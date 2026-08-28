"""Convert Astribot schema-v4 dexterous-hand HDF5 recordings to LeRobot.

The schema-v4 convenience ``joints_dict`` collapses left/right hand joints with
identical names. This converter intentionally reads the per-side streams instead:

* 20 stationary-task upper-body position states/commands (torso, arms, head);
* 16 driver-space joints selected from each 21-joint hand state/command stream;
* the left view of the side-by-side head stereo stream plus left/right wrist RGB cameras.

The chassis is excluded because it was not recorded on the first campaign day
and is not controlled for this stationary task. The resulting state and action
vectors are both 52-dimensional. Wrist camera
orientation metadata is honored during JPEG decoding.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime as dt
import io
import json
from pathlib import Path
import shutil
import time
from typing import Any

import h5py
import numpy as np
from PIL import Image
import tyro

BODY_PARTS: tuple[tuple[str, int], ...] = (
    ("astribot_torso", 4),
    ("astribot_arm_left", 7),
    ("astribot_arm_right", 7),
    ("astribot_head", 2),
)
CAMERAS: tuple[tuple[str, str], ...] = (
    ("head_stereo", "images.head"),
    ("left_wrist_rgbd", "images.left_wrist"),
    ("right_wrist_rgbd", "images.right_wrist"),
)
BODY_DIM = sum(dim for _, dim in BODY_PARTS)
HAND_DRIVER_DIM = 16
VECTOR_DIM = BODY_DIM + 2 * HAND_DRIVER_DIM
COMPLETION_MARKER = "conversion_complete.json"
PROGRESS_MARKER = "conversion_progress.json"


@dataclasses.dataclass(frozen=True)
class Args:
    data_dir: str = "/astribot/astribot_light_candle"
    output_dir: str = "/data2/202608/astribot/astribot_light_candle_lerobot"
    repo_id: str = "astribot/astribot_light_candle_lerobot"
    default_task: str = "Light the candle"
    fps: int = 30
    image_width: int = 224
    image_height: int = 224
    # Applied per camera. The three cameras decode concurrently.
    decode_workers: int = 48
    image_writer_processes: int = 16
    image_writer_threads: int = 4
    video_encode_workers: int = 3
    frame_stride: int = 1
    trim_start_seconds: float = 0.0
    trim_end_seconds: float = 5.0
    max_episodes: int | None = None
    overwrite: bool = False
    resume: bool = False
    validate_only: bool = False


@dataclasses.dataclass(frozen=True)
class FileInspection:
    path: str
    size_bytes: int
    sample_count: int
    valid_frame_count: int
    eligible: bool
    reason: str | None = None
    has_head_rgbd: bool = False
    head_rgbd_valid_frames: int = 0
    head_stereo_valid_frames: int = 0


@dataclasses.dataclass(frozen=True)
class CameraPayload:
    camera: str
    output_key: str
    raw_indices: np.ndarray
    offsets: np.ndarray
    blob: np.ndarray
    rotation_degrees: int


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values.tolist()]


def _body_joint_names() -> list[str]:
    names: list[str] = []
    for part, dim in BODY_PARTS:
        names.extend(f"{part}_{index}" for index in range(dim))
    return names


def _hand_name_selection(f: h5py.File, side: str) -> tuple[list[str], np.ndarray]:
    event_root = f"streams/joint_events/hands/{side}"
    state_names = _decode_strings(np.asarray(f[f"{event_root}/state/joint_names"]))
    driver_names = _decode_strings(np.asarray(f[f"{event_root}/command_driver_input/joint_names"]))
    if len(driver_names) != HAND_DRIVER_DIM:
        raise ValueError(f"{side} hand has {len(driver_names)} driver joints, expected {HAND_DRIVER_DIM}")
    if len(state_names) != len(set(state_names)) or len(driver_names) != len(set(driver_names)):
        raise ValueError(f"{side} hand joint names are not unique")
    state_index = {name: index for index, name in enumerate(state_names)}
    missing = [name for name in driver_names if name not in state_index]
    if missing:
        raise ValueError(f"{side} hand driver joints missing from state: {missing}")
    return driver_names, np.asarray([state_index[name] for name in driver_names], dtype=np.int64)


def extract_state_actions(f: h5py.File) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract aligned 52-D state/action arrays without left/right name collisions."""

    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    names = _body_joint_names()
    sample_count: int | None = None

    for part, expected_dim in BODY_PARTS:
        root = f"streams/joints/{part}"
        state = np.asarray(f[f"{root}/state/position"], dtype=np.float32)
        action = np.asarray(f[f"{root}/command/position"], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != expected_dim:
            raise ValueError(f"Unexpected {part} state shape: {state.shape}")
        if action.shape != state.shape:
            raise ValueError(f"Unexpected {part} action shape: {action.shape}, state={state.shape}")
        sample_count = state.shape[0] if sample_count is None else sample_count
        if state.shape[0] != sample_count:
            raise ValueError(f"Mismatched sample count for {part}: {state.shape[0]} != {sample_count}")
        state_parts.append(state)
        action_parts.append(action)

    for side in ("left", "right"):
        driver_names, selected_indices = _hand_name_selection(f, side)
        root = f"streams/joints/hand_{side}"
        hand_state = np.asarray(f[f"{root}/state/position"], dtype=np.float32)
        hand_action = np.asarray(f[f"{root}/driver_input/position"], dtype=np.float32)
        if hand_state.ndim != 2 or hand_action.shape != hand_state.shape:
            raise ValueError(
                f"Unexpected {side} hand state/action shapes: state={hand_state.shape}, action={hand_action.shape}"
            )
        if hand_state.shape[0] != sample_count:
            raise ValueError(f"Mismatched sample count for {side} hand: {hand_state.shape[0]} != {sample_count}")
        if int(selected_indices.max()) >= hand_state.shape[1]:
            raise ValueError(f"{side} hand selection exceeds recorded state dimension {hand_state.shape[1]}")
        state_parts.append(hand_state[:, selected_indices])
        action_parts.append(hand_action[:, selected_indices])
        names.extend(f"{side}_hand_{name}" for name in driver_names)

    state = np.concatenate(state_parts, axis=1)
    actions = np.concatenate(action_parts, axis=1)
    if state.shape != actions.shape or state.shape[1] != VECTOR_DIM:
        raise ValueError(f"Expected matching (*,{VECTOR_DIM}) state/actions, got {state.shape} and {actions.shape}")
    return state, actions, names


def _camera_valid_mask(f: h5py.File, sample_count: int) -> np.ndarray:
    valid = np.ones(sample_count, dtype=bool)
    for camera, _ in CAMERAS:
        base = f"images_dict/{camera}"
        raw_indices = np.asarray(f[f"{base}/aligned/rgb_raw_index"])
        raw_sizes = np.asarray(f[f"{base}/rgb_size"])
        if raw_indices.shape != (sample_count,):
            raise ValueError(f"Unexpected {camera} aligned index shape: {raw_indices.shape}")
        finite = np.isfinite(raw_indices)
        integer_indices = np.where(finite, raw_indices, -1).astype(np.int64)
        valid &= finite & (integer_indices >= 0) & (integer_indices < len(raw_sizes))
        safe_indices = np.clip(integer_indices, 0, max(len(raw_sizes) - 1, 0))
        if len(raw_sizes) == 0:
            valid &= False
        else:
            valid &= np.isfinite(raw_sizes[safe_indices]) & (raw_sizes[safe_indices] > 0)
    return valid


def _select_trimmed_rows(
    f: h5py.File,
    valid: np.ndarray,
    *,
    trim_start_seconds: float,
    trim_end_seconds: float,
) -> np.ndarray:
    timestamps = np.asarray(f["time"], dtype=np.float64)
    if timestamps.shape != valid.shape:
        raise ValueError(f"Unexpected time shape: {timestamps.shape}, valid={valid.shape}")
    valid = valid & np.isfinite(timestamps)
    valid_rows = np.flatnonzero(valid)
    if len(valid_rows) == 0:
        return valid_rows

    first_time = float(timestamps[valid_rows[0]])
    last_time = float(timestamps[valid_rows[-1]])
    start_time = first_time + trim_start_seconds
    end_time = last_time - trim_end_seconds
    if end_time < start_time:
        return np.empty((0,), dtype=np.int64)
    return np.flatnonzero(valid & (timestamps >= start_time) & (timestamps <= end_time))


def inspect_hdf5(
    path: Path,
    *,
    trim_start_seconds: float = 0.0,
    trim_end_seconds: float = 5.0,
) -> FileInspection:
    try:
        with h5py.File(path, "r") as f:
            if int(f.attrs.get("schema_version", -1)) != 4:
                raise ValueError(f"schema_version={f.attrs.get('schema_version')!r}, expected 4")
            if not bool(f.attrs.get("completed", False)) or bool(f.attrs.get("capture_incomplete", True)):
                raise ValueError("capture is not marked complete")
            state, actions, _ = extract_state_actions(f)
            valid = np.isfinite(state).all(axis=1) & np.isfinite(actions).all(axis=1)
            valid &= _camera_valid_mask(f, state.shape[0])
            rows = _select_trimmed_rows(
                f,
                valid,
                trim_start_seconds=trim_start_seconds,
                trim_end_seconds=trim_end_seconds,
            )
            valid_count = len(rows)
            if valid_count == 0:
                raise ValueError("no frames remain after validation and time trimming")
            has_head_rgbd = "head_rgbd" in f["images_dict"]
            head_rgbd_valid_frames = 0
            if has_head_rgbd:
                head_rgbd_indices = np.asarray(f["images_dict/head_rgbd/aligned/rgb_raw_index"])
                head_rgbd_valid_frames = int((np.isfinite(head_rgbd_indices) & (head_rgbd_indices >= 0)).sum())
            head_stereo_indices = np.asarray(f["images_dict/head_stereo/aligned/rgb_raw_index"])
            return FileInspection(
                path=str(path),
                size_bytes=path.stat().st_size,
                sample_count=state.shape[0],
                valid_frame_count=valid_count,
                eligible=True,
                has_head_rgbd=has_head_rgbd,
                head_rgbd_valid_frames=head_rgbd_valid_frames,
                head_stereo_valid_frames=int(
                    (np.isfinite(head_stereo_indices) & (head_stereo_indices >= 0)).sum()
                ),
            )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        return FileInspection(
            path=str(path),
            size_bytes=path.stat().st_size if path.exists() else 0,
            sample_count=0,
            valid_frame_count=0,
            eligible=False,
            reason=f"{type(exc).__name__}: {exc}",
        )


def inspect_corpus(
    data_dir: Path,
    max_episodes: int | None = None,
    *,
    trim_start_seconds: float = 0.0,
    trim_end_seconds: float = 5.0,
) -> list[FileInspection]:
    paths = sorted(path for path in data_dir.rglob("*.hdf5") if path.is_file())
    if max_episodes is not None:
        paths = paths[:max_episodes]
    inspections = []
    for index, path in enumerate(paths, start=1):
        inspection = inspect_hdf5(
            path,
            trim_start_seconds=trim_start_seconds,
            trim_end_seconds=trim_end_seconds,
        )
        inspections.append(inspection)
        print(
            f"[scan {index}/{len(paths)}] eligible={inspection.eligible} "
            f"valid_frames={inspection.valid_frame_count} path={path}"
        )
    return inspections


def _jpeg_offsets(sizes: np.ndarray) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=np.int64)
    offsets = np.empty(len(sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    return offsets


def _decode_resize_rgb(
    jpeg_bytes: bytes,
    width: int,
    height: int,
    rotation_degrees: int,
    *,
    crop_left_half: bool = False,
) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg_bytes)) as source_image:
        image = source_image.convert("RGB")
        if crop_left_half:
            source_width, source_height = image.size
            if source_width % 2 != 0:
                raise ValueError(f"Stereo image width must be even, got {image.size}")
            image = image.crop((0, 0, source_width // 2, source_height))
        if rotation_degrees == 180:
            image = image.transpose(Image.Transpose.ROTATE_180)
        elif rotation_degrees not in (0,):
            raise ValueError(f"Unsupported camera rotation: {rotation_degrees}")
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8).copy()


def _read_camera_payload(
    f: h5py.File,
    camera: str,
    output_key: str,
    sample_rows: np.ndarray,
) -> CameraPayload:
    base = f"images_dict/{camera}"
    raw_indices = np.asarray(f[f"{base}/aligned/rgb_raw_index"], dtype=np.int64)[sample_rows]
    raw_sizes = np.asarray(f[f"{base}/rgb_size"], dtype=np.int64)
    return CameraPayload(
        camera=camera,
        output_key=output_key,
        raw_indices=raw_indices,
        offsets=_jpeg_offsets(raw_sizes),
        blob=np.asarray(f[f"{base}/rgb"], dtype=np.uint8),
        rotation_degrees=int(f[f"{base}/rgb"].attrs.get("rotation_degrees", 0)),
    )


def _decode_camera_payload(
    payload: CameraPayload,
    *,
    width: int,
    height: int,
    workers: int,
) -> tuple[str, list[np.ndarray]]:
    raw_indices = payload.raw_indices
    offsets = payload.offsets
    blob = payload.blob

    unique_indices = sorted(set(raw_indices.tolist()))

    def decode(index: int) -> tuple[int, np.ndarray]:
        start, end = int(offsets[index]), int(offsets[index + 1])
        return index, _decode_resize_rgb(
            bytes(blob[start:end]),
            width,
            height,
            payload.rotation_degrees,
            crop_left_half=payload.camera == "head_stereo",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        decoded = dict(executor.map(decode, unique_indices))
    return payload.output_key, [decoded[int(index)] for index in raw_indices]


def _decode_camera_frames(
    f: h5py.File,
    camera: str,
    sample_rows: np.ndarray,
    *,
    width: int,
    height: int,
    workers: int,
) -> list[np.ndarray]:
    """Compatibility wrapper used by focused tests and one-camera callers."""

    payload = _read_camera_payload(f, camera, camera, sample_rows)
    _, frames = _decode_camera_payload(payload, width=width, height=height, workers=workers)
    return frames


def _feature_names() -> list[str]:
    body = _body_joint_names()
    hand_template = (
        "thumb_j0",
        "thumb_j1",
        "thumb_j2",
        "thumb_j3",
        "index_j0",
        "index_j1",
        "index_j2",
        "middle_j0",
        "middle_j1",
        "middle_j2",
        "ring_j0",
        "ring_j1",
        "ring_j2",
        "pinky_j0",
        "pinky_j1",
        "pinky_j2",
    )
    return [*body, *(f"left_hand_{name}" for name in hand_template), *(f"right_hand_{name}" for name in hand_template)]


def _dataset_features(args: Args) -> dict[str, dict[str, Any]]:
    image_feature = {
        "dtype": "video",
        "shape": (3, args.image_height, args.image_width),
        "names": ["channel", "height", "width"],
    }
    names = _feature_names()
    return {
        "images.head": dict(image_feature),
        "images.left_wrist": dict(image_feature),
        "images.right_wrist": dict(image_feature),
        "state": {"dtype": "float32", "shape": (VECTOR_DIM,), "names": names},
        "actions": {"dtype": "float32", "shape": (VECTOR_DIM,), "names": names},
    }


def _summary(inspections: list[FileInspection]) -> dict[str, Any]:
    eligible = [item for item in inspections if item.eligible]
    return {
        "source_hdf5_files": len(inspections),
        "eligible_hdf5_files": len(eligible),
        "excluded_hdf5_files": len(inspections) - len(eligible),
        "eligible_source_bytes": sum(item.size_bytes for item in eligible),
        "eligible_valid_frames": sum(item.valid_frame_count for item in eligible),
        "head_rgbd_files": sum(item.has_head_rgbd for item in eligible),
        "head_rgbd_valid_frames": sum(item.head_rgbd_valid_frames for item in eligible),
        "head_stereo_files": sum(item.head_stereo_valid_frames > 0 for item in eligible),
        "head_stereo_valid_frames": sum(item.head_stereo_valid_frames for item in eligible),
        "excluded": [dataclasses.asdict(item) for item in inspections if not item.eligible],
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _converted_frame_count(item: FileInspection, frame_stride: int) -> int:
    return (item.valid_frame_count + frame_stride - 1) // frame_stride


def _converted_file_record(item: FileInspection, frame_stride: int) -> dict[str, Any]:
    return {
        "path": item.path,
        "size_bytes": item.size_bytes,
        "source_samples": item.sample_count,
        "converted_frames": _converted_frame_count(item, frame_stride),
    }


def _manifest(
    args: Args,
    data_dir: Path,
    output_dir: Path,
    summary: dict[str, Any],
    converted_files: list[dict[str, Any]],
    *,
    status: str,
    started_at_utc: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "started_at_utc": started_at_utc,
        "updated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_root": str(data_dir),
        "output_root": str(output_dir),
        "repo_id": args.repo_id,
        "task": args.default_task,
        "fps": args.fps,
        "state_dim": VECTOR_DIM,
        "action_dim": VECTOR_DIM,
        "action_layout": _feature_names(),
        "frame_stride": args.frame_stride,
        "image_size": [args.image_height, args.image_width],
        "time_trim_seconds": {
            "start": args.trim_start_seconds,
            "end": args.trim_end_seconds,
            "basis": "recorded_time",
        },
        "head_camera": {
            "source": "head_stereo",
            "view": "left",
            "crop": "left_half_before_resize",
        },
        "provenance": summary,
        "converted_episodes": len(converted_files),
        "converted_frames": sum(item["converted_frames"] for item in converted_files),
        "files": converted_files,
    }


def _validate_progress_contract(progress: dict[str, Any], args: Args, data_dir: Path) -> None:
    expected = {
        "source_root": str(data_dir),
        "repo_id": args.repo_id,
        "task": args.default_task,
        "fps": args.fps,
        "state_dim": VECTOR_DIM,
        "action_dim": VECTOR_DIM,
        "action_layout": _feature_names(),
        "frame_stride": args.frame_stride,
        "image_size": [args.image_height, args.image_width],
        "time_trim_seconds": {
            "start": args.trim_start_seconds,
            "end": args.trim_end_seconds,
            "basis": "recorded_time",
        },
        "head_camera": {
            "source": "head_stereo",
            "view": "left",
            "crop": "left_half_before_resize",
        },
    }
    mismatches = {key: (progress.get(key), value) for key, value in expected.items() if progress.get(key) != value}
    if mismatches:
        raise ValueError(f"Resume progress contract does not match requested conversion: {mismatches}")


def _validate_metadata_contract(info: dict[str, Any], args: Args) -> None:
    if info.get("robot_type") != "astribot_dexhand":
        raise ValueError(f"Resume robot_type mismatch: {info.get('robot_type')!r}")
    if info.get("fps") != args.fps:
        raise ValueError(f"Resume fps mismatch: {info.get('fps')!r} != {args.fps}")
    actual_features = info.get("features", {})
    for key, expected in _dataset_features(args).items():
        if key not in actual_features:
            raise ValueError(f"Resume dataset is missing feature {key!r}")
        actual = actual_features[key]
        if actual.get("dtype") != expected["dtype"]:
            raise ValueError(f"Resume feature {key!r} dtype mismatch: {actual.get('dtype')!r}")
        if tuple(actual.get("shape", ())) != tuple(expected["shape"]):
            raise ValueError(f"Resume feature {key!r} shape mismatch: {actual.get('shape')!r}")
        if actual.get("names") != expected["names"]:
            raise ValueError(f"Resume feature {key!r} names mismatch")


def _quarantine_uncommitted_artifacts(output_dir: Path, info: dict[str, Any], episode_index: int) -> Path | None:
    chunk = episode_index // int(info["chunks_size"])
    candidates = [output_dir / "images"]
    candidates.append(
        output_dir / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    )
    for key, feature in info["features"].items():
        if feature.get("dtype") == "video":
            candidates.append(
                output_dir
                / info["video_path"].format(
                    episode_chunk=chunk,
                    video_key=key,
                    episode_index=episode_index,
                )
            )
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = output_dir.parent / "resume_orphans" / f"{output_dir.name}_episode_{episode_index:06d}_{stamp}"
    if quarantine.exists():
        raise FileExistsError(f"Resume quarantine already exists: {quarantine}")
    for source in candidates:
        relative = source.relative_to(output_dir)
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(f"RESUME_QUARANTINE episode={episode_index} path={quarantine}")
    return quarantine


def _prepare_resume(
    output_dir: Path,
    args: Args,
    data_dir: Path,
    eligible: list[FileInspection],
) -> tuple[int, list[dict[str, Any]], str]:
    info_path = output_dir / "meta/info.json"
    episodes_path = output_dir / "meta/episodes.jsonl"
    stats_path = output_dir / "meta/episodes_stats.jsonl"
    for path in (info_path, episodes_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot append-resume without {path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    _validate_metadata_contract(info, args)
    episodes = _read_jsonlines(episodes_path)
    episodes_stats = _read_jsonlines(stats_path)
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    if total_episodes < 0 or total_episodes > len(eligible):
        raise ValueError(f"Resume episode count {total_episodes} is outside eligible range 0..{len(eligible)}")
    if len(episodes) != total_episodes or len(episodes_stats) != total_episodes:
        raise ValueError(
            "Resume metadata is not at a committed boundary: "
            f"info={total_episodes}, episodes={len(episodes)}, stats={len(episodes_stats)}"
        )
    expected_indices = list(range(total_episodes))
    if [item.get("episode_index") for item in episodes] != expected_indices:
        raise ValueError("Resume episodes are not a contiguous zero-based prefix")
    if [item.get("episode_index") for item in episodes_stats] != expected_indices:
        raise ValueError("Resume episode stats are not a contiguous zero-based prefix")

    expected_lengths = [_converted_frame_count(item, args.frame_stride) for item in eligible[:total_episodes]]
    actual_lengths = [int(item.get("length", -1)) for item in episodes]
    if actual_lengths != expected_lengths:
        raise ValueError(
            "Resume episode lengths do not match this source/trim/stride contract: "
            f"actual={actual_lengths}, expected={expected_lengths}"
        )
    if total_frames != sum(actual_lengths):
        raise ValueError(f"Resume total_frames mismatch: info={total_frames}, episodes={sum(actual_lengths)}")

    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    expected_total_videos = total_episodes * len(video_keys)
    if int(info.get("total_videos", -1)) != expected_total_videos:
        raise ValueError(
            f"Resume total_videos mismatch: info={info.get('total_videos')}, expected={expected_total_videos}"
        )

    progress_path = output_dir / PROGRESS_MARKER
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        _validate_progress_contract(progress, args, data_dir)
        progress_episodes = int(progress.get("converted_episodes", -1))
        if progress_episodes > total_episodes:
            raise ValueError(
                "Resume progress marker is ahead of committed metadata: "
                f"progress={progress_episodes}, metadata={total_episodes}"
            )
        if progress_episodes < total_episodes:
            print(
                f"RESUME_PROGRESS_LAG progress_episodes={progress_episodes} "
                f"committed_episodes={total_episodes} reconstruction=metadata_prefix"
            )
        started_at_utc = str(progress.get("started_at_utc") or dt.datetime.now(dt.UTC).isoformat())
    else:
        started_at_utc = dt.datetime.now(dt.UTC).isoformat()
        print("RESUME_BOOTSTRAP progress_marker=missing metadata_contract=validated")

    _quarantine_uncommitted_artifacts(output_dir, info, total_episodes)

    expected_parquet = {
        output_dir
        / info["data_path"].format(
            episode_chunk=index // int(info["chunks_size"]),
            episode_index=index,
        )
        for index in expected_indices
    }
    expected_videos = {
        output_dir
        / info["video_path"].format(
            episode_chunk=index // int(info["chunks_size"]),
            video_key=key,
            episode_index=index,
        )
        for index in expected_indices
        for key in video_keys
    }
    actual_parquet = set(output_dir.rglob("*.parquet"))
    actual_videos = set(output_dir.rglob("*.mp4"))
    if actual_parquet != expected_parquet:
        raise ValueError(
            f"Resume parquet inventory mismatch: missing={expected_parquet - actual_parquet}, "
            f"extra={actual_parquet - expected_parquet}"
        )
    if actual_videos != expected_videos:
        raise ValueError(
            f"Resume video inventory mismatch: missing={expected_videos - actual_videos}, "
            f"extra={actual_videos - expected_videos}"
        )

    converted_files = [_converted_file_record(item, args.frame_stride) for item in eligible[:total_episodes]]
    print(
        f"APPEND_RESUME_READY episodes={total_episodes} frames={total_frames} "
        f"videos={expected_total_videos} next_episode={total_episodes}"
    )
    return total_episodes, converted_files, started_at_utc


def convert(args: Args) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {data_dir}")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.trim_start_seconds < 0 or args.trim_end_seconds < 0:
        raise ValueError("trim durations must be non-negative")
    if args.decode_workers < 1 or args.image_writer_processes < 0 or args.image_writer_threads < 1:
        raise ValueError("decode/image writer worker counts are invalid")
    if args.video_encode_workers < 1:
        raise ValueError("--video-encode-workers must be at least 1")

    inspections = inspect_corpus(
        data_dir,
        args.max_episodes,
        trim_start_seconds=args.trim_start_seconds,
        trim_end_seconds=args.trim_end_seconds,
    )
    summary = _summary(inspections)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.validate_only:
        return

    eligible = [item for item in inspections if item.eligible]
    if not eligible:
        raise ValueError("No eligible source episodes were found")

    marker = output_dir / COMPLETION_MARKER
    if marker.is_file() and not args.overwrite:
        completed = json.loads(marker.read_text(encoding="utf-8"))
        _validate_progress_contract(completed, args, data_dir)
        if completed.get("status") != "complete" or completed.get("converted_episodes") != len(eligible):
            raise ValueError(f"Invalid completion marker at {marker}")
        print(f"Complete dataset already exists at {output_dir}; leaving it unchanged")
        return

    start_episode = 0
    is_resume = False
    converted_files: list[dict[str, Any]] = []
    started_at_utc = dt.datetime.now(dt.UTC).isoformat()
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif args.resume:
            is_resume = True
            start_episode, converted_files, started_at_utc = _prepare_resume(
                output_dir,
                args,
                data_dir,
                eligible,
            )
        else:
            raise FileExistsError(
                f"Output exists without a completion marker: {output_dir}; pass --resume or --overwrite"
            )
    elif args.resume:
        raise FileNotFoundError(f"Cannot append-resume because output does not exist: {output_dir}")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.video_utils import encode_video_frames

    class ParallelVideoLeRobotDataset(LeRobotDataset):
        def encode_episode_videos(self, episode_index: int) -> dict[str, str]:
            def encode(key: str) -> tuple[str, str]:
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                if not video_path.is_file():
                    image_dir = self._get_image_file_path(
                        episode_index=episode_index,
                        image_key=key,
                        frame_index=0,
                    ).parent
                    encode_video_frames(
                        image_dir,
                        video_path,
                        self.fps,
                        overwrite=True,
                        log_level=None,
                    )
                return key, str(video_path)

            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.video_encode_workers, len(self.meta.video_keys))
            ) as executor:
                video_paths = dict(executor.map(encode, self.meta.video_keys))
            print(
                f"VIDEO_ENCODE_PARALLEL episode={episode_index} cameras={len(video_paths)} "
                f"workers={args.video_encode_workers} seconds={time.monotonic() - started:.3f}"
            )
            return video_paths

    if is_resume:
        dataset = ParallelVideoLeRobotDataset(repo_id=args.repo_id, root=output_dir)
        if dataset.meta.total_episodes != start_episode:
            raise RuntimeError(
                f"Loaded resume episode count changed: {dataset.meta.total_episodes} != {start_episode}"
            )
        dataset.episode_buffer = dataset.create_episode_buffer()
        dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
    else:
        dataset = ParallelVideoLeRobotDataset.create(
            repo_id=args.repo_id,
            root=output_dir,
            robot_type="astribot_dexhand",
            fps=args.fps,
            features=_dataset_features(args),
            use_videos=True,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )

    progress_path = output_dir / PROGRESS_MARKER
    _atomic_write_json(
        progress_path,
        _manifest(
            args,
            data_dir,
            output_dir,
            summary,
            converted_files,
            status="in_progress",
            started_at_utc=started_at_utc,
        ),
    )

    try:
        for episode_index, item in enumerate(eligible[start_episode:], start=start_episode):
            episode_started = time.monotonic()
            path = Path(item.path)
            print(f"[{episode_index + 1}/{len(eligible)}] {path}")
            with h5py.File(path, "r") as f:
                state, actions, names = extract_state_actions(f)
                if names != _feature_names():
                    raise ValueError(f"Unexpected action layout in {path}")
                valid = np.isfinite(state).all(axis=1) & np.isfinite(actions).all(axis=1)
                valid &= _camera_valid_mask(f, state.shape[0])
                rows = _select_trimmed_rows(
                    f,
                    valid,
                    trim_start_seconds=args.trim_start_seconds,
                    trim_end_seconds=args.trim_end_seconds,
                )[:: args.frame_stride]
                payloads = [
                    _read_camera_payload(f, camera, output_key, rows)
                    for camera, output_key in CAMERAS
                ]

            decode_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as executor:
                camera_frames = dict(
                    executor.map(
                        lambda payload: _decode_camera_payload(
                            payload,
                            width=args.image_width,
                            height=args.image_height,
                            workers=args.decode_workers,
                        ),
                        payloads,
                    )
                )
            print(
                f"CAMERA_DECODE_PARALLEL episode={episode_index} cameras={len(payloads)} "
                f"workers_per_camera={args.decode_workers} frames={len(rows)} "
                f"seconds={time.monotonic() - decode_started:.3f}"
            )

            for output_index, source_row in enumerate(rows.tolist()):
                frame: dict[str, Any] = {
                    "state": state[source_row],
                    "actions": actions[source_row],
                    "task": args.default_task,
                }
                for _, output_key in CAMERAS:
                    frame[output_key] = camera_frames[output_key][output_index]
                dataset.add_frame(frame)
            dataset.save_episode()

            converted_files.append(_converted_file_record(item, args.frame_stride))
            if converted_files[-1]["converted_frames"] != len(rows):
                raise RuntimeError(
                    f"Converted frame accounting mismatch for {path}: "
                    f"expected={converted_files[-1]['converted_frames']} actual={len(rows)}"
                )
            _atomic_write_json(
                progress_path,
                _manifest(
                    args,
                    data_dir,
                    output_dir,
                    summary,
                    converted_files,
                    status="in_progress",
                    started_at_utc=started_at_utc,
                ),
            )
            print(
                f"EPISODE_COMMITTED episode={episode_index} frames={len(rows)} "
                f"total_episodes={len(converted_files)} seconds={time.monotonic() - episode_started:.3f}"
            )

        if hasattr(dataset, "consolidate"):
            dataset.consolidate()
    finally:
        dataset.stop_image_writer()

    manifest = _manifest(
        args,
        data_dir,
        output_dir,
        summary,
        converted_files,
        status="complete",
        started_at_utc=started_at_utc,
    )
    _atomic_write_json(progress_path, manifest)
    _atomic_write_json(marker, manifest)
    print(
        json.dumps(
            {key: manifest[key] for key in ("status", "output_root", "converted_episodes", "converted_frames")},
            indent=2,
        )
    )


if __name__ == "__main__":
    convert(tyro.cli(Args))
