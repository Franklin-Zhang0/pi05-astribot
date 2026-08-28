"""Convert Astribot HDF5 episodes to LeRobot dataset format for openpi fine-tuning.

This script is tailored to the Astribot recording format (see `astribot_data_example/*.hdf5`).
It:
- reads per-segment ("*_0", "*_1", ...) image streams and command/state streams
- aligns camera frames to nearest joint state + nearest joint command by timestamp
- writes a LeRobot dataset to $HF_LEROBOT_HOME/<repo_id>

Usage (no uv required):

/data/250010203/franklin/project/pi_astribot_finetune/openpi/.venv/bin/python \
  openpi/examples/astribot/convert_astribot_hdf5_to_lerobot.py \
    --data-dir /astribot_data/serve_small_mm_beans \
    --repo-id <org>/<dataset_name> \
    --default-task "Serve small MM beans" \
    --max-episodes 0

Notes:
- This converter assumes *absolute* joint position commands (25-DoF) at ~100Hz.
- Images are stored compressed as JPEG byte streams in HDF5 (data_* + data_size_*).
- We only use a subset of cameras for training by default: head_rgbd, left_wrist_rgbd, right_wrist_rgbd.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
from pathlib import Path

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

try:
    import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclasses.dataclass(frozen=True)
class Args:
    data_dir: str
    repo_id: str

    # If provided, used as the dataset task string for all episodes.
    default_task: str = "do something"

    # Cameras to export.
    use_head_rgbd: bool = True
    use_left_wrist_rgbd: bool = True
    use_right_wrist_rgbd: bool = True

    # Image resize for dataset storage.
    image_width: int = 640
    image_height: int = 360

    # If >0, stop after converting N episodes.
    max_episodes: int = 0

    # Subsample camera frames to speed up conversion (1 = keep all frames).
    frame_stride: int = 1

    # Per-segment frame limit (0 = no limit). Helpful for smoke tests.
    max_frames_per_segment: int = 0

    # Parallelize JPEG decode/resize.
    # Set >0 to enable. Recommended for large datasets.
    decode_workers: int = 8

    # Backend for parallel decode.
    # - "process": true multiprocessing (more CPU, more overhead)
    # - "thread": lower overhead; OpenCV decode/resize releases the GIL in practice
    decode_backend: str = "process"


CAMERA_KEYS = ("head_rgbd", "left_wrist_rgbd", "right_wrist_rgbd")

# Whole-body order used by SDK examples.
WHOLE_BODY_NAMES = (
    "astribot_chassis",
    "astribot_torso",
    "astribot_arm_left",
    "astribot_gripper_left",
    "astribot_arm_right",
    "astribot_gripper_right",
    "astribot_head",
)
WHOLE_BODY_DIMS = {
    "astribot_chassis": 3,
    "astribot_torso": 4,
    "astribot_arm_left": 7,
    "astribot_gripper_left": 1,
    "astribot_arm_right": 7,
    "astribot_gripper_right": 1,
    "astribot_head": 2,
}


class EmptyJointSeriesError(RuntimeError):
    """Raised when an HDF5 segment contains empty joint state/command series."""


def _iter_hdf5_files(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.rglob("*.hdf5") if p.is_file()])


def _segment_ids_in_h5(f: h5py.File) -> list[int]:
    """Segments are stored as *_0, *_1, ..."""
    segs = []
    # Use command timestamps as reference for segments.
    keys = [k for k in f["joints_dict/astribot_torso/command"].keys() if k.startswith("timestamp_")]
    for k in keys:
        try:
            segs.append(int(k.split("_")[-1]))
        except ValueError:
            pass
    return sorted(set(segs))


def _jpeg_offsets(sizes: np.ndarray) -> np.ndarray:
    """Offsets into concatenated jpeg byte stream."""
    sizes_i64 = sizes.astype(np.int64)
    offsets = np.empty((len(sizes_i64) + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes_i64, out=offsets[1:])
    return offsets


def _decode_jpeg_at(data: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    """Decode a single JPEG frame from a concatenated HDF5 byte stream."""
    start = int(offsets[index])
    end = int(offsets[index + 1])
    chunk = data[start:end]
    img = cv2.imdecode(chunk, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img


def _decode_resize_rgb(jpeg_bytes: bytes, image_w: int, image_h: int) -> np.ndarray:
    """Decode JPEG bytes -> resized RGB uint8 HWC."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed")
    if bgr.shape[0] != image_h or bgr.shape[1] != image_w:
        bgr = cv2.resize(bgr, (image_w, image_h), interpolation=cv2.INTER_AREA)
    return bgr[..., ::-1]


def _resize_bgr(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _concat_whole_body(joint_dict: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for name in WHOLE_BODY_NAMES:
        parts.append(np.asarray(joint_dict[name], dtype=np.float32).reshape(-1))
    out = np.concatenate(parts, axis=0)
    if out.shape != (25,):
        raise ValueError(f"Expected 25-DoF whole-body vector, got {out.shape}")
    return out


def _nearest_index(t: np.ndarray, t_query: float) -> int:
    # t is sorted-ish. Use argmin(abs).
    return int(np.argmin(np.abs(t - t_query)))


def _read_segment_joint_series(f: h5py.File, *, seg_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_state, state_vecs, t_cmd, cmd_vecs).

    The Astribot recorder can produce tiny off-by-1 mismatches between timestamp arrays and
    per-part arrays; we defensively truncate to the minimum consistent length.
    """

    # State is ~250Hz.
    t_state_ref = np.asarray(f[f"joints_dict/astribot_torso/states/timestamp_{seg_id}"][()])
    # Command is ~100Hz.
    t_cmd_ref = np.asarray(f[f"joints_dict/astribot_torso/command/timestamp_{seg_id}"][()])

    state_parts: dict[str, np.ndarray] = {}
    cmd_parts: dict[str, np.ndarray] = {}

    for name in WHOLE_BODY_NAMES:
        state_parts[name] = np.asarray(f[f"joints_dict/{name}/states/position_{seg_id}"][()], dtype=np.float32)
        cmd_parts[name] = np.asarray(f[f"joints_dict/{name}/command/command_{seg_id}"][()], dtype=np.float32)

    state_len = min(int(len(t_state_ref)), *(int(a.shape[0]) for a in state_parts.values()))
    cmd_len = min(int(len(t_cmd_ref)), *(int(a.shape[0]) for a in cmd_parts.values()))

    # numpy cannot infer the -1 dim when the first dim is 0; skip these segments upstream.
    if state_len == 0 or cmd_len == 0:
        state_lens = {k: int(v.shape[0]) for k, v in state_parts.items()}
        cmd_lens = {k: int(v.shape[0]) for k, v in cmd_parts.items()}
        zero_state = [k for k, n in state_lens.items() if n == 0]
        zero_cmd = [k for k, n in cmd_lens.items() if n == 0]
        raise EmptyJointSeriesError(
            "Empty joint series in HDF5 segment; cannot build (T,25) vectors. "
            f"seg_id={seg_id} t_state_len={len(t_state_ref)} t_cmd_len={len(t_cmd_ref)} "
            f"state_len={state_len} cmd_len={cmd_len} "
            f"zero_state_parts={zero_state} zero_cmd_parts={zero_cmd}"
        )

    t_state_ref = t_state_ref[:state_len]
    t_cmd_ref = t_cmd_ref[:cmd_len]

    state_vecs = np.concatenate(
        [state_parts[name][:state_len].reshape(state_len, -1) for name in WHOLE_BODY_NAMES], axis=1
    ).astype(np.float32)
    cmd_vecs = np.concatenate(
        [cmd_parts[name][:cmd_len].reshape(cmd_len, -1) for name in WHOLE_BODY_NAMES], axis=1
    ).astype(np.float32)

    if state_vecs.shape[1] != 25:
        raise ValueError(f"Expected state dim 25, got {state_vecs.shape}")
    if cmd_vecs.shape[1] != 25:
        raise ValueError(f"Expected action dim 25, got {cmd_vecs.shape}")

    return t_state_ref, state_vecs, t_cmd_ref, cmd_vecs


def _read_segment_camera_raw(
    f: h5py.File,
    *,
    seg_id: int,
    camera: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps, data_size, concatenated_bytes)."""
    base = f"images_dict/{camera}/color"
    t = np.asarray(f[f"{base}/timestamp_{seg_id}"][()])
    sizes = np.asarray(f[f"{base}/data_size_{seg_id}"][()])
    data = np.asarray(f[f"{base}/data_{seg_id}"][()])
    return t, sizes, data


def _episode_task(h5_path: Path, task_info: dict, default_task: str) -> str:
    # Your real dataset likely has richer task_info.json; the example only has dataset-level metadata.
    # This hook tries common patterns.
    if "episodes" in task_info:
        # e.g. {"episodes": {"file.hdf5": {"task": "..."}}}
        ep = task_info["episodes"].get(h5_path.name)
        if isinstance(ep, dict):
            for k in ("task", "prompt", "instruction"):
                if k in ep:
                    return str(ep[k])
    return default_task


def main(args: Args) -> None:
    data_dir = Path(args.data_dir)
    h5_paths = _iter_hdf5_files(data_dir)
    if args.max_episodes and args.max_episodes > 0:
        h5_paths = h5_paths[: args.max_episodes]

    task_info_path = data_dir / "task_info.json"
    task_info = {}
    if task_info_path.exists():
        task_info = json.loads(task_info_path.read_text())

    # Clean output.
    out_dir = HF_LEROBOT_HOME / args.repo_id
    if out_dir.exists():
        # LeRobotDataset.create expects clean dir.
        import shutil

        shutil.rmtree(out_dir)

    # Define LeRobot features matching mm22_5_4_lerobot structure.
    # Use video format with nested dict keys and (channels, height, width) shape.
    features = {
        "images_dict.head.rgb": {
            "dtype": "video",
            "shape": (3, args.image_height, args.image_width),
            "names": ["channel", "height", "width"],
        },
        "images_dict.left.rgb": {
            "dtype": "video",
            "shape": (3, args.image_height, args.image_width),
            "names": ["channel", "height", "width"],
        },
        "images_dict.right.rgb": {
            "dtype": "video",
            "shape": (3, args.image_height, args.image_width),
            "names": ["channel", "height", "width"],
        },
        "joints_dict.joints_position_state": {
            "dtype": "float64",
            "shape": (25,),
            "names": [
                "chassis_state_0",
                "chassis_state_1",
                "chassis_state_2",
                "torso_state_0",
                "torso_state_1",
                "torso_state_2",
                "torso_state_3",
                "left_arm_state_0",
                "left_arm_state_1",
                "left_arm_state_2",
                "left_arm_state_3",
                "left_arm_state_4",
                "left_arm_state_5",
                "left_arm_state_6",
                "left_gripper_state_0",
                "right_arm_state_0",
                "right_arm_state_1",
                "right_arm_state_2",
                "right_arm_state_3",
                "right_arm_state_4",
                "right_arm_state_5",
                "right_arm_state_6",
                "right_gripper_state_0",
                "head_state_yaw",
                "head_state_pitch",
            ],
        },
        "joints_dict.joints_position_command": {
            "dtype": "float64",
            "shape": (25,),
            "names": [
                "chassis_action_0",
                "chassis_action_1",
                "chassis_action_2",
                "torso_action_0",
                "torso_action_1",
                "torso_action_2",
                "torso_action_3",
                "left_arm_action_0",
                "left_arm_action_1",
                "left_arm_action_2",
                "left_arm_action_3",
                "left_arm_action_4",
                "left_arm_action_5",
                "left_arm_action_6",
                "left_gripper_action_0",
                "right_arm_action_0",
                "right_arm_action_1",
                "right_arm_action_2",
                "right_arm_action_3",
                "right_arm_action_4",
                "right_arm_action_5",
                "right_arm_action_6",
                "right_gripper_action_0",
                "head_action_yaw",
                "head_action_pitch",
            ],
        },
    }

    # Optionally drop cameras.
    if not args.use_head_rgbd:
        features.pop("images_dict.head.rgb")
    if not args.use_left_wrist_rgbd:
        features.pop("images_dict.left.rgb")
    if not args.use_right_wrist_rgbd:
        features.pop("images_dict.right.rgb")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="astribot",
        fps=30,  # camera rate; actions are chunked during training anyway
        features=features,
        use_videos=True,  # Use video format to match mm22_5_4_lerobot
        image_writer_threads=10,
        image_writer_processes=32,
    )

    print(f"Found {len(h5_paths)} hdf5 episodes")

    for ep_i, h5_path in enumerate(h5_paths):
        task = _episode_task(h5_path, task_info, args.default_task)
        print(f"[{ep_i + 1}/{len(h5_paths)}] converting {h5_path.name} task={task!r}")

        frames_written = 0

        with h5py.File(h5_path, "r") as f:
            seg_ids = _segment_ids_in_h5(f)
            print(f"  segments: {len(seg_ids)} -> {seg_ids[:8]}{'...' if len(seg_ids) > 8 else ''}")

            for seg_i, seg_id in enumerate(seg_ids):
                print(f"  - segment {seg_id} ({seg_i + 1}/{len(seg_ids)})")

                # Load joints.
                try:
                    t_state, state_vecs, t_cmd, cmd_vecs = _read_segment_joint_series(f, seg_id=seg_id)
                except EmptyJointSeriesError as e:
                    print(f"WARNING: {h5_path.name}: skipping seg_id={seg_id} due to empty joints: {e}")
                    continue

                # Load cameras for this segment (raw compressed streams).
                # Map internal camera names to output feature keys.
                cams: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                cam_key_map: dict[str, str] = {}  # Maps internal name to output key
                if args.use_head_rgbd:
                    cams["head_rgbd"] = _read_segment_camera_raw(f, seg_id=seg_id, camera="head_rgbd")
                    cam_key_map["head_rgbd"] = "images_dict.head.rgb"
                if args.use_left_wrist_rgbd:
                    cams["left_wrist_rgbd"] = _read_segment_camera_raw(f, seg_id=seg_id, camera="left_wrist_rgbd")
                    cam_key_map["left_wrist_rgbd"] = "images_dict.left.rgb"
                if args.use_right_wrist_rgbd:
                    cams["right_wrist_rgbd"] = _read_segment_camera_raw(f, seg_id=seg_id, camera="right_wrist_rgbd")
                    cam_key_map["right_wrist_rgbd"] = "images_dict.right.rgb"

                if not cams:
                    raise ValueError("No cameras enabled; at least one camera is required for pi05.")

                # Choose main camera timeline.
                main_cam_internal = "head_rgbd" if "head_rgbd" in cams else next(iter(cams.keys()))
                t_main = cams[main_cam_internal][0]

                # Subsample / limit.
                frame_indices = np.arange(len(t_main), dtype=np.int64)
                if args.frame_stride > 1:
                    frame_indices = frame_indices[:: args.frame_stride]
                if args.max_frames_per_segment and args.max_frames_per_segment > 0:
                    frame_indices = frame_indices[: args.max_frames_per_segment]

                # Precompute offsets per camera.
                cam_offsets: dict[str, np.ndarray] = {
                    cam_name: _jpeg_offsets(sizes) for cam_name, (_t, sizes, _d) in cams.items()
                }

                # Precompute aligned camera indices for each output frame.
                aligned: dict[str, np.ndarray] = {}
                for cam_name, (t_cam, _sizes, _data) in cams.items():
                    if len(t_cam) == len(t_main) and np.allclose(t_cam, t_main):
                        aligned[cam_name] = frame_indices.copy()
                    else:
                        aligned[cam_name] = np.asarray(
                            [_nearest_index(t_cam, float(t_main[int(k)])) for k in frame_indices], dtype=np.int64
                        )

                # Optionally decode/resize in parallel per camera (cache unique frame indices).
                decoded_cache: dict[str, dict[int, np.ndarray]] = {cam_name: {} for cam_name in cams}

                if args.decode_workers and args.decode_workers > 0:
                    backend = args.decode_backend.lower().strip()
                    if backend not in ("process", "thread"):
                        raise ValueError(f"Unsupported --decode-backend: {args.decode_backend}")

                    Executor = (
                        concurrent.futures.ProcessPoolExecutor
                        if backend == "process"
                        else concurrent.futures.ThreadPoolExecutor
                    )

                    for cam_name, (_t_cam, _sizes, data) in cams.items():
                        unique_indices = np.unique(aligned[cam_name])
                        offsets = cam_offsets[cam_name]

                        def jobs():
                            for idx in unique_indices.tolist():
                                start = int(offsets[idx])
                                end = int(offsets[idx + 1])
                                yield idx, bytes(data[start:end])

                        print(
                            f"    decoding {cam_name}: unique_frames={len(unique_indices)} backend={backend} workers={args.decode_workers}"
                        )
                        with Executor(max_workers=args.decode_workers) as ex:
                            fut_to_idx = {
                                ex.submit(_decode_resize_rgb, jpeg_bytes, args.image_width, args.image_height): idx
                                for idx, jpeg_bytes in jobs()
                            }
                            it = concurrent.futures.as_completed(fut_to_idx)
                            if tqdm is not None:
                                it = tqdm.tqdm(it, total=len(fut_to_idx), desc=f"    decode({cam_name})", unit="img")
                            for fut in it:
                                idx = fut_to_idx[fut]
                                decoded_cache[cam_name][idx] = fut.result()

                # Progress bar for writing frames.
                iterator = frame_indices
                if tqdm is not None:
                    iterator = tqdm.tqdm(frame_indices, desc=f"    write(seg={seg_id})", unit="frame")

                for j, k in enumerate(iterator):
                    tk = float(t_main[int(k)])
                    i_state = _nearest_index(t_state, tk)
                    i_cmd = _nearest_index(t_cmd, tk)

                    actions = cmd_vecs[i_cmd].astype(np.float64)
                    state = state_vecs[i_state].astype(np.float64)
                    # if np.allclose(actions, 0.0):
                    #     # Skip "idle" frames where the aligned action vector is all zeros.
                    #     # Do this before decoding any camera images.
                    #     continue

                    frame = {
                        "joints_dict.joints_position_state": state,
                        "joints_dict.joints_position_command": actions,
                        "task": task,
                    }

                    for cam_name_internal, (_t_cam, _sizes, data) in cams.items():
                        idx = int(aligned[cam_name_internal][j])
                        cam_key_output = cam_key_map[cam_name_internal]

                        if idx in decoded_cache[cam_name_internal]:
                            img_rgb = decoded_cache[cam_name_internal][idx]
                        else:
                            # Fallback: decode inline.
                            bgr = _decode_jpeg_at(data, cam_offsets[cam_name_internal], idx)
                            bgr = _resize_bgr(bgr, args.image_width, args.image_height)
                            img_rgb = bgr[..., ::-1]

                        # Convert from HWC to CHW format for video storage
                        img_chw = np.transpose(img_rgb, (2, 0, 1))
                        frame[cam_key_output] = img_chw

                    dataset.add_frame(frame)
                    frames_written += 1

        # In this LeRobot version, episodes are saved without args; task strings are passed via per-frame `task`.
        if frames_written > 0:
            dataset.save_episode()
        else:
            print(f"WARNING: {h5_path.name}: no nonzero-action frames; skipping episode save")

    # LeRobot API differs by version; some versions don't have consolidate().
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()  # type: ignore[attr-defined]
    print(f"Wrote LeRobot dataset to: {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
