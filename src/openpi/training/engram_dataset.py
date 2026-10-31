"""Local LeRobot v3 reader for the Dex W1 Engram sandwich demonstrations.

The OpenPI checkout pins LeRobot v2.1, whose loader cannot read v3 metadata.
This adapter uses v3 episode locations and the source timestamps directly and
keeps the original episode timeline intact even when an anchor is invalid.
"""

from __future__ import annotations

from collections import OrderedDict
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal, SupportsIndex

import av
import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize

STATE_KEY = "observation.joint_states"
ACTION_KEY = "action.joint_actions"
CAMERA_KEY = "observation.images.kfc_compressed"
LEFT_STATE = "observation.raw.hand_left_fb_position"
RIGHT_STATE = "observation.raw.hand_right_fb_position"
LEFT_ACTION = "action.raw.hand_left_cmd_values"
RIGHT_ACTION = "action.raw.hand_right_cmd_values"
LEFT_MODE = "action.raw.hand_left_cmd_mode"
RIGHT_MODE = "action.raw.hand_right_cmd_mode"
JOINT_NAMES = (
    "ANKLE",
    "KNEE",
    "BUTTOCK",
    "WAIST",
    "NECK1",
    "NECK2",
    "LEFT_J1",
    "LEFT_J2",
    "LEFT_J3",
    "LEFT_J4",
    "LEFT_J5",
    "LEFT_J6",
    "LEFT_J7",
    "RIGHT_J1",
    "RIGHT_J2",
    "RIGHT_J3",
    "RIGHT_J4",
    "RIGHT_J5",
    "RIGHT_J6",
    "RIGHT_J7",
)
# Source packages selected by SHA-256 ranking within each major task label,
# for 23/450 held-out episodes while keeping recording packages disjoint.
VAL_PACKAGE_IDS = frozenset(
    {
        "5af748f3-6e02-4506-9938-90bb165c0953",
        "07bc4327-6a40-42f0-8a30-35168b9d73b2",
    }
)
STATS_CONTRACT = "engram_w1_physical_delta20_gripper_abs_quantilefloor_pruned_exactnoise_v4"
_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    STATE_KEY,
    ACTION_KEY,
    LEFT_STATE,
    RIGHT_STATE,
    LEFT_ACTION,
    RIGHT_ACTION,
    LEFT_MODE,
    RIGHT_MODE,
)


def floor_quantile_ranges(stats: dict[str, normalize.NormStats]) -> dict[str, normalize.NormStats]:
    """Floor the full q01--q99 span, in physical units, without clipping values.

    The first 20 channels are radians; the last two are gripper percentages.
    Persisting the adjusted quantiles lets OpenPI's normal inverse remain exact.
    """
    minimum = np.asarray([0.1] * 20 + [5.0] * 2)
    result = {}
    for key in ("state", "actions"):
        value = stats[key]
        q01, q99 = np.asarray(value.q01), np.asarray(value.q99)
        if q01.shape != (22,) or q99.shape != (22,) or not np.isfinite(q01).all() or not np.isfinite(q99).all():
            raise ValueError(f"Invalid Engram {key} quantiles")
        if np.any(q99 < q01):
            raise ValueError(f"Reversed Engram {key} quantiles")
        midpoint = (q01 + q99) / 2
        half_width = np.maximum(q99 - q01, minimum) / 2
        result[key] = dataclasses.replace(value, q01=midpoint - half_width, q99=midpoint + half_width)
    return result


def stats_identity(
    root: str | Path, split: str, action_horizon: int, prune_threshold_rad: float, gripper_threshold_percent: float
) -> dict[str, Any]:
    root = Path(root)
    identity = {
        "contract": STATS_CONTRACT,
        "split": "train" if split == "val" else split,
        "action_horizon": action_horizon,
        "prune_threshold_rad": prune_threshold_rad,
        "gripper_threshold_percent": gripper_threshold_percent,
        "info_sha256": hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest(),
    }
    # A metadata inventory catches accidental replacement of data shards without
    # reading many gigabytes of unused raw modalities on every startup. This is
    # deliberately documented as file identity, not a full content checksum.
    inventory = hashlib.sha256()
    shard_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not shard_paths:
        raise ValueError("No data shards found for statistics provenance")
    for path in shard_paths:
        stat = path.stat()
        inventory.update(f"{path.relative_to(root)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    identity["data_inventory_sha256"] = inventory.hexdigest()
    metadata = hashlib.sha256()
    metadata_paths = [root / "meta/tasks.parquet", *sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))]
    for path in metadata_paths:
        metadata.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    identity["task_episode_metadata_sha256"] = metadata.hexdigest()
    if split != "all":
        identity["export_plan_sha256"] = hashlib.sha256((root.parent / "export_plan.json").read_bytes()).hexdigest()
    return identity


def check_stats_identity(
    manifest_path: str | Path,
    root: str | Path,
    split: str,
    action_horizon: int,
    prune_threshold_rad: float,
    gripper_threshold_percent: float,
    *,
    allow_sampled_stats: bool = False,
) -> None:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ValueError(f"Engram statistics provenance missing: {manifest_path}; recompute training stats")
    stored = json.loads(manifest_path.read_text())
    expected = stats_identity(root, split, action_horizon, prune_threshold_rad, gripper_threshold_percent)
    if any(stored.get(key) != value for key, value in expected.items()):
        raise ValueError("Engram statistics do not match this source, split, horizon, or transform; recompute them")
    stats_path = manifest_path.parent / "norm_stats.json"
    if (
        not stats_path.is_file()
        or stored.get("norm_stats_sha256") != hashlib.sha256(stats_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Engram normalization stats do not match the saved manifest; recompute them")
    if stored.get("max_frames") is not None and not allow_sampled_stats:
        raise ValueError("Sampled Engram statistics are diagnostic only; recompute full stats or explicitly opt in")


@dataclasses.dataclass(frozen=True)
class _Episode:
    episode_index: int
    timestamps: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    task_indices: np.ndarray
    anchors: np.ndarray
    video_path: str
    video_start: float


def _vector(table: Any, key: str, width: int) -> np.ndarray:
    try:
        array = np.asarray(table.column(key).to_pylist(), dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or malformed {key}") from exc
    if array.shape != (len(table), width):
        raise ValueError(f"{key} must have shape ({len(table)}, {width}); got {array.shape}")
    return array


def _relative_action_indices(timestamps: np.ndarray, anchor: int, horizon: int, fps: float) -> np.ndarray:
    """Nearest recorded action at each physical 1/fps interval."""
    targets = timestamps[anchor] + np.arange(horizon, dtype=np.float64) / fps
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, anchor, len(timestamps) - 1)
    left = np.maximum(right - 1, anchor)
    choose_left = np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets)
    return np.where(choose_left, left, right)


def _moving_anchors(
    timestamps: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    anchors: np.ndarray,
    horizon: int,
    fps: float,
    threshold: float,
    gripper_threshold_percent: float,
) -> np.ndarray:
    """Keep a chunk when either joints or grippers have a meaningful target change."""
    keep = np.zeros(len(anchors), dtype=bool)
    offsets = np.arange(horizon, dtype=np.float64) / fps
    for start in range(0, len(anchors), 512):
        selected = anchors[start : start + 512]
        targets = timestamps[selected, None] + offsets
        right = np.searchsorted(timestamps, targets, side="left")
        right = np.minimum(np.maximum(right, selected[:, None]), len(timestamps) - 1)
        left = np.maximum(right - 1, selected[:, None])
        indices = np.where(np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets), left, right)
        differences = np.abs(actions[indices, :20] - states[selected, None, :20])
        gripper_differences = np.abs(actions[indices, 20:] - states[selected, None, 20:])
        gripper_moving = (
            gripper_differences > 0
            if gripper_threshold_percent == 0
            else gripper_differences >= gripper_threshold_percent
        )
        keep[start : start + len(selected)] = np.any(differences >= threshold, axis=(1, 2)) | np.any(
            gripper_moving, axis=(1, 2)
        )
    return anchors[keep]


class EngramDataset:
    """Read full successful W1 episodes; reject invalid actions and exclude invalid anchors.

    The 22D layout is 20 joint radians in ``JOINT_NAMES`` order, followed by
    left and right gripper positions in percent. Images are the left half of
    the side-by-side kfc stereo video; wrist cameras are intentionally absent.
    """

    def __init__(
        self,
        root: str | Path,
        action_horizon: int,
        *,
        split: Literal["all", "train", "val"] = "all",
        prune_threshold_rad: float = 0.01,
        gripper_threshold_percent: float = 0.0,
        decode_images: bool = True,
        max_gap_s: float = 0.25,
        video_tolerance_s: float = 0.0001,
        video_cache_size: int = 4,
    ) -> None:
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.decode_images = decode_images
        self.video_tolerance_s = video_tolerance_s
        self.video_cache_size = video_cache_size
        self._video_cache: OrderedDict[str, Any] = OrderedDict()
        if (
            action_horizon < 1
            or max_gap_s <= 0
            or video_tolerance_s <= 0
            or video_cache_size < 1
            or not np.isfinite(prune_threshold_rad)
            or prune_threshold_rad < 0
            or not np.isfinite(gripper_threshold_percent)
            or gripper_threshold_percent < 0
        ):
            raise ValueError("Horizon, gap/tolerance, and video cache size must be positive")
        if split not in ("all", "train", "val"):
            raise ValueError(f"Invalid split: {split}")
        info = json.loads((self.root / "meta/info.json").read_text())
        if info.get("codebase_version") != "v3.0" or info.get("robot_type") != "dex_w1":
            raise ValueError("Expected an Engram Dex W1 LeRobot v3 dataset")
        self.fps = float(info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Dataset FPS must be positive")
        for key in (STATE_KEY, ACTION_KEY):
            if tuple(info["features"][key].get("names") or ()) != JOINT_NAMES:
                raise ValueError(f"Unexpected joint order for {key}")
        task_rows = pq.read_table(self.root / "meta/tasks.parquet").to_pylist()
        self.tasks = {int(row["task_index"]): row["task"] for row in task_rows}
        metadata_paths = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        if not metadata_paths:
            raise ValueError("No episode metadata found")
        metadata = [row for path in metadata_paths for row in pq.read_table(path).to_pylist()]
        selected = [
            row for row in metadata if row.get("completeness") == "FULL" and row.get("annotation_result") == "SUCCESS"
        ]
        if split != "all":
            # Split by source recording package, not episode, to avoid leakage
            # between neighboring task segments from the same recording.
            plan = json.loads((self.root.parent / "export_plan.json").read_text())
            annotation_packages: dict[str, str] = {}
            for item in plan:
                annotation, package = item["annotation_id"], item["data_package_id"]
                if annotation in annotation_packages and annotation_packages[annotation] != package:
                    raise ValueError(f"Annotation {annotation} occurs in multiple packages")
                annotation_packages[annotation] = package

            def package_for(row):
                annotation = row["annotation_id"]
                if annotation not in annotation_packages:
                    raise ValueError(f"No package mapping for annotation {annotation}")
                return annotation_packages[annotation]

            observed_packages = {package_for(row) for row in selected}
            if missing := VAL_PACKAGE_IDS - observed_packages:
                raise ValueError(f"Expected validation packages absent from export plan: {sorted(missing)}")
            selected = [row for row in selected if (package_for(row) in VAL_PACKAGE_IDS) == (split == "val")]
        selected.sort(key=lambda row: int(row["episode_index"]))
        ids = [int(row["episode_index"]) for row in selected]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("No full successful episodes or duplicate episode indices")
        episodes = []
        invalid_anchors = 0
        short_horizon_anchors = 0
        static_anchors = 0
        for row in selected:
            episode_id = int(row["episode_index"])
            data_path = self.root / info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            )
            table = pq.read_table(data_path, columns=list(_COLUMNS), filters=[("episode_index", "=", episode_id)])
            n = len(table)
            if n != int(row["length"]) or n == 0:
                raise ValueError(f"Episode {episode_id}: metadata length {row['length']} != {n} rows")
            frame_indices = np.asarray(table.column("frame_index"), dtype=np.int64)
            if not np.array_equal(frame_indices, np.arange(n)):
                raise ValueError(f"Episode {episode_id}: frame indices are not contiguous and ordered")
            timestamps = np.asarray(table.column("timestamp"), dtype=np.float64)
            gaps = np.diff(timestamps)
            if not np.isfinite(timestamps).all() or np.any(gaps <= 0) or np.any(gaps > max_gap_s):
                raise ValueError(f"Episode {episode_id}: non-monotonic or missing timestamps (max gap {max_gap_s}s)")
            states = np.concatenate(
                [_vector(table, STATE_KEY, 20), _vector(table, LEFT_STATE, 1), _vector(table, RIGHT_STATE, 1)], axis=1
            )
            actions = np.concatenate(
                [_vector(table, ACTION_KEY, 20), _vector(table, LEFT_ACTION, 1), _vector(table, RIGHT_ACTION, 1)],
                axis=1,
            )
            modes = np.concatenate([_vector(table, LEFT_MODE, 1), _vector(table, RIGHT_MODE, 1)], axis=1)
            if not np.isfinite(actions).all() or not np.all(modes == 0):
                raise ValueError(f"Episode {episode_id}: nonfinite action or gripper mode other than POSITION")
            if np.any((actions[:, 20:] < 0) | (actions[:, 20:] > 100)):
                raise ValueError(f"Episode {episode_id}: gripper action outside [0, 100] percent")
            valid = np.isfinite(states).all(axis=1) & np.all((states[:, 20:] >= 0) & (states[:, 20:] <= 100), axis=1)
            invalid_anchors += n - int(valid.sum())
            # The standard Pi0.5 loss does not mask padded actions: only train
            # anchors with a complete future horizon in physical time.
            enough_future = timestamps + (action_horizon - 1) / self.fps <= timestamps[-1] + 1e-9
            short_horizon_anchors += int(np.sum(valid & ~enough_future))
            valid &= enough_future
            anchors = np.flatnonzero(valid)
            if split != "val" and prune_threshold_rad > 0:
                moving = _moving_anchors(
                    timestamps,
                    states,
                    actions,
                    anchors,
                    action_horizon,
                    self.fps,
                    prune_threshold_rad,
                    gripper_threshold_percent,
                )
                static_anchors += len(anchors) - len(moving)
                anchors = moving
            task_indices = np.asarray(table.column("task_index"), dtype=np.int64)
            if not set(task_indices.tolist()).issubset(self.tasks):
                raise ValueError(f"Episode {episode_id}: task indices absent from task metadata")
            prefix = f"videos/{CAMERA_KEY}"
            video_path = self.root / info["video_path"].format(
                video_key=CAMERA_KEY,
                chunk_index=int(row[f"{prefix}/chunk_index"]),
                file_index=int(row[f"{prefix}/file_index"]),
            )
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            episodes.append(
                _Episode(
                    episode_id,
                    timestamps,
                    states,
                    actions,
                    task_indices,
                    anchors,
                    str(video_path),
                    float(row[f"{prefix}/from_timestamp"]),
                )
            )
        self.episodes = episodes
        self.invalid_anchors = invalid_anchors
        self.short_horizon_anchors = short_horizon_anchors
        self.static_anchors = static_anchors
        self.offsets = np.cumsum([0, *[len(ep.anchors) for ep in episodes]], dtype=np.int64)
        if self.offsets[-1] == 0:
            raise ValueError("No valid state anchors in selected episodes")
        logging.info(
            "Engram %s: %d episodes, %d anchors, %d invalid state, %d short-horizon, %d pruned static",
            split,
            len(episodes),
            len(self),
            invalid_anchors,
            short_horizon_anchors,
            static_anchors,
        )

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        for container in self._video_cache.values():
            container.close()
        self._video_cache.clear()

    def _read_image(self, episode: _Episode, timestamp: float) -> np.ndarray:
        if not self.decode_images:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        path = episode.video_path
        container = self._video_cache.pop(path, None)
        if container is None:
            container = av.open(path)
        self._video_cache[path] = container
        while len(self._video_cache) > self.video_cache_size:
            _, old = self._video_cache.popitem(last=False)
            old.close()
        stream = container.streams.video[0]
        target = episode.video_start + timestamp
        container.seek(max(0, int(target / float(stream.time_base))), stream=stream, backward=True, any_frame=False)
        previous = None
        selected = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            if frame_time >= target:
                selected = min(
                    (x for x in (previous, (frame_time, frame)) if x is not None), key=lambda x: abs(x[0] - target)
                )
                break
            previous = (frame_time, frame)
        if selected is None:
            selected = previous
        if selected is None or abs(selected[0] - target) > self.video_tolerance_s:
            got = None if selected is None else selected[0]
            raise ValueError(f"Episode {episode.episode_index}: no camera frame near {target:.6f}s (got {got})")
        image = selected[1].to_ndarray(format="rgb24")
        if image.ndim != 3 or image.shape[1] % 2:
            raise ValueError(f"Expected side-by-side stereo image, got {image.shape}")
        return image[:, : image.shape[1] // 2].copy()

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        ep_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        episode = self.episodes[ep_idx]
        anchor = int(episode.anchors[idx - self.offsets[ep_idx]])
        indices = _relative_action_indices(episode.timestamps, anchor, self.action_horizon, self.fps)
        return {
            "observation/head_rgbd": self._read_image(episode, float(episode.timestamps[anchor])),
            "observation/joint_position": episode.states[anchor].copy(),
            "actions": episode.actions[indices].copy(),
            "prompt": self.tasks[int(episode.task_indices[anchor])],
        }
