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
STATS_CONTRACT = "engram_w1_delta20_mixed_end_effector_mask_stats_v6_joint_filter_unlabeled"
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


def _read_source_manifest(path: str | Path) -> dict[str, Any]:
    """Validate a pinned collection of exports and their global package split."""
    spec = json.loads(Path(path).read_text())
    if spec.get("format") != "engram-multisource-v1" or not spec.get("sources"):
        raise ValueError("Expected an engram-multisource-v1 manifest with sources")
    roots: set[str] = set()
    annotations: set[str] = set()
    heldout: set[str] = set()
    inventories = []
    for source in spec["sources"]:
        root = Path(source["root"])
        canonical = str(root.resolve())
        if canonical in roots or not isinstance(source.get("has_gripper"), bool):
            raise ValueError("Duplicate source root or unspecified end-effector label availability")
        roots.add(canonical)
        info = json.loads((root / "meta/info.json").read_text())
        grip_keys = {LEFT_STATE, RIGHT_STATE, LEFT_ACTION, RIGHT_ACTION, LEFT_MODE, RIGHT_MODE}
        present = grip_keys.intersection(info["features"])
        if present != (grip_keys if source["has_gripper"] else set()):
            raise ValueError(f"Incomplete or unexpected end-effector schema in {root}")
        paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
        rows = [row for p in paths for row in pq.read_table(p).to_pylist()]
        selected = [r for r in rows if r.get("completeness") == "FULL" and r.get("annotation_result") == "SUCCESS"]
        if len(selected) != source.get("expected_episodes"):
            raise ValueError(f"Unexpected selected episode count in {root}")
        mapping = {}
        for entry in json.loads((root.parent / "export_plan.json").read_text()):
            annotation, package = entry["annotation_id"], entry["data_package_id"]
            if annotation in mapping and mapping[annotation] != package:
                raise ValueError(f"Ambiguous package mapping for {annotation}")
            mapping[annotation] = package
        source_packages = set()
        for row in selected:
            annotation = row["annotation_id"]
            if annotation in annotations:
                raise ValueError(f"Duplicate annotation across selected exports: {annotation}")
            annotations.add(annotation)
            if annotation not in mapping:
                raise ValueError(f"Missing source package for {annotation}")
            source_packages.add(mapping[annotation])
        local_heldout = set(source.get("validation_packages", ()))
        if not local_heldout.issubset(source_packages):
            raise ValueError(f"Unknown validation package in {root}")
        heldout.update(local_heldout)
        inventories.append(source_packages)
    # A package appearing in several exports is held out from all of them.
    for source, packages in zip(spec["sources"], inventories, strict=True):
        source["validation_packages"] = sorted(packages & heldout)
        if not source["validation_packages"]:
            raise ValueError(f"No validation package selected in {source['root']}")
    return spec


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
    if root.is_file():
        spec = _read_source_manifest(root)
        return {
            "contract": STATS_CONTRACT,
            "split": "train" if split == "val" else split,
            "action_horizon": action_horizon,
            "prune_threshold_rad": prune_threshold_rad,
            "gripper_threshold_percent": gripper_threshold_percent,
            "source_manifest_sha256": hashlib.sha256(root.read_bytes()).hexdigest(),
            "source_spec": spec,
            "sources": [
                stats_identity(item["root"], split, action_horizon, prune_threshold_rad, gripper_threshold_percent)
                for item in spec["sources"]
            ],
        }
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
    *,
    has_gripper: bool = True,
) -> np.ndarray:
    """Keep joint motion or, when labeled, meaningful gripper target change."""
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
        moving = np.any(differences >= threshold, axis=(1, 2))
        if has_gripper:
            gripper_differences = np.abs(actions[indices, 20:] - states[selected, None, 20:])
            gripper_moving = (
                gripper_differences > 0
                if gripper_threshold_percent == 0
                else gripper_differences >= gripper_threshold_percent
            )
            moving |= np.any(gripper_moving, axis=(1, 2))
        keep[start : start + len(selected)] = moving
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
        has_gripper: bool = True,
        validation_packages: frozenset[str] | None = None,
        video_mode: Literal["original", "proxy_640"] = "original",
        proxy_root: str | Path | None = None,
        proxy_subset_only: bool = False,
    ) -> None:
        self.root = Path(root)
        self.has_gripper = has_gripper
        if video_mode not in ("original", "proxy_640"):
            raise ValueError("Invalid video mode")
        self.video_mode = video_mode
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
            heldout = VAL_PACKAGE_IDS if validation_packages is None else validation_packages
            if missing := heldout - observed_packages:
                raise ValueError(f"Expected validation packages absent from export plan: {sorted(missing)}")
            selected = [row for row in selected if (package_for(row) in heldout) == (split == "val")]
        selected.sort(key=lambda row: int(row["episode_index"]))
        if proxy_subset_only:
            from openpi.training import engram_video_proxy

            prefix = f"videos/{CAMERA_KEY}"
            def has_proxy(row):
                original = self.root / info["video_path"].format(video_key=CAMERA_KEY,
                    chunk_index=int(row[f"{prefix}/chunk_index"]), file_index=int(row[f"{prefix}/file_index"]))
                proxy = engram_video_proxy.proxy_path(self.root, original, None if proxy_root is None else Path(proxy_root))
                return proxy.is_file() and engram_video_proxy.sidecar_path(proxy).is_file()
            before = len(selected)
            selected = [row for row in selected if has_proxy(row)]
            logging.warning("PROFILING SUBSET: %d of %d %s episodes have completed proxies", len(selected), before, split)
        ids = [int(row["episode_index"]) for row in selected]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("No full successful episodes or duplicate episode indices")
        episodes = []
        invalid_anchors = 0
        short_horizon_anchors = 0
        static_anchors = 0
        video_identity = hashlib.sha256() if video_mode == "proxy_640" or proxy_subset_only else None
        for row in selected:
            episode_id = int(row["episode_index"])
            data_path = self.root / info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            )
            columns = _COLUMNS if has_gripper else _COLUMNS[:6]
            table = pq.read_table(data_path, columns=list(columns), filters=[("episode_index", "=", episode_id)])
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
            states = _vector(table, STATE_KEY, 20)
            actions = _vector(table, ACTION_KEY, 20)
            if has_gripper:
                states = np.concatenate([states, _vector(table, LEFT_STATE, 1), _vector(table, RIGHT_STATE, 1)], axis=1)
                actions = np.concatenate([actions, _vector(table, LEFT_ACTION, 1), _vector(table, RIGHT_ACTION, 1)], axis=1)
                modes = np.concatenate([_vector(table, LEFT_MODE, 1), _vector(table, RIGHT_MODE, 1)], axis=1)
            else:
                # Storage placeholders are never valid labels and never become
                # physical feedback tokens; `action_mask` records availability.
                states = np.pad(states, ((0, 0), (0, 2)))
                actions = np.pad(actions, ((0, 0), (0, 2)))
                modes = np.zeros((n, 2), dtype=np.float32)
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
            # Apply the joint threshold to every source. The gripper-only
            # exception applies only where end-effector labels are available.
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
                    has_gripper=has_gripper,
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
            if video_mode == "proxy_640" or proxy_subset_only:
                from openpi.training import engram_video_proxy

                original = video_path
                proxy_video_path = engram_video_proxy.proxy_path(
                    self.root, original, None if proxy_root is None else Path(proxy_root)
                )
                record = engram_video_proxy.validate_sidecar(self.root, original, proxy_video_path, episode_id, n)
                video_identity.update(f"{self.root.parent.name}\0{episode_id}\0{record['source_sha256']}\0{record['proxy_sha256']}\n".encode())
                if video_mode == "proxy_640":
                    video_path = proxy_video_path
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
        self.video_identity = None if video_identity is None else video_identity.hexdigest()
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
        if self.video_mode == "proxy_640":
            if image.shape != (360, 640, 3):
                raise ValueError(f"Expected already-cropped 640x360 RGB proxy, got {image.shape}")
            return image
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
            "action_mask": np.array([True] * 20 + [self.has_gripper] * 2),
            "prompt": self.tasks[int(episode.task_indices[anchor])],
        }


class EngramMultiDataset:
    """Concatenate pinned exports while preserving recording-package split boundaries."""

    def __init__(self, manifest: str | Path, action_horizon: int, **kwargs) -> None:
        spec = _read_source_manifest(manifest)
        self.datasets = [
            EngramDataset(
                item["root"],
                action_horizon,
                has_gripper=item["has_gripper"],
                validation_packages=frozenset(item["validation_packages"]),
                **kwargs,
            )
            for item in spec["sources"]
        ]
        self.offsets = np.cumsum([0, *[len(dataset) for dataset in self.datasets]], dtype=np.int64)
        self.video_identity = (
            None if self.datasets[0].video_identity is None else hashlib.sha256(
                "\n".join(dataset.video_identity for dataset in self.datasets).encode()
            ).hexdigest()
        )

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_index = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        return self.datasets[dataset_index][idx - self.offsets[dataset_index]]

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()


def compute_training_stats(
    dataset: EngramDataset | EngramMultiDataset, *, seed: int, max_frames: int | None = None, batch_size: int = 512
) -> tuple[dict[str, normalize.NormStats], dict[str, int]]:
    """Fit all retained anchors; never include unlabeled end-effector placeholders.

    Each anchor contributes one state and its complete future action horizon.
    The same joint-delta and Gaussian-noise distribution used by training is
    applied in vectorized batches. Quantiles use OpenPI's running histograms.
    """
    if batch_size < 1 or (max_frames is not None and max_frames < 2):
        raise ValueError("Statistics batch size must be positive and max_frames at least two")
    rng = np.random.default_rng(seed)
    count = len(dataset)
    if max_frames is None or max_frames >= count:
        selected = None
    else:
        selected = np.sort(rng.choice(count, size=max_frames, replace=False))
    accumulators = {key: normalize.RunningStats() for key in ("joint_state", "joint_action", "grip_state", "grip_action")}
    parts = dataset.datasets if isinstance(dataset, EngramMultiDataset) else [dataset]
    global_offset = 0
    used = 0
    grip_used = 0
    episode_count = 0
    for part in parts:
        for episode, begin in zip(part.episodes, part.offsets[:-1], strict=True):
            lo = global_offset + int(begin)
            hi = lo + len(episode.anchors)
            if selected is None:
                anchors = episode.anchors
            else:
                subset = selected[np.searchsorted(selected, lo) : np.searchsorted(selected, hi)] - lo
                anchors = episode.anchors[subset]
            for start in range(0, len(anchors), batch_size):
                anchors_batch = anchors[start : start + batch_size]
                offsets = np.arange(part.action_horizon, dtype=np.float64) / part.fps
                targets = episode.timestamps[anchors_batch, None] + offsets
                right = np.searchsorted(episode.timestamps, targets, side="left")
                right = np.minimum(np.maximum(right, anchors_batch[:, None]), len(episode.timestamps) - 1)
                left = np.maximum(right - 1, anchors_batch[:, None])
                indices = np.where(
                    np.abs(episode.timestamps[left] - targets) <= np.abs(episode.timestamps[right] - targets), left, right
                )
                state = episode.states[anchors_batch].copy()
                action = episode.actions[indices].copy()
                action[..., :20] -= state[:, None, :20]
                noise = rng.normal(0, 0.001, size=(len(anchors_batch), 20)).astype(np.float32)
                state[:, :20] += noise
                action[..., :20] -= noise[:, None, :]
                accumulators["joint_state"].update(state[:, :20])
                accumulators["joint_action"].update(action[..., :20])
                used += len(anchors_batch)
                if part.has_gripper:
                    accumulators["grip_state"].update(state[:, 20:])
                    accumulators["grip_action"].update(action[..., 20:])
                    grip_used += len(anchors_batch)
            episode_count += 1
            if episode_count % 20 == 0:
                print(f"Engram stats: {episode_count} episodes, {used} anchors, {grip_used} labeled-gripper anchors", flush=True)
        global_offset += len(part)
    if used != (len(selected) if selected is not None else count):
        raise ValueError("Statistics did not cover the requested anchors exactly")
    if grip_used < 2:
        raise ValueError("Cannot fit gripper statistics without labeled training anchors")
    result = {}
    for output_key, joint_key, grip_key in (
        ("state", "joint_state", "grip_state"), ("actions", "joint_action", "grip_action")
    ):
        joint = accumulators[joint_key].get_statistics()
        grip = accumulators[grip_key].get_statistics()
        result[output_key] = normalize.NormStats(
            **{name: np.concatenate([getattr(joint, name), getattr(grip, name)]) for name in ("mean", "std", "q01", "q99")}
        )
    return floor_quantile_ranges(result), {"used_anchors": used, "labeled_gripper_anchors": grip_used, "total_anchors": count}
