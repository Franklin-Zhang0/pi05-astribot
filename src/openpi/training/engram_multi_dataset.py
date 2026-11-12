"""LeRobot-v3 reader for the deduplicated mixed Engram embodiment corpus."""

from __future__ import annotations

# Reuse the legacy timestamp/horizon implementation to keep recipes consistent.
# ruff: noqa: SLF001
from collections import OrderedDict
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import SupportsIndex

import av
import numpy as np
import pyarrow.parquet as pq

from openpi.policies.engram_multi_policy import camera_size
from openpi.policies.engram_multi_policy import prepare_camera
from openpi.shared import normalize
from openpi.shared.engram_multi import ACTION_DIM
from openpi.shared.engram_multi import EMBODIMENTS
from openpi.shared.engram_multi import HAND_START
from openpi.shared.engram_multi import ROUTE_NAMES
from openpi.shared.engram_multi import ROUTES
from openpi.shared.engram_multi import RouteStats
from openpi.shared.engram_multi import load_route_stats
from openpi.shared.engram_multi import physical_mask
from openpi.shared.engram_multi import route_id
from openpi.training import engram_dataset as legacy

MANIFEST_FORMAT = "engram-multi-embodiment-v1"
STATS_CONTRACT = "engram_multi_delta_joints_absolute_hands_v1"
CAMERAS = {
    "dex_w1": {"head": "kfc_compressed", "left_wrist": "camera_l", "right_wrist": "camera_r"},
    "galbot": {"head": "head_left_camera", "left_wrist": "arm_left_camera", "right_wrist": "arm_right_camera"},
}
SLOTS = {"head": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}


def read_manifest(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if value.get("format") != MANIFEST_FORMAT or not isinstance(value.get("sources"), list):
        raise ValueError("Expected an engram-multi-embodiment-v1 source manifest")
    roots = set()
    for source in value["sources"]:
        root = str(Path(source["root"]).resolve())
        if root in roots:
            raise ValueError(f"Duplicate source root: {root}")
        roots.add(root)
        route_id(source["robot_type"], source["end_effector_type"])
        if "selected_episode_indices" not in source:
            raise ValueError("A deduplicated explicit selected_episode_indices list is required")
        selected = source["selected_episode_indices"]
        if (
            not isinstance(selected, list)
            or any(type(index) is not int or index < 0 for index in selected)
            or len(selected) != len(set(selected))
        ):
            raise ValueError("Selected episode indices must be unique")
    return value


def stats_identity(
    manifest_path: str | Path,
    split: str,
    action_horizon: int,
    prune_threshold_rad: float,
    hand_threshold_percent: float,
) -> dict:
    """Bind normalization to exact source metadata and explicit episode selection."""
    path = Path(manifest_path)
    spec = read_manifest(path)
    sources = []
    for source in spec["sources"]:
        if not source["selected_episode_indices"]:
            continue
        root = Path(source["root"])
        digest = hashlib.sha256()
        paths = [
            root / "meta/info.json",
            root / "meta/tasks.parquet",
            root.parent / "export_plan.json",
            *sorted((root / "meta/episodes").glob("chunk-*/*.parquet")),
        ]
        for item in paths:
            if not item.is_file():
                raise ValueError(f"Missing source provenance file: {item}")
            digest.update(item.name.encode() + b"\0" + item.read_bytes())
        inventory = hashlib.sha256()
        for item in sorted((root / "data").glob("chunk-*/*.parquet")):
            inventory.update(f"{item.relative_to(root)}\0{item.stat().st_size}\n".encode())
        sources.append(
            {
                "source_id": source.get("source_id", root.parent.name),
                "metadata_sha256": digest.hexdigest(),
                "data_size_inventory_sha256": inventory.hexdigest(),
            }
        )
    return {
        "contract": STATS_CONTRACT,
        "format": MANIFEST_FORMAT,
        "split": "train" if split == "val" else split,
        "action_horizon": action_horizon,
        "prune_threshold_rad": prune_threshold_rad,
        "hand_threshold_percent": hand_threshold_percent,
        "model_action_dim": ACTION_DIM,
        "route_names": list(ROUTE_NAMES),
        "hand_availability": "finite_current_feedback_and_full_horizon_position_targets_v1",
        "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_spec": spec,
        "sources": sources,
    }


def check_stats_identity(
    stats_manifest_path: str | Path,
    manifest_path: str | Path,
    split: str,
    action_horizon: int,
    prune_threshold_rad: float,
    hand_threshold_percent: float,
    *,
    allow_sampled_stats: bool = False,
) -> None:
    p = Path(stats_manifest_path)
    saved = json.loads(p.read_text())
    expected = stats_identity(manifest_path, split, action_horizon, prune_threshold_rad, hand_threshold_percent)
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("Multi-embodiment statistics differ from source/split/transform identity")
    filename = saved.get("stats_filename", "engram_multi_stats.json")
    stats = p.parent / filename
    if not stats.is_file() or hashlib.sha256(stats.read_bytes()).hexdigest() != saved.get("stats_sha256"):
        raise ValueError("Typed normalization file differs from its saved manifest")
    if saved.get("max_frames") is not None and not allow_sampled_stats:
        raise ValueError("Sampled statistics are diagnostic; fit full training statistics")
    routes = load_route_stats(stats)
    used, sampled, total = (saved.get(key) for key in ("used_anchors", "sampled_anchors", "total_anchors"))
    if any(type(count) is not int or count <= 0 for count in (used, sampled, total)):
        raise ValueError("Statistics coverage requires positive integer anchor counts")
    if used != sampled or used > total or (saved.get("max_frames") is None and used != total):
        raise ValueError("Statistics coverage does not include all requested training anchors")
    route_counts = saved.get("routes", {})
    if set(route_counts) != set(ROUTE_NAMES):
        raise ValueError("Statistics coverage omits an embodiment route")
    for index, name in enumerate(ROUTE_NAMES):
        count = route_counts[name]
        anchors, hands = count.get("anchors"), count.get("labeled_hand_anchors")
        if (
            type(anchors) is not int
            or anchors < 2
            or type(hands) is not int
            or not 0 <= hands <= anchors
            or (index < 3 and hands < 2)
            or (index == 3 and hands != 0)
        ):
            raise ValueError(f"Invalid route or hand-label statistics coverage for {name}")
        if count != routes[name].counts:
            raise ValueError("Saved normalization and provenance coverage counts disagree")
        expected_mask = physical_mask(index, has_hand_labels=hands > 0)
        if not np.array_equal(routes[name].supervised_dims, expected_mask):
            raise ValueError("Normalization availability mask disagrees with measured label coverage")
    if sum(count["anchors"] for count in route_counts.values()) != used:
        raise ValueError("Statistics route counts do not sum to the retained anchor total")


@dataclasses.dataclass
class Episode:
    source_id: str
    root: Path
    episode_index: int
    annotation_id: str
    package_id: str
    embodiment_id: int
    fps: float
    timestamps: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    task_indices: np.ndarray
    tasks: dict[int, str]
    anchors: np.ndarray
    action_mask: np.ndarray
    hand_available: np.ndarray
    videos: dict[str, tuple[str, float, bool]]


class EngramMultiEmbodimentDataset:
    """Read explicitly selected full-success episodes and preserve each route."""

    def __init__(
        self,
        manifest: str | Path,
        action_horizon: int = 64,
        *,
        split: str = "train",
        decode_images: bool = True,
        video_mode: str = "original",
        proxy_root: str | Path | None = None,
        prune_threshold_rad: float = 0.01,
        hand_threshold_percent: float = 0.0,
        video_tolerance_s: float = 0.0001,
        max_gap_s: float = 0.25,
        video_cache_size: int = 8,
        verify_proxy_hash: bool = True,
    ):
        if split not in ("train", "val", "all") or video_mode not in ("original", "proxy_640"):
            raise ValueError("Invalid dataset split or video mode")
        if action_horizon < 1 or min(video_tolerance_s, max_gap_s, video_cache_size) <= 0:
            raise ValueError("Invalid horizon, tolerance, gap, or video cache")
        if (
            not np.isfinite(prune_threshold_rad)
            or prune_threshold_rad < 0
            or not np.isfinite(hand_threshold_percent)
            or hand_threshold_percent < 0
        ):
            raise ValueError("Motion thresholds must be finite and nonnegative")
        self.manifest_path = Path(manifest)
        self.manifest = read_manifest(manifest)
        self.action_horizon = action_horizon
        self.split = split
        self.decode_images = decode_images
        self.video_mode = video_mode
        self.video_tolerance_s = video_tolerance_s
        self.video_cache_size = video_cache_size
        self._video_cache = OrderedDict()
        self.episodes = []
        self.invalid_anchors = self.short_horizon_anchors = self.static_anchors = 0
        self.source_counts = []
        self.missing_hand_anchors = dict.fromkeys(ROUTE_NAMES, 0)
        seen_annotations = set()
        seen_windows = set()
        video_identity = hashlib.sha256()
        proxy_checks = {}
        for source in self.manifest["sources"]:
            selected_ids = set(map(int, source["selected_episode_indices"]))
            if not selected_ids:
                continue
            root = Path(source["root"])
            index = route_id(source["robot_type"], source["end_effector_type"])
            route = EMBODIMENTS[index]
            info = json.loads((root / "meta/info.json").read_text())
            features = info["features"]
            if info.get("codebase_version") != "v3.0" or info.get("robot_type") != route.robot_type:
                raise ValueError(f"Unexpected source robot/schema at {root}")
            for key in (legacy.STATE_KEY, legacy.ACTION_KEY):
                if (
                    tuple(features[key].get("shape", ())) != (route.joint_dim,)
                    or features[key].get("names") != ROUTES[route.name]["joint_names"]
                ):
                    raise ValueError(f"Unexpected {key} joint layout at {root}")
            fps = float(info["fps"])
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Invalid nominal action rate")
            hand_keys = (
                legacy.LEFT_STATE,
                legacy.RIGHT_STATE,
                legacy.LEFT_ACTION,
                legacy.RIGHT_ACTION,
                legacy.LEFT_MODE,
                legacy.RIGHT_MODE,
            )
            present = set(hand_keys) & set(features)
            if present and present != set(hand_keys):
                raise ValueError(f"Partial end-effector annotation schema at {root}")
            has_hands = bool(present)
            if source.get("has_hand_labels", has_hands) != has_hands:
                raise ValueError("Manifest hand-label availability differs from the actual schema")
            if has_hands:
                for key in (legacy.LEFT_STATE, legacy.RIGHT_STATE, legacy.LEFT_ACTION, legacy.RIGHT_ACTION):
                    if features[key].get("shape") != [route.hand_dim // 2]:
                        raise ValueError(f"Unexpected hand width for {key} at {root}")
            tasks = {int(r["task_index"]): r["task"] for r in pq.read_table(root / "meta/tasks.parquet").to_pylist()}
            rows = [
                row
                for p in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
                for row in pq.read_table(p).to_pylist()
            ]
            metadata_by_id = {int(r["episode_index"]): r for r in rows}
            if len(metadata_by_id) != len(rows) or not selected_ids.issubset(metadata_by_id):
                raise ValueError("Duplicate or nonexistent selected episodes")
            expected = source.get("expected_episodes")
            if expected is not None and expected != len(selected_ids):
                raise ValueError("Selected episode count differs from pinned source")
            plan = json.loads((root.parent / "export_plan.json").read_text())
            packages = {}
            for entry in plan:
                annotation, package = entry["annotation_id"], entry["data_package_id"]
                if annotation in packages and packages[annotation] != package:
                    raise ValueError("Ambiguous annotation/package mapping")
                packages[annotation] = package
            heldout = set(self.manifest.get("heldout_packages", source.get("validation_packages", [])))
            start_count = len(self.episodes)
            for episode_id in sorted(selected_ids):
                row = metadata_by_id[episode_id]
                if row.get("completeness") != "FULL" or row.get("annotation_result") != "SUCCESS":
                    raise ValueError("Explicit selection contains an incomplete or unsuccessful episode")
                annotation = row["annotation_id"]
                if annotation not in packages:
                    raise ValueError("Selected annotation has no source recording package")
                package = packages[annotation]
                if split != "all" and (package in heldout) != (split == "val"):
                    continue
                if annotation in seen_annotations:
                    raise ValueError("Duplicate selected annotation across source exports")
                seen_annotations.add(annotation)
                data_path = root / info["data_path"].format(
                    chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
                )
                columns = list(legacy._COLUMNS if has_hands else legacy._COLUMNS[:6])
                table = pq.read_table(data_path, columns=columns, filters=[("episode_index", "=", episode_id)])
                n = len(table)
                if n != int(row["length"]) or not n:
                    raise ValueError("Episode metadata/data lengths disagree")
                if not np.array_equal(np.asarray(table["frame_index"]), np.arange(n)):
                    raise ValueError("Episode frame indices are not contiguous")
                timestamps = np.asarray(table["timestamp"], dtype=np.float64)
                if (
                    not np.isfinite(timestamps).all()
                    or np.any(np.diff(timestamps) <= 0)
                    or np.any(np.diff(timestamps) > max_gap_s)
                ):
                    raise ValueError("Invalid or missing physical timestamps")
                # Exporters express episode time relative to the same source reference.
                # Exact annotation duplicates are rejected here; overlap selection is
                # additionally pinned by the manifest builder's interval audit.
                window = (package, row.get("frame_start_time"), row.get("frame_end_time"))
                if None not in window and window in seen_windows:
                    raise ValueError("Duplicate package time window across exports")
                seen_windows.add(window)
                states = np.zeros((n, ACTION_DIM), np.float32)
                actions = np.zeros((n, ACTION_DIM), np.float32)
                states[:, : route.joint_dim] = legacy._vector(table, legacy.STATE_KEY, route.joint_dim)
                actions[:, : route.joint_dim] = legacy._vector(table, legacy.ACTION_KEY, route.joint_dim)
                mask = physical_mask(index, has_hands)
                hand_command_valid = np.zeros(n, dtype=bool)
                hand_feedback_valid = np.zeros(n, dtype=bool)
                if has_hands:
                    hand_command_valid[:] = True
                    width = route.hand_dim // 2
                    for side, state_key, action_key, mode_key in (
                        (0, legacy.LEFT_STATE, legacy.LEFT_ACTION, legacy.LEFT_MODE),
                        (1, legacy.RIGHT_STATE, legacy.RIGHT_ACTION, legacy.RIGHT_MODE),
                    ):
                        offset = HAND_START + side * width
                        states[:, offset : offset + width] = legacy._vector(table, state_key, width)
                        actions[:, offset : offset + width] = legacy._vector(table, action_key, width)
                        modes = legacy._vector(table, mode_key, 1)[:, 0]
                        if np.any(np.isfinite(modes) & (modes != 0)):
                            raise ValueError(f"End-effector action mode must be POSITION: {root}, episode {episode_id}")
                        hand_command_valid &= np.isfinite(modes)
                    lo = -100 if route.end_effector_type == "linker_l20" else 0
                    if np.any(
                        (actions[:, HAND_START : HAND_START + route.hand_dim] < lo)
                        | (actions[:, HAND_START : HAND_START + route.hand_dim] > 100)
                    ):
                        raise ValueError("End-effector targets outside their schema units")
                    hand_command_valid &= np.isfinite(actions[:, HAND_START : HAND_START + route.hand_dim]).all(axis=1)
                    hand_state = states[:, HAND_START : HAND_START + route.hand_dim]
                    hand_feedback_valid = np.all(
                        np.isfinite(hand_state) & (hand_state >= lo) & (hand_state <= 100), axis=1
                    )
                if not np.isfinite(actions[:, : route.joint_dim]).all():
                    raise ValueError("Nonfinite supervised joint targets")
                valid = np.isfinite(states[:, : route.joint_dim]).all(axis=1)
                self.invalid_anchors += int((~valid).sum())
                enough = timestamps + (action_horizon - 1) / fps <= timestamps[-1] + 1e-9
                self.short_horizon_anchors += int(np.sum(valid & ~enough))
                anchors = np.flatnonzero(valid & enough)
                hand_available = np.zeros(n, dtype=bool)
                if has_hands:
                    for begin in range(0, len(anchors), 512):
                        batch = anchors[begin : begin + 512]
                        targets = timestamps[batch, None] + np.arange(action_horizon)[None, :] / fps
                        right = np.minimum(np.maximum(np.searchsorted(timestamps, targets), batch[:, None]), n - 1)
                        left = np.maximum(right - 1, batch[:, None])
                        indices = np.where(
                            np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets), left, right
                        )
                        hand_available[batch] = hand_feedback_valid[batch] & hand_command_valid[indices].all(axis=1)
                if split != "val" and prune_threshold_rad > 0:
                    before = len(anchors)
                    anchors = legacy._moving_anchors(
                        timestamps,
                        states,
                        actions,
                        anchors,
                        action_horizon,
                        fps,
                        prune_threshold_rad,
                        hand_threshold_percent,
                        has_gripper=False,
                    )
                    if has_hands:
                        available_anchors = np.flatnonzero(valid & enough & hand_available)
                        hand_motion = legacy._moving_anchors(
                            timestamps,
                            states,
                            actions,
                            available_anchors,
                            action_horizon,
                            fps,
                            prune_threshold_rad,
                            hand_threshold_percent,
                            has_gripper=True,
                        )
                        anchors = np.union1d(anchors, hand_motion)
                    self.static_anchors += before - len(anchors)
                task_indices = np.asarray(table["task_index"], dtype=np.int64)
                if not set(task_indices.tolist()).issubset(tasks):
                    raise ValueError("Frame task indices have no task definitions")
                videos = {}
                for role, camera in CAMERAS[route.robot_type].items():
                    camera_key = f"observation.images.{camera}"
                    if camera_key not in features:
                        if role == "head":
                            raise ValueError("Missing required head camera")
                        continue
                    prefix = f"videos/{camera_key}"
                    original = root / info["video_path"].format(
                        video_key=camera_key,
                        chunk_index=int(row[f"{prefix}/chunk_index"]),
                        file_index=int(row[f"{prefix}/file_index"]),
                    )
                    if not original.is_file():
                        raise FileNotFoundError(original)
                    path = original
                    stereo = route.robot_type == "dex_w1" and role == "head"
                    if video_mode == "proxy_640":
                        from openpi.training import engram_multi_video

                        path = engram_multi_video.proxy_path(root, original, proxy_root)
                        cache_key = (str(path), episode_id, n)
                        if cache_key not in proxy_checks:
                            proxy_checks[cache_key] = engram_multi_video.validate_proxy(
                                root,
                                original,
                                path,
                                route.robot_type,
                                role,
                                episode_id,
                                n,
                                verify_hash=verify_proxy_hash,
                            )
                        record = proxy_checks[cache_key]
                        video_identity.update(f"{root}\0{episode_id}\0{role}\0{record['proxy_sha256']}\n".encode())
                        stereo = False
                    videos[role] = (str(path), float(row[f"{prefix}/from_timestamp"]), stereo)
                self.episodes.append(
                    Episode(
                        source.get("source_id", root.parent.name),
                        root,
                        episode_id,
                        annotation,
                        package,
                        index,
                        fps,
                        timestamps,
                        states,
                        actions,
                        task_indices,
                        tasks,
                        anchors,
                        mask,
                        hand_available,
                        videos,
                    )
                )
            source_episodes = self.episodes[start_count:]
            self.source_counts.append(
                {
                    "source_id": source.get("source_id", root.parent.name),
                    "root": str(root),
                    "embodiment_id": index,
                    "episodes": len(source_episodes),
                    "anchors": sum(len(e.anchors) for e in source_episodes),
                    "missing_hand_anchors": sum(int((~e.hand_available[e.anchors]).sum()) for e in source_episodes),
                }
            )
            self.missing_hand_anchors[route.name] += self.source_counts[-1]["missing_hand_anchors"]
            if len(self.source_counts) % 25 == 0:
                logging.info("Read %d sources, %d selected episodes", len(self.source_counts), len(self.episodes))
        self.offsets = np.cumsum([0, *[len(e.anchors) for e in self.episodes]], dtype=np.int64)
        self.video_identity = video_identity.hexdigest() if video_mode == "proxy_640" else None
        if not len(self):
            raise ValueError(f"No retained {split} anchors in mixed Engram dataset")
        logging.info(
            "Engram multi %s: %d episodes, %d anchors, %d invalid, %d short, %d static",
            split,
            len(self.episodes),
            len(self),
            self.invalid_anchors,
            self.short_horizon_anchors,
            self.static_anchors,
        )

    def __len__(self):
        return int(self.offsets[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_cache"] = OrderedDict()
        return state

    def close(self):
        for container in self._video_cache.values():
            container.close()
        self._video_cache.clear()

    def _read_image(self, episode: Episode, role: str, timestamp: float) -> np.ndarray:
        route = EMBODIMENTS[episode.embodiment_id]
        width, height = camera_size(route.robot_type, role)
        if not self.decode_images:
            return np.zeros((height, width, 3), np.uint8)
        path, start, stereo = episode.videos[role]
        container = self._video_cache.pop(path, None) or av.open(path)
        self._video_cache[path] = container
        while len(self._video_cache) > self.video_cache_size:
            self._video_cache.popitem(last=False)[1].close()
        stream = container.streams.video[0]
        if not stream.codec_context.is_open:
            stream.codec_context.thread_count = 1
        target = start + timestamp
        container.seek(max(0, int(target / float(stream.time_base))), stream=stream, backward=True, any_frame=False)
        previous = selected = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            if time >= target:
                selected = min(
                    (x for x in (previous, (time, frame)) if x is not None), key=lambda x: abs(x[0] - target)
                )
                break
            previous = (time, frame)
        selected = selected or previous
        if selected is None or abs(selected[0] - target) > self.video_tolerance_s:
            raise ValueError(f"No {role} image aligned to episode {episode.episode_index} timestamp {target}")
        return prepare_camera(selected[1].to_ndarray(format="rgb24"), route.robot_type, role, stereo=stereo)

    def __getitem__(self, index: SupportsIndex):
        value = index.__index__()
        if not 0 <= value < len(self):
            raise IndexError(value)
        part = int(np.searchsorted(self.offsets, value, side="right") - 1)
        episode = self.episodes[part]
        anchor = int(episode.anchors[value - self.offsets[part]])
        indices = legacy._relative_action_indices(episode.timestamps, anchor, self.action_horizon, episode.fps)
        route = EMBODIMENTS[episode.embodiment_id]
        images, masks = {}, {}
        for role, slot in SLOTS.items():
            masks[slot] = np.bool_(role in episode.videos)
            if role in episode.videos:
                images[slot] = self._read_image(episode, role, float(episode.timestamps[anchor]))
            else:
                width, height = camera_size(route.robot_type, role)
                images[slot] = np.zeros((height, width, 3), np.uint8)
        mask = episode.action_mask.copy()
        if not episode.hand_available[anchor]:
            mask[HAND_START : HAND_START + route.hand_dim] = False
        state, actions = episode.states[anchor].copy(), episode.actions[indices].copy()
        state[~mask], actions[:, ~mask] = 0, 0
        return {
            "state": state,
            "actions": actions,
            "action_mask": mask,
            "embodiment_id": np.int32(episode.embodiment_id),
            "robot_type": route.robot_type,
            "end_effector_type": route.end_effector_type,
            "prompt": episode.tasks[int(episode.task_indices[anchor])],
            "image": images,
            "image_mask": masks,
        }


def compute_training_stats(
    dataset: EngramMultiEmbodimentDataset, *, seed: int, max_frames: int | None = None, batch_size: int = 512
) -> tuple[dict[str, RouteStats], dict]:
    """Fit route-separated train-only statistics, ignoring all absent labels."""
    if dataset.split != "train" or batch_size < 1 or (max_frames is not None and max_frames < 2):
        raise ValueError("Statistics require a training split and positive batch/sample counts")
    rng = np.random.default_rng(seed)
    selected = None
    if max_frames is not None and max_frames < len(dataset):
        if max_frames < 2 * len(ROUTE_NAMES):
            raise ValueError("Diagnostic statistics need at least two anchors per embodiment")
        by_route = [[] for _ in ROUTE_NAMES]
        labeled_by_route = [[] for _ in ROUTE_NAMES]
        for episode, begin in zip(dataset.episodes, dataset.offsets[:-1], strict=True):
            values = np.arange(begin, begin + len(episode.anchors), dtype=np.int64)
            by_route[episode.embodiment_id].append(values)
            route = EMBODIMENTS[episode.embodiment_id]
            labeled_values = values[episode.hand_available[episode.anchors]]
            if len(labeled_values):
                labeled_by_route[episode.embodiment_id].append(labeled_values)
        mandatory = []
        for parts, labeled in zip(by_route, labeled_by_route, strict=True):
            values = np.concatenate(parts) if parts else np.empty(0, np.int64)
            if labeled:
                labeled_values = np.concatenate(labeled)
                if len(labeled_values) >= 2:
                    values = labeled_values
            if len(values) < 2:
                raise ValueError("Every embodiment requires at least two retained statistics anchors")
            mandatory.extend(rng.choice(values, 2, replace=False).tolist())
        remainder = np.setdiff1d(np.arange(len(dataset), dtype=np.int64), mandatory, assume_unique=False)
        extra = rng.choice(remainder, max_frames - len(mandatory), replace=False)
        selected = np.sort(np.concatenate((np.asarray(mandatory, np.int64), extra)))
    running = {
        name: {key: normalize.RunningStats() for key in ("joint_state", "joint_action", "hand_state", "hand_action")}
        for name in ROUTE_NAMES
    }
    counts = {name: {"anchors": 0, "labeled_hand_anchors": 0} for name in ROUTE_NAMES}
    for episode, begin in zip(dataset.episodes, dataset.offsets[:-1], strict=True):
        route = EMBODIMENTS[episode.embodiment_id]
        anchors = episode.anchors
        if selected is not None:
            chosen = (
                selected[np.searchsorted(selected, begin) : np.searchsorted(selected, begin + len(anchors))] - begin
            )
            anchors = anchors[chosen]
        for start in range(0, len(anchors), batch_size):
            batch = anchors[start : start + batch_size]
            targets = episode.timestamps[batch, None] + np.arange(dataset.action_horizon)[None, :] / episode.fps
            right = np.minimum(
                np.maximum(np.searchsorted(episode.timestamps, targets), batch[:, None]), len(episode.timestamps) - 1
            )
            left = np.maximum(right - 1, batch[:, None])
            indices = np.where(
                np.abs(episode.timestamps[left] - targets) <= np.abs(episode.timestamps[right] - targets), left, right
            )
            states = episode.states[batch].copy()
            actions = episode.actions[indices].copy()
            actions[..., : route.joint_dim] -= states[:, None, : route.joint_dim]
            noise = rng.normal(0, 0.001, (len(batch), route.joint_dim)).astype(np.float32)
            states[:, : route.joint_dim] += noise
            actions[..., : route.joint_dim] -= noise[:, None, :]
            acc = running[route.name]
            acc["joint_state"].update(states[:, : route.joint_dim])
            acc["joint_action"].update(actions[..., : route.joint_dim])
            counts[route.name]["anchors"] += len(batch)
            labeled = episode.hand_available[batch]
            if np.any(labeled):
                acc["hand_state"].update(states[labeled, HAND_START : HAND_START + route.hand_dim])
                acc["hand_action"].update(actions[labeled, :, HAND_START : HAND_START + route.hand_dim])
                counts[route.name]["labeled_hand_anchors"] += int(labeled.sum())
    result = {}
    for index, route in enumerate(EMBODIMENTS):
        count = counts[route.name]
        if count["anchors"] < 2:
            raise ValueError(f"No sufficient training anchors for {route.name}")
        has_hands = count["labeled_hand_anchors"] >= 2
        if count["labeled_hand_anchors"] == 1:
            raise ValueError("Insufficient labeled hand samples for stable normalization")
        output = {}
        for key, joint_key, hand_key in (
            ("state", "joint_state", "hand_state"),
            ("actions", "joint_action", "hand_action"),
        ):
            arrays = {
                "mean": np.zeros(ACTION_DIM),
                "std": np.ones(ACTION_DIM),
                "q01": -np.ones(ACTION_DIM),
                "q99": np.ones(ACTION_DIM),
            }
            parts = [(slice(0, route.joint_dim), running[route.name][joint_key].get_statistics(), 0.1)]
            if has_hands:
                parts.append(
                    (
                        slice(HAND_START, HAND_START + route.hand_dim),
                        running[route.name][hand_key].get_statistics(),
                        5.0,
                    )
                )
            for target, stat, floor in parts:
                for field, array in arrays.items():
                    array[target] = getattr(stat, field)
                midpoint = (arrays["q99"][target] + arrays["q01"][target]) / 2
                half = np.maximum(arrays["q99"][target] - arrays["q01"][target], floor) / 2
                arrays["q01"][target], arrays["q99"][target] = midpoint - half, midpoint + half
            output[key] = normalize.NormStats(**arrays)
        result[route.name] = RouteStats(output["state"], output["actions"], physical_mask(index, has_hands), count)
    used = sum(v["anchors"] for v in counts.values())
    expected = len(dataset) if selected is None else len(selected)
    if used != expected:
        raise ValueError("Statistics coverage omitted requested retained anchors")
    return result, {
        "used_anchors": used,
        "total_anchors": len(dataset),
        "routes": counts,
        "invalid_anchors": dataset.invalid_anchors,
        "short_horizon_anchors": dataset.short_horizon_anchors,
        "static_anchors": dataset.static_anchors,
        "missing_hand_anchors": dataset.missing_hand_anchors,
        "source_counts": dataset.source_counts,
    }
