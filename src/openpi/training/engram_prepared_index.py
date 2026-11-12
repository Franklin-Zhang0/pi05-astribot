"""Immutable node-local Engram annotation snapshots shared by spawned workers."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

FORMAT = "engram-prepared-index-v1"
_ARRAYS = {
    "states": (np.dtype(np.float32), 64),
    "actions": (np.dtype(np.float32), 64),
    "timestamps": (np.dtype(np.float64), None),
    "task_indices": (np.dtype(np.int64), None),
    "hand_available": (np.dtype(bool), None),
    "anchors": (np.dtype(np.int64), None),
}
_LOCAL_FILESYSTEMS = {"tmpfs", "overlay", "ext4", "xfs", "btrfs", "zfs", "ramfs"}


class InvalidPreparedIndexError(ValueError):
    """A prepared snapshot is incomplete or incompatible and must be rebuilt."""


def local_directory(path: str | Path, *, create: bool = True) -> Path:
    """Reject remote/FUSE cache mounts before any arrays can be memory mapped."""
    directory = Path(path).expanduser().resolve()
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), left.split()[4]))
        if directory.is_relative_to(mount):
            matches.append((len(mount.parts), right.split()[0]))
    filesystem = max(matches)[1] if matches else None
    if filesystem not in _LOCAL_FILESYSTEMS:
        raise ValueError(f"Prepared Engram index must use node-local storage, not {filesystem}: {directory}")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def code_identity() -> str:
    """Small source hashes invalidate snapshots when parsing or layouts change."""
    from openpi.policies import engram_multi_policy
    from openpi.shared import engram_multi
    from openpi.training import engram_dataset
    from openpi.training import engram_multi_dataset
    from openpi.training import engram_multi_video

    digest = hashlib.sha256()
    for file in (
        __file__,
        engram_dataset.__file__,
        engram_multi_dataset.__file__,
        engram_multi.__file__,
        engram_multi_video.__file__,
        engram_multi_policy.__file__,
    ):
        path = Path(file)
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


class PreparedIndexStore:
    def __init__(self, directory: str | Path, identity: dict):
        self.directory = local_directory(directory)
        self.identity = {"format": FORMAT, **identity}
        self.key = hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.entry = self.directory / self.key

    @contextmanager
    def lock(self):
        """Only one cold builder per key; readers of immutable generations never lock."""
        with (self.directory / f"{self.key}.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def current(self) -> dict | None:
        pointer = self.entry / "current.json"
        try:
            descriptor = json.loads(pointer.read_text())
            if (
                not isinstance(descriptor, dict)
                or descriptor.get("identity") != self.identity
                or descriptor.get("key") != self.key
            ):
                return None
            generation = descriptor["generation"]
            if not re.fullmatch(r"generation-[0-9a-f]{32}", generation):
                return None
            return {**descriptor, "path": str(self.entry / generation)}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def publish(self, dataset) -> dict:
        """Write six flat arrays once, then atomically publish a complete generation."""
        self.entry.mkdir(parents=True, exist_ok=True)
        generation = f"generation-{uuid.uuid4().hex}"
        staging = self.entry / f".building-{uuid.uuid4().hex}"
        staging.mkdir()
        frames = sum(len(episode.timestamps) for episode in dataset.episodes)
        anchors = len(dataset)
        mappings = {}
        try:
            for name, (dtype, width) in _ARRAYS.items():
                length = anchors if name == "anchors" else frames
                shape = (length,) if width is None else (length, width)
                mappings[name] = np.lib.format.open_memmap(staging / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            episodes = []
            frame_begin = anchor_begin = 0
            for episode in dataset.episodes:
                frame_end = frame_begin + len(episode.timestamps)
                anchor_end = anchor_begin + len(episode.anchors)
                for name, mapping in mappings.items():
                    selection = slice(anchor_begin, anchor_end) if name == "anchors" else slice(frame_begin, frame_end)
                    mapping[selection] = getattr(episode, name)
                episodes.append(
                    {
                        "source_id": episode.source_id,
                        "root": str(episode.root),
                        "episode_index": episode.episode_index,
                        "annotation_id": episode.annotation_id,
                        "package_id": episode.package_id,
                        "embodiment_id": episode.embodiment_id,
                        "fps": episode.fps,
                        "frame_begin": frame_begin,
                        "frame_end": frame_end,
                        "anchor_begin": anchor_begin,
                        "anchor_end": anchor_end,
                        "tasks": episode.tasks,
                        "action_mask": episode.action_mask.tolist(),
                        "videos": episode.videos,
                    }
                )
                frame_begin, anchor_begin = frame_end, anchor_end
            for mapping in mappings.values():
                mapping.flush()
            mappings.clear()
            fields = (
                "manifest",
                "action_horizon",
                "split",
                "video_mode",
                "invalid_anchors",
                "short_horizon_anchors",
                "static_anchors",
                "source_counts",
                "missing_hand_anchors",
                "video_identity",
            )
            metadata = {
                "format": FORMAT,
                "identity": self.identity,
                "frames": frames,
                "anchors": anchors,
                "state": {name: getattr(dataset, name) for name in fields},
                "episodes": episodes,
                "offsets": dataset.offsets.tolist(),
                "arrays": {name: {"bytes": (staging / f"{name}.npy").stat().st_size} for name in _ARRAYS},
            }
            payload = (json.dumps(metadata, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
            (staging / "metadata.json").write_bytes(payload)
            destination = self.entry / generation
            os.rename(staging, destination)
            descriptor = {
                "key": self.key,
                "identity": self.identity,
                "generation": generation,
                "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            }
            pending = self.entry / f".current-{uuid.uuid4().hex}.json"
            pending.write_text(json.dumps(descriptor) + "\n")
            os.replace(pending, self.entry / "current.json")
            return {**descriptor, "path": str(destination)}
        finally:
            mappings.clear()
            if staging.exists():
                shutil.rmtree(staging)


def open_index(descriptor: dict) -> tuple[dict, dict[str, np.memmap]]:
    """Open only local snapshot metadata and read-only array headers, never source files."""
    path = Path(descriptor["path"])
    try:
        # Workers check the actual backing mount, too; remote mmap is never implicit.
        path = local_directory(path, create=False)
        payload = (path / "metadata.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != descriptor["metadata_sha256"]:
            raise InvalidPreparedIndexError("Prepared index metadata checksum mismatch")
        metadata = json.loads(payload)
        if metadata["format"] != FORMAT or metadata["identity"] != descriptor["identity"]:
            raise InvalidPreparedIndexError("Prepared index identity mismatch")
        frames, anchors = metadata["frames"], metadata["anchors"]
        if type(frames) is not int or type(anchors) is not int or min(frames, anchors) <= 0:
            raise InvalidPreparedIndexError("Prepared index has invalid counts")
        if set(metadata["arrays"]) != set(_ARRAYS):
            raise InvalidPreparedIndexError("Prepared index array inventory mismatch")
        arrays = {}
        for name, (dtype, width) in _ARRAYS.items():
            file = path / f"{name}.npy"
            if file.is_symlink() or file.stat().st_size != metadata["arrays"][name]["bytes"]:
                raise InvalidPreparedIndexError(f"Prepared array size mismatch: {name}")
            array = np.load(file, mmap_mode="r", allow_pickle=False)
            length = anchors if name == "anchors" else frames
            shape = (length,) if width is None else (length, width)
            if array.dtype != dtype or array.shape != shape or array.flags.writeable:
                raise InvalidPreparedIndexError(f"Prepared array shape/dtype mismatch: {name}")
            arrays[name] = array
        return metadata, arrays
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise InvalidPreparedIndexError(f"Cannot open prepared Engram index {path}: {error}") from error
