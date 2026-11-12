"""Prepared snapshots preserve annotation semantics without worker payload copies."""

# ruff: noqa: SLF001
from __future__ import annotations

from collections import OrderedDict
import json
import multiprocessing
from pathlib import Path
import pickle
import traceback

import numpy as np
import pyarrow.parquet as pq
import pytest

from openpi.training import engram_dataset as legacy
from openpi.training import engram_multi_dataset as multi
from openpi.training import engram_multi_video as video
from openpi.training import engram_prepared_index as prepared
from openpi.training.engram_multi_dataset_test import make_source


@pytest.fixture
def source_manifest(tmp_path):
    sources = [make_source(tmp_path, i, hands=i != 3, wrists=i != 2) for i in range(4)]
    file = Path(sources[1]["root"]) / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(file).to_pylist()
    rows[2][legacy.LEFT_MODE] = [None]
    rows[2][legacy.LEFT_ACTION] = [None] * 6
    rows[3][legacy.RIGHT_STATE] = [None] * 6
    pq.write_table(__import__("pyarrow").Table.from_pylist(rows), file)
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"format": multi.MANIFEST_FORMAT, "sources": sources}))
    return manifest


def read(manifest, cache=None, **kwargs):
    return multi.EngramMultiEmbodimentDataset(manifest, 3, decode_images=False, prepared_index_dir=cache, **kwargs)


def assert_identical(left, right):
    assert len(left) == len(right)
    assert left.video_identity == right.video_identity
    assert left.source_counts == right.source_counts
    assert left.missing_hand_anchors == right.missing_hand_anchors
    assert left.invalid_anchors == right.invalid_anchors
    assert left.short_horizon_anchors == right.short_horizon_anchors
    assert left.static_anchors == right.static_anchors
    np.testing.assert_array_equal(left.offsets, right.offsets)
    for a, b in zip(left.episodes, right.episodes, strict=True):
        for key in (
            "source_id",
            "root",
            "episode_index",
            "annotation_id",
            "package_id",
            "embodiment_id",
            "fps",
            "tasks",
            "videos",
        ):
            assert getattr(a, key) == getattr(b, key)
        for key in ("states", "actions", "timestamps", "task_indices", "anchors", "hand_available", "action_mask"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        assert all(type(key) is int for key in b.tasks)
    for value in np.unique(np.concatenate((left.offsets[:-1], left.offsets[1:] - 1))):
        a, b = left[int(value)], right[int(value)]
        for key in ("state", "actions", "action_mask", "embodiment_id"):
            np.testing.assert_array_equal(a[key], b[key])
        assert a["prompt"] == b["prompt"]
        assert a["image_mask"] == b["image_mask"]
        for key in a["image"]:
            np.testing.assert_array_equal(a["image"][key], b["image"][key])


def test_cold_and_warm_cache_preserve_full_timelines_masks_order_and_splits(source_manifest, tmp_path, monkeypatch):
    cache = tmp_path / "prepared"
    original = read(source_manifest)
    cold = read(source_manifest, cache)
    assert_identical(original, cold)
    assert sum(len(episode.timestamps) for episode in cold.episodes) > len(cold)
    assert not cold[6]["action_mask"][20:].any()
    assert all(isinstance(episode.states, np.memmap) for episode in cold.episodes)

    def forbidden(*args, **kwargs):
        raise AssertionError("Warm startup touched source metadata or data")

    with monkeypatch.context() as patch:
        patch.setattr(pq, "read_table", forbidden)
        patch.setattr(multi, "read_manifest", forbidden)
        patch.setattr(video, "validate_proxy", forbidden)
        patch.setattr(multi.EngramMultiEmbodimentDataset, "_build_episodes", forbidden)
        warm = read(source_manifest, cache)
        assert_identical(original, warm)
    validation = read(source_manifest, cache, split="val")
    assert {episode.package_id for episode in cold.episodes}.isdisjoint(
        episode.package_id for episode in validation.episodes
    )
    assert len(validation.episodes) == 4
    assert validation._prepared_descriptor["key"] != cold._prepared_descriptor["key"]


def test_descriptor_pickle_contains_no_bulk_arrays_and_is_read_only(source_manifest, tmp_path):
    dataset = read(source_manifest, tmp_path / "prepared")
    state = dataset.__getstate__()
    assert set(state) == {"_prepared_descriptor", "_prepared_runtime"}
    payload = pickle.dumps(dataset, protocol=5)
    assert len(payload) < 10000
    reconstructed = pickle.loads(payload)
    assert_identical(dataset, reconstructed)
    assert isinstance(reconstructed.episodes[0].states, np.memmap)
    assert reconstructed.episodes[0].states.filename is not None
    with pytest.raises(ValueError, match="read-only"):
        reconstructed.episodes[0].states[0, 0] = 123
    assert reconstructed._video_cache == OrderedDict()


def _spawn_load(dataset, queue):
    try:
        pq.read_table = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Worker reparsed a source parquet")
        )
        state = dataset.episodes[0].states
        queue.put(
            {
                "length": len(dataset),
                "readonly": not state.flags.writeable,
                "filename": str(state.filename),
                "mask": dataset[6]["action_mask"].tolist(),
                "payload_bytes": len(pickle.dumps(dataset, protocol=5)),
            }
        )
    except BaseException:
        queue.put({"error": traceback.format_exc()})


def test_real_spawn_reopens_shared_read_only_arrays(source_manifest, tmp_path):
    dataset = read(source_manifest, tmp_path / "prepared")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [context.Process(target=_spawn_load, args=(dataset, queue)) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        results = [queue.get(timeout=120) for _ in workers]
        for result in results:
            assert "error" not in result, result
            assert result["length"] == len(dataset)
            assert result["readonly"]
            assert result["payload_bytes"] < 10000
            assert not any(result["mask"][20:])
        assert results[0]["filename"] == results[1]["filename"]
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        queue.close()


def test_invalidated_configuration_and_explicit_audits_rebuild(source_manifest, tmp_path, monkeypatch):
    cache = tmp_path / "prepared"
    initial = read(source_manifest, cache)
    old_descriptor = initial._prepared_descriptor
    for kwargs in (
        {"prune_threshold_rad": 0.02},
        {"hand_threshold_percent": 1.0},
        {"max_gap_s": 0.3},
        {"video_tolerance_s": 0.0002},
        {"proxy_root": str(tmp_path / "otherproxy")},
    ):
        assert read(source_manifest, cache, **kwargs)._prepared_descriptor["key"] != old_descriptor["key"]
    source_manifest.write_text(source_manifest.read_text() + "\n")
    changed = read(source_manifest, cache)
    assert changed._prepared_descriptor["key"] != old_descriptor["key"]
    for kwargs in ({"verify_proxy_hash": True}, {"audit_source_metadata": True}, {"rebuild_prepared_index": True}):
        rebuilt = read(source_manifest, cache, **kwargs)
        assert rebuilt._prepared_descriptor["key"] == changed._prepared_descriptor["key"]
        assert rebuilt._prepared_descriptor["generation"] != changed._prepared_descriptor["generation"]
        assert_identical(changed, rebuilt)
    monkeypatch.setattr(prepared, "code_identity", lambda: "changed-parser-code")
    assert read(source_manifest, cache)._prepared_descriptor["key"] != changed._prepared_descriptor["key"]


def test_incomplete_or_truncated_generation_is_rebuilt_without_replacing_active_maps(source_manifest, tmp_path):
    cache = tmp_path / "prepared"
    initial = read(source_manifest, cache)
    generation = Path(initial._prepared_descriptor["path"])
    entry = generation.parent
    (entry / ".building-abandoned").mkdir()
    (entry / ".building-abandoned" / "metadata.json").write_text("incomplete")
    assert read(source_manifest, cache)._prepared_descriptor == initial._prepared_descriptor
    # Publish a new immutable generation; old worker mappings remain usable.
    (entry / "current.json").write_text("null")
    rebuilt = read(source_manifest, cache)
    assert generation.is_dir()
    assert_identical(initial, rebuilt)
    damaged = Path(rebuilt._prepared_descriptor["path"]) / "actions.npy"
    with damaged.open("r+b") as stream:
        stream.truncate(32)
    recovered = read(source_manifest, cache)
    assert recovered._prepared_descriptor["generation"] != rebuilt._prepared_descriptor["generation"]
    assert_identical(initial, recovered)


def _concurrent_build(manifest, directory, marker, queue):
    original = multi.EngramMultiEmbodimentDataset._build_episodes

    def build(self, **kwargs):
        with open(marker, "a") as stream:
            stream.write("build\n")
        original(self, **kwargs)

    multi.EngramMultiEmbodimentDataset._build_episodes = build
    try:
        dataset = read(manifest, directory)
        queue.put({"generation": dataset._prepared_descriptor["generation"], "length": len(dataset)})
    except BaseException:
        queue.put({"error": traceback.format_exc()})


def test_concurrent_cold_builders_publish_one_complete_generation(source_manifest, tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    marker = tmp_path / "builds.txt"
    workers = [
        context.Process(target=_concurrent_build, args=(source_manifest, tmp_path / "prepared", marker, queue))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    try:
        results = [queue.get(timeout=120) for _ in workers]
        assert all("error" not in result for result in results), results
        assert results[0] == results[1]
        assert marker.read_text().splitlines() == ["build"]
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        queue.close()


def test_remote_cache_mount_is_rejected_without_creating_files(tmp_path, monkeypatch):
    original_read = Path.read_text

    def read_mountinfo(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return "1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / /hand_keypoint_data rw - fuse.geesefs bucket rw\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_mountinfo)
    with pytest.raises(ValueError, match="node-local storage"):
        prepared.local_directory("/hand_keypoint_data/bad-prepared-index")


def test_proxy_snapshot_keeps_video_identity_without_reopening_sidecars(source_manifest, tmp_path, monkeypatch):
    spec = json.loads(source_manifest.read_text())
    for source in spec["sources"]:
        root = Path(source["root"])
        for role, camera in multi.CAMERAS[source["robot_type"]].items():
            original = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
            if not original.exists():
                continue
            proxy = video.proxy_path(root, original)
            proxy.write_bytes(b"prepared camera payload")
            width, height = multi.camera_size(source["robot_type"], role)
            record = {
                "version": video.PROXY_VERSION,
                "robot_type": source["robot_type"],
                "camera_role": role,
                "source_relative": str(original.relative_to(root)),
                "source_bytes": original.stat().st_size,
                "proxy_bytes": proxy.stat().st_size,
                "proxy_sha256": "a" * 64,
                "timing": {
                    "proxy": {"width": width, "height": height, "color_range": 2},
                    "max_gop": 4,
                    "max_pts_error_us": 0,
                    "episodes": [{"episode_index": i, "rows": 8} for i in range(2)],
                },
            }
            video.sidecar_path(proxy).write_text(json.dumps(record))
    cache = tmp_path / "prepared"
    cold = read(source_manifest, cache, video_mode="proxy_640")
    assert cold.video_identity is not None
    original_open = Path.open
    roots = [Path(source["root"]) for source in spec["sources"]]

    def guard(path, *args, **kwargs):
        if any(path.is_relative_to(root) for root in roots):
            raise AssertionError(f"Warm snapshot reopened source/camera metadata: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    warm = read(source_manifest, cache, video_mode="proxy_640")
    assert_identical(cold, warm)


def test_resolved_relative_source_roots_distinguish_launch_directories(source_manifest, tmp_path, monkeypatch):
    spec = json.loads(source_manifest.read_text())
    for source in spec["sources"]:
        source["root"] = str(Path(source["root"]).relative_to(tmp_path))
    source_manifest.write_text(json.dumps(spec))
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "prepared"
    cold = read(source_manifest, cache)
    assert all(Path(root).is_absolute() for root in cold._prepared_descriptor["identity"]["source_roots"])
    other = tmp_path / "different_working_directory"
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(FileNotFoundError):
        read(source_manifest, cache)
