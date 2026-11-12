import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.policies.engram_multi_policy import MultiInputs
from openpi.policies.engram_multi_policy import MultiOutputs
from openpi.policies.engram_multi_policy import prepare_camera
from openpi.shared.engram_multi import EMBODIMENTS
from openpi.shared.engram_multi import ROUTE_NAMES
from openpi.shared.engram_multi import ROUTES
from openpi.shared.engram_multi import load_route_stats
from openpi.shared.engram_multi import save_route_stats
from openpi.training import engram_dataset as old
from openpi.training import engram_multi_dataset as multi


def make_source(base: Path, route_id: int, *, hands: bool, wrists: bool = True, stationary: bool = False):
    route = EMBODIMENTS[route_id]
    root = base / route.name / "trainable"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    names = ROUTES[route.name]["joint_names"]
    features = {key: {"shape": [route.joint_dim], "names": names} for key in (old.STATE_KEY, old.ACTION_KEY)}
    for role, camera in multi.CAMERAS[route.robot_type].items():
        if not wrists and role != "head":
            continue
        key = f"observation.images.{camera}"
        features[key] = {"shape": [1080, 3840, 3] if camera == "kfc_compressed" else [480, 640, 3]}
        directory = root / f"videos/{key}/chunk-000"
        directory.mkdir(parents=True)
        (directory / "file-000.mp4").touch()
    if hands:
        for key in (old.LEFT_STATE, old.RIGHT_STATE, old.LEFT_ACTION, old.RIGHT_ACTION):
            features[key] = {"shape": [route.hand_dim // 2]}
        for key in (old.LEFT_MODE, old.RIGHT_MODE):
            features[key] = {"shape": [1]}
    info = {
        "codebase_version": "v3.0",
        "robot_type": route.robot_type,
        "fps": 30,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pylist([{"task_index": 0, "task": "pick_test / exact TEXT"}]), root / "meta/tasks.parquet"
    )
    data, metadata, plan = [], [], []
    for episode in range(2):
        annotation = f"{route.name}-annotation-{episode}"
        package = f"{route.name}-package-{episode}"
        row = {
            "episode_index": episode,
            "annotation_id": annotation,
            "completeness": "FULL",
            "annotation_result": "SUCCESS",
            "length": 8,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "frame_start_time": episode * 10.0,
            "frame_end_time": episode * 10.0 + 0.3,
        }
        for camera in multi.CAMERAS[route.robot_type].values():
            if f"observation.images.{camera}" in features:
                prefix = f"videos/observation.images.{camera}"
                row.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": 0, f"{prefix}/from_timestamp": 0.0})
        metadata.append(row)
        plan.append({"annotation_id": annotation, "data_package_id": package})
        for frame in range(8):
            state = [route_id + 0.1] * route.joint_dim
            action = state if stationary else [route_id + 0.2 + frame / 100] * route.joint_dim
            sample = {
                "episode_index": episode,
                "frame_index": frame,
                "task_index": 0,
                "timestamp": frame / 30,
                old.STATE_KEY: state,
                old.ACTION_KEY: action,
            }
            if hands:
                hand_state = -25.0 if route_id == 2 else 20.0
                sample.update(
                    {
                        old.LEFT_STATE: [hand_state] * (route.hand_dim // 2),
                        old.RIGHT_STATE: [hand_state] * (route.hand_dim // 2),
                        old.LEFT_ACTION: [hand_state + frame] * (route.hand_dim // 2),
                        old.RIGHT_ACTION: [hand_state + frame] * (route.hand_dim // 2),
                        old.LEFT_MODE: [0.0],
                        old.RIGHT_MODE: [0.0],
                    }
                )
            data.append(sample)
    pq.write_table(pa.Table.from_pylist(data), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    (root.parent / "export_plan.json").write_text(json.dumps(plan))
    return {
        "root": str(root),
        "source_id": route.name,
        "robot_type": route.robot_type,
        "end_effector_type": route.end_effector_type,
        "has_hand_labels": hands,
        "selected_episode_indices": [0, 1],
        "expected_episodes": 2,
        "validation_packages": [f"{route.name}-package-1"],
    }


@pytest.fixture
def mixed_manifest(tmp_path):
    sources = [make_source(tmp_path, i, hands=i != 3, wrists=i != 2) for i in range(4)]
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"format": multi.MANIFEST_FORMAT, "sources": sources}))
    return path


def test_mixed_dataset_routes_masks_and_package_split(mixed_manifest):
    train = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, split="train", decode_images=False)
    val = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, split="val", decode_images=False)
    assert len(train.episodes) == len(val.episodes) == 4
    assert len(train) == len(val) == 24
    assert not {e.package_id for e in train.episodes} & {e.package_id for e in val.episodes}
    for episode, begin in zip(train.episodes, train.offsets[:-1], strict=True):
        sample = train[int(begin)]
        spec = EMBODIMENTS[episode.embodiment_id]
        assert sample["state"].shape == (64,)
        assert sample["actions"].shape == (3, 64)
        assert not sample["action_mask"][52:].any()
        assert sample["action_mask"][: spec.joint_dim].all()
        if episode.embodiment_id == 3:
            assert not sample["action_mask"][14:].any()
        if episode.embodiment_id == 2:
            assert not sample["image_mask"]["left_wrist_0_rgb"]
        else:
            assert sample["image_mask"]["left_wrist_0_rgb"]
    train.close()
    val.close()


def test_route_stats_and_transform_roundtrip(mixed_manifest, tmp_path):
    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, split="train", decode_images=False)
    stats, counts = multi.compute_training_stats(dataset, seed=4, batch_size=5)
    assert counts["used_anchors"] == len(dataset)
    path = tmp_path / "stats.json"
    save_route_stats(path, stats)
    stats = load_route_stats(path)
    assert set(stats) == set(ROUTE_NAMES)
    assert not stats[ROUTE_NAMES[3]].supervised_dims[14:].any()
    for episode, begin in zip(dataset.episodes, dataset.offsets[:-1], strict=True):
        sample = dataset[int(begin)]
        original = sample["actions"].copy()
        transformed = MultiInputs(stats, structured_probability=1, train=False)(sample)
        prompt = json.loads(transformed["prompt"])
        assert prompt["instruction"] == sample["prompt"]
        assert prompt["robot_type"] == sample["robot_type"]
        assert prompt["end_effector_state_available"] == (episode.embodiment_id != 3)
        recovered = MultiOutputs(stats)(transformed)
        np.testing.assert_allclose(
            recovered["actions"][:, sample["action_mask"]], original[:, sample["action_mask"]], atol=1e-5
        )
        np.testing.assert_array_equal(recovered["actions"][:, ~sample["action_mask"]], 0)
    # Diagnostic selection is stratified, including rare routes and labeled hand examples.
    sampled, sampled_counts = multi.compute_training_stats(dataset, seed=123, max_frames=8)
    assert sampled_counts["used_anchors"] == 8
    assert all(r.counts["anchors"] >= 2 for r in sampled.values())


def test_mixed_prompts_do_not_change_route_or_instruction(mixed_manifest):
    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)
    stats, _ = multi.compute_training_stats(dataset, seed=0)
    raw = dataset[0]
    np.random.seed(12)
    transform = MultiInputs(stats, train=True, noise_std=0)
    rendered = [transform(raw)["prompt"] for _ in range(50)]
    assert any(p.startswith("{") for p in rendered)
    assert any(p == raw["prompt"] for p in rendered)
    raw["prompt_format"] = "plain"
    assert transform(raw)["prompt"] == raw["prompt"]
    absent = dataset[int(dataset.offsets[3])]
    absent["prompt_format"] = "plain"
    assert transform(absent)["prompt"] == "End-effector state unavailable. " + absent["prompt"]
    assert transform(absent)["embodiment_id"] == 3


def test_noise_compensation_preserves_absolute_targets(mixed_manifest):
    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)
    stats, _ = multi.compute_training_stats(dataset, seed=0)
    raw = dataset[0]
    result = MultiInputs(stats, train=True)(raw)
    recovered = MultiOutputs(stats)(result)
    np.testing.assert_allclose(recovered["actions"][:, :22], raw["actions"][:, :22], atol=1e-5)


def test_preserves_labeled_hand_only_chunks(tmp_path):
    source = make_source(tmp_path, 1, hands=True, stationary=True)
    p = tmp_path / "handonly.json"
    p.write_text(json.dumps({"format": multi.MANIFEST_FORMAT, "sources": [source]}))
    dataset = multi.EngramMultiEmbodimentDataset(p, 3, decode_images=False)
    assert len(dataset) == 6  # All horizons include changing hand targets.
    assert not dataset[0]["action_mask"][32:].any()


def test_rejects_partial_hand_schema_and_incomplete_selection(mixed_manifest):
    spec = json.loads(mixed_manifest.read_text())
    source = Path(spec["sources"][0]["root"])
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    del info["features"][old.RIGHT_STATE]
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Partial end-effector"):
        multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)


def test_camera_preprocessing_preserves_role_geometry():
    stereo = np.zeros((90, 320, 3), np.uint8)
    stereo[:, :160, 0] = 255
    stereo[:, 160:, 2] = 255
    left = prepare_camera(stereo, "dex_w1", "head", stereo=True)
    assert left.shape == (360, 640, 3)
    assert left[..., 0].min() == 255
    assert left[..., 2].max() == 0
    galbot = prepare_camera(np.zeros((96, 128, 3), np.uint8), "galbot", "head")
    assert galbot.shape == (360, 480, 3)
    wrist = prepare_camera(np.zeros((48, 64, 3), np.uint8), "dex_w1", "left_wrist")
    assert wrist.shape == (480, 640, 3)
    with pytest.raises(ValueError, match="aspect"):
        prepare_camera(np.zeros((100, 100, 3), np.uint8), "galbot", "head")


def test_stats_reject_undersized_ranges_and_fake_galbot_hand_labels(mixed_manifest, tmp_path):
    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)
    stats, _ = multi.compute_training_stats(dataset, seed=0)
    p = tmp_path / "stats.json"
    save_route_stats(p, stats)
    raw = json.loads(p.read_text())
    raw["routes"][ROUTE_NAMES[3]]["supervised_dims"][20] = True
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unlabeled Galbot"):
        load_route_stats(p)


def test_missing_hand_rows_keep_joint_training_and_mask_full_chunk(mixed_manifest):
    sources = json.loads(mixed_manifest.read_text())["sources"]
    path = Path(sources[1]["root"]) / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[2][old.LEFT_MODE] = [None]
    rows[2][old.LEFT_ACTION] = [None] * 6
    rows[3][old.RIGHT_STATE] = [None] * 6
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)
    begin = int(dataset.offsets[1])
    for index in (0, 1, 2, 3):
        sample = dataset[begin + index]
        assert sample["action_mask"][:20].all()
        assert not sample["action_mask"][20:].any()
        assert np.isfinite(sample["state"]).all()
        assert np.isfinite(sample["actions"]).all()
    assert dataset[begin + 4]["action_mask"][20:32].all()
    stats, counts = multi.compute_training_stats(dataset, seed=0)
    assert counts["routes"][ROUTE_NAMES[1]]["anchors"] == 6
    assert counts["routes"][ROUTE_NAMES[1]]["labeled_hand_anchors"] == 2
    assert dataset.missing_hand_anchors[ROUTE_NAMES[1]] == 4
    transformed = MultiInputs(stats, train=False)(dataset[begin])
    assert not json.loads(transformed["prompt"])["end_effector_state_available"]


def test_finite_nonposition_mode_still_rejected(mixed_manifest):
    sources = json.loads(mixed_manifest.read_text())["sources"]
    path = Path(sources[1]["root"]) / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[2][old.LEFT_MODE] = [1.0]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="must be POSITION"):
        multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)


def test_saved_statistics_require_full_route_and_anchor_coverage(mixed_manifest, tmp_path):
    import hashlib

    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)
    stats, counts = multi.compute_training_stats(dataset, seed=0)
    path = tmp_path / "engram_multi_stats.json"
    save_route_stats(path, stats)
    identity = multi.stats_identity(mixed_manifest, "train", 3, 0.01, 0.0)
    identity.update(
        counts,
        sampled_anchors=counts["used_anchors"],
        max_frames=None,
        stats_filename=path.name,
        stats_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    provenance = tmp_path / "engram_multi_manifest.json"

    def check(value, **kwargs):
        provenance.write_text(json.dumps(value))
        multi.check_stats_identity(provenance, mixed_manifest, "train", 3, 0.01, 0.0, **kwargs)

    check(identity)
    for key in ("used_anchors", "sampled_anchors", "total_anchors"):
        bad = json.loads(json.dumps(identity))
        bad[key] -= 1
        with pytest.raises(ValueError, match="coverage|counts"):
            check(bad)
    bad = json.loads(json.dumps(identity))
    bad["routes"][ROUTE_NAMES[1]]["labeled_hand_anchors"] = 0
    with pytest.raises(ValueError, match="coverage|counts"):
        check(bad)
    sampled = {**identity, "max_frames": 8}
    with pytest.raises(ValueError, match="diagnostic"):
        check(sampled)
    check(sampled, allow_sampled_stats=True)


def test_stats_startup_skips_source_reads_but_retains_manifest_and_opt_in_audit(mixed_manifest, tmp_path, monkeypatch):
    import hashlib

    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False)
    stats, counts = multi.compute_training_stats(dataset, seed=0)
    stats_file = tmp_path / "engram_multi_stats.json"
    save_route_stats(stats_file, stats)
    identity = multi.stats_identity(mixed_manifest, "train", 3, 0.01, 0.0)
    assert len(identity["sources"]) == 4  # Generation still performs the complete audit once.
    identity.update(
        counts,
        sampled_anchors=counts["used_anchors"],
        max_frames=None,
        stats_filename=stats_file.name,
        stats_sha256=hashlib.sha256(stats_file.read_bytes()).hexdigest(),
    )
    provenance = tmp_path / "engram_multi_manifest.json"
    provenance.write_text(json.dumps(identity))
    source_roots = [Path(source["root"]) for source in identity["source_spec"]["sources"]]
    original_read = Path.read_bytes
    reads = []

    def guarded_read(path):
        if any(path.is_relative_to(root) for root in source_roots):
            raise AssertionError("Startup unexpectedly reread source metadata")
        reads.append(path)
        return original_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", guarded_read)
        multi.check_stats_identity(provenance, mixed_manifest, "train", 3, 0.01, 0.0)
        assert mixed_manifest in reads
        assert stats_file in reads
        with pytest.raises(AssertionError, match="reread source"):
            multi.check_stats_identity(provenance, mixed_manifest, "train", 3, 0.01, 0.0, audit_source_metadata=True)
    # Explicit audits still detect changed source metadata; ordinary startup only
    # checks the exact pinned source manifest and transform, then runtime schemas.
    info_file = source_roots[0] / "meta/info.json"
    info = json.loads(info_file.read_text())
    info["audit_test_annotation"] = "changed after statistics generation"
    info_file.write_text(json.dumps(info))
    multi.check_stats_identity(provenance, mixed_manifest, "train", 3, 0.01, 0.0)
    with pytest.raises(ValueError, match="source/split/transform"):
        multi.check_stats_identity(provenance, mixed_manifest, "train", 3, 0.01, 0.0, audit_source_metadata=True)
    with pytest.raises(ValueError, match="source/split/transform"):
        multi.check_stats_identity(provenance, mixed_manifest, "train", 4, 0.01, 0.0)
    mixed_manifest.write_text(mixed_manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="source/split/transform"):
        multi.check_stats_identity(provenance, mixed_manifest, "train", 3, 0.01, 0.0)


def test_proxy_startup_skips_payload_reads_unless_full_hash_requested(mixed_manifest, monkeypatch):
    from openpi.training import engram_multi_video as video

    spec = json.loads(mixed_manifest.read_text())
    for source in spec["sources"]:
        root = Path(source["root"])
        for role, camera in multi.CAMERAS[source["robot_type"]].items():
            original = root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
            if not original.exists():
                continue
            proxy = video.proxy_path(root, original)
            proxy.write_bytes(b"unchecked video payload")
            width, height = multi.camera_size(source["robot_type"], role)
            record = {
                "version": video.PROXY_VERSION,
                "robot_type": source["robot_type"],
                "camera_role": role,
                "source_relative": str(original.relative_to(root)),
                "source_bytes": original.stat().st_size,
                "proxy_bytes": proxy.stat().st_size,
                "proxy_sha256": "f" * 64,
                "timing": {
                    "proxy": {"width": width, "height": height, "color_range": 2},
                    "max_gop": 4,
                    "max_pts_error_us": 0,
                    "episodes": [{"episode_index": i, "rows": 8} for i in range(2)],
                },
            }
            video.sidecar_path(proxy).write_text(json.dumps(record))
    original_open = Path.open
    audited = False
    payload_reads = []

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".mp4":
            if not audited:
                raise AssertionError("Default startup read a video payload")
            payload_reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    dataset = multi.EngramMultiEmbodimentDataset(mixed_manifest, 3, decode_images=False, video_mode="proxy_640")
    assert len(dataset) == 24
    assert dataset.video_identity is not None
    dataset.close()
    assert payload_reads == []
    audited = True
    with pytest.raises(ValueError, match="payload checksum"):
        multi.EngramMultiEmbodimentDataset(
            mixed_manifest, 3, decode_images=False, video_mode="proxy_640", verify_proxy_hash=True
        )
    assert len(payload_reads) == 1
