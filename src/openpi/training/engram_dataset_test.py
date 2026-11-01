import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.models.model import ModelType
from openpi.policies import engram_policy
from openpi.policies.astribot_policy import AstribotInputs
from openpi.policies.engram_policy import ExactJointNoiseCompensation
from openpi.shared import normalize
from openpi.training import engram_dataset as engram


@pytest.fixture
def v3_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "trainable"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    video_dir = root / f"videos/{engram.CAMERA_KEY}/chunk-000"
    video_dir.mkdir(parents=True)
    (video_dir / "file-000.mp4").touch()
    info = {
        "codebase_version": "v3.0",
        "robot_type": "dex_w1",
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {key: {"names": list(engram.JOINT_NAMES)} for key in (engram.STATE_KEY, engram.ACTION_KEY)},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "sandwich"}]), root / "meta/tasks.parquet")
    records, metadata = [], []
    for episode, timestamps in enumerate(([0.0, 0.03, 0.07, 0.1, 0.14], [0.0, 0.035, 0.07, 0.105])):
        for frame, timestamp in enumerate(timestamps):
            records.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "timestamp": timestamp,
                    "task_index": 0,
                    engram.STATE_KEY: [float(frame)] * 20,
                    engram.ACTION_KEY: [float(episode * 100 + frame)] * 20,
                    engram.LEFT_STATE: [float("nan") if episode == 0 and frame == 1 else 20.0],
                    engram.RIGHT_STATE: [80.0],
                    engram.LEFT_ACTION: [20.0 + frame],
                    engram.RIGHT_ACTION: [80.0 - frame],
                    engram.LEFT_MODE: [0.0],
                    engram.RIGHT_MODE: [0.0],
                }
            )
        prefix = f"videos/{engram.CAMERA_KEY}"
        metadata.append(
            {
                "episode_index": episode,
                "annotation_id": f"annotation-{episode}",
                "completeness": "FULL",
                "annotation_result": "SUCCESS",
                "length": len(timestamps),
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"{prefix}/chunk_index": 0,
                f"{prefix}/file_index": 0,
                f"{prefix}/from_timestamp": 0.0,
            }
        )
    pq.write_table(pa.Table.from_pylist(records), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


def test_v3_windows_keep_invalid_anchor_actions_and_exclude_short_horizon(v3_fixture):
    dataset = engram.EngramDataset(v3_fixture, 3, decode_images=False)
    assert len(dataset.episodes) == 2
    assert dataset.invalid_anchors == 1
    assert dataset.short_horizon_anchors == 4
    assert len(dataset) == 4
    first = dataset[0]
    np.testing.assert_array_equal(first["actions"][:, 0], [0, 1, 2])
    assert first["actions"][1, 20] == 21  # Invalid feedback does not erase a valid target.
    assert first["prompt"] == "sandwich"
    assert first["observation/joint_position"].shape == (22,)
    # A second episode shares the data file but its horizon never crosses episodes.
    np.testing.assert_array_equal(dataset[2]["actions"][:, 0], [100, 101, 102])
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_reject_non_position_gripper_mode(v3_fixture):
    path = v3_fixture / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][engram.LEFT_MODE] = [2.0]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="POSITION"):
        engram.EngramDataset(v3_fixture, 3, decode_images=False)


def test_reject_time_gaps(v3_fixture):
    path = v3_fixture / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[3]["timestamp"] = 0.5
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="timestamps"):
        engram.EngramDataset(v3_fixture, 3, decode_images=False)


def test_delta_round_trip_and_noise_only_on_joints(v3_fixture):
    data = engram.EngramDataset(v3_fixture, 3, decode_images=False)[0]
    canonical = AstribotInputs(model_type=ModelType.PI05)(data)
    original_actions = canonical["actions"].copy()
    mask = transforms.make_bool_mask(20, -2)
    delta = transforms.DeltaActions(mask)(canonical)
    np.testing.assert_allclose(delta["actions"][:, 20:], original_actions[:, 20:])
    restored = transforms.AbsoluteActions(mask)({"state": delta["state"], "actions": delta["actions"].copy()})
    np.testing.assert_allclose(restored["actions"], original_actions)
    np.random.seed(4)
    before_state, before_action = delta["state"].copy(), delta["actions"].copy()
    noised = ExactJointNoiseCompensation(std=0.001)(delta)
    np.testing.assert_array_equal(noised["state"][20:], before_state[20:])
    np.testing.assert_array_equal(noised["actions"][:, 20:], before_action[:, 20:])
    assert not np.array_equal(noised["state"][:20], before_state[:20])
    np.testing.assert_allclose(
        noised["state"][None, :20] + noised["actions"][:, :20], original_actions[:, :20], atol=1e-6
    )
    assert bool(noised["image_mask"]["base_0_rgb"])
    assert not bool(noised["image_mask"]["left_wrist_0_rgb"])
    assert not bool(noised["image_mask"]["right_wrist_0_rgb"])


def test_quantile_floor_has_exact_openpi_inverse():
    stat = normalize.NormStats(mean=np.zeros(22), std=np.ones(22), q01=np.zeros(22), q99=np.full(22, 0.001))
    stats = engram.floor_quantile_ranges({"state": stat, "actions": stat})
    np.testing.assert_allclose(stats["state"].q99[:20] - stats["state"].q01[:20], 0.1)
    np.testing.assert_allclose(stats["state"].q99[20:] - stats["state"].q01[20:], 5.0)
    raw = {"state": np.linspace(-0.2, 0.3, 22), "actions": np.linspace(-0.3, 0.4, 22)[None, :]}
    normalized = transforms.Normalize(stats, use_quantiles=True)(raw)
    recovered = transforms.Unnormalize(stats, use_quantiles=True)(normalized)
    np.testing.assert_allclose(recovered["state"], raw["state"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(recovered["actions"], raw["actions"], rtol=1e-6, atol=1e-6)


def test_resize_preserves_entire_stereo_half():
    image = np.zeros((90, 160, 3), np.uint8)
    image[:, :16, 0] = 255
    image[:, -16:, 1] = 255
    out = transforms.ResizeImages(224, 224)({"image": {"base": image}})["image"]["base"]
    assert out.shape == (224, 224, 3)
    assert out[:, :20, 0].max() > 200
    assert out[:, -20:, 1].max() > 200


def test_pruning_considers_whole_horizon_and_keeps_gripper_targets():
    timestamps = np.arange(5, dtype=np.float64) / 30
    states = np.zeros((5, 22), dtype=np.float32)
    actions = np.zeros((5, 22), dtype=np.float32)
    anchors = np.array([0, 1, 2])
    assert len(engram._moving_anchors(timestamps, states, actions, anchors, 3, 30, 0.01, 1.0)) == 0  # noqa: SLF001
    # A meaningful gripper-only target retains the whole chunk.
    actions[2, 20] = 2.0
    np.testing.assert_array_equal(
        engram._moving_anchors(timestamps, states, actions, anchors, 3, 30, 0.01, 1.0),  # noqa: SLF001
        anchors,
    )
    actions[2, 20] = 0
    # An initially tiny target must not discard meaningful motion later in the chunk.
    actions[2, 7] = 0.011
    np.testing.assert_array_equal(
        engram._moving_anchors(timestamps, states, actions, anchors, 3, 30, 0.01, 1.0),  # noqa: SLF001
        anchors,
    )


def test_joint_only_pruning_ignores_unavailable_end_effector_placeholders():
    timestamps = np.arange(5, dtype=np.float64) / 30
    states = np.zeros((5, 22), dtype=np.float32)
    actions = np.zeros((5, 22), dtype=np.float32)
    anchors = np.array([0, 1, 2])
    # Synthetic values prove that absent end-effector placeholders are not
    # treated as labels or used to decide whether to keep a chunk.
    actions[2, 20] = 50.0
    assert not len(engram._moving_anchors(  # noqa: SLF001
        timestamps, states, actions, anchors, 3, 30, 0.01, 0.0, has_gripper=False
    ))
    np.testing.assert_array_equal(
        engram._moving_anchors(timestamps, states, actions, anchors, 3, 30, 0.01, 0.0),  # noqa: SLF001
        anchors,
    )
    # The threshold is inclusive and applied to all steps of the chunk.
    actions[2, 7] = np.float32(0.01)
    np.testing.assert_array_equal(
        engram._moving_anchors(  # noqa: SLF001
            timestamps, states, actions, anchors, 3, 30, 0.01, 0.0, has_gripper=False
        ),
        anchors,
    )


def test_package_split_prevents_recording_leakage(v3_fixture):
    data_path = v3_fixture / "data/chunk-000/file-000.parquet"
    records = pq.read_table(data_path).to_pylist()
    for row in list(records):
        if row["episode_index"] == 0:
            records.append({**row, "episode_index": 2})
    pq.write_table(pa.Table.from_pylist(records), data_path)
    metadata_path = v3_fixture / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pq.read_table(metadata_path).to_pylist()
    metadata.append({**metadata[0], "episode_index": 2, "annotation_id": "annotation-2"})
    pq.write_table(pa.Table.from_pylist(metadata), metadata_path)
    packages = sorted(engram.VAL_PACKAGE_IDS)
    plan = [
        {"annotation_id": "annotation-0", "data_package_id": packages[0]},
        {"annotation_id": "annotation-1", "data_package_id": packages[1]},
        {"annotation_id": "annotation-2", "data_package_id": "training-package"},
    ]
    (v3_fixture.parent / "export_plan.json").write_text(json.dumps(plan))
    train = engram.EngramDataset(v3_fixture, 3, split="train", decode_images=False)
    val = engram.EngramDataset(v3_fixture, 3, split="val", decode_images=False)
    assert [ep.episode_index for ep in train.episodes] == [2]
    assert [ep.episode_index for ep in val.episodes] == [0, 1]
    assert val.static_anchors == 0


def test_stats_provenance_detects_changed_threshold(v3_fixture, tmp_path):
    manifest = tmp_path / "manifest.json"
    stats_file = tmp_path / "norm_stats.json"
    stats_file.write_text("{}")
    identity = engram.stats_identity(v3_fixture, "all", 64, 0.01, 1.0)
    identity["norm_stats_sha256"] = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(identity))
    engram.check_stats_identity(manifest, v3_fixture, "all", 64, 0.01, 1.0)
    sampled = json.loads(manifest.read_text())
    sampled["max_frames"] = 8
    manifest.write_text(json.dumps(sampled))
    with pytest.raises(ValueError, match="Sampled"):
        engram.check_stats_identity(manifest, v3_fixture, "all", 64, 0.01, 1.0)
    engram.check_stats_identity(manifest, v3_fixture, "all", 64, 0.01, 1.0, allow_sampled_stats=True)
    sampled["max_frames"] = None
    manifest.write_text(json.dumps(sampled))
    stats_file.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="do not match the saved manifest"):
        engram.check_stats_identity(manifest, v3_fixture, "all", 64, 0.01, 1.0)
    stats_file.write_text("{}")
    with pytest.raises(ValueError, match="do not match"):
        engram.check_stats_identity(manifest, v3_fixture, "all", 64, 0.02, 1.0)
    shard = v3_fixture / "data/chunk-000/file-000.parquet"
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(ValueError, match="do not match"):
        engram.check_stats_identity(manifest, v3_fixture, "all", 64, 0.01, 1.0)


def test_video_frame_must_match_anchor_pts(v3_fixture, monkeypatch):
    from fractions import Fraction
    from types import SimpleNamespace

    class Frame:
        def __init__(self, pts):
            self.pts = pts

        def to_ndarray(self, format):
            return np.zeros((4, 8, 3), np.uint8)

    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 1000))])

        def seek(self, *args, **kwargs):
            pass

        def decode(self, stream):
            return iter([Frame(0), Frame(40), Frame(80), Frame(120)])

        def close(self):
            pass

    monkeypatch.setattr(engram.av, "open", lambda path: Container())
    dataset = engram.EngramDataset(v3_fixture, 3)
    assert dataset[0]["observation/head_rgbd"].shape == (4, 4, 3)
    # Anchor at 70ms is missing. It must not silently consume the 80ms frame.
    with pytest.raises(ValueError, match="no camera frame near"):
        _ = dataset[1]


def test_normalization_hash_survives_checkpoint_roundtrip(v3_fixture, tmp_path):
    stat = normalize.NormStats(
        mean=np.arange(22, dtype=np.float32) / 7,
        std=np.ones(22, dtype=np.float32),
        q01=np.arange(22, dtype=np.float32) / 10,
        q99=np.arange(22, dtype=np.float32) / 10 + 1,
    )
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    normalize.save(source, {"state": stat, "actions": stat})
    normalize.save(checkpoint, normalize.load(source))
    first = hashlib.sha256((source / "norm_stats.json").read_bytes()).hexdigest()
    second = hashlib.sha256((checkpoint / "norm_stats.json").read_bytes()).hexdigest()
    assert first == second
    identity = engram.stats_identity(v3_fixture, "all", 64, 0.01, 0.0)
    identity["norm_stats_sha256"] = second
    (checkpoint / "engram_stats_manifest.json").write_text(json.dumps(identity))
    engram.check_stats_identity(checkpoint / "engram_stats_manifest.json", v3_fixture, "all", 64, 0.01, 0.0)


def _mixed_manifest(v3_fixture, tmp_path):
    full = tmp_path / "full/trainable"
    partial = tmp_path / "partial/trainable"
    shutil.copytree(v3_fixture, full)
    shutil.copytree(v3_fixture, partial)
    spec = {"format": "engram-multisource-v1", "sources": []}
    for name, root, has_gripper in (("full", full, True), ("partial", partial, False)):
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        if has_gripper:
            for key in (engram.LEFT_STATE, engram.RIGHT_STATE, engram.LEFT_ACTION, engram.RIGHT_ACTION, engram.LEFT_MODE, engram.RIGHT_MODE):
                info["features"][key] = {}
        info_path.write_text(json.dumps(info))
        metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
        metadata = pq.read_table(metadata_path).to_pylist()
        for row in metadata:
            row["annotation_id"] = name + "-" + row["annotation_id"]
        pq.write_table(pa.Table.from_pylist(metadata), metadata_path)
        (root.parent / "export_plan.json").write_text(json.dumps([
            {"annotation_id": f"{name}-annotation-0", "data_package_id": f"{name}-train"},
            {"annotation_id": f"{name}-annotation-1", "data_package_id": f"{name}-val"},
        ]))
        if not has_gripper:
            path = root / "data/chunk-000/file-000.parquet"
            table = pq.read_table(path)
            table = table.drop([engram.LEFT_STATE, engram.RIGHT_STATE, engram.LEFT_ACTION, engram.RIGHT_ACTION, engram.LEFT_MODE, engram.RIGHT_MODE])
            pq.write_table(table, path)
        spec["sources"].append({"root": str(root), "has_gripper": has_gripper, "expected_episodes": 2,
                                "validation_packages": [f"{name}-val"]})
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(spec))
    return path


def test_mixed_sources_preserve_masks_and_only_fit_labeled_grippers(v3_fixture, tmp_path):
    manifest = _mixed_manifest(v3_fixture, tmp_path)
    dataset = engram.EngramMultiDataset(manifest, 2, split="train", decode_images=False)
    assert len(dataset.datasets) == 2
    labeled, partial = dataset.datasets
    assert labeled[0]["action_mask"].all()
    assert partial[0]["action_mask"][:20].all()
    assert not partial[0]["action_mask"][20:].any()
    mixed, counts = engram.compute_training_stats(dataset, seed=4, batch_size=2)
    full, full_counts = engram.compute_training_stats(labeled, seed=4, batch_size=2)
    assert counts["used_anchors"] == len(dataset)
    assert counts["labeled_gripper_anchors"] == full_counts["used_anchors"] == len(labeled)
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            np.testing.assert_allclose(getattr(mixed[key], field)[20:], getattr(full[key], field)[20:])
    raw = partial[0]
    adapted = engram_policy.EngramInputs(model_type=ModelType.PI05, model_action_dim=32)(raw)
    assert "End-effector state unavailable." in adapted["prompt"]
    adapted["state"][20:] = 999
    adapted = engram_policy.MaskUnavailableState()(adapted)
    np.testing.assert_array_equal(adapted["state"][20:], [0, 0])
    assert not adapted["action_mask"][20:22].any()


def test_joint_only_source_prunes_train_but_not_validation(v3_fixture, tmp_path):
    manifest = _mixed_manifest(v3_fixture, tmp_path)
    spec = json.loads(manifest.read_text())
    path = Path(spec["sources"][1]["root"]) / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row[engram.STATE_KEY] = [0.0] * 20
        row[engram.ACTION_KEY] = [0.0] * 20
        if row["episode_index"] == 0 and row["frame_index"] == 2:
            row[engram.ACTION_KEY][7] = 0.01
    pq.write_table(pa.Table.from_pylist(rows), path)

    dataset = engram.EngramMultiDataset(manifest, 2, split="train", decode_images=False)
    partial = dataset.datasets[1]
    np.testing.assert_array_equal(partial.episodes[0].anchors, [1, 2])
    assert partial.static_anchors == 2
    assert not partial[0]["action_mask"][20:].any()
    unfiltered = engram.EngramMultiDataset(manifest, 2, split="train", decode_images=False, prune_threshold_rad=0)
    assert len(unfiltered.datasets[1]) == 4
    validation = engram.EngramMultiDataset(manifest, 2, split="val", decode_images=False)
    assert len(validation.datasets[1]) == 3
    assert validation.datasets[1].static_anchors == 0


def test_joint_only_pruning_rejects_old_statistics_contract(v3_fixture, tmp_path):
    manifest = tmp_path / "engram_stats_manifest.json"
    identity = engram.stats_identity(v3_fixture, "all", 64, 0.01, 0.0)
    identity["contract"] = "engram_w1_delta20_mixed_end_effector_mask_stats_v5"
    manifest.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="statistics do not match"):
        engram.check_stats_identity(manifest, v3_fixture, "all", 64, 0.01, 0.0)


def test_multisource_rejects_duplicate_annotations(v3_fixture, tmp_path):
    manifest = _mixed_manifest(v3_fixture, tmp_path)
    spec = json.loads(manifest.read_text())
    full, partial = [Path(item["root"]) for item in spec["sources"]]
    metadata_path = partial / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(metadata_path).to_pylist()
    rows[0]["annotation_id"] = "full-annotation-0"
    pq.write_table(pa.Table.from_pylist(rows), metadata_path)
    plan_path = partial.parent / "export_plan.json"
    plan = json.loads(plan_path.read_text())
    plan[0]["annotation_id"] = "full-annotation-0"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Duplicate annotation"):
        engram.EngramMultiDataset(manifest, 2, split="train", decode_images=False)


def test_opt_in_single_camera_does_not_discard_supplied_wrist(v3_fixture):
    sample = engram.EngramDataset(v3_fixture, 2, decode_images=False)[0]
    transform = engram_policy.EngramInputs(model_type=ModelType.PI05, model_action_dim=32, single_camera=True)
    out = transform(sample)
    assert tuple(out["image"]) == ("base_0_rgb",)
    assert tuple(out["image_mask"]) == ("base_0_rgb",)
    with pytest.raises(ValueError, match="cannot discard"):
        transform({**sample, "observation/left_wrist_rgbd": np.zeros((1, 1, 3), np.uint8)})


def test_proxy_reader_fails_closed_and_does_not_crop_square_twice(v3_fixture, monkeypatch):
    from fractions import Fraction
    from types import SimpleNamespace

    from openpi.training import engram_video_proxy

    original = v3_fixture / f"videos/{engram.CAMERA_KEY}/chunk-000/file-000.mp4"
    proxy = engram_video_proxy.proxy_path(v3_fixture, original)
    with pytest.raises(ValueError, match="proxy sidecar"):
        engram.EngramDataset(v3_fixture, 2, video_mode="proxy_640")
    proxy.write_bytes(b"test proxy content")
    record = {
        "version": engram_video_proxy.PROXY_VERSION, "source_relative": str(original.relative_to(v3_fixture)),
        "source_bytes": 0, "source_sha256": hashlib.sha256(b"").hexdigest(), "proxy_bytes": proxy.stat().st_size,
        "proxy_sha256": hashlib.sha256(proxy.read_bytes()).hexdigest(),
        "timing": {"proxy": {"width": 640, "height": 360, "color_range": 2}, "max_gop": 4,
                   "max_pts_error_us": 0, "episodes": [{"episode_index": 0, "rows": 5}, {"episode_index": 1, "rows": 4}]},
    }
    engram_video_proxy.sidecar_path(proxy).write_text(json.dumps(record))
    old_record = {**record, "version": "engram_left640_h264_gop4_v1"}
    engram_video_proxy.sidecar_path(proxy).write_text(json.dumps(old_record))
    with pytest.raises(ValueError, match="sidecar does not match"):
        engram.EngramDataset(v3_fixture, 2, video_mode="proxy_640")
    engram_video_proxy.sidecar_path(proxy).write_text(json.dumps(record))

    class Frame:
        pts = 0
        def to_ndarray(self, format):
            image = np.zeros((360, 640, 3), np.uint8)
            image[:, 500, 0] = 123
            return image

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 1000))])
        def seek(self, *args, **kwargs):
            pass
        def decode(self, stream):
            return iter([Frame()])
        def close(self):
            pass

    monkeypatch.setattr(engram.av, "open", lambda path: Container())
    dataset = engram.EngramDataset(v3_fixture, 2, video_mode="proxy_640")
    image = dataset[0]["observation/head_rgbd"]
    assert image.shape == (360, 640, 3)
    assert image[0, 500, 0] == 123
    assert dataset.video_identity is not None
    proxy.write_bytes(b"bad proxy content!")  # Same length, changed hash.
    with pytest.raises(ValueError, match="checksum"):
        engram.EngramDataset(v3_fixture, 2, video_mode="proxy_640")
