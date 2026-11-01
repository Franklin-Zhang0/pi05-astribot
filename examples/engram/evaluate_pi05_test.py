import dataclasses
import hashlib
import json

import numpy as np
import pytest

from examples.engram import evaluate_pi05 as evaluator
from openpi import transforms
from openpi.policies import engram_policy
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config as train_config

evaluate_samples = evaluator.evaluate_samples


class _Dataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        target = np.zeros((4, 22), dtype=np.float32)
        target[:, :20] = 0.5
        target[:, 20:] = [20, 40]
        return {"actions": target, "observation/joint_position": np.zeros(22), "prompt": "sandwich"}


class _ZeroPolicy:
    def infer(self, observation):
        assert "actions" not in observation
        return {"actions": np.zeros((4, 22), dtype=np.float32)}


def test_validation_reports_action_errors_in_physical_units():
    result = evaluate_samples(_Dataset(), _ZeroPolicy(), max_samples=2, seed=42)
    assert result["num_anchors"] == 2
    assert result["action_steps"] == 8
    assert result["joint_rmse_rad"] == 0.5
    assert result["left_gripper_rmse_percent"] == 20
    assert result["right_gripper_rmse_percent"] == 40


def test_validation_excludes_missing_gripper_labels_from_metrics():
    class MixedDataset(_Dataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["action_mask"] = np.array([True] * 20 + [index == 0] * 2)
            if index != 0:
                sample["actions"][:, 20:] = 1000
            return sample

    result = evaluate_samples(MixedDataset(), _ZeroPolicy(), max_samples=3, seed=42)
    assert result["action_steps"] == 12
    assert result["gripper_action_steps"] == 4
    assert result["left_gripper_rmse_percent"] == 20
    assert result["right_gripper_rmse_percent"] == 40


@pytest.mark.parametrize(("use_proxy", "proxy_root"), [(False, None), (True, None), (True, "/validated/proxies")])
def test_main_propagates_input_options_and_reports_provenance(tmp_path, monkeypatch, capsys, use_proxy, proxy_root):
    monkeypatch.setattr(train_config._tokenizer, "PaligemmaTokenizer", lambda *_: object())  # noqa: SLF001
    monkeypatch.setattr(train_config.DataConfigFactory, "_load_norm_stats", lambda *_: None)
    checkpoint = tmp_path / "checkpoint"
    asset = checkpoint / "assets/engram/dex_w1_sandwich_train"
    asset.mkdir(parents=True)
    manifest = asset / "engram_stats_manifest.json"
    manifest.write_text('{"contract": "test"}')
    stats = asset / "norm_stats.json"
    stats.write_text('{"norm_stats": "test"}')
    data_root = tmp_path / ("sources.json" if use_proxy else "source")
    if use_proxy:
        data_root.write_text("{}")
    else:
        data_root.mkdir()
    observed = {}

    class FakeDataset(_Dataset):
        def __init__(self, root, horizon, **kwargs):
            observed["dataset"] = root, horizon, kwargs
            self.video_identity = "a" * 64 if kwargs["video_mode"] == "proxy_640" else None
            self.episodes = [object(), object()]

        def close(self):
            observed["closed"] = True

    def check_identity(*args):
        observed["identity"] = args

    def create_policy(config, path):
        observed["config"] = config
        observed["checkpoint"] = path
        return _ZeroPolicy()

    monkeypatch.setattr(evaluator.engram_dataset, "EngramDataset", FakeDataset)
    monkeypatch.setattr(evaluator.engram_dataset, "EngramMultiDataset", FakeDataset)
    monkeypatch.setattr(evaluator.engram_dataset, "check_stats_identity", check_identity)
    monkeypatch.setattr(evaluator.policy_config, "create_trained_policy", create_policy)
    options = {"video_mode": "proxy_640", "proxy_root": proxy_root, "single_camera": True} if use_proxy else {}
    evaluator.main(str(checkpoint), data_root=str(data_root), max_samples=2, **options)
    result = json.loads(capsys.readouterr().out)
    expected_mode = "proxy_640" if use_proxy else "original"
    expected_root = proxy_root
    root, horizon, loader = observed["dataset"]
    assert root == str(data_root)
    assert horizon == 64
    assert loader["split"] == "val"
    assert loader["video_mode"] == expected_mode
    assert loader["proxy_root"] == expected_root
    config = observed["config"]
    assert config.data.split == "val"
    assert config.data.single_camera is use_proxy
    assert config.data.video_mode == expected_mode
    assert config.data.proxy_root == expected_root
    assert observed["identity"][:3] == (manifest, str(data_root), "val")
    assert observed["checkpoint"] == checkpoint
    assert observed["closed"]
    assert result["checkpoint"] == str(checkpoint)
    assert result["data_root"] == str(data_root)
    assert result["single_camera"] is use_proxy
    assert result["video_mode"] == expected_mode
    assert result["proxy_root"] == expected_root
    assert result["validation_video_identity"] == ("a" * 64 if use_proxy else None)
    assert result["normalization_asset_id"] == "engram/dex_w1_sandwich_train"
    assert result["normalization_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result["normalization_stats_sha256"] == hashlib.sha256(stats.read_bytes()).hexdigest()
    assert result["joint_rmse_rad"] == 0.5
    assert result["eligible_anchors"] == 3


def test_evaluator_rejects_proxy_root_without_proxy_mode(tmp_path):
    with pytest.raises(ValueError, match="proxy_root requires"):
        evaluator.main(str(tmp_path), proxy_root="/proxies")


def test_validation_policy_uses_checkpoint_train_stats_without_noise_or_target_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(train_config._tokenizer, "PaligemmaTokenizer", lambda *_: object())  # noqa: SLF001
    monkeypatch.setattr(train_config.DataConfigFactory, "_load_norm_stats", lambda *_: None)
    base = train_config.get_config("pi05_engram_w1_sandwich")
    config = dataclasses.replace(base, data=dataclasses.replace(base.data, split="val", single_camera=True))
    data = config.data.create(config.assets_dirs, config.model)
    assert data.asset_id == "engram/dex_w1_sandwich_train"
    assert not any(isinstance(item, engram_policy.ExactJointNoiseCompensation) for item in data.data_transforms.inputs)
    assert any(isinstance(item, engram_policy.EngramInputs) and item.single_camera for item in data.data_transforms.inputs)

    stat = normalize.NormStats(mean=np.zeros(22), std=np.ones(22), q01=-np.ones(22), q99=np.ones(22))
    checkpoint_stats = {"state": stat, "actions": stat}
    captured = {}
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda path: tmp_path)
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *_args, **_kwargs: object())  # noqa: SLF001
    monkeypatch.setattr(type(config.model), "load", lambda *_args: object())

    def load_norm_stats(path, asset_id):
        captured["stats_path"] = path, asset_id
        return checkpoint_stats

    def make_policy(_model, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(policy_config._checkpoints, "load_norm_stats", load_norm_stats)  # noqa: SLF001
    monkeypatch.setattr(policy_config._policy, "Policy", make_policy)  # noqa: SLF001
    policy_config.create_trained_policy(config, tmp_path)
    assert captured["stats_path"] == (tmp_path / "assets", "engram/dex_w1_sandwich_train")
    norm = next(item for item in captured["transforms"] if isinstance(item, transforms.Normalize))
    unnorm = next(item for item in captured["output_transforms"] if isinstance(item, transforms.Unnormalize))
    assert norm.norm_stats is checkpoint_stats
    assert unnorm.norm_stats is checkpoint_stats

    target = np.arange(4 * 22, dtype=np.float32).reshape(4, 22)
    saved = target.copy()
    observation = {"observation/joint_position": np.arange(22, dtype=np.float32),
                   "observation/head_rgbd": np.zeros((4, 4, 3), dtype=np.uint8), "prompt": "sandwich"}
    for transform in data.data_transforms.inputs:
        observation = transform(observation)
    assert set(observation["image"]) == {"base_0_rgb"}
    np.testing.assert_array_equal(observation["state"], np.arange(22, dtype=np.float32))
    np.testing.assert_array_equal(target, saved)
