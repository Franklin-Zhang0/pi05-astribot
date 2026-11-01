import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from openpi_client import websocket_client_policy
import pytest
import websockets.asyncio.server

from examples.engram import serve_pi05 as serving
from openpi import transforms
from openpi.models.model import ModelType
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import engram_policy
from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config as train_config
from openpi.training import engram_dataset


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "10000"
    (path / "params").mkdir(parents=True)
    (path / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 123456789}')
    assets = path / "assets" / serving.ASSET_ID
    state_q01 = np.array([-5.0] * 20 + [0.0] * 2)
    state_q99 = np.array([5.0] * 20 + [100.0] * 2)
    action_q01 = np.array([-2.0] * 20 + [0.0] * 2)
    action_q99 = np.array([2.0] * 20 + [100.0] * 2)
    def stat(lo, hi):
        return normalize.NormStats(mean=np.zeros(22), std=np.ones(22), q01=lo, q99=hi)

    normalize.save(assets, {"state": stat(state_q01, state_q99), "actions": stat(action_q01, action_q99)})
    digest = hashlib.sha256((assets / "norm_stats.json").read_bytes()).hexdigest()
    manifest = {"contract": engram_dataset.STATS_CONTRACT, "split": "train", "action_horizon": 64,
                "max_frames": None, "norm_stats_sha256": digest, "prune_threshold_rad": 0.01,
                "gripper_threshold_percent": 0.0, "joint_quantile_span_floor_radians": 0.1,
                "gripper_quantile_span_floor_percent": 5.0, "total_training_anchors": 4,
                "sampled_anchors": 4, "labeled_gripper_anchors": 2,
                "source_manifest_sha256": "b" * 64,
                "source_root": "/training/mount/deliberately/unavailable"}
    (assets / "engram_stats_manifest.json").write_text(json.dumps(manifest))
    (assets / "engram_video_contract.json").write_text(json.dumps({
        "version": 1, "mode": "proxy_640", "profiling_subset_only": False, "proxy_identity": "a" * 64,
    }))
    return path


def test_checkpoint_assets_are_local_pinned_and_fail_closed(checkpoint, monkeypatch):
    def no_dataset_access(*_args, **_kwargs):
        raise AssertionError("inference must not inspect training dataset paths")

    monkeypatch.setattr(engram_dataset, "check_stats_identity", no_dataset_access)
    stats, metadata = serving.read_checkpoint_assets(checkpoint)
    assert set(stats) == {"state", "actions"}
    assert metadata["training_proxy_identity"] == "a" * 64
    assert metadata["training_source_manifest_sha256"] == "b" * 64
    assert metadata["normalization_asset_id"] == serving.ASSET_ID
    serving.read_checkpoint_assets(checkpoint, expected_source_manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="must be 64 lowercase"):
        serving.read_checkpoint_assets(checkpoint, expected_source_manifest_sha256="malformed")
    with pytest.raises(ValueError, match="does not match the expected"):
        serving.read_checkpoint_assets(checkpoint, expected_source_manifest_sha256="c" * 64)
    assets = checkpoint / "assets" / serving.ASSET_ID
    norm_file = assets / "norm_stats.json"
    norm_file.write_text(norm_file.read_text() + " ")
    with pytest.raises(ValueError, match="contract or hash"):
        serving.read_checkpoint_assets(checkpoint)
    norm_file.unlink()
    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        serving.read_checkpoint_assets(checkpoint)


def test_checkpoint_requires_completed_params_and_correct_video_contract(checkpoint):
    marker = checkpoint / "_CHECKPOINT_METADATA"
    marker.unlink()
    with pytest.raises(FileNotFoundError, match="_CHECKPOINT_METADATA"):
        serving.read_checkpoint_assets(checkpoint)
    marker.write_text('{"commit_timestamp_nsecs": 123456789}')
    path = checkpoint / "assets" / serving.ASSET_ID / "engram_video_contract.json"
    video = json.loads(path.read_text())
    video["profiling_subset_only"] = True
    path.write_text(json.dumps(video))
    with pytest.raises(ValueError, match="video contract"):
        serving.read_checkpoint_assets(checkpoint)


@pytest.mark.parametrize("commit_time", [None, 0, -1, True, "123"])
def test_checkpoint_rejects_incomplete_or_invalid_orbax_commit(checkpoint, commit_time):
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": commit_time}))
    with pytest.raises(ValueError, match="completed Orbax commit timestamp"):
        serving.read_checkpoint_assets(checkpoint)


def test_checkpoint_rejects_orbax_temporary_name_but_allows_copied_name(checkpoint):
    temporary = checkpoint.with_name("10000.orbax-checkpoint-tmp-1")
    checkpoint.rename(temporary)
    with pytest.raises(ValueError, match="temporary checkpoint"):
        serving.read_checkpoint_assets(temporary)
    copied = temporary.with_name("my-copied-checkpoint")
    temporary.rename(copied)
    serving.read_checkpoint_assets(copied)


def test_checkpoint_accepts_completed_local_orbax_manager_save(checkpoint, tmp_path):
    parent = tmp_path / "orbax-manager"
    manager, resumed = checkpoints.initialize_checkpoint_dir(parent, keep_period=10000, overwrite=False, resume=False)
    assert not resumed

    def save_assets(directory):
        shutil.copytree(checkpoint / "assets", str(directory), dirs_exist_ok=True)

    try:
        saved = manager.save(10000, {"assets": save_assets, "train_state": {"step": np.array(1)},
                                     "params": {"params": {"weight": np.array([0.1])}}})
        assert saved
        manager.wait_until_finished()
    finally:
        manager.close()
    completed = parent / "10000"
    metadata = json.loads((completed / "_CHECKPOINT_METADATA").read_text())
    assert type(metadata["commit_timestamp_nsecs"]) is int
    assert metadata["commit_timestamp_nsecs"] > 0
    serving.read_checkpoint_assets(completed)


def test_checkpoint_rejects_malformed_source_or_ambiguous_weight_file(checkpoint):
    path = checkpoint / "assets" / serving.ASSET_ID / "engram_stats_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_manifest_sha256"] = "not-a-sha256"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="multisource manifest SHA-256"):
        serving.read_checkpoint_assets(checkpoint)
    manifest["source_manifest_sha256"] = "b" * 64
    path.write_text(json.dumps(manifest))
    (checkpoint / "model.safetensors").write_bytes(b"not-the-orbax-weights")
    with pytest.raises(ValueError, match="would override"):
        serving.read_checkpoint_assets(checkpoint)


def test_camera_left_and_stereo_are_explicit_and_area_resized():
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    np.testing.assert_array_equal(serving.prepare_camera(left, "left"), left)
    large_left = np.repeat(np.repeat(left, 2, axis=0), 2, axis=1)
    right = np.full_like(large_left, 17)
    stereo = np.concatenate((large_left, right), axis=1)
    np.testing.assert_array_equal(serving.prepare_camera(stereo, "stereo"), left)
    np.testing.assert_array_equal(serving.prepare_camera(large_left, "left"), left)
    with pytest.raises(ValueError, match="aspect ratio"):
        serving.prepare_camera(stereo, "left")
    with pytest.raises(ValueError, match="aspect ratio"):
        serving.prepare_camera(left, "stereo")


class RecordingPolicy:
    def __init__(self):
        self.seen = []

    def infer(self, observation):
        self.seen.append(observation)
        return {"actions": np.tile(np.arange(22, dtype=np.float32), (64, 1))}


def observation(state, **kwargs):
    return {"observation/head_rgbd": np.zeros((360, 640, 3), dtype=np.uint8),
            "observation/joint_position": state, "prompt": "open the bag", **kwargs}


def test_missing_gripper_mask_controls_state_and_returned_commands():
    backing = RecordingPolicy()
    policy = serving.EngramServingPolicy(backing, camera_layout="left", metadata={})
    joints = np.linspace(-1, 1, 20, dtype=np.float32)
    short = policy.infer(observation(joints))
    assert short["actions"].shape == (64, 20)
    assert short["gripper_available"] is False
    np.testing.assert_array_equal(backing.seen[-1]["observation/joint_position"][20:], [0, 0])
    np.testing.assert_array_equal(backing.seen[-1]["action_mask"], [True] * 20 + [False] * 2)
    np.testing.assert_array_equal(short["actions"][0], np.arange(20))
    masked = policy.infer(observation(np.r_[joints, 5, 95], action_mask=np.array([True] * 20 + [False] * 2)))
    assert masked["actions"].shape == (64, 20)
    assert masked["gripper_available"] is False
    np.testing.assert_array_equal(backing.seen[-1]["observation/joint_position"][20:], [0, 0])
    labeled = policy.infer(observation(np.r_[joints, 5, 95]))
    assert labeled["actions"].shape == (64, 22)
    assert labeled["gripper_available"] is True
    np.testing.assert_array_equal(labeled["actions"][0], np.arange(22))


def test_reject_ambiguous_or_invalid_observations():
    invalid = [
        (observation(np.zeros(21)), "Current state"),
        (observation(np.r_[np.zeros(20), -1, 20]), "gripper positions"),
        (observation(np.zeros(20), action_mask=np.ones(22, dtype=bool)), "action_mask"),
        (observation(np.zeros(22), action_mask=np.array([True] * 21 + [False])), "action_mask"),
        (observation(np.zeros(20), prompt=""), "task prompt"),
        (observation(np.zeros(20), **{"observation/left_wrist_rgbd": np.zeros((1, 1, 3), np.uint8)}), "wrist"),
    ]
    for item, message in invalid:
        with pytest.raises(ValueError, match=message):
            serving.prepare_observation(item, "left")


def test_real_policy_transforms_delta_to_absolute_once_for_both_masks(checkpoint, monkeypatch):
    """Only model restore/tokenizer are stubbed; OpenPI transformations are real."""
    observed = {}
    delta = np.linspace(-0.2, 0.2, 20, dtype=np.float32)
    physical_delta = np.r_[delta, [25.0, 75.0]]
    q01 = np.array([-2.0] * 20 + [0.0] * 2)
    q99 = np.array([2.0] * 20 + [100.0] * 2)
    normalized = ((physical_delta - q01) / (q99 - q01 + 1e-6)) * 2 - 1
    normalized = np.r_[normalized, np.zeros(10, dtype=np.float64)].astype(np.float32)

    class Model:
        def sample_actions(self, _rng, obs, **_kwargs):
            observed["model_state"] = np.asarray(obs.state)
            observed["model_images"] = obs.images
            observed["mask"] = np.asarray(obs.action_mask)
            return jnp.broadcast_to(jnp.asarray(normalized), (1, 64, 32))

    class Tokenizer:
        def __init__(self, max_len):
            self._max_len = max_len

            class Piece:
                def encode(self, *_args, **_kwargs):
                    return [1, 2, 3]

            self._tokenizer = Piece()

        def tokenize(self, prompt, state):
            observed["prompt"] = prompt
            observed["token_state"] = np.asarray(state).copy()
            self._tokenizer.encode(prompt, add_bos=True)
            return np.zeros(200, np.int32), np.ones(200, bool)

    tokenizer = checkpoint.parent / "cache/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("stub")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tokenizer.parent.parent))
    monkeypatch.setattr(train_config._tokenizer, "PaligemmaTokenizer", Tokenizer)  # noqa: SLF001
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *_a, **_kw: object())  # noqa: SLF001
    monkeypatch.setattr(type(train_config.get_config("pi05_engram_w1_sandwich").model), "load", lambda *_a: Model())
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda function: function)
    policy = serving.create_policy(checkpoint, camera_layout="left")
    joints = np.linspace(-1, 1, 20, dtype=np.float32)
    left = np.random.default_rng(4).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    request = observation(np.r_[joints, 10.0, 90.0], **{"observation/head_rgbd": left})
    available = policy.infer(request)
    np.testing.assert_allclose(available["actions"][:, :20], np.broadcast_to(joints + delta, (64, 20)), atol=2e-6)
    np.testing.assert_allclose(available["actions"][:, 20:], np.broadcast_to([25, 75], (64, 2)), atol=2e-5)
    assert observed["prompt"] == "open the bag"
    assert set(observed["model_images"]) == {"base_0_rgb"}
    assert observed["model_images"]["base_0_rgb"].shape == (1, 224, 224, 3)
    assert observed["mask"][0, 20:22].all()
    left_model_image = np.asarray(observed["model_images"]["base_0_rgb"]).copy()
    stereo = np.concatenate((left, np.full_like(left, 193)), axis=1)
    stereo_policy = serving.EngramServingPolicy(policy._policy, camera_layout="stereo", metadata={})  # noqa: SLF001
    stereo_request = observation(np.r_[joints, 10.0, 90.0], **{"observation/head_rgbd": stereo})
    stereo_result = stereo_policy.infer(stereo_request)
    np.testing.assert_array_equal(observed["model_images"]["base_0_rgb"], left_model_image)
    np.testing.assert_allclose(stereo_result["actions"], available["actions"], atol=2e-6)
    missing = policy.infer(observation(np.r_[joints, 10.0, 90.0], action_mask=np.array([True] * 20 + [False] * 2)))
    np.testing.assert_allclose(missing["actions"], np.broadcast_to(joints + delta, (64, 20)), atol=2e-6)
    assert observed["prompt"] == "End-effector state unavailable. open the bag"
    np.testing.assert_array_equal(observed["token_state"][20:22], [0, 0])
    assert not observed["mask"][0, 20:22].any()


def test_missing_tokenizer_fails_before_model_load(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "no_tokenizer"))

    def reject_model_load(*_args, **_kwargs):
        raise AssertionError("Model load must not run without the tokenizer")

    monkeypatch.setattr(policy_config, "create_trained_policy", reject_model_load)
    with pytest.raises(FileNotFoundError, match="PaliGemma tokenizer is missing"):
        serving.create_policy(checkpoint, camera_layout="left")


def test_prompt_guard_rejects_an_unexpected_policy_pipeline():
    policy = SimpleNamespace(_input_transform=transforms.compose([]))
    with pytest.raises(ValueError, match="exactly one discrete-state"):
        serving.reject_truncated_prompts(policy, 200)


@pytest.mark.parametrize("has_gripper", [False, True])
def test_real_tokenizer_keeps_full_state_at_boundary_and_rejects_long_prompts(checkpoint, has_gripper):
    cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    if not (cache / "big_vision/paligemma_tokenizer.model").is_file():
        pytest.skip("Set OPENPI_DATA_HOME to a local PaliGemma tokenizer cache")
    tokenizer = PaligemmaTokenizer(200)
    reference = copy.copy(tokenizer)
    reference._max_len = 512  # noqa: SLF001
    stats, _ = serving.read_checkpoint_assets(checkpoint)
    prefix = [engram_policy.EngramInputs(model_type=ModelType.PI05, model_action_dim=32, single_camera=True),
              transforms.Normalize(stats, use_quantiles=True), engram_policy.MaskUnavailableState()]
    policy = SimpleNamespace(_input_transform=transforms.compose([
        *prefix, transforms.TokenizePrompt(tokenizer, discrete_state_input=True),
    ]))
    serving.reject_truncated_prompts(policy, 200)
    state = np.r_[np.linspace(-1, 1, 20), 10, 90]
    mask = np.array([True] * 20 + [has_gripper] * 2)

    def sample_and_reference(n):
        raw = observation(state, prompt="open " * n, action_mask=mask)
        prepared, _ = serving.prepare_observation(raw, "left")
        processed = transforms.compose(prefix)(prepared)
        token_ids, token_mask = reference.tokenize(processed["prompt"], processed["state"])
        return prepared, processed, token_ids, token_mask

    boundary = oversize = None
    for n in range(1, 240):
        sample, processed, token_ids, token_mask = sample_and_reference(n)
        count = int(token_mask.sum())
        if count == 200:
            boundary = sample, processed, token_ids
        if count == 201:
            oversize = sample
            break
    assert boundary is not None
    assert oversize is not None
    sample, processed, token_ids = boundary
    assert processed["prompt"].startswith("End-effector state unavailable.") is (not has_gripper)
    if not has_gripper:
        np.testing.assert_array_equal(processed["state"][20:], [0, 0])
    result = policy._input_transform(sample)  # noqa: SLF001
    assert result["tokenized_prompt_mask"].sum() == 200
    np.testing.assert_array_equal(result["tokenized_prompt"], token_ids[:200])
    with pytest.raises(ValueError, match="201 tokens"):
        policy._input_transform(oversize)  # noqa: SLF001


def test_cli_propagates_source_pin_camera_layout_and_bind(tmp_path, monkeypatch, capsys):
    captured = {}
    policy = serving.EngramServingPolicy(RecordingPolicy(), camera_layout="stereo", metadata={"snapshot": "ok"})

    def create(checkpoint, *, camera_layout, expected_source_manifest_sha256):
        captured["create"] = checkpoint, camera_layout, expected_source_manifest_sha256
        return policy

    class Server:
        def __init__(self, *, policy, host, port, metadata):
            captured["server"] = policy, host, port, metadata

        def serve_forever(self):
            captured["served"] = True

    monkeypatch.setattr(serving, "create_policy", create)
    monkeypatch.setattr(serving.websocket_policy_server, "WebsocketPolicyServer", Server)
    monkeypatch.setattr(sys, "argv", ["serve_pi05.py", "--checkpoint-dir", str(tmp_path), "--host", "127.0.0.1",
                                      "--port", "8011", "--camera-layout", "stereo",
                                      "--expected-source-manifest-sha256", "b" * 64])
    serving.main()
    assert captured["create"] == (tmp_path, "stereo", "b" * 64)
    assert captured["server"] == (policy, "127.0.0.1", 8011, policy.metadata)
    assert captured["served"]
    assert json.loads(capsys.readouterr().out)["event"] == "ENGRAM_POLICY_LOADED"


def test_existing_websocket_protocol_round_trip():
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Sandbox does not permit localhost sockets")

    async def exercise():
        policy = serving.EngramServingPolicy(RecordingPolicy(), camera_layout="left", metadata={"example": "engram"})
        wrapper = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=0,
                                                                 metadata=policy.metadata)
        async with websockets.asyncio.server.serve(wrapper._handler, "127.0.0.1", 0, compression=None,  # noqa: SLF001
                                                    max_size=None) as server:
            port = server.sockets[0].getsockname()[1]

            def client_calls():
                client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
                try:
                    return (client.get_server_metadata(), client.infer(observation(np.zeros(20))),
                            client.infer(observation(np.r_[np.zeros(20), 20, 80])))
                finally:
                    client._ws.close()  # noqa: SLF001

            metadata, missing, labeled = await asyncio.wait_for(asyncio.to_thread(client_calls), timeout=10)
            assert metadata["example"] == "engram"
            assert missing["actions"].shape == (64, 20)
            assert not missing["gripper_available"]
            assert labeled["actions"].shape == (64, 22)
            assert labeled["gripper_available"]

    asyncio.run(exercise())
