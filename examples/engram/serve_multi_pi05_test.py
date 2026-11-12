"""Physical request contracts and real OpenPI transforms for typed inference."""

import copy
import hashlib
import json
from types import SimpleNamespace

from etils import epath
import jax.numpy as jnp
import numpy as np
import pytest

from examples.engram import serve_multi_pi05 as serving
from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.shared.engram_multi import ACTION_DIM
from openpi.shared.engram_multi import EMBODIMENTS
from openpi.shared.engram_multi import ROUTE_NAMES
from openpi.shared.engram_multi import ROUTES
from openpi.shared.engram_multi import RouteStats
from openpi.shared.engram_multi import load_route_stats
from openpi.shared.engram_multi import physical_mask
from openpi.shared.engram_multi import save_route_stats
from openpi.training import checkpoints
from openpi.training import config as train_config
from openpi.training import engram_multi_dataset


@pytest.fixture
def route_stats():
    result = {}
    for index, spec in enumerate(EMBODIMENTS):
        mask = physical_mask(index, index != 3)
        # Route-dependent ranges catch accidental normalization with another head.
        lo = np.full(ACTION_DIM, -float(index + 1))
        hi = -lo
        lo[20:], hi[20:] = -100, 100
        state = normalize.NormStats(np.zeros(ACTION_DIM), np.ones(ACTION_DIM), lo * 2, hi * 2)
        actions = normalize.NormStats(np.zeros(ACTION_DIM), np.ones(ACTION_DIM), lo, hi)
        result[spec.name] = RouteStats(state, actions, mask, {
            "anchors": 2, "labeled_hand_anchors": 0 if index == 3 else 2,
        })
    return result


@pytest.fixture
def checkpoint(tmp_path, route_stats):
    root = tmp_path / "10000"
    (root / "params").mkdir(parents=True)
    (root / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 123}')
    assets = root / "assets" / serving.ASSET_ID
    save_route_stats(assets / "engram_multi_stats.json", route_stats)
    digest = hashlib.sha256((assets / "engram_multi_stats.json").read_bytes()).hexdigest()
    manifest = {
        "action_head_names": list(ROUTE_NAMES), "stats_sha256": digest, "action_horizon": 64,
        "max_frames": None, "split": "train", "total_anchors": 8, "used_anchors": 8,
        "sampled_anchors": 8, "max_token_len": 320,
        "routes": {name: stat.counts for name, stat in route_stats.items()},
        "source_root": "/unavailable/training/dataset",
        "prompts_by_route": {ROUTE_NAMES[0]: ["pick_up the cup"], ROUTE_NAMES[1]: ["pick_up the cup", "open drawer"]},
    }
    contract = {
        "format": "engram-multi-policy-v1", "action_dim": 64, "action_horizon": 64,
        "action_head_names": list(ROUTE_NAMES), "stats_sha256": digest, "routes": ROUTES,
        "preserve_prompt_text": True, "max_token_len": 320,
        "video": {"mode": "proxy_640", "profiling_subset_only": False},
        "camera_sizes": {robot: {role: list(serving.camera_size(robot, role))
                                 for role in ("head", "left_wrist", "right_wrist")}
                         for robot in ("dex_w1", "galbot")},
    }
    (assets / "engram_multi_manifest.json").write_text(json.dumps(manifest))
    (assets / "engram_multi_contract.json").write_text(json.dumps(contract))
    return root


def request(index=0, *, hands=True, structured=False):
    spec = EMBODIMENTS[index]
    width, height = serving.camera_size(spec.robot_type, "head")
    metadata = {"instruction": "pick_up the cup", "robot_type": spec.robot_type,
                "end_effector_type": spec.end_effector_type}
    result = {"prompt": metadata if structured else metadata["instruction"],
              "robot_type": spec.robot_type, "end_effector_type": spec.end_effector_type,
              "observation/joint_position": np.linspace(-0.8, 0.8, spec.joint_dim, dtype=np.float32),
              "observation/head_rgbd": np.zeros((height, width, 3), np.uint8)}
    if hands and index != 3:
        result["observation/hand_position"] = np.linspace(0, 100, spec.hand_dim, dtype=np.float32)
    return result


@pytest.mark.parametrize("index", range(4))
def test_prompt_formats_preserve_numeric_routing_and_missing_wrists(index, route_stats):
    seen = []
    for style in ("plain", "dict", "json"):
        data = request(index, structured=style != "plain")
        if style == "json":
            data["prompt"] = json.dumps(data["prompt"])
        sample, actual, available = serving.prepare_observation(data, route_stats)
        assert actual == index
        assert sample["embodiment_id"] == index
        assert sample["prompt"] == "pick_up the cup"
        assert sample["prompt_format"] == ("plain" if style == "plain" else "json")
        assert available == (index != 3)
        assert sample["image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": False,
                                        "right_wrist_0_rgb": False}
        assert not sample["image"]["left_wrist_0_rgb"].any()
        seen.append(sample["action_mask"])
    np.testing.assert_array_equal(seen[0], seen[1])
    np.testing.assert_array_equal(seen[1], seen[2])


def test_conflicting_metadata_and_unavailable_feedback_fail_closed(route_stats):
    data = request(structured=True)
    data["prompt"]["robot_type"] = "galbot"
    with pytest.raises(ValueError, match="Conflicting robot_type"):
        serving.prepare_observation(data, route_stats)
    data = request(hands=False, structured=True)
    data["prompt"]["end_effector_state_available"] = True
    with pytest.raises(ValueError, match="no measurement"):
        serving.prepare_observation(data, route_stats)
    data = request(structured=True)
    data["prompt"]["end_effector_state_available"] = False
    sample, _, available = serving.prepare_observation(data, route_stats)
    assert not available
    assert not sample["action_mask"][20:].any()
    assert not sample["state"][20:].any()
    data = request(3)
    data["observation/hand_position"] = np.zeros(12)
    with pytest.raises(ValueError, match="no supervised hand"):
        serving.prepare_observation(data, route_stats)
    data = request()
    data["action_mask"] = np.ones(64, bool)
    with pytest.raises(ValueError, match="action_mask"):
        serving.prepare_observation(data, route_stats)


def test_stereo_left_eye_and_robot_specific_wrist_geometry(route_stats):
    data = request()
    left = np.full((360, 640, 3), 17, np.uint8)
    data["observation/head_rgbd"] = np.concatenate([left, np.full_like(left, 200)], axis=1)
    data["head_camera_layout"] = "stereo"
    data["observation/left_wrist_rgbd"] = np.full((480, 640, 3), 31, np.uint8)
    sample, _, _ = serving.prepare_observation(data, route_stats)
    np.testing.assert_array_equal(sample["image"]["base_0_rgb"], left)
    assert sample["image_mask"]["left_wrist_0_rgb"]
    data = request(3)
    data["observation/right_wrist_rgbd"] = np.ones((360, 640, 3), np.uint8)
    sample, _, _ = serving.prepare_observation(data, route_stats)
    assert sample["image"]["base_0_rgb"].shape == (360, 480, 3)
    assert sample["image"]["right_wrist_0_rgb"].shape == (360, 640, 3)
    data["head_camera_layout"] = "stereo"
    with pytest.raises(ValueError, match="stereo|Stereo"):
        serving.prepare_observation(data, route_stats)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("hands", [False, True])
def test_physical_output_layouts_do_not_expose_padding(index, hands, route_stats):
    class Model:
        def infer(self, sample):
            assert sample["embodiment_id"] == index
            return {"actions": np.broadcast_to(np.arange(64), (64, 64))}

    output = serving.MultiEngramServingPolicy(Model(), route_stats, {}).infer(request(index, hands=hands))
    spec = EMBODIMENTS[index]
    has_hand = hands and index != 3
    assert output["actions"].shape == (64, spec.joint_dim + (spec.hand_dim if has_hand else 0))
    assert output["joint_names"] == ROUTES[spec.name]["joint_names"]
    np.testing.assert_array_equal(output["joint_actions"][0], np.arange(spec.joint_dim))
    if has_hand:
        np.testing.assert_array_equal(output["end_effector_actions"][0], np.arange(20, 20 + spec.hand_dim))
    else:
        assert "end_effector_actions" not in output


def test_checkpoint_assets_are_self_contained(checkpoint, monkeypatch):
    def no_source_access(*_args, **_kwargs):
        raise AssertionError("serving must not inspect the training source")

    monkeypatch.setattr(engram_multi_dataset, "stats_identity", no_source_access)
    stats, contract, manifest = serving.read_checkpoint_assets(checkpoint)
    assert tuple(stats) == ROUTE_NAMES
    assert manifest["used_anchors"] == 8
    assert contract["action_dim"] == 64
    path = checkpoint / "assets" / serving.ASSET_ID / "engram_multi_stats.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="contract"):
        serving.read_checkpoint_assets(checkpoint)


@pytest.mark.parametrize("mutation", ["camera", "routes", "subset", "token_limit", "coverage", "route_counts"])
def test_checkpoint_rejects_incompatible_training_contract(checkpoint, mutation):
    assets = checkpoint / "assets" / serving.ASSET_ID
    contract_file, manifest_file = assets / "engram_multi_contract.json", assets / "engram_multi_manifest.json"
    contract, manifest = json.loads(contract_file.read_text()), json.loads(manifest_file.read_text())
    if mutation == "camera":
        contract["camera_sizes"]["galbot"]["head"] = [640, 360]
    elif mutation == "routes":
        contract["action_head_names"].reverse()
    elif mutation == "subset":
        manifest["sampled_anchors"] = 7
    elif mutation == "token_limit":
        contract["max_token_len"] = manifest["max_token_len"] = None
    elif mutation == "coverage":
        manifest["routes"][ROUTE_NAMES[0]]["anchors"] = 0
    else:
        manifest["routes"][ROUTE_NAMES[0]]["labeled_hand_anchors"] = 0
    contract_file.write_text(json.dumps(contract))
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Checkpoint|checkpoint"):
        serving.read_checkpoint_assets(checkpoint)


@pytest.mark.parametrize("mutation", ["zero_range", "padded_supervision", "missing_route"])
def test_checkpoint_rejects_invalid_typed_stats_even_with_matching_hash(checkpoint, mutation):
    assets = checkpoint / "assets" / serving.ASSET_ID
    path = assets / "engram_multi_stats.json"
    stats = json.loads(path.read_text())
    route = stats["routes"][ROUTE_NAMES[0]]
    if mutation == "zero_range":
        route["norm_stats"]["actions"]["q99"][0] = route["norm_stats"]["actions"]["q01"][0]
    elif mutation == "padded_supervision":
        route["supervised_dims"][63] = True
    else:
        del stats["routes"][ROUTE_NAMES[3]]
    path.write_text(json.dumps(stats))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for filename in ("engram_multi_contract.json", "engram_multi_manifest.json"):
        target = assets / filename
        contract = json.loads(target.read_text())
        contract["stats_sha256"] = digest
        target.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="normalization|Statistics"):
        serving.read_checkpoint_assets(checkpoint)


def test_training_save_and_resume_preserve_typed_assets_and_reject_drift(checkpoint, monkeypatch, tmp_path):
    """Exercise production checkpoint callbacks without a heavyweight train state."""
    assets = checkpoint / "assets" / serving.ASSET_ID
    source = tmp_path / "training_assets"
    source.mkdir()
    stat_file = source / "engram_multi_stats.json"
    manifest_file = source / "engram_multi_manifest.json"
    stat_file.write_bytes((assets / stat_file.name).read_bytes())
    manifest_file.write_bytes((assets / manifest_file.name).read_bytes())
    config = SimpleNamespace(
        engram_multi_data_dir="/unavailable/training/dataset", asset_id=serving.ASSET_ID,
        engram_multi_stats_path=str(stat_file), engram_multi_manifest_path=str(manifest_file),
        engram_multi_stats_sha256=hashlib.sha256(stat_file.read_bytes()).hexdigest(),
        engram_multi_manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        engram_multi_structured_probability=0.5, engram_video_mode="proxy_640",
        engram_proxy_subset_only=False, engram_video_identity="proxy-identity", engram_stats_manifest=None,
    )
    loader = SimpleNamespace(data_config=lambda: config)
    captured = {}

    class Manager:
        directory = epath.Path(checkpoint.parent)

        def latest_step(self):
            return 10000

        def save(self, step, items):
            captured["callback"] = items["assets"]
            items["assets"](epath.Path(tmp_path / "saved_assets"))
            assert step == 10000

        def restore(self, step, *, items):
            captured["restored"] = True
            return {"train_state": "restored-state", "params": {"params": "restored-params"}}

    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ("state", "params"))
    monkeypatch.setattr(checkpoints, "_merge_params", lambda state, params: (state, params))
    checkpoints.save_state(Manager(), object(), loader, 10000)
    saved = tmp_path / "saved_assets" / serving.ASSET_ID
    assert (saved / "engram_multi_stats.json").read_bytes() == stat_file.read_bytes()
    assert (saved / "engram_multi_manifest.json").read_bytes() == manifest_file.read_bytes()
    assert json.loads((saved / "prompt.json").read_text()) == ["open drawer", "pick_up the cup"]
    # Resume compares the same saved camera, prompt and route contract.
    contract_path = assets / "engram_multi_contract.json"
    contract_path.write_bytes((saved / "engram_multi_contract.json").read_bytes())
    assert checkpoints.restore_state(Manager(), object(), loader) == ("restored-state", {"params": "restored-params"})
    config.engram_multi_structured_probability = 0.75
    with pytest.raises(ValueError, match="prompt, route, or camera"):
        checkpoints.restore_state(Manager(), object(), loader)
    config.engram_multi_structured_probability = 0.5
    stat_file.write_text(stat_file.read_text() + " ")
    with pytest.raises(ValueError, match="statistics or source selection"):
        checkpoints.restore_state(Manager(), object(), loader)
    with pytest.raises(ValueError, match="changed while the trainer"):
        captured["callback"](epath.Path(tmp_path / "changed_assets"))


@pytest.mark.parametrize("index", range(4))
def test_real_policy_normalization_restores_current_joints_exactly_once(checkpoint, monkeypatch, index):
    """Stub heavyweight weights, preserve Policy and every physical transform."""
    stats = load_route_stats(checkpoint / "assets" / serving.ASSET_ID / "engram_multi_stats.json")
    spec = EMBODIMENTS[index]
    captured = {}
    physical = np.zeros(64, np.float32)
    physical[:spec.joint_dim] = np.linspace(-0.2, 0.2, spec.joint_dim)
    physical[20:20 + spec.hand_dim] = np.linspace(-25 if index == 2 else 25, 75, spec.hand_dim)
    stat = stats[spec.name].actions
    normalized = (physical - stat.q01) / (stat.q99 - stat.q01 + 1e-6) * 2 - 1

    class Model:
        def sample_actions(self, _rng, observation, **_kwargs):
            captured["id"] = np.asarray(observation.embodiment_id)
            captured["mask"] = np.asarray(observation.action_mask)
            return jnp.broadcast_to(jnp.asarray(normalized), (1, 64, 64))

    class Tokenizer:
        def __init__(self, max_len, *, preserve_text=False, reject_truncation=False):
            assert preserve_text
            assert reject_truncation
            self.max_len = max_len

        def tokenize(self, prompt, state):
            captured["prompt"] = prompt
            captured["token_state"] = np.asarray(state)
            return np.zeros(self.max_len, np.int32), np.ones(self.max_len, bool)

    tokenizer = checkpoint.parent / "cache/big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("stub")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tokenizer.parent.parent))
    monkeypatch.setattr(train_config._tokenizer, "PaligemmaTokenizer", Tokenizer)  # noqa: SLF001
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *_a, **_kw: object())  # noqa: SLF001
    monkeypatch.setattr(type(train_config.get_config("pi05_engram_multi").model), "load", lambda *_a: Model())
    monkeypatch.setattr(policy_module.nnx_utils, "module_jit", lambda fn: fn)
    policy = serving.create_policy(checkpoint)
    for structured in (False, True):
        data = request(index, structured=structured)
        output = policy.infer(copy.deepcopy(data))
        expected = data["observation/joint_position"] + physical[:spec.joint_dim]
        np.testing.assert_allclose(output["joint_actions"], np.broadcast_to(expected, (64, spec.joint_dim)), atol=2e-5)
        np.testing.assert_array_equal(captured["id"], [index])
        np.testing.assert_array_equal(captured["mask"][0], physical_mask(index, index != 3))
        assert not captured["token_state"][~captured["mask"][0]].any()
        if index != 3:
            np.testing.assert_allclose(output["end_effector_actions"][0], physical[20:20 + spec.hand_dim], atol=2e-5)
        else:
            assert output["actions"].shape == (64, 14)
        if structured:
            prompt = json.loads(captured["prompt"])
            assert prompt["robot_type"] == spec.robot_type
            assert prompt["instruction"] == "pick_up the cup"

