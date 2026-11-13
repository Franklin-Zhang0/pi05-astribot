"""CPU checks for physical units, chunk anchors and optional input contracts."""

import json
from copy import deepcopy

import numpy as np
import pytest
from prism_engram.adapter import (
    ROUTE_NAMES,
    ROUTES,
    EngramAdapter,
    quaternion_wxyz_to_matrix,
    rotation_6d_to_matrix,
)


@pytest.fixture
def normalization():
    native = {}
    eef = {}
    for route in ROUTES:
        mask = np.zeros(64, bool)
        mask[: route.joint_dim] = True
        low, high = np.full(64, -1.0), np.full(64, 1.0)
        if route.hand_units != "unavailable":
            mask[20 : 20 + route.hand_dim] = True
            low[20 : 20 + route.hand_dim], high[20 : 20 + route.hand_dim] = -50, 50
        native[route.name] = {
            "supervised_dims": mask.tolist(),
            "norm_stats": {kind: {"q01": low.tolist(), "q99": high.tolist()} for kind in ("state", "actions")},
        }
        mean = np.zeros((5, 15))
        mean[[1, 3], 3:9] = [1, 0, 0, 0, 1, 0]
        eef[route.name] = (
            None
            if route.robot_type == "galbot"
            else {
                "mode": "mean_std_v1",
                "mean": mean.tolist(),
                "std": np.ones((5, 15)).tolist(),
            }
        )
    return {
        "layout": "engram_delta_joints_delta_hands_packed75_v2",
        "target_horizon": 32,
        "eligibility_horizon": 64,
        "route_statistics": {
            "format": "engram-route-delta-hand-stats-v2",
            "model_action_dim": 64,
            "route_names": list(ROUTE_NAMES),
            "routes": native,
        },
        "robot_eef_normalization": {
            "format": "prism_engram_observed_eef_normalization_v1",
            "horizon": 32,
            "frame_contract": {"frame": "inferred_native_torso_arm_base", "position_unit": "meters"},
            "routes": eef,
        },
    }


def request(route_id=0, hands=True, *, wrists=False):
    route = ROUTES[route_id]
    value = {
        "robot_type": route.robot_type,
        "end_effector_type": route.end_effector_type,
        "prompt": "pick up the cup",
        "observation/joint_position": np.linspace(-0.4, 0.4, route.joint_dim),
        "observation/head_rgbd": np.zeros((360, 640 if route_id < 3 else 480, 3), np.uint8),
    }
    if hands and route_id < 3:
        value["observation/hand_position"] = np.full(route.hand_dim, 20.0)
    if route_id < 3:
        value["observation/ee_poses"] = np.array(
            [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0]]
        )
    if wrists:
        value["observation/left_wrist_rgbd"] = np.zeros((480 if route_id < 3 else 360, 640, 3), np.uint8)
    return value


def predictions(adapter, prepared, *, joint_delta=0.125, hand_delta=7.5, delta_rotation=None):
    route = ROUTES[prepared.route_id]
    physical = np.zeros((32, 64), np.float32)
    physical[:, : route.joint_dim] = joint_delta
    physical[:, 20 : 20 + route.hand_dim] = hand_delta
    low, high = adapter.ranges[route.name, "actions"]
    native = np.zeros((32, 75), np.float32)
    native[:, :64] = 2 * (physical - low) / (high - low) - 1
    eef = np.zeros((32, 5, 15), np.float32)
    if route.robot_type == "dex_w1":
        eef[:, [1, 3], :3] = [0.1, -0.2, 0.3]
        if delta_rotation is None:
            delta_rotation = np.eye(3)
        eef[:, [1, 3], 3:9] = delta_rotation[:2].reshape(6)
        mean, std = adapter.eef_stats[route.name]
        eef = (eef - mean) / std
    return {"native_joint": native.reshape(1, 160, 15), "robot_eef": eef.reshape(1, 160, 15)}


@pytest.mark.parametrize("route_id", range(4))
def test_normalize_and_decode_physical_targets(normalization, route_id):
    adapter = EngramAdapter(normalization)
    sample = request(route_id)
    prepared = adapter.prepare(sample)
    route = ROUTES[route_id]
    low, high = adapter.ranges[route.name, "state"]
    expected = (
        2 * (prepared.joint_anchor - low[: route.joint_dim]) / (high[: route.joint_dim] - low[: route.joint_dim]) - 1
    )
    np.testing.assert_allclose(prepared.normalized_state[: route.joint_dim], expected)
    assert prepared.normalized_state.shape == (75,)
    assert prepared.native_action_mask.shape == (160, 15)
    assert prepared.robot_eef_mask.shape == (160, 15)
    assert not prepared.state_mask[64:].any()
    assert (prepared.camera_mask == [True, False, False]).all()
    decoded = adapter.decode(predictions(adapter, prepared), prepared)
    if route_id < 3:
        assert "joint_actions" not in decoded  # W1 arm joints are not EEF poses.
        np.testing.assert_allclose(
            decoded["body_head_actions"], np.broadcast_to(prepared.joint_anchor[:6] + 0.125, (32, 6)), atol=1e-6
        )
        np.testing.assert_allclose(decoded["end_effector_actions"], 27.5, atol=1e-5)
        np.testing.assert_allclose(
            decoded["eef_position"],
            np.broadcast_to(prepared.eef_anchor[:, :3] + [0.1, -0.2, 0.3], (32, 2, 3)),
            atol=1e-6,
        )
        assert decoded["eef_actions"].shape == (32, 2, 9)
        assert not prepared.native_action_mask.reshape(32, 75)[:, 6:20].any()
        assert prepared.robot_eef_mask.sum() == 32 * 2 * 9
        np.testing.assert_array_equal(decoded["eef_tick_offsets"], np.arange(1, 33))
    else:
        np.testing.assert_allclose(
            decoded["joint_actions"], np.broadcast_to(prepared.joint_anchor + 0.125, (32, 14)), atol=1e-6
        )
        assert decoded["joint_names"][0] == "right_arm_joint1"
        assert "eef_actions" not in decoded and "end_effector_actions" not in decoded
        assert not prepared.robot_eef_mask.any()
    np.testing.assert_array_equal(decoded["joint_hand_tick_offsets"], np.arange(32))


def test_rotation_uses_first_two_rows_and_left_composition(normalization):
    adapter = EngramAdapter(normalization)
    sample = request()
    root_half = np.sqrt(0.5)
    sample["observation/ee_poses"][:, 3:] = [root_half, 0, 0, root_half]  # Current Rz90.
    prepared = adapter.prepare(sample)
    delta = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # Delta Rx90.
    result = adapter.decode(predictions(adapter, prepared, delta_rotation=delta), prepared)
    anchor = quaternion_wxyz_to_matrix(prepared.eef_anchor[:, 3:])
    expected = delta @ anchor
    np.testing.assert_allclose(result["eef_rotation_matrix"], np.broadcast_to(expected, (32, 2, 3, 3)), atol=1e-6)
    assert not np.allclose(expected, anchor @ delta)
    np.testing.assert_allclose(result["eef_rotation_6d"][0], expected[:, :2].reshape(2, 6), atol=1e-6)
    np.testing.assert_allclose(
        rotation_6d_to_matrix(result["eef_rotation_6d"]), result["eef_rotation_matrix"], atol=1e-6
    )


@pytest.mark.parametrize("route_id", range(3))
def test_missing_hand_feedback_never_imputes_anchor(normalization, route_id):
    adapter = EngramAdapter(normalization)
    prepared = adapter.prepare(request(route_id, hands=False))
    assert prepared.hand_anchor is None
    assert not prepared.state_mask[20:].any()
    assert not prepared.normalized_state[20:].any()
    decoded = adapter.decode(predictions(adapter, prepared), prepared)
    assert not decoded["end_effector_available"]
    assert "end_effector_actions" not in decoded


def test_prompt_json_matches_training_order_and_route(normalization):
    adapter = EngramAdapter(normalization)
    sample = request(1)
    sample["prompt"] = {"instruction": "pick up the cup", "robot_type": "dex_w1", "end_effector_type": "linker_l6"}
    result = adapter.prepare(sample)
    assert result.prompt_text == (
        '{"task instruction":"pick up the cup","end effector type":"hand",'
        '"end effector number":"dual","embodiment":"robot","setting":"engram_dex_w1_linker_l6"}'
    )
    sample["prompt"] = {"task_instruction": "pick up the cup", "setting": "engram_dex_w1_linker_l6", "speed": None}
    assert adapter.prepare(sample).prompt_text == result.prompt_text
    sample["prompt"] = {"task instruction": "pick up the cup", "setting": None, "subtask instruction": None}
    assert json.loads(adapter.prepare(sample).prompt_text) == {
        "task instruction": "pick up the cup",
        "subtask instruction": "",
        "end effector type": "hand",
        "end effector number": "dual",
        "embodiment": "robot",
    }
    sample["prompt"] = "  literal {task}  "
    assert adapter.prepare(sample).prompt_text == sample["prompt"]
    sample["prompt"] = "{literal task"
    sample["prompt_format"] = "plain"
    assert adapter.prepare(sample).prompt_text == sample["prompt"]


def test_duplicate_or_conflicting_prompt_is_rejected(normalization):
    adapter = EngramAdapter(normalization)
    sample = request()
    sample["prompt"] = '{"task instruction":"a","task instruction":"b"}'
    with pytest.raises(ValueError, match="Malformed JSON"):
        adapter.prepare(sample)
    sample["prompt"] = {"task instruction": "pick", "setting": "engram_dex_w1_linker_l20"}
    with pytest.raises(ValueError, match="conflicts"):
        adapter.prepare(sample)
    sample["prompt"] = {"task instruction": "pick", "end effector type": "hand"}
    with pytest.raises(ValueError, match="conflicts"):
        adapter.prepare(sample)


def test_camera_slots_and_stereo_left_crop(normalization):
    adapter = EngramAdapter(normalization)
    sample = request(wrists=True)
    left = np.full((360, 640, 3), 17, np.uint8)
    sample["observation/head_rgbd"] = np.concatenate((left, np.full_like(left, 239)), axis=1)
    sample["head_camera_layout"] = "stereo"
    result = adapter.prepare(sample)
    np.testing.assert_array_equal(result.frames["image"], left)
    np.testing.assert_array_equal(result.camera_mask, [True, True, False])
    sample["observation/head_rgbd"] = np.zeros((360, 640, 3), np.uint8)
    with pytest.raises(ValueError, match="aspect ratio"):
        adapter.prepare(sample)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"observation/ee_poses": None}, "requires current"),
        ({"observation/ee_poses": np.zeros((2, 7))}, "quaternions"),
        ({"observation/ee_poses": np.zeros((2, 6))}, "Current left/right"),
        ({"observation/joint_position": np.zeros(75)}, "Measured joint"),
        ({"observation/hand_position": [101, 20]}, "percentage"),
        ({"observation/hand_position": [20, 20], "end_effector_state_available": False}, "unavailable"),
        ({"end_effector_state_available": "false"}, "boolean"),
        ({"observation/head_rgbd": np.zeros((360, 640, 3), np.float32)}, "uint8 RGB"),
        ({"action_mask": np.ones(75)}, "derived internally"),
        ({"eef_frame": "world"}, "saved training frame"),
        ({"eef_position_units": "millimeters"}, "meters"),
        ({"joint_units": "degrees"}, "radians"),
        ({"hand_units": "radians"}, "recorded route"),
    ],
)
def test_invalid_measurements_fail_closed(normalization, change, match):
    sample = request()
    sample.update(change)
    with pytest.raises(ValueError, match=match):
        EngramAdapter(normalization).prepare(sample)


def test_galbot_rejects_unsupervised_hand_state(normalization):
    sample = request(3)
    sample["observation/hand_position"] = np.zeros(12)
    with pytest.raises(ValueError, match="no training supervision"):
        EngramAdapter(normalization).prepare(sample)


def test_bad_stats_and_rotation_predictions_fail_closed(normalization):
    bad = deepcopy(normalization)
    bad["route_statistics"]["routes"][ROUTE_NAMES[0]]["norm_stats"]["actions"]["q99"][0] = -0.99
    with pytest.raises(ValueError, match="physical minimum"):
        EngramAdapter(bad)
    adapter = EngramAdapter(normalization)
    sample = adapter.prepare(request())
    prediction = predictions(adapter, sample)
    prediction["native_joint"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        adapter.decode(prediction, sample)
    prediction = predictions(adapter, sample)
    # Padding is never interpreted as a physical value.
    prediction["native_joint"].reshape(32, 75)[:, 64:] = np.nan
    adapter.decode(prediction, sample)
    with pytest.raises(ValueError, match="Degenerate"):
        rotation_6d_to_matrix(np.zeros(6))


def test_actual_checkpoint_statistics_are_accepted_when_available():
    # Optional developer integration check; exported releases never require this path.
    # A standalone fixture covers the same schema above.
    import os
    from pathlib import Path

    filename = os.environ.get("PRISM_TEST_NORMALIZATION")
    if not filename:
        pytest.skip("Set PRISM_TEST_NORMALIZATION for a checkpoint-owned statistics integration check")
    adapter = EngramAdapter(json.loads(Path(filename).read_text()))
    for index in range(4):
        prepared = adapter.prepare(request(index))
        decoded = adapter.decode(predictions(adapter, prepared), prepared)
        assert decoded["action_horizon"] == 32
