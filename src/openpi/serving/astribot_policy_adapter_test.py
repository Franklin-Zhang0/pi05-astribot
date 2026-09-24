import io
from typing import ClassVar

import numpy as np
from PIL import Image
import pytest

from openpi.serving.astribot_policy_adapter import GRIPPER_MODEL_ACTION_DIM
from openpi.serving.astribot_policy_adapter import MODEL_ACTION_DIM
from openpi.serving.astribot_policy_adapter import ACTTemporalEnsembler
from openpi.serving.astribot_policy_adapter import AstribotPolicyAdapter
from openpi.serving.astribot_policy_adapter import decode_camera_payload
from openpi.serving.astribot_policy_adapter import gripper_model_state_from_wire
from openpi.serving.astribot_policy_adapter import hold_action_from_wire
from openpi.serving.astribot_policy_adapter import map_gripper_model_action
from openpi.serving.astribot_policy_adapter import map_model_action_to_canonical
from openpi.serving.astribot_policy_adapter import model_state_from_wire
from openpi.serving.astribot_policy_schema import ACTION_DIM
from openpi.serving.astribot_policy_schema import CANONICAL_BODY_JOINTS
from openpi.serving.astribot_policy_schema import HAND_CONTROL_JOINTS
from openpi.serving.astribot_policy_schema import HAND_STATE_JOINTS
from openpi.serving.astribot_policy_schema import LEFT_HAND_ACTION_SLICE
from openpi.serving.astribot_policy_schema import RIGHT_HAND_ACTION_SLICE
from openpi.serving.astribot_policy_schema import enabled_action_mask


def _encoded_png(image: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _named_values(names: tuple[str, ...], canonical_values: np.ndarray) -> tuple[list[str], np.ndarray]:
    received_names = list(reversed(names))
    index = {name: position for position, name in enumerate(names)}
    received_values = np.asarray([canonical_values[index[name]] for name in received_names])
    return received_names, received_values


def _request() -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    body_values = np.arange(23, dtype=np.float64) + 0.5
    left_values = np.arange(21, dtype=np.float64) + 100.5
    right_values = np.arange(21, dtype=np.float64) + 200.5
    body_names, body_position = _named_values(CANONICAL_BODY_JOINTS, body_values)
    left_names, left_position = _named_values(HAND_STATE_JOINTS, left_values)
    right_names, right_position = _named_values(HAND_STATE_JOINTS, right_values)

    head = np.zeros((2, 4, 3), dtype=np.uint8)
    head[:, :2] = (10, 20, 30)
    head[:, 2:] = (200, 210, 220)
    left = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    right = (left + 30).astype(np.uint8)
    request = {
        "type": "observation",
        "enabled_action_mask": enabled_action_mask(
            control_scope="upper_body",
            body=True,
            left_hand=True,
            right_hand=True,
        ),
        "prompt": "test prompt",
        "observation": {
            "body": {
                "valid": True,
                "joint_names": body_names,
                "position": body_position,
            },
            "hand_left": {
                "valid": True,
                "joint_names": left_names,
                "position": left_position,
            },
            "hand_right": {
                "valid": True,
                "joint_names": right_names,
                "position": right_position,
            },
            "images": {
                "head_stereo": {"valid": True, "encoding": "png", "payload": _encoded_png(head)},
                "left_wrist_rgbd": {"valid": True, "encoding": "png", "payload": _encoded_png(left)},
                "right_wrist_rgbd": {"valid": True, "encoding": "png", "payload": _encoded_png(right)},
            },
        },
    }
    return request, body_values, left_values, right_values


def test_camera_decode_crops_stereo_and_rotates_wrist_before_resize() -> None:
    request, _, _, _ = _request()
    images = request["observation"]["images"]  # type: ignore[index]
    head = decode_camera_payload("head_stereo", images["head_stereo"])  # type: ignore[index]
    left = decode_camera_payload("left_wrist_rgbd", images["left_wrist_rgbd"])  # type: ignore[index]

    assert head.shape == (2, 2, 3)
    assert np.all(head == (10, 20, 30))
    assert np.array_equal(
        left,
        np.asarray(
            Image.fromarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3)).transpose(Image.Transpose.ROTATE_180)
        ),
    )


def test_act_temporal_ensembler_weights_oldest_surviving_prediction() -> None:
    ensembler = ACTTemporalEnsembler(action_horizon=3, ensemble_decay=float(np.log(2.0)))
    enabled = np.ones(ACTION_DIM, dtype=np.bool_)
    hold = np.zeros(ACTION_DIM, dtype=np.float64)

    first_chunk = np.stack([np.full(ACTION_DIM, value, dtype=np.float64) for value in (1.0, 2.0, 3.0)])
    second_chunk = np.stack([np.full(ACTION_DIM, value, dtype=np.float64) for value in (10.0, 11.0, 12.0)])
    third_chunk = np.stack([np.full(ACTION_DIM, value, dtype=np.float64) for value in (20.0, 21.0, 22.0)])

    first = ensembler.update(
        session_id="test",
        control_epoch=0,
        sequence=0,
        actions=first_chunk,
        enabled_mask=enabled,
        hold_action=hold,
    )
    second = ensembler.update(
        session_id="test",
        control_epoch=0,
        sequence=1,
        actions=second_chunk,
        enabled_mask=enabled,
        hold_action=hold,
    )
    third = ensembler.update(
        session_id="test",
        control_epoch=0,
        sequence=2,
        actions=third_chunk,
        enabled_mask=enabled,
        hold_action=hold,
    )

    assert np.allclose(first, 1.0)
    assert np.allclose(second, (2.0 + 0.5 * 10.0) / 1.5)
    assert np.allclose(third, (3.0 + 0.5 * 11.0 + 0.25 * 20.0) / 1.75)


def test_adapter_ensembles_chunks_and_holds_unsupported_joints() -> None:
    request, body_values, _, _ = _request()
    request = {**request, "request_sequence": 0, "session_id": "ensemble-test"}

    class ChunkPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def infer(self, observation: dict[str, object]) -> dict[str, object]:
            del observation
            base = self.calls * 100.0
            self.calls += 1
            return {
                "actions": np.stack(
                    [np.full(MODEL_ACTION_DIM, base + row, dtype=np.float32) for row in range(3)]
                )
            }

    policy = ChunkPolicy()
    adapter = AstribotPolicyAdapter(
        policy,
        checkpoint_id="test",
        action_horizon=2,
        ensemble_decay=float(np.log(2.0)),
    )
    first, _ = adapter.infer_observation(request)
    second_request = {**request, "request_sequence": 1}
    second, _ = adapter.infer_observation(second_request)

    # The configured horizon discards the model's third row. At sequence 1,
    # ACT weighting combines row 1 from the first chunk and row 0 from the
    # second chunk as 1.0 and 100.0 with weights 1.0 and 0.5.
    assert np.isclose(first[6], 0.0)
    assert np.isclose(second[6], 34.0)
    assert np.array_equal(second[:3], body_values[:3])
    assert np.array_equal(second[3:6], body_values[3:6])
    assert adapter.metadata["action_horizon"] == 2
    assert adapter.metadata["ensemble_decay"] == float(np.log(2.0))


def test_adapter_accepts_normal_head_rgbd_stream_when_stereo_is_not_selected() -> None:
    request, _, _, _ = _request()
    observation = dict(request["observation"])
    images = dict(observation["images"])
    images["head_rgbd"] = images.pop("head_stereo")
    observation["images"] = images
    request["observation"] = observation

    class FakePolicy:
        def __init__(self) -> None:
            self.observation = None

        def infer(self, policy_observation: dict[str, object]) -> dict[str, object]:
            self.observation = policy_observation
            return {"actions": np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32)}

    policy = FakePolicy()
    AstribotPolicyAdapter(policy, checkpoint_id="test").infer_observation(request)

    assert policy.observation is not None
    assert policy.observation["observation/head_rgbd"].shape == (2, 4, 3)  # type: ignore[index]


def test_mapping_keeps_non_policy_dimensions_at_proprioception() -> None:
    request, body_values, left_values, right_values = _request()
    hold = hold_action_from_wire(request)
    model_action = 1000.0 + np.arange(MODEL_ACTION_DIM, dtype=np.float64)
    mask = request["enabled_action_mask"]
    mapped = map_model_action_to_canonical(model_action, hold, mask)  # type: ignore[arg-type]

    assert np.array_equal(mapped[:3], body_values[:3])
    assert np.array_equal(mapped[3:6], body_values[3:6])  # upper-body lean is disabled
    assert np.array_equal(mapped[6], model_action[3])
    assert np.array_equal(mapped[9:16], model_action[4:11])
    assert np.array_equal(mapped[16:23], model_action[11:18])
    assert np.array_equal(mapped[7:9], model_action[18:20])
    expected_left = left_values[[HAND_STATE_JOINTS.index(name) for name in HAND_CONTROL_JOINTS]]
    expected_right = right_values[[HAND_STATE_JOINTS.index(name) for name in HAND_CONTROL_JOINTS]]
    assert np.array_equal(hold[LEFT_HAND_ACTION_SLICE], expected_left)
    assert np.array_equal(hold[RIGHT_HAND_ACTION_SLICE], expected_right)
    assert np.array_equal(mapped[LEFT_HAND_ACTION_SLICE], model_action[20:36])
    assert np.array_equal(mapped[RIGHT_HAND_ACTION_SLICE], model_action[36:52])


def test_adapter_reorders_state_and_maps_policy_output() -> None:
    request, body_values, left_values, right_values = _request()
    expected_state = np.concatenate(
        (
            body_values[3:7],
            body_values[9:16],
            body_values[16:23],
            body_values[7:9],
            left_values[[HAND_STATE_JOINTS.index(name) for name in HAND_CONTROL_JOINTS]],
            right_values[[HAND_STATE_JOINTS.index(name) for name in HAND_CONTROL_JOINTS]],
        )
    ).astype(np.float32)
    assert np.array_equal(model_state_from_wire(request), expected_state)

    class FakePolicy:
        metadata: ClassVar[dict[str, str]] = {"policy_name": "fake"}

        def __init__(self) -> None:
            self.observation = None

        def infer(self, observation: dict[str, object]) -> dict[str, object]:
            self.observation = observation
            return {"actions": (1000.0 + np.arange(52, dtype=np.float32))[None, :]}

    policy = FakePolicy()
    adapter = AstribotPolicyAdapter(policy, checkpoint_id="test")
    action, timing = adapter.infer_observation(request)

    assert action.shape == (ACTION_DIM,)
    assert timing["infer_ms"] >= 0.0
    assert policy.observation is not None
    assert np.array_equal(policy.observation["observation/joint_position"], expected_state)  # type: ignore[index]
    assert np.array_equal(action[:3], body_values[:3])
    assert np.array_equal(action[6], 1003.0)
    assert np.isfinite(action).all()


def test_stock_gripper_layout_maps_body_and_mm_envelope() -> None:
    request, body_values, _, _ = _request()
    observation = dict(request["observation"])
    observation["gripper_left"] = {
        "valid": True,
        "joint_names": ["astribot_gripper_left"],
        "position": [12.0],
    }
    observation["gripper_right"] = {
        "valid": True,
        "joint_names": ["astribot_gripper_right"],
        "position": [34.0],
    }
    request = {
        **request,
        "observation": observation,
        "enabled_action_mask": enabled_action_mask(
            control_scope="whole_body", body=True, left_hand=False, right_hand=False
        ),
        "enabled_grippers": {"left": True, "right": True},
        "request_sequence": 0,
        "session_id": "gripper-test",
    }
    expected_state = np.concatenate(
        (body_values[:3], body_values[3:7], body_values[9:16], [12.0], body_values[16:23], [34.0], body_values[7:9])
    ).astype(np.float32)
    assert np.array_equal(gripper_model_state_from_wire(request), expected_state)

    class GripperPolicy:
        def infer(self, observation: dict[str, object]) -> dict[str, object]:
            assert np.array_equal(observation["observation/joint_position"], expected_state)
            return {"actions": (np.arange(GRIPPER_MODEL_ACTION_DIM, dtype=np.float32) + 1.0)[None, :]}

    adapter = AstribotPolicyAdapter(GripperPolicy(), checkpoint_id="gripper-test", end_effector="gripper")
    action, _ = adapter.infer_observation(request)
    assert np.array_equal(action[:3], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(action[3:7], np.array([4.0, 5.0, 6.0, 7.0]))
    assert np.array_equal(action[9:16], np.arange(8.0, 15.0))
    assert np.array_equal(action[16:23], np.arange(16.0, 23.0))
    assert np.array_equal(action[7:9], np.array([24.0, 25.0]))
    assert np.array_equal(adapter.last_gripper_action, np.array([15.0, 23.0]))


def test_stock_gripper_mapping_holds_disabled_body_and_rejects_bad_mm() -> None:
    hold = np.arange(ACTION_DIM, dtype=np.float64)
    mask = np.ones(ACTION_DIM, dtype=np.bool_)
    mask[:3] = False
    model = np.arange(GRIPPER_MODEL_ACTION_DIM, dtype=np.float64)
    action, gripper = map_gripper_model_action(model, hold, mask, np.array([10.0, 20.0]))
    assert np.array_equal(action[:3], hold[:3])
    assert np.array_equal(gripper, np.array([14.0, 22.0]))
    model[14] = 101.0
    with pytest.raises(ValueError, match=r"outside \[0, 100\]"):
        map_gripper_model_action(model, hold, mask, np.array([10.0, 20.0]))
