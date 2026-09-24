"""Astribot wire-protocol to light-candle policy-space adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
import time
from typing import Any

import numpy as np
from PIL import Image

from openpi.serving.astribot_policy_schema import ACTION_DIM
from openpi.serving.astribot_policy_schema import BODY_ACTION_SLICE
from openpi.serving.astribot_policy_schema import CANONICAL_BODY_JOINTS
from openpi.serving.astribot_policy_schema import GRIPPER_JOINTS
from openpi.serving.astribot_policy_schema import HAND_CONTROL_JOINTS
from openpi.serving.astribot_policy_schema import LEFT_HAND_ACTION_SLICE
from openpi.serving.astribot_policy_schema import POLICY_PROTOCOL_VERSION
from openpi.serving.astribot_policy_schema import POLICY_SCHEMA_HASH
from openpi.serving.astribot_policy_schema import RIGHT_HAND_ACTION_SLICE
from openpi.serving.astribot_policy_schema import canonical_vector
from openpi.serving.astribot_policy_schema import hand_control_from_state

# The checkpoint was trained in this order, which differs from the Orin body
# order because chassis is omitted and head follows both arms.
MODEL_BODY_JOINTS = (
    *CANONICAL_BODY_JOINTS[3:7],
    *CANONICAL_BODY_JOINTS[9:16],
    *CANONICAL_BODY_JOINTS[16:23],
    *CANONICAL_BODY_JOINTS[7:9],
)
MODEL_JOINT_NAMES = MODEL_BODY_JOINTS + HAND_CONTROL_JOINTS + HAND_CONTROL_JOINTS
MODEL_ACTION_DIM = len(MODEL_JOINT_NAMES)
MODEL_TORSO_SLICE = slice(0, 4)
MODEL_LEFT_ARM_SLICE = slice(4, 11)
MODEL_RIGHT_ARM_SLICE = slice(11, 18)
MODEL_HEAD_SLICE = slice(18, 20)
MODEL_LEFT_HAND_SLICE = slice(20, 36)
MODEL_RIGHT_HAND_SLICE = slice(36, 52)

# Stock-gripper checkpoints use the 25-D dataset order: chassis, torso, left
# arm, left gripper, right arm, right gripper, head. Their prismatic outputs
# are carried in a separate wire envelope.
GRIPPER_MODEL_ACTION_DIM = 25
GRIPPER_MODEL_CHASSIS_SLICE = slice(0, 3)
GRIPPER_MODEL_TORSO_SLICE = slice(3, 7)
GRIPPER_MODEL_LEFT_ARM_SLICE = slice(7, 14)
GRIPPER_MODEL_LEFT_GRIPPER_INDEX = 14
GRIPPER_MODEL_RIGHT_ARM_SLICE = slice(15, 22)
GRIPPER_MODEL_RIGHT_GRIPPER_INDEX = 22
GRIPPER_MODEL_HEAD_SLICE = slice(23, 25)

# The checkpoint predicts every canonical action except the three chassis
# joints.  This mask is separate from the per-observation control mask: a
# whole-body request may enable chassis output even though the model cannot
# produce it.
POLICY_ACTION_MASK = np.zeros(ACTION_DIM, dtype=np.bool_)
POLICY_ACTION_MASK[3 : BODY_ACTION_SLICE.stop] = True
POLICY_ACTION_MASK[LEFT_HAND_ACTION_SLICE] = True
POLICY_ACTION_MASK[RIGHT_HAND_ACTION_SLICE] = True
POLICY_ACTION_MASK.setflags(write=False)


@dataclass(frozen=True)
class _ActionPrediction:
    source_sequence: int
    actions: np.ndarray
    enabled_mask: np.ndarray


class ACTTemporalEnsembler:
    """Online ACT-style temporal ensembling for action chunks.

    At each observation sequence, the row aligned with that sequence is
    selected from every still-valid prediction chunk.  The oldest surviving
    chunk receives ``exp(-decay * 0)`` and newer chunks receive progressively
    smaller weights, matching ACT's online ensembler.
    """

    def __init__(self, action_horizon: int, ensemble_decay: float, *, action_dim: int = ACTION_DIM) -> None:
        if isinstance(action_horizon, bool) or not isinstance(action_horizon, int | np.integer) or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        if not np.isfinite(ensemble_decay):
            raise ValueError("ensemble_decay must be finite")
        self._action_horizon = int(action_horizon)
        if isinstance(action_dim, bool) or not isinstance(action_dim, int | np.integer) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        self._action_dim = int(action_dim)
        self._ensemble_decay = float(ensemble_decay)
        self._history: dict[tuple[str, int], list[_ActionPrediction]] = {}
        self._last_sequences: dict[tuple[str, int], int] = {}

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def ensemble_decay(self) -> float:
        return self._ensemble_decay

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._history.clear()
            self._last_sequences.clear()
            return
        keys = [key for key in self._history if key[0] == session_id]
        for key in keys:
            self._history.pop(key, None)
            self._last_sequences.pop(key, None)

    def update(
        self,
        *,
        session_id: str,
        control_epoch: int,
        sequence: int,
        actions: np.ndarray,
        enabled_mask: np.ndarray,
        hold_action: np.ndarray,
    ) -> np.ndarray:
        """Add one chunk and return the ensembled action at ``sequence``."""

        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self._action_dim or not 1 <= actions.shape[0] <= self._action_horizon:
            raise ValueError(
                f"actions must have shape (1..{self._action_horizon}, {self._action_dim}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("actions contain NaN or infinity")
        enabled_mask = np.asarray(enabled_mask, dtype=np.bool_).reshape(-1)
        if enabled_mask.shape != (self._action_dim,):
            raise ValueError(f"enabled_mask must contain {self._action_dim} values")
        hold_action = np.asarray(hold_action, dtype=np.float64).reshape(-1)
        if hold_action.shape != (self._action_dim,) or not np.isfinite(hold_action).all():
            raise ValueError(f"hold_action must contain {self._action_dim} finite values")

        key = (str(session_id), int(control_epoch))
        history = self._history.get(key, [])
        last_sequence = self._last_sequences.get(key)
        if last_sequence is not None and sequence <= last_sequence:
            # A restarted stream can reuse sequence numbers. Do not mix its
            # chunks with predictions from the previous stream instance.
            history = []

        history = [
            prediction
            for prediction in history
            if 0 <= sequence - prediction.source_sequence < self._action_horizon
        ]
        history.append(
            _ActionPrediction(
                source_sequence=sequence,
                actions=actions.copy(),
                enabled_mask=enabled_mask.copy(),
            )
        )

        values = []
        masks = []
        for prediction in history:
            offset = sequence - prediction.source_sequence
            if 0 <= offset < prediction.actions.shape[0]:
                values.append(prediction.actions[offset])
                masks.append(prediction.enabled_mask)
        if not values:
            raise AssertionError("the newly added action chunk must provide a current action")

        values_array = np.stack(values)
        masks_array = np.stack(masks)
        exponents = -self._ensemble_decay * np.arange(len(values), dtype=np.float64)
        # Subtracting the largest exponent keeps extreme user-supplied decay
        # values finite without changing the normalized weighted average.
        weights = np.exp(exponents - np.max(exponents))
        weighted_masks = weights[:, None] * masks_array
        denominator = weighted_masks.sum(axis=0)
        result = hold_action.copy()
        np.divide(
            (values_array * weighted_masks).sum(axis=0),
            denominator,
            out=result,
            where=denominator > 0.0,
        )
        result[~enabled_mask] = hold_action[~enabled_mask]

        self._history[key] = history
        self._last_sequences[key] = sequence
        return result.astype(np.float32)


def decode_camera_payload(camera: str, image: Mapping[str, object]) -> np.ndarray:
    """Decode one Orin compressed image and apply training camera geometry.

    The stereo crop happens before the model's resize-with-padding transform.
    Wrist camera streams are mounted upside down in the Orin deployment and
    therefore receive the 180-degree rotation recorded by the dataset
    converter.

    The policy client can provide either the normal head RGB-D color stream or
    the side-by-side head stereo stream. Only the latter is cropped.
    """

    if not bool(image.get("valid", False)):
        raise ValueError(f"{camera}: image is not valid")
    payload = image.get("payload")
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise ValueError(f"{camera}: payload must contain encoded image bytes")

    with Image.open(io.BytesIO(bytes(payload))) as source:
        decoded = source.convert("RGB")
    if camera == "head_stereo":
        width, height = decoded.size
        if width % 2:
            raise ValueError(f"{camera}: stereo image width must be even, got {decoded.size}")
        decoded = decoded.crop((0, 0, width // 2, height))
    elif camera == "head_rgbd":
        pass
    elif camera in {"left_wrist_rgbd", "right_wrist_rgbd"}:
        decoded = decoded.transpose(Image.Transpose.ROTATE_180)
    else:
        raise ValueError(f"unsupported Astribot camera {camera!r}")
    return np.asarray(decoded, dtype=np.uint8).copy()


def _observation_payload(request: Mapping[str, object]) -> Mapping[str, object]:
    observation = request.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    return observation


def _component_position(
    component: object,
    expected_names: tuple[str, ...],
    *,
    field: str,
    required: bool,
) -> np.ndarray:
    if not isinstance(component, Mapping) or not bool(component.get("valid", False)):
        if required:
            raise ValueError(f"{field} is missing or invalid")
        return np.zeros(len(expected_names), dtype=np.float64)
    return canonical_vector(
        component.get("joint_names", ()),
        component.get("position", ()),
        expected_names,
        field=f"{field}.position",
    )


def _wire_measurements(request: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation = _observation_payload(request)
    body = _component_position(
        observation.get("body"),
        CANONICAL_BODY_JOINTS,
        field="observation.body",
        required=True,
    )
    left_hand = np.zeros(len(HAND_CONTROL_JOINTS), dtype=np.float64)
    right_hand = np.zeros(len(HAND_CONTROL_JOINTS), dtype=np.float64)
    if isinstance(observation.get("hand_left"), Mapping) and bool(observation["hand_left"].get("valid", False)):
        left_hand = hand_control_from_state(
            observation["hand_left"].get("joint_names", ()),
            observation["hand_left"].get("position", ()),
            field="observation.hand_left.position",
        )
    if isinstance(observation.get("hand_right"), Mapping) and bool(observation["hand_right"].get("valid", False)):
        right_hand = hand_control_from_state(
            observation["hand_right"].get("joint_names", ()),
            observation["hand_right"].get("position", ()),
            field="observation.hand_right.position",
        )
    return body, left_hand, right_hand


def model_state_from_wire(request: Mapping[str, object]) -> np.ndarray:
    """Convert canonical 23+21+21 measured joints to the trained 52-D state."""

    body, left_hand, right_hand = _wire_measurements(request)
    model_body = np.concatenate(
        (
            body[3:7],  # torso
            body[9:16],  # left arm
            body[16:23],  # right arm
            body[7:9],  # head/neck
        )
    )
    state = np.concatenate((model_body, left_hand, right_hand))
    if state.shape != (MODEL_ACTION_DIM,):
        raise AssertionError(f"unexpected model state shape {state.shape}")
    return state.astype(np.float32)


def _wire_gripper_measurements(request: Mapping[str, object]) -> np.ndarray:
    observation = _observation_payload(request)
    values: list[float] = []
    for side, joint_name in zip(("left", "right"), GRIPPER_JOINTS, strict=True):
        component = observation.get(f"gripper_{side}")
        if not isinstance(component, Mapping) or not bool(component.get("valid", False)):
            raise ValueError(f"observation.gripper_{side} is missing or invalid")
        single = canonical_vector(
            component.get("joint_names", ()),
            component.get("position", ()),
            (joint_name,),
            field=f"observation.gripper_{side}.position",
        )
        if not 0.0 <= float(single[0]) <= 100.0:
            raise ValueError(f"observation.gripper_{side}.position must be within [0, 100] mm")
        values.append(float(single[0]))
    return np.asarray(values, dtype=np.float64)


def gripper_model_state_from_wire(request: Mapping[str, object]) -> np.ndarray:
    """Convert canonical measured state to the 25-D stock-gripper layout."""

    body, _, _ = _wire_measurements(request)
    grippers = _wire_gripper_measurements(request)
    state = np.concatenate((body[:3], body[3:7], body[9:16], grippers[:1], body[16:23], grippers[1:], body[7:9]))
    if state.shape != (GRIPPER_MODEL_ACTION_DIM,):
        raise AssertionError(f"unexpected gripper model state shape {state.shape}")
    return state.astype(np.float32)


def hold_action_from_wire(request: Mapping[str, object]) -> np.ndarray:
    """Build a full canonical action from measured state only."""

    body, left_hand, right_hand = _wire_measurements(request)
    action = np.zeros(ACTION_DIM, dtype=np.float64)
    action[BODY_ACTION_SLICE] = body
    action[LEFT_HAND_ACTION_SLICE] = left_hand
    action[RIGHT_HAND_ACTION_SLICE] = right_hand
    return action


def _enabled_action_mask(request: Mapping[str, object]) -> np.ndarray:
    mask = np.asarray(request.get("enabled_action_mask"), dtype=np.bool_).reshape(-1)
    if mask.shape != (ACTION_DIM,):
        raise ValueError(f"enabled_action_mask must contain {ACTION_DIM} entries")
    return mask


def map_model_action_to_canonical(
    model_action: np.ndarray,
    hold_action: np.ndarray,
    enabled_mask: np.ndarray,
) -> np.ndarray:
    """Map one 52-D model action to a held 55-D canonical absolute action."""

    model_action = np.asarray(model_action, dtype=np.float64).reshape(-1)
    if model_action.shape != (MODEL_ACTION_DIM,):
        raise ValueError(f"model action must contain {MODEL_ACTION_DIM} values, got {model_action.shape}")
    hold_action = np.asarray(hold_action, dtype=np.float64).reshape(-1)
    if hold_action.shape != (ACTION_DIM,):
        raise ValueError(f"hold action must contain {ACTION_DIM} values, got {hold_action.shape}")
    enabled_mask = np.asarray(enabled_mask, dtype=np.bool_).reshape(-1)
    if enabled_mask.shape != (ACTION_DIM,):
        raise ValueError(f"enabled mask must contain {ACTION_DIM} values, got {enabled_mask.shape}")
    if not np.isfinite(model_action).all() or not np.isfinite(hold_action).all():
        raise ValueError("action values contain NaN or infinity")

    action = hold_action.copy()
    action[3:7] = model_action[MODEL_TORSO_SLICE]
    action[9:16] = model_action[MODEL_LEFT_ARM_SLICE]
    action[16:23] = model_action[MODEL_RIGHT_ARM_SLICE]
    action[7:9] = model_action[MODEL_HEAD_SLICE]
    action[LEFT_HAND_ACTION_SLICE] = model_action[MODEL_LEFT_HAND_SLICE]
    action[RIGHT_HAND_ACTION_SLICE] = model_action[MODEL_RIGHT_HAND_SLICE]

    # Chassis is not predicted by this checkpoint.  Scope-disabled joints and
    # every other non-policy component remain at measured proprioception.
    action[~enabled_mask] = hold_action[~enabled_mask]
    return action


def map_gripper_model_action(
    model_action: np.ndarray,
    hold_action: np.ndarray,
    enabled_mask: np.ndarray,
    hold_gripper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map one 25-D stock-gripper action to body and millimetre outputs."""

    model_action = np.asarray(model_action, dtype=np.float64).reshape(-1)
    if model_action.shape != (GRIPPER_MODEL_ACTION_DIM,):
        raise ValueError(
            f"gripper model action must contain {GRIPPER_MODEL_ACTION_DIM} values, got {model_action.shape}"
        )
    hold_action = np.asarray(hold_action, dtype=np.float64).reshape(-1)
    enabled_mask = np.asarray(enabled_mask, dtype=np.bool_).reshape(-1)
    hold_gripper = np.asarray(hold_gripper, dtype=np.float64).reshape(-1)
    if hold_action.shape != (ACTION_DIM,) or enabled_mask.shape != (ACTION_DIM,):
        raise ValueError("invalid canonical hold action or enabled mask")
    if hold_gripper.shape != (2,) or not np.isfinite(hold_gripper).all():
        raise ValueError("hold gripper action must contain two finite values")
    if not np.isfinite(model_action).all():
        raise ValueError("gripper model action contains NaN or infinity")

    action = hold_action.copy()
    action[0:3] = model_action[GRIPPER_MODEL_CHASSIS_SLICE]
    action[3:7] = model_action[GRIPPER_MODEL_TORSO_SLICE]
    action[9:16] = model_action[GRIPPER_MODEL_LEFT_ARM_SLICE]
    action[16:23] = model_action[GRIPPER_MODEL_RIGHT_ARM_SLICE]
    action[7:9] = model_action[GRIPPER_MODEL_HEAD_SLICE]
    action[~enabled_mask] = hold_action[~enabled_mask]
    gripper = np.asarray(
        [model_action[GRIPPER_MODEL_LEFT_GRIPPER_INDEX], model_action[GRIPPER_MODEL_RIGHT_GRIPPER_INDEX]],
        dtype=np.float64,
    )
    if np.any(gripper < 0.0) or np.any(gripper > 100.0):
        raise ValueError(f"gripper model action is outside [0, 100] mm: {gripper}")
    return action, gripper


class AstribotPolicyAdapter:
    """Adapt the documented Orin envelope to an OpenPI Astribot policy."""

    def __init__(
        self,
        policy: Any,
        *,
        checkpoint_id: str,
        default_prompt: str | None = None,
        action_horizon: int | None = None,
        ensemble_decay: float = 0.01,
        end_effector: str = "apexhand",
    ) -> None:
        if action_horizon is not None:
            if (
                isinstance(action_horizon, bool)
                or not isinstance(action_horizon, int | np.integer)
                or action_horizon < 1
            ):
                raise ValueError("action_horizon must be a positive integer")
            action_horizon = int(action_horizon)
        if not np.isfinite(ensemble_decay):
            raise ValueError("ensemble_decay must be finite")
        self._policy = policy
        self._end_effector = str(end_effector).strip().lower()
        if self._end_effector not in {"apexhand", "gripper"}:
            raise ValueError("end_effector must be 'apexhand' or 'gripper'")
        self._default_prompt = default_prompt
        self._configured_action_horizon = action_horizon
        self._ensemble_decay = float(ensemble_decay)
        self._ensembler = (
            ACTTemporalEnsembler(action_horizon, self._ensemble_decay) if action_horizon is not None else None
        )
        self._gripper_ensembler: ACTTemporalEnsembler | None = (
            ACTTemporalEnsembler(action_horizon, self._ensemble_decay, action_dim=2)
            if self._end_effector == "gripper" and action_horizon is not None
            else None
        )
        self._last_gripper_action: np.ndarray | None = None
        policy_metadata = getattr(policy, "metadata", {}) or {}
        self._metadata = {
            **dict(policy_metadata),
            "astribot_policy_protocol": POLICY_PROTOCOL_VERSION,
            "action_schema_hash": POLICY_SCHEMA_HASH,
            "checkpoint_id": checkpoint_id,
            "transport": "zmq",
            "action_semantics": "absolute_joint_position_single_frame",
            "action_ensembling": "act_temporal",
            "action_horizon": action_horizon if action_horizon is not None else "model_native",
            "ensemble_decay": self._ensemble_decay,
            "end_effector": self._end_effector,
        }

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    @property
    def last_gripper_action(self) -> np.ndarray | None:
        return None if self._last_gripper_action is None else self._last_gripper_action.copy()

    def infer_observation(self, request: Mapping[str, object]) -> tuple[np.ndarray, dict[str, float]]:
        """Run one inference and return an ensembled canonical action plus timing."""

        started = time.perf_counter()
        hold_action = hold_action_from_wire(request)
        hold_gripper = _wire_gripper_measurements(request) if self._end_effector == "gripper" else None
        policy_observation = self._policy_observation(request)
        policy_output = self._policy.infer(policy_observation)
        model_actions = np.asarray(policy_output.get("actions"), dtype=np.float64)
        if model_actions.ndim == 1:
            model_actions = model_actions[None, :]
        elif model_actions.ndim != 2:
            raise ValueError(f"policy actions must be 1-D or 2-D, got {model_actions.shape}")
        expected_model_dim = GRIPPER_MODEL_ACTION_DIM if self._end_effector == "gripper" else MODEL_ACTION_DIM
        if model_actions.shape[1] != expected_model_dim:
            raise ValueError(f"policy actions must contain {expected_model_dim} values per row, got {model_actions.shape}")
        if model_actions.shape[0] == 0:
            raise ValueError("policy returned an empty action chunk")
        if self._configured_action_horizon is not None:
            if model_actions.shape[0] < self._configured_action_horizon:
                raise ValueError(
                    "configured action_horizon "
                    f"{self._configured_action_horizon} exceeds model action horizon {model_actions.shape[0]}"
                )
            model_actions = model_actions[: self._configured_action_horizon]

        if self._ensembler is None:
            self._ensembler = ACTTemporalEnsembler(model_actions.shape[0], self._ensemble_decay)
        elif model_actions.shape[0] != self._ensembler.action_horizon:
            raise ValueError(
                f"policy action horizon changed from {self._ensembler.action_horizon} to {model_actions.shape[0]}"
            )

        enabled_mask = _enabled_action_mask(request)
        if self._end_effector == "gripper":
            assert hold_gripper is not None
            predicted_pairs = [
                map_gripper_model_action(row, hold_action, enabled_mask, hold_gripper) for row in model_actions
            ]
            predicted_actions = np.stack([pair[0] for pair in predicted_pairs])
            predicted_grippers = np.stack([pair[1] for pair in predicted_pairs])
        else:
            predicted_actions = np.stack(
                [map_model_action_to_canonical(row, hold_action, POLICY_ACTION_MASK) for row in model_actions]
            )
            predicted_grippers = None
        try:
            sequence = int(request.get("request_sequence", 0))
            control_epoch = int(request.get("control_epoch", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("request_sequence and control_epoch must be integers") from exc
        action = self._ensembler.update(
            session_id=str(request.get("session_id", "default")),
            control_epoch=control_epoch,
            sequence=sequence,
            actions=predicted_actions,
            enabled_mask=enabled_mask & (np.ones(ACTION_DIM, dtype=np.bool_) if self._end_effector == "gripper" else POLICY_ACTION_MASK),
            hold_action=hold_action,
        )
        if predicted_grippers is not None:
            if self._gripper_ensembler is None:
                self._gripper_ensembler = ACTTemporalEnsembler(predicted_grippers.shape[0], self._ensemble_decay, action_dim=2)
            elif predicted_grippers.shape[0] != self._gripper_ensembler.action_horizon:
                raise ValueError("policy gripper action horizon changed")
            enabled_grippers = request.get("enabled_grippers", {})
            if not isinstance(enabled_grippers, Mapping):
                raise ValueError("enabled_grippers must be a mapping")
            self._last_gripper_action = self._gripper_ensembler.update(
                session_id=str(request.get("session_id", "default")),
                control_epoch=control_epoch,
                sequence=sequence,
                actions=predicted_grippers,
                enabled_mask=np.asarray(
                    [bool(enabled_grippers.get(side, False)) for side in ("left", "right")],
                    dtype=np.bool_,
                ),
                hold_action=hold_gripper,
            )
        else:
            self._last_gripper_action = None
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timing = {"infer_ms": elapsed_ms}
        policy_timing = policy_output.get("policy_timing", {})
        if isinstance(policy_timing, Mapping) and policy_timing.get("infer_ms") is not None:
            timing["policy_infer_ms"] = float(policy_timing["infer_ms"])
        return action, timing

    def _policy_observation(self, request: Mapping[str, object]) -> dict[str, object]:
        observation = _observation_payload(request)
        images = observation.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("observation.images must be a mapping")
        head_camera = None
        for camera in ("head_stereo", "head_rgbd"):
            image = images.get(camera)
            if isinstance(image, Mapping) and bool(image.get("valid", False)):
                head_camera = camera
                break
        if head_camera is None:
            raise ValueError("observation.images requires a valid head_stereo or head_rgbd image")
        head = images[head_camera]

        policy_observation: dict[str, object] = {
            "observation/head_rgbd": decode_camera_payload(head_camera, head),
            "observation/joint_position": (
                gripper_model_state_from_wire(request)
                if self._end_effector == "gripper"
                else model_state_from_wire(request)
            ),
        }
        for camera, key in (
            ("left_wrist_rgbd", "observation/left_wrist_rgbd"),
            ("right_wrist_rgbd", "observation/right_wrist_rgbd"),
        ):
            image = images.get(camera)
            if isinstance(image, Mapping) and bool(image.get("valid", False)):
                policy_observation[key] = decode_camera_payload(camera, image)

        prompt = request.get("prompt", self._default_prompt)
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        if prompt:
            policy_observation["prompt"] = str(prompt)
        return policy_observation
