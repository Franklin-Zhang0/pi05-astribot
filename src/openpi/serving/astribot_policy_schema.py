"""Vendored Astribot policy schema used by the Orin policy client.

The joint tables and schema payload intentionally mirror
``avp_teleop_core.policy_schema`` from the Orin deployment workspace.  Keep
this module dependency-free so the policy server can validate and emit the
same wire contract without importing the ROS workspace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

import numpy as np

POLICY_PROTOCOL_VERSION = 1

CHASSIS_JOINTS = (
    "astribot_chassis_x",
    "astribot_chassis_y",
    "astribot_chassis_zrot",
)
TORSO_JOINTS = tuple(f"astribot_torso_joint_{index}" for index in range(1, 5))
NECK_JOINTS = ("astribot_head_joint_1", "astribot_head_joint_2")
ARM_JOINTS = (
    *(f"astribot_arm_left_joint_{index}" for index in range(1, 8)),
    *(f"astribot_arm_right_joint_{index}" for index in range(1, 8)),
)
CANONICAL_BODY_JOINTS = CHASSIS_JOINTS + TORSO_JOINTS + NECK_JOINTS + ARM_JOINTS

HAND_CONTROL_JOINTS = tuple(
    [f"thumb_j{index}" for index in range(4)]
    + [f"{finger}_j{index}" for finger in ("index", "middle", "ring", "pinky") for index in range(3)]
)
HAND_STATE_JOINTS = tuple(
    [f"thumb_j{index}" for index in range(5)]
    + [f"{finger}_j{index}" for finger in ("index", "middle", "ring", "pinky") for index in range(4)]
)
CANONICAL_ACTION_JOINTS = (
    tuple(f"body/{name}" for name in CANONICAL_BODY_JOINTS)
    + tuple(f"hand_left/{name}" for name in HAND_CONTROL_JOINTS)
    + tuple(f"hand_right/{name}" for name in HAND_CONTROL_JOINTS)
)
ACTION_JOINT_NAMES = CANONICAL_ACTION_JOINTS

BODY_ACTION_SLICE = slice(0, len(CANONICAL_BODY_JOINTS))
LEFT_HAND_ACTION_SLICE = slice(
    BODY_ACTION_SLICE.stop,
    BODY_ACTION_SLICE.stop + len(HAND_CONTROL_JOINTS),
)
RIGHT_HAND_ACTION_SLICE = slice(
    LEFT_HAND_ACTION_SLICE.stop,
    LEFT_HAND_ACTION_SLICE.stop + len(HAND_CONTROL_JOINTS),
)
ACTION_DIM = RIGHT_HAND_ACTION_SLICE.stop


def _schema_payload() -> dict[str, object]:
    return {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "body_joint_names": CANONICAL_BODY_JOINTS,
        "hand_control_joint_names": HAND_CONTROL_JOINTS,
        "hand_state_joint_names": HAND_STATE_JOINTS,
        "action_joint_names": CANONICAL_ACTION_JOINTS,
        "action_semantics": "absolute_joint_position",
    }


POLICY_SCHEMA_HASH = hashlib.sha256(
    json.dumps(_schema_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def canonical_vector(
    names: Sequence[str],
    values: Sequence[float],
    expected_names: Sequence[str],
    *,
    field: str,
) -> np.ndarray:
    """Validate and reorder one named finite vector into canonical order."""

    received_names = tuple(str(name) for name in names)
    expected = tuple(str(name) for name in expected_names)
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(received_names) != vector.size:
        raise ValueError(f"{field}: {len(received_names)} names for {vector.size} values")
    if len(set(received_names)) != len(received_names):
        raise ValueError(f"{field}: duplicate joint names")
    if set(received_names) != set(expected):
        missing = sorted(set(expected) - set(received_names))
        unknown = sorted(set(received_names) - set(expected))
        raise ValueError(f"{field}: joint schema mismatch; missing={missing}, unknown={unknown}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field}: values contain NaN or infinity")
    received_index = {name: index for index, name in enumerate(received_names)}
    return np.asarray([vector[received_index[name]] for name in expected], dtype=np.float64)


def optional_canonical_vector(
    names: Sequence[str],
    values: Sequence[float],
    expected_names: Sequence[str],
    *,
    field: str,
) -> np.ndarray:
    """Canonicalize an optional vector, returning zeros when it is absent."""

    if len(values) == 0:
        return np.zeros(len(expected_names), dtype=np.float64)
    return canonical_vector(names, values, expected_names, field=field)


def hand_control_from_state(names: Sequence[str], values: Sequence[float], *, field: str) -> np.ndarray:
    """Select the 16 independent controls from a canonical 21-joint hand state."""

    state = canonical_vector(names, values, HAND_STATE_JOINTS, field=field)
    state_index = {name: index for index, name in enumerate(HAND_STATE_JOINTS)}
    return np.asarray([state[state_index[name]] for name in HAND_CONTROL_JOINTS], dtype=np.float64)


def normalize_control_scope(scope: str) -> str:
    aliases = {
        "upper": "upper_body",
        "lean": "upper_body_with_lean",
        "upper_with_lean": "upper_body_with_lean",
        "chassis": "whole_body",
        "full_body": "whole_body",
    }
    normalized = aliases.get(str(scope).strip().lower(), str(scope).strip().lower())
    if normalized not in {"upper_body", "upper_body_with_lean", "whole_body"}:
        raise ValueError(f"unknown control scope '{scope}'; expected upper_body, upper_body_with_lean, or whole_body")
    return normalized


def joint_enabled(scope: str, joint_name: str) -> bool:
    normalized = normalize_control_scope(scope)
    if joint_name in CHASSIS_JOINTS:
        return normalized == "whole_body"
    if joint_name in TORSO_JOINTS[:3]:
        return normalized != "upper_body"
    return True


def enabled_action_mask(*, control_scope: str, body: bool, left_hand: bool, right_hand: bool) -> np.ndarray:
    """Return the 55-element component and control-scope mask."""

    mask = np.zeros(ACTION_DIM, dtype=np.bool_)
    if body:
        mask[BODY_ACTION_SLICE] = [joint_enabled(control_scope, name) for name in CANONICAL_BODY_JOINTS]
    if left_hand:
        mask[LEFT_HAND_ACTION_SLICE] = True
    if right_hand:
        mask[RIGHT_HAND_ACTION_SLICE] = True
    return mask


def validate_server_metadata(metadata: Mapping[str, object]) -> None:
    """Require the Astribot wrapper metadata sent before OpenPI inference."""

    version = metadata.get("astribot_policy_protocol")
    if version != POLICY_PROTOCOL_VERSION:
        raise ValueError(f"server protocol mismatch: expected {POLICY_PROTOCOL_VERSION}, got {version!r}")
    schema_hash = metadata.get("action_schema_hash")
    if schema_hash != POLICY_SCHEMA_HASH:
        raise ValueError(f"server action schema mismatch: expected {POLICY_SCHEMA_HASH}, got {schema_hash!r}")


def validate_action_response(
    response: Mapping[str, object],
    *,
    request_sequence: int,
    session_id: str,
    expected_mask: np.ndarray,
    min_action_dt_s: float = 0.001,
    max_action_dt_s: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Validate the legacy websocket action-chunk response."""

    if response.get("protocol_version") != POLICY_PROTOCOL_VERSION:
        raise ValueError("action response protocol version mismatch")
    if response.get("schema_hash") != POLICY_SCHEMA_HASH:
        raise ValueError("action response schema hash mismatch")
    if response.get("session_id") != session_id:
        raise ValueError("action response belongs to another policy session")
    if response.get("request_sequence") != request_sequence:
        raise ValueError("late or out-of-order action response")
    if tuple(response.get("body_joint_names", ())) != CANONICAL_BODY_JOINTS:
        raise ValueError("body action joint order mismatch")
    if tuple(response.get("hand_control_joint_names", ())) != HAND_CONTROL_JOINTS:
        raise ValueError("hand action joint order mismatch")

    mask = np.asarray(response.get("enabled_action_mask"), dtype=np.bool_).reshape(-1)
    if mask.shape != (ACTION_DIM,) or not np.array_equal(mask, expected_mask):
        raise ValueError("action response enabled mask mismatch")

    actions = np.asarray(response.get("actions"), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] < 1:
        raise ValueError(f"actions must have shape (horizon, {ACTION_DIM}), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or infinity")
    try:
        action_dt_s = float(response.get("action_dt_s"))
    except (TypeError, ValueError) as exc:
        raise ValueError("action_dt_s must be numeric") from exc
    if not np.isfinite(action_dt_s) or not min_action_dt_s <= action_dt_s <= max_action_dt_s:
        raise ValueError(f"action_dt_s must be in [{min_action_dt_s}, {max_action_dt_s}]")
    horizon = response.get("horizon")
    if horizon is not None and int(horizon) != actions.shape[0]:
        raise ValueError("declared horizon does not match actions")
    return actions, action_dt_s


def validate_single_action_response(
    response: Mapping[str, object], *, session_id: str, expected_mask: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Validate one streamed action frame as the Orin client does."""

    if response.get("type") != "action":
        raise ValueError("single action response type must be 'action'")
    if response.get("protocol_version") != POLICY_PROTOCOL_VERSION:
        raise ValueError("action response protocol version mismatch")
    if response.get("schema_hash") != POLICY_SCHEMA_HASH:
        raise ValueError("action response schema hash mismatch")
    if response.get("session_id") != session_id:
        raise ValueError("action response belongs to another policy session")
    if tuple(response.get("body_joint_names", ())) != CANONICAL_BODY_JOINTS:
        raise ValueError("body action joint order mismatch")
    if tuple(response.get("hand_control_joint_names", ())) != HAND_CONTROL_JOINTS:
        raise ValueError("hand action joint order mismatch")

    mask = np.asarray(response.get("enabled_action_mask"), dtype=np.bool_).reshape(-1)
    if mask.shape != (ACTION_DIM,) or not np.array_equal(mask, expected_mask):
        raise ValueError("action response enabled mask mismatch")

    try:
        action_sequence = int(response.get("action_sequence"))
        source_observation_sequence = int(response.get("source_observation_sequence", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("action sequence fields must be integers") from exc
    if action_sequence < 0 or source_observation_sequence < -1:
        raise ValueError("action sequence fields must be non-negative")

    action = np.asarray(response.get("action"), dtype=np.float64).reshape(-1)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"action must contain exactly {ACTION_DIM} values, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("action contains NaN or infinity")
    return action, action_sequence, source_observation_sequence
