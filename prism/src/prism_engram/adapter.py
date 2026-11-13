"""Physical Engram observations and targets for the trained hybrid PRISM policy.

The model predicts two normalized tensors; these helpers keep their distinct
geometry, units, availability and target clocks explicit. No training dataset
or private package is needed to prepare an observation or decode a prediction.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

HORIZON = 32
PACKED_WIDTH = 75
TOKEN_WIDTH = 15
ROUTE_NAMES = (
    "dex_w1_gripper",
    "dex_w1_linker_l6",
    "dex_w1_linker_l20",
    "galbot_inspire_rh56dfx_hand",
)
W1_JOINT_NAMES = (
    "ANKLE",
    "KNEE",
    "BUTTOCK",
    "WAIST",
    "NECK1",
    "NECK2",
    *(f"LEFT_J{i}" for i in range(1, 8)),
    *(f"RIGHT_J{i}" for i in range(1, 8)),
)
GALBOT_JOINT_NAMES = (
    *(f"right_arm_joint{i}" for i in range(1, 8)),
    *(f"left_arm_joint{i}" for i in range(1, 8)),
)
CONDITIONING_FIELDS = (
    "task instruction",
    "subtask instruction",
    "end effector type",
    "end effector number",
    "embodiment",
    "setting",
    "speed",
    "motion quality",
)
_CATEGORIES = {
    "end effector type": {"hand", "gripper"},
    "end effector number": {"single", "dual", "none"},
    "embodiment": {"human", "robot"},
    "speed": {"fast", "moderate", "slow"},
    "motion quality": {"good", "normal", "bad"},
}
_IMAGE_KEYS = {
    "image": "observation/head_rgbd",
    "left_wrist_image": "observation/left_wrist_rgbd",
    "right_wrist_image": "observation/right_wrist_rgbd",
}


@dataclass(frozen=True)
class Route:
    name: str
    robot_type: str
    end_effector_type: str
    joint_dim: int
    hand_dim: int
    hand_units: str


ROUTES = (
    Route(ROUTE_NAMES[0], "dex_w1", "gripper", 20, 2, "percent"),
    Route(ROUTE_NAMES[1], "dex_w1", "linker_l6", 20, 12, "percent"),
    Route(ROUTE_NAMES[2], "dex_w1", "linker_l20", 20, 32, "signed_percent"),
    Route(ROUTE_NAMES[3], "galbot", "inspire_rh56dfx_hand", 14, 12, "unavailable"),
)


@dataclass(frozen=True)
class PreparedObservation:
    """One complete current measurement; all anchors belong to this same tick."""

    route_id: int
    route_name: str
    robot_type: str
    end_effector_type: str
    prompt_text: str
    prompt_format: str
    frames: dict[str, np.ndarray]
    camera_mask: np.ndarray
    normalized_state: np.ndarray
    state_mask: np.ndarray
    native_action_mask: np.ndarray
    robot_eef_mask: np.ndarray
    joint_anchor: np.ndarray
    hand_anchor: np.ndarray | None
    eef_anchor: np.ndarray | None


def _finite_array(value: Any, shape: tuple[int, ...], name: str, *, dtype=np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} requires {shape} finite values")
    return array.copy()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate prompt field {key!r}")
        result[key] = value
    return result


def _canonical_fields(value: Mapping[str, Any]) -> dict[str, str]:
    aliases = {name.replace(" ", "_"): name for name in CONDITIONING_FIELDS}
    result: dict[str, str] = {}
    seen: set[str] = set()
    for name, raw in value.items():
        if not isinstance(name, str):
            raise ValueError("Prompt keys must be strings")
        key = aliases.get(name, name)
        if key not in CONDITIONING_FIELDS or key in seen:
            raise ValueError(f"Unknown or duplicate canonical prompt field {name!r}")
        seen.add(key)
        if raw is None:
            if key == "subtask instruction":
                result[key] = ""
            continue
        if not isinstance(raw, str):
            raise ValueError(f"Prompt field {name!r} must be a string or null")
        text = raw.strip()
        if not text:
            # Training preserves an explicitly empty subtask through formatting.
            if key == "subtask instruction":
                result[key] = ""
            continue
        if key in _CATEGORIES and text not in _CATEGORIES[key]:
            raise ValueError(f"Unsupported canonical {key}: {text!r}")
        result[key] = text
    return result


def _request_prompt(request: Mapping[str, Any]) -> tuple[Route, int, str, str, bool | None]:
    prompt = request.get("prompt", request.get("instruction"))
    requested_format = request.get("prompt_format")
    if requested_format not in (None, "plain", "json"):
        raise ValueError("prompt_format must be plain or json")
    if isinstance(prompt, str) and requested_format != "plain" and prompt.lstrip().startswith("{"):
        try:
            prompt = json.loads(prompt, object_pairs_hook=_unique_object)
        except (TypeError, ValueError) as error:
            raise ValueError("Malformed JSON prompt; use prompt_format=plain for literal task text") from error
    fields: dict[str, str] = {}
    explicit_keys: set[str] = set()
    robot, effector = request.get("robot_type"), request.get("end_effector_type")
    available = request.get("end_effector_state_available")
    if available is not None and type(available) is not bool:
        raise ValueError("end_effector_state_available must be a boolean")
    if isinstance(prompt, Mapping):
        if requested_format == "plain":
            raise ValueError("A mapping prompt requires JSON formatting")
        if "instruction" in prompt:
            allowed = {"instruction", "robot_type", "end_effector_type", "end_effector_state_available"}
            if set(prompt) - allowed:
                raise ValueError("Unknown convenience prompt fields")
            for key, existing in (("robot_type", robot), ("end_effector_type", effector)):
                if key in prompt and existing is not None and prompt[key] != existing:
                    raise ValueError(f"Conflicting {key} in prompt and request")
            robot, effector = prompt.get("robot_type", robot), prompt.get("end_effector_type", effector)
            if "end_effector_state_available" in prompt:
                value = prompt["end_effector_state_available"]
                if type(value) is not bool or (available is not None and available != value):
                    raise ValueError("Conflicting or invalid end-effector availability")
                available = value
            fields = _canonical_fields({"task instruction": prompt["instruction"]})
        else:
            fields = _canonical_fields(prompt)
            aliases = {name.replace(" ", "_"): name for name in CONDITIONING_FIELDS}
            explicit_keys = {aliases.get(name, name) for name in prompt}
        setting = fields.get("setting")
        if setting and setting.startswith("engram_"):
            candidates = [route for route in ROUTES if setting == f"engram_{route.name}"]
            if not candidates:
                raise ValueError("Unknown Engram setting")
            inferred = candidates[0]
            if (robot is not None and robot != inferred.robot_type) or (
                effector is not None and effector != inferred.end_effector_type
            ):
                raise ValueError("Prompt setting conflicts with numeric robot/end-effector route")
            robot, effector = inferred.robot_type, inferred.end_effector_type
        text = fields.get("task instruction")
        format_name = "json"
    elif isinstance(prompt, str):
        text = prompt
        format_name = requested_format or "plain"
    else:
        raise ValueError("Supply a plain task instruction or a canonical JSON prompt")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A nonempty task instruction is required")
    if "prompt" in request and "instruction" in request and request["instruction"] != text:
        raise ValueError("Conflicting instruction and prompt")
    matches = [
        (index, route)
        for index, route in enumerate(ROUTES)
        if (route.robot_type, route.end_effector_type) == (robot, effector)
    ]
    if not matches:
        raise ValueError("Supply a supported explicit robot_type/end_effector_type route")
    index, route = matches[0]
    if format_name == "json":
        setup = {
            "end effector type": "gripper" if index == 0 else "hand",
            "end effector number": "dual",
            "embodiment": "robot",
            "setting": f"engram_{route.name}",
        }
        for key, expected in setup.items():
            if key in fields and fields[key] != expected:
                raise ValueError(f"Prompt {key} conflicts with the trained route")
        # Explicit unknown/empty fields remove source defaults just as the
        # training formatter does; numeric routing remains independent.
        setup = {key: value for key, value in setup.items() if key not in explicit_keys}
        payload = {"task instruction": text.strip(), **setup, **fields}
        prompt_text = json.dumps(
            {key: payload[key] for key in CONDITIONING_FIELDS if key in payload},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    else:
        prompt_text = text
    return route, index, prompt_text, format_name, available


def _proxy_image(value: Any, *, robot: str, role: str, layout: str = "left") -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"{role} camera requires nonempty HWC uint8 RGB")
    target = (
        ((640, 360) if robot == "dex_w1" else (480, 360))
        if role == "head"
        else ((640, 480) if robot == "dex_w1" else (640, 360))
    )
    if role == "head":
        if layout not in {"left", "stereo"} or (layout == "stereo" and robot != "dex_w1"):
            raise ValueError("Only W1 supports head_camera_layout=stereo; otherwise use left")
        if layout == "stereo":
            if image.shape[1] % 2:
                raise ValueError("Side-by-side stereo image width must be even")
            image = image[:, : image.shape[1] // 2]
    width, height = target
    if image.shape[1] * height != image.shape[0] * width:
        raise ValueError(f"{role} camera aspect ratio differs from training ({width}:{height})")
    if image.shape[:2] != (height, width):
        import cv2

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Normalize a measured WXYZ quaternion and return its rotation matrix."""
    q = np.asarray(quaternion, np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all() or np.any(norm <= 1e-8):
        raise ValueError("Measured EEF quaternions require finite nonzero WXYZ values")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotation_6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    """Gram-Schmidt the first two matrix rows, matching pretraining's 6D convention."""
    values = np.asarray(rotation, np.float64)
    if values.shape[-1:] != (6,) or not np.isfinite(values).all():
        raise ValueError("Rotation predictions require six finite row-major values")
    first, second = values[..., :3], values[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Degenerate predicted 6D rotation first row")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Degenerate predicted 6D rotation second row")
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


class EngramAdapter:
    """Normalize current measurements and decode physical 32-tick hybrid targets."""

    def __init__(self, normalization: Mapping[str, Any]):
        if normalization.get("layout") != "engram_delta_joints_delta_hands_packed75_v2":
            raise ValueError("Unsupported Engram native action layout")
        if normalization.get("target_horizon") != HORIZON or normalization.get("eligibility_horizon") != 64:
            raise ValueError("The published Engram model requires 32 target/64 eligibility ticks")
        native = normalization.get("route_statistics", {})
        if (
            native.get("format") != "engram-route-delta-hand-stats-v2"
            or native.get("model_action_dim") != 64
            or native.get("route_names") != list(ROUTE_NAMES)
        ):
            raise ValueError("Invalid route-owned native statistics")
        self.ranges: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.supervised: dict[str, np.ndarray] = {}
        for route in ROUTES:
            entry = native["routes"][route.name]
            supervised = np.asarray(entry["supervised_dims"], bool)
            expected = np.zeros(64, bool)
            expected[: route.joint_dim] = True
            if route.hand_units != "unavailable":
                expected[20 : 20 + route.hand_dim] = True
            if supervised.shape != (64,) or not np.array_equal(supervised, expected):
                raise ValueError("Native statistics availability differs from the trained route")
            self.supervised[route.name] = supervised
            for kind in ("state", "actions"):
                raw = entry["norm_stats"][kind]
                low = _finite_array(raw["q01"], (64,), "q01")
                high = _finite_array(raw["q99"], (64,), "q99")
                minimum = np.zeros(64, np.float32)
                minimum[: route.joint_dim] = 0.1
                minimum[20 : 20 + route.hand_dim] = 5
                if np.any(high <= low) or np.any((high - low)[supervised] < minimum[supervised] - 1e-6):
                    raise ValueError("Normalization span is below its physical minimum")
                self.ranges[route.name, kind] = low, high
        eef = normalization.get("robot_eef_normalization", {})
        if eef.get("format") != "prism_engram_observed_eef_normalization_v1" or eef.get("horizon") != HORIZON:
            raise ValueError("Missing or incompatible observed EEF normalization")
        self.eef_frame = dict(eef["frame_contract"])
        if self.eef_frame.get("position_unit") != "meters" or self.eef_frame.get("frame") not in {
            "inferred_native_torso_arm_base",
            "common_robot_base",
            "per_arm_robot_base",
        }:
            raise ValueError("EEF frame or metric units differ from the trained contract")
        self.eef_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for route in ROUTES[:3]:
            entry = eef["routes"][route.name]
            if entry.get("mode") != "mean_std_v1":
                raise ValueError("EEF normalization must use TRAIN mean/std statistics")
            mean = _finite_array(entry["mean"], (5, 15), "EEF mean")
            std = _finite_array(entry["std"], (5, 15), "EEF std")
            if np.any(std <= 0):
                raise ValueError("EEF standard deviations must be positive")
            self.eef_stats[route.name] = mean, std
        if eef["routes"].get(ROUTE_NAMES[3]) is not None:
            raise ValueError("Galbot has no recorded EEF supervision")

    def prepare(self, request: Mapping[str, Any]) -> PreparedObservation:
        """Validate one measured observation; missing hand/wrist channels stay masked."""
        if not isinstance(request, Mapping):
            raise ValueError("Observation must be a mapping")
        if any(key in request for key in ("action_mask", "state_mask", "engram_state_mask", "engram_route_ids")):
            raise ValueError("Masks and numeric embodiment routes are derived internally")
        route, index, prompt, prompt_format, declared = _request_prompt(request)
        if request.get("joint_units", "radians") != "radians":
            raise ValueError("Measured joint positions must use radians")
        if request.get("hand_units", route.hand_units) != route.hand_units:
            raise ValueError("Measured hand units differ from the recorded route")
        joints = _finite_array(
            request.get("observation/joint_position"), (route.joint_dim,), "Measured joint positions"
        )
        hands_raw = request.get("observation/hand_position")
        if declared is False and hands_raw is not None:
            raise ValueError("Hand feedback supplied while declared unavailable; omit placeholder values")
        if declared is True and hands_raw is None:
            raise ValueError("Hand state declared available without measured feedback")
        hands = None
        if hands_raw is not None:
            if route.hand_units == "unavailable":
                raise ValueError("Galbot hand channels have no training supervision")
            hands = _finite_array(hands_raw, (route.hand_dim,), "Measured hand feedback")
            minimum = -100 if route.hand_units == "signed_percent" else 0
            if np.any((hands < minimum) | (hands > 100)):
                raise ValueError("Measured hand feedback is outside its recorded percentage units")
        state = np.zeros(75, np.float32)
        state_mask = np.zeros(75, bool)
        state[: route.joint_dim], state_mask[: route.joint_dim] = joints, True
        if hands is not None:
            state[20 : 20 + route.hand_dim], state_mask[20 : 20 + route.hand_dim] = hands, True
        low, high = self.ranges[route.name, "state"]
        normalized = np.zeros(75, np.float32)
        normalized[:64] = np.where(state_mask[:64], 2 * (state[:64] - low) / (high - low) - 1, 0)
        native_mask = np.broadcast_to(state_mask, (HORIZON, 75)).copy()
        eef_mask = np.zeros((HORIZON, 5, 15), bool)
        anchor = None
        if route.robot_type == "dex_w1":
            if request.get("eef_frame", self.eef_frame["frame"]) != self.eef_frame["frame"]:
                raise ValueError("Current EEF feedback frame differs from the saved training frame")
            if request.get("eef_position_units", "meters") != "meters":
                raise ValueError("Current EEF positions must use meters")
            raw = request.get("observation/ee_poses")
            if raw is None:
                raise ValueError("W1 EEF inference requires current observation/ee_poses (left/right XYZ+QWXYZ)")
            anchor = np.asarray(raw, np.float64)
            if anchor.shape == (14,):
                anchor = anchor.reshape(2, 7)
            anchor = _finite_array(anchor, (2, 7), "Current left/right EEF poses", dtype=np.float64)
            quaternion_wxyz_to_matrix(anchor[:, 3:])
            anchor[:, 3:] /= np.linalg.norm(anchor[:, 3:], axis=-1, keepdims=True)
            eef_mask[:, [1, 3], :9] = True
            native_mask[:, 6:20] = False
        elif request.get("observation/ee_poses") is not None:
            raise ValueError("Galbot EEF poses are not used by this checkpoint")
        frames: dict[str, np.ndarray] = {}
        camera_mask = np.zeros(3, bool)
        for camera, (key, input_key) in enumerate(_IMAGE_KEYS.items()):
            value = request.get(input_key)
            if value is None:
                if camera == 0:
                    raise ValueError("The head RGB camera is required")
                continue
            frames[key] = _proxy_image(
                value,
                robot=route.robot_type,
                role="head" if camera == 0 else "wrist",
                layout=request.get("head_camera_layout", "left"),
            )
            camera_mask[camera] = True
        return PreparedObservation(
            index,
            route.name,
            route.robot_type,
            route.end_effector_type,
            prompt,
            prompt_format,
            frames,
            camera_mask,
            normalized,
            state_mask,
            native_mask.reshape(HORIZON * 5, 15),
            eef_mask.reshape(HORIZON * 5, 15),
            joints,
            hands,
            anchor,
        )

    @staticmethod
    def _prediction(value: Any, mask: np.ndarray, name: str) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value, np.float32)
        if array.shape == (1, HORIZON * 5, 15):
            array = array[0]
        if array.shape != (HORIZON * 5, 15):
            raise ValueError(f"{name} must have shape [160,15] or [1,160,15]")
        if not np.isfinite(array[mask]).all():
            raise ValueError(f"Available {name} predictions are nonfinite")
        return np.where(mask, array, 0)

    def decode(self, prediction: Mapping[str, Any], observation: PreparedObservation) -> dict[str, Any]:
        """Return physical absolute targets, never packed padding or fabricated arm joints."""
        if not isinstance(prediction, Mapping) or set(prediction) != {"native_joint", "robot_eef"}:
            raise ValueError("Hybrid prediction requires exactly native_joint and robot_eef tensors")
        route = ROUTES[observation.route_id]
        native = self._prediction(prediction["native_joint"], observation.native_action_mask, "native_joint").reshape(
            HORIZON, 75
        )
        low, high = self.ranges[route.name, "actions"]
        physical = (native[:, :64] + 1) * 0.5 * (high - low) + low
        output: dict[str, Any] = {
            "route": route.name,
            "action_layout": "engram_hybrid_absolute_targets_v1",
            "action_horizon": HORIZON,
            "observation_hz": 10.0,
            "action_hz": 30.0,
            "end_effector_available": observation.hand_anchor is not None,
            "joint_hand_tick_offsets": np.arange(HORIZON, dtype=np.int32),
        }
        if route.robot_type == "galbot":
            output.update(
                joint_actions=(physical[:, :14] + observation.joint_anchor).astype(np.float32),
                joint_names=list(GALBOT_JOINT_NAMES),
                joint_units="radians",
            )
        else:
            output.update(
                body_head_actions=(physical[:, :6] + observation.joint_anchor[:6]).astype(np.float32),
                body_head_joint_names=list(W1_JOINT_NAMES[:6]),
                body_head_units="radians",
            )
        if observation.hand_anchor is not None:
            output.update(
                end_effector_actions=(physical[:, 20 : 20 + route.hand_dim] + observation.hand_anchor).astype(
                    np.float32
                ),
                end_effector_units=route.hand_units,
                end_effector_order="left_then_right",
            )
        eef = self._prediction(prediction["robot_eef"], observation.robot_eef_mask, "robot_eef")
        if route.robot_type == "dex_w1":
            mean, std = self.eef_stats[route.name]
            grouped = eef.reshape(HORIZON, 5, 15) * std + mean
            poses = grouped[:, [1, 3], :9]
            delta_rotation = rotation_6d_to_matrix(poses[..., 3:9])
            anchor_rotation = quaternion_wxyz_to_matrix(observation.eef_anchor[:, 3:])
            rotation = delta_rotation @ anchor_rotation
            position = poses[..., :3] + observation.eef_anchor[None, :, :3]
            absolute6d = rotation[..., :2, :].reshape(HORIZON, 2, 6)
            output.update(
                eef_actions=np.concatenate((position, absolute6d), axis=-1).astype(np.float32),
                eef_position=position.astype(np.float32),
                eef_rotation_matrix=rotation.astype(np.float32),
                eef_rotation_6d=absolute6d.astype(np.float32),
                eef_order="left_then_right",
                eef_position_units="meters",
                eef_frame=self.eef_frame["frame"],
                eef_rotation_convention="absolute_first_two_matrix_rows",
                eef_tick_offsets=np.arange(1, HORIZON + 1, dtype=np.int32),
            )
        return output
