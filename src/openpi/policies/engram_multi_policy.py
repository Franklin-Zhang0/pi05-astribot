"""Typed Engram preprocessing shared by mixed training and inference."""

from __future__ import annotations

import dataclasses
import json

import cv2
import numpy as np

from openpi import transforms
from openpi.shared.engram_multi import ACTION_DIM
from openpi.shared.engram_multi import EMBODIMENTS
from openpi.shared.engram_multi import HAND_START
from openpi.shared.engram_multi import ROUTE_NAMES as ROUTE_NAMES  # noqa: PLC0414
from openpi.shared.engram_multi import RouteStats
from openpi.shared.engram_multi import physical_mask


def camera_size(robot_type: str, role: str) -> tuple[int, int]:
    """Return aspect-preserving proxy dimensions as width, height."""
    if robot_type not in {"dex_w1", "galbot"} or role not in {"head", "left_wrist", "right_wrist"}:
        raise ValueError(f"Unsupported camera geometry {robot_type}/{role}")
    if role == "head":
        return (640, 360) if robot_type == "dex_w1" else (480, 360)
    return (640, 480) if robot_type == "dex_w1" else (640, 360)


def prepare_camera(image, robot_type: str, role: str, *, stereo: bool = False) -> np.ndarray:
    """Crop a declared stereo head and area-resize without aspect distortion."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Expected HWC RGB image")
    if image.dtype != np.uint8:
        if (
            not np.issubdtype(image.dtype, np.floating)
            or not np.isfinite(image).all()
            or np.any((image < 0) | (image > 1))
        ):
            raise ValueError("RGB must be uint8 or finite floating [0, 1]")
        image = (image * 255).astype(np.uint8)
    if stereo:
        if robot_type != "dex_w1" or role != "head" or image.shape[1] % 2:
            raise ValueError("Stereo layout is only supported for the W1 head camera")
        image = image[:, : image.shape[1] // 2]
    width, height = camera_size(robot_type, role)
    if abs(image.shape[1] / image.shape[0] - width / height) > 0.005:
        raise ValueError(f"Unexpected {robot_type}/{role} aspect ratio: {image.shape}")
    if image.shape[:2] == (height, width):
        return image.copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def format_prompt(
    instruction: str,
    robot_type: str,
    end_effector_type: str,
    available: bool,  # noqa: FBT001
    *,
    structured: bool,
) -> str:
    """Keep exact instruction text in either the JSON or plain prompt form."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction string is required")
    if structured:
        return json.dumps(
            {
                "instruction": instruction,
                "robot_type": robot_type,
                "end_effector_type": end_effector_type,
                "end_effector_state_available": bool(available),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    prefix = "" if available else "End-effector state unavailable. "
    return prefix + instruction


def _normalize(values, stat):
    return (values - stat.q01) / (stat.q99 - stat.q01 + 1e-6) * 2 - 1


def _unnormalize(values, stat):
    return (values + 1) / 2 * (stat.q99 - stat.q01 + 1e-6) + stat.q01


@dataclasses.dataclass(frozen=True)
class MultiInputs(transforms.DataTransformFn):
    """Apply joint deltas, route-owned stats, and mixed prompt formatting."""

    route_stats: dict[str, RouteStats]
    structured_probability: float = 0.5
    train: bool = True
    noise_std: float = 0.001

    def __post_init__(self):
        if not 0 <= self.structured_probability <= 1 or self.noise_std < 0:
            raise ValueError("Invalid prompt mixture or observation noise")

    def __call__(self, data: dict) -> dict:
        index = int(np.asarray(data["embodiment_id"]))
        if not 0 <= index < len(EMBODIMENTS):
            raise ValueError("Invalid embodiment_id")
        spec = EMBODIMENTS[index]
        if (
            data.get("robot_type", spec.robot_type) != spec.robot_type
            or data.get("end_effector_type", spec.end_effector_type) != spec.end_effector_type
        ):
            raise ValueError("Prompt metadata conflicts with numeric embodiment routing")
        stats = self.route_stats[spec.name]
        state = np.asarray(data["state"], dtype=np.float32).copy()
        mask = np.asarray(data["action_mask"])
        if state.shape != (ACTION_DIM,) or mask.shape != (ACTION_DIM,) or mask.dtype != np.dtype(bool):
            raise ValueError("Expected packed 64D state and boolean availability mask")
        has_hands = bool(mask[HAND_START : HAND_START + spec.hand_dim].all())
        if not np.array_equal(mask, physical_mask(index, has_hands)) or np.any(mask & ~stats.supervised_dims):
            raise ValueError("Observation contains untrained or unavailable action dimensions")
        if not np.isfinite(state[mask]).all():
            raise ValueError("Available state values must be finite")
        state[~mask] = 0
        result = {
            "state": state,
            "action_mask": mask.copy(),
            "embodiment_id": np.int32(index),
            "image": data["image"],
            "image_mask": data["image_mask"],
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32).copy()
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM or not np.isfinite(actions[:, mask]).all():
                raise ValueError("Expected finite packed action horizon")
            actions[:, : spec.joint_dim] -= state[None, : spec.joint_dim]
            if self.train and self.noise_std:
                noise = np.random.normal(0, self.noise_std, spec.joint_dim).astype(np.float32)
                state[: spec.joint_dim] += noise
                actions[:, : spec.joint_dim] -= noise
            normalized_actions = _normalize(actions, stats.actions)
            normalized_actions[:, ~mask] = 0
            result["actions"] = normalized_actions.astype(np.float32)
        normalized_state = _normalize(state, stats.state)
        normalized_state[~mask] = 0
        result["state"] = normalized_state.astype(np.float32)
        requested = data.get("prompt_format")
        if requested not in (None, "json", "plain"):
            raise ValueError("prompt_format must be json or plain")
        structured = (
            requested == "json"
            if requested is not None
            else np.random.random() < self.structured_probability
            if self.train
            else self.structured_probability >= 0.5
        )
        instruction = data.get("instruction", data.get("prompt"))
        available = bool(mask[HAND_START : HAND_START + spec.hand_dim].all())
        result["prompt"] = format_prompt(
            instruction, spec.robot_type, spec.end_effector_type, available, structured=structured
        )
        return result


@dataclasses.dataclass(frozen=True)
class MultiOutputs(transforms.DataTransformFn):
    """Undo the selected route normalization and restore absolute joints once."""

    route_stats: dict[str, RouteStats]

    def __call__(self, data: dict) -> dict:
        index = int(np.asarray(data["embodiment_id"]))
        if not 0 <= index < len(EMBODIMENTS):
            raise ValueError("Invalid embodiment_id")
        spec = EMBODIMENTS[index]
        stats = self.route_stats[spec.name]
        mask = np.asarray(data["action_mask"], dtype=bool)
        state = np.asarray(data["state"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if (
            state.shape != (ACTION_DIM,)
            or mask.shape != (ACTION_DIM,)
            or actions.ndim != 2
            or actions.shape[-1] != ACTION_DIM
        ):
            raise ValueError("Expected unbatched packed state, action mask, and action horizon")
        state = _unnormalize(state, stats.state)
        actions = _unnormalize(actions, stats.actions)
        actions[:, : spec.joint_dim] += state[None, : spec.joint_dim]
        actions[:, ~mask] = 0
        return {"actions": actions.astype(np.float32), "embodiment_id": np.int32(index), "action_mask": mask}
