"""Engram observation, action, and proprioception transforms."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert CHW images to HWC and scale floating RGB to uint8."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = (255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class EngramInputs(transforms.DataTransformFn):
    """Map Engram observations to OpenPI inputs and preserve label availability."""

    model_type: _model.ModelType
    model_action_dim: int
    single_camera: bool = False

    def __call__(self, data: dict) -> dict:
        if self.single_camera and ("observation/left_wrist_rgbd" in data or "observation/right_wrist_rgbd" in data):
            raise ValueError("Single-camera Engram mode cannot discard supplied wrist images")
        head = _parse_image(data["observation/head_rgbd"])
        images = {"base_0_rgb": head}
        image_masks = {"base_0_rgb": np.True_}
        if not self.single_camera:
            for side in ("left", "right"):
                key = f"observation/{side}_wrist_rgbd"
                images[f"{side}_wrist_0_rgb"] = _parse_image(data.get(key, np.zeros_like(head)))
                image_masks[f"{side}_wrist_0_rgb"] = np.asarray(key in data)
        out = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["observation/joint_position"], dtype=np.float32),
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            out["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        mask = np.asarray(data.get("action_mask", np.ones(22, dtype=bool)), dtype=bool)
        if mask.shape != (22,) or not mask[:20].all() or mask[20] != mask[21]:
            raise ValueError("Expected all joint labels and paired gripper availability")
        if self.model_action_dim < 22:
            raise ValueError("Model action dimension must hold the 22 physical channels")
        # Padded dimensions retain the pre-existing OpenPI zero-padding objective.
        out["action_mask"] = np.pad(mask, (0, self.model_action_dim - 22), constant_values=True)
        if not mask[20]:
            out["prompt"] = "End-effector state unavailable. " + str(out.get("prompt", ""))
        return out


@dataclasses.dataclass(frozen=True)
class EngramOutputs(transforms.DataTransformFn):
    """Remove model padding from the 22 physical action channels."""

    action_dim: int = 22

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, : self.action_dim]}


@dataclasses.dataclass(frozen=True)
class MaskUnavailableState(transforms.DataTransformFn):
    """Use an explicit neutral normalized token for unavailable gripper state."""

    def __call__(self, data: dict) -> dict:
        mask = data.get("action_mask")
        if mask is not None and not np.asarray(mask, dtype=bool)[20]:
            state = np.asarray(data["state"]).copy()
            state[20:22] = 0.0
            data["state"] = state
        return data


@dataclasses.dataclass(frozen=True)
class ExactJointNoiseCompensation(transforms.DataTransformFn):
    """Perturb 20 joint observations and preserve their absolute targets exactly."""

    std: float = 0.001

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"])
        if state.shape[-1] != 22 or actions.shape[-1] != 22:
            raise ValueError("Expected 20 joints followed by two grippers")
        noise = np.random.normal(0.0, self.std, size=state.shape[:-1] + (20,)).astype(state.dtype)
        state = state.copy()
        actions = actions.copy()
        state[..., :20] += noise
        actions[..., :20] -= np.expand_dims(noise, axis=-2)
        data["state"] = state
        data["actions"] = actions
        return data
