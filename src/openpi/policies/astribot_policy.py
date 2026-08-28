import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Ensure HWC uint8 RGB."""
    image = np.asarray(image)
    # If CHW, convert to HWC.
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    # If float images (0..1), convert to uint8.
    if np.issubdtype(image.dtype, np.floating):
        image = (255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)

    # If BGR (common in ROS/OpenCV), caller should convert. We keep as-is.
    return image


@dataclasses.dataclass(frozen=True)
class AstribotInputs(transforms.DataTransformFn):
    """Map Astribot observation dicts into openpi common format.

    Expected input dict (training + inference):
      - observation/head_rgbd: HWC uint8 RGB
      - observation/left_wrist_rgbd: HWC uint8 RGB (optional)
      - observation/right_wrist_rgbd: HWC uint8 RGB (optional)
      - observation/joint_position: (D,) float (robot joint positions in the configured order)
      - actions: (T, D) float (optional; only during training)
      - prompt: str (optional; will be injected by ModelTransformFactory default prompt or prompt_from_task)

    Output dict (openpi common format):
      - image: dict[str, HWC uint8]
      - image_mask: dict[str, bool]
      - state: float32[D]
      - actions: float32[T,D] (if present)
      - prompt: str (if present)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # State.
        state = np.asarray(data["observation/joint_position"], dtype=np.float32)

        # Images.
        head = _parse_image(data["observation/head_rgbd"])
        left = _parse_image(data.get("observation/left_wrist_rgbd", np.zeros_like(head)))
        right = _parse_image(data.get("observation/right_wrist_rgbd", np.zeros_like(head)))

        images = {
            "base_0_rgb": head,
            "left_wrist_0_rgb": left,
            "right_wrist_0_rgb": right,
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.asarray("observation/left_wrist_rgbd" in data),
            "right_wrist_0_rgb": np.asarray("observation/right_wrist_rgbd" in data),
        }

        out = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            out["prompt"] = prompt

        return out


@dataclasses.dataclass(frozen=True)
class AstribotOutputs(transforms.DataTransformFn):
    """Postprocess model outputs back to Astribot action space."""

    action_dim: int = 25

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}
