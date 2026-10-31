"""Engram-specific training-time proprioceptive augmentation."""

import dataclasses

import numpy as np

from openpi import transforms


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
