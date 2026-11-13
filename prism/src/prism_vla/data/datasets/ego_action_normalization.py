"""Checkpoint-portable statistics for completed, masked ego action chunks."""
from __future__ import annotations

import json

import numpy as np
import torch


def validate_ego_action_normalization(metadata):
    """Validate TRAIN-fitted per-source statistics without altering their values."""
    if metadata is None:
        return None
    result = json.loads(json.dumps(metadata, allow_nan=False))
    if result.get("mode") != "completed_chunk_masked_mean_std_v1":
        raise ValueError("Unsupported ego action normalization mode")
    policy = result.get("action_normalization_max_abs_policy", "mask")
    if not isinstance(policy, str) or policy not in {"mask", "clip"}:
        raise ValueError("action_normalization_max_abs_policy must be mask or clip")
    for name in ("normalized_action_clip_abs_max", "normalized_action_mask_abs_max"):
        bound = result.get(name)
        if bound is not None and (
            isinstance(bound, bool) or not isinstance(bound, (int, float))
            or not np.isfinite(bound) or bound <= 0
        ):
            raise ValueError(f"{name} must be finite and positive or null")
    allow_empty_datasets = result.get("allow_empty_datasets", False)
    if type(allow_empty_datasets) is not bool:
        raise ValueError("allow_empty_datasets must be an explicit boolean")
    if not isinstance(result.get("horizon"), int) or result["horizon"] < 1:
        raise ValueError("Ego action normalization requires a positive horizon")
    from .ego_action_representation import normalize_action_representation
    result["representation"] = normalize_action_representation(result.get("representation"))
    if not np.isfinite(result.get("position_scale", 0)) or result.get("position_scale", 0) <= 0:
        raise ValueError("Ego action normalization requires a positive position scale")
    datasets = result.get("by_dataset")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Ego action normalization requires per-dataset TRAIN statistics")
    for name, stats in datasets.items():
        mean, std, count = (np.asarray(stats.get(key), np.float64) for key in ("mean", "std", "count"))
        if any(value.shape != (5, 15) or not np.isfinite(value).all() for value in (mean, std, count)):
            raise ValueError(f"Invalid ego action normalization arrays for {name}")
        if ((std <= 0).any() or (count < 0).any()
                or not np.equal(count, np.floor(count)).all()
                or (not count.any() and not allow_empty_datasets)):
            raise ValueError(f"Invalid ego action normalization counts/scales for {name}")
        missing = count == 0
        if (mean[missing] != 0).any() or (std[missing] != 1).any():
            raise ValueError(f"Unsupported coordinates must use mean=0/std=1 for {name}")
    return result


def denormalize_ego_action_chunks(values, metadata, *, dataset_name: str, mask=None):
    """Restore physical target coordinates before geometric decoding or metrics.

    Select the source's checkpoint-owned statistics explicitly. This reverses
    normalization; the stored camera/anchor/hand contract still owns geometry.
    """
    stats = validate_ego_action_normalization(metadata)
    if stats is None:
        return values
    if values.shape[-1] != 15 or values.shape[-2] % 5:
        raise ValueError("Ego action chunks must end in [H*5,15]")
    selected = stats["by_dataset"][dataset_name]
    shape = values.shape
    grouped = values.reshape(*shape[:-2], shape[-2] // 5, 5, 15)
    if torch.is_tensor(values):
        mean = torch.as_tensor(selected["mean"], device=values.device, dtype=values.dtype)
        std = torch.as_tensor(selected["std"], device=values.device, dtype=values.dtype)
        result = (grouped * std + mean).reshape(shape)
        if mask is not None:
            result = torch.where(torch.as_tensor(mask, device=values.device) > 0, result, 0)
    else:
        result = (grouped * np.asarray(selected["std"], values.dtype)
                  + np.asarray(selected["mean"], values.dtype)).reshape(shape)
        if mask is not None:
            result = np.where(np.asarray(mask) > 0, result, 0)
    return result
