"""Downstream TRAIN-only normalization of completed five-token robot chunks.

Statistics pool valid horizon endpoints per token/coordinate, including robot
controller auxiliaries. They never describe the human pretraining distribution.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import torch

ROBOT_ACTION_STATS_VERSION = "downstream_train_chunk_mean_std_v1"
VARIANCE_POLICY = "population_nonzero_std_floor_eps_constant_unit"


def normalization_identity_sha256(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_action_normalization_metadata(metadata):
    """Return a detached JSON dictionary, or None for disabled normalization."""
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping) or metadata.get("mode") != "mean_std_v1":
        raise ValueError("Robot action normalization must use mode='mean_std_v1'")
    if not isinstance(metadata.get("identity"), Mapping):
        raise ValueError("Robot action normalization requires a TRAIN statistics identity")
    try:
        result = json.loads(json.dumps(dict(metadata), allow_nan=False))
        mean = np.asarray(result["mean"], dtype=np.float64)
        std = np.asarray(result["std"], dtype=np.float64)
        count = np.asarray(result["count"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Robot normalization metadata must contain JSON mean/std/count") from exc
    if any(value.shape != (5, 15) for value in (mean, std, count)):
        raise ValueError("Robot action mean/std/count must each have shape [5,15]")
    if not all(np.isfinite(value).all() for value in (mean, std, count)) or (std <= 0).any():
        raise ValueError("Robot action mean/std must be finite with strictly positive std")
    if (count < 0).any() or not np.equal(count, np.floor(count)).all():
        raise ValueError("Robot action counts must be nonnegative integers")
    missing = count == 0
    if (mean[missing] != 0).any() or (std[missing] != 1).any():
        raise ValueError("Unobserved robot action coordinates require mean0/std1")
    if "stats_horizon" in result:
        horizon = result["stats_horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Robot action statistics horizon must be a positive integer")
    if "eps" in result and (not np.isfinite(result["eps"]) or result["eps"] <= 0):
        raise ValueError("Robot action normalization eps must be positive")
    digest = normalization_identity_sha256(result["identity"])
    if result.get("identity_sha256", digest) != digest:
        raise ValueError("Robot action statistics identity SHA256 mismatch")
    result["identity_sha256"] = digest
    result["count"] = count.astype(np.int64).tolist()
    return result


def _transform(values, metadata, loss_mask, *, inverse):
    stats = validate_action_normalization_metadata(metadata)
    if values.ndim < 2 or values.shape[-1] != 15 or values.shape[-2] % 5:
        raise ValueError("Robot action arrays must end in [H*5,15] or [H,5,15]")
    shape = values.shape
    grouped_shape = (*shape[:-2], shape[-2]//5, 5, 15)
    if isinstance(values, torch.Tensor):
        if not values.is_floating_point():
            raise ValueError("Robot action tensors must have floating dtype")
        grouped = values.reshape(grouped_shape)
        if stats is not None:
            mean = torch.as_tensor(stats["mean"], dtype=values.dtype, device=values.device)
            std = torch.as_tensor(stats["std"], dtype=values.dtype, device=values.device)
            grouped = grouped*std+mean if inverse else (grouped-mean)/std
        result = grouped.reshape(shape)
        if loss_mask is not None:
            valid = torch.as_tensor(loss_mask, device=values.device) > 0
            result = torch.where(valid, result, torch.zeros_like(result))
        return result
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Robot action arrays must have floating dtype")
    grouped = values.reshape(grouped_shape)
    if stats is not None:
        mean = np.asarray(stats["mean"], dtype=values.dtype)
        std = np.asarray(stats["std"], dtype=values.dtype)
        grouped = grouped*std+mean if inverse else (grouped-mean)/std
    result = grouped.reshape(shape)
    if loss_mask is not None:
        result = np.where(np.asarray(loss_mask) > 0, result, np.zeros((), dtype=values.dtype))
    return result


def normalize_robot_actions(values, metadata, loss_mask=None):
    """Normalize complete geometric+auxiliary chunks, retaining masked zeros."""
    return _transform(values, metadata, loss_mask, inverse=False)


def denormalize_robot_actions(values, metadata, loss_mask=None):
    """Restore physical-scale poses and controller auxiliaries before execution."""
    return _transform(values, metadata, loss_mask, inverse=True)


class RobotActionMoments:
    """Merge stable masked population moments independently for all75coordinates."""

    def __init__(self):
        self.count = np.zeros((5, 15), dtype=np.int64)
        self.mean = np.zeros((5, 15), dtype=np.float64)
        self.m2 = np.zeros((5, 15), dtype=np.float64)

    def update(self, values, mask):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float64).reshape(-1, 5, 15)
        valid = np.asarray(mask).reshape(values.shape) > 0
        if not np.isfinite(values[valid]).all():
            raise ValueError("Nonfinite valid robot action encountered during TRAIN statistics")
        n = valid.sum(axis=0, dtype=np.int64)
        safe = np.where(valid, values, 0)
        mean = safe.sum(axis=0)/np.maximum(n, 1)
        m2 = np.where(valid, (safe-mean)**2, 0).sum(axis=0)
        total = self.count+n
        delta = mean-self.mean
        self.mean += delta*n/np.maximum(total, 1)
        self.m2 += m2+delta**2*self.count*n/np.maximum(total, 1)
        self.count = total

    def metadata(self, *, identity, horizon, eps, fit_summary):
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("Robot action normalization eps must be positive")
        if not self.count.any():
            raise ValueError("No valid downstream TRAIN action coordinates for normalization")
        variance = np.maximum(self.m2/np.maximum(self.count, 1), 0)
        std = np.ones((5, 15), dtype=np.float64)
        varying = (self.count > 0) & (variance > 0)
        std[varying] = np.maximum(np.sqrt(variance[varying]), eps)
        mean = np.where(self.count > 0, self.mean, 0)
        return validate_action_normalization_metadata({
            "mode": "mean_std_v1", "mean": mean.tolist(), "std": std.tolist(),
            "count": self.count.tolist(), "identity": identity,
            "identity_sha256": normalization_identity_sha256(identity),
            "stats_horizon": int(horizon), "eps": float(eps), "variance_policy": VARIANCE_POLICY,
            "fit_summary": {**fit_summary, "zero_count_coordinates": int((self.count == 0).sum()),
                            "constant_coordinates": int(((self.count > 0) & ~varying).sum()),
                            "std_floored_coordinates": int((varying & (np.sqrt(variance) < eps)).sum())},
        })
