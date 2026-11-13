"""TRAIN-only scalar normalization for RoboDojo's recorded 28-D proprio.

The two WXYZ quaternions stay on their recorded unit sphere. Positions,
grippers and joints use per-coordinate population moments; constant channels
have unit scale. This is separate from the five-token action normalizer.
"""
from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np
import torch

from prism_vla.data.datasets.robot_action_normalization import normalization_identity_sha256

ROBOT_PROPRIO_STATS_VERSION = "robodojo_native28_scalar_mean_std_unit_quaternion_v1"
QUATERNION_INDICES = (3, 4, 5, 6, 17, 18, 19, 20)
EEF_TRANSLATION_INDICES = (0, 1, 2, 14, 15, 16)


def validate_proprio_normalization_metadata(metadata):
    """Validate and detach the complete physical-to-model coordinate contract."""
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping) or metadata.get("mode") != ROBOT_PROPRIO_STATS_VERSION:
        raise ValueError("Native RoboDojo proprio normalization has an unsupported mode")
    if not isinstance(metadata.get("identity"), Mapping) or not isinstance(metadata.get("fit_summary"), Mapping):
        raise ValueError("Native proprio normalization requires TRAIN identity and fit provenance")
    try:
        result = json.loads(json.dumps(dict(metadata), allow_nan=False))
        mean, std, count = (np.asarray(result[key], dtype=np.float64) for key in ("mean", "std", "count"))
        digest = normalization_identity_sha256(result["identity"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Native proprio normalization requires JSON mean/std/count and identity") from exc
    if any(x.shape != (28,) for x in (mean, std, count)):
        raise ValueError("Native proprio mean/std/count must each have shape [28]")
    if not all(np.isfinite(x).all() for x in (mean, std, count)) or (std <= 0).any():
        raise ValueError("Native proprio statistics must be finite with positive std")
    if (count < 0).any() or not np.equal(count, np.floor(count)).all():
        raise ValueError("Native proprio counts must be nonnegative integers")
    if not np.equal(mean[list(QUATERNION_INDICES)], 0).all() or not np.equal(std[list(QUATERNION_INDICES)], 1).all():
        raise ValueError("Native proprio quaternions must retain mean0/std1")
    if (mean[count == 0] != 0).any() or (std[count == 0] != 1).any():
        raise ValueError("Unobserved native proprio channels require mean0/std1")
    if result.get("identity_sha256") != digest:
        raise ValueError("Native proprio statistics identity SHA256 mismatch")
    if result.get("fit_summary", {}).get("split") != "train":
        raise ValueError("Native proprio statistics must be fitted on TRAIN")
    result["count"] = count.astype(np.int64).tolist()
    return result


def normalize_robot_proprio(values, metadata, *, inverse=False):
    """Apply scalar affine normalization while preserving quaternion values."""
    stats = validate_proprio_normalization_metadata(metadata)
    if values.shape[-1] != 28:
        raise ValueError("Native RoboDojo proprio must end in dimension28")
    if stats is None:
        return values
    if isinstance(values, torch.Tensor):
        mean = torch.as_tensor(stats["mean"], dtype=values.dtype, device=values.device)
        std = torch.as_tensor(stats["std"], dtype=values.dtype, device=values.device)
    else:
        values = np.asarray(values)
        mean = np.asarray(stats["mean"], dtype=values.dtype)
        std = np.asarray(stats["std"], dtype=values.dtype)
    return values * std + mean if inverse else (values - mean) / std


class RobotProprioMoments:
    """Stable mergeable population moments over recorded native proprio rows."""

    def __init__(self):
        self.count = 0
        self.mean = np.zeros(28, dtype=np.float64)
        self.m2 = np.zeros(28, dtype=np.float64)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 28 or not np.isfinite(values).all():
            raise ValueError("TRAIN native proprio must be finite [T,28]")
        n = len(values)
        if not n:
            return
        mean = values.mean(axis=0)
        m2 = ((values - mean) ** 2).sum(axis=0)
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.m2 += m2 + delta ** 2 * self.count * n / total
        self.count = total

    def metadata(self, *, identity, eps, fit_summary):
        if not self.count or not np.isfinite(eps) or eps <= 0:
            raise ValueError("Native proprio statistics need TRAIN rows and a positive epsilon")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0))
        # Truly or numerically constant dimensions must not amplify noise.
        std[std <= eps] = 1.0
        mean = self.mean.copy()
        mean[list(QUATERNION_INDICES)] = 0.0
        std[list(QUATERNION_INDICES)] = 1.0
        return validate_proprio_normalization_metadata({
            "mode": ROBOT_PROPRIO_STATS_VERSION, "identity": identity,
            "identity_sha256": normalization_identity_sha256(identity),
            "mean": mean.tolist(), "std": std.tolist(), "count": [self.count] * 28,
            "eps": float(eps), "fit_summary": fit_summary,
            "augmentation": "normalized_eef_xyz_only_physical_camera_anchor_compensation_v1",
        })
