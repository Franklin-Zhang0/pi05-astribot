"""Geometric supervision for camera-relative wrists with unobserved camera motion.

The loss constrains the composition of local head and wrist increments, not their
otherwise unidentifiable decomposition. Metric sources must anchor that decomposition.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


WEAK_POSE_PAYLOAD_DIM = 217
WEAK_POSE_CURRENT_SLICE = slice(0, 32)
WEAK_POSE_TARGET_SLICE = slice(32, 64)
WEAK_POSE_MASK_SLICE = slice(64, 66)
WEAK_POSE_MEAN_SLICE = slice(66, 141)
WEAK_POSE_STD_SLICE = slice(141, 216)
WEAK_POSE_WEIGHT_INDEX = 216


def pack_weak_pose_payload(
    current_wrist: Tensor,
    target_wrist: Tensor,
    target_mask: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
    weight: Tensor,
) -> Tensor:
    """Pack per-transition geometry into a float32 tensor ending in 217 values.

    ``current_wrist`` and ``target_wrist`` end in [2,4,4], masks in [2], and
    normalization statistics in [5,15]. Leading dimensions broadcast; for example
    use [N,1,5,15] statistics for [N,S,2,4,4] poses. Weight has no trailing axis.
    """
    device = current_wrist.device
    parts = [
        current_wrist.flatten(-3), target_wrist.flatten(-3),
        target_mask, action_mean.flatten(-2), action_std.flatten(-2),
        weight.unsqueeze(-1),
    ]
    leading = torch.broadcast_shapes(*(part.shape[:-1] for part in parts))
    return torch.cat(
        [part.to(device=device, dtype=torch.float32).expand(*leading, part.shape[-1])
         for part in parts], dim=-1,
    )


def unpack_weak_pose_payload(payload: Tensor) -> dict[str, Tensor]:
    """Return views of a packed per-transition geometry payload."""
    if payload.shape[-1] != WEAK_POSE_PAYLOAD_DIM:
        raise ValueError(f"Weak pose payload must end in {WEAK_POSE_PAYLOAD_DIM} values")
    leading = payload.shape[:-1]
    return {
        "current_wrist": payload[..., WEAK_POSE_CURRENT_SLICE].reshape(*leading, 2, 4, 4),
        "target_wrist": payload[..., WEAK_POSE_TARGET_SLICE].reshape(*leading, 2, 4, 4),
        "target_mask": payload[..., WEAK_POSE_MASK_SLICE] > 0.5,
        "action_mean": payload[..., WEAK_POSE_MEAN_SLICE].reshape(*leading, 5, 15),
        "action_std": payload[..., WEAK_POSE_STD_SLICE].reshape(*leading, 5, 15),
        "weight": payload[..., WEAK_POSE_WEIGHT_INDEX],
    }


def _rotation_6d_to_matrix(rotation: Tensor) -> Tensor:
    """Decode the schema's first-two-rows convention, including degenerate inputs."""
    first, second = rotation[..., :3], rotation[..., 3:]
    x_axis = torch.zeros_like(first)
    x_axis[..., 0] = 1.0
    # The fallback remains differentiable at an all-zero model prediction.
    first = torch.where(first.norm(dim=-1, keepdim=True) > 1e-6, first, first + x_axis)
    first = F.normalize(first, dim=-1, eps=1e-6)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    basis = F.one_hot(first.abs().argmin(dim=-1), num_classes=3).to(first.dtype)
    fallback = basis - (basis * first).sum(dim=-1, keepdim=True) * first
    second = torch.where(second.norm(dim=-1, keepdim=True) > 1e-6, second, second + fallback)
    second = F.normalize(second, dim=-1, eps=1e-6)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-2)


def weak_action_supervision_weight(
    payload: Tensor, action_chunk_valid_steps: Tensor | None = None,
) -> Tensor:
    """Count nine supervised action coordinates per valid weak pose.

    Payload is [N,S,217]; optional valid-step counts are in flattened action
    tokens, so only complete groups of five contribute. Dataset coefficients
    scale the numerator elsewhere and do not enter this denominator.
    """
    if payload.ndim != 3:
        raise ValueError("Weak pose payload must have shape [N,S,217]")
    fields = unpack_weak_pose_payload(payload)
    valid = fields["target_mask"] & (fields["weight"] > 0).unsqueeze(-1)
    valid = valid & torch.isfinite(fields["current_wrist"]).all(dim=(-1, -2))
    valid = valid & torch.isfinite(fields["target_wrist"]).all(dim=(-1, -2))
    if action_chunk_valid_steps is not None:
        steps = torch.as_tensor(action_chunk_valid_steps, device=payload.device).reshape(-1)
        if steps.numel() != payload.shape[0]:
            raise ValueError("Valid action token counts must match the weak pose batch")
        horizon = torch.arange(payload.shape[1], device=payload.device)
        valid = valid & (horizon[None, :] < (steps[:, None] // 5)).unsqueeze(-1)
    return valid.float().sum() * 9.0


def camera_relative_pose_loss(
    pred_actions: Tensor,
    payload: Tensor,
    *,
    translation_scale: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Return weighted pose-loss sum and unweighted valid-pose count.

    ``pred_actions`` is [N,5*S,15] in normalized UMI-local action space; payload
    is [N,S,217]. For each observed camera-from-wrist H, reconstruct its next
    pose as inv(delta_camera) @ H @ delta_wrist. This uses observed geometry
    solely as a training target, independently per transition, so missing poses
    do not corrupt later valid transitions.

    Translation uses mean Huber error in units of ``translation_scale`` metres;
    rotation uses squared chordal distance / 8 (sin^2 of half the angle).
    Divide the first result by count.clamp_min(1). Dataset weights scale the
    loss rather than cancelling against a weighted denominator. Zero weights
    and invalid poses contribute neither loss nor count. Finger/padding tokens
    are unused. Arithmetic stays float32 under mixed-precision autocast.
    """
    if translation_scale <= 0:
        raise ValueError("translation_scale must be positive")
    if pred_actions.ndim != 3 or pred_actions.shape[-1] != 15 or pred_actions.shape[1] % 5:
        raise ValueError("Weak pose loss requires actions with shape [N,5*S,15]")
    n, token_count, _ = pred_actions.shape
    if payload.shape != (n, token_count // 5, WEAK_POSE_PAYLOAD_DIM):
        raise ValueError("Weak pose payload must match the action batch and physical horizon")
    with torch.autocast(device_type=pred_actions.device.type, enabled=False):
        fields = unpack_weak_pose_payload(payload.to(device=pred_actions.device, dtype=torch.float32))
        current, target = fields["current_wrist"], fields["target_wrist"]
        weight = fields["weight"]
        valid = fields["target_mask"] & (weight > 0).unsqueeze(-1)
        valid = valid & torch.isfinite(current).all(dim=(-1, -2)) & torch.isfinite(target).all(dim=(-1, -2))
        identity = torch.eye(4, dtype=torch.float32, device=pred_actions.device)
        current = torch.where(valid[..., None, None], current, identity)
        target = torch.where(valid[..., None, None], target, identity)
        # Mask before decoding: even NaN padding cannot affect the loss or gradients.
        token_valid = torch.stack((valid.any(-1), valid[..., 0], valid[..., 1]), dim=-1)
        actions = pred_actions.float().reshape(n, token_count // 5, 5, 15)[..., [0, 1, 3], :9]
        mean = fields["action_mean"][..., [0, 1, 3], :9]
        std = fields["action_std"][..., [0, 1, 3], :9]
        actions = torch.where(token_valid[..., None], actions, 0.0)
        mean = torch.where(token_valid[..., None], mean, 0.0)
        std = torch.where(token_valid[..., None], std, 1.0)
        decoded = actions * std + mean
        rotations = _rotation_6d_to_matrix(decoded[..., 3:9])
        camera_r_inv = rotations[..., 0, :, :].transpose(-1, -2).unsqueeze(-3)
        wrist_r = rotations[..., 1:, :, :]
        wrist_t = decoded[..., 1:, :3]
        camera_t = decoded[..., 0, :3].unsqueeze(-2)
        next_r = camera_r_inv @ current[..., :3, :3] @ wrist_r
        next_t = (current[..., :3, :3] @ wrist_t.unsqueeze(-1)).squeeze(-1)
        next_t = next_t + current[..., :3, 3] - camera_t
        next_t = (camera_r_inv @ next_t.unsqueeze(-1)).squeeze(-1)
        translation_loss = F.smooth_l1_loss(
            next_t / translation_scale, target[..., :3, 3] / translation_scale,
            reduction="none",
        ).mean(dim=-1)
        rotation_loss = (next_r - target[..., :3, :3]).square().sum(dim=(-1, -2)) / 8.0
        pose_loss = translation_loss + rotation_loss
        weights = torch.where(valid, weight.unsqueeze(-1), 0.0)
        return (pose_loss * weights).sum(), valid.float().sum()
