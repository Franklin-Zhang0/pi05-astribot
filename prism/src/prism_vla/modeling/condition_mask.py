"""Condition-list metadata for camera slots ignored by action decoders.

Plain Python lists preserve the original decoder path. Camera condition lists
carry immutable-by-convention presence metadata; an all-valid list uses that
same original path too. Constructing a list checks this metadata once, rather
than synchronizing separately in every action head.
"""
from __future__ import annotations

import torch


class ConditionList(list):
    """A condition sequence with a boolean ``[B, L]`` token-validity mask."""

    def __init__(self, values, *, valid_mask: torch.Tensor,
                 end_effector_type_ids: torch.Tensor | None = None, backbone_kv=None):
        super().__init__(values)
        if not self or not all(torch.is_tensor(value) and value.ndim == 2 for value in self):
            raise ValueError("Conditions must be a nonempty list of [B, D] tensors")
        batch = self[0].shape[0]
        if any(value.shape[0] != batch for value in self):
            raise ValueError("Condition batch sizes differ")
        if (not torch.is_tensor(valid_mask) or valid_mask.dtype != torch.bool
                or tuple(valid_mask.shape) != (batch, len(self))):
            raise ValueError("Condition validity must be boolean [B, number_of_conditions]")
        if end_effector_type_ids is not None:
            if (end_effector_type_ids.dtype != torch.long
                    or tuple(end_effector_type_ids.shape) != (batch, 2)):
                raise ValueError("Effector IDs must be int64 [B, 2]")
            end_effector_type_ids = end_effector_type_ids.detach().to(device=self[0].device)
        self.backbone_kv = backbone_kv
        self.end_effector_type_ids = end_effector_type_ids
        self.all_valid = bool(valid_mask.all())
        self.valid_mask = valid_mask.detach().to(device=self[0].device).clone()


def condition_mask(conditions, *, device=None):
    """Return an effective mask, or None for the exact legacy/all-valid path."""
    if not isinstance(conditions, ConditionList) or conditions.all_valid:
        return None
    mask = conditions.valid_mask
    if tuple(mask.shape) != (conditions[0].shape[0], len(conditions)):
        raise ValueError("Condition validity no longer matches the condition list")
    return mask if device is None else mask.to(device=device)


def replace_conditions(conditions, values):
    """Keep presence metadata when replacing condition values, for example CFG."""
    if isinstance(conditions, ConditionList):
        return ConditionList(values, valid_mask=conditions.valid_mask,
            end_effector_type_ids=conditions.end_effector_type_ids, backbone_kv=conditions.backbone_kv)
    return list(values)


def select_conditions(conditions, rows):
    """Select the same batch rows from conditions and their presence inventory."""
    values = [value[rows] for value in conditions]
    if isinstance(conditions, ConditionList):
        return ConditionList(values, valid_mask=conditions.valid_mask[rows],
            end_effector_type_ids=(None if conditions.end_effector_type_ids is None
                else conditions.end_effector_type_ids[rows]),
            backbone_kv=None if conditions.backbone_kv is None else conditions.backbone_kv.select(rows))
    return values


def repeat_conditions(conditions, repeats: int, *, pattern: str = "tile"):
    """Repeat batch rows in block/tile order, or explicitly interleave rows."""
    if type(repeats) is not int or repeats < 1 or pattern not in ("tile", "interleave"):
        raise ValueError("Condition repetition needs a positive count and tile/interleave pattern")
    def repeat(value):
        return value.repeat(repeats, 1) if pattern == "tile" else value.repeat_interleave(repeats, dim=0)
    values = [repeat(value) for value in conditions]
    if isinstance(conditions, ConditionList):
        return ConditionList(values, valid_mask=repeat(conditions.valid_mask),
            end_effector_type_ids=(None if conditions.end_effector_type_ids is None
                else repeat(conditions.end_effector_type_ids)),
            backbone_kv=None if conditions.backbone_kv is None else conditions.backbone_kv.repeat(repeats, interleave=pattern == "interleave"))
    return values


def masked_condition_values(conditions):
    """Zero invalid raw values before biased projections or normalization."""
    mask = condition_mask(conditions)
    if mask is None:
        return conditions
    return [torch.where(mask[:, index:index + 1].to(value.device), value, 0)
        for index, value in enumerate(conditions)]


def mask_projected_conditions(tokens, conditions):
    """Remove projection biases from absent condition tokens."""
    mask = condition_mask(conditions, device=tokens.device)
    return tokens if mask is None else torch.where(mask.unsqueeze(-1), tokens, 0)


def last_valid_condition(conditions, *, device):
    """Find the BOS context without using an absent final physical slot."""
    mask = condition_mask(conditions, device=device)
    if mask is None:
        return None
    indices = torch.arange(mask.shape[1], device=device).expand(mask.shape[0], -1)
    last = torch.where(mask, indices, -1).amax(dim=-1)
    if last.device.type == "cuda":
        torch._assert_async((last >= 0).all(), "FAST needs at least one valid condition per row")
    elif not bool((last >= 0).all()):
        raise ValueError("FAST needs at least one valid condition per row")
    return last
