"""Ordered transition-row selection without a device-side nonzero query."""

from __future__ import annotations

import torch


def cpu_transition_row_indices(num_transitions: torch.Tensor, width: int) -> torch.Tensor:
    """Return exactly ``(arange(width) < counts).flatten().nonzero()`` on CPU.

    Construct this in the collator, before DataLoader pinning. The comparison
    deliberately retains the original mask semantics for padded/ragged rows.
    """
    if (not torch.is_tensor(num_transitions) or num_transitions.device.type != "cpu"
            or num_transitions.dtype != torch.long or num_transitions.ndim != 1):
        raise ValueError("num_transitions must be a one-dimensional CPU int64 tensor")
    if type(width) is not int or width < 0:
        raise ValueError("transition width must be a non-negative integer")
    valid = torch.arange(width, device="cpu", dtype=torch.long).unsqueeze(0) < num_transitions.unsqueeze(1)
    return valid.reshape(-1).nonzero(as_tuple=True)[0]


def validate_transition_row_indices(indices: torch.Tensor, valid_mask: torch.Tensor) -> None:
    """Check tensor metadata only; CPU collator construction owns row values.

    Reading index values here would reintroduce a CUDA-to-host fence. Like the
    other batch tensors, these values must come from the matching collated batch.
    """
    if (not torch.is_tensor(indices) or indices.dtype != torch.long or indices.ndim != 1
            or indices.device != valid_mask.device or indices.numel() > valid_mask.numel()):
        raise ValueError("transition_row_indices must be same-device rank-1 int64 batch indices")


def select_transition_rows(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    row_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select identical, ascending unique rows using collator indices if supplied."""
    if value.shape[0] != valid_mask.numel():
        raise ValueError("transition values and valid mask have different row counts")
    if row_indices is None:
        return value[valid_mask.reshape(-1)]
    validate_transition_row_indices(row_indices, valid_mask)
    return value.index_select(0, row_indices)
