"""Utilities for selecting and fusing visual encoder streams."""

from __future__ import annotations

import torch
import torch.nn as nn


VALID_VISUAL_ENCODER_MODES = {"dino", "semantic", "both"}


def normalize_visual_encoder_mode(mode: str | None) -> str:
    normalized = str(mode or "dino").strip().lower()
    aliases = {
        "dinov3": "dino",
        "qwen_vl": "semantic",
        "qwenvl": "semantic",
        "sem": "semantic",
        "dual": "both",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VALID_VISUAL_ENCODER_MODES:
        raise ValueError(
            f"Unsupported visual_encoder_mode={mode!r}. "
            f"Expected one of {sorted(VALID_VISUAL_ENCODER_MODES)}."
        )
    return normalized


def uses_dino(mode: str | None) -> bool:
    return normalize_visual_encoder_mode(mode) in {"dino", "both"}


def uses_semantic(mode: str | None) -> bool:
    return normalize_visual_encoder_mode(mode) in {"semantic", "both"}


class ResidualMLPFusion(nn.Module):
    """Fuse same-width latents with a residual pooled from the input streams."""

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 2,
        *,
        num_inputs: int = 2,
        residual_pooling: str = "primary",
        residual_weights: list[float] | None = None,
    ):
        super().__init__()
        if num_inputs < 2:
            raise ValueError(f"num_inputs must be >= 2, got {num_inputs}")
        residual_pooling = str(residual_pooling).strip().lower()
        if residual_pooling not in {"primary", "mean", "weighted"}:
            raise ValueError(
                f"Unsupported residual_pooling={residual_pooling!r}; "
                "expected 'primary', 'mean', or 'weighted'."
            )
        if residual_pooling == "weighted":
            if residual_weights is None or len(residual_weights) != num_inputs:
                raise ValueError(
                    "residual_pooling='weighted' requires one residual weight per input."
                )
            weights = torch.as_tensor(residual_weights, dtype=torch.float32)
            if not torch.isfinite(weights).all() or float(weights.abs().sum()) <= 0.0:
                raise ValueError("residual_weights must be finite and not all zero.")
            weights = weights / weights.sum()
            self.register_buffer("_residual_weights", weights, persistent=False)
        else:
            self.register_buffer("_residual_weights", None, persistent=False)

        self.num_inputs = int(num_inputs)
        self.residual_pooling = residual_pooling
        hidden_dim = int(dim) * int(hidden_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(self.num_inputs * dim),
            nn.Linear(self.num_inputs * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # Start from the residual pool and let training learn the cross-stream delta.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _residual(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.residual_pooling == "primary":
            return inputs[0]
        stacked = torch.stack(inputs, dim=0)
        if self.residual_pooling == "mean":
            return stacked.mean(dim=0)
        weights = self._residual_weights.to(device=stacked.device, dtype=stacked.dtype)
        view_shape = (self.num_inputs,) + (1,) * (stacked.ndim - 1)
        return (stacked * weights.view(view_shape)).sum(dim=0)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        return self._residual(inputs) + self.net(torch.cat(list(inputs), dim=-1))
