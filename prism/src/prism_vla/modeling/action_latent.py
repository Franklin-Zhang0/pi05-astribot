"""Action-chunk CVAE posterior and target preparation for transient ACT inputs.

The encoder follows ACT's action/proprio posterior pattern. Its target inputs
are training-only; the backbone's ACT observer never writes persistent state.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _Float32Linear(nn.Linear):
    def forward(self, inputs):
        # Calling this submodule normally preserves FSDP/Muon parameter hooks.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.float(), self.weight.float(), self.bias.float())


class ActionChunkPosterior(nn.Module):
    """Encode one masked action chunk per row into a diagonal Gaussian."""

    def __init__(self, *, action_dim: int, horizon: int, proprio_dim: int,
                 latent_dim: int, encoder_dim: int, layers: int, heads: int):
        super().__init__()
        self.action_dim, self.horizon = action_dim, horizon
        self.proprio_dim, self.latent_dim = proprio_dim, latent_dim
        # Include element validity explicitly: missing labels are not zero actions.
        self.action_projection = nn.Linear(2 * action_dim, encoder_dim)
        self.proprio_projection = nn.Linear(proprio_dim, encoder_dim) if proprio_dim else None
        self.summary = nn.Parameter(torch.empty(1, 1, encoder_dim))
        self.position = nn.Parameter(torch.empty(1, horizon + 2, encoder_dim))
        layer = nn.TransformerEncoderLayer(encoder_dim, heads, 4 * encoder_dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(encoder_dim)
        self.statistics = _Float32Linear(encoder_dim, 2 * latent_dim)
        nn.init.normal_(self.summary, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, actions: torch.Tensor, mask: torch.Tensor,
                proprio: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("CVAE actions must have the configured [N, horizon, action_dim] shape")
        if mask.shape != actions.shape:
            raise ValueError("CVAE mask must match actions")
        # Mask before every encoder operation, including masked NaN payloads.
        safe = torch.where(mask, actions, torch.zeros_like(actions))
        dtype = self.action_projection.weight.dtype
        tokens = self.action_projection(torch.cat((safe, mask.to(safe)), dim=-1).to(dtype))
        batch = actions.shape[0]
        prefix = [self.summary.expand(batch, -1, -1)]
        if self.proprio_projection is not None:
            if proprio is None or proprio.shape != (batch, self.proprio_dim):
                raise ValueError("CVAE posterior requires current proprio for every action row")
            prefix.append(self.proprio_projection(proprio.to(dtype)).unsqueeze(1))
        inputs = torch.cat((*prefix, tokens), dim=1)
        padding = torch.cat((torch.zeros(batch, len(prefix), device=mask.device, dtype=torch.bool),
                             ~mask.any(dim=-1)), dim=1)
        hidden = self.encoder(inputs + self.position[:, :inputs.shape[1]].to(inputs),
                              src_key_padding_mask=padding)
        pooled = self.norm(hidden[:, 0])
        # The small distribution head and KL arithmetic remain FP32 under BF16 autocast.
        stats = self.statistics(pooled)
        mu, logvar = stats.chunk(2, dim=-1)
        # Bound variance only for exp's numerical range; this is not an action cap.
        return mu, logvar.clamp(-20.0, 10.0)


def action_cvae_target_payload(model, action_chunks, valid_steps, loss_mask, head_targets):
    """Use the primary head's already normalized/clipped training targets.

    ``action_clip_abs_max`` belongs to sampling for the existing experts. It
    must not silently clip this posterior's copy of the dataset's targets.
    """
    backend = model.idm
    if hasattr(backend, "heads"):
        name = backend.primary_head
        if head_targets is None or name not in head_targets:
            raise ValueError("CVAE multi-head training requires primary action-head targets")
        payload = head_targets[name]
        action_chunks = payload.get("action_chunks")
        valid_steps = payload.get("action_chunk_valid_steps")
        loss_mask = payload.get("action_chunk_loss_mask")
        backend = backend.heads[name]
    return action_chunks, valid_steps, loss_mask, None


def prepare_action_cvae_targets(actions: torch.Tensor, *, valid_rows: torch.Tensor,
                                valid_steps: torch.Tensor | None = None,
                                loss_mask: torch.Tensor | None = None,
                                clip_abs_max: float | None = None):
    """Return safe primary targets, element masks and valid posterior rows."""
    from prism_vla.modeling.idm_backends import _combine_action_chunk_masks
    if actions.ndim != 3:
        raise ValueError("CVAE targets require flattened [N, H, A] action chunks")
    mask = _combine_action_chunk_masks(valid_steps, loss_mask,
        horizon=actions.shape[1], action_dim=actions.shape[2], device=actions.device,
        dtype=torch.float32)
    if mask is None:
        mask = torch.ones_like(actions, dtype=torch.bool)
    else:
        mask = mask > 0
    mask = mask & valid_rows.to(device=actions.device, dtype=torch.bool)[:, None, None]
    mask = mask & torch.isfinite(actions)
    safe = torch.where(mask, actions, torch.zeros_like(actions))
    if clip_abs_max is not None:
        safe = safe.clamp(-float(clip_abs_max), float(clip_abs_max))
    return safe, mask, mask.any(dim=(-2, -1))


def action_cvae_kl(mu: torch.Tensor, logvar: torch.Tensor, valid_rows: torch.Tensor):
    """Sum latent dimensions, then average only valid action-chunk rows."""
    rows = 0.5 * (mu.float().square() + logvar.float().exp() - 1.0 - logvar.float()).sum(-1)
    numerator = torch.where(valid_rows, rows, torch.zeros_like(rows)).sum()
    count = valid_rows.float().sum()
    return numerator, count
