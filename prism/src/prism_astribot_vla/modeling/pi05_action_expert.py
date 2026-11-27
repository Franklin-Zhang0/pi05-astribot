"""Knowledge-insulated Qwen action expert with native shared K/V attention.

This is a PRISM/Qwen adaptation of pi0.5 knowledge insulation, not a Gemma
checkpoint implementation. Every native full-attention layer supplies its own
detached K/V to one independent action block. Qwen recurrent layers have no
per-token K/V; their information already influences the full-attention keys.
The expert has bidirectional action-suffix attention, time-adaptive norms and
no compact/proprio/CLS shortcut. Only the FM and FAST heads train the compact
bottleneck in this default mode. Token reduction opts into a separate compact-
conditioned variant: per-block K/V project only final visual conditions, with
gradients enabled into those conditions and no intermediate backbone K/V.
Reference: https://www.pi.website/research/knowledge_insulation
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from prism_astribot_vla.modeling.idm_backends import (
    FlowMatchingIDM, AdaLNZeroFinalLayer, timestep_embedding,
    _configure_flow_regression_loss, _configure_action_clip,
    _configure_sampled_action_clip,
)
from prism_astribot_vla.modeling.config import validate_flow_output_projection


def _rope(tensor, positions, *, rotary_dim, theta):
    if not rotary_dim:
        return tensor
    frequency = theta ** (-torch.arange(0, rotary_dim, 2, device=tensor.device).float() / rotary_dim)
    angle = positions.float()[..., None] * frequency
    angle = torch.cat((angle, angle), dim=-1)[:, None]
    first, rest = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat((-first[..., half:], first[..., :half]), dim=-1)
    return torch.cat((first * angle.cos().to(tensor.dtype) + rotated * angle.sin().to(tensor.dtype), rest), dim=-1)


class _ActionBlock(nn.Module):
    def __init__(self, width, nhead, head_dim, ff_mult, rotary_dim, rope_theta,
                 compact_cond_dim=None):
        super().__init__()
        self.nhead, self.head_dim = nhead, head_dim
        self.rotary_dim, self.rope_theta = rotary_dim, rope_theta
        self.norm = nn.RMSNorm(width)
        self.modulation = nn.Linear(width, width)
        self.q_proj = nn.Linear(width, nhead * head_dim, bias=False)
        self.k_proj = nn.Linear(width, nhead * head_dim, bias=False)
        self.v_proj = nn.Linear(width, nhead * head_dim, bias=False)
        self.q_norm, self.k_norm = nn.RMSNorm(head_dim), nn.RMSNorm(head_dim)
        self.o_proj = nn.Linear(nhead * head_dim, width, bias=False)
        self.ff_norm = nn.RMSNorm(width)
        self.ff_modulation = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, width * ff_mult), nn.SiLU(), nn.Linear(width * ff_mult, width))
        # The opt-in dense-patch architecture supplies only final visual states.
        # Each expert block selects its own feature subspaces directly from D.
        self.compact_k_proj = (nn.Linear(compact_cond_dim, nhead * head_dim, bias=False)
                               if compact_cond_dim is not None else None)
        self.compact_v_proj = (nn.Linear(compact_cond_dim, nhead * head_dim, bias=False)
                               if compact_cond_dim is not None else None)

    def forward(self, hidden, time, rows, layer):
        batch, horizon, _ = hidden.shape
        normalized = self.norm(hidden) * (1 + self.modulation(time)[:, None])
        shape = (batch, horizon, self.nhead, self.head_dim)
        query = self.q_norm(self.q_proj(normalized).view(shape)).transpose(1, 2)
        action_key = self.k_norm(self.k_proj(normalized).view(shape)).transpose(1, 2)
        action_value = self.v_proj(normalized).view(shape).transpose(1, 2)
        if self.compact_k_proj is not None:
            from prism_astribot_vla.modeling.condition_mask import condition_mask, masked_condition_values

            compact = torch.stack(masked_condition_values(rows), dim=1).to(hidden)
            count = compact.shape[1]
            condition_shape = (batch, count, self.nhead, self.head_dim)
            key = self.k_norm(self.compact_k_proj(compact).view(condition_shape)).transpose(1, 2)
            value = self.compact_v_proj(compact).view(condition_shape).transpose(1, 2)
            # Visual states already carry their camera/spatial/time identity.
            # Do not invent spatial rotary coordinates for these projected keys.
            positions = count + torch.arange(horizon, device=hidden.device)[None].expand(batch, -1)
            query = _rope(query, positions, rotary_dim=self.rotary_dim, theta=self.rope_theta)
            action_key = _rope(action_key, positions, rotary_dim=self.rotary_dim, theta=self.rope_theta)
            key, value = torch.cat((key, action_key), dim=-2), torch.cat((value, action_value), dim=-2)
            valid = condition_mask(rows, device=hidden.device)
            if valid is not None:
                valid = F.pad(valid, (0, horizon), value=True)[:, None, None, :]
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=valid)
            hidden = hidden + self.o_proj(attended.transpose(1, 2).reshape(batch, horizon, -1))
            return hidden + self.ff(self.ff_norm(hidden) * (1 + self.ff_modulation(time)[:, None]))
        # K/V cache rows may repeat over transitions. Gather only small row
        # groups so the native B-sized cache is never copied N times at once.
        attended = torch.zeros_like(query)
        for context_index, context in enumerate(rows.contexts):
            selected = (rows.routes[:, 0] == context_index).nonzero().flatten()
            native_key, native_value = context.layers[layer]
            if self.nhead % native_key.shape[1]:
                raise ValueError("Expert query heads must be divisible by native K/V heads")
            for index in selected.split(16):
                route = rows.routes[index]
                source, source_query = route[:, 1], route[:, 2]
                position = context.positions[source, source_query][:, None] + 1 + torch.arange(horizon, device=hidden.device)[None]
                q = _rope(query[index], position, rotary_dim=self.rotary_dim, theta=self.rope_theta)
                ak = _rope(action_key[index], position, rotary_dim=self.rotary_dim, theta=self.rope_theta)
                groups = self.nhead // native_key.shape[1]
                key = native_key[source].to(query).repeat_interleave(groups, dim=1)
                value = native_value[source].to(query).repeat_interleave(groups, dim=1)
                key = torch.cat((key, ak), dim=-2)
                value = torch.cat((value, action_value[index]), dim=-2)
                mask = F.pad(context.allowed[source, source_query], (0, horizon), value=True)
                attended[index] = F.scaled_dot_product_attention(q, key, value, attn_mask=mask[:, None, None, :])
        hidden = hidden + self.o_proj(attended.transpose(1, 2).reshape(batch, horizon, -1))
        return hidden + self.ff(self.ff_norm(hidden) * (1 + self.ff_modulation(time)[:, None]))


class Pi05ActionExpert(FlowMatchingIDM):
    """Native insulated K/V flow, or explicit final-compact conditioning.

    ``pi05_compact_conditioning`` is a distinct opt-in architecture for token
    reduction. It projects the shared final conditions independently in every
    action block and propagates gradients into them. Native K/V is unchanged.
    """
    idm_type = "flow_matching"

    def __init__(self, *, native_kv_layers, native_head_dim, native_rotary_dim,
                 native_rope_theta, cond_dim, d_model=1024, nhead=8, ff_mult=4,
                 action_dim, action_horizon, n_cond_tokens, flow_time_embed_dim=128,
                 flow_sampling_steps=16, loss_type="mse", huber_delta=.1,
                 action_clip_abs_max=None, sampled_action_clip_abs_max=None,
                 robot_action_layout="", flow_output_projection="shared",
                 action_token_types=(), pi05_block_gradient_checkpointing=False,
                 pi05_compact_conditioning=False, **unused):
        nn.Module.__init__(self)
        if type(pi05_block_gradient_checkpointing) is not bool:
            raise ValueError("pi05_block_gradient_checkpointing must be a bool")
        if type(pi05_compact_conditioning) is not bool:
            raise ValueError("pi05_compact_conditioning must be a bool")
        self.pi05_block_gradient_checkpointing = pi05_block_gradient_checkpointing
        self.pi05_compact_conditioning = pi05_compact_conditioning
        self.cond_dim, self.hidden_dim = cond_dim, d_model
        self.action_dim, self.action_horizon = action_dim, action_horizon
        self.n_cond_tokens, self.flow_time_embed_dim = n_cond_tokens, flow_time_embed_dim
        self.flow_sampling_steps, self.flow_sampling_cache_static_kv = flow_sampling_steps, False
        self.flow_output_projection, self.action_token_types = validate_flow_output_projection(
            flow_output_projection, action_token_types=action_token_types, action_horizon=action_horizon)
        _configure_flow_regression_loss(self, loss_type=loss_type, huber_delta=huber_delta)
        _configure_action_clip(self, action_clip_abs_max)
        _configure_sampled_action_clip(self, sampled_action_clip_abs_max)
        self._configure_sampling_layout(robot_action_layout)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.time_proj = nn.Sequential(nn.Linear(flow_time_embed_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(_ActionBlock(d_model, nhead, native_head_dim, ff_mult,
            native_rotary_dim, native_rope_theta,
            compact_cond_dim=cond_dim if pi05_compact_conditioning else None)
            for _ in range(native_kv_layers))
        self.final_layer = AdaLNZeroFinalLayer(d_model, action_dim,
            flow_output_projection=flow_output_projection, action_token_types=self.action_token_types)

    def _vector_field(self, cond_list, noisy_actions, times, proprio_token=None):
        if proprio_token is not None:
            raise ValueError("pi05 has no direct proprio bypass")
        if self.pi05_compact_conditioning:
            if (not cond_list or any(value.ndim != 2 or value.shape != (noisy_actions.shape[0], self.cond_dim)
                                     for value in cond_list)):
                raise ValueError("Compact-conditioned pi05 requires shared final [B,D] visual conditions")
            if getattr(cond_list, "backbone_kv", None) is not None:
                raise ValueError("Compact-conditioned pi05 does not accept intermediate native backbone K/V")
            rows = cond_list
        else:
            rows = getattr(cond_list, "backbone_kv", None)
            if rows is None or rows.routes.shape[0] != noisy_actions.shape[0]:
                raise ValueError("pi05 requires routed native causal backbone K/V")
            if any(len(context.layers) != len(self.blocks) for context in rows.contexts):
                raise ValueError("Expert depth must cover every native full-attention layer")
            if any(key.requires_grad or value.requires_grad
                   for context in rows.contexts for key, value in context.layers):
                raise ValueError("Knowledge insulation requires detached native K/V")
        time = self.time_proj(timestep_embedding(times, self.flow_time_embed_dim))
        hidden = self.action_proj(noisy_actions)
        for index, block in enumerate(self.blocks):
            if self.pi05_block_gradient_checkpointing and self.training and torch.is_grad_enabled():
                # Include native K/V gather, head expansion and concatenation:
                # saving only the attention input would retain those copies.
                hidden = checkpoint(block, hidden, time, rows=rows, layer=index,
                    use_reentrant=False, preserve_rng_state=True, early_stop=False)
            else:
                hidden = block(hidden, time, rows, index)
        return self.final_layer(hidden, time,
            end_effector_type_ids=getattr(cond_list, "end_effector_type_ids", None))
