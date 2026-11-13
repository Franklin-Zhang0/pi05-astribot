from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from prism_vla.data.fast_tokenizer import FASTActionTokenizer
from prism_vla.data.fast_targets import (
    prepare_fast_targets, fast_semantic_mask, fast_token_validation_contract,
    trust_dataset_fast_targets, validate_fast_token_ids,
)
from prism_vla.modeling.action_layout import robot_action_sampling_mask
from prism_vla.modeling.config import (
    validate_flow_output_projection, validate_idm_attention_limits,
    validate_idm_recovery_controls,
)
from prism_vla.modeling.idm_qk_control import ControlledIDMSelfAttention
from prism_vla.modeling.idm_muonclip import (
    MuonClipMultiheadAttention, suspend_muonclip_observations,
)
from prism_vla.modeling.idm_attention_cache import CachedConditionMultiheadAttention
from prism_vla.modeling.fast_loss import bounded_fp32_cross_entropy_sum
from prism_vla.modeling.losses import masked_regression_loss, normalize_regression_loss_type
from prism_vla.modeling.condition_mask import (
    ConditionList, condition_mask, masked_condition_values, mask_projected_conditions,
    last_valid_condition, select_conditions, replace_conditions,
)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for scalar flow times in [0, 1]."""
    half = dim // 2
    device = t.device
    dtype = t.dtype
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def _expand_action_chunk_mask(
    action_chunk_valid_steps: Optional[torch.Tensor],
    *,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Expand per-sample valid horizon lengths into a broadcastable mask."""
    if action_chunk_valid_steps is None:
        return None

    valid_steps = action_chunk_valid_steps.to(device=device).long().clamp(min=0, max=horizon)
    step_ids = torch.arange(horizon, device=device).view(1, horizon, 1)
    mask = (step_ids < valid_steps.view(-1, 1, 1)).to(dtype=dtype)
    return mask.expand(-1, -1, action_dim)


def _configure_flow_regression_loss(
    module: nn.Module,
    *,
    loss_type: str,
    huber_delta: float,
) -> None:
    module.loss_type = normalize_regression_loss_type(loss_type)
    module.huber_delta = float(huber_delta)
    if module.huber_delta <= 0.0:
        raise ValueError(f"huber_delta must be positive, got {module.huber_delta}.")


def _configure_action_clip(module: nn.Module, value: Optional[float]) -> None:
    if value is None:
        module.action_clip_abs_max = None
        return
    bound = float(value)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(
            f"action_clip_abs_max must be finite and positive or null, got {value}"
        )
    module.action_clip_abs_max = bound


def _configure_sampled_action_clip(module: nn.Module, value: Optional[float]) -> None:
    """Bound generated actions independently of continuous training targets.

    Null retains the historical action_clip_abs_max sampling policy. This
    option never clips the flow target, noise bridge or regression objective.
    """
    if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
        raise ValueError("sampled_action_clip_abs_max must be finite and positive or null")
    module.sampled_action_clip_abs_max = None if value is None else float(value)


def _combine_action_chunk_masks(
    action_chunk_valid_steps: Optional[torch.Tensor],
    action_chunk_loss_mask: Optional[torch.Tensor],
    *,
    horizon: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build an elementwise action supervision mask."""
    step_mask = _expand_action_chunk_mask(
        action_chunk_valid_steps,
        horizon=horizon,
        action_dim=action_dim,
        device=device,
        dtype=dtype,
    )
    if action_chunk_loss_mask is None:
        return step_mask

    elem_mask = action_chunk_loss_mask.to(device=device, dtype=dtype)
    if elem_mask.ndim != 3:
        raise ValueError(
            f"Expected action_chunk_loss_mask [B, H, A], got {tuple(elem_mask.shape)}"
        )
    if int(elem_mask.shape[1]) < horizon or int(elem_mask.shape[2]) != action_dim:
        raise ValueError(
            "action_chunk_loss_mask shape mismatch: "
            f"got {tuple(elem_mask.shape)}, expected second dim >= {horizon} "
            f"and action_dim={action_dim}"
        )
    elem_mask = elem_mask[:, :horizon]
    if step_mask is None:
        return elem_mask
    return elem_mask * step_mask


def _modulate_layer_norm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _limit_modulation(value: torch.Tensor, limit: float | None) -> torch.Tensor:
    return value if limit is None else value.clamp(-limit, limit)


def _limit_condition_rms(value: torch.Tensor, limit: float | None) -> torch.Tensor:
    """Cap the combined latent/time conditioning per row, preserving healthy rows."""
    if limit is None:
        return value
    with torch.autocast(device_type=value.device.type, enabled=False):
        value32 = value.float()
        rms = torch.linalg.vector_norm(value32, dim=-1, keepdim=True) / math.sqrt(value.shape[-1])
        # Roundoff reserve only applies to rows actually exceeding the ceiling.
        target = limit * (1.0 - 2.0 * torch.finfo(value.dtype).eps)
        scale = torch.where(rms > limit, target / rms.clamp_min(1e-12), torch.ones_like(rms))
        return (value32 * scale).to(dtype=value.dtype)


class AdaLNZeroTransformerBlock(nn.Module):
    """DiT-style transformer block with adaptive LayerNorm zero conditioning."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_mult: int,
        *,
        idm_attention_scale_limit: float | None = None,
        idm_attention_shift_limit: float | None = None,
        idm_qk_control_mode: str = "none",
        idm_qk_logit_limit: float = 100.0,
        idm_qk_norm_temperature: float = 8.0,
        idm_attention_gate_limit: float | None = None,
        idm_mlp_scale_limit: float | None = None,
        idm_mlp_shift_limit: float | None = None,
        idm_mlp_gate_limit: float | None = None,
    ):
        super().__init__()
        self.idm_attention_scale_limit, self.idm_attention_shift_limit = validate_idm_attention_limits(
            idm_attention_scale_limit,
            idm_attention_shift_limit,
            idm_type="flow_matching",
            flow_idm_arch="temporal_transformer",
            flow_conditioning="adaln_zero",
        )
        controls = validate_idm_recovery_controls(
            idm_qk_control_mode=idm_qk_control_mode,
            idm_qk_logit_limit=idm_qk_logit_limit,
            idm_qk_norm_temperature=idm_qk_norm_temperature,
            idm_attention_gate_limit=idm_attention_gate_limit,
            idm_mlp_scale_limit=idm_mlp_scale_limit,
            idm_mlp_shift_limit=idm_mlp_shift_limit,
            idm_mlp_gate_limit=idm_mlp_gate_limit,
        )
        for name, value in controls.items():
            setattr(self, name, value)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if self.idm_qk_control_mode == "none":
            self.attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True, dropout=0.0)
        else:
            self.attn = ControlledIDMSelfAttention(
                d_model, nhead,
                qk_control_mode=self.idm_qk_control_mode,
                qk_logit_limit=self.idm_qk_logit_limit,
                qk_norm_temperature=self.idm_qk_norm_temperature,
            )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        # Attention deltas, residual gates and feed-forward modulation have
        # independent optional bounds; defaults preserve the original function.
        if self.idm_attention_scale_limit is not None:
            scale_msa = scale_msa.clamp(
                -self.idm_attention_scale_limit, self.idm_attention_scale_limit
            )
        if self.idm_attention_shift_limit is not None:
            shift_msa = shift_msa.clamp(
                -self.idm_attention_shift_limit, self.idm_attention_shift_limit
            )
        gate_msa = _limit_modulation(gate_msa, self.idm_attention_gate_limit)
        scale_mlp = _limit_modulation(scale_mlp, self.idm_mlp_scale_limit)
        shift_mlp = _limit_modulation(shift_mlp, self.idm_mlp_shift_limit)
        gate_mlp = _limit_modulation(gate_mlp, self.idm_mlp_gate_limit)
        attn_in = _modulate_layer_norm(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = _modulate_layer_norm(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class AdaLNZeroFinalLayer(nn.Module):
    """DiT final projection with adaptive LayerNorm conditioning."""

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        *,
        flow_output_projection: str = "shared",
        action_token_types: tuple[str, ...] = (),
        idm_final_scale_limit: float | None = None,
        idm_final_shift_limit: float | None = None,
    ):
        super().__init__()
        controls = validate_idm_recovery_controls(
            idm_final_scale_limit=idm_final_scale_limit,
            idm_final_shift_limit=idm_final_shift_limit,
        )
        self.idm_final_scale_limit = controls["idm_final_scale_limit"]
        self.idm_final_shift_limit = controls["idm_final_shift_limit"]
        self.flow_output_projection, self.action_token_types = validate_flow_output_projection(
            flow_output_projection, action_token_types=action_token_types
        )
        self.action_dim = int(action_dim)
        if self.flow_output_projection == "per_effector_type" and self.action_dim != 15:
            raise ValueError("per_effector_type requires the canonical 15-D action tokens")
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        if self.flow_output_projection == "shared":
            self.linear = nn.Linear(d_model, action_dim)
            linears = (self.linear,)
        else:
            self.linears = nn.ModuleDict({
                token_type: nn.Linear(d_model, action_dim)
                for token_type in dict.fromkeys(self.action_token_types)
            })
            if self.flow_output_projection == "per_effector_type":
                self.linears["gripper"] = nn.Linear(d_model, 1)
            self._type_indices = {
                token_type: tuple(
                    index for index, value in enumerate(self.action_token_types)
                    if value == token_type
                )
                for token_type in dict.fromkeys(self.action_token_types)
            }
            # Follow module device moves without copying Python indices to
            # CUDA at every gather/scatter. These derived indices are not
            # checkpoint state and do not change parameter ownership.
            self._type_index_buffer_names = {}
            for index, (token_type, indices) in enumerate(self._type_indices.items()):
                name = f"_type_index_{index}"
                self.register_buffer(name, torch.tensor(indices, dtype=torch.long), persistent=False)
                self._type_index_buffer_names[token_type] = name
            linears = self.linears.values()
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        for linear in linears:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, *,
                end_effector_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        shift = _limit_modulation(shift, self.idm_final_shift_limit)
        scale = _limit_modulation(scale, self.idm_final_scale_limit)
        x = _modulate_layer_norm(self.norm_final(x), shift, scale)
        if self.flow_output_projection == "shared":
            return self.linear(x)
        tokens_per_step = len(self.action_token_types)
        if x.shape[1] % tokens_per_step != 0:
            raise ValueError("Action sequence length must be divisible by len(action_token_types).")
        # Decode all horizons and both sides together, once per action type.
        x = x.reshape(x.shape[0], -1, tokens_per_step, x.shape[-1])
        output = None
        for token_type, buffer_name in self._type_index_buffer_names.items():
            linear = self.linears[token_type]
            indices = getattr(self, buffer_name)
            values = linear(x[:, :, indices, :])
            if token_type == "hand" and self.flow_output_projection == "per_effector_type":
                if end_effector_type_ids is None or tuple(end_effector_type_ids.shape) != (x.shape[0], 2):
                    raise ValueError("per_effector_type requires explicit left/right effector IDs [B, 2]")
                ids = end_effector_type_ids.to(device=x.device)
                if ids.dtype != torch.long:
                    raise ValueError("end_effector_type_ids must be int64")
                valid = ((ids == 0) | (ids == 1)).all()
                if ids.device.type == "cuda":
                    torch._assert_async(valid, "Unknown hand/gripper identity")
                elif not bool(valid):
                    raise ValueError("Unknown hand/gripper identity")
                if len(self._type_indices["hand"]) != 2:
                    raise ValueError("Effector routing requires two canonical hand slots")
                gripper = F.pad(self.linears["gripper"](x[:, :, indices, :]), (0, self.action_dim - 1))
                values = torch.where(ids[:, None, :, None] == 1, gripper, values)
            if output is None:
                output = values.new_empty(*x.shape[:3], values.shape[-1])
            output[:, :, indices, :] = values
        return output.flatten(1, 2)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.flow_output_projection == "per_effector_type":
            for suffix in ("weight", "bias"):
                target = prefix + "linears.gripper." + suffix
                source = prefix + "linears.hand." + suffix
                if target not in state_dict and source in state_dict:
                    state_dict[target] = state_dict[source][:1].clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class TimeAdaLNCrossAttentionTransformerBlock(nn.Module):
    """DiT block with zero-gated time AdaLN and latent condition cross-attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_mult: int,
        cross_attn_gate_init: float = 0.0,
        idm_qk_control_mode: str = "none",
        idm_qk_logit_limit: float = 100.0,
        cond_dim: int | None = None,
        gradient_checkpointing: bool = False,
        flow_qk_norm: bool = False,
    ):
        super().__init__()
        if type(gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a bool")
        self.gradient_checkpointing = gradient_checkpointing
        if idm_qk_control_mode not in {"none", "muonclip"}:
            raise ValueError("Cross-attention IDM supports none or post-update muonclip")
        if type(flow_qk_norm) is not bool:
            raise ValueError("flow_qk_norm must be a bool")
        if flow_qk_norm and idm_qk_control_mode != "muonclip":
            raise ValueError("flow_qk_norm currently requires cross-attention flow with muonclip")
        attention_options = dict(batch_first=True, dropout=0.0)
        attention_class = nn.MultiheadAttention
        if idm_qk_control_mode == "muonclip":
            attention_class = MuonClipMultiheadAttention
            attention_options["muonclip_logit_limit"] = idm_qk_logit_limit
            attention_options["flow_qk_norm"] = flow_qk_norm
        cond_dim = d_model if cond_dim is None else cond_dim
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = attention_class(
            d_model,
            num_heads=nhead,
            **attention_options,
        )
        self.norm_cross = nn.LayerNorm(d_model, eps=1e-6)
        self.cond_norm = nn.LayerNorm(cond_dim, eps=1e-6)
        self.cross_attn = (CachedConditionMultiheadAttention
            if idm_qk_control_mode == "none" else attention_class)(
            d_model,
            num_heads=nhead,
            kdim=cond_dim,
            vdim=cond_dim,
            **attention_options,
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 7 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        with torch.no_grad():
            self.adaLN_modulation[-1].bias[3 * d_model : 4 * d_model].fill_(
                float(cross_attn_gate_init)
            )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Recompute the cache-free FFN while retaining ordinary module hooks.

        Attention and its MuonClip observations execute only once. Complete
        recomputation preserves dedicated Linear post-forward hook boundaries,
        following the existing FAST FFN checkpoint implementation below.
        """
        if self.training and torch.is_grad_enabled() and not self.gradient_checkpointing:
            return checkpoint(
                self.mlp, x, use_reentrant=False,
                preserve_rng_state=True, early_stop=False,
            )
        return self.mlp(x)

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        time_cond: torch.Tensor,
        cond_key_padding_mask: torch.Tensor | None = None,
        static_kv_cache: dict | None = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            if static_kv_cache is not None:
                raise ValueError("Static condition K/V caching requires no-grad sampling")
            # This action block has no KDA or retained-stream state. Recompute
            # the complete module boundaries so DMuon hooks finish normally,
            # but do not repeat the detached MuonClip score observer. Avoid a
            # nested FFN checkpoint: this outer scope already saves its inputs.
            return checkpoint(
                self._forward_impl, x, cond_tokens, time_cond, cond_key_padding_mask,
                use_reentrant=False, preserve_rng_state=True, early_stop=False,
                context_fn=lambda: (nullcontext(), suspend_muonclip_observations()),
            )
        return self._forward_impl(x, cond_tokens, time_cond, cond_key_padding_mask, static_kv_cache)

    def _forward_impl(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        time_cond: torch.Tensor,
        cond_key_padding_mask: torch.Tensor | None = None,
        static_kv_cache: dict | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, gate_cross, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(time_cond).chunk(7, dim=-1)
        )
        attn_in = _modulate_layer_norm(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        cond_tokens = self.cond_norm(cond_tokens)
        if cond_key_padding_mask is None:
            cross_out, _ = self.cross_attn(
                self.norm_cross(x), cond_tokens, cond_tokens, need_weights=False,
                **({"static_kv_cache": static_kv_cache} if static_kv_cache is not None else {}),
            )
        else:
            empty = cond_key_padding_mask.all(dim=-1)
            first_key = torch.arange(cond_key_padding_mask.shape[-1], device=x.device) == 0
            safe_padding_mask = cond_key_padding_mask & ~(empty[:, None] & first_key[None])
            cond_tokens = torch.where(empty[:, None, None], 0, cond_tokens)
            cross_out, _ = self.cross_attn(
                self.norm_cross(x), cond_tokens, cond_tokens, need_weights=False,
                key_padding_mask=safe_padding_mask,
                **({"observation_key_padding_mask": cond_key_padding_mask}
                    if isinstance(self.cross_attn, MuonClipMultiheadAttention) else {}),
                **({"static_kv_cache": static_kv_cache} if static_kv_cache is not None else {}),
            )
            cross_out = torch.where(empty[:, None, None], 0, cross_out)
        x = x + gate_cross.unsqueeze(1) * cross_out

        mlp_in = _modulate_layer_norm(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self._ff_block(mlp_in)
        return x


class _FASTFFNCheckpointEncoderLayer(nn.TransformerEncoderLayer):
    """Recompute only FAST's cache-free FFN, leaving attention untouched."""

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            # Complete recompute so dedicated Linear forward hooks finish too.
            # Native DMuon recognizes backward recompute and skips forward
            # prefetch/order bookkeeping. Retain the ordinary FFN operation
            # order, autocast state and any future dropout RNG consumption.
            return checkpoint(
                super()._ff_block, x, use_reentrant=False,
                preserve_rng_state=True, early_stop=False,
            )
        return super()._ff_block(x)


class _FASTLayerCheckpointEncoderLayer(nn.TransformerEncoderLayer):
    """Recompute a cache-free FAST layer, including its unchanged attention."""

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            # Complete replay preserves MHA and Linear pre/post-hook pairs.
            # Only FAST's stateless encoder layer is covered: no Qwen/KDA
            # cache, optimizer, or persistent normalization state is replayed.
            return checkpoint(
                super().forward, src, src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask, is_causal=is_causal,
                use_reentrant=False, preserve_rng_state=True, early_stop=False,
            )
        return super().forward(
            src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )


class AutoregressiveIDM(nn.Module):
    """Small transformer that decodes FAST tokens autoregressively."""

    idm_type = "fast_ar"

    def consume_training_rng(self, **kwargs) -> None:
        """The teacher-forced FAST decoder has no stochastic layers."""
        return None

    def __init__(
        self,
        cond_dim: int = 2560,
        d_model: int = 512,
        nhead: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        fast_vocab_size: int = 2049,
        max_fast_len: int = 20,
        action_dim: int = 7,
        action_horizon: int = 10,
        fast_pretrained_path: str = "physical-intelligence/fast",
        n_cond_tokens: int = 4,
        action_token_types: tuple[str, ...] = (),
        robot_action_layout: str = "",
        fast_ffn_gradient_checkpointing: bool = False,
        fast_layer_gradient_checkpointing: bool = False,
        fast_broadcast_attention: bool = False,
    ):
        super().__init__()
        if type(fast_broadcast_attention) is not bool:
            raise ValueError("fast_broadcast_attention must be a bool")
        self.fast_broadcast_attention = fast_broadcast_attention
        if type(fast_ffn_gradient_checkpointing) is not bool:
            raise ValueError("fast_ffn_gradient_checkpointing must be a bool")
        if type(fast_layer_gradient_checkpointing) is not bool:
            raise ValueError("fast_layer_gradient_checkpointing must be a bool")
        if fast_ffn_gradient_checkpointing and fast_layer_gradient_checkpointing:
            raise ValueError("FAST FFN and whole-layer checkpointing are mutually exclusive")
        self.fast_ffn_gradient_checkpointing = fast_ffn_gradient_checkpointing
        self.fast_layer_gradient_checkpointing = fast_layer_gradient_checkpointing
        if int(n_cond_tokens) <= 0:
            raise ValueError("n_cond_tokens must be positive.")
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.fast_vocab_size = fast_vocab_size
        self.max_fast_len = max_fast_len
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.fast_pretrained_path = fast_pretrained_path
        self.n_cond_tokens = int(n_cond_tokens)
        self._trust_dataset_fast_targets = trust_dataset_fast_targets()
        if int(max_fast_len) <= 0:
            raise ValueError("max_fast_len must be positive")
        self.action_token_types = tuple(action_token_types)
        self.robot_action_layout = robot_action_layout
        self._semantic_mask = fast_semantic_mask(
            action_horizon, action_dim, self.action_token_types, robot_action_layout,
        )
        physical_horizon = action_horizon // len(self.action_token_types) if self.action_token_types else action_horizon
        physical_dim = action_dim * len(self.action_token_types) if self.action_token_types else action_dim

        self.fast_tokenizer = FASTActionTokenizer.from_pretrained(
            fast_pretrained_path,
            action_dim=physical_dim,
            action_horizon=physical_horizon,
        )
        if fast_vocab_size != self.fast_tokenizer.vocab_size:
            raise ValueError(
                f"fast_vocab_size={fast_vocab_size} differs from tokenizer vocabulary "
                f"{self.fast_tokenizer.vocab_size} including EOS"
            )

        self.cond_projs = nn.ModuleList([
            nn.Linear(cond_dim, d_model) for _ in range(self.n_cond_tokens)
        ])
        self.token_embed = nn.Embedding(fast_vocab_size, d_model)
        self.pos_embed = nn.Embedding(self.n_cond_tokens + max_fast_len, d_model)

        encoder_layer_type = (
            _FASTLayerCheckpointEncoderLayer if fast_layer_gradient_checkpointing else
            _FASTFFNCheckpointEncoderLayer if fast_ffn_gradient_checkpointing
            else nn.TransformerEncoderLayer
        )
        if fast_broadcast_attention:
            from prism_vla.modeling.fast_attention import (
                FASTBroadcastEncoderLayer, FASTBroadcastLayerCheckpointEncoderLayer,
                FASTBroadcastFFNCheckpointEncoderLayer,
            )
            encoder_layer_type = (FASTBroadcastLayerCheckpointEncoderLayer if fast_layer_gradient_checkpointing
                else FASTBroadcastFFNCheckpointEncoderLayer if fast_ffn_gradient_checkpointing
                else FASTBroadcastEncoderLayer)
        encoder_layer = encoder_layer_type(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, fast_vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.cond_projs.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)

    def _build_prefix(self, cond_list: list[torch.Tensor]) -> torch.Tensor:
        if len(cond_list) != self.n_cond_tokens:
            raise ValueError(
                f"Expected {self.n_cond_tokens} condition tensors, got {len(cond_list)}"
            )
        values = masked_condition_values(cond_list)
        prefix = torch.stack([proj(c) for proj, c in zip(self.cond_projs, values)], dim=1)
        return mask_projected_conditions(prefix, cond_list)

    def _masked_condition_transformer(self, sequence, sequence_valid):
        """Keep absent keys invisible without wholly masked padding queries.

        Some native evaluation attention paths produce NaNs for an invalid
        leading query with no permitted keys. Such query outputs are unused;
        give only those rows a private self-diagonal. Real query rows retain
        causal attention exclusively to real keys, including all action tokens.
        """
        length = sequence.shape[1]
        causal = torch.ones(length, length, device=sequence.device, dtype=torch.bool).tril()
        allowed = causal.unsqueeze(0) & sequence_valid.unsqueeze(1)
        diagonal = torch.eye(length, device=sequence.device, dtype=torch.bool).unsqueeze(0)
        allowed = torch.where(sequence_valid.unsqueeze(-1), allowed, diagonal)
        heads = self.transformer.layers[0].self_attn.num_heads
        if self.fast_broadcast_attention:
            # Memory-efficient CUDA SDPA requires aligned bias row strides.
            # Preserve the logical LxL mask while padding only its storage.
            aligned = (length + 7) // 8 * 8
            storage = sequence.new_full((*allowed.shape[:-1], aligned), float("-inf"))
            attention_mask = storage[..., :length]
            attention_mask.masked_fill_(allowed, 0)
            attention_mask = attention_mask[:, None]
        else:
            attention_mask = sequence.new_full(allowed.shape, float("-inf")).masked_fill(allowed, 0)
            attention_mask = attention_mask.repeat_interleave(heads, dim=0)
        return self.transformer(sequence, mask=attention_mask, is_causal=False)

    def forward(self, cond_list: list[torch.Tensor], fast_tokens: torch.Tensor,
                *, fast_token_validation: dict | None = None) -> torch.Tensor:
        if fast_tokens.ndim != 2 or fast_tokens.dtype != torch.long:
            raise ValueError("FAST targets must be a torch.long tensor [B, L]")
        bsz, seq_len = fast_tokens.shape
        if seq_len < 1 or seq_len > self.max_fast_len:
            raise ValueError(f"FAST target length {seq_len} must be in [1, {self.max_fast_len}]")
        if bsz != cond_list[0].shape[0]:
            raise ValueError("FAST targets and condition tensors have different batch sizes")
        if self._trust_dataset_fast_targets and fast_token_validation is not None:
            expected = fast_token_validation_contract(vocab_size=self.fast_vocab_size,
                eos_token_id=self.fast_tokenizer.eos_token_id, max_fast_len=self.max_fast_len)
            if fast_token_validation != expected:
                raise ValueError("FAST CPU validation contract differs from the decoder")
        else:
            validate_fast_token_ids(fast_tokens, vocab_size=self.fast_vocab_size,
                eos_token_id=self.fast_tokenizer.eos_token_id, max_fast_len=self.max_fast_len)
        device = fast_tokens.device
        valid_conditions = condition_mask(cond_list, device=device)
        bos_indices = last_valid_condition(cond_list, device=device)
        prefix = self._build_prefix(cond_list)
        safe_tokens = fast_tokens.clamp(min=0)
        # The final condition position is the BOS context. It predicts target
        # 0; target 0 predicts target 1, etc. No extra BOS action ID is needed.
        tok_embeds = self.token_embed(safe_tokens[:, :-1])
        seq = torch.cat([prefix, tok_embeds], dim=1)
        positions = torch.arange(seq.shape[1], device=device)
        seq = seq + self.pos_embed(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq.shape[1], device=device, dtype=seq.dtype,
        )
        if valid_conditions is None:
            hidden = self.transformer(seq, mask=causal_mask, is_causal=True)
        else:
            sequence_valid = torch.cat((valid_conditions,
                torch.ones(bsz, seq_len - 1, device=device, dtype=torch.bool)), dim=1)
            seq = torch.where(sequence_valid.unsqueeze(-1), seq, 0)
            hidden = self._masked_condition_transformer(seq, sequence_valid)
        hidden = self.ln_f(hidden)
        if bos_indices is None:
            prediction_hidden = hidden[:, self.n_cond_tokens - 1: self.n_cond_tokens - 1 + seq_len]
        else:
            read_indices = (torch.arange(seq_len, device=device) + self.n_cond_tokens - 1).expand(bsz, -1).clone()
            read_indices[:, 0] = bos_indices
            prediction_hidden = hidden.gather(1, read_indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logits = self.head(prediction_hidden)
        return logits

    def training_step(
        self,
        cond_list: list[torch.Tensor],
        *,
        fast_tokens: Optional[torch.Tensor] = None,
        action_chunks: Optional[torch.Tensor] = None,
        action_chunk_valid_steps: Optional[torch.Tensor] = None,
        action_chunk_loss_mask: Optional[torch.Tensor] = None,
        replan_horizon: Optional[int] = None,
        compute_action_metrics: bool = False,
        fast_token_validation: dict | None = None,
        action_semantic_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if action_semantic_mask is not None and (
                not torch.is_tensor(action_semantic_mask) or action_semantic_mask.dtype != torch.bool
                or tuple(action_semantic_mask.shape) != (cond_list[0].shape[0], self.action_horizon, self.action_dim)):
            raise ValueError("action_semantic_mask must be boolean [B, action_horizon, action_dim]")
        supervised_rows = None
        if fast_tokens is None:
            if action_chunks is None:
                raise ValueError("FAST training requires fast_tokens or head-local action_chunks")
            fast_tokens, supervised_rows = self.prepare_targets(
                action_chunks,
                action_chunk_valid_steps=action_chunk_valid_steps,
                action_chunk_loss_mask=action_chunk_loss_mask,
                action_semantic_mask=action_semantic_mask,
            )

        logits = self.forward(cond_list, fast_tokens, fast_token_validation=fast_token_validation)
        token_weight = (fast_tokens != -1).sum()
        loss = bounded_fp32_cross_entropy_sum(logits, fast_tokens) / token_weight.clamp_min(1)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = fast_tokens != -1
            if self._trust_dataset_fast_targets:
                # Keep accuracy present on every call without a GPU-to-host
                # mask.any() synchronization. The all-ignored value stays zero.
                action_acc = ((preds == fast_tokens) & mask).sum().float() / token_weight.clamp_min(1)
            elif mask.any():
                action_acc = (preds[mask] == fast_tokens[mask]).float().mean()
            else:
                action_acc = torch.zeros((), device=logits.device)

        metrics = {
            "loss": loss,
            "action_acc": action_acc,
            "action_loss_weight": token_weight.detach(),
        }
        if supervised_rows is None:
            supervised_rows = (fast_tokens != -1).any(dim=1)
        metrics["fast_supervised_chunks"] = supervised_rows.sum().detach()
        metrics["fast_total_chunks"] = token_weight.new_full((), fast_tokens.shape[0])
        metrics["fast_chunk_coverage"] = (
            supervised_rows.sum().float() / max(fast_tokens.shape[0], 1)
        ).detach()
        if compute_action_metrics and action_chunks is not None:
            pred_actions, decoded_valid = self._generate_action_chunks(cond_list)
            metrics["fast_decode_valid_fraction"] = decoded_valid.float().mean()
            horizon = min(replan_horizon or self.action_horizon, self.action_horizon)
            delta = pred_actions[:, :horizon] - action_chunks[:, :horizon]
            metric_mask = _combine_action_chunk_masks(
                action_chunk_valid_steps,
                action_chunk_loss_mask,
                horizon=horizon,
                action_dim=self.action_dim,
                device=delta.device,
                dtype=delta.dtype,
            )
            if metric_mask is None:
                metrics["action_chunk_mse"] = delta.square().mean()
                metrics["action_chunk_l1"] = delta.abs().mean()
            else:
                delta = torch.where(metric_mask > 0, delta, 0.0)
                denom = metric_mask.sum().clamp(min=1.0)
                metrics["action_chunk_mse"] = (delta.square() * metric_mask).sum() / denom
                metrics["action_chunk_l1"] = (delta.abs() * metric_mask).sum() / denom
        return metrics

    def prepare_targets(
        self,
        action_chunks: torch.Tensor,
        *,
        action_chunk_valid_steps: Optional[torch.Tensor] = None,
        action_chunk_loss_mask: Optional[torch.Tensor] = None,
        action_semantic_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build targets from this head's action representation and normalization."""
        if action_chunks.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("FAST head-local action chunk shape differs from configured H x A")
        return prepare_fast_targets(
            action_chunks, tokenizer=self.fast_tokenizer, max_fast_len=self.max_fast_len,
            action_chunk_valid_steps=action_chunk_valid_steps,
            action_chunk_loss_mask=action_chunk_loss_mask,
            action_token_types=self.action_token_types,
            robot_action_layout=self.robot_action_layout,
            action_semantic_mask=action_semantic_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        cond_list: list[torch.Tensor],
        max_len: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        max_len = self.max_fast_len if max_len is None else int(max_len)
        if max_len < 1 or max_len > self.max_fast_len:
            raise ValueError(f"FAST generation max_len must be in [1, {self.max_fast_len}]")
        bsz = cond_list[0].shape[0]
        device = cond_list[0].device
        eos_token_id = eos_token_id if eos_token_id is not None else self.fast_vocab_size - 1
        if eos_token_id != self.fast_tokenizer.eos_token_id:
            raise ValueError("FAST generation EOS must match the tokenizer")

        valid_conditions = condition_mask(cond_list, device=device)
        bos_indices = last_valid_condition(cond_list, device=device)
        prefix = self._build_prefix(cond_list)
        generated = torch.full((bsz, max_len), eos_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(bsz, device=device, dtype=torch.bool)
        seq = prefix

        for step in range(max_len):
            positions = torch.arange(seq.shape[1], device=device)
            seq_pos = seq + self.pos_embed(positions)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq.shape[1], device=device, dtype=seq.dtype,
            )
            if valid_conditions is None:
                hidden = self.transformer(seq_pos, mask=causal_mask, is_causal=True)
            else:
                sequence_valid = torch.cat((valid_conditions,
                    torch.ones(bsz, seq.shape[1] - self.n_cond_tokens, device=device, dtype=torch.bool)), dim=1)
                seq_pos = torch.where(sequence_valid.unsqueeze(-1), seq_pos, 0)
                hidden = self._masked_condition_transformer(seq_pos, sequence_valid)
            hidden = self.ln_f(hidden)
            if step == 0 and bos_indices is not None:
                context = hidden[torch.arange(bsz, device=device), bos_indices]
            else:
                context = hidden[:, -1]
            logits = self.head(context)
            next_token = logits.argmax(dim=-1)
            generated[:, step] = torch.where(finished, eos_token_id, next_token)
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break
            next_embed = self.token_embed(next_token).unsqueeze(1)
            seq = torch.cat([seq, next_embed], dim=1)
        return generated

    @torch.no_grad()
    def generate_actions(self, cond_list: list[torch.Tensor]) -> torch.Tensor:
        """Decode complete chunks; malformed predictions return a neutral chunk.

        Untrained autoregressive heads often lack EOS or valid coefficient
        lengths. Such predictions are not silently shortened or padded.
        Evaluation exposes fast_decode_valid_fraction for this fail-closed path.
        """
        return self._generate_action_chunks(cond_list)[0]

    @torch.no_grad()
    def _generate_action_chunks(self, cond_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = self.generate(cond_list)
        actions = []
        valid = []
        for sample in token_ids.detach().cpu().tolist():
            try:
                if self.fast_tokenizer.eos_token_id not in sample:
                    raise ValueError("FAST prediction reached max_fast_len without EOS")
                action_chunk = self.fast_tokenizer.decode(sample, strict=True)
                action_chunk = action_chunk.reshape(self.action_horizon, self.action_dim)
                valid.append(True)
            except (ValueError, OverflowError):
                action_chunk = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
                valid.append(False)
            action_chunk = np.where(self._semantic_mask.numpy(), action_chunk, 0.0)
            actions.append(np.clip(action_chunk, -1.0, 1.0))
        return (
            torch.as_tensor(np.stack(actions), device=cond_list[0].device, dtype=cond_list[0].dtype),
            torch.tensor(valid, device=cond_list[0].device),
        )


class ParallelL1ActionHead(nn.Module):
    """OFT-style residual MLP predicting an entire continuous chunk in parallel.

    Inspired by OpenVLA-OFT's L1RegressionActionHead (MIT):
    https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py
    Unlike that recipe's action slots inside the VLM, this adaptation uses the
    same compact condition tensors as every other PRISM head. It adds no VLM
    tokens and no iterative sampling. The loss is always scalar-masked L1.
    """

    idm_type = "oft"

    def consume_training_rng(self, **kwargs) -> None:
        """The parallel action decoder consumes no training random draws."""
        return None

    def __init__(
        self,
        cond_dim: int = 2560,
        d_model: int = 512,
        n_layers: int = 2,
        action_dim: int = 15,
        action_horizon: int = 160,
        n_cond_tokens: int = 4,
        action_clip_abs_max: float | None = None,
        robot_action_layout: str = "",
    ):
        super().__init__()
        if min(cond_dim, d_model, n_layers, action_dim, action_horizon, n_cond_tokens) < 1:
            raise ValueError("OFT head dimensions and layer count must be positive")
        self.cond_dim = cond_dim
        self.n_cond_tokens = n_cond_tokens
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.loss_type = "l1"
        _configure_action_clip(self, action_clip_abs_max)
        self.input_norm = nn.LayerNorm(cond_dim * n_cond_tokens)
        self.input_proj = nn.Linear(cond_dim * n_cond_tokens, d_model)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU())
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, action_horizon * action_dim)
        sampling_mask = robot_action_sampling_mask(
            robot_action_layout, action_horizon=action_horizon, action_dim=action_dim,
        )
        self.register_buffer("sampling_active_mask", sampling_mask, persistent=False)

    def forward(self, cond_list: list[torch.Tensor]) -> torch.Tensor:
        if len(cond_list) != self.n_cond_tokens:
            raise ValueError(f"OFT expected {self.n_cond_tokens} condition tensors")
        shape = (cond_list[0].shape[0], self.cond_dim)
        if any(tuple(condition.shape) != shape for condition in cond_list):
            raise ValueError(f"OFT condition tensors must all have shape {shape}")
        hidden = F.relu(self.input_proj(self.input_norm(torch.cat(masked_condition_values(cond_list), dim=-1))))
        for block in self.res_blocks:
            hidden = hidden + block(hidden)
        return self.output_proj(self.output_norm(hidden)).reshape(-1, self.action_horizon, self.action_dim)

    def training_step(
        self,
        cond_list: list[torch.Tensor],
        *,
        fast_tokens: torch.Tensor | None = None,
        action_chunks: torch.Tensor | None,
        action_chunk_valid_steps: torch.Tensor | None = None,
        action_chunk_loss_mask: torch.Tensor | None = None,
        replan_horizon: int | None = None,
        compute_action_metrics: bool = False,
    ) -> dict[str, torch.Tensor]:
        if action_chunks is None:
            raise ValueError("OFT training requires head-local action_chunks")
        prediction = self.forward(cond_list).float()
        if action_chunks.shape != prediction.shape:
            raise ValueError("OFT action target shape differs from configured chunk")
        target = action_chunks.to(device=prediction.device, dtype=torch.float32)
        mask = _combine_action_chunk_masks(
            action_chunk_valid_steps, action_chunk_loss_mask,
            horizon=self.action_horizon, action_dim=self.action_dim,
            device=prediction.device, dtype=prediction.dtype,
        )
        if mask is not None:
            target = torch.where(mask > 0, target, 0.0)
        loss = masked_regression_loss(prediction, target, mask=mask, loss_type="l1")
        weight = mask.sum() if mask is not None else prediction.new_full((), prediction.numel())
        metrics = {"loss": loss, "action_loss_weight": weight.detach(), "action_acc": loss.new_zeros(())}
        if compute_action_metrics:
            horizon = min(self.action_horizon, replan_horizon or self.action_horizon)
            metric_prediction = prediction[:, :horizon]
            if self.action_clip_abs_max is not None:
                metric_prediction = metric_prediction.clamp(-self.action_clip_abs_max, self.action_clip_abs_max)
            metric_mask = None if mask is None else mask[:, :horizon]
            metrics["action_chunk_l1"] = masked_regression_loss(
                metric_prediction, target[:, :horizon], mask=metric_mask, loss_type="l1",
            )
            metrics["action_chunk_mse"] = masked_regression_loss(
                metric_prediction, target[:, :horizon], mask=metric_mask, loss_type="mse",
            )
        return metrics

    @torch.no_grad()
    def generate_actions(self, cond_list: list[torch.Tensor]) -> torch.Tensor:
        prediction = self.forward(cond_list)
        if self.sampling_active_mask is not None:
            prediction = prediction.masked_fill(~self.sampling_active_mask, 0.0)
        if self.action_clip_abs_max is not None:
            prediction = prediction.clamp(-self.action_clip_abs_max, self.action_clip_abs_max)
        return prediction


class FlowMatchingIDM(nn.Module):
    """Conditional flow-matching IDM over raw action chunks."""

    idm_type = "flow_matching"
    flow_idm_arch = "mlp"

    def consume_training_rng(self, *, action_chunks, dtype, **kwargs) -> None:
        """Retain the disabled GT endpoint's flow draws before predicted flow.

        Weak-supervision and metric rollouts retain the complete endpoint
        forward instead. This method mirrors the ordinary bridge draw order.
        """
        if action_chunks is None:
            raise ValueError("action_chunks are required for flow_matching IDM training")
        target = action_chunks.to(dtype=dtype)
        torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        torch.randn_like(target)

    def __init__(
        self,
        cond_dim: int = 2560,
        d_model: int = 512,
        nhead: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        fast_vocab_size: int = 2049,
        max_fast_len: int = 20,
        action_dim: int = 7,
        action_horizon: int = 10,
        flow_time_embed_dim: int = 128,
        flow_sampling_steps: int = 16,
        n_cond_tokens: int = 4,
        loss_type: str = "mse",
        huber_delta: float = 0.1,
        action_clip_abs_max: Optional[float] = 1.0,
        sampled_action_clip_abs_max: Optional[float] = None,
        robot_action_layout: str = "",
    ):
        super().__init__()
        if int(n_cond_tokens) <= 0:
            raise ValueError("n_cond_tokens must be positive.")
        self.cond_dim = cond_dim
        self.hidden_dim = d_model
        self.fast_vocab_size = fast_vocab_size
        self.max_fast_len = max_fast_len
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flow_time_embed_dim = flow_time_embed_dim
        self.flow_sampling_steps = flow_sampling_steps
        self.action_size = action_dim * action_horizon
        self.n_cond_tokens = int(n_cond_tokens)
        _configure_flow_regression_loss(
            self,
            loss_type=loss_type,
            huber_delta=huber_delta,
        )
        _configure_action_clip(self, action_clip_abs_max)
        _configure_sampled_action_clip(self, sampled_action_clip_abs_max)
        self._configure_sampling_layout(robot_action_layout)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim * self.n_cond_tokens, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.state_proj = nn.Linear(self.action_size, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(flow_time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * ff_mult),
                nn.SiLU(),
                nn.Linear(d_model * ff_mult, d_model),
            )
            for _ in range(max(n_layers, 1))
        ])
        self.out = nn.Linear(d_model, self.action_size)
        self._init_weights()

    def _configure_sampling_layout(self, robot_action_layout: str) -> None:
        self.robot_action_layout = robot_action_layout
        self.register_buffer(
            "sampling_active_mask",
            robot_action_sampling_mask(
                robot_action_layout,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                action_token_types=getattr(self, "action_token_types", ()),
            ),
            persistent=False,
        )

    def _init_weights(self) -> None:
        for module in [
            self.cond_proj[0],
            self.state_proj,
            self.time_proj[0],
            self.time_proj[2],
            self.out,
        ]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for block in self.res_blocks:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def _vector_field(
        self,
        cond_list: list[torch.Tensor],
        noisy_actions: torch.Tensor,
        times: torch.Tensor,
        proprio_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if proprio_token is not None:
            raise ValueError(
                "proprio_token is only supported by the temporal flow-matching IDM."
            )
        if len(cond_list) != self.n_cond_tokens:
            raise ValueError(
                f"Expected {self.n_cond_tokens} condition tensors, got {len(cond_list)}"
            )
        cond = torch.cat(masked_condition_values(cond_list), dim=-1)
        hidden = self.cond_proj(cond)
        hidden = hidden + self.state_proj(noisy_actions.reshape(noisy_actions.shape[0], -1))
        hidden = hidden + self.time_proj(timestep_embedding(times, self.flow_time_embed_dim))
        for block in self.res_blocks:
            hidden = hidden + block(hidden)
        return self.out(F.silu(hidden)).reshape(-1, self.action_horizon, self.action_dim)

    def training_step(
        self,
        cond_list: list[torch.Tensor],
        *,
        fast_tokens: Optional[torch.Tensor] = None,
        action_chunks: Optional[torch.Tensor],
        action_chunk_valid_steps: Optional[torch.Tensor] = None,
        action_chunk_loss_mask: Optional[torch.Tensor] = None,
        ego_weak_supervision: Optional[torch.Tensor] = None,
        replan_horizon: Optional[int] = None,
        compute_action_metrics: bool = False,
        proprio_token: Optional[torch.Tensor] = None,
        flow_times: Optional[torch.Tensor] = None,
        flow_noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Native flow loss with optional deterministic held-out bridge draws.

        Explicit draws match the effective action dtype/device and use FP32
        error/count reductions. Default training draws/arithmetic are unchanged.
        """
        if action_chunks is None:
            raise ValueError("action_chunks are required for flow_matching IDM training")

        action_chunks = action_chunks.to(cond_list[0].dtype)
        if (flow_times is None) != (flow_noise is None):
            raise ValueError("flow_times and flow_noise must be provided together")
        if flow_times is not None:
            for name, value, shape in (
                ("flow_times", flow_times, (action_chunks.shape[0],)),
                ("flow_noise", flow_noise, tuple(action_chunks.shape)),
            ):
                if not torch.is_tensor(value) or tuple(value.shape) != shape:
                    raise ValueError(f"{name} must be a tensor with shape {shape}")
                if value.device != action_chunks.device or value.dtype != action_chunks.dtype:
                    raise ValueError(f"{name} must match action_chunks device and dtype")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{name} must contain only finite values")
            if not bool(((flow_times >= 0) & (flow_times <= 1)).all()):
                raise ValueError("flow_times must lie in [0, 1]")
        if ego_weak_supervision is not None and action_chunk_loss_mask is not None:
            weak_rows = (ego_weak_supervision[..., 216] > 0).any(dim=-1)
            action_chunks = torch.where(
                weak_rows[:, None, None] & ~(action_chunk_loss_mask > 0),
                torch.zeros_like(action_chunks), action_chunks,
            )
        if flow_times is None:
            times = torch.rand(action_chunks.shape[0], device=action_chunks.device, dtype=action_chunks.dtype)
            noise = torch.randn_like(action_chunks)
        else:
            times, noise = flow_times, flow_noise
        xt = (1.0 - times).view(-1, 1, 1) * noise + times.view(-1, 1, 1) * action_chunks
        if ego_weak_supervision is not None:
            if self.action_dim != 15 or self.action_horizon % 5:
                raise ValueError("Ego weak supervision requires five-token rotation-6D actions")
            if action_chunk_loss_mask is None:
                raise ValueError("Ego weak supervision requires an explicit direct-supervision mask")
            # Unknown action coordinates must not reveal arbitrary zero pseudo-labels.
            weak_rows = (ego_weak_supervision[..., 216] > 0).any(dim=-1)
            observed = action_chunk_loss_mask > 0
            xt = torch.where(weak_rows[:, None, None] & ~observed, noise, xt)
        target_velocity = action_chunks - noise
        pred_velocity = self._vector_field(
            cond_list,
            xt,
            times,
            proprio_token=proprio_token,
        )
        loss_mask = _combine_action_chunk_masks(
            action_chunk_valid_steps,
            action_chunk_loss_mask,
            horizon=self.action_horizon,
            action_dim=self.action_dim,
            device=action_chunks.device,
            dtype=action_chunks.dtype,
        )
        loss = masked_regression_loss(
            pred_velocity.float() if flow_times is not None else pred_velocity,
            target_velocity.float() if flow_times is not None else target_velocity,
            mask=(loss_mask.float() if flow_times is not None and loss_mask is not None else loss_mask),
            loss_type=self.loss_type,
            huber_delta=self.huber_delta,
        )
        if flow_times is not None:
            action_loss_weight = (loss_mask.float().sum() if loss_mask is not None
                else loss.new_full((), pred_velocity.numel()))
        else:
            action_loss_weight = (loss_mask.sum() if loss_mask is not None
                else pred_velocity.new_full((), pred_velocity.numel()))

        weak_loss = loss.new_zeros(())
        weak_count = loss.new_zeros(())
        payload = None
        selected = None
        if ego_weak_supervision is not None:
            payload = ego_weak_supervision.float().clone()
            if action_chunk_valid_steps is not None:
                steps = torch.arange(payload.shape[1], device=payload.device)
                valid = steps[None] < (action_chunk_valid_steps[:, None] // 5)
                payload[..., 64:66] *= valid[..., None]
            selected = ((payload[..., 64:66] > 0) & (payload[..., 216:217] > 0)).any(dim=(1, 2))

        has_local_weak = selected is not None and bool(selected.any())
        run_weak_rollout = has_local_weak
        if bool(getattr(self, "_hsdp_evaluation", False)):
            # Evaluation can mix directly supervised and weakly supervised
            # sources. Every rank must enter the same dedicated-parameter
            # forwards, even when it has no weak payload or valid weak poses.
            if not torch.distributed.is_initialized():
                raise RuntimeError("HSDP weak evaluation requires an initialized process group")
            any_weak = torch.tensor(
                int(has_local_weak), device=cond_list[0].device, dtype=torch.int64,
            )
            torch.distributed.all_reduce(any_weak, op=torch.distributed.ReduceOp.MAX)
            run_weak_rollout = bool(any_weak.item())

        if run_weak_rollout:
            # A separate rollout starts from noise, with no unknown target
            # endpoint or future observed pose as model input. The evaluator
            # guarantees nonempty local transitions; a dummy row only matches
            # the forward schedule and never contributes loss or coverage.
            rollout_rows = selected if has_local_weak else slice(0, 1)
            predicted = self._sample_actions(
                select_conditions(cond_list, rollout_rows),
                proprio_token=proprio_token[rollout_rows] if proprio_token is not None else None,
                sampling_steps=int(getattr(self, "weak_flow_sampling_steps", 4)),
            )
            if has_local_weak:
                from prism_vla.modeling.ego_weak_supervision import camera_relative_pose_loss

                weak_sum, pose_count = camera_relative_pose_loss(predicted, payload[selected])
                weak_count = pose_count * 9.0
                weak_loss = weak_sum / pose_count.clamp(min=1.0)
                total_count = action_loss_weight.float() + weak_count
                loss = (loss.float() * action_loss_weight + weak_sum * 9.0) / total_count.clamp(min=1.0)
                action_loss_weight = total_count

        metrics = {
            "loss": loss,
            "weak_pose_loss": weak_loss.detach(),
            "weak_pose_weight": weak_count.detach(),
            "action_acc": action_chunks.new_zeros(()),
            "action_loss_weight": action_loss_weight.detach(),
        }
        if compute_action_metrics:
            pred_actions = self.generate_actions(
                cond_list,
                proprio_token=proprio_token,
            )
            horizon = min(replan_horizon or self.action_horizon, self.action_horizon)
            delta = pred_actions[:, :horizon] - action_chunks[:, :horizon]
            metric_mask = _combine_action_chunk_masks(
                action_chunk_valid_steps,
                action_chunk_loss_mask,
                horizon=horizon,
                action_dim=self.action_dim,
                device=action_chunks.device,
                dtype=action_chunks.dtype,
            )
            if metric_mask is None:
                metrics["action_chunk_mse"] = delta.square().mean()
                metrics["action_chunk_l1"] = delta.abs().mean()
            else:
                denom = metric_mask.sum().clamp(min=1.0)
                metrics["action_chunk_mse"] = (delta.square() * metric_mask).sum() / denom
                metrics["action_chunk_l1"] = (delta.abs() * metric_mask).sum() / denom
        return metrics

    @torch.no_grad()
    def generate_actions(self, cond_list: list[torch.Tensor], **kwargs) -> torch.Tensor:
        return self._sample_actions(cond_list, **kwargs)

    def _sample_actions(
        self,
        cond_list: list[torch.Tensor],
        *,
        initial_noise: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_cond_list: Optional[list[torch.Tensor]] = None,
        proprio_token: Optional[torch.Tensor] = None,
        uncond_proprio_token: Optional[torch.Tensor] = None,
        sampling_steps: Optional[int] = None,
        cache_static_kv: Optional[bool] = None,
    ) -> torch.Tensor:
        device = cond_list[0].device
        dtype = cond_list[0].dtype
        guidance_scale = float(guidance_scale)
        if not math.isfinite(guidance_scale):
            raise ValueError(f"guidance_scale must be finite, got {guidance_scale}")
        if proprio_token is not None:
            proprio_token = proprio_token.to(device=device, dtype=dtype)
        if guidance_scale != 1.0:
            if uncond_cond_list is None:
                uncond_cond_list = replace_conditions(cond_list, [torch.zeros_like(cond) for cond in cond_list])
            elif condition_mask(cond_list) is not None and not isinstance(uncond_cond_list, ConditionList):
                uncond_cond_list = replace_conditions(cond_list, uncond_cond_list)
            if len(uncond_cond_list) != len(cond_list):
                raise ValueError(
                    f"Expected {len(cond_list)} unconditional condition tensors, "
                    f"got {len(uncond_cond_list)}"
                )
            for idx, (cond, uncond_cond) in enumerate(zip(cond_list, uncond_cond_list)):
                if tuple(uncond_cond.shape) != tuple(cond.shape):
                    raise ValueError(
                        "Unconditional condition tensor shape mismatch at index "
                        f"{idx}: got {tuple(uncond_cond.shape)}, expected {tuple(cond.shape)}"
                    )
                uncond_cond_list[idx] = uncond_cond.to(device=device, dtype=dtype)
            if proprio_token is not None:
                if uncond_proprio_token is None:
                    # Proprio is part of the denoising token sequence in this
                    # mode, so CFG drops latent conditions but preserves state.
                    uncond_proprio_token = proprio_token
                if tuple(uncond_proprio_token.shape) != tuple(proprio_token.shape):
                    raise ValueError(
                        "Unconditional proprio token shape mismatch: got "
                        f"{tuple(uncond_proprio_token.shape)}, expected "
                        f"{tuple(proprio_token.shape)}"
                    )
                uncond_proprio_token = uncond_proprio_token.to(
                    device=device,
                    dtype=dtype,
                )
            elif uncond_proprio_token is not None:
                raise ValueError(
                    "uncond_proprio_token requires a conditioned proprio_token."
                )
        steps = max(self.flow_sampling_steps if sampling_steps is None else sampling_steps, 1)
        expected_shape = (cond_list[0].shape[0], self.action_horizon, self.action_dim)
        if initial_noise is None:
            xt = torch.randn(*expected_shape, device=device, dtype=dtype)
        else:
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} does not match "
                    f"expected {expected_shape}"
                )
            xt = initial_noise.to(device=device, dtype=dtype)
        initial_state = xt
        active_mask = self.sampling_active_mask
        if active_mask is not None:
            active_mask = active_mask.to(device=device)
        if getattr(self, "flow_output_projection", "shared") == "per_effector_type":
            ids = getattr(cond_list, "end_effector_type_ids", None)
            if ids is None or tuple(ids.shape) != (xt.shape[0], 2):
                raise ValueError("per_effector_type sampling requires effector IDs [B, 2]")
            step_tokens = len(self.action_token_types)
            token_index = torch.arange(self.action_horizon, device=device) % step_tokens
            grip = ((token_index[None, :] == 2) & (ids[:, :1].to(device) == 1)
                | (token_index[None, :] == 4) & (ids[:, 1:].to(device) == 1))
            effector_mask = ~(grip[:, :, None] & (torch.arange(self.action_dim, device=device)[None, None, :] > 0))
            active_mask = effector_mask if active_mask is None else active_mask & effector_mask
        time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
        use_static_kv = (getattr(self, "flow_sampling_cache_static_kv", True)
            if cache_static_kv is None else cache_static_kv)
        # An installed CUDA-graph/compiled vector field owns its own execution
        # contract. Do not replace or bypass it to introduce this cache.
        use_static_kv = (use_static_kv and not torch.is_grad_enabled()
            and getattr(self._vector_field, "__func__", None)
                is TemporalCrossAttentionFlowMatchingIDM._vector_field)
        condition_caches = ([{} for _ in self.blocks] if use_static_kv else None)
        unconditional_caches = ([{} for _ in self.blocks]
            if use_static_kv and guidance_scale != 1.0 else None)
        for idx in range(steps):
            # A tensor fill value in torch.full would extract a CUDA scalar to the host.
            t = time_grid[idx:idx + 1].expand(xt.shape[0]).contiguous()
            if active_mask is not None:
                # Match training's zero-target bridge before BOTH CFG branches;
                # reproject each step to avoid low-precision integration drift.
                xt = torch.where(active_mask, xt, (1.0 - t[:, None, None]) * initial_state)
            velocity = self._vector_field(
                cond_list,
                xt,
                t,
                proprio_token=proprio_token,
                **({"static_kv_caches": condition_caches} if condition_caches is not None else {}),
            )
            if guidance_scale != 1.0:
                assert uncond_cond_list is not None
                uncond_velocity = self._vector_field(
                    uncond_cond_list,
                    xt,
                    t,
                    proprio_token=uncond_proprio_token,
                    **({"static_kv_caches": unconditional_caches} if unconditional_caches is not None else {}),
                )
                velocity = uncond_velocity + guidance_scale * (velocity - uncond_velocity)
            if active_mask is not None:
                velocity = torch.where(active_mask, velocity, -initial_state)
            dt = time_grid[idx + 1] - time_grid[idx]
            xt = xt + dt * velocity
        if active_mask is not None:
            xt = xt.masked_fill(~active_mask, 0.0)
        bound = self.sampled_action_clip_abs_max
        if bound is None:
            bound = self.action_clip_abs_max
        return xt if bound is None else xt.clamp(-bound, bound)

    def _append_proprio_action_token(
        self,
        hidden: torch.Tensor,
        proprio_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        enabled = bool(getattr(self, "proprio_as_action_token", False))
        if not enabled:
            if proprio_token is not None:
                raise ValueError(
                    "Received proprio_token but this IDM was not configured with "
                    "proprio_as_action_token=True."
                )
            return hidden
        if proprio_token is None:
            raise ValueError(
                "proprio_token is required when proprio_as_action_token=True."
            )
        if proprio_token.ndim != 2:
            raise ValueError(
                "Expected proprio_token [B, cond_dim], got "
                f"shape={tuple(proprio_token.shape)}."
            )
        if proprio_token.shape[0] != hidden.shape[0] or proprio_token.shape[1] != self.cond_dim:
            raise ValueError(
                "Proprio token shape is incompatible with the action-token sequence: "
                f"got {tuple(proprio_token.shape)}, expected "
                f"({hidden.shape[0]}, {self.cond_dim})."
            )
        if self.proprio_token_proj is None or self.proprio_token_pos_embed is None:
            raise RuntimeError("Proprio action-token modules are not initialized.")
        token = self.proprio_token_proj(proprio_token).unsqueeze(1)
        token = token + self.proprio_token_pos_embed.to(dtype=token.dtype)
        return torch.cat([hidden, token.to(dtype=hidden.dtype)], dim=1)

    @torch.no_grad()
    def generate(
        self,
        cond_list: list[torch.Tensor],
        max_len: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "FlowMatchingIDM operates in continuous action space; use generate_actions() instead."
        )


class TemporalFlowMatchingIDM(FlowMatchingIDM):
    """Flow-matching IDM with DiT-style AdaLN-Zero conditioning."""

    flow_idm_arch = "temporal_transformer"
    flow_conditioning = "adaln_zero"

    def __init__(
        self,
        cond_dim: int = 2560,
        d_model: int = 512,
        nhead: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        fast_vocab_size: int = 2049,
        max_fast_len: int = 20,
        action_dim: int = 7,
        action_horizon: int = 10,
        flow_time_embed_dim: int = 128,
        flow_sampling_steps: int = 16,
        n_cond_tokens: int = 4,
        loss_type: str = "mse",
        huber_delta: float = 0.1,
        action_clip_abs_max: Optional[float] = 1.0,
        sampled_action_clip_abs_max: Optional[float] = None,
        proprio_as_action_token: bool = False,
        flow_output_projection: str = "shared",
        action_token_types: tuple[str, ...] = (),
        robot_action_layout: str = "",
        idm_attention_scale_limit: float | None = None,
        idm_attention_shift_limit: float | None = None,
        idm_qk_control_mode: str = "none",
        idm_qk_logit_limit: float = 100.0,
        idm_qk_norm_temperature: float = 8.0,
        idm_condition_rms_limit: float | None = None,
        idm_attention_gate_limit: float | None = None,
        idm_mlp_scale_limit: float | None = None,
        idm_mlp_shift_limit: float | None = None,
        idm_mlp_gate_limit: float | None = None,
        idm_final_scale_limit: float | None = None,
        idm_final_shift_limit: float | None = None,
    ):
        nn.Module.__init__(self)
        self.idm_attention_scale_limit, self.idm_attention_shift_limit = validate_idm_attention_limits(
            idm_attention_scale_limit,
            idm_attention_shift_limit,
            idm_type="flow_matching",
            flow_idm_arch="temporal_transformer",
            flow_conditioning="adaln_zero",
        )
        controls = validate_idm_recovery_controls(
            idm_qk_control_mode=idm_qk_control_mode,
            idm_qk_logit_limit=idm_qk_logit_limit,
            idm_qk_norm_temperature=idm_qk_norm_temperature,
            idm_condition_rms_limit=idm_condition_rms_limit,
            idm_attention_gate_limit=idm_attention_gate_limit,
            idm_mlp_scale_limit=idm_mlp_scale_limit,
            idm_mlp_shift_limit=idm_mlp_shift_limit,
            idm_mlp_gate_limit=idm_mlp_gate_limit,
            idm_final_scale_limit=idm_final_scale_limit,
            idm_final_shift_limit=idm_final_shift_limit,
        )
        for name, value in controls.items():
            setattr(self, name, value)
        self.flow_output_projection, self.action_token_types = validate_flow_output_projection(
            flow_output_projection,
            action_token_types=action_token_types,
            action_horizon=action_horizon,
        )
        if int(n_cond_tokens) <= 0:
            raise ValueError("n_cond_tokens must be positive.")
        self.cond_dim = cond_dim
        self.hidden_dim = d_model
        self.fast_vocab_size = fast_vocab_size
        self.max_fast_len = max_fast_len
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flow_time_embed_dim = flow_time_embed_dim
        self.flow_sampling_steps = flow_sampling_steps
        self.action_size = action_dim * action_horizon
        self.n_cond_tokens = int(n_cond_tokens)
        self.proprio_as_action_token = bool(proprio_as_action_token)
        _configure_flow_regression_loss(
            self,
            loss_type=loss_type,
            huber_delta=huber_delta,
        )
        _configure_action_clip(self, action_clip_abs_max)
        _configure_sampled_action_clip(self, sampled_action_clip_abs_max)
        self._configure_sampling_layout(robot_action_layout)

        self.cond_projs = nn.ModuleList([
            nn.Linear(cond_dim, d_model) for _ in range(self.n_cond_tokens)
        ])
        self.cond_mixer = nn.Sequential(
            nn.Linear(d_model * self.n_cond_tokens, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_proj = nn.Linear(action_dim, d_model)
        self.proprio_token_proj = (
            nn.Linear(cond_dim, d_model)
            if self.proprio_as_action_token
            else None
        )
        self.proprio_token_pos_embed = (
            nn.Parameter(torch.empty(1, 1, d_model))
            if self.proprio_as_action_token
            else None
        )
        self.time_proj = nn.Sequential(
            nn.Linear(flow_time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embed = nn.Embedding(action_horizon, d_model)
        self.blocks = nn.ModuleList([
            AdaLNZeroTransformerBlock(
                d_model, nhead, ff_mult,
                idm_attention_scale_limit=self.idm_attention_scale_limit,
                idm_attention_shift_limit=self.idm_attention_shift_limit,
                idm_qk_control_mode=self.idm_qk_control_mode,
                idm_qk_logit_limit=self.idm_qk_logit_limit,
                idm_qk_norm_temperature=self.idm_qk_norm_temperature,
                idm_attention_gate_limit=self.idm_attention_gate_limit,
                idm_mlp_scale_limit=self.idm_mlp_scale_limit,
                idm_mlp_shift_limit=self.idm_mlp_shift_limit,
                idm_mlp_gate_limit=self.idm_mlp_gate_limit,
            )
            for _ in range(max(n_layers, 1))
        ])
        self.final_layer = AdaLNZeroFinalLayer(
            d_model, action_dim,
            flow_output_projection=self.flow_output_projection,
            action_token_types=self.action_token_types,
            idm_final_scale_limit=self.idm_final_scale_limit,
            idm_final_shift_limit=self.idm_final_shift_limit,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for proj in self.cond_projs:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            nn.init.zeros_(proj.bias)
        for module in [self.cond_mixer[0], self.cond_mixer[2]]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
        for module in [
            self.action_proj,
            self.time_proj[0],
            self.time_proj[2],
        ]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.proprio_token_proj is not None:
            nn.init.xavier_uniform_(self.proprio_token_proj.weight, gain=0.1)
            nn.init.zeros_(self.proprio_token_proj.bias)
        if self.proprio_token_pos_embed is not None:
            nn.init.normal_(self.proprio_token_pos_embed, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def _conditioning_embedding(
        self,
        cond_list: list[torch.Tensor],
        times: torch.Tensor,
    ) -> torch.Tensor:
        cond_tokens = torch.stack(
            [proj(cond) for proj, cond in zip(self.cond_projs, masked_condition_values(cond_list))],
            dim=1,
        )
        cond_tokens = mask_projected_conditions(cond_tokens, cond_list)
        cond_embed = self.cond_mixer(cond_tokens.flatten(1))
        time_embed = self.time_proj(timestep_embedding(times, self.flow_time_embed_dim))
        return _limit_condition_rms(cond_embed + time_embed, self.idm_condition_rms_limit)

    def _vector_field(
        self,
        cond_list: list[torch.Tensor],
        noisy_actions: torch.Tensor,
        times: torch.Tensor,
        proprio_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(cond_list) != self.n_cond_tokens:
            raise ValueError(
                f"Expected {self.n_cond_tokens} condition tensors, got {len(cond_list)}"
            )

        bsz = noisy_actions.shape[0]
        device = noisy_actions.device
        pos_ids = torch.arange(self.action_horizon, device=device)
        pos = self.pos_embed(pos_ids).unsqueeze(0)
        action_tokens = self.action_proj(noisy_actions)
        hidden = action_tokens + pos.to(dtype=action_tokens.dtype)
        hidden = self._append_proprio_action_token(hidden, proprio_token)
        cond_embed = self._conditioning_embedding(cond_list, times).to(dtype=hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, cond_embed)
        action_hidden = hidden[:, : self.action_horizon]
        return self.final_layer(action_hidden, cond_embed,
            end_effector_type_ids=getattr(cond_list, "end_effector_type_ids", None)).reshape(
            bsz,
            self.action_horizon,
            self.action_dim,
        )


class TemporalCrossAttentionFlowMatchingIDM(FlowMatchingIDM):
    """Time-only AdaLN with independent per-block K/V from full-width latents.

    Canonical condition slots are never packed: their additive identities jointly
    identify current/future time, camera and compact slot. Every block normalizes
    the original-width tokens and independently projects its own K and V.
    """

    flow_idm_arch = "temporal_transformer"
    flow_conditioning = "cross_attn"

    def __init__(
        self,
        cond_dim: int = 2560,
        d_model: int = 512,
        nhead: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        fast_vocab_size: int = 2049,
        max_fast_len: int = 20,
        action_dim: int = 7,
        action_horizon: int = 10,
        flow_time_embed_dim: int = 128,
        flow_sampling_steps: int = 16,
        n_cond_tokens: int = 4,
        loss_type: str = "mse",
        huber_delta: float = 0.1,
        action_clip_abs_max: Optional[float] = 1.0,
        sampled_action_clip_abs_max: Optional[float] = None,
        proprio_as_action_token: bool = False,
        flow_cross_attn_gate_init: float = 0.0,
        flow_output_projection: str = "shared",
        action_token_types: tuple[str, ...] = (),
        robot_action_layout: str = "",
        idm_qk_control_mode: str = "none",
        idm_qk_logit_limit: float = 100.0,
        flow_cross_block_gradient_checkpointing: bool = False,
        flow_qk_norm: bool = False,
    ):
        nn.Module.__init__(self)
        if type(flow_cross_block_gradient_checkpointing) is not bool:
            raise ValueError("flow_cross_block_gradient_checkpointing must be a bool")
        self.flow_cross_block_gradient_checkpointing = flow_cross_block_gradient_checkpointing
        if type(flow_qk_norm) is not bool:
            raise ValueError("flow_qk_norm must be a bool")
        if flow_qk_norm and idm_qk_control_mode != "muonclip":
            raise ValueError("flow_qk_norm currently requires cross-attention flow with muonclip")
        self.flow_qk_norm = flow_qk_norm
        if idm_qk_control_mode not in {"none", "muonclip"}:
            raise ValueError("Cross-attention IDM supports none or post-update muonclip")
        self.idm_qk_control_mode = idm_qk_control_mode
        self.idm_qk_logit_limit = float(idm_qk_logit_limit)
        self.flow_output_projection, self.action_token_types = validate_flow_output_projection(
            flow_output_projection,
            action_token_types=action_token_types,
            action_horizon=action_horizon,
        )
        if int(n_cond_tokens) <= 0:
            raise ValueError("n_cond_tokens must be positive.")
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive.")
        if int(d_model) % int(nhead) != 0:
            raise ValueError(
                "d_model must be divisible by nhead for cross-attention "
                f"flow conditioning; got d_model={d_model}, nhead={nhead}."
            )
        self.cond_dim = cond_dim
        self.hidden_dim = int(d_model)
        self.idm_d_model = d_model
        self.fast_vocab_size = fast_vocab_size
        self.max_fast_len = max_fast_len
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flow_time_embed_dim = flow_time_embed_dim
        self.flow_sampling_steps = flow_sampling_steps
        self.action_size = action_dim * action_horizon
        self.n_cond_tokens = int(n_cond_tokens)
        self.proprio_as_action_token = bool(proprio_as_action_token)
        self.flow_cross_attn_gate_init = float(flow_cross_attn_gate_init)
        _configure_flow_regression_loss(
            self,
            loss_type=loss_type,
            huber_delta=huber_delta,
        )
        _configure_action_clip(self, action_clip_abs_max)
        _configure_sampled_action_clip(self, sampled_action_clip_abs_max)
        self._configure_sampling_layout(robot_action_layout)

        token_dim = self.hidden_dim
        self.condition_role_embeddings = nn.Parameter(torch.empty(self.n_cond_tokens, cond_dim))
        self.action_proj = nn.Linear(action_dim, token_dim)
        self.proprio_token_proj = (
            nn.Linear(cond_dim, token_dim)
            if self.proprio_as_action_token
            else None
        )
        self.proprio_token_pos_embed = (
            nn.Parameter(torch.empty(1, 1, token_dim))
            if self.proprio_as_action_token
            else None
        )
        self.time_proj = nn.Sequential(
            nn.Linear(flow_time_embed_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.pos_embed = nn.Embedding(action_horizon, token_dim)
        self.blocks = nn.ModuleList([
            TimeAdaLNCrossAttentionTransformerBlock(
                token_dim,
                nhead,
                ff_mult,
                cross_attn_gate_init=self.flow_cross_attn_gate_init,
                idm_qk_control_mode=idm_qk_control_mode,
                idm_qk_logit_limit=idm_qk_logit_limit,
                cond_dim=cond_dim,
                gradient_checkpointing=flow_cross_block_gradient_checkpointing,
                flow_qk_norm=flow_qk_norm,
            )
            for _ in range(max(n_layers, 1))
        ])
        self.final_layer = AdaLNZeroFinalLayer(
            token_dim, action_dim,
            flow_output_projection=self.flow_output_projection,
            action_token_types=self.action_token_types,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.condition_role_embeddings, std=0.02)
        for module in [
            self.action_proj,
            self.time_proj[0],
            self.time_proj[2],
        ]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.proprio_token_proj is not None:
            nn.init.xavier_uniform_(self.proprio_token_proj.weight, gain=0.1)
            nn.init.zeros_(self.proprio_token_proj.bias)
        if self.proprio_token_pos_embed is not None:
            nn.init.normal_(self.proprio_token_pos_embed, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def _condition_tokens(self, cond_list: list[torch.Tensor]) -> torch.Tensor:
        if len(cond_list) != self.n_cond_tokens:
            raise ValueError(
                f"Expected {self.n_cond_tokens} condition tensors, got {len(cond_list)}"
            )
        tokens = torch.stack(
            masked_condition_values(cond_list),
            dim=1,
        )
        tokens = tokens + self.condition_role_embeddings.unsqueeze(0).to(dtype=tokens.dtype)
        return mask_projected_conditions(tokens, cond_list)

    def _time_embedding(self, times: torch.Tensor) -> torch.Tensor:
        return self.time_proj(timestep_embedding(times, self.flow_time_embed_dim))

    def _vector_field(
        self,
        cond_list: list[torch.Tensor],
        noisy_actions: torch.Tensor,
        times: torch.Tensor,
        proprio_token: Optional[torch.Tensor] = None,
        static_kv_caches: Optional[list[dict]] = None,
    ) -> torch.Tensor:
        bsz = noisy_actions.shape[0]
        device = noisy_actions.device
        pos_ids = torch.arange(self.action_horizon, device=device)
        pos = self.pos_embed(pos_ids).unsqueeze(0)
        action_tokens = self.action_proj(noisy_actions)
        hidden = action_tokens + pos.to(dtype=action_tokens.dtype)
        hidden = self._append_proprio_action_token(hidden, proprio_token)
        cond_tokens = self._condition_tokens(cond_list).to(dtype=hidden.dtype)
        valid_conditions = condition_mask(cond_list, device=hidden.device)
        time_embed = self._time_embedding(times).to(dtype=hidden.dtype)
        for index, block in enumerate(self.blocks):
            cache_kwargs = ({"static_kv_cache": static_kv_caches[index]}
                if static_kv_caches is not None else {})
            if valid_conditions is None:
                hidden = block(hidden, cond_tokens, time_embed, **cache_kwargs)
            else:
                hidden = block(hidden, cond_tokens, time_embed,
                    cond_key_padding_mask=~valid_conditions, **cache_kwargs)
        action_hidden = hidden[:, : self.action_horizon]
        return self.final_layer(action_hidden, time_embed,
            end_effector_type_ids=getattr(cond_list, "end_effector_type_ids", None)).reshape(
            bsz,
            self.action_horizon,
            self.action_dim,
        )


def build_idm_backend(
    idm_type: str,
    *,
    cond_dim: int,
    d_model: int,
    nhead: int,
    n_layers: int,
    ff_mult: int,
    fast_vocab_size: int,
    max_fast_len: int,
    action_dim: int,
    action_horizon: int,
    fast_pretrained_path: str,
    flow_idm_arch: str = "mlp",
    flow_conditioning: str = "adaln_zero",
    flow_time_embed_dim: int = 128,
    flow_sampling_steps: int = 16,
    flow_sampling_cache_static_kv: bool = True,
    n_cond_tokens: int = 4,
    loss_type: str = "mse",
    huber_delta: float = 0.1,
    action_clip_abs_max: Optional[float] = 1.0,
    sampled_action_clip_abs_max: Optional[float] = None,
    proprio_as_action_token: bool = False,
    flow_cross_attn_gate_init: float = 0.0,
    flow_cross_block_gradient_checkpointing: bool = False,
    flow_qk_norm: bool = False,
    flow_output_projection: str = "shared",
    action_token_types: tuple[str, ...] = (),
    action_tokens_per_step: int = 1,
    robot_action_layout: str = "",
    idm_attention_scale_limit: float | None = None,
    idm_attention_shift_limit: float | None = None,
    idm_qk_control_mode: str = "none",
    idm_qk_logit_limit: float = 100.0,
    idm_qk_norm_temperature: float = 8.0,
    idm_condition_rms_limit: float | None = None,
    idm_attention_gate_limit: float | None = None,
    idm_mlp_scale_limit: float | None = None,
    idm_mlp_shift_limit: float | None = None,
    idm_mlp_gate_limit: float | None = None,
    idm_final_scale_limit: float | None = None,
    idm_final_shift_limit: float | None = None,
    fast_ffn_gradient_checkpointing: bool = False,
    fast_layer_gradient_checkpointing: bool = False,
    fast_broadcast_attention: bool = False,
) -> nn.Module:
    if type(flow_qk_norm) is not bool:
        raise ValueError("flow_qk_norm must be a bool")
    if flow_qk_norm and not (
        idm_type == "flow_matching"
        and str(flow_idm_arch).lower() in {"temporal_transformer", "transformer"}
        and str(flow_conditioning).lower().replace("-", "_")
        in {"cross_attn", "cross_attention", "crossattn"}
        and idm_qk_control_mode == "muonclip"
    ):
        raise ValueError("flow_qk_norm currently requires temporal cross-attention flow with muonclip")
    if type(fast_broadcast_attention) is not bool:
        raise ValueError("fast_broadcast_attention must be a bool")
    if fast_broadcast_attention and idm_type != "fast_ar":
        raise ValueError("FAST broadcast attention is only supported by fast_ar")
    if type(flow_cross_block_gradient_checkpointing) is not bool:
        raise ValueError("flow_cross_block_gradient_checkpointing must be a bool")
    if flow_cross_block_gradient_checkpointing and not (
        idm_type == "flow_matching"
        and str(flow_idm_arch).lower() in {"temporal_transformer", "transformer"}
        and str(flow_conditioning).lower().replace("-", "_")
        in {"cross_attn", "cross_attention", "crossattn"}
    ):
        raise ValueError("Whole-block flow checkpointing requires temporal cross-attention flow matching")
    if str(idm_type).strip().lower() in {"act_patch", "patchwam"}:
        if proprio_as_action_token:
            raise ValueError("act_patch requires condition-placed proprio fused into ACT")
        from prism_vla.modeling.action_patch import ActionPatchIDM
        return ActionPatchIDM(cond_dim=cond_dim, action_dim=action_dim,
            action_horizon=action_horizon, n_cond_tokens=n_cond_tokens,
            action_tokens_per_step=action_tokens_per_step,
            loss_type=loss_type, huber_delta=huber_delta,
            action_clip_abs_max=action_clip_abs_max,
            sampled_action_clip_abs_max=sampled_action_clip_abs_max,
            robot_action_layout=robot_action_layout)
    if type(flow_sampling_cache_static_kv) is not bool:
        raise ValueError("flow_sampling_cache_static_kv must be a bool")
    if type(fast_ffn_gradient_checkpointing) is not bool:
        raise ValueError("fast_ffn_gradient_checkpointing must be a bool")
    if type(fast_layer_gradient_checkpointing) is not bool:
        raise ValueError("fast_layer_gradient_checkpointing must be a bool")
    if fast_ffn_gradient_checkpointing and fast_layer_gradient_checkpointing:
        raise ValueError("FAST FFN and whole-layer checkpointing are mutually exclusive")
    if (fast_ffn_gradient_checkpointing or fast_layer_gradient_checkpointing) and idm_type != "fast_ar":
        raise ValueError("FAST checkpointing is only supported by fast_ar")
    idm_attention_scale_limit, idm_attention_shift_limit = validate_idm_attention_limits(
        idm_attention_scale_limit,
        idm_attention_shift_limit,
        idm_type=idm_type,
        flow_idm_arch=flow_idm_arch,
        flow_conditioning=flow_conditioning,
    )
    controls = validate_idm_recovery_controls(
        idm_qk_control_mode=idm_qk_control_mode,
        idm_qk_logit_limit=idm_qk_logit_limit,
        idm_qk_norm_temperature=idm_qk_norm_temperature,
        idm_condition_rms_limit=idm_condition_rms_limit,
        idm_attention_gate_limit=idm_attention_gate_limit,
        idm_mlp_scale_limit=idm_mlp_scale_limit,
        idm_mlp_shift_limit=idm_mlp_shift_limit,
        idm_mlp_gate_limit=idm_mlp_gate_limit,
        idm_final_scale_limit=idm_final_scale_limit,
        idm_final_shift_limit=idm_final_shift_limit,
        idm_type=idm_type,
        flow_idm_arch=flow_idm_arch,
        flow_conditioning=flow_conditioning,
    )
    flow_output_projection, action_token_types = validate_flow_output_projection(
        flow_output_projection,
        action_token_types=action_token_types,
        action_horizon=action_horizon,
        idm_type=idm_type,
        flow_idm_arch=flow_idm_arch,
    )
    if proprio_as_action_token and idm_type != "flow_matching":
        raise ValueError(
            "proprio_as_action_token is only supported by the temporal "
            "flow-matching IDM."
        )
    if idm_type == "fast_ar":
        if action_clip_abs_max is None or not math.isclose(
            float(action_clip_abs_max), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "fast_ar requires action_clip_abs_max=1.0 because FAST actions "
                "are normalized to [-1, 1]."
            )
        return AutoregressiveIDM(
            cond_dim=cond_dim,
            d_model=d_model,
            nhead=nhead,
            n_layers=n_layers,
            ff_mult=ff_mult,
            fast_vocab_size=fast_vocab_size,
            max_fast_len=max_fast_len,
            action_dim=action_dim,
            action_horizon=action_horizon,
            fast_pretrained_path=fast_pretrained_path,
            n_cond_tokens=n_cond_tokens,
            action_token_types=action_token_types,
            robot_action_layout=robot_action_layout,
            fast_ffn_gradient_checkpointing=fast_ffn_gradient_checkpointing,
            fast_layer_gradient_checkpointing=fast_layer_gradient_checkpointing,
            fast_broadcast_attention=fast_broadcast_attention,
        )
    if idm_type == "oft":
        return ParallelL1ActionHead(
            cond_dim=cond_dim, d_model=d_model, n_layers=n_layers,
            action_dim=action_dim, action_horizon=action_horizon,
            n_cond_tokens=n_cond_tokens, action_clip_abs_max=action_clip_abs_max,
            robot_action_layout=robot_action_layout,
        )
    if idm_type == "flow_matching":
        flow_arch = str(flow_idm_arch).lower()
        if flow_arch in {"mlp", "flat_mlp", "flat"}:
            if proprio_as_action_token:
                raise ValueError(
                    "proprio_as_action_token is not supported by the flat MLP flow IDM."
                )
            return FlowMatchingIDM(
                cond_dim=cond_dim,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                ff_mult=ff_mult,
                fast_vocab_size=fast_vocab_size,
                max_fast_len=max_fast_len,
                action_dim=action_dim,
                action_horizon=action_horizon,
                flow_time_embed_dim=flow_time_embed_dim,
                flow_sampling_steps=flow_sampling_steps,
                n_cond_tokens=n_cond_tokens,
                loss_type=loss_type,
                huber_delta=huber_delta,
                action_clip_abs_max=action_clip_abs_max,
                sampled_action_clip_abs_max=sampled_action_clip_abs_max,
                robot_action_layout=robot_action_layout,
            )
        if flow_arch in {"temporal_transformer", "transformer"}:
            conditioning = str(flow_conditioning).lower().replace("-", "_")
            if conditioning in {"adaln_zero", "adaln0"}:
                return TemporalFlowMatchingIDM(
                    cond_dim=cond_dim,
                    d_model=d_model,
                    nhead=nhead,
                    n_layers=n_layers,
                    ff_mult=ff_mult,
                    fast_vocab_size=fast_vocab_size,
                    max_fast_len=max_fast_len,
                    action_dim=action_dim,
                    action_horizon=action_horizon,
                    flow_time_embed_dim=flow_time_embed_dim,
                    flow_sampling_steps=flow_sampling_steps,
                    n_cond_tokens=n_cond_tokens,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                    action_clip_abs_max=action_clip_abs_max,
                    sampled_action_clip_abs_max=sampled_action_clip_abs_max,
                    robot_action_layout=robot_action_layout,
                    proprio_as_action_token=proprio_as_action_token,
                    flow_output_projection=flow_output_projection,
                    action_token_types=action_token_types,
                    idm_attention_scale_limit=idm_attention_scale_limit,
                    idm_attention_shift_limit=idm_attention_shift_limit,
                    **controls,
                )
            if conditioning in {"cross_attn", "cross_attention", "crossattn"}:
                backend = TemporalCrossAttentionFlowMatchingIDM(
                    cond_dim=cond_dim,
                    d_model=d_model,
                    nhead=nhead,
                    n_layers=n_layers,
                    ff_mult=ff_mult,
                    fast_vocab_size=fast_vocab_size,
                    max_fast_len=max_fast_len,
                    action_dim=action_dim,
                    action_horizon=action_horizon,
                    flow_time_embed_dim=flow_time_embed_dim,
                    flow_sampling_steps=flow_sampling_steps,
                    n_cond_tokens=n_cond_tokens,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                    action_clip_abs_max=action_clip_abs_max,
                    sampled_action_clip_abs_max=sampled_action_clip_abs_max,
                    robot_action_layout=robot_action_layout,
                    proprio_as_action_token=proprio_as_action_token,
                    flow_cross_attn_gate_init=flow_cross_attn_gate_init,
                    flow_cross_block_gradient_checkpointing=flow_cross_block_gradient_checkpointing,
                    flow_qk_norm=flow_qk_norm,
                    flow_output_projection=flow_output_projection,
                    action_token_types=action_token_types,
                    idm_qk_control_mode=controls["idm_qk_control_mode"],
                    idm_qk_logit_limit=controls["idm_qk_logit_limit"],
                )
                backend.flow_sampling_cache_static_kv = flow_sampling_cache_static_kv
                return backend
            if conditioning not in {"adaln_zero", "adaln0", "cross_attn", "cross_attention", "crossattn"}:
                raise ValueError(
                    f"Unsupported temporal flow conditioning: {flow_conditioning}. "
                    "Expected 'adaln_zero' or 'cross_attn'."
                )
        raise ValueError(
            f"Unsupported flow_idm_arch: {flow_idm_arch}. "
            "Expected 'mlp' or 'temporal_transformer'."
        )
    raise ValueError(f"Unsupported idm_type: {idm_type}")
