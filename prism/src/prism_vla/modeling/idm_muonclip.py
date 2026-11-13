"""Attention observations for post-update MuonClip weight rescaling.

The forward computes ordinary attention. The optimizer consumes detached,
per-head positive Q/K score maxima after an update and owns all rescaling of
authoritative weights. No observation is part of checkpoint state.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F

from prism_vla.modeling.idm_attention_cache import project_attention_qkv


_MUONCLIP_OBSERVATIONS_SUSPENDED = ContextVar("muonclip_observations_suspended", default=False)


@contextmanager
def suspend_muonclip_observations():
    """Skip diagnostic Q/K work during replay of an already observed forward.

    The flag is local to this execution context and restores its previous value
    even for nested checkpoints or exceptions. Attention outputs and autograd
    are unaffected; only detached optimizer observations are suspended.
    """
    token = _MUONCLIP_OBSERVATIONS_SUSPENDED.set(True)
    try:
        yield
    finally:
        _MUONCLIP_OBSERVATIONS_SUSPENDED.reset(token)


class MuonClipMultiheadAttention(nn.MultiheadAttention):
    """Native packed/separate MHA parameters with an update-window observer.

    Training forwards with autograd enabled contribute to the observer. This
    excludes diagnostic/no-grad action endpoints, evaluation, and the first
    no-grad pass of reentrant activation checkpointing. Nonreentrant whole-block
    replay explicitly suspends observations, retaining the original forward's
    maxima without repeating its FP32 score computation.

    Self-attention retains packed ``[Q; K; V]`` parameters. Cross-attention with
    original-width conditions retains native separate Q/K/V projection weights.
    Distributed Muon ownership and the parent MHA hook boundary are retained.
    The production path uses fused SDPA. Optional parameter-free Q/K RMS
    normalization acts after projection/bias, before observation and attention;
    V projections and inherited checkpoint parameter topology remain unchanged.
    """

    def __init__(self, embed_dim: int, num_heads: int, *,
                 muonclip_logit_limit: float = 100.0,
                 flow_qk_norm: bool = False, **kwargs):
        if type(flow_qk_norm) is not bool:
            raise ValueError("flow_qk_norm must be a bool")
        if (isinstance(muonclip_logit_limit, bool)
                or not math.isfinite(float(muonclip_logit_limit))
                or float(muonclip_logit_limit) <= 0):
            raise ValueError("muonclip_logit_limit must be finite and positive")
        kwargs.setdefault("batch_first", True)
        kwargs.setdefault("dropout", 0.0)
        super().__init__(embed_dim, num_heads, **kwargs)
        if (not self.batch_first or self.bias_k is not None or self.bias_v is not None
                or self.add_zero_attn or self.dropout != 0.0):
            raise ValueError("MuonClip requires batch-first MHA without dropout or extra keys")
        self.muonclip_logit_limit = float(muonclip_logit_limit)
        self.flow_qk_norm = flow_qk_norm
        self._muonclip_max_logits: torch.Tensor | None = None

    def _normalize_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize each projected token/head in FP32 without learned gains.

        RMS normalization retains the ordinary SDPA 1/sqrt(head_dim) scale.
        Its ideal-arithmetic score bound is sqrt(head_dim); BF16 cast-back
        introduces small rounding error. No parameter, persistent buffer,
        host read or detached normalization factor is introduced.
        """
        if not self.flow_qk_norm:
            return q, k
        with torch.autocast(device_type=q.device.type, enabled=False):
            return tuple(F.rms_norm(value.float(), (self.head_dim,), eps=1e-6).to(value.dtype)
                         for value in (q, k))

    def consume_muonclip_max_logits(self) -> torch.Tensor:
        """Return/reset detached FP32 maxima on the attention parameter device.

        Unobserved heads return zero. Every configured module must be consumed
        in deterministic order, including ranks with no valid observations.
        """
        observed = self._muonclip_max_logits
        self._muonclip_max_logits = None
        if observed is None:
            weight = self.in_proj_weight if self._qkv_same_embed_dim else self.q_proj_weight
            return torch.zeros(self.num_heads, device=weight.device, dtype=torch.float32)
        return observed

    def _observe(self, q: torch.Tensor, k: torch.Tensor,
                 allowed: torch.Tensor | None) -> None:
        if (not self.training or not torch.is_grad_enabled()
                or _MUONCLIP_OBSERVATIONS_SUSPENDED.get()):
            return
        with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
            q32, k32 = q.detach().float(), k.detach().float()
            maximum = torch.zeros(self.num_heads, device=q.device, dtype=torch.float32)
            # Keep the diagnostic score workspace bounded for long action chunks.
            for start in range(0, q.shape[-2], 64):
                scores = torch.matmul(q32[..., start:start + 64, :], k32.transpose(-2, -1))
                scores = scores / math.sqrt(self.head_dim)
                if allowed is not None:
                    mask = allowed[..., start:start + 64, :] if allowed.shape[-2] != 1 else allowed
                    scores = torch.where(mask, scores, -torch.inf)
                if scores.numel():
                    maximum = torch.maximum(maximum, scores.amax(dim=(0, 2, 3)).clamp_min(0))
            if self._muonclip_max_logits is not None:
                maximum = torch.maximum(self._muonclip_max_logits.to(maximum.device), maximum)
            self._muonclip_max_logits = maximum

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None, need_weights: bool = True,
                attn_mask: torch.Tensor | None = None, average_attn_weights: bool = True,
                is_causal: bool = False, *,
                observation_key_padding_mask: torch.Tensor | None = None,
                static_kv_cache: dict | None = None):
        """Apply ordinary MHA; optionally exclude synthetic keys from observation.

        ``observation_key_padding_mask`` retains the original all-missing camera
        inventory when the caller exposes one finite synthetic key to attention.
        It affects observation only, never the attention result or its gradients.
        """
        if (query.ndim != 3 or key.ndim != 3 or value.ndim != 3
                or query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]
                or query.shape[-1] != self.embed_dim or key.shape[-1] != self.kdim
                or value.shape[-1] != self.vdim
                or is_causal):
            raise ValueError("MuonClip expects aligned batch-first noncausal Q/K/V")
        batch, queries = query.shape[:2]
        keys = key.shape[1]
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, keys):
                raise ValueError("key_padding_mask must be [batch, keys]")
            missing = key_padding_mask if key_padding_mask.dtype == torch.bool else torch.isneginf(key_padding_mask)
            if key is value:
                key = value = torch.where(missing[..., None], 0.0, key)
            else:
                key = torch.where(missing[..., None], 0.0, key)
                value = torch.where(missing[..., None], 0.0, value)
        q, k, v = project_attention_qkv(self, query, key, value, static_kv_cache)
        q = q.reshape(batch, queries, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = (tensor.reshape(batch, keys, self.num_heads, self.head_dim).transpose(1, 2)
            for tensor in (k, v))
        q, k = self._normalize_qk(q, k)
        mask = None
        if attn_mask is not None:
            if attn_mask.shape == (queries, keys):
                mask = attn_mask[None, None]
            elif attn_mask.shape == (batch * self.num_heads, queries, keys):
                mask = attn_mask.reshape(batch, self.num_heads, queries, keys)
            else:
                raise ValueError("attn_mask must be [queries, keys] or [batch*heads, queries, keys]")
            mask = (torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, -torch.inf)
                if mask.dtype == torch.bool else mask.to(dtype=q.dtype))
        if key_padding_mask is not None:
            padding = key_padding_mask[:, None, None]
            padding = (torch.zeros_like(padding, dtype=q.dtype).masked_fill(padding, -torch.inf)
                if padding.dtype == torch.bool else padding.to(dtype=q.dtype))
            mask = padding if mask is None else mask + padding
        allowed = None if mask is None else ~torch.isneginf(mask)
        if observation_key_padding_mask is not None:
            if observation_key_padding_mask.shape != (batch, keys) or observation_key_padding_mask.dtype != torch.bool:
                raise ValueError("observation_key_padding_mask must be boolean [batch, keys]")
            valid_keys = ~observation_key_padding_mask[:, None, None]
            allowed = valid_keys if allowed is None else allowed & valid_keys
        self._observe(q, k, allowed)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().reshape(batch, queries, self.embed_dim)
        output = F.linear(output, self.out_proj.weight, self.out_proj.bias)
        weights = None
        if need_weights:
            scores = torch.matmul(q * (self.head_dim ** -0.5), k.transpose(-2, -1))
            if mask is not None:
                scores = scores + mask
            weights = torch.softmax(scores, dim=-1)
            if average_attn_weights:
                weights = weights.mean(dim=1)
        return output, weights


def iter_muonclip_attention_modules(model: nn.Module) -> tuple[tuple[str, MuonClipMultiheadAttention], ...]:
    """Return a stable module-name inventory for distributed optimizer callbacks."""
    return tuple(sorted((name, module) for name, module in model.named_modules()
        if isinstance(module, MuonClipMultiheadAttention)))
