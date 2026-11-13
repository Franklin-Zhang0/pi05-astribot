"""FAST self-attention with a head-broadcast mask and native MHA parameters.

The native MHA module remains the call/hook boundary, including its functional
QKV and output weights. Only its attention execution changes: SDPA receives
[B,1,L,L] rather than an eagerly repeated [B*heads,L,L] mask. The layer avoids
the native inference fused path so parameter hooks run in training and rollout.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel


class BroadcastSDPASelfAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if (not self.batch_first
                or not self._qkv_same_embed_dim or self.bias_k is not None
                or self.bias_v is not None or self.add_zero_attn or need_weights
                or key_padding_mask is not None):
            raise ValueError("FAST broadcast SDPA requires batch-first self-attention without returned weights")
        batch, length, width = query.shape
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("FAST broadcast SDPA requires equal Q/K/V sequence shapes")
        # DMuon's backward sentinel wraps the first positional argument (Q),
        # breaking the original Q/K/V alias. Native packed projection retains
        # that sentinel and the actual K/V inputs without requiring aliases.
        q, k, v = F._in_projection_packed(query, key, value,
            self.in_proj_weight, self.in_proj_bias)
        def heads(tensor):
            return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        if attn_mask is not None and attn_mask.ndim not in (2, 4):
            raise ValueError("FAST SDPA mask must be common [L,L] or broadcast [B,1,L,L]")
        # Native MHA's is_causal=True hint discards a matching common mask.
        # Camera masks have is_causal=False because their private invalid-query
        # diagonals and key exclusions must be represented explicitly.
        mask = None if is_causal else attn_mask
        if mask is not None and mask.dtype == torch.bool:
            mask = ~mask  # MHA True=forbidden; SDPA True=allowed.
        # An unsupported CUDA bias must fail explicitly rather than silently
        # allocating a quadratic per-head math attention matrix again.
        backend = (sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
            if query.is_cuda else nullcontext())
        with backend:
            attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v),
                attn_mask=mask, dropout_p=self.dropout if self.training else 0., is_causal=is_causal)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return F.linear(attended, self.out_proj.weight, self.out_proj.bias), None


class FASTBroadcastEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the exact native object, Parameters, initialization order,
        # and RNG draws. This subclass adds no state or storage of its own.
        self.self_attn.__class__ = BroadcastSDPASelfAttention

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if src_key_padding_mask is not None:
            raise ValueError("FAST broadcast key presence belongs to its explicit attention mask")
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, None, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, None, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


class FASTBroadcastLayerCheckpointEncoderLayer(FASTBroadcastEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if self.training and torch.is_grad_enabled():
            return checkpoint(super().forward, src, src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask, is_causal=is_causal,
                use_reentrant=False, preserve_rng_state=True, early_stop=False)
        return super().forward(src, src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)


class FASTBroadcastFFNCheckpointEncoderLayer(FASTBroadcastEncoderLayer):
    def _ff_block(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(super()._ff_block, x, use_reentrant=False,
                preserve_rng_state=True, early_stop=False)
        return super()._ff_block(x)
