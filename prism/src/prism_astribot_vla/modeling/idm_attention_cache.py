"""Ephemeral static-condition projections for no-grad action sampling."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def project_attention_qkv(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    static_kv_cache: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project inside the attention forward, after materialization hooks run.

    The caller owns a fresh cache for each sampling call, block and CFG branch.
    It must never reuse it with different conditions or after a weight update.
    Gradient-bearing training forwards always use the ordinary uncached path.
    """
    if static_kv_cache is not None and torch.is_grad_enabled():
        raise ValueError("Static condition K/V caching requires no-grad sampling")
    cached = None if static_kv_cache is None else static_kv_cache.get("projected_kv")
    if cached is not None:
        if static_kv_cache["owner"] is not module:
            raise ValueError("A static K/V cache belongs to one attention module")
        q_weight = (module.in_proj_weight[:module.embed_dim] if module._qkv_same_embed_dim
                    else module.q_proj_weight)
        q_bias = None if module.in_proj_bias is None else module.in_proj_bias[:module.embed_dim]
        return F.linear(query, q_weight, q_bias), cached[0], cached[1]

    if not module._qkv_same_embed_dim:
        biases = (None, None, None) if module.in_proj_bias is None else module.in_proj_bias.chunk(3)
        q, k, v = (F.linear(tensor, weight, bias) for tensor, weight, bias in
            zip((query, key, value), (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight), biases))
    elif query is key and key is value:
        q, k, v = F.linear(query, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
    elif key is value:
        q_weight, kv_weight = module.in_proj_weight.split((module.embed_dim, 2 * module.embed_dim))
        q_bias, kv_bias = ((None, None) if module.in_proj_bias is None else
            module.in_proj_bias.split((module.embed_dim, 2 * module.embed_dim)))
        q = F.linear(query, q_weight, q_bias)
        k, v = F.linear(key, kv_weight, kv_bias).chunk(2, dim=-1)
    else:
        biases = (None, None, None) if module.in_proj_bias is None else module.in_proj_bias.chunk(3)
        q, k, v = (F.linear(tensor, weight, bias) for tensor, weight, bias in
            zip((query, key, value), module.in_proj_weight.chunk(3), biases))
    if static_kv_cache is not None:
        static_kv_cache.update(owner=module, projected_kv=(k, v))
    return q, k, v


class CachedConditionMultiheadAttention(nn.MultiheadAttention):
    """Native MHA parameters/training with optional static K/V at inference."""

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False, *,
                static_kv_cache: dict | None = None):
        if static_kv_cache is None:
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                average_attn_weights=average_attn_weights, is_causal=is_causal)
        if (not self.batch_first or self.dropout != 0.0 or self.bias_k is not None
                or self.bias_v is not None or self.add_zero_attn or is_causal
                or query.ndim != 3 or key.ndim != 3 or value.ndim != 3):
            raise ValueError("Static K/V caching requires batch-first noncausal MHA without dropout")
        batch, queries = query.shape[:2]
        keys = key.shape[1]
        q, k, v = project_attention_qkv(self, query, key, value, static_kv_cache)
        q = q.reshape(batch, queries, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = (tensor.reshape(batch, keys, self.num_heads, self.head_dim).transpose(1, 2)
                for tensor in (k, v))
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
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().reshape(batch, queries, self.embed_dim)
        output = F.linear(output, self.out_proj.weight, self.out_proj.bias)
        weights = None
        if need_weights:
            scores = torch.matmul(q * (self.head_dim ** -0.5), k.transpose(-2, -1))
            weights = torch.softmax(scores if mask is None else scores + mask, dim=-1)
            if average_attn_weights:
                weights = weights.mean(dim=1)
        return output, weights
