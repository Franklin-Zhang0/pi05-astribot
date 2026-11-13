"""Detached native full-attention K/V for a knowledge-insulated action expert.

Qwen3.5's recurrent layers do not have a token K/V cache. This contract exposes
every native full-attention layer, including its actual rotary-transformed keys,
and preserves the backbone's causal/window/camera mask. It never substitutes
last-layer hidden states for K/V. Routing is cheap metadata: repeated transition
rows do not duplicate the underlying cache until a bounded attention microbatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeKV:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    allowed: torch.Tensor
    query_length: int
    prefix_tokens: int
    positions: torch.Tensor


@dataclass(frozen=True)
class KVRows:
    contexts: tuple[NativeKV, ...]
    # [N, 3]: context index, source batch index, source query index.
    routes: torch.Tensor

    def select(self, rows):
        return KVRows(self.contexts, self.routes[rows])

    def repeat(self, repeats, *, interleave=False):
        routes = (self.routes.repeat_interleave(repeats, dim=0) if interleave
                  else self.routes.repeat(repeats, 1))
        return KVRows(self.contexts, routes)


def needs_action_expert(model):
    if getattr(model.config, "engram_native", None) is not None:
        # The retained camera_pi05 head is dormant in native DiT finetuning.
        return False
    return any(spec["type"] == "pi05" for spec in getattr(model.config, "action_heads", ()))


def capture_native_kv(output, *, attention_mask, query_length, prefix_tokens, position_ids=None):
    """Read completed native cache tensors before the caller crops its history."""
    cache = output.past_key_values
    if cache is None or not hasattr(cache, "key_cache"):
        raise ValueError("pi05 requires a native full-attention K/V cache")
    layers = tuple((key.detach(), value.detach())
        for key, value in zip(cache.key_cache, cache.value_cache)
        if torch.is_tensor(key) and key.ndim == 4)
    if not layers:
        raise ValueError("pi05 requires at least one full-attention backbone layer")
    key_length = layers[0][0].shape[-2]
    batch = layers[0][0].shape[0]
    device = layers[0][0].device
    if any(key.shape[-2] != key_length for key, _ in layers):
        raise ValueError("Native full-attention layers have different cache lengths")
    if attention_mask is not None and attention_mask.ndim == 4:
        if attention_mask.dtype != torch.bool:
            raise ValueError("Action expert requires an explicit boolean causal attention mask")
        allowed = attention_mask.any(dim=1).detach()
    else:
        keys = torch.arange(key_length, device=device)
        queries = torch.arange(key_length - query_length, key_length, device=device)
        allowed = (keys[None, None, :] <= queries[None, :, None]).expand(batch, -1, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, :key_length].bool()
    if tuple(allowed.shape) != (batch, query_length, key_length):
        raise ValueError("Native K/V visibility does not match the current main-stream query")
    if position_ids is None:
        position_ids = torch.arange(key_length - query_length, key_length, device=device)[None].expand(batch, -1)
    if position_ids.ndim != 2 or tuple(position_ids.shape) != (batch, query_length):
        raise ValueError("Action expert requires explicit text positions [B, Q]")
    output.action_expert_kv = NativeKV(layers, allowed, query_length, prefix_tokens, position_ids.detach())
    return output


def select_action_expert_kv(outputs, valid, *, tokens_per_step, row_indices=None):
    """Route valid B-major transitions to their current causal backbone prefix."""
    values = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
    contexts = tuple(getattr(value, "action_expert_kv", None) for value in values)
    if not any(context is not None for context in contexts):
        return None
    if any(context is None for context in contexts):
        raise ValueError("Missing action-expert K/V in a main-stream forward")
    valid = valid.reshape(-1)
    batch = contexts[0].allowed.shape[0]
    if valid.numel() % batch:
        raise ValueError("Action transition routing does not match source batch")
    steps = valid.numel() // batch
    flat = torch.arange(valid.numel(), device=valid.device)
    indices = flat[valid] if row_indices is None else row_indices.to(device=valid.device)
    source_batch, timestep = indices // steps, indices % steps
    if len(contexts) == 1:
        query = contexts[0].prefix_tokens + (timestep + 1) * tokens_per_step - 1
        context_ids = torch.zeros_like(source_batch)
    else:
        if len(contexts) < steps:
            raise ValueError("Missing per-timestep K/V cache")
        query = torch.full_like(timestep, tokens_per_step - 1)
        context_ids = timestep
    return KVRows(contexts, torch.stack((context_ids, source_batch, query), dim=-1))
