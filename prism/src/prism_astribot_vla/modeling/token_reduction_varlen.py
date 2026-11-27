"""Ragged joint attention over the exact per-frame rolling and snapshot key sets."""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch


_COUNTS = {"calls": 0, "query_rows": 0, "kv_rows": 0}


def ragged_attention_counts(*, reset=False):
    """Read process-local counters collected with PRISM_VERIFY_VARLEN_GQA=1."""
    result = _COUNTS.copy()
    if reset:
        _COUNTS.update(dict.fromkeys(_COUNTS, 0))
    return result


@dataclass(frozen=True)
class RaggedFramePlan:
    query_rows: torch.Tensor
    key_rows: torch.Tensor
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    max_query: int
    max_key: int


def build_ragged_frame_plan(valid_cpu, packed_indices, native_length, prefix_mask, prefix_owners, device):
    """Build linear indices into query rows and [unique prefix bank, native bank].

    Inputs are small CPU validity/index tensors already known by the backbone.
    Empty query frames are omitted and the output scatter restores their zeros.
    All spatial positions at a timestep use the same allowed keys.
    """
    batch, frames, tokens = valid_cpu.shape
    q_valid = valid_cpu.reshape(batch * frames, tokens)
    native_ids, native_valid = packed_indices
    row_batch = torch.arange(batch).repeat_interleave(frames)
    native_ids = native_ids + row_batch[:, None] * native_length
    if prefix_mask is not None:
        width = prefix_mask.shape[1]
        prefix_ids = prefix_owners[:, None] * width + torch.arange(width)
        key_ids = torch.cat((prefix_ids, native_ids + prefix_mask.numel()), dim=1)
        key_valid = torch.cat((prefix_mask.index_select(0, prefix_owners), native_valid), dim=1)
    else:
        key_ids, key_valid = native_ids, native_valid

    q_lengths = q_valid.sum(1, dtype=torch.int32)
    used = q_lengths > 0
    q_lengths = q_lengths[used]
    key_valid = key_valid & used[:, None]
    k_lengths = key_valid.sum(1, dtype=torch.int32)[used]
    if q_lengths.numel() == 0 or k_lengths.min().item() < 1:
        raise ValueError("Variable-length frame attention requires keys for each nonempty query frame")

    def offsets(lengths):
        return torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32))).to(device)

    return RaggedFramePlan(q_valid.flatten().nonzero().flatten().to(device),
                           key_ids.masked_select(key_valid).to(device),
                           offsets(q_lengths), offsets(k_lengths),
                           int(q_lengths.max()), int(k_lengths.max()))


def _flash_varlen(query, key, value, plan, scale):
    from torch.nn.attention.varlen import varlen_attn
    return varlen_attn(query, key, value, plan.cu_query, plan.cu_key,
                       plan.max_query, plan.max_key, scale=scale)


def ragged_frame_attention(query, key, value, prefix_key, prefix_value, plan, scale):
    """Compute native grouped-query FlashAttention without expanding K/V heads."""
    batch, heads, queries, dim = query.shape

    def rows(tensor):
        return tensor.transpose(1, 2).reshape(-1, tensor.shape[1], dim)

    def gather_keys(native, prefix):
        source = rows(native)
        if prefix is not None:
            source = torch.cat((rows(prefix), source), dim=0)
        return source.index_select(0, plan.key_rows)

    q = rows(query).index_select(0, plan.query_rows)
    k = gather_keys(key, prefix_key)
    v = gather_keys(value, prefix_value)
    out = _flash_varlen(q, k, v, plan, scale)
    if os.environ.get("PRISM_VERIFY_VARLEN_GQA") == "1":
        _COUNTS["calls"] += 1
        _COUNTS["query_rows"] += q.shape[0]
        _COUNTS["kv_rows"] += k.shape[0]
    restored = out.new_zeros((batch * queries, heads, dim)).index_copy_(0, plan.query_rows, out)
    return restored.reshape(batch, queries, heads * dim)
