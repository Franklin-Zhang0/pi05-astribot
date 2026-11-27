"""Stage-aware Qwen execution for fixed spatial token selection.

Full attention packs individual query frames and their eligible KV windows into
the SDPA batch dimension.  It never computes a dense all-chunk attention matrix.
KDA still scans the complete chronological stage sequence and keeps global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
import types
import os
from typing import Any, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F

from prism_astribot_vla.modeling.qwen35_deltanet_patch import qwen35_valid_token_scope

if TYPE_CHECKING:
    from prism_astribot_vla.modeling.text_deepstack import TextDeepstackMemory


@dataclass(frozen=True)
class FrameDeepstackMemory:
    """Frozen prefix snapshots and the snapshot visible to each [batch, frame] query."""
    memory: TextDeepstackMemory
    indices: torch.Tensor

    @property
    def attention_mask(self) -> torch.Tensor:
        return self.memory.attention_mask


@dataclass
class LayerTokenMetadata:
    """CPU timestamps/roles and device validity for one layer's OWN KV bank."""

    times: torch.Tensor
    valid: torch.Tensor
    prefix: torch.Tensor
    valid_cpu: torch.Tensor


@dataclass
class TokenReductionCache:
    """Native KDA/attention state with independent temporal KV metadata."""

    native_cache: Any
    batch_size: int
    metadata: dict[int, LayerTokenMetadata] = field(default_factory=dict)
    prefix_length: int = 0
    prefix_template: dict[str, list[torch.Tensor | None]] = field(default_factory=dict)
    prefix_valid: torch.Tensor | None = None
    prefix_valid_cpu: torch.Tensor | None = None
    last_times: torch.Tensor | None = None

    def detach(self) -> "TokenReductionCache":
        """Detach at a training-chunk boundary, never between its timesteps."""
        for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            values = getattr(self.native_cache, name)
            setattr(self.native_cache, name, [x.detach() if x is not None else None for x in values])
        return self

    def reset_rows(self, mask: torch.Tensor) -> "TokenReductionCache":
        """Restore selected episodes to their initial instruction-prefix state."""
        mask = torch.as_tensor(mask, dtype=torch.bool).detach().cpu()
        if tuple(mask.shape) != (self.batch_size,):
            raise ValueError("Token-reduction reset mask must have one entry per batch row")
        for name in ("conv_states", "recurrent_states", "key_cache", "value_cache"):
            for index, value in enumerate(getattr(self.native_cache, name)):
                if value is None:
                    continue
                row_mask = mask.to(value.device).view(self.batch_size, *([1] * (value.ndim - 1)))
                replacement = torch.zeros_like(value)
                template = self.prefix_template.get(name, [None] * len(self.native_cache.layer_types))[index]
                if template is not None:
                    if name in {"key_cache", "value_cache"}:
                        replacement[..., :template.shape[-2], :] = template.to(value)
                    else:
                        replacement = template.to(value)
                getattr(self.native_cache, name)[index] = torch.where(row_mask, replacement, value)
        for meta in self.metadata.values():
            valid = meta.valid.clone()
            valid[mask.to(valid.device)] = False
            if self.prefix_length and self.prefix_valid is not None:
                valid[mask.to(valid.device), :self.prefix_length] = self.prefix_valid.to(valid.device)[mask.to(valid.device)]
            meta.valid = valid
            meta.valid_cpu = meta.valid_cpu.clone()
            meta.valid_cpu[mask] = False
            if self.prefix_length and self.prefix_valid_cpu is not None:
                meta.valid_cpu[mask, :self.prefix_length] = self.prefix_valid_cpu[mask]
            meta.times = torch.where(mask[:, None] & ~meta.prefix,
                torch.full_like(meta.times, -torch.inf), meta.times)
        if self.last_times is not None:
            self.last_times = torch.where(mask, -torch.inf, self.last_times)
        return self


@dataclass
class TokenReductionOutput:
    hidden_states: torch.Tensor
    valid: torch.Tensor
    cache: TokenReductionCache
    stage_info: tuple[dict[str, Any], ...]
    position_ids: torch.Tensor

    @property
    def last_hidden_state(self):
        return self.hidden_states

    @property
    def past_key_values(self):
        return self.cache


def _text_model(qwen):
    model = qwen
    for _ in range(4):
        if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
            return model
        model = getattr(model, "model", None)
        if model is None:
            break
    raise ValueError("Token reduction requires a Qwen text model with native decoder layers")


def _packed_window_indices(meta: LayerTokenMetadata, frame_times: torch.Tensor, window: int):
    """Build fixed-shape packed gathers from small CPU frame metadata."""
    if os.getenv("PRISM_BATCHED_WINDOW_INDICES") == "1":
        return _batched_window_indices(meta, frame_times, window)
    rows = []
    for row in range(frame_times.shape[0]):
        for time in frame_times[row]:
            allowed = meta.valid_cpu[row] & (meta.prefix[row] | ((meta.times[row] <= time)
                & (meta.times[row] >= time - window + 1)))
            rows.append(allowed.nonzero().flatten())
    width = max(1, max(x.numel() for x in rows))
    indices = torch.zeros(len(rows), width, dtype=torch.long)
    present = torch.zeros(len(rows), width, dtype=torch.bool)
    for row, selected in enumerate(rows):
        indices[row, :selected.numel()] = selected
        present[row, :selected.numel()] = True
    return indices, present


def _batched_window_indices(meta, frame_times, window):
    """Pack each row's eligible native columns in order, including prefix keys.

    Compact the validity mask first. This avoids building a three-dimensional
    mask over all spatial keys for all query times in a long chunk.
    """
    import numpy as np

    columns = []
    counts = []
    source_times = meta.times.numpy()
    query_times = frame_times.numpy()
    prefix = meta.prefix.numpy()
    for batch, valid in enumerate(meta.valid_cpu.numpy()):
        ids = np.flatnonzero(valid)
        times = source_times[batch, ids]
        prefix_ids = ids[prefix[batch, ids]]
        native = ~prefix[batch, ids]
        native_times, native_ids = times[native], ids[native]
        # Cached heterogeneous resets can leave nonmonotonic physical keys.
        # Search sorted bounds only when they preserve the native column order.
        prefix_first = not prefix_ids.size or not native_ids.size or prefix_ids[-1] < native_ids[0]
        ordered = np.all(native_times[:-1] <= native_times[1:])
        if prefix_first and ordered and not np.isnan(query_times[batch]).any() and not np.isnan(native_times).any():
            qs = query_times[batch]
            start = np.searchsorted(native_times, qs-window+1, side="left")
            end = np.searchsorted(native_times, qs, side="right")
            for a, b in zip(start, end):
                row = native_ids[a:b]
                if prefix_ids.size:
                    row = np.concatenate((prefix_ids, row))
                columns.append(row)
                counts.append(len(row))
        else:
            for t in query_times[batch]:
                row = ids[prefix[batch, ids] | ((times <= t) & (times >= t-window+1))]
                columns.append(row)
                counts.append(len(row))
    width = max(1, max(counts))
    present = np.arange(width)[None, :] < np.asarray(counts)[:, None]
    indices = np.zeros(present.shape, dtype=np.int64)
    indices[present] = np.concatenate(columns)
    return torch.from_numpy(indices), torch.from_numpy(present)


def _gather_frame_keys(value, indices, frames):
    batch, heads, _, dim = value.shape
    groups = batch * frames
    row_ids = torch.arange(batch, device=value.device).repeat_interleave(frames)
    source = value.permute(0, 2, 1, 3)
    return source[row_ids[:, None], indices.to(value.device)].reshape(groups, -1, heads, dim).transpose(1, 2)


def _packed_full_attention(self, hidden_states, position_embeddings, attention_mask=None,
        past_key_values=None, text_deepstack_memory=None, token_reduction_context=None, **kwargs):
    if token_reduction_context is None:
        return self._token_reduction_native_forward(hidden_states=hidden_states,
            position_embeddings=position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, text_deepstack_memory=text_deepstack_memory, **kwargs)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    context = token_reduction_context
    batch, length, _ = hidden_states.shape
    frames, tokens = context["frames"], context["tokens"]
    dim = self.head_dim
    projection = self.q_proj(hidden_states)
    heads = int(self.config.num_attention_heads)
    if projection.shape[-1] == 2 * heads * dim:
        query, gate = projection.reshape(batch, length, heads, 2 * dim).chunk(2, -1)
    else:
        query, gate = projection.reshape(batch, length, heads, dim), None
    query = self.q_norm(query).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).reshape(batch, length, -1, dim)).transpose(1, 2)
    value = self.v_proj(hidden_states).reshape(batch, length, -1, dim).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    if past_key_values is not None:
        boundary = getattr(self, "_prism_qwen_cache_update", None)
        key, value = (boundary(past_key_values, key, value) if callable(boundary)
            else past_key_values.update(key, value, self.layer_idx))
    meta = context["metadata"]
    if key.shape[-2] != meta.times.shape[-1]:
        raise RuntimeError("Per-layer token metadata does not match the native KV bank")
    if "ragged_plan" in context:
        from prism_astribot_vla.modeling.token_reduction_varlen import ragged_frame_attention

        prefix_key = prefix_value = None
        if text_deepstack_memory is not None:
            snapshots = (text_deepstack_memory.memory if isinstance(text_deepstack_memory, FrameDeepstackMemory)
                         else text_deepstack_memory)
            reference = query.new_empty((snapshots.attention_mask.shape[0], 0))
            prefix_key, prefix_value, _ = self.text_deepstack_adapter(snapshots, reference)
        output = ragged_frame_attention(query, key, value, prefix_key, prefix_value,
                                        context["ragged_plan"], self.scaling)
        if gate is not None:
            output = output * torch.sigmoid(gate.reshape(batch, length, heads * dim))
        return self.o_proj(output), None
    indices, selected = context["packed_indices"]
    packed_query = query.reshape(batch, heads, frames, tokens, dim).permute(0, 2, 1, 3, 4).reshape(batch * frames, heads, tokens, dim)
    packed_key = _gather_frame_keys(key, indices, frames)
    packed_value = _gather_frame_keys(value, indices, frames)
    row_ids = torch.arange(batch, device=key.device).repeat_interleave(frames)
    key_valid = meta.valid[row_ids[:, None], indices.to(key.device)] & selected.to(key.device)
    if text_deepstack_memory is not None:
        adapter = getattr(self, "text_deepstack_adapter", None)
        if adapter is None:
            raise ValueError("Text deepstack memory requires the existing per-layer text adapter")
        if isinstance(text_deepstack_memory, FrameDeepstackMemory):
            snapshots = text_deepstack_memory.memory
            indices = text_deepstack_memory.indices
            if indices.shape != (batch, frames) or indices.dtype != torch.long:
                raise ValueError("VLM snapshot indices must be int64 [batch, frames]")
            # Project each snapshot once. Match the frame-major ordering of packed_query.
            reference = query.new_empty((snapshots.attention_mask.shape[0], 0))
            keys, values, masks = adapter(snapshots, reference)
            indices = indices.flatten().to(keys.device)
            text_key, text_value, text_valid = (tensor.index_select(0, indices)
                                                 for tensor in (keys, values, masks))
        else:
            keys, values, masks = adapter(text_deepstack_memory, query)
            text_key, text_value, text_valid = (tensor.repeat_interleave(frames, 0)
                                                 for tensor in (keys, values, masks))
        packed_key = torch.cat((text_key, packed_key), dim=-2)
        packed_value = torch.cat((text_value, packed_value), dim=-2)
        key_valid = torch.cat((text_valid, key_valid), dim=-1)
    groups = heads // packed_key.shape[1]
    packed_key, packed_value = repeat_kv(packed_key, groups), repeat_kv(packed_value, groups)
    query_valid = context["valid"].reshape(batch * frames, tokens)
    mask = query_valid[:, None, :, None] & key_valid[:, None, None, :]
    output = F.scaled_dot_product_attention(packed_query, packed_key, packed_value,
        attn_mask=mask, dropout_p=self.attention_dropout if self.training else 0.,
        is_causal=False, scale=self.scaling)
    output = output.reshape(batch, frames, heads, tokens, dim).permute(0, 1, 3, 2, 4).reshape(batch, length, heads * dim)
    if gate is not None:
        output = output * torch.sigmoid(gate.reshape(batch, length, -1))
    return self.o_proj(output), None


def _fused_qwen_norm(module, x):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32) or not x.numel():
        return Qwen3_5RMSNorm.forward(module, x)
    if getattr(module, "_prism_static_rms", False):
        from prism_astribot_vla.modeling.qwen_rms_static import qwen_rms_norm

        return qwen_rms_norm(x, module.weight, module.eps)
    from fla.modules.layernorm import rms_norm

    # Qwen applies the offset and the scale in FP32 before converting to x.dtype.
    # FLA's backward saves the input and per-row reciprocal standard deviation.
    return rms_norm(x, 1.0 + module.weight.float(), bias=None, eps=module.eps)


def install_fused_qwen_norms(qwen) -> int:
    """Use FLA RMSNorm for the trainable Qwen3.5 text backbone, preserving parameter identity."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    static = os.getenv("PRISM_QWEN_STATIC_RMSNORM", "0")
    if static not in ("0", "1", "all"):
        raise ValueError("PRISM_QWEN_STATIC_RMSNORM must be 0, 1 (width <= 512), or all")
    count = 0
    for module in _text_model(qwen).modules():
        if type(module) is Qwen3_5RMSNorm:
            module._prism_static_rms = static == "all" or (static == "1" and module.weight.numel() <= 512)
            module.forward = types.MethodType(_fused_qwen_norm, module)
            count += 1
    return count


def install_token_reduction_attention(qwen) -> int:
    """Install a conditional frame-packed attention path before FSDP wrapping."""
    if os.getenv("PRISM_QWEN_FUSED_RMSNORM") == "1" and not getattr(qwen, "_prism_fused_qwen_norms", False):
        qwen._prism_fused_qwen_norms = install_fused_qwen_norms(qwen)
    count = 0
    for layer in _text_model(qwen).layers:
        if getattr(layer, "layer_type", None) != "full_attention":
            continue
        attention = layer.self_attn
        if not hasattr(attention, "_token_reduction_native_forward"):
            attention._token_reduction_native_forward = attention.forward
            attention.forward = types.MethodType(_packed_full_attention, attention)
        count += 1
    return count


def _new_cache(model, batch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
    return TokenReductionCache(Qwen3_5DynamicCache(model.config), batch)


def _prefill_prefix(model, cache, prefix_embeds, prefix_mask):
    if prefix_embeds is None or prefix_embeds.shape[1] == 0:
        return
    batch, length = prefix_embeds.shape[:2]
    valid = (torch.ones(batch, length, dtype=torch.bool, device=prefix_embeds.device)
        if prefix_mask is None else prefix_mask.to(device=prefix_embeds.device, dtype=torch.bool))
    if tuple(valid.shape) != (batch, length):
        raise ValueError("Instruction prefix validity must match prefix embeddings")
    positions = torch.arange(length, device=prefix_embeds.device).expand(batch, -1)
    with qwen35_valid_token_scope(valid, stream="prefix"):
        model(inputs_embeds=torch.where(valid[..., None], prefix_embeds, 0),
            attention_mask=valid, position_ids=positions,
            past_key_values=cache.native_cache, use_cache=True)
    cache.prefix_length, cache.prefix_valid = length, valid.detach().clone()
    cache.prefix_valid_cpu = valid.detach().cpu().clone()
    for index in cache.native_cache.transformer_layers:
        cache.metadata[index] = LayerTokenMetadata(torch.full((batch, length), -torch.inf, dtype=torch.float64),
            valid, torch.ones(batch, length, dtype=torch.bool), cache.prefix_valid_cpu)
    cache.prefix_template = {name: [x.detach().clone() if x is not None else None
        for x in getattr(cache.native_cache, name)]
        for name in ("key_cache", "value_cache", "conv_states", "recurrent_states")}


def create_token_reduction_cache(qwen, batch_size: int, *, prefix_embeds=None,
        prefix_mask=None) -> TokenReductionCache:
    """Create a native state and optionally prefill its instruction once."""
    model = _text_model(qwen)
    install_token_reduction_attention(qwen)
    cache = _new_cache(model, batch_size)
    _prefill_prefix(model, cache, prefix_embeds, prefix_mask)
    return cache


def refresh_token_reduction_prefix(qwen, cache: TokenReductionCache,
        prefix_embeds: torch.Tensor, prefix_mask: torch.Tensor | None = None,
        row_mask: torch.Tensor | None = None) -> TokenReductionCache:
    """Replace instruction KV without resetting observed visual/KDA history.

    Changed instructions cannot rewrite nonlinear recurrent history already
    observed under a previous instruction. They replace the explicit instruction
    KV immediately. Reset templates are also refreshed, so a subsequent episode
    reset starts from the new instruction. Padded prefix width must stay fixed
    for partial-batch updates, matching the native streaming text protocol.
    """
    if prefix_embeds.ndim != 3 or prefix_embeds.shape[:2] != (cache.batch_size, cache.prefix_length):
        raise ValueError("Instruction refresh requires the same padded prefix width and batch size")
    selected = (torch.ones(cache.batch_size, dtype=torch.bool) if row_mask is None
        else torch.as_tensor(row_mask, dtype=torch.bool).detach().cpu())
    if tuple(selected.shape) != (cache.batch_size,):
        raise ValueError("Instruction refresh mask must have one entry per batch row")
    fresh = create_token_reduction_cache(qwen, cache.batch_size,
        prefix_embeds=prefix_embeds, prefix_mask=prefix_mask)
    for index in cache.native_cache.transformer_layers:
        for name in ("key_cache", "value_cache"):
            old = getattr(cache.native_cache, name)[index]
            new = getattr(fresh.native_cache, name)[index]
            row = selected.to(old.device).view(cache.batch_size, 1, 1, 1)
            prefix = torch.where(row, new.to(old), old[..., :cache.prefix_length, :])
            getattr(cache.native_cache, name)[index] = torch.cat((prefix,
                old[..., cache.prefix_length:, :]), dim=-2)
        meta = cache.metadata[index]
        valid = meta.valid.clone()
        valid[selected.to(valid.device), :cache.prefix_length] = fresh.prefix_valid.to(valid.device)[selected.to(valid.device)]
        meta.valid = valid
        meta.valid_cpu = meta.valid_cpu.clone()
        meta.valid_cpu[selected, :cache.prefix_length] = fresh.prefix_valid_cpu[selected]
    for name, values in fresh.prefix_template.items():
        old_values = cache.prefix_template[name]
        for index, new in enumerate(values):
            if new is not None:
                old = old_values[index]
                row = selected.to(old.device).view(cache.batch_size, *([1] * (old.ndim - 1)))
                old_values[index] = torch.where(row, new.to(old), old)
    cache.prefix_valid = torch.where(selected.to(cache.prefix_valid.device)[:, None],
        fresh.prefix_valid.to(cache.prefix_valid.device), cache.prefix_valid)
    cache.prefix_valid_cpu = torch.where(selected[:, None],
        fresh.prefix_valid_cpu, cache.prefix_valid_cpu)
    return cache


def _crop_layer(cache, index, window, latest):
    meta = cache.metadata[index]
    allowed = meta.prefix | (meta.valid_cpu & (meta.times > latest[:, None] - window + 1))
    keep = allowed.any(0).nonzero().flatten()
    cache.metadata[index] = LayerTokenMetadata(meta.times.index_select(1, keep),
        (meta.valid & allowed.to(meta.valid.device)).index_select(1, keep.to(meta.valid.device)),
        meta.prefix.index_select(1, keep),
        (meta.valid_cpu & allowed).index_select(1, keep))
    for name in ("key_cache", "value_cache"):
        value = getattr(cache.native_cache, name)[index]
        getattr(cache.native_cache, name)[index] = value.index_select(-2, keep.to(value.device))


def run_token_reduction_backbone(qwen, hidden_states: torch.Tensor,
        valid: torch.Tensor, timestep_positions: torch.Tensor, layout, *,
        cache: TokenReductionCache | None = None,
        stage_boundaries: Sequence[int] = (4, 8, 12, 16, 20),
        windows: Sequence[int] = (1, 2, 4, 8, 16, 32),
        max_attention_window_timesteps: int | None = None,
        kda_update_granularity: str = "token",
        prefix_embeds: torch.Tensor | None = None, prefix_mask: torch.Tensor | None = None,
        text_deepstack_memory=None, detach_cache: bool = True) -> TokenReductionOutput:
    """Run a chronological chunk while preserving original token identities.

    ``timestep_positions`` are physical elapsed times in reference-frame units;
    spatial IDs originate from the initial layout and are only gathered later.
    Prefix embeddings are consumed once when creating a fresh cache. Production
    frozen text-deepstack memory stays outside the main token sequence.
    """
    model = _text_model(qwen)
    if hidden_states.ndim != 4 or tuple(valid.shape) != tuple(hidden_states.shape[:3]):
        raise ValueError("Token-reduction inputs must be [B,T,N,D] with [B,T,N] validity")
    batch, frames, tokens, width = hidden_states.shape
    if tuple(timestep_positions.shape) != (batch, frames) or tokens != layout.initial_token_count:
        raise ValueError("Token-reduction timestamps/layout must match the initial frame sequence")
    bounds, windows = tuple(stage_boundaries), tuple(windows)
    if (max_attention_window_timesteps is not None
            and (type(max_attention_window_timesteps) is not int or max_attention_window_timesteps < 1)):
        raise ValueError("Maximum attention window must be None or a positive integer")
    if kda_update_granularity not in {"token", "timestep"}:
        raise ValueError("kda_update_granularity must be token or timestep")
    if (len(windows) != len(bounds) + 1 or layout.num_stages != len(windows)
            or any(type(x) is not int or x < 1 for x in (*bounds, *windows))
            or tuple(sorted(set(bounds))) != bounds or (bounds and bounds[-1] >= len(model.layers))):
        raise ValueError("Token-reduction stage boundaries/windows must form a valid decoder partition")
    if max_attention_window_timesteps is not None:
        windows = tuple(min(window, max_attention_window_timesteps) for window in windows)
    if getattr(model, "gradient_checkpointing", False) and model.training:
        raise ValueError("Token reduction requires stateless MLP checkpointing, not mutable decoder checkpoint replay")
    if cache is not None and cache.batch_size != batch:
        raise ValueError("Token-reduction cache batch size changed without an explicit reset")
    times = timestep_positions.detach().to(device="cpu", dtype=torch.float64)
    valid_cpu = valid.detach().to(device="cpu", dtype=torch.bool)
    if not torch.isfinite(times).all():
        raise ValueError("Token reduction requires finite timestamps")
    preserved_cache = None
    if not valid_cpu.any():
        if not isinstance(text_deepstack_memory, FrameDeepstackMemory) or model.training:
            raise ValueError("Token reduction requires at least one valid visual token")
        # HSDP peers must enter every wrapped layer on padded evaluation ticks.
        # Packed KDA kernels need a nonempty input. Run one zero frame against
        # scratch state, then discard both its outputs and its cache.
        preserved_cache = cache if cache is not None else _new_cache(model, batch)
        cache = None
        hidden_states = torch.zeros_like(hidden_states)
        valid_cpu = torch.zeros_like(valid_cpu)
        valid_cpu[0, 0] = True
        valid = valid_cpu.to(valid.device)
    for row in range(batch):
        observed = times[row, valid_cpu[row].any(-1)]
        if observed.numel() > 1 and not (observed.diff() > 0).all():
            raise ValueError("Observation timestamps must be strictly increasing")
        if (observed.numel() and cache is not None and cache.last_times is not None
                and observed[0] <= cache.last_times[row]):
            raise ValueError("Cached observation timestamps must start strictly after the previous valid observation; reset the row before reusing an episode")
    install_token_reduction_attention(qwen)
    if cache is None:
        cache = _new_cache(model, batch)
        _prefill_prefix(model, cache, prefix_embeds, prefix_mask)
    latest = torch.where(valid_cpu.any(-1), times, -torch.inf).max(-1).values
    if cache.last_times is not None:
        latest = torch.maximum(latest, cache.last_times)
    prefix_offset = (int(text_deepstack_memory.attention_mask.shape[-1])
        if text_deepstack_memory is not None else cache.prefix_length)
    spatial = layout.metadata(0, device=hidden_states.device).spatial_positions
    time_positions = timestep_positions.to(hidden_states.device)[:, :, None].expand(batch, frames, tokens)
    positions = torch.stack((time_positions,
        spatial[:, 0].view(1, 1, tokens).expand(batch, frames, -1),
        spatial[:, 1].view(1, 1, tokens).expand(batch, frames, -1))) + prefix_offset
    positions = positions.reshape(3, batch, frames * tokens)
    hidden = torch.where(valid[..., None], hidden_states, 0).reshape(batch, frames * tokens, width)
    # Match native Qwen: compute this call's rotary phases once, before decoder
    # execution. Later gathers retain these exact phases and cannot change a
    # dynamic RoPE base simply because a spatial endpoint was discarded.
    # Padded timestamps must not change a dynamic RoPE frequency base. Keep
    # semantic metadata intact, but omit invalid tokens from base selection.
    rotary_positions = torch.where(valid.flatten(1).unsqueeze(0), positions, prefix_offset)
    rotary = model.rotary_emb(hidden, rotary_positions)
    stage, stage_info = 0, []
    current_valid = valid.to(device=hidden.device, dtype=torch.bool)
    current_valid_cpu = valid_cpu
    stage_start = 0
    use_varlen = os.environ.get("PRISM_PACKED_VARLEN_GQA", "0") == "1"
    prefix_mask_cpu = prefix_owners = None
    if use_varlen and text_deepstack_memory is not None:
        prefix_mask_cpu = text_deepstack_memory.attention_mask.detach().to(device="cpu", dtype=torch.bool)
        prefix_owners = (text_deepstack_memory.indices.detach().reshape(-1).cpu()
                         if isinstance(text_deepstack_memory, FrameDeepstackMemory)
                         else torch.arange(batch).repeat_interleave(frames))
    for index, layer in enumerate(model.layers):
        flat_valid = current_valid.flatten(1)
        token_times = times[:, :, None].expand(batch, frames, tokens).reshape(batch, -1)
        context = None
        if model.config.layer_types[index] == "full_attention":
            previous = cache.metadata.get(index)
            meta = LayerTokenMetadata(token_times, flat_valid,
                torch.zeros(batch, frames * tokens, dtype=torch.bool),
                current_valid_cpu.flatten(1))
            if previous is not None:
                meta = LayerTokenMetadata(torch.cat((previous.times, token_times), 1),
                    torch.cat((previous.valid.to(flat_valid.device), flat_valid), 1),
                    torch.cat((previous.prefix, meta.prefix), 1),
                    torch.cat((previous.valid_cpu, meta.valid_cpu), 1))
            cache.metadata[index] = meta
            context = {"frames": frames, "tokens": tokens, "valid": current_valid,
                "metadata": meta, "packed_indices": _packed_window_indices(meta, times, windows[stage])}
            if use_varlen and (not layer.training or not layer.self_attn.attention_dropout):
                from prism_astribot_vla.modeling.token_reduction_varlen import build_ragged_frame_plan
                context["ragged_plan"] = build_ragged_frame_plan(
                    current_valid_cpu, context["packed_indices"], meta.times.shape[1],
                    prefix_mask_cpu, prefix_owners, hidden.device)

        # The scope can build its packed token indices from the already-copied
        # CPU metadata; do not synchronize a CUDA validity mask at every layer.
        timestep_scope = nullcontext()
        if kda_update_granularity == "timestep" and model.config.layer_types[index] == "linear_attention":
            from prism_astribot_vla.modeling.qwen35_timestep_kda import qwen35_timestep_scope
            timestep_scope = qwen35_timestep_scope(current_valid, valid_cpu=current_valid_cpu)
        with qwen35_valid_token_scope(current_valid_cpu.flatten(1), stream="main"), timestep_scope:
            hidden = layer(hidden, position_embeddings=rotary, attention_mask=None,
                position_ids=positions[0], past_key_values=cache.native_cache,
                use_cache=True, text_deepstack_memory=text_deepstack_memory,
                token_reduction_context=context)
        hidden = torch.where(flat_valid[..., None], hidden, 0)
        if context is not None:
            _crop_layer(cache, index, windows[stage], latest)
        if index + 1 in bounds or index + 1 == len(model.layers):
            stage_info.append({"stage": stage, "layers": (stage_start, index),
                "tokens_per_frame": tokens, "camera_counts": layout.stage_counts(stage),
                "window": windows[stage]})
            if index + 1 in bounds:
                selected = layout.transition_indices(stage, stage + 1, device=hidden.device)
                hidden = hidden.reshape(batch, frames, tokens, width).index_select(2, selected)
                current_valid = current_valid.index_select(2, selected)
                selected_cpu = layout.transition_indices(stage, stage + 1, device="cpu")
                current_valid_cpu = current_valid_cpu.index_select(2, selected_cpu)
                positions = positions.reshape(3, batch, frames, tokens).index_select(3, selected)
                rotary = tuple(phase.reshape(batch, frames, tokens, phase.shape[-1])
                    .index_select(2, selected).reshape(batch, frames * selected.numel(), phase.shape[-1])
                    for phase in rotary)
                tokens = selected.numel()
                hidden = hidden.reshape(batch, frames * tokens, width)
                positions = positions.reshape(3, batch, frames * tokens)
                stage += 1
                stage_start = index + 1
    hidden = model.norm(hidden).reshape(batch, frames, tokens, width)
    hidden = torch.where(current_valid[..., None], hidden, 0)
    cache.last_times = latest
    if detach_cache:
        cache.detach()
    if preserved_cache is not None:
        cache = preserved_cache
        hidden = torch.zeros_like(hidden)
        current_valid = torch.zeros_like(current_valid)
    return TokenReductionOutput(hidden, current_valid, cache, tuple(stage_info),
        positions.reshape(3, batch, frames, tokens))
