"""Frozen text K/V as an external bank in native Qwen full attention.

Attention math follows Hugging Face Transformers' Qwen3 and Qwen3.5 native
implementations (Apache-2.0). Native parameters, projections, normalization,
own-cache writes and Qwen3.5's output gate retain their module boundaries.
External keys are stored before RoPE, then encoded once in the target frame.
No source convolution or recurrent cache is retained or passed to the target.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import MethodType
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TextLayerKV:
    """One physical source layer's detached canonical keys and values [B,H,T,D]."""

    layer_index: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class TextDeepstackMemory:
    """Immutable per-call text context; safe to retain in checkpoint closures."""

    layers: tuple[TextLayerKV, ...]
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    @property
    def position_offsets(self) -> torch.Tensor:
        # Reserve the padded instruction width, so changing a row's real text
        # length cannot change coordinates of native keys retained in its cache.
        return torch.full((self.attention_mask.shape[0],), self.attention_mask.shape[1],
            dtype=torch.long, device=self.position_ids.device)

    def layer(self, index: int) -> TextLayerKV:
        for layer in self.layers:
            if layer.layer_index == index:
                return layer
        raise ValueError(f"Text deepstack source layer {index} is absent")

    @torch.inference_mode(False)
    @torch.no_grad()
    def index_select(self, indices: torch.Tensor) -> TextDeepstackMemory:
        def select(value):
            return value.index_select(0, indices.to(device=value.device))
        return TextDeepstackMemory(tuple(TextLayerKV(layer.layer_index,
            select(layer.key), select(layer.value)) for layer in self.layers),
            select(self.attention_mask), select(self.position_ids))

    @torch.inference_mode(False)
    @torch.no_grad()
    def replace_rows(self, replacement: TextDeepstackMemory,
                     mask: torch.Tensor) -> TextDeepstackMemory:
        """Return a fresh snapshot, preserving rows outside a reset/refresh mask."""
        if (self.attention_mask.shape != replacement.attention_mask.shape
                or self.position_ids.shape != replacement.position_ids.shape
                or tuple(layer.layer_index for layer in self.layers)
                != tuple(layer.layer_index for layer in replacement.layers)
                or mask.ndim != 1 or mask.shape[0] != self.attention_mask.shape[0]):
            raise ValueError("Text deepstack row replacement requires identical layout and batch shape")
        def replace(current, new):
            if current.shape != new.shape:
                raise ValueError("Text deepstack row replacement requires identical K/V geometry")
            selected = mask.to(device=current.device, dtype=torch.bool).view(
                current.shape[0], *((1,) * (current.ndim - 1)))
            return torch.where(selected, new.to(device=current.device, dtype=current.dtype), current)
        return TextDeepstackMemory(tuple(TextLayerKV(old.layer_index,
            replace(old.key, new.key), replace(old.value, new.value))
            for old, new in zip(self.layers, replacement.layers)),
            replace(self.attention_mask, replacement.attention_mask),
            replace(self.position_ids, replacement.position_ids))

    @torch.inference_mode(False)
    @torch.no_grad()
    def update_rows(self, indices: torch.Tensor,
                    replacement: TextDeepstackMemory) -> TextDeepstackMemory:
        """Refresh a selected subbatch without recomputing other source rows."""
        batch = self.attention_mask.shape[0]
        if (indices.ndim != 1 or indices.dtype != torch.long
                or indices.unique().numel() != indices.numel()
                or bool(((indices < 0) | (indices >= batch)).any())):
            raise ValueError("Text deepstack row updates require unique in-range integer indices")
        count = indices.numel()
        if (replacement.attention_mask.shape != (count, self.attention_mask.shape[1])
                or replacement.position_ids.shape != (count, self.position_ids.shape[1])
                or tuple(layer.layer_index for layer in self.layers)
                != tuple(layer.layer_index for layer in replacement.layers)):
            raise ValueError("Text deepstack row updates require identical layer layout and text width")
        def update(current, new):
            if new.shape != (count, *current.shape[1:]):
                raise ValueError("Text deepstack row updates require identical K/V geometry")
            return current.index_copy(0, indices.to(device=current.device),
                new.to(device=current.device, dtype=current.dtype))
        return TextDeepstackMemory(tuple(TextLayerKV(old.layer_index,
            update(old.key, new.key), update(old.value, new.value))
            for old, new in zip(self.layers, replacement.layers)),
            update(self.attention_mask, replacement.attention_mask),
            update(self.position_ids, replacement.position_ids))


def _text_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
        return model
    if hasattr(model, "model"):
        return _text_model(model.model)
    raise ValueError("Text deepstack requires a native Qwen text decoder")


def _attention_layers(model: nn.Module) -> dict[int, nn.Module]:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Model
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextModel

    model = _text_model(model)
    native_classes = (Qwen3Model, Qwen3_5TextModel)
    native_decoder = type(model) in native_classes
    if not native_decoder:
        from torch.distributed.fsdp import FSDPModule

        # Composable FSDP mutates the class in place to (FSDPModule, Orig).
        # Permit that known wrapper over the exact native architecture; an
        # arbitrary Qwen subclass (including one wrapped by FSDP) stays unsupported.
        bases = type(model).__bases__
        native_decoder = (isinstance(model, FSDPModule) and len(bases) == 2
            and bases[0] is FSDPModule and bases[1] in native_classes
            and isinstance(model, bases[1]))
    if not native_decoder:
        raise ValueError(f"Unsupported text deepstack decoder: {type(model).__name__}")
    result = {}
    for index, layer in enumerate(model.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            if getattr(layer, "layer_type", None) != "linear_attention":
                raise ValueError(f"Unsupported text deepstack mixer at layer {index}")
            continue
        if type(attention) not in (Qwen3Attention, Qwen3_5Attention):
            raise ValueError(f"Unsupported text deepstack attention at layer {index}")
        if getattr(attention, "sliding_window", None) is not None:
            raise ValueError("Text deepstack currently requires full, non-sliding attention")
        result[index] = attention
    if not result:
        raise ValueError("Text deepstack requires at least one full-attention layer")
    return result


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def canonicalize_rotary_key(key: torch.Tensor, cos: torch.Tensor,
                           sin: torch.Tensor) -> torch.Tensor:
    """Invert the actual source RoPE, including partial rotation and scaling."""
    if cos.shape != sin.shape or cos.ndim != 3 or cos.shape[-1] % 2:
        raise ValueError("Text deepstack requires native half-split rotary cos/sin [B,T,R]")
    width = cos.shape[-1]
    if width > key.shape[-1] or cos.shape[:2] != (key.shape[0], key.shape[-2]):
        raise ValueError("Source rotary metadata does not match its cached keys")
    cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    rotated = key[..., :width].float()
    canonical = (rotated * cos - _rotate_half(rotated) * sin) / (cos.square() + sin.square())
    return torch.cat((canonical.to(key.dtype), key[..., width:]), dim=-1)


@torch.inference_mode(False)
@torch.no_grad()
def extract_text_deepstack_memory(encoder: nn.Module, *, input_ids: torch.Tensor,
        attention_mask: torch.Tensor, position_ids: torch.Tensor | None = None,
        layer_indices: Sequence[int] | None = None) -> TextDeepstackMemory:
    """Run the frozen text model once and retain only canonical full-layer K/V."""
    encoder = _text_model(encoder)
    layers = _attention_layers(encoder)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Text deepstack instruction encoder must be frozen")
    if encoder.training:
        raise ValueError("Text deepstack instruction encoder must be in eval mode")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or input_ids.shape[1] == 0:
        raise ValueError("Text deepstack requires nonempty text IDs and a matching [B,T] mask")
    mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    if not bool(mask.any(-1).all()):
        raise ValueError("Each text deepstack row must contain a valid text token")
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0) if position_ids is None else position_ids
    if positions.shape != input_ids.shape or positions.ndim != 2:
        raise ValueError("Text deepstack requires ordinary text positions [B,T]")
    selected = tuple(layers) if layer_indices is None else tuple(layer_indices)
    if not selected or len(set(selected)) != len(selected) or any(index not in layers for index in selected):
        raise ValueError("Text deepstack source mapping must select distinct full-attention layers")
    rotary = []
    handle = encoder.rotary_emb.register_forward_hook(lambda module, args, output: rotary.append(output))
    try:
        output = encoder(input_ids=input_ids, attention_mask=mask, position_ids=positions,
            use_cache=True, return_dict=True)
    finally:
        handle.remove()
    if len(rotary) != 1:
        raise ValueError("Cannot identify the frozen encoder's actual rotary metadata")
    cache = output.past_key_values
    if cache is None:
        raise ValueError("Frozen text encoder did not produce native full-attention K/V")
    result = []
    for index in selected:
        if hasattr(cache, "key_cache"):
            key, value = cache.key_cache[index], cache.value_cache[index]
        elif hasattr(cache, "layers"):
            key, value = cache.layers[index].keys, cache.layers[index].values
        else:
            raise ValueError("Unsupported frozen text full-attention cache format")
        if not torch.is_tensor(key) or key.ndim != 4 or value.shape != key.shape:
            raise ValueError("Frozen text source layer has no native full-attention K/V")
        canonical = canonicalize_rotary_key(key, *rotary[0])
        # Zero padding before projection, too: masked NaNs must never enter matmul.
        valid = mask[:, None, :, None]
        result.append(TextLayerKV(index, canonical.masked_fill(~valid, 0).detach().clone(),
            value.masked_fill(~valid, 0).detach().clone()))
    return TextDeepstackMemory(tuple(result), mask.detach().clone(), positions.detach().clone())


class TextDeepstackAdapter(nn.Module):
    """Identity at matching KV geometry; otherwise independent K and V maps."""

    def __init__(self, *, source_heads: int, source_head_dim: int,
            target_heads: int, target_head_dim: int, rotary_emb: nn.Module,
            source_layer: int) -> None:
        super().__init__()
        if getattr(rotary_emb, "rope_type", None) != "default":
            raise ValueError("Text deepstack target currently requires default deterministic RoPE")
        self.source_layer = source_layer
        self.source_geometry = (source_heads, source_head_dim)
        self.target_geometry = (target_heads, target_head_dim)
        compatible = self.source_geometry == self.target_geometry
        self.source_num_kv_heads = source_heads
        self.source_head_dim = source_head_dim
        self.target_num_kv_heads = target_heads
        self.target_head_dim = target_head_dim
        self.is_identity = compatible
        self.k_projection = nn.Identity() if compatible else nn.Linear(
            source_heads * source_head_dim, target_heads * target_head_dim, bias=False)
        self.v_projection = nn.Identity() if compatible else nn.Linear(
            source_heads * source_head_dim, target_heads * target_head_dim, bias=False)
        self.register_buffer("inv_freq", rotary_emb.inv_freq.detach().float().clone(), persistent=False)
        self.attention_scaling = float(rotary_emb.attention_scaling)

    def forward(self, memory: TextDeepstackMemory, reference: torch.Tensor):
        layer = memory.layer(self.source_layer)
        if layer.key.requires_grad or layer.value.requires_grad:
            raise ValueError("Frozen text deepstack K/V must be detached")
        if layer.key.shape != layer.value.shape or layer.key.ndim != 4:
            raise ValueError("Text deepstack K/V geometry is invalid")
        batch, heads, length, dim = layer.key.shape
        if (heads, dim) != self.source_geometry or batch != reference.shape[0]:
            raise ValueError("Text deepstack K/V geometry does not match mapped attention")
        if memory.attention_mask.shape != (batch, length) or memory.position_ids.shape != (batch, length):
            raise ValueError("Text deepstack mask and positions do not match K/V")
        mask = memory.attention_mask.to(device=reference.device, dtype=torch.bool)
        def project(value, projection):
            value = value.to(device=reference.device, dtype=reference.dtype).masked_fill(~mask[:, None, :, None], 0)
            if isinstance(projection, nn.Identity):
                return projection(value)
            value = projection(value.transpose(1, 2).reshape(batch, length, heads * dim))
            return value.view(batch, length, *self.target_geometry).transpose(1, 2)
        key, value = project(layer.key, self.k_projection), project(layer.value, self.v_projection)
        positions = memory.position_ids.to(device=reference.device)
        angles = positions.float()[..., None] * self.inv_freq.float()[None, None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos = (angles.cos() * self.attention_scaling).to(key.dtype).unsqueeze(1)
        sin = (angles.sin() * self.attention_scaling).to(key.dtype).unsqueeze(1)
        width = cos.shape[-1]
        rotated = key[..., :width] * cos + _rotate_half(key[..., :width]) * sin
        return torch.cat((rotated, key[..., width:]), dim=-1), value, mask


def _joint_attention_mask(mask, text_valid, *, queries, own_keys, dtype, device):
    batch, text_length = text_valid.shape
    if mask is None:
        query_slots = torch.arange(own_keys - queries, own_keys, device=device)
        own = torch.arange(own_keys, device=device)[None, :] <= query_slots[:, None]
        own = own[None, None].expand(batch, 1, -1, -1)
    elif mask.ndim == 4 and mask.shape[-2:] == (queries, own_keys):
        own = mask.to(device=device)
    else:
        raise ValueError("Text deepstack attention requires a native explicit 4D own-token mask")
    text = text_valid[:, None, None, :].expand(batch, own.shape[1], queries, text_length)
    if own.dtype == torch.bool:
        allowed = torch.cat((text, own.expand(batch, -1, -1, -1)), dim=-1)
        return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, float("-inf"))
    text_bias = torch.zeros(text.shape, dtype=dtype, device=device).masked_fill(~text, float("-inf"))
    return torch.cat((text_bias, own.to(dtype=dtype).expand(batch, -1, -1, -1)), dim=-1)


def _deepstack_forward(self, hidden_states, position_embeddings, attention_mask,
        past_key_values=None, text_deepstack_memory=None, **kwargs):
    if text_deepstack_memory is None:
        return self._text_deepstack_native_forward(hidden_states=hidden_states,
            position_embeddings=position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs)
    if type(self) is not self._text_deepstack_native_type:
        raise ValueError("Text deepstack requires unwrapped native full-attention modules; wrap decoder layers instead")
    module = importlib.import_module(type(self).__module__)
    shape = hidden_states.shape[:-1]
    head_shape = (*shape, -1, self.head_dim)
    gated = type(self).__name__ == "Qwen3_5Attention"
    if gated:
        query, gate = self.q_proj(hidden_states).view(*shape, -1, 2 * self.head_dim).chunk(2, -1)
        query = query.reshape(head_shape)
    else:
        query = self.q_proj(hidden_states).view(head_shape)
    query = self.q_norm(query).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    query, key = module.apply_rotary_pos_emb(query, key, *position_embeddings)
    if past_key_values is not None:
        boundary = getattr(self, "_prism_qwen_cache_update", None)
        key, value = (boundary(past_key_values, key, value) if callable(boundary)
            else past_key_values.update(key, value, self.layer_idx))
    # Cache update is complete before external tensors are joined; native cache
    # contains only PRISM's own states, including under rolling continuation.
    text_key, text_value, valid = self.text_deepstack_adapter(text_deepstack_memory, query)
    mask = _joint_attention_mask(attention_mask, valid, queries=query.shape[-2],
        own_keys=key.shape[-2], dtype=query.dtype, device=query.device)
    key, value = torch.cat((text_key, key), -2), torch.cat((text_value, value), -2)
    # Explicit mask is authoritative; native FlashAttention's inferred causal
    # alignment cannot describe an external noncausal text prefix with holes.
    groups = query.shape[1] // key.shape[1]
    key, value = module.repeat_kv(key, groups), module.repeat_kv(value, groups)
    implementation = self.config._attn_implementation
    if implementation == "eager":
        weights = torch.softmax(query @ key.transpose(-2, -1) * self.scaling + mask,
            dim=-1, dtype=torch.float32).to(query.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        output = (weights @ value).transpose(1, 2)
    elif implementation == "sdpa":
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False, scale=self.scaling).transpose(1, 2)
        weights = None
    else:
        raise ValueError("Text deepstack supports native eager or SDPA attention only")
    output = output.reshape(*shape, -1).contiguous()
    if gated:
        output = output * torch.sigmoid(gate.reshape(*shape, -1))
    return self.o_proj(output), weights


def install_text_deepstack_attention(qwen: nn.Module, source_encoder: nn.Module, *,
        layer_mapping: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Install explicit (physical target, physical source) full-attention routing.

    Call before wrapping/sharding/compiling the trainable model. Each target
    full-attention layer can be mapped at most once. Source layers may repeat.
    No source modules or tensors are registered beneath the target backbone.
    """
    target, source = _text_model(qwen), _text_model(source_encoder)
    targets, sources = _attention_layers(target), _attention_layers(source)
    mapping = tuple(tuple(pair) for pair in layer_mapping)
    if not mapping or any(len(pair) != 2 or any(type(index) is not int for index in pair) for pair in mapping):
        raise ValueError("Text deepstack layer mapping must contain explicit integer target/source pairs")
    if len({pair[0] for pair in mapping}) != len(mapping) or not {pair[0] for pair in mapping}.issubset(targets):
        raise ValueError("Text deepstack mapping must select distinct target full-attention layers")
    if any(source_index not in sources for _, source_index in mapping):
        raise ValueError("Text deepstack mapping may select only source full-attention layers")
    adapters = []
    for target_index, source_index in mapping:
        attention, source_attention = targets[target_index], sources[source_index]
        if getattr(attention, "_text_deepstack_native_forward", None) is not None:
            raise ValueError("Text deepstack attention is already installed")
        if attention.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Text deepstack supports native eager or SDPA attention only")
        adapter = TextDeepstackAdapter(source_heads=source_attention.config.num_key_value_heads,
            source_head_dim=source_attention.head_dim,
            target_heads=attention.config.num_key_value_heads, target_head_dim=attention.head_dim,
            rotary_emb=target.rotary_emb, source_layer=source_index)
        adapter.to(device=attention.k_proj.weight.device, dtype=attention.k_proj.weight.dtype)
        adapters.append((attention, adapter))
    for attention, adapter in adapters:
        attention.text_deepstack_adapter = adapter
        attention._text_deepstack_native_type = type(attention)
        attention._text_deepstack_native_forward = attention.forward
        attention.forward = MethodType(_deepstack_forward, attention)
    return mapping
