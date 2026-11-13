"""Activation recomputation and optional valid-row packing for Qwen MLPs."""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
from types import MethodType
from typing import Iterator

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


_ENABLED: ContextVar[bool] = ContextVar('prism_qwen_mlp_checkpoint_enabled', default=True)
_MARKER = '_prism_qwen_mlp_checkpoint_v1'
_PACK_MARKER = '_prism_qwen_mlp_pack_valid_tokens_v1'
_PACKING_COUNTS: dict[str, dict[str, int]] = {}


def _is_native_silu(activation: nn.Module) -> bool:
    # Transformers uses its own stateless F.silu wrapper for hidden_act=silu;
    # torch.nn.SiLU is the equivalent native module used for swish/CPU fixtures.
    from transformers.activations import SiLUActivation

    return (type(activation) in (nn.SiLU, SiLUActivation)
        and getattr(activation.forward, '__func__', None) is type(activation).forward)


def _is_native_bias_free_linear(layer: nn.Module) -> bool:
    return (type(layer) is nn.Linear and layer.bias is None
        and getattr(layer.forward, '__func__', None) is nn.Linear.forward)


def qwen_mlp_packing_counts(*, reset: bool = False) -> dict[str, dict[str, int]]:
    """Read opt-in CPU row counters, without inspecting device tensor values.

    Set ``RB_OPT_VERIFY_MLP_PACKING=1`` before a qualification forward. Keys
    identify the shared token scope and retained physical decoder layer. All
    valid ordinary scopes use native fallback and have no layout, so their
    counters are explicitly labelled ``unscoped_or_all_valid``. Replay is not
    counted twice: observation happens outside the checkpoint closure.
    """
    result = {key: dict(value) for key, value in _PACKING_COUNTS.items()}
    if reset:
        _PACKING_COUNTS.clear()
    return result


def _record_packing_rows(stream: str, layer: int, rows: int, selected: int,
                         *, packed: bool, checkpointed: bool) -> None:
    if os.environ.get('RB_OPT_VERIFY_MLP_PACKING', '0') != '1':
        return
    counts = _PACKING_COUNTS.setdefault(f'{stream}.layer_{layer}',
        dict(calls=0, input_rows=0, mlp_rows=0, packed_calls=0, checkpointed_calls=0))
    counts['calls'] += 1
    counts['input_rows'] += rows
    counts['mlp_rows'] += selected
    counts['packed_calls'] += int(packed)
    counts['checkpointed_calls'] += int(checkpointed)


def validate_qwen_mlp_packing_backbone(backbone: nn.Module) -> None:
    """Require the native zero-preserving architecture used by camera PRISM.

    A missing current camera can reappear at its prediction target. Its MAIN
    hidden row is consequently still consumed by loss/action code. Native
    zero-masked compacts, RMS norms, bias-free attention and SwiGLU preserve
    that exact-zero row; SwiGLU also has zero input/parameter Jacobians there.
    Query-stream missing rows are instead explicitly masked after the model.
    Do not broaden this guard to architectures with offsets or stochastic MLPs.
    Installation precedes FSDP/DMuon. Opaque pre-existing activation hooks are
    rejected; later native materialization/diagnostic hooks must preserve tensor
    values. No hooks are removed or bypassed by packing/recomputation.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    if (type(backbone) is not hf.Qwen3_5TextModel
            or getattr(backbone.forward, '__func__', None) is not hf.Qwen3_5TextModel.forward):
        raise ValueError('Qwen MLP packing requires the native Qwen3.5 backbone without boundary operators')
    hook_fields = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
    if any(getattr(module, name, {}) for module in backbone.modules() for name in hook_fields):
        raise ValueError('Qwen MLP packing requires a fresh backbone without pre-existing activation hooks')
    module_hooks = torch.nn.modules.module
    if any(getattr(module_hooks, '_global' + name, {}) for name in hook_fields):
        raise ValueError('Qwen MLP packing requires no pre-existing global activation hooks')
    layers = tuple(getattr(backbone, 'layers', ()))
    if not layers:
        raise ValueError('Qwen MLP packing requires native Qwen3.5 decoder layers')
    for layer in layers:
        if (type(layer) is not hf.Qwen3_5DecoderLayer
                or getattr(layer.forward, '__func__', None) is not hf.Qwen3_5DecoderLayer.forward):
            raise ValueError('Qwen MLP packing requires native Qwen3.5 decoder layers')
        if (type(layer.mlp) is not hf.Qwen3_5MLP
                or not _is_native_silu(layer.mlp.act_fn)
                or getattr(layer.mlp.forward, '__func__', None) is not hf.Qwen3_5MLP.forward):
            raise ValueError('Qwen MLP packing requires native bias-free SiLU-gated MLPs')
        norms = [layer.input_layernorm, layer.post_attention_layernorm, backbone.norm]
        if layer.layer_type == 'full_attention':
            norms.extend((layer.self_attn.q_norm, layer.self_attn.k_norm))
        for norm in norms:
            if (type(norm) is not hf.Qwen3_5RMSNorm
                    or getattr(norm.forward, '__func__', None) is not hf.Qwen3_5RMSNorm.forward
                    or type(norm.eps) not in (int, float)
                    or not math.isfinite(float(norm.eps)) or float(norm.eps) <= 0):
                raise ValueError('Qwen MLP packing requires native zero-preserving RMS norms')
        if layer.layer_type == 'full_attention':
            attention = layer.self_attn
            if (type(attention) is not hf.Qwen3_5Attention
                    or getattr(attention.forward, '__func__', None) is not hf.Qwen3_5Attention.forward
                    or type(attention.attention_dropout) not in (int, float)
                    or float(attention.attention_dropout) != 0.0
                    or any(not _is_native_bias_free_linear(getattr(attention, name))
                        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))):
                raise ValueError('Qwen MLP packing requires bias-free native attention with dropout=0')
        elif layer.layer_type == 'linear_attention':
            if (type(layer.linear_attn) is not hf.Qwen3_5GatedDeltaNet
                    or getattr(layer.linear_attn.forward, '__func__', None) is not hf.Qwen3_5GatedDeltaNet.forward
                    or not getattr(layer.linear_attn, '_prism_chunk_cache_patch', False)):
                raise ValueError('Qwen MLP packing requires the scoped Qwen3.5 linear-attention patch')
        else:
            raise ValueError('Qwen MLP packing requires native hybrid Qwen3.5 layers')
        if any(not _is_native_bias_free_linear(getattr(layer.mlp, name))
                for name in ('gate_proj', 'up_proj', 'down_proj')):
            raise ValueError('Qwen MLP packing requires bias-free native MLP projections')


@contextmanager
def qwen_mlp_checkpointing(enabled: bool) -> Iterator[None]:
    """Temporarily select checkpointed or eager MLPs without replacing weights."""
    if type(enabled) is not bool:
        raise ValueError('Qwen MLP checkpoint selection must be a boolean')
    token = _ENABLED.set(enabled)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def install_qwen_mlp_checkpointing(
    layers: Sequence[nn.Module], *, pack_valid_tokens: bool = False,
) -> dict[str, int]:
    """Wrap only native gated MLP forwards before FSDP/DMuon installation.

    Every Linear still executes its ordinary module hooks. Complete nonreentrant
    replay preserves materialization-hook pairs; the parent decoder's native
    backward boundary must restore its ordinary/dedicated weights as usual.
    No attention, mutable cache, KDA operation, module or parameter is replaced.
    """
    if type(pack_valid_tokens) is not bool:
        raise ValueError('Qwen MLP valid-token packing selection must be a boolean')
    layers = list(layers)
    if not layers:
        raise ValueError('Qwen MLP checkpointing requires decoder layers')
    mlps = [getattr(layer, 'mlp', None) for layer in layers]
    if len({id(mlp) for mlp in mlps}) != len(mlps):
        raise ValueError('Qwen decoder MLPs must be distinct')
    for mlp in mlps:
        if not isinstance(mlp, nn.Module) or not all(isinstance(getattr(mlp, key, None), nn.Linear)
                for key in ('gate_proj', 'up_proj', 'down_proj')):
            raise ValueError('Expected native Qwen gated Linear MLPs')
        if not getattr(mlp, _MARKER, False) and not isinstance(mlp.forward, MethodType):
            raise ValueError('Install Qwen MLP checkpointing before forward compilation')
        if pack_valid_tokens and (not _is_native_silu(getattr(mlp, 'act_fn', None))
                or any(not _is_native_bias_free_linear(getattr(mlp, key))
                    for key in ('gate_proj', 'up_proj', 'down_proj'))):
            raise ValueError('Qwen MLP valid-token packing requires bias-free SiLU-gated MLPs')
    installed = 0
    for layer_index, mlp in enumerate(mlps):
        if pack_valid_tokens:
            setattr(mlp, _PACK_MARKER, True)
        if getattr(mlp, _MARKER, False):
            continue
        # Capture the function, not a bound original module: deepcopy of an
        # installed model must recompute against the copied module's weights.
        original = mlp.forward.__func__

        def bind(forward, physical_layer):
            def checkpointed(module, x):
                replay = module.training and torch.is_grad_enabled() and _ENABLED.get()
                value, indices = x, None
                packing = bool(getattr(module, _PACK_MARKER, False))
                if packing:
                    from prism_vla.modeling.qwen35_deltanet_patch import get_qwen35_valid_token_layout

                    layout = get_qwen35_valid_token_layout()
                    rows = x.numel() // x.shape[-1]
                    if layout is not None and (x.ndim != 3 or tuple(x.shape[:2]) != layout.shape):
                        raise ValueError('Scoped Qwen valid-token mask must match the current [B, T] query')
                    selected = rows if layout is None else layout.flat_indices.numel()
                    if layout is not None and selected != rows:
                        # Reuse the KDA scope's cached indices (including every
                        # private ACT), before entering any replay closure.
                        indices = layout.indices_for(x)
                        value = x.flatten(0, 1).index_select(0, indices)
                    _record_packing_rows('unscoped_or_all_valid' if layout is None else layout.stream,
                        physical_layer, rows, selected, packed=indices is not None,
                        checkpointed=replay)
                if replay:
                    result = checkpoint(lambda values: forward(module, values), value,
                        use_reentrant=False, preserve_rng_state=True, early_stop=False)
                else:
                    result = forward(module, value)
                if indices is not None:
                    return result.new_zeros((x.shape[0] * x.shape[1], x.shape[-1])).index_copy(
                        0, indices, result).reshape_as(x)
                return result
            return checkpointed

        mlp.forward = MethodType(bind(original, layer_index), mlp)
        setattr(mlp, _MARKER, True)
        installed += 1
    result = dict(installed_layers=installed, total_layers=len(layers))
    if pack_valid_tokens:
        result['packed_layers'] = len(layers)
    return result
