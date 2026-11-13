"""Qwen backbone topology and compilation utilities."""

from __future__ import annotations

import ast
import importlib
import re
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn


_QWEN_LAYER_KEY_RE = re.compile(
    r"^(qwen\.(?:model|transformer)\.(?:layers|h)\.)(\d+)(\..*)$"
)


@dataclass(frozen=True)
class BackboneLayerSkipInfo:
    """Metadata describing the active Qwen layer subset."""

    keep_layers: tuple[int, ...]
    original_num_layers: int
    active_num_layers: int

    @property
    def enabled(self) -> bool:
        return bool(self.keep_layers) and self.keep_layers != tuple(
            range(self.original_num_layers)
        )


@dataclass(frozen=True)
class QwenLayerStateRemapInfo:
    """Checkpoint key remapping statistics."""

    remapped_keys: int
    dropped_keys: int


@dataclass(frozen=True)
class QwenBackboneCompileInfo:
    """Installed decoder compilation coverage and eager boundaries."""

    installed_layers: int
    total_layers: int
    linear_attention_layers: int
    full_attention_layers: int
    installed_stateful_boundaries: int
    installed_cache_boundaries: int


def normalize_qwen_keep_layers(value: Any) -> tuple[int, ...]:
    """Normalize a config/CLI layer list.

    Empty/``None`` means keep the full pretrained backbone. Non-empty lists are
    original pretrained layer indices, for example ``[0, 4, 8, 12, 16, 20, 24,
    28]`` for a 32-layer Qwen backbone.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"all", "none", "null", "false", "no", "[]"}:
            return ()
        if text.startswith(("[", "(")):
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid qwen_keep_layers literal: {value!r}") from exc
        else:
            value = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(value, int) and not isinstance(value, bool):
        value = [value]

    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ValueError(
            "qwen_keep_layers must be empty, a comma-separated string, or a sequence of ints; "
            f"got {type(value).__name__}."
        )

    layers: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("qwen_keep_layers entries must be integer indices, not booleans.")
        try:
            idx = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid qwen_keep_layers entry: {item!r}") from exc
        if idx < 0:
            raise ValueError(f"qwen_keep_layers entries must be non-negative, got {idx}.")
        layers.append(idx)

    if not layers:
        return ()
    if len(set(layers)) != len(layers):
        raise ValueError(f"qwen_keep_layers must not contain duplicates, got {layers}.")
    if layers != sorted(layers):
        raise ValueError(
            "qwen_keep_layers must be strictly increasing original layer indices, "
            f"got {layers}."
        )
    return tuple(layers)


def _get_nested_attr(root: Any, path: tuple[str, ...]) -> Any:
    value = root
    for name in path:
        value = getattr(value, name)
    return value


def _find_decoder_layers(qwen: nn.Module) -> tuple[Any, str, nn.ModuleList]:
    """Return ``(parent, attr_name, layers)`` for common causal LM layouts."""
    for path in (("model", "layers"), ("transformer", "h")):
        try:
            parent = _get_nested_attr(qwen, path[:-1])
            layers = getattr(parent, path[-1])
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList):
            return parent, path[-1], layers
    raise ValueError(
        "Could not locate Qwen decoder layers. Expected qwen.model.layers "
        "or qwen.transformer.h to be an nn.ModuleList."
    )


def _compile_disable(function: Callable[..., Any], *, reason: str) -> Callable[..., Any]:
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if not callable(disable):
        raise RuntimeError("Qwen backbone compilation requires torch.compiler.disable().")
    try:
        disabled = disable(function, reason=reason)
    except TypeError:
        disabled = disable(function)
    if not callable(disabled):
        raise RuntimeError("torch.compiler.disable() returned a non-callable value.")
    return disabled


def _prepare_qwen_attention_cache_boundary(
    attention: nn.Module,
) -> tuple[Callable[..., Any] | None, Callable[..., Any]]:
    """Build an eager cache helper and, for upstream Qwen3.5, its caller."""
    method_name = "_prism_qwen_cache_update"
    existing = getattr(attention, method_name, None)
    if callable(existing):
        return None, _compile_disable(
            existing,
            reason="PRISM Qwen mutable KV-cache update",
        )

    if type(attention).__name__ != "Qwen3_5Attention":
        raise RuntimeError(
            "Qwen full-attention compilation requires a cache-boundary adapter for "
            f"{type(attention).__module__}.{type(attention).__name__}."
        )
    module = importlib.import_module(type(attention).__module__)
    apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb", None)
    attention_functions = getattr(module, "ALL_ATTENTION_FUNCTIONS", None)
    eager_attention_forward = getattr(module, "eager_attention_forward", None)
    if (
        not callable(apply_rotary_pos_emb)
        or attention_functions is None
        or not callable(eager_attention_forward)
    ):
        raise RuntimeError(
            "Installed Transformers Qwen3.5 attention internals are incompatible "
            "with decoder-region compilation."
        )

    def cache_update(
        past_key_values: Any,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep layer_idx lookup and Python-list mutation outside Dynamo.
        return past_key_values.update(
            key_states,
            value_states,
            attention.layer_idx,
        )

    disabled_update = _compile_disable(
        cache_update,
        reason="PRISM Qwen mutable KV-cache update",
    )

    def forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )
        if past_key_values is not None:
            key_states, value_states = self._prism_qwen_cache_update(
                past_key_values,
                key_states,
                value_states,
            )

        attention_interface = attention_functions.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights

    return types.MethodType(forward, attention), disabled_update


def compile_qwen_backbone_decoder_regions(
    qwen: nn.Module,
    *,
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = False,
    compiler: Callable[..., Callable[..., Any]] | None = None,
) -> QwenBackboneCompileInfo:
    """Compile decoder tensor regions while keeping mutable cache paths eager.

    Each decoder ``forward`` is compiled with graph breaks enabled. Qwen3.5's
    DeltaNet convolution/recurrent core and dynamic KV-cache update are marked
    as eager boundaries, so Dynamo can capture the projections, attention math,
    output projections, residuals, normalization, and MLP around them without
    guarding on mutable Python cache state.
    """

    _, _, layers = _find_decoder_layers(qwen)
    if any(hasattr(getattr(layer, "self_attn", None), "text_deepstack_adapter") for layer in layers):
        raise ValueError(
            "Text input deepstack currently requires compile_qwen_backbone=false; "
            "native decoder compilation would replace its external-text attention"
        )
    if len({id(layer) for layer in layers}) != len(layers):
        raise RuntimeError("Qwen backbone compilation found duplicate decoder layers.")

    linear_mixers: list[nn.Module] = []
    attention_mixers: list[nn.Module] = []
    for index, layer in enumerate(layers):
        if not isinstance(getattr(layer, "mlp", None), nn.Module):
            raise RuntimeError(
                f"Qwen backbone compilation could not locate an MLP at layer {index}."
            )
        linear_mixer = getattr(layer, "linear_attn", None)
        attention_mixer = getattr(layer, "self_attn", None)
        if isinstance(linear_mixer, nn.Module) and not isinstance(attention_mixer, nn.Module):
            stateful_core = getattr(linear_mixer, "_prism_qwen35_stateful_core", None)
            if not callable(stateful_core):
                raise RuntimeError(
                    "Qwen linear-attention compilation requires the PRISM Qwen3.5 "
                    f"DeltaNet cache patch at layer {index}."
                )
            linear_mixers.append(linear_mixer)
        elif isinstance(attention_mixer, nn.Module) and not isinstance(linear_mixer, nn.Module):
            attention_mixers.append(attention_mixer)
        else:
            raise RuntimeError(
                "Qwen backbone compilation expected exactly one token mixer at "
                f"layer {index}."
            )

    all_mixers = [*linear_mixers, *attention_mixers]
    if len({id(target) for target in all_mixers}) != len(all_mixers):
        raise RuntimeError("Qwen backbone compilation found duplicate token mixers.")

    compile_fn = torch.compile if compiler is None else compiler
    if not callable(compile_fn):
        raise RuntimeError("Qwen backbone compiler is not callable.")
    requested = (str(backend), str(mode), bool(dynamic))
    root_marker = "_prism_qwen_decoder_compile_spec"
    installed_root = getattr(qwen, root_marker, None)
    if installed_root is not None:
        if tuple(installed_root) != requested:
            raise RuntimeError(
                "Qwen backbone compiler options changed after installation: "
                f"installed={tuple(installed_root)!r}, requested={requested!r}."
            )
        return QwenBackboneCompileInfo(
            installed_layers=0,
            total_layers=len(layers),
            linear_attention_layers=len(linear_mixers),
            full_attention_layers=len(attention_mixers),
            installed_stateful_boundaries=0,
            installed_cache_boundaries=0,
        )

    pending_layers: list[tuple[nn.Module, Callable[..., Any]]] = []
    layer_marker = "_prism_qwen_decoder_compile_spec"
    for index, layer in enumerate(layers):
        installed = getattr(layer, layer_marker, None)
        if installed is not None:
            if tuple(installed) != requested:
                raise RuntimeError(
                    "Qwen backbone compiler options changed after installation at "
                    f"layer {index}: installed={tuple(installed)!r}, requested={requested!r}."
                )
            continue
        eager_forward = layer.forward
        try:
            compiled_forward = compile_fn(
                eager_forward,
                backend=requested[0],
                mode=requested[1],
                dynamic=requested[2],
                fullgraph=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to install Qwen backbone compiler at layer {index}."
            ) from exc
        if not callable(compiled_forward):
            raise RuntimeError(
                f"Qwen backbone compiler returned a non-callable value at layer {index}."
            )
        pending_layers.append((layer, compiled_forward))

    core_marker = "_prism_qwen_stateful_core_boundary"
    pending_cores: list[tuple[nn.Module, Callable[..., Any]]] = []
    for index, mixer in enumerate(linear_mixers):
        if bool(getattr(mixer, core_marker, False)):
            raise RuntimeError(
                "Qwen backbone compilation found an incomplete prior DeltaNet "
                f"boundary installation at linear layer {index}."
            )
        pending_cores.append(
            (
                mixer,
                _compile_disable(
                    mixer._prism_qwen35_stateful_core,
                    reason="PRISM Qwen3.5 mutable DeltaNet cache core",
                ),
            )
        )

    cache_marker = "_prism_qwen_cache_update_boundary"
    pending_caches: list[
        tuple[nn.Module, Callable[..., Any] | None, Callable[..., Any]]
    ] = []
    for index, attention in enumerate(attention_mixers):
        if bool(getattr(attention, cache_marker, False)):
            raise RuntimeError(
                "Qwen backbone compilation found an incomplete prior cache-boundary "
                f"installation at attention layer {index}."
            )
        replacement_forward, disabled_update = _prepare_qwen_attention_cache_boundary(
            attention
        )
        pending_caches.append((attention, replacement_forward, disabled_update))

    # Install only after every compiler/boundary wrapper has been constructed,
    # so setup failures cannot leave a partially modified model.
    for mixer, disabled_core in pending_cores:
        mixer._prism_qwen35_stateful_core = disabled_core
        setattr(mixer, core_marker, True)
    for attention, replacement_forward, disabled_update in pending_caches:
        attention._prism_qwen_cache_update = disabled_update
        if replacement_forward is not None:
            attention.forward = replacement_forward
        setattr(attention, cache_marker, True)
    for layer, compiled_forward in pending_layers:
        layer.forward = compiled_forward
        setattr(layer, layer_marker, requested)
    setattr(qwen, root_marker, requested)
    return QwenBackboneCompileInfo(
        installed_layers=len(pending_layers),
        total_layers=len(layers),
        linear_attention_layers=len(linear_mixers),
        full_attention_layers=len(attention_mixers),
        installed_stateful_boundaries=len(pending_cores),
        installed_cache_boundaries=len(pending_caches),
    )


def _iter_hf_configs(qwen: nn.Module) -> list[Any]:
    configs: list[Any] = []

    def add_config(cfg: Any) -> None:
        if cfg is not None and all(existing is not cfg for existing in configs):
            configs.append(cfg)

    root_cfg = getattr(qwen, "config", None)
    add_config(root_cfg)
    add_config(getattr(root_cfg, "text_config", None))
    for child_name in ("model", "transformer"):
        child = getattr(qwen, child_name, None)
        add_config(getattr(child, "config", None))
    return configs


def _set_config_attr_if_present(configs: Sequence[Any], name: str, value: Any) -> None:
    for cfg in configs:
        if hasattr(cfg, name):
            setattr(cfg, name, value)


def _shrink_layer_indexed_config_lists(
    configs: Sequence[Any],
    *,
    keep_layers: tuple[int, ...],
    original_num_layers: int,
) -> None:
    # Qwen3.5 stores hybrid attention layout here. Keep it aligned with the
    # physical ModuleList length so cache construction and debug metadata match.
    for cfg in configs:
        for name in ("layer_types", "layers_block_type", "layers_type"):
            value = getattr(cfg, name, None)
            if isinstance(value, list) and len(value) == original_num_layers:
                setattr(cfg, name, [value[idx] for idx in keep_layers])
            elif isinstance(value, tuple) and len(value) == original_num_layers:
                setattr(cfg, name, tuple(value[idx] for idx in keep_layers))


def _record_and_reindex_layer_idx(
    layers: Sequence[nn.Module],
    *,
    keep_layers: tuple[int, ...],
) -> None:
    for new_idx, (original_idx, layer) in enumerate(zip(keep_layers, layers, strict=True)):
        for module in layer.modules():
            if hasattr(module, "layer_idx"):
                if not hasattr(module, "prism_original_layer_idx"):
                    setattr(module, "prism_original_layer_idx", getattr(module, "layer_idx"))
                setattr(module, "layer_idx", new_idx)
            if not hasattr(module, "prism_source_layer_idx"):
                setattr(module, "prism_source_layer_idx", original_idx)


def apply_qwen_layer_skip(qwen: nn.Module, keep_layers: Any) -> BackboneLayerSkipInfo:
    """Physically keep only selected original Qwen layers.

    Non-kept layers are removed from the decoder ``ModuleList``. They no longer
    participate in forward/backward and do not appear in ``state_dict()``.
    """
    normalized = normalize_qwen_keep_layers(keep_layers)
    parent, attr_name, layers = _find_decoder_layers(qwen)
    original_num_layers = len(layers)
    if not normalized:
        return BackboneLayerSkipInfo(
            keep_layers=(),
            original_num_layers=original_num_layers,
            active_num_layers=original_num_layers,
        )

    invalid = [idx for idx in normalized if idx >= original_num_layers]
    if invalid:
        raise ValueError(
            f"qwen_keep_layers contains indices outside the loaded backbone depth "
            f"{original_num_layers}: {invalid}."
        )

    selected_layers = nn.ModuleList([layers[idx] for idx in normalized])
    setattr(parent, attr_name, selected_layers)

    configs = _iter_hf_configs(qwen)
    _set_config_attr_if_present(configs, "num_hidden_layers", len(selected_layers))
    for cfg in configs:
        setattr(cfg, "prism_original_num_hidden_layers", original_num_layers)
        setattr(cfg, "prism_qwen_keep_layers", list(normalized))
    _shrink_layer_indexed_config_lists(
        configs,
        keep_layers=normalized,
        original_num_layers=original_num_layers,
    )
    _record_and_reindex_layer_idx(selected_layers, keep_layers=normalized)

    return BackboneLayerSkipInfo(
        keep_layers=normalized,
        original_num_layers=original_num_layers,
        active_num_layers=len(selected_layers),
    )


def remap_qwen_layer_state_dict_for_keep_layers(
    state_dict: dict[str, Any],
    *,
    source_keep_layers: Any,
    target_keep_layers: Any,
) -> tuple[dict[str, Any], QwenLayerStateRemapInfo]:
    """Remap Qwen layer keys between full and pruned backbone layouts.

    Saved pruned checkpoints use compact layer positions ``0..K-1`` while their
    config records the original pretrained indices. This helper maps by original
    index so full checkpoints can initialize pruned models without accidentally
    loading layer 4 weights into the fourth kept layer when that layer is
    actually original layer 16.
    """
    source_keep = normalize_qwen_keep_layers(source_keep_layers)
    target_keep = normalize_qwen_keep_layers(target_keep_layers)
    if source_keep == target_keep:
        return state_dict, QwenLayerStateRemapInfo(remapped_keys=0, dropped_keys=0)

    target_orig_to_pos = (
        {original_idx: pos for pos, original_idx in enumerate(target_keep)}
        if target_keep
        else None
    )
    remapped: dict[str, Any] = {}
    remapped_keys = 0
    dropped_keys = 0

    for key, value in state_dict.items():
        match = _QWEN_LAYER_KEY_RE.match(key)
        if match is None:
            remapped[key] = value
            continue

        source_pos = int(match.group(2))
        if source_keep:
            if source_pos >= len(source_keep):
                dropped_keys += 1
                continue
            original_idx = source_keep[source_pos]
        else:
            original_idx = source_pos

        if target_orig_to_pos is None:
            target_pos = original_idx
        else:
            target_pos = target_orig_to_pos.get(original_idx)
            if target_pos is None:
                dropped_keys += 1
                continue

        new_key = f"{match.group(1)}{target_pos}{match.group(3)}"
        if new_key != key:
            remapped_keys += 1
        remapped[new_key] = value

    return remapped, QwenLayerStateRemapInfo(
        remapped_keys=remapped_keys,
        dropped_keys=dropped_keys,
    )
