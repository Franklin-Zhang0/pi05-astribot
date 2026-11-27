"""Directed ACT attention and transient K/V storage for the main stream.

ACT occupies a logical/output position, but only its own query may read its
key/value. Its K/V is available during the current forward and never enters
the persistent cache. The independent query-generation stream has no scope.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Iterator

import torch


@dataclass
class ActionObserverLayout:
    """Current-query ACT identities shared by all decoder layers."""

    observer_mask: torch.Tensor
    _device_masks: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    _device_indices: dict[torch.device, torch.Tensor] = field(default_factory=dict)

    def mask_for(self, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        if device not in self._device_masks:
            self._device_masks[device] = self.observer_mask.to(device=device)
        return self._device_masks[device]

    def persistent_indices(self, device: torch.device | str) -> torch.Tensor:
        device = torch.device(device)
        if device not in self._device_indices:
            self._device_indices[device] = (~self.observer_mask[0]).nonzero().flatten().to(device)
        return self._device_indices[device]


_ACTION_OBSERVER: ContextVar[ActionObserverLayout | None] = ContextVar(
    "prism_main_stream_action_observer", default=None,
)


def get_action_observer_layout() -> ActionObserverLayout | None:
    """Return metadata for this main forward, or None for all other streams."""
    return _ACTION_OBSERVER.get()


@contextmanager
def action_observer_scope(observer_mask: torch.Tensor) -> Iterator[ActionObserverLayout]:
    """Scope fixed logical ACT positions without retaining their K/V."""
    if (not torch.is_tensor(observer_mask) or observer_mask.dtype != torch.bool
            or observer_mask.ndim != 2 or min(observer_mask.shape) < 1):
        raise ValueError("ACT observer mask must be a nonempty boolean [B, Q] tensor")
    mask = observer_mask.detach().to(device="cpu").clone()
    if not bool(mask.any()) or not torch.equal(mask, mask[:1].expand_as(mask)):
        raise ValueError("ACT positions must be nonempty and identical across batch rows")
    layout = ActionObserverLayout(mask)
    token = _ACTION_OBSERVER.set(layout)
    try:
        yield layout
    finally:
        _ACTION_OBSERVER.reset(token)


def mask_action_observer_attention(
    attention_mask: torch.Tensor, observer_mask: torch.Tensor,
) -> torch.Tensor:
    """Allow ACT keys only on their own diagonal; preserve all other masks.

    Cached keys already exclude ACT. The final Q key columns correspond to
    the current Q query positions, including any instruction prefix.
    """
    if (attention_mask.dtype != torch.bool or attention_mask.ndim != 4
            or observer_mask.dtype != torch.bool or observer_mask.ndim != 2):
        raise ValueError("ACT attention requires boolean [B,H,Q,K] and [B,Q] masks")
    batch, queries = observer_mask.shape
    if (attention_mask.shape[0] not in (1, batch)
            or attention_mask.shape[-2] != queries or attention_mask.shape[-1] < queries):
        raise ValueError("ACT observer identities do not match attention query/key geometry")
    observers = observer_mask.to(device=attention_mask.device)
    past = attention_mask.shape[-1] - queries
    key_observers = torch.nn.functional.pad(observers, (past, 0))
    diagonal = torch.nn.functional.pad(
        torch.eye(queries, dtype=torch.bool, device=attention_mask.device), (past, 0),
    )
    allowed = (~key_observers[:, None, None, :]
        | (observers[:, None, :, None] & diagonal[None, None]))
    return attention_mask & allowed


def install_action_observer_cache_filter(cache_type: type) -> None:
    """Filter writes of an append-style cache while returning transient K/V.

    The original update receives only non-ACT entries. Its returned prefix is
    combined with all current entries for this layer's attention call. Thus
    storage is physically compressed, without changing the current SDPA mask.
    Calls outside the observer scope delegate to the original method exactly.
    """
    original = getattr(cache_type, "update", None)
    if not callable(original):
        raise TypeError("ACT observer caching requires a callable cache.update")
    if getattr(original, "_prism_action_observer_filter", False):
        return

    @wraps(original)
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None, **kwargs):
        layout = get_action_observer_layout()
        if layout is None:
            if cache_kwargs is None:
                return original(self, key_states, value_states, layer_idx, **kwargs)
            return original(self, key_states, value_states, layer_idx, cache_kwargs, **kwargs)
        if (key_states.ndim != 4 or value_states.ndim != 4
                or (key_states.shape[0], key_states.shape[-2]) != tuple(layout.observer_mask.shape)
                or value_states.shape[0] != key_states.shape[0]
                or value_states.shape[-2] != key_states.shape[-2]):
            raise ValueError("ACT-scoped cache update does not match current query identities")
        keep = layout.persistent_indices(key_states.device)
        current_keys = key_states.index_select(-2, keep)
        current_values = value_states.index_select(-2, keep.to(value_states.device))
        filtered_kwargs: Any = cache_kwargs
        if isinstance(cache_kwargs, dict):
            filtered_kwargs = dict(cache_kwargs)
            position = filtered_kwargs.get("cache_position")
            if torch.is_tensor(position) and position.shape[-1] == key_states.shape[-2]:
                filtered_kwargs["cache_position"] = position.index_select(-1, keep.to(position.device))
        if filtered_kwargs is None:
            persistent_keys, persistent_values = original(
                self, current_keys, current_values, layer_idx, **kwargs,
            )
        else:
            persistent_keys, persistent_values = original(
                self, current_keys, current_values, layer_idx, filtered_kwargs, **kwargs,
            )
        past = persistent_keys.shape[-2] - keep.numel()
        if past < 0:
            raise RuntimeError("ACT observer filter requires an append-style dynamic cache")
        if past == 0:
            return key_states, value_states
        return (
            torch.cat((persistent_keys[..., :past, :], key_states), dim=-2),
            torch.cat((persistent_values[..., :past, :], value_states), dim=-2),
        )

    update._prism_action_observer_filter = True
    cache_type.update = update


def install_qwen_action_observer_cache_filters() -> None:
    """Install filters once for the dynamic cache types supported by Qwen."""
    from transformers.cache_utils import DynamicCache
    install_action_observer_cache_filter(DynamicCache)
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
    except ImportError:
        return
    install_action_observer_cache_filter(Qwen3_5DynamicCache)
