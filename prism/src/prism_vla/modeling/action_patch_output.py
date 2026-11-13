"""Expose raw ACT outputs while preserving native final-norm execution."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn


_RAW_ACT_CALL: ContextVar[object | None] = ContextVar("prism_raw_act_output", default=None)


def requires_raw_action_token(model) -> bool:
    """Read the instantiated backend contract, with a simple-fixture fallback."""
    backend = getattr(model, "idm", None)
    if backend is not None:
        return bool(getattr(backend, "requires_raw_action_token", False))
    kind = getattr(getattr(model, "config", None), "idm_type", "")
    return kind in {"act_patch", "patchwam"}


def _final_norm(backbone) -> nn.Module:
    # HF causal text backbones expose .norm. Some multimodal adapters keep the
    # text model one level below; resolve explicit final-norm paths only.
    for path in ("norm", "language_model.norm", "model.norm", "text_model.norm"):
        candidate = backbone
        for component in path.split("."):
            candidate = getattr(candidate, component, None)
            if candidate is None:
                break
        if isinstance(candidate, nn.Module):
            return candidate
    raise ValueError("Raw ACT output requires an identifiable native final norm module")


@contextmanager
def raw_action_token_output_scope(backbone, observer_mask, *, enabled: bool):
    """Replace only ACT rows after calling the real final norm normally.

    There is no parameter replacement, forward bypass, cache mutation, or
    persistent activation storage. Query/prefix-only calls clear any outer
    scope, and nested calls cannot trigger another call's temporary hook.
    """
    if not enabled:
        yield
        return
    identity = object() if observer_mask is not None else None
    context_token = _RAW_ACT_CALL.set(identity)
    handle = None
    calls = 0
    try:
        if observer_mask is None:
            yield
            return
        if (not torch.is_tensor(observer_mask) or observer_mask.dtype != torch.bool
                or observer_mask.ndim != 2):
            raise ValueError("Raw ACT output needs a boolean [B, Q] observer mask")
        norm = _final_norm(backbone)

        def raw_rows(_norm, args, kwargs, output):
            nonlocal calls
            if _RAW_ACT_CALL.get() is not identity:
                return output
            source = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
            if (not torch.is_tensor(source) or not torch.is_tensor(output)
                    or source.ndim != 3 or source.shape != output.shape
                    or tuple(source.shape[:2]) != tuple(observer_mask.shape)):
                raise ValueError("Native final norm did not receive the complete ACT-bearing main sequence")
            calls += 1
            mask = observer_mask.to(device=source.device).unsqueeze(-1)
            return torch.where(mask, source, output)

        handle = norm.register_forward_hook(raw_rows, with_kwargs=True)
        yield
        if calls != 1:
            raise RuntimeError("Raw ACT output requires exactly one native final norm call per main forward")
    finally:
        if handle is not None:
            handle.remove()
        _RAW_ACT_CALL.reset(context_token)
