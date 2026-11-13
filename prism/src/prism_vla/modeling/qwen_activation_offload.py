"""Optional exact CPU storage for large, non-matrix Qwen saved activations.

Only autograd's saved copies move. Forward outputs, recurrent/KV caches and
model parameters retain their ordinary devices and native execution order.
The initial implementation uses blocking D2H copies for a completed CPU
snapshot, then PyTorch's standard pinned-memory nonblocking H2D restoration.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class _CPUActivation:
    device: torch.device
    value: torch.Tensor


_COUNTS = dict(scopes=0, packed_tensors=0, packed_bytes=0,
               unpacked_tensors=0, unpacked_bytes=0)


def qwen_activation_offload_counts(*, reset: bool = False) -> dict[str, int]:
    """Read CPU metadata, without inspecting or synchronizing device values."""
    result = dict(_COUNTS)
    if reset:
        for key in _COUNTS:
            _COUNTS[key] = 0
    return result


def validate_qwen_activation_offload_backbone(backbone: nn.Module, minimum_bytes: int) -> None:
    """Bound the non-matrix selection to the native Qwen3.5 parameter layout.

    Qwen's Linear/embedding matrices are rank two; norms/gates are vectors.
    Its only higher-rank learned tensors are small convolution kernels. Reject
    an architecture whose learned higher-rank storage would meet our selection
    threshold rather than silently copying large parameter views into CPU.
    """
    if type(minimum_bytes) is not int or minimum_bytes < 0:
        raise ValueError('Qwen saved-activation offload minimum must be a nonnegative integer')
    if not minimum_bytes:
        return
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    if type(backbone) is not Qwen3_5TextModel:
        raise ValueError('Qwen saved-activation offload currently requires the native Qwen3.5 backbone')
    for name, parameter in backbone.named_parameters():
        if parameter.ndim >= 3 and parameter.numel() * parameter.element_size() >= minimum_bytes:
            raise ValueError('Qwen saved-activation threshold includes a higher-rank parameter: ' + name)


def _selected(tensor: torch.Tensor, minimum_bytes: int) -> bool:
    if (type(tensor) is not torch.Tensor or tensor.device.type != 'cuda'
            or tensor.ndim < 3 or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or tensor.numel() * tensor.element_size() < minimum_bytes):
        return False
    # Native projections save rank-two weights. Also protect an explicitly
    # reshaped Parameter, rather than treating its rank-three view as a feature.
    base = getattr(tensor, '_base', None)
    while base is not None:
        if isinstance(base, nn.Parameter):
            return False
        base = getattr(base, '_base', None)
    return True


@contextmanager
def qwen_saved_activation_cpu_offload(minimum_bytes: int, *, training: bool) -> Iterator[None]:
    """Offload selected saved values; keep checkpoint replay and RNG untouched.

    This deliberately excludes rank-two saved matrices, subclasses/Parameters
    and their explicit views, and non-contiguous layouts. Native FSDP/DMuon
    materialization hooks remain in place. Selected activations use an exact CPU
    snapshot. The ordinary autograd edges still return gradients to the
    original inputs/parameters; no attention or state update is recomputed.
    PyTorch owns the pinned allocator's copy-stream lifetime, as in its native
    ``torch.autograd.graph.save_on_cpu(pin_memory=True)`` implementation.
    """
    if type(minimum_bytes) is not int or minimum_bytes < 0:
        raise ValueError('Qwen saved-activation offload minimum must be a nonnegative integer')
    if not minimum_bytes or not training or not torch.is_grad_enabled():
        yield
        return
    _COUNTS['scopes'] += 1

    def pack(tensor):
        if not _selected(tensor, minimum_bytes):
            return tensor.detach()  # Avoid a reference cycle through grad_fn.
        cpu = torch.empty_like(tensor, device='cpu', pin_memory=True,
                               memory_format=torch.preserve_format)
        with torch.no_grad():
            cpu.copy_(tensor)  # Finish the D2H snapshot before its input can be reused.
        _COUNTS['packed_tensors'] += 1
        _COUNTS['packed_bytes'] += tensor.numel() * tensor.element_size()
        return _CPUActivation(tensor.device, cpu)

    def unpack(value):
        if isinstance(value, _CPUActivation):
            _COUNTS['unpacked_tensors'] += 1
            _COUNTS['unpacked_bytes'] += value.value.numel() * value.value.element_size()
            return value.value.to(device=value.device, non_blocking=True)
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield
