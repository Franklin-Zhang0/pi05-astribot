"""Bounded-memory framewise KDA preparation and optional compiled updates.

This is an implementation of the same simultaneous masked-mean delta rule as
``qwen35_timestep_kda.timestep_kda_reference``. It changes neither the write
operator nor the number/order of persistent state transitions. Preparation is
batched across a small number of frames; the dependent state updates remain in
chronological order. In particular, it does not use tokenwise FLA as a shortcut.

Compilation is explicit and lazy. Compiling the small frame body rather than
unrolling a whole sequence bounds graph size, and dynamic shapes allow the
same code to serve different token-reduction stages and resolution buckets.
No cache mutation or trainable module calls occur inside compiled functions.
"""
from __future__ import annotations

from functools import lru_cache, wraps
import os
from typing import Callable

import torch


def compile_frame_function(function: Callable, *, backend: str | Callable | None = None) -> Callable:
    """Compile a pure function for AMP forward and backward outside autocast.

    AOTAutograd traces the backward graph during the first compiled forward.
    Its default ``same_as_forward`` policy is inappropriate for the trainer's
    forward-only AMP context. Scope ``off`` around creation and every invocation
    (including lazy tracing, recompiles and checkpoint replay), then restore the
    caller's setting. The already-traced backward graph retains this policy;
    this wrapper does not enable/disable ambient forward autocast or leave a
    global configuration change behind. See PyTorch's compiler backward docs:
    https://docs.pytorch.org/docs/stable/torch.compiler_backward.html
    """
    from torch._functorch import config

    options = dict(fullgraph=True, dynamic=True)
    if backend is not None:
        options["backend"] = backend
    with config.patch(backward_pass_autocast="off"):
        compiled = torch.compile(function, **options)

    @wraps(function)
    def call(*args, **kwargs):
        with config.patch(backward_pass_autocast="off"):
            return compiled(*args, **kwargs)

    return call


def _prepare_frames(q, k, v, log_decay, beta, valid):
    """Prepare independent frames without changing the recurrence precision."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    mask = valid[..., None, None]
    query = torch.where(mask, q, 0)
    key = torch.where(mask, k, 0)
    value = torch.where(mask, v, 0).to(dtype)
    if q.is_cuda and dtype == torch.float32 and os.environ.get("PRISM_KDA_FUSED_L2NORM") == "1":
        # FLA evaluates the sum of squares and inverse root in FP32. Keep its
        # FP32 normalized outputs for the state rule and autograd.
        from fla.modules.l2norm import l2norm
        query = l2norm(query, eps=1e-6, output_dtype=dtype)
        key = l2norm(key, eps=1e-6, output_dtype=dtype)
    else:
        query = query.to(dtype)
        key = key.to(dtype)
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    count = valid.sum(-1).clamp_min(1).to(dtype)
    decay = torch.where(mask, log_decay, 0).to(dtype).sum(2) / count[..., None, None]
    strength = torch.where(valid[..., None], beta, 0).to(dtype)
    strength = (strength / count[..., None, None]).transpose(2, 3).unsqueeze(-1)
    return (query.transpose(2, 3), key.transpose(2, 3), value.transpose(2, 3),
        decay.exp().unsqueeze(-1), strength)


def _frame_update(query, key, value, decay, strength, state, scale):
    """Publish exactly one mean correction, then read it with every query."""
    decayed = state * decay
    prediction = torch.matmul(key, decayed)
    residual = (value - prediction) * strength
    state = decayed + torch.matmul(key.transpose(-1, -2), residual)
    output = (torch.matmul(query, state) * scale).transpose(1, 2)
    return output, state


@lru_cache(maxsize=1)
def _compiled_functions() -> tuple[Callable, Callable]:
    # No CUDA graphs: outputs become the next frame's state and participate in
    # autograd/checkpoint replay. A static sequence-sized graph is unnecessary.
    return compile_frame_function(_prepare_frames), compile_frame_function(_frame_update)


def optimized_timestep_kda_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        log_decay: torch.Tensor, beta: torch.Tensor, valid: torch.Tensor, *,
        initial_state: torch.Tensor | None = None, scale: float | None = None,
        use_compile: bool = False, prepare_frames: int = 4,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the reference framewise rule with bounded preparation batches.

    Input/return shapes, normalization, masking, empty-frame behavior, grouped
    heads and ambient autocast semantics match the reference. Recurrent state
    stays FP32 (FP64 for double-precision tests). ``prepare_frames`` bounds the
    additional FP32 preparation memory independently of the sequence length;
    one frame minimizes memory while larger batches amortize launch overhead.
    Compilation errors propagate rather than silently changing the backend.
    The caller retains the unchanged reference path as its default fallback.
    """
    if (q.ndim != 5 or v.ndim != 5 or min(q.shape) < 1
            or k.shape != q.shape or v.shape[:3] != q.shape[:3]):
        raise ValueError("Timestep KDA requires q/k/v [B,T,N,H,K/V]")
    if (log_decay.shape != (*q.shape[:3], v.shape[3], q.shape[-1])
            or beta.shape != (*q.shape[:3], v.shape[3]) or valid.shape != q.shape[:3]
            or valid.dtype != torch.bool):
        raise ValueError("Timestep KDA gate/validity shapes disagree")
    heads, key_dim, value_dim = v.shape[-2], q.shape[-1], v.shape[-1]
    if heads % q.shape[-2]:
        raise ValueError("KDA value heads must be divisible by query/key heads")
    if initial_state is not None and initial_state.shape != (q.shape[0], heads, key_dim, value_dim):
        raise ValueError("Timestep KDA initial state must match [B,HV,K,V]")
    if type(prepare_frames) is not int or prepare_frames < 1:
        raise ValueError("Timestep KDA prepare_frames must be a positive integer")
    if type(use_compile) is not bool:
        raise ValueError("Timestep KDA use_compile must be boolean")
    repeat = heads // q.shape[-2]
    if repeat != 1:
        q, k = q.repeat_interleave(repeat, -2), k.repeat_interleave(repeat, -2)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    state = (torch.zeros(q.shape[0], heads, key_dim, value_dim, dtype=dtype, device=q.device)
        if initial_state is None else initial_state.to(dtype))
    scale = key_dim ** -0.5 if scale is None else scale
    prepare, update = _compiled_functions() if use_compile else (_prepare_frames, _frame_update)
    outputs = []
    scan = os.environ.get("PRISM_KDA_BLOCK_SCAN", "off")
    if scan != "off":
        from prism_astribot_vla.modeling.qwen35_kda_scan import scan_prepared_frames
        block_size = int(os.environ.get("PRISM_KDA_SCAN_FRAMES", "8"))
        if scan not in {"readout","affine"} or not 1 <= block_size <= 32:
            raise ValueError("KDA block scan requires readout/affine and 1..32 frames")
        blocks = (tensor.split(block_size,dim=1) for tensor in (q,k,v,log_decay,beta,valid))
        for inputs in zip(*blocks):
            block_output, state = scan_prepared_frames(*prepare(*inputs), state, scale, scan)
            outputs.append(block_output)
        return torch.cat(outputs, dim=1).to(v.dtype), state
    if os.environ.get("PRISM_KDA_DISJOINT_GRADS") == "1":
        # One split/unbind node assembles gradients of disjoint views once.
        # Independent slices create full-sized zero+scatter gradients per view.
        blocks = (tensor.split(prepare_frames, dim=1) for tensor in (q, k, v, log_decay, beta, valid))
        for inputs in zip(*blocks):
            prepared = prepare(*inputs)
            for frame in zip(*(tensor.unbind(dim=1) for tensor in prepared)):
                output, state = update(*frame, state, scale)
                outputs.append(output)
        return torch.stack(outputs, 1).to(v.dtype), state
    for start in range(0, q.shape[1], prepare_frames):
        end = min(start + prepare_frames, q.shape[1])
        prepared = prepare(q[:, start:end], k[:, start:end], v[:, start:end],
            log_decay[:, start:end], beta[:, start:end], valid[:, start:end])
        for index in range(end - start):
            output, state = update(*(x[:, index] for x in prepared), state, scale)
            outputs.append(output)
    return torch.stack(outputs, 1).to(v.dtype), state
