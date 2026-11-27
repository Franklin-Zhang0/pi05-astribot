"""Bounded-memory vectorization of the unchanged framewise KDA convolution.

History stores the BF16/FP16 rounded mean of each nonempty frame. Prefix-index
gathers recover the exact previous K frames even with arbitrary empty frames;
the convolution's current term remains specific to each spatial token.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable

import torch
import torch.nn.functional as F


def _pool_frames(mixed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if mixed.dtype == torch.float64 else torch.float32
    clean = torch.where(valid[..., None], mixed, 0)
    count = valid.sum(-1).clamp_min(1).to(dtype)
    # dtype accumulation avoids a full sequence-sized FP32 input allocation.
    return (clean.sum(2, dtype=dtype) / count[..., None]).to(mixed.dtype)


def _convolve_block(mixed, valid, past, last_weight, bias):
    dtype = torch.float64 if mixed.dtype == torch.float64 else torch.float32
    clean = torch.where(valid[..., None], mixed, 0).to(dtype)
    output = clean * last_weight.to(dtype) + past[:, :, None]
    if bias is not None:
        output = output + bias.to(dtype)
    return torch.where(valid[..., None], F.silu(output).to(mixed.dtype), 0)


@lru_cache(maxsize=1)
def _compiled_helpers() -> tuple[Callable, Callable]:
    from prism_astribot_vla.modeling.qwen35_timestep_kda_kernels import compile_frame_function
    return compile_frame_function(_pool_frames), compile_frame_function(_convolve_block)


def timestep_convolution_vectorized(mixed: torch.Tensor, valid: torch.Tensor,
        weight: torch.Tensor, bias: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None, *, block_frames: int = 4,
        compile_pointwise: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reference convolution/output history with bounded temporaries.

    Only independent frame work is batched. Gathered history means are rounded
    to the input dtype before use, as in the sequential implementation. The
    returned history remains differentiable; cache writes stay in the caller.
    """
    if mixed.ndim != 4 or valid.shape != mixed.shape[:3] or valid.dtype != torch.bool:
        raise ValueError("Timestep convolution requires [B,T,N,C] and boolean [B,T,N]")
    if min(mixed.shape) < 1 or type(block_frames) is not int or block_frames < 1:
        raise ValueError("Timestep convolution requires nonempty dimensions and positive block_frames")
    if type(compile_pointwise) is not bool:
        raise ValueError("Timestep convolution compile_pointwise must be boolean")
    channels = mixed.shape[-1]
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight.squeeze(1)
    if weight.ndim != 2 or weight.shape[0] != channels or weight.shape[1] < 1:
        raise ValueError("Timestep convolution weight requires [C,K]")
    batch, frames = mixed.shape[:2]
    kernel = weight.shape[-1]
    history = (mixed.new_zeros(batch, channels, kernel) if initial_state is None
        else initial_state.to(device=mixed.device, dtype=mixed.dtype))
    if history.shape != (batch, channels, kernel):
        raise ValueError("Timestep convolution cache has incompatible shape")
    pool, convolve = _compiled_helpers() if compile_pointwise else (_pool_frames, _convolve_block)
    disjoint = os.environ.get("PRISM_KDA_DISJOINT_GRADS") == "1"
    if disjoint:
        mixed_blocks = mixed.split(block_frames, dim=1)
        valid_blocks = valid.split(block_frames, dim=1)
        pooled = torch.cat([pool(block, mask) for block, mask in zip(mixed_blocks, valid_blocks)], 1)
    else:
        pooled = torch.cat([pool(mixed[:, start:start + block_frames], valid[:, start:start + block_frames])
            for start in range(0, frames, block_frames)], 1)
    active = valid.any(-1)
    positions = torch.arange(frames, device=mixed.device).expand(batch, -1)
    inclusive = torch.where(active, positions, -1).cummax(1).values
    # Include a dummy frame T so the same lag gathers also give final history.
    previous = torch.cat((positions.new_full((batch, 1), -1), inclusive), 1)
    counts_before = torch.cat((positions.new_zeros((batch, 1)), active.long().cumsum(1)), 1)
    sources = torch.cat((history.transpose(1, 2), pooled), 1)
    cursor = previous
    lagged = []
    for lag in range(1, kernel + 1):
        index = torch.where(counts_before >= lag,
            kernel + cursor, kernel + counts_before - lag)
        lagged.append(sources.gather(1, index[..., None].expand(-1, -1, channels)))
        cursor = torch.where(cursor >= 0,
            previous.gather(1, cursor.clamp_min(0)), -1)
    dtype = torch.float64 if mixed.dtype == torch.float64 else torch.float32
    weights = weight.to(dtype)
    if kernel > 1:
        past = (torch.stack([entry[:, :frames] for entry in reversed(lagged[:-1])], -1).to(dtype)
            * weights[:, :-1]).sum(-1)
    else:
        past = mixed.new_zeros(batch, frames, channels, dtype=dtype)
    if disjoint:
        last_weight = weights[:, -1]
        outputs = [convolve(block, mask, prior, last_weight, bias)
                   for block, mask, prior in zip(mixed_blocks, valid_blocks, past.split(block_frames, dim=1))]
    else:
        outputs = [convolve(mixed[:, start:start + block_frames], valid[:, start:start + block_frames],
            past[:, start:start + block_frames], weights[:, -1], bias)
            for start in range(0, frames, block_frames)]
    final_history = torch.stack([entry[:, -1] for entry in reversed(lagged)], -1)
    return torch.cat(outputs, 1), final_history
