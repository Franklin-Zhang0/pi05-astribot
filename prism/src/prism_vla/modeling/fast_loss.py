"""Bound the temporary FP32 workspace of long FAST token objectives."""
from __future__ import annotations

from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


FAST_CROSS_ENTROPY_CHUNK_TOKENS = 16_384


def _fp32_cross_entropy_sum(
    logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.float(), targets, ignore_index=ignore_index, reduction="sum",
    )


def bounded_fp32_cross_entropy_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -1,
    chunk_tokens: int = FAST_CROSS_ENTROPY_CHUNK_TOKENS,
) -> torch.Tensor:
    """Return the unchanged summed token CE with bounded FP32 intermediates.

    The caller divides once by the total number of nonignored tokens. Long
    inputs split only the token axis, retaining every target and vocabulary
    entry. Stateless nonreentrant checkpoints recompute each chunk's FP32
    softmax during backward instead of retaining the entire FP32 matrix.
    No RNG, model parameters, or distributed collectives run in a checkpoint.
    Small inputs use the original direct loss; inference skips checkpointing.
    """
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int) or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if logits.ndim < 2 or tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError("FAST logits must have shape [*targets.shape, vocabulary]")
    vocabulary = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary)
    flat_targets = targets.reshape(-1)
    loss_fn = partial(_fp32_cross_entropy_sum, ignore_index=ignore_index)
    if flat_targets.numel() <= chunk_tokens:
        return loss_fn(flat_logits, flat_targets)

    recompute = torch.is_grad_enabled() and flat_logits.requires_grad
    losses = []
    # SplitBackward concatenates chunk gradients once. Repeated basic slices
    # would each allocate a zero-filled gradient the size of the whole input.
    for chunk_logits, chunk_targets in zip(
        flat_logits.split(chunk_tokens), flat_targets.split(chunk_tokens),
    ):
        losses.append(
            checkpoint(
                loss_fn, chunk_logits, chunk_targets,
                use_reentrant=False, preserve_rng_state=False,
            ) if recompute else loss_fn(chunk_logits, chunk_targets)
        )
    return torch.stack(losses).sum()
