from __future__ import annotations

import torch


LATENT_NORM_EPS = 1e-6


def rms_normalize_last_dim(x: torch.Tensor, eps: float = LATENT_NORM_EPS) -> torch.Tensor:
    """Apply stateless RMS normalization over the last dimension.

    This keeps latent vectors in a stable, Qwen-like scale without introducing
    extra checkpoint parameters, so older Stage-1 and Stage-2 checkpoints remain
    load-compatible.
    """
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv_rms.to(dtype=x.dtype)
