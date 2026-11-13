"""Shared regression losses for PRISM-VLA continuous targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_LOSS_ALIASES = {
    "mse": "mse",
    "l2": "mse",
    "l1": "l1",
    "mae": "l1",
    "huber": "huber",
    "smooth_l1": "huber",
    "smooth-l1": "huber",
}


def normalize_regression_loss_type(loss_type: str | None) -> str:
    """Normalize user-facing regression loss names."""
    normalized = str(loss_type or "mse").lower().replace(" ", "_")
    if normalized not in _LOSS_ALIASES:
        valid = ", ".join(sorted(_LOSS_ALIASES))
        raise ValueError(f"Unsupported regression loss_type={loss_type!r}. Expected one of: {valid}.")
    return _LOSS_ALIASES[normalized]


def regression_loss_elementwise(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "mse",
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Return unreduced elementwise regression loss."""
    kind = normalize_regression_loss_type(loss_type)
    diff = pred - target
    if kind == "mse":
        return diff.square()
    if kind == "l1":
        return diff.abs()

    delta_value = float(huber_delta)
    if delta_value <= 0.0:
        raise ValueError(f"huber_delta must be positive, got {delta_value}.")
    abs_diff = diff.abs()
    delta = torch.as_tensor(delta_value, device=abs_diff.device, dtype=abs_diff.dtype)
    quadratic = 0.5 * diff.square()
    linear = delta * (abs_diff - 0.5 * delta)
    return torch.where(abs_diff <= delta, quadratic, linear)


def masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    loss_type: str = "mse",
    huber_delta: float = 0.1,
    mean_dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute a masked mean regression loss.

    ``mean_dim`` is applied before masking, which is useful for latent-token
    losses where a [B, T, D] loss should first average over D and then average
    over valid timesteps.
    """
    loss = regression_loss_elementwise(
        pred,
        target,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )
    if mean_dim is not None:
        loss = loss.mean(dim=mean_dim)
    if mask is None:
        return loss.mean()
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def sigreg_epps_pulley_loss(
    tokens: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    num_slices: int = 256,
    num_knots: int = 17,
    t_max: float = 3.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply LeJEPA-style sliced Epps-Pulley regularization.

    ``tokens`` has shape ``[..., N, D]``. The penultimate dimension is the
    statistical sample axis, so callers can independently regularize each
    frame's set of cross-attention query tokens. Random projection directions
    are drawn from ``generator``. When one is not supplied, a function-local
    generator is used so SIGReg never consumes the process-global RNG.
    ``sample_mask`` excludes missing camera queries from both the empirical
    distribution and sample count. Frames with fewer than two real samples
    contribute neither loss nor denominator weight.

    This follows the characteristic-function statistic from LeJEPA while using
    the symmetric positive-half trapezoidal integration from its reference
    implementation: https://arxiv.org/abs/2511.08544
    """
    if tokens.ndim < 2:
        raise ValueError(f"SIGReg expects [..., N, D], got shape={tuple(tokens.shape)}")
    num_samples = int(tokens.shape[-2])
    feature_dim = int(tokens.shape[-1])
    if num_samples < 2:
        raise ValueError(f"SIGReg needs at least two samples, got N={num_samples}.")
    if feature_dim <= 0:
        raise ValueError(f"SIGReg feature dimension must be positive, got D={feature_dim}.")
    if int(num_slices) <= 0:
        raise ValueError(f"SIGReg num_slices must be positive, got {num_slices}.")
    if int(num_knots) < 2:
        raise ValueError(f"SIGReg num_knots must be at least 2, got {num_knots}.")
    if float(t_max) <= 0.0:
        raise ValueError(f"SIGReg t_max must be positive, got {t_max}.")

    # Characteristic-function evaluation is numerically fragile in bf16.
    samples = tokens.float()
    if sample_mask is not None:
        if tuple(sample_mask.shape) != tuple(tokens.shape[:-1]):
            raise ValueError("SIGReg sample_mask must match the statistical sample dimensions")
        sample_mask = sample_mask.to(device=samples.device, dtype=torch.bool)
        samples = torch.where(sample_mask.unsqueeze(-1), samples, 0)
    if generator is None:
        generator = torch.Generator(device=samples.device).manual_seed(0)
    directions = torch.randn(
        feature_dim,
        int(num_slices),
        device=samples.device,
        dtype=samples.dtype,
        generator=generator,
    )
    directions = F.normalize(directions, dim=0)
    projected = samples @ directions

    knots = torch.linspace(
        0.0,
        float(t_max),
        int(num_knots),
        device=samples.device,
        dtype=samples.dtype,
    )
    step = float(t_max) / (int(num_knots) - 1)
    trapezoid = torch.full_like(knots, 2.0 * step)
    trapezoid[0].fill_(step)
    trapezoid[-1].fill_(step)
    gaussian_cf = torch.exp(-0.5 * knots.square())
    integration_weights = trapezoid * gaussian_cf

    phases = projected.unsqueeze(-1) * knots
    if sample_mask is None:
        empirical_real = phases.cos().mean(dim=-3)
        empirical_imag = phases.sin().mean(dim=-3)
    else:
        valid = sample_mask.to(phases.dtype)
        counts = valid.sum(dim=-1)
        denominator = counts[..., None, None].clamp_min(1)
        empirical_real = (phases.cos() * valid[..., :, None, None]).sum(dim=-3) / denominator
        empirical_imag = (phases.sin() * valid[..., :, None, None]).sum(dim=-3) / denominator
    error = (empirical_real - gaussian_cf).square() + empirical_imag.square()
    if sample_mask is None:
        statistic = float(num_samples) * (error * integration_weights).sum(dim=-1)
        return statistic.mean()
    statistic = counts[..., None] * (error * integration_weights).sum(dim=-1)
    supported = (counts >= 2).to(statistic.dtype)
    return (statistic.mean(dim=-1) * supported).sum() / supported.sum().clamp_min(1)
