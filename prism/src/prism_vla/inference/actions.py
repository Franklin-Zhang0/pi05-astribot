"""Action inference helpers shared by offline and LIBERO evaluation."""

from __future__ import annotations

from typing import Optional

import torch


def aggregate_flow_action_samples(
    flow_output: torch.Tensor,
    *,
    aggregation: str = "first",
) -> torch.Tensor:
    """Collapse optional multi-sample flow output to one action chunk per item."""
    if flow_output.ndim != 4:
        return flow_output

    if aggregation == "first":
        return flow_output[0]
    if aggregation == "mean":
        return flow_output.mean(dim=0)
    raise ValueError(f"Unsupported flow sample aggregation: {aggregation}")


def make_flow_initial_noise(
    *,
    idm_type: str,
    num_action_samples: int,
    flow_noise_seed: Optional[int],
    plan_index: int,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Create the same optional initial flow noise used by LIBERO inference.

    When a single sample is requested without a seed, returning ``None`` lets
    the IDM sample internally, matching the online eval client behavior.
    """
    if idm_type != "flow_matching":
        return None

    n_samples = int(num_action_samples)
    if n_samples <= 0:
        raise ValueError(f"num_action_samples must be positive, got {n_samples}")
    if flow_noise_seed is None and n_samples == 1:
        return None

    shape = (
        (n_samples, batch_size, action_horizon, action_dim)
        if n_samples > 1
        else (batch_size, action_horizon, action_dim)
    )
    generator = None
    if flow_noise_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(
            int(flow_noise_seed) + int(plan_index)
        )
    noise = torch.randn(*shape, generator=generator, dtype=torch.float32)
    if device is not None or dtype != torch.float32:
        noise = noise.to(device=device, dtype=dtype)
    return noise
