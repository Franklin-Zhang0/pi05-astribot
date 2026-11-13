"""Raw-frame horizons for target-only future CLS supervision."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


def future_cls_frame_offset(seconds: Optional[float], source_fps: Optional[float]) -> Optional[int]:
    """Resolve a horizon with nearest-frame rounding, breaking ties forward.

    ``None, None`` preserves legacy next-observation supervision. A physical
    horizon is measured in source frames, independently of observation stride.
    """
    if seconds is None and source_fps is None:
        return None
    if seconds is None or source_fps is None:
        raise ValueError("future_cls_horizon_seconds and future_cls_source_fps must be configured together")
    for name, value in (("future_cls_horizon_seconds", seconds), ("future_cls_source_fps", source_fps)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    offset = int(math.floor(float(seconds) * float(source_fps) + 0.5))
    if offset < 1:
        raise ValueError("future CLS horizon must round to at least one source frame")
    return offset


def future_cls_target_frames(
    observation_frames: Sequence[int], *, episode_length: int,
    frame_offset: int, num_transitions: int, allocated_transitions: int,
) -> tuple[list[int], torch.Tensor]:
    """Return same-episode target frames and true endpoint validity.

    The caller may supply a short observation chunk: targets deliberately use
    episode bounds rather than chunk bounds. Invalid/padded rows decode the
    episode's last frame but never contribute to CLS supervision.
    """
    if episode_length < 1 or frame_offset < 1 or allocated_transitions < 0:
        raise ValueError("Future CLS frame selection requires valid episode/horizon/allocation")
    if not 0 <= num_transitions <= allocated_transitions:
        raise ValueError("num_transitions must fit allocated_transitions")
    if len(observation_frames) < num_transitions:
        raise ValueError("Missing observation identities for future CLS targets")
    frames, valid = [], []
    for row in range(allocated_transitions):
        anchor = int(observation_frames[row]) if row < num_transitions else episode_length - 1
        target = anchor + frame_offset
        active = row < num_transitions and 0 <= anchor < episode_length and target < episode_length
        frames.append(min(max(target, 0), episode_length - 1))
        valid.append(active)
    return frames, torch.tensor(valid, dtype=torch.bool)
