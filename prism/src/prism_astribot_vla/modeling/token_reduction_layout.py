"""Fixed spatial token identities for progressive visual-token selection.

All selection tables depend only on the camera grids and stage.  They are built
on the CPU, reused across observations, and returned in original raster order.
The only averaging operation here is the explicitly configured wrist reduction
*before* the backbone; stage transitions are index gathers.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F


FIXED_SELECTION_VERSION = "checkerboard_original_positions_v1"
DEFAULT_STAGE_STARTS = (0, 4, 8, 12, 16, 20)
DEFAULT_ATTENTION_WINDOWS = (1, 2, 4, 8, 16, 32)


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _grid(grid_hw: Sequence[int]) -> tuple[int, int]:
    if len(grid_hw) != 2:
        raise ValueError("A camera patch grid must have two dimensions")
    return (_positive_int(grid_hw[0], "grid height"),
            _positive_int(grid_hw[1], "grid width"))


def _stage(stage: int) -> int:
    if type(stage) is not int or stage < 0:
        raise ValueError("Selection stage must be a nonnegative integer")
    return stage


@lru_cache(maxsize=512)
def _priority_order(height: int, width: int) -> tuple[int, ...]:
    """Rank coordinates by nested checkerboards with deterministic edge ties.

    The first priority bits select even checkerboard parity, then odd/odd
    coordinates. Subsequent pairs select the coarse checkerboard and its even
    sub-grid. Thus stage five on a 16x16 grid retains the agreed eight anchors.
    The complete ordering also defines exact ceiling counts on odd/1-D grids.
    """
    bits = max(height - 1, width - 1, 1).bit_length()

    def priority(index: int) -> tuple[int, ...]:
        row, column = divmod(index, width)
        result = []
        for bit in range(bits):
            r, c = (row >> bit) & 1, (column >> bit) & 1
            result.extend((r ^ c, 1 - r if bit == 0 else r))
        return (*result, index)

    return tuple(sorted(range(height * width), key=priority))


@lru_cache(maxsize=2048)
def _spatial_indices(height: int, width: int, stage: int) -> tuple[int, ...]:
    count = (height * width + (1 << stage) - 1) // (1 << stage)
    return tuple(sorted(_priority_order(height, width)[:count]))


def fixed_spatial_indices(
    grid_hw: Sequence[int], stage: int, *, device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the retained ORIGINAL raster indices for one camera grid."""
    height, width = _grid(grid_hw)
    return torch.tensor(_spatial_indices(height, width, _stage(stage)),
                        dtype=torch.long, device=device)


def camera_patch_mean2x2(
    patches: torch.Tensor, grid_hw: Sequence[int],
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Apply the fixed front-end wrist mean, preserving partial edge blocks.

    Input has shape ``[..., H*W, D]``. Padding never counts as a real patch;
    incomplete right/bottom blocks average only existing patches. BF16/FP16
    inputs accumulate in FP32 and return their original dtype. Callers mask
    absent camera rows before this operation.
    """
    height, width = _grid(grid_hw)
    if patches.ndim < 2 or patches.shape[-2] != height * width:
        raise ValueError("Patch tensor must end in [grid_height*grid_width, width]")
    if not patches.is_floating_point():
        raise ValueError("Patch features must be floating point")
    out_h, out_w = (height + 1) // 2, (width + 1) // 2
    # F.pad's final two pairs refer to the channel-last W/H axes here.
    image = patches.reshape(*patches.shape[:-2], height, width, patches.shape[-1])
    image = F.pad(image, (0, 0, 0, width % 2, 0, height % 2))
    image = image.reshape(*patches.shape[:-2], out_h, 2, out_w, 2, patches.shape[-1])
    accumulation_dtype = (torch.float32 if patches.dtype in (torch.float16, torch.bfloat16)
                          else patches.dtype)
    summed = image.sum(dim=(-4, -2), dtype=accumulation_dtype)
    rows = torch.arange(out_h, device=patches.device)
    columns = torch.arange(out_w, device=patches.device)
    row_counts = (height - 2 * rows).clamp(max=2)
    column_counts = (width - 2 * columns).clamp(max=2)
    counts = (row_counts[:, None] * column_counts[None, :]).to(accumulation_dtype)
    merged = (summed / counts[..., None]).to(patches.dtype)
    return merged.reshape(*patches.shape[:-2], out_h * out_w, patches.shape[-1]), (out_h, out_w)


# Readable aliases for callers that name the front-end operation by role.
fixed_wrist_mean2x2 = camera_patch_mean2x2
camera_mean2x2 = camera_patch_mean2x2


@dataclass(frozen=True)
class FixedTokenMetadata:
    """Per-frame token identities; time/rotary IDs are owned by the caller."""

    camera_ids: torch.Tensor
    spatial_positions: torch.Tensor
    source_indices: torch.Tensor
    camera_local_indices: torch.Tensor
    camera_counts: tuple[int, ...]
    stage: int


@dataclass(frozen=True)
class FixedTokenLayout:
    """Camera-major fixed gathers for a collection of post-front-end grids.

    ``camera_ids`` are semantic identities and may contain reserved gaps, e.g.
    ``(0, 2)`` for main plus right wrist. Missing cameras within a batch should
    use the caller's validity mask; they do not require a different layout.
    """

    camera_grids: tuple[tuple[int, int], ...]
    camera_ids: tuple[int, ...] | None = None
    num_stages: int = 6

    def __post_init__(self) -> None:
        grids = tuple(_grid(grid) for grid in self.camera_grids)
        if not grids:
            raise ValueError("A visual layout needs at least one camera grid")
        ids = tuple(range(len(grids))) if self.camera_ids is None else tuple(self.camera_ids)
        if (len(ids) != len(grids) or len(set(ids)) != len(ids)
                or any(type(value) is not int or value < 0 for value in ids)):
            raise ValueError("Camera IDs must be unique nonnegative integers, one per grid")
        _positive_int(self.num_stages, "number of stages")
        object.__setattr__(self, "camera_grids", grids)
        object.__setattr__(self, "camera_ids", ids)

    @property
    def initial_token_count(self) -> int:
        return sum(height * width for height, width in self.camera_grids)

    @property
    def final_token_count(self) -> int:
        return self.stage_token_count(self.num_stages - 1)

    @property
    def num_cameras(self) -> int:
        return len(self.camera_grids)

    def _validate_stage(self, stage: int) -> int:
        stage = _stage(stage)
        if stage >= self.num_stages:
            raise ValueError(f"Stage {stage} exceeds configured {self.num_stages} stages")
        return stage

    def stage_counts(self, stage: int) -> tuple[int, ...]:
        """Return ceil(initial count / 2**stage) for each camera."""
        stage = self._validate_stage(stage)
        denominator = 1 << stage
        return tuple((height * width + denominator - 1) // denominator
                     for height, width in self.camera_grids)

    def stage_token_count(self, stage: int) -> int:
        return sum(self.stage_counts(stage))

    def stage_indices(
        self, stage: int, *, device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Select a stage directly from the original stage-zero token axis."""
        stage = self._validate_stage(stage)
        values, offset = [], 0
        for height, width in self.camera_grids:
            values.extend(offset + index for index in _spatial_indices(height, width, stage))
            offset += height * width
        return torch.tensor(values, dtype=torch.long, device=device)

    def transition_indices(
        self, from_stage: int, to_stage: int, *, device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Gather a deeper stage from an existing stage, without reordering."""
        from_stage, to_stage = self._validate_stage(from_stage), self._validate_stage(to_stage)
        if to_stage < from_stage:
            raise ValueError("A stage transition cannot restore discarded tokens")
        values, offset = [], 0
        for height, width in self.camera_grids:
            previous = _spatial_indices(height, width, from_stage)
            lookup = {original: local for local, original in enumerate(previous)}
            values.extend(offset + lookup[index] for index in _spatial_indices(height, width, to_stage))
            offset += len(previous)
        return torch.tensor(values, dtype=torch.long, device=device)

    def metadata(
        self, stage: int = 0, *, device: torch.device | str | None = None,
    ) -> FixedTokenMetadata:
        """Return the stage's original camera/grid IDs, never new centroids."""
        stage = self._validate_stage(stage)
        cameras, coordinates, local_indices = [], [], []
        for camera, (height, width) in zip(self.camera_ids, self.camera_grids):
            kept = _spatial_indices(height, width, stage)
            cameras.extend([camera] * len(kept))
            coordinates.extend(divmod(index, width) for index in kept)
            local_indices.extend(kept)
        return FixedTokenMetadata(
            camera_ids=torch.tensor(cameras, dtype=torch.long, device=device),
            spatial_positions=torch.tensor(coordinates, dtype=torch.long, device=device),
            source_indices=self.stage_indices(stage, device=device),
            camera_local_indices=torch.tensor(local_indices, dtype=torch.long, device=device),
            camera_counts=self.stage_counts(stage), stage=stage,
        )

    def gather_stage(
        self, values: torch.Tensor, from_stage: int, to_stage: int, *, dim: int = -2,
    ) -> torch.Tensor:
        """Apply the exact same selection to hidden states or caller metadata."""
        if values.shape[dim] != self.stage_token_count(from_stage):
            raise ValueError("Tensor token axis differs from the source layout stage")
        return values.index_select(dim, self.transition_indices(from_stage, to_stage, device=values.device))


TokenReductionLayout = FixedTokenLayout


def fixed_stage_indices(
    camera_grids: Sequence[Sequence[int]], stage: int, *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convenience selection for default semantic camera IDs."""
    stage = _stage(stage)
    layout = FixedTokenLayout(tuple(tuple(grid) for grid in camera_grids),
                              num_stages=max(6, stage + 1))
    return layout.stage_indices(stage, device=device)
