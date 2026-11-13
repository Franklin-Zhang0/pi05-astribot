"""Reconstruct original visual patches from camera-local global compact history.

A decoder row contains one camera and one selected global compact slot. Its
memory is exactly ``group_size`` consecutive frames; neither other cameras nor
other global slots can enter that row. Metadata-only window construction is
shared with the trainer so loss normalization follows the same temporal rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CameraReconstructionMetadata:
    """Detached, unfiltered metadata for the last ``k - 1`` physical frames."""

    frame_valid: torch.Tensor
    camera_present: torch.Tensor
    frame_ids: torch.Tensor


@dataclass
class CameraReconstructionState(CameraReconstructionMetadata):
    """Bounded history; ``predicted_globals`` has shape ``[B,k-1,C,G,H]``."""

    predicted_globals: torch.Tensor


def _validate_metadata(frame_valid, camera_present, frame_ids) -> tuple[int, int, int]:
    if (not torch.is_tensor(frame_valid) or frame_valid.dtype != torch.bool
            or frame_valid.ndim != 2):
        raise ValueError("frame_valid must be a boolean tensor [B, L]")
    batch, length = frame_valid.shape
    if (not torch.is_tensor(camera_present) or camera_present.dtype != torch.bool
            or camera_present.ndim != 3 or camera_present.shape[:2] != (batch, length)):
        raise ValueError("camera_present must be a boolean tensor [B, L, C]")
    if (not torch.is_tensor(frame_ids) or frame_ids.dtype not in (torch.int32, torch.int64)
            or frame_ids.shape != (batch, length)):
        raise ValueError("frame_ids must be an integer tensor [B, L]")
    return batch, length, camera_present.shape[2]


def _metadata_history(
    frame_valid: torch.Tensor,
    camera_present: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    group_size: int,
    previous_state: CameraReconstructionMetadata | None,
    reset_rows: torch.Tensor | None,
) -> CameraReconstructionMetadata:
    batch, _, cameras = _validate_metadata(frame_valid, camera_present, frame_ids)
    # Keep CPU metadata on CPU. Mixed metadata uses the first accelerator
    # device, with no accelerator-to-host scalar reads or dynamic row selection.
    device = next((value.device for value in (frame_valid, camera_present, frame_ids)
        if value.device.type != "cpu"), frame_valid.device)
    frame_valid = frame_valid.to(device=device)
    camera_present = camera_present.to(device=device)
    frame_ids = frame_ids.to(device=device, dtype=torch.long)
    history_length = group_size - 1
    if previous_state is None:
        old_valid = frame_valid.new_zeros(batch, history_length)
        old_present = camera_present.new_zeros(batch, history_length, cameras)
        old_ids = frame_ids.new_zeros(batch, history_length)
    else:
        if not isinstance(previous_state, CameraReconstructionMetadata):
            raise ValueError("previous_state must be CameraReconstructionMetadata or None")
        old_batch, old_length, old_cameras = _validate_metadata(
            previous_state.frame_valid, previous_state.camera_present, previous_state.frame_ids)
        if old_batch != batch or old_cameras != cameras or old_length > history_length:
            raise ValueError("Reconstruction history must match B, C and contain at most k-1 frames")
        old_valid = previous_state.frame_valid.detach().to(device=device)
        old_present = previous_state.camera_present.detach().to(device=device)
        old_ids = previous_state.frame_ids.detach().to(device=device, dtype=torch.long)
        padding = history_length - old_length
        if padding:
            old_valid = torch.cat((frame_valid.new_zeros(batch, padding), old_valid), dim=1)
            old_present = torch.cat((camera_present.new_zeros(batch, padding, cameras),
                old_present), dim=1)
            old_ids = torch.cat((frame_ids.new_zeros(batch, padding), old_ids), dim=1)
    if reset_rows is not None:
        if (not torch.is_tensor(reset_rows) or reset_rows.dtype != torch.bool
                or reset_rows.shape != (batch,)):
            raise ValueError("reset_rows must be a boolean tensor [B]")
        keep = ~reset_rows.to(device=device)
        old_valid = old_valid & keep[:, None]
        old_present = old_present & keep[:, None, None]
    return CameraReconstructionMetadata(
        frame_valid=torch.cat((old_valid, frame_valid), dim=1),
        camera_present=torch.cat((old_present, camera_present), dim=1),
        frame_ids=torch.cat((old_ids, frame_ids), dim=1),
    )


def _metadata_tail(metadata: CameraReconstructionMetadata, length: int):
    # ``-0:`` would keep every frame rather than the empty k=1 history.
    start = metadata.frame_valid.shape[1] - length
    return CameraReconstructionMetadata(
        frame_valid=metadata.frame_valid[:, start:].detach().clone(),
        camera_present=metadata.camera_present[:, start:].detach().clone(),
        frame_ids=metadata.frame_ids[:, start:].detach().clone(),
    )


def camera_reconstruction_window_metadata(
    frame_valid: torch.Tensor,
    camera_present: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    group_size: int,
    previous_state: CameraReconstructionMetadata | None = None,
    reset_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, CameraReconstructionMetadata]:
    """Return valid ``[B,L,C]`` windows and the next detached metadata tail.

    Every physical frame remains in place, including padding and absent views.
    A window requires all ``k`` valid frames, the same camera present in each,
    and frame IDs increasing by exactly one. Previous history is invalidated for
    ``reset_rows`` before constructing windows. No predictions or decoder are
    needed, so this helper also supports CPU trainer denominator preparation.
    Multiply the returned boolean mask's sum by the number of selected global
    slots to obtain the module's counting unit.
    """
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    batch, length, cameras = _validate_metadata(frame_valid, camera_present, frame_ids)
    metadata = _metadata_history(frame_valid, camera_present, frame_ids,
        group_size=group_size, previous_state=previous_state, reset_rows=reset_rows)
    if length:
        valid_frames = metadata.frame_valid.unfold(1, group_size, 1).all(dim=-1)
        valid_cameras = metadata.camera_present.unfold(1, group_size, 1).all(dim=-1)
        ids = metadata.frame_ids.unfold(1, group_size, 1)
        consecutive = (ids[..., 1:] == ids[..., :-1] + 1).all(dim=-1)
        windows = valid_cameras & (valid_frames & consecutive).unsqueeze(-1)
    else:
        windows = metadata.camera_present.new_zeros(batch, 0, cameras)
    return windows, _metadata_tail(metadata, group_size - 1)


def _axis_sincos(positions: torch.Tensor, width: int) -> torch.Tensor:
    if width == 0:
        return positions.new_empty((*positions.shape, 0))
    frequencies = torch.arange((width + 1) // 2, device=positions.device, dtype=torch.float32)
    frequencies = torch.exp(-frequencies * (torch.log(positions.new_tensor(10000.0))
        / max((width + 1) // 2 - 1, 1)))
    angles = positions[..., None] * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :width]


def spatial_patch_queries(
    grid_hw: tuple[int, int], dim: int, *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build row-major 2D sin/cos queries from an explicit rectangular grid."""
    if (not isinstance(grid_hw, (tuple, list)) or len(grid_hw) != 2
            or any(type(value) is not int or value <= 0 for value in grid_hw)):
        raise ValueError("grid_hw must contain two positive integers (height, width)")
    if type(dim) is not int or dim <= 0:
        raise ValueError("Spatial query width must be a positive integer")
    height, width = grid_hw
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    if dim == 1:
        encoding = (y.sin() + x.cos()).unsqueeze(-1)
    else:
        encoding = torch.cat((_axis_sincos(y, dim // 2),
            _axis_sincos(x, dim - dim // 2)), dim=-1)
    return encoding.reshape(height * width, dim).to(dtype=dtype)


class CameraPatchReconstruction(nn.Module):
    """Decode last-frame visual patches independently per camera/global slot.

    The Qwen compact width, decoder width, and original visual feature width
    are separate. Decoder weights are shared across cameras and global slots.
    All rows use exactly ``group_size`` temporal memory tokens, and dropout is
    disabled so partitioning an episode does not change its objective.

    Every ordinary forward executes the same projection/decoder boundaries,
    including ranks without valid windows. A finite synthetic row makes this
    possible even when B, L, or C is zero. Invalid real rows are masked before
    arithmetic, and the synthetic row contributes no loss or count.
    """

    def __init__(
        self,
        compact_dim: int,
        patch_dim: int,
        global_compact_indices: tuple[int, ...],
        group_size: int = 1,
        decoder_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        options = (compact_dim, patch_dim, group_size, decoder_dim, num_layers,
            num_heads, ff_mult)
        if any(type(value) is not int or value <= 0 for value in options):
            raise ValueError("Reconstruction dimensions, group size and depth must be positive integers")
        if decoder_dim % num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        if (not global_compact_indices
                or any(type(value) is not int or value < 0 for value in global_compact_indices)
                or len(set(global_compact_indices)) != len(global_compact_indices)):
            raise ValueError("global_compact_indices must contain distinct nonnegative integers")
        self.compact_dim = compact_dim
        self.patch_dim = patch_dim
        self.global_compact_indices = tuple(global_compact_indices)
        self.group_size = group_size
        self.decoder_dim = decoder_dim
        self.memory_projection = nn.Linear(compact_dim, decoder_dim)
        self.temporal_embeddings = nn.Parameter(torch.empty(group_size, decoder_dim))
        self.query_seed = nn.Parameter(torch.empty(1, 1, decoder_dim))
        layer = nn.TransformerDecoderLayer(decoder_dim, num_heads,
            dim_feedforward=decoder_dim * ff_mult, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(decoder_dim))
        self.output_projection = nn.Linear(decoder_dim, patch_dim)
        nn.init.normal_(self.temporal_embeddings, std=decoder_dim ** -0.5)
        nn.init.normal_(self.query_seed, std=decoder_dim ** -0.5)

    def _prediction_history(self, selected, previous_state, reset_rows):
        batch, _, cameras, globals_count, width = selected.shape
        history_length = self.group_size - 1
        if previous_state is None:
            history = selected.new_zeros(batch, history_length, cameras, globals_count, width)
        else:
            if not isinstance(previous_state, CameraReconstructionState):
                raise ValueError("previous_state must be CameraReconstructionState or None")
            history = previous_state.predicted_globals
            previous_length = previous_state.frame_valid.shape[1]
            if (not torch.is_tensor(history) or not history.is_floating_point()
                    or history.shape != (batch, previous_length, cameras, globals_count, width)):
                raise ValueError("predicted_globals history must have shape [B,history,C,G,compact_dim]")
            history = history.detach().to(device=selected.device, dtype=selected.dtype)
            if previous_length < history_length:
                history = torch.cat((selected.new_zeros(batch, history_length - previous_length,
                    cameras, globals_count, width), history), dim=1)
            previous_valid = (previous_state.frame_valid[:, :, None]
                & previous_state.camera_present).to(device=selected.device)
            if previous_length < history_length:
                previous_valid = torch.cat((previous_valid.new_zeros(batch,
                    history_length - previous_length, cameras), previous_valid), dim=1)
            history = torch.where(previous_valid[..., None, None], history, 0.0)
        if reset_rows is not None:
            history = torch.where(reset_rows.to(selected.device)[:, None, None, None, None],
                0.0, history)
        return torch.cat((history, selected), dim=1)

    def forward(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        frame_valid: torch.Tensor,
        camera_present: torch.Tensor,
        grid_hw: tuple[int, int] | None = None,
        frame_ids: torch.Tensor,
        previous_state: CameraReconstructionState | None = None,
        reset_rows: torch.Tensor | None = None,
        state_only: bool = False,
    ) -> dict[str, torch.Tensor | CameraReconstructionState]:
        """Return FP32 ``loss``, ``loss_sum``, ``weight`` and detached ``state``.

        ``predicted`` is ``[B,L,C,M,H]``; targets are original visual features
        ``[B,L,C,P,D]``. Each valid camera/global-slot window first averages
        FP32 ``1 - cosine`` over its last frame's P patches. ``weight`` counts
        these windows. Targets are always detached. ``state_only`` updates
        history without invoking any decoder/projection module or using targets.
        """
        batch, length, cameras = _validate_metadata(frame_valid, camera_present, frame_ids)
        if (not torch.is_tensor(predicted) or not predicted.is_floating_point()
                or predicted.ndim != 5 or predicted.shape[:3] != (batch, length, cameras)
                or predicted.shape[-1] != self.compact_dim):
            raise ValueError("predicted must be floating point [B, L, C, M, compact_dim]")
        if max(self.global_compact_indices) >= predicted.shape[3]:
            raise ValueError("global_compact_indices must select existing per-camera compact slots")
        windows, tail = camera_reconstruction_window_metadata(frame_valid, camera_present,
            frame_ids, group_size=self.group_size, previous_state=previous_state,
            reset_rows=reset_rows)
        slots = torch.tensor(self.global_compact_indices, dtype=torch.long, device=predicted.device)
        selected = predicted.index_select(3, slots)
        present = (frame_valid[:, :, None].to(predicted.device)
            & camera_present.to(predicted.device))
        selected = torch.where(present[..., None, None], selected, 0.0)
        history = self._prediction_history(selected, previous_state, reset_rows)
        start = history.shape[1] - (self.group_size - 1)
        state = CameraReconstructionState(frame_valid=tail.frame_valid,
            camera_present=tail.camera_present, frame_ids=tail.frame_ids,
            predicted_globals=history[:, start:].detach().clone())
        if state_only:
            zero = selected[:, :0].sum().float()
            return dict(loss=zero, loss_sum=zero, weight=zero.detach(), state=state)
        if (not torch.is_tensor(targets) or not targets.is_floating_point()
                or targets.ndim != 5 or targets.shape[:3] != (batch, length, cameras)
                or targets.shape[-1] != self.patch_dim or targets.shape[3] <= 0):
            raise ValueError("targets must be floating point [B, L, C, P>0, patch_dim]")
        if targets.device != predicted.device:
            raise ValueError("predicted and targets must share a device")
        spatial_queries = spatial_patch_queries(grid_hw, self.decoder_dim,
            device=predicted.device, dtype=self.query_seed.dtype)
        patches = targets.shape[3]
        if spatial_queries.shape[0] != patches:
            raise ValueError("grid_hw height*width must equal the number of target patches")
        globals_count = len(self.global_compact_indices)
        valid = windows.to(predicted.device)
        if length:
            memory = history.unfold(1, self.group_size, 1).permute(0, 1, 2, 3, 5, 4)
            memory = torch.where(valid[..., None, None, None], memory, 0.0)
            memory = memory.reshape(-1, self.group_size, self.compact_dim)
        else:
            # Keep an empty dependency on current predictions for zero gradients.
            memory = selected.reshape(-1, self.group_size, self.compact_dim)
        # Unconditionally append a finite dummy: module call order and backward
        # boundaries do not depend on any rank-local valid-window inventory.
        memory = torch.cat((memory, predicted.new_zeros(1, self.group_size, self.compact_dim)), dim=0)
        memory = self.memory_projection(memory) + self.temporal_embeddings.unsqueeze(0)
        queries = (self.query_seed + spatial_queries.unsqueeze(0)).to(memory.dtype)
        decoded = self.decoder(queries.expand(memory.shape[0], -1, -1), memory)
        reconstructed = self.output_projection(decoded)[:-1].reshape(
            batch, length, cameras, globals_count, patches, self.patch_dim)
        # Mask before normalizing: invalid targets may contain NaNs, and a
        # multiply by zero after cosine would retain NaNs in both pass directions.
        target = torch.where(valid[..., None, None], targets.detach(), 0.0).float()
        reconstructed = torch.where(valid[..., None, None, None], reconstructed, 0.0).float()
        patch_loss = 1.0 - F.cosine_similarity(reconstructed, target.unsqueeze(3), dim=-1, eps=1e-8)
        window_loss = patch_loss.mean(dim=-1)
        loss_sum = torch.where(valid[..., None], window_loss, 0.0).sum()
        weight = valid.sum(dtype=torch.float32) * globals_count
        return dict(loss=loss_sum / weight.clamp_min(1.0), loss_sum=loss_sum,
            weight=weight, state=state)
