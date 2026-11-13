"""Camera-local direct visual queries with a stable role-major token layout.

Each query reads its camera's spatial features directly.  The caller owns the
query seeds and their instruction/history conditioning; this module does not
construct a query stream or predict a future query.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


_ROLES = ("global", "task", "generated")
Slot = tuple[str, int, int]


@dataclass(frozen=True)
class DirectQueryLayout:
    """Map independent camera/role inventories to a fixed role-major order.

    A slot is ``(role, camera_index, local_index)``.  Camera indices refer to
    positions in the input camera inventory, not externally assigned camera IDs.
    Zero counts are allowed for any role/camera combination.
    """

    global_counts: tuple[int, ...]
    task_counts: tuple[int, ...]
    generated_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("global_counts", "task_counts", "generated_counts"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{name} must be an ordered tuple/list of camera counts")
            counts = tuple(value)
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError(f"{name} must contain nonnegative integer counts")
            object.__setattr__(self, name, counts)
        if not self.global_counts or not (
            len(self.global_counts) == len(self.task_counts) == len(self.generated_counts)
        ):
            raise ValueError("Direct-query count tuples must have one entry per camera")
        if not self.num_compacts:
            raise ValueError("Direct-query layout requires at least one visual query")

    @classmethod
    def from_config(cls, config: object) -> DirectQueryLayout:
        return cls(
            getattr(config, "direct_query_global_tokens_per_camera"),
            getattr(config, "direct_query_task_tokens_per_camera"),
            getattr(config, "direct_query_generated_tokens_per_camera"),
        )

    @property
    def num_cameras(self) -> int:
        return len(self.global_counts)

    @property
    def num_compacts(self) -> int:
        return sum(self.global_counts) + sum(self.task_counts) + sum(self.generated_counts)

    @property
    def num_generated(self) -> int:
        return sum(self.generated_counts)

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(
            (role, camera, local)
            for role, counts in zip(_ROLES, (self.global_counts, self.task_counts, self.generated_counts))
            for camera, count in enumerate(counts)
            for local in range(count)
        )

    def _roles(self, roles: str | Sequence[str] | None) -> tuple[str, ...]:
        if roles is None:
            return _ROLES
        selected = (roles,) if isinstance(roles, str) else tuple(roles)
        if len(set(selected)) != len(selected) or any(role not in _ROLES for role in selected):
            raise ValueError(f"roles must be distinct members of {_ROLES}")
        return tuple(role for role in _ROLES if role in selected)

    def slots_for(self, roles: str | Sequence[str] | None = None) -> tuple[Slot, ...]:
        selected = self._roles(roles)
        return tuple(slot for slot in self.slots if slot[0] in selected)

    def indices_for(self, roles: str | Sequence[str] | None = None) -> tuple[int, ...]:
        selected = self._roles(roles)
        return tuple(index for index, slot in enumerate(self.slots) if slot[0] in selected)

    def camera_major_indices(self, roles: str | Sequence[str] | None = None) -> tuple[int, ...]:
        """Permutation from selected role-major slots to camera-major slots."""
        slots = self.slots_for(roles)
        return tuple(sorted(range(len(slots)), key=lambda index: (slots[index][1], index)))

    def role_major_indices(self, roles: str | Sequence[str] | None = None) -> tuple[int, ...]:
        """Inverse permutation from camera-major back to selected role-major."""
        order = self.camera_major_indices(roles)
        return tuple(sorted(range(len(order)), key=order.__getitem__))

    def camera_mask(
        self, camera_present: torch.Tensor, roles: str | Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Expand ``[..., cameras]`` camera presence to selected role-major slots."""
        if (not torch.is_tensor(camera_present) or camera_present.dtype != torch.bool
                or camera_present.ndim < 1 or camera_present.shape[-1] != self.num_cameras):
            raise ValueError("camera_present must be boolean [..., num_cameras]")
        indices = torch.tensor(
            [slot[1] for slot in self.slots_for(roles)],
            device=camera_present.device, dtype=torch.long,
        )
        return camera_present.index_select(-1, indices)


def _patch_position_embeddings(
    height: int, width: int, dim: int, *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build fixed row-major 2D sine/cosine keys for a spatial patch grid."""
    quarter = dim // 4
    frequencies = torch.arange(quarter, device=device, dtype=torch.float32) / quarter
    frequencies = torch.pow(10000.0, -frequencies)
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij",
    )
    y = rows.reshape(-1, 1) * frequencies
    x = columns.reshape(-1, 1) * frequencies
    return torch.cat((y.sin(), y.cos(), x.sin(), x.cos()), dim=-1).to(dtype=dtype)


class _DirectQueryBlock(nn.Module):
    """One residual cross-attention from caller-supplied queries to visual keys."""

    def __init__(self, qwen_dim: int, vision_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(qwen_dim)
        self.visual_norm = nn.LayerNorm(vision_dim)
        self.attention = nn.MultiheadAttention(
            qwen_dim, num_heads, kdim=vision_dim, vdim=vision_dim, batch_first=True,
        )

    def forward(self, query: torch.Tensor, visual: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        values = self.visual_norm(visual)
        keys = values + position.to(dtype=values.dtype)
        update, _ = self.attention(self.query_norm(query), keys, values, need_weights=False)
        return query + update


class DirectQueryResampler(nn.Module):
    """Produce one compact per query by reading that query's camera directly.

    ``queries`` has shape ``[..., N, qwen_dim]`` in the layout's selected
    role-major order. ``patches`` is ``[..., C, P, vision_dim]`` and optional
    ``registers`` is ``[..., C, 7, vision_dim]``.  Every leading dimension must
    match exactly.  A missing camera is masked before attention and produces
    zero outputs, even if its visual inputs contain NaNs.
    """

    def __init__(
        self, layout: DirectQueryLayout, qwen_dim: int, vision_dim: int,
        num_heads: int = 8, depth: int = 1, include_registers: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(layout, DirectQueryLayout):
            raise ValueError("layout must be a DirectQueryLayout")
        if (type(qwen_dim) is not int or qwen_dim < 1 or type(vision_dim) is not int
                or vision_dim < 4 or vision_dim % 4):
            raise ValueError("qwen_dim must be positive and vision_dim divisible by four")
        if (type(num_heads) is not int or num_heads < 1 or qwen_dim % num_heads
                or type(depth) is not int or depth < 1):
            raise ValueError("num_heads must divide qwen_dim and depth must be positive")
        if type(include_registers) is not bool:
            raise ValueError("include_registers must be a bool")
        self.layout = layout
        self.qwen_dim = qwen_dim
        self.vision_dim = vision_dim
        self.include_registers = include_registers
        self.blocks = nn.ModuleList(
            _DirectQueryBlock(qwen_dim, vision_dim, num_heads) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(qwen_dim)
        self.register_embeddings = (
            nn.Parameter(torch.empty(7, vision_dim)) if include_registers else None
        )
        if self.register_embeddings is not None:
            nn.init.normal_(self.register_embeddings, std=vision_dim ** -0.5)

    def forward(
        self,
        queries: torch.Tensor,
        patches: torch.Tensor,
        registers: torch.Tensor | None = None,
        *,
        camera_mask: torch.Tensor | None = None,
        roles: str | Sequence[str] | None = None,
        patch_grid_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        slots = self.layout.slots_for(roles)
        if (not torch.is_tensor(queries) or not queries.is_floating_point()
                or queries.ndim < 2 or queries.shape[-2:] != (len(slots), self.qwen_dim)):
            raise ValueError("queries must be floating [..., selected_slots, qwen_dim]")
        prefix = queries.shape[:-2]
        if (not torch.is_tensor(patches) or not patches.is_floating_point()
                or patches.ndim != len(prefix) + 3 or patches.shape[:-3] != prefix
                or patches.shape[-3] != self.layout.num_cameras
                or patches.shape[-1] != self.vision_dim or patches.shape[-2] < 1):
            raise ValueError("patches must be floating [..., cameras, patches, vision_dim]")
        if patches.device != queries.device:
            raise ValueError("queries and visual features must share a device")
        if self.include_registers:
            if (not torch.is_tensor(registers) or not registers.is_floating_point()
                    or registers.shape != (*prefix, self.layout.num_cameras, 7, self.vision_dim)
                    or registers.device != queries.device):
                raise ValueError("registers must be floating [..., cameras, 7, vision_dim]")
        elif registers is not None:
            raise ValueError("registers were supplied while include_registers=False")
        if camera_mask is None:
            camera_mask = torch.ones(*prefix, self.layout.num_cameras, device=queries.device, dtype=torch.bool)
        if (not torch.is_tensor(camera_mask) or camera_mask.dtype != torch.bool
                or camera_mask.shape != (*prefix, self.layout.num_cameras)):
            raise ValueError("camera_mask must be boolean [..., cameras]")
        camera_mask = camera_mask.to(queries.device)
        patch_count = patches.shape[-2]
        if patch_grid_hw is None:
            side = math.isqrt(patch_count)
            patch_grid_hw = (side, side)
        if (not isinstance(patch_grid_hw, (tuple, list)) or len(patch_grid_hw) != 2
                or any(type(size) is not int or size < 1 for size in patch_grid_hw)
                or patch_grid_hw[0] * patch_grid_hw[1] != patch_count):
            raise ValueError("patch_grid_hw must be positive and match the patch count")

        if not slots:
            # The generated read always uses these shared parameters later
            # (total R >= 1). Do not touch sharded MHA parameters outside their
            # forward hooks just to attach an empty static-role tensor.
            return queries

        batch = math.prod(prefix)
        flat_queries = queries.reshape(batch, len(slots), self.qwen_dim)
        flat_patches = patches.reshape(batch, self.layout.num_cameras, patch_count, self.vision_dim)
        flat_registers = (
            registers.reshape(batch, self.layout.num_cameras, 7, self.vision_dim)
            if registers is not None else None
        )
        flat_mask = camera_mask.reshape(batch, self.layout.num_cameras)
        patch_positions = _patch_position_embeddings(
            *patch_grid_hw, self.vision_dim, device=queries.device, dtype=patches.dtype,
        )
        position = patch_positions
        if self.register_embeddings is not None:
            position = torch.cat((patch_positions, self.register_embeddings.to(patch_positions.dtype)), dim=0)
        outputs = []
        for camera in range(self.layout.num_cameras):
            selected = [index for index, slot in enumerate(slots) if slot[1] == camera]
            if not selected:
                continue
            index = torch.tensor(selected, device=queries.device, dtype=torch.long)
            present = flat_mask[:, camera, None, None]
            camera_queries = torch.where(present, flat_queries.index_select(1, index), 0)
            camera_visual = torch.where(present, flat_patches[:, camera], 0)
            if flat_registers is not None:
                camera_registers = torch.where(present, flat_registers[:, camera], 0)
                camera_visual = torch.cat((camera_visual, camera_registers), dim=1)
            for block in self.blocks:
                camera_queries = block(camera_queries, camera_visual, position)
            outputs.append(torch.where(present, self.output_norm(camera_queries), 0))
        camera_major = torch.cat(outputs, dim=1)
        order = torch.tensor(self.layout.role_major_indices(roles), device=queries.device, dtype=torch.long)
        return camera_major.index_select(1, order).reshape(*prefix, len(slots), self.qwen_dim)
