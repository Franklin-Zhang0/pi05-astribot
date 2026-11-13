"""Camera-local latent resampling, independent of the number of camera slots.

The returned dense container preserves input camera order. Its mask identifies
real tokens; callers must pack or otherwise handle missing slots before feeding
a recurrent backbone. A masked slot is not a no-op for cached linear attention.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import nn
from prism_vla.modeling.camera_token_budget import camera_token_counts


def resolve_camera_pool_num_heads(dim: int, num_heads: int | None = None) -> int:
    """Validate an explicit split or infer one with at most 128 channels/head."""
    if type(dim) is not int or dim <= 0:
        raise ValueError("Pooling width must be a positive integer")
    if num_heads is not None:
        if type(num_heads) is not int or num_heads <= 0 or dim % num_heads:
            raise ValueError("pool_num_heads must be a positive integer dividing qwen_dim")
        return num_heads
    head_dim = next(value for value in range(min(dim, 128), 0, -1) if dim % value == 0)
    return dim // head_dim


class CameraSpecificLinearPredictor(nn.Module):
    """Independent output maps keyed by stable, configured camera identities.

    Only configured cameras allocate matrices. Every forward executes every
    configured map in the same order, including absent views; callers mask the
    resulting tokens so distributed module schedules stay identical on all ranks.
    """

    def __init__(self, dim: int, camera_ids: tuple[int, ...], *,
                 tokens_per_camera: int | tuple[int, ...] = 1,
                 allow_zero_tokens: bool = False):
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise ValueError("Prediction width must be a positive integer")
        if (not camera_ids or any(type(value) is not int or value < 0 for value in camera_ids)
                or len(set(camera_ids)) != len(camera_ids)):
            raise ValueError("Prediction camera_ids must be distinct nonnegative integers")
        if type(allow_zero_tokens) is not bool:
            raise ValueError("allow_zero_tokens must be a bool")
        if allow_zero_tokens:
            if type(tokens_per_camera) is int:
                counts = (tokens_per_camera,) * len(camera_ids)
            elif isinstance(tokens_per_camera, (tuple, list)):
                counts = tuple(tokens_per_camera)
            else:
                raise ValueError("tokens_per_camera must have one nonnegative count per camera")
            if (len(counts) != len(camera_ids)
                    or any(type(count) is not int or count < 0 for count in counts)):
                raise ValueError("tokens_per_camera must have one nonnegative count per camera")
        else:
            counts = camera_token_counts(tokens_per_camera, len(camera_ids))
        self.dim = dim
        self.camera_ids = tuple(camera_ids)
        self.tokens_per_camera = tokens_per_camera
        self.camera_token_counts = counts
        self.heads = nn.ModuleDict({str(camera_id): nn.Linear(dim, count * dim)
            for camera_id, count in zip(camera_ids, counts) if count})

    def forward(self, hidden: torch.Tensor, *, positional: bool = False) -> torch.Tensor:
        """Map ``[..., H]`` to ``[..., sum(counts), H]`` in camera-major order.

        Each slot occupies independent rows of its camera's linear map. The
        one-slot case retains the original parameter names, shapes and path.
        With ``positional=True``, the input is already camera-major slots and
        each output slot uses its corresponding input hidden state.
        """
        if positional:
            if hidden.shape[-2:] != (sum(self.camera_token_counts), self.dim):
                raise ValueError("Positional prediction requires camera-major slot hidden states")
            predicted = []
            offset = 0
            for camera_id, slots in zip(self.camera_ids, self.camera_token_counts):
                if not slots:
                    continue
                head = self.heads[str(camera_id)]
                camera_hidden = hidden[..., offset:offset + slots, :]
                offset += slots
                # Keep the original camera map and its independent output-row
                # blocks. Only matching input/output slots contribute gradients.
                camera_prediction = head(camera_hidden).reshape(
                    *hidden.shape[:-2], slots, slots, self.dim
                )
                predicted.append(camera_prediction.diagonal(dim1=-3, dim2=-2).movedim(-1, -2))
            return torch.cat(predicted, dim=-2) if predicted else hidden[..., :0, :]
        predicted = [self.heads[str(camera_id)](hidden).reshape(*hidden.shape[:-1], count, self.dim)
            for camera_id, count in zip(self.camera_ids, self.camera_token_counts) if count]
        return torch.cat(predicted, dim=-2) if predicted else hidden.unsqueeze(-2)[..., :0, :]


class CameraCompactResult(NamedTuple):
    tokens: torch.Tensor
    camera_mask: torch.Tensor
    camera_ids: torch.Tensor
    refined_tokens: torch.Tensor


class CameraCompactBlock(nn.Module):
    """Pre-normalized cross-attention, camera-local self-attention and MLP."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int):
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, queries: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        normalized_patches = self.patch_norm(patches)
        delta, _ = self.cross_attn(
            self.cross_query_norm(queries), normalized_patches, normalized_patches,
            need_weights=False,
        )
        queries = queries + delta
        normalized_queries = self.self_attn_norm(queries)
        delta, _ = self.self_attn(
            normalized_queries, normalized_queries, normalized_queries,
            need_weights=False,
        )
        queries = queries + delta
        return queries + self.ffn(self.ffn_norm(queries))


def _require_values(condition: torch.Tensor, message: str) -> None:
    """Validate CPU metadata directly; CUDA metadata needs no host scalar read."""
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class CameraCompactResampler(nn.Module):
    """Produce the configured number of compacts for every present camera.

    ``queries_per_source`` and weights are shared across cameras. Every depth
    layer has its own parameters, and self-attention mixes sources only within
    the same camera. Identity IDs belong to a stable configured vocabulary,
    rather than being reassigned when cameras are reordered or missing.
    Refinement uses ``dino_dim``; final pooling uses ``qwen_dim`` directly,
    with its own head split and no post-pooling width projection.

    CPU masks/IDs are supported so callers can prepare valid-row indices without
    reading a CUDA scalar. CUDA masks also work, but dynamic ``nonzero`` may
    synchronize while finding their valid-row inventory.
    A tuple/list assigns counts to stable input camera slots and returns a
    packed ``[B, sum(counts), H]`` inventory; scalar counts keep the legacy API.
    """

    def __init__(
        self,
        qwen_dim: int,
        dino_dim: int,
        *,
        queries_per_source: int = 4,
        depth: int = 1,
        num_heads: int = 8,
        pool_num_heads: int | None = None,
        ff_mult: int = 4,
        camera_identity_count: int = 32,
        tokens_per_camera: int | tuple[int, ...] = 1,
    ):
        super().__init__()
        options = dict(qwen_dim=qwen_dim, dino_dim=dino_dim,
            queries_per_source=queries_per_source, depth=depth, num_heads=num_heads,
            ff_mult=ff_mult, camera_identity_count=camera_identity_count)
        if any(type(value) is not int or value <= 0 for value in options.values()):
            raise ValueError("Camera resampler dimensions and depth must be positive integers")
        if dino_dim % num_heads:
            raise ValueError("dino_dim must be divisible by num_heads")
        self.qwen_dim = qwen_dim
        self.dino_dim = dino_dim
        self.queries_per_source = queries_per_source
        self.camera_identity_count = camera_identity_count
        if isinstance(tokens_per_camera, (tuple, list)):
            self.camera_token_counts = camera_token_counts(tokens_per_camera, len(tokens_per_camera))
        else:
            camera_token_counts(tokens_per_camera, 1)
            self.camera_token_counts = None
        self.tokens_per_camera = tokens_per_camera
        self.max_tokens_per_camera = max(self.camera_token_counts) if self.camera_token_counts else tokens_per_camera
        self.pool_num_heads = resolve_camera_pool_num_heads(qwen_dim, pool_num_heads)
        self.query_proj = nn.Linear(qwen_dim, queries_per_source * dino_dim)
        self.blocks = nn.ModuleList(
            CameraCompactBlock(dino_dim, num_heads, ff_mult) for _ in range(depth)
        )
        self.pool_seed = nn.Parameter(torch.empty(1, self.max_tokens_per_camera, qwen_dim))
        self.camera_identity_embedding = nn.Embedding(camera_identity_count, qwen_dim)
        self.pool_query_norm = nn.LayerNorm(qwen_dim)
        self.pool_tokens_norm = nn.LayerNorm(dino_dim)
        self.pool_attn = nn.MultiheadAttention(qwen_dim, self.pool_num_heads,
            kdim=dino_dim, vdim=dino_dim, batch_first=True)
        self.output_norm = nn.LayerNorm(qwen_dim)
        nn.init.normal_(self.pool_seed, std=qwen_dim ** -0.5)
        nn.init.normal_(self.camera_identity_embedding.weight, std=qwen_dim ** -0.5)

    def _metadata(self, patches, camera_mask, camera_ids):
        batch, cameras = patches.shape[:2]
        if not torch.is_tensor(camera_mask) or camera_mask.dtype != torch.bool:
            raise ValueError("camera_mask must be a boolean tensor [B, C]")
        if tuple(camera_mask.shape) != (batch, cameras):
            raise ValueError("camera_mask must have shape [B, C]")
        if (not torch.is_tensor(camera_ids)
                or camera_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("camera_ids must be an integer tensor [C] or [B, C]")
        if tuple(camera_ids.shape) == (cameras,):
            camera_ids = camera_ids.unsqueeze(0).expand(batch, -1)
        elif tuple(camera_ids.shape) != (batch, cameras):
            raise ValueError("camera_ids must have shape [C] or [B, C]")
        # Keep CPU-only validation on CPU when both pieces of metadata are CPU.
        # Otherwise validate on the model device rather than copying GPU values
        # to the host. Masked slots may use the explicit -1 padding identity.
        metadata_device = (torch.device("cpu") if camera_mask.device.type == "cpu"
            and camera_ids.device.type == "cpu" else patches.device)
        mask = camera_mask.to(device=metadata_device)
        ids = camera_ids.to(device=metadata_device, dtype=torch.long)
        valid_ids = (ids >= 0) & (ids < self.camera_identity_count)
        _require_values((valid_ids | (~mask & (ids == -1))).all(),
            "Present camera IDs must be inside the identity vocabulary; only masked IDs may be -1")
        # Masked IDs are replaced by distinct sentinels before checking that
        # the same real camera is not accidentally supplied twice in one row.
        sentinels = self.camera_identity_count + torch.arange(cameras, device=metadata_device)
        ordered = torch.where(mask, ids, sentinels).sort(dim=-1).values
        duplicate = ((ordered[:, 1:] == ordered[:, :-1])
            & (ordered[:, 1:] < self.camera_identity_count))
        _require_values(~duplicate.any(), "Present camera IDs must be unique within each batch row")
        valid_rows = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        return (mask.to(patches.device), ids.to(patches.device),
            valid_rows.to(patches.device))

    def _empty_result(self, sources, patches, mask, ids, *, dummy=None):
        # Empty slices never read absent-camera NaNs. Distributed callers retain
        # the executed dummy graph instead of accessing dedicated parameters
        # outside their normal module-forward callback boundaries.
        zero = sources[:0].sum() + patches[:0].sum()
        if dummy is None:
            for parameter in self.parameters():
                zero = zero + parameter.reshape(-1)[:0].sum()
        else:
            zero = zero + dummy.tokens[:0].sum() + dummy.refined_tokens[:0].sum()
        dtype = self.pool_attn.out_proj.weight.dtype
        if torch.is_autocast_enabled(sources.device.type):
            dtype = torch.get_autocast_dtype(sources.device.type)
        zero = zero.to(dtype)
        if self.camera_token_counts is not None:
            token_shape = (mask.shape[0], sum(self.camera_token_counts), self.qwen_dim)
        else:
            token_shape = (*mask.shape, self.qwen_dim) if self.tokens_per_camera == 1 else (
                *mask.shape, self.tokens_per_camera, self.qwen_dim)
        tokens = zero.expand(token_shape)
        refined = zero.expand(*mask.shape,
            sources.shape[1] * self.queries_per_source, self.dino_dim)
        return CameraCompactResult(tokens, mask, ids, refined)

    def forward(
        self,
        source_queries: torch.Tensor,
        camera_patches: torch.Tensor,
        *,
        camera_mask: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> CameraCompactResult:
        """Map sources/patches to tokens ``[B,C,M,H]`` and refined queries.

        M=1 retains the legacy squeezed ``[B,C,H]`` result; per-camera counts
        return packed ``[B,sum(counts),H]``. Presence and stable identities
        remain camera metadata ``[B,C]`` for either inventory.
        """
        if (source_queries.ndim != 3 or source_queries.shape[-1] != self.qwen_dim
                or source_queries.shape[1] == 0 or not source_queries.is_floating_point()):
            raise ValueError("source_queries must be floating point [B, S>0, qwen_dim]")
        if (camera_patches.ndim != 4 or camera_patches.shape[-1] != self.dino_dim
                or camera_patches.shape[0] != source_queries.shape[0]
                or camera_patches.shape[2] == 0 or not camera_patches.is_floating_point()):
            raise ValueError("camera_patches must be floating point [B, C, P>0, dino_dim]")
        if source_queries.device != camera_patches.device:
            raise ValueError("Source queries and camera patches must share a device")
        if self.camera_token_counts is not None and camera_patches.shape[1] != len(self.camera_token_counts):
            raise ValueError("Camera patches must match the configured per-camera token inventory")
        mask, ids, rows = self._metadata(camera_patches, camera_mask, camera_ids)
        if rows.numel() == 0:
            dummy = None
            if dist.is_initialized():
                # Every rank must enter the same projection/MHA/FFN boundaries.
                # One synthetic row is enough, including when the real B or C
                # dimension is zero. No real absent-camera values are selected.
                dummy_cameras = len(self.camera_token_counts) if self.camera_token_counts else 1
                dummy = self.forward(
                    source_queries.new_zeros((1, source_queries.shape[1], self.qwen_dim)),
                    camera_patches.new_zeros((1, dummy_cameras, camera_patches.shape[2], self.dino_dim)),
                    camera_mask=torch.ones(1, dummy_cameras, dtype=torch.bool),
                    camera_ids=torch.arange(dummy_cameras, dtype=torch.long),
                )
            return self._empty_result(source_queries, camera_patches, mask, ids, dummy=dummy)
        batch, cameras, patches_per_camera, _ = camera_patches.shape
        queries = self.query_proj(source_queries).reshape(
            batch, source_queries.shape[1] * self.queries_per_source, self.dino_dim)
        queries = queries.index_select(0, torch.div(rows, cameras, rounding_mode="floor"))
        patches = camera_patches.reshape(batch * cameras, patches_per_camera, self.dino_dim)
        patches = patches.index_select(0, rows).to(queries.dtype)
        for block in self.blocks:
            queries = block(queries, patches)
        selected_ids = ids.reshape(-1).index_select(0, rows)
        seed = self.pool_seed + self.camera_identity_embedding(selected_ids).unsqueeze(1)
        normalized_queries = self.pool_tokens_norm(queries)
        pooled, _ = self.pool_attn(self.pool_query_norm(seed),
            normalized_queries, normalized_queries, need_weights=False)
        # Match the former projection's autocast output policy even when the
        # residual and normalization promote the intermediate back to FP32.
        tokens = self.output_norm(seed + pooled).to(pooled.dtype)
        if self.camera_token_counts is not None:
            output = tokens.new_zeros(batch * cameras, self.max_tokens_per_camera, self.qwen_dim)
            dense = output.index_copy(0, rows, tokens).reshape(batch, cameras, self.max_tokens_per_camera, self.qwen_dim)
            output = torch.cat([dense[:, camera, :count] for camera, count in enumerate(self.camera_token_counts)], dim=1)
        elif self.tokens_per_camera == 1:
            tokens = tokens.squeeze(1)
            output = tokens.new_zeros(batch * cameras, self.qwen_dim).index_copy(0, rows, tokens)
            output = output.reshape(batch, cameras, self.qwen_dim)
        else:
            output = tokens.new_zeros(batch * cameras, self.tokens_per_camera, self.qwen_dim)
            output = output.index_copy(0, rows, tokens).reshape(
                batch, cameras, self.tokens_per_camera, self.qwen_dim)
        refined = queries.new_zeros(batch * cameras, queries.shape[1], self.dino_dim)
        refined = refined.index_copy(0, rows, queries)
        return CameraCompactResult(output,
            mask, ids, refined.reshape(batch, cameras, queries.shape[1], self.dino_dim))
