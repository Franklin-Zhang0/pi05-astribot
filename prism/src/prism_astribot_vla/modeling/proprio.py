"""Proprioception encoders for PRISM-VLA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PROPRIO_ENCODER_MLP = "mlp"
PROPRIO_ENCODER_EGO_STRUCTURED_ROPE = "ego_structured_rope"
_PROPRIO_ENCODER_TYPES = {
    PROPRIO_ENCODER_MLP,
    PROPRIO_ENCODER_EGO_STRUCTURED_ROPE,
}


def normalize_proprio_encoder_type(value: str | None) -> str:
    normalized = str(value or PROPRIO_ENCODER_MLP).strip().lower().replace("-", "_")
    if normalized not in _PROPRIO_ENCODER_TYPES:
        raise ValueError(
            f"Unsupported proprio_encoder_type={value!r}; expected one of "
            f"{sorted(_PROPRIO_ENCODER_TYPES)}."
        )
    return normalized


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to ``[B, H, N, D]`` attention states."""

    head_dim = int(x.shape[-1])
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {head_dim}.")
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    frequency_indices = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inv_freq = torch.pow(10000.0, -frequency_indices / float(head_dim))
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(dtype=x.dtype).view(1, 1, x.shape[-2], head_dim // 2)
    sin = angles.sin().to(dtype=x.dtype).view(1, 1, x.shape[-2], head_dim // 2)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = odd * cos + even * sin
    return rotated


class RotarySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "Structured proprio hidden_dim must be positive and divisible by "
                f"num_heads, got hidden_dim={hidden_dim}, num_heads={num_heads}."
            )
        head_dim = hidden_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                "Structured proprio attention head dimension must be even for RoPE, "
                f"got {head_dim}."
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = _apply_rope(query.transpose(1, 2))
        key = _apply_rope(key.transpose(1, 2))
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            self.hidden_dim,
        )
        return self.out_proj(attended)


class RotaryTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}.")
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = RotarySelfAttention(hidden_dim, num_heads, dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.dropout(self.attn(self.attn_norm(hidden)))
        return hidden + self.dropout(self.ff(self.ff_norm(hidden)))


class FingertipConvEncoder(nn.Module):
    """Encode a hand's ordered 5-by-3 fingertip grid with 2D convolutions."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        conv_dim = max(32, hidden_dim // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_dim, kernel_size=(3, 3), padding=(1, 0)),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(conv_dim, hidden_dim)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        if keypoints.shape[-2:] != (5, 3):
            raise ValueError(
                "FingertipConvEncoder expects [..., 5, 3], got "
                f"{tuple(keypoints.shape)}."
            )
        leading_shape = keypoints.shape[:-2]
        flat = keypoints.reshape(-1, 1, 5, 3)
        encoded = self.conv(flat).flatten(1)
        return self.proj(encoded).reshape(*leading_shape, -1)


class EgoStructuredProprioEncoder(nn.Module):
    """Encode compact or separated ego proprio by semantic component."""

    input_dim = 63
    supported_input_dims = (63, 72, 75)
    num_component_tokens = 7

    def __init__(
        self,
        *,
        output_dim: int,
        input_dim: int = 63,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
        action_tokens_per_step: int | None = None,
        rotation_representation: str | None = None,
        proprio_delta_policy: dict[str, bool] | None = None,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if int(input_dim) not in self.supported_input_dims:
            raise ValueError(
                "Structured ego proprio input_dim must be 63 (compact Euler), "
                "72 (compact rotation_6d), or 75 (five-token); got "
                f"{input_dim}."
            )
        self.input_dim = int(input_dim)
        if action_tokens_per_step is None:
            action_tokens_per_step = 5 if self.input_dim == 75 else 3
        self.action_tokens_per_step = int(action_tokens_per_step)
        if self.action_tokens_per_step not in {3, 5}:
            raise ValueError(
                "Structured ego proprio requires 3 or 5 tokens per step, got "
                f"{self.action_tokens_per_step}."
            )
        if rotation_representation is None:
            rotation_representation = (
                "euler" if self.input_dim == 63 else "rotation_6d"
            )
        self.rotation_dim = 3 if str(rotation_representation) == "euler" else 6
        self.pose_dim = 3 + self.rotation_dim
        self.separated_layout = self.action_tokens_per_step == 5
        if self.separated_layout:
            self.token_dim = 15
            self.num_component_tokens = 5
        else:
            self.token_dim = self.input_dim // 3
            self.compact_head_chest = any(
                bool((proprio_delta_policy or {}).get(field_name, False))
                for field_name in (
                    "delta_head_translation",
                    "delta_chest_translation",
                    "delta_chest_rotation",
                )
            )
            self.num_component_tokens = 6 if self.compact_head_chest else 7
        expected_input_dim = self.action_tokens_per_step * self.token_dim
        if self.input_dim != expected_input_dim:
            raise ValueError(
                "Structured ego proprio layout mismatch: "
                f"{self.action_tokens_per_step} tokens * {self.token_dim} dims != "
                f"input_dim {self.input_dim}."
            )
        self.hidden_dim = int(hidden_dim)
        self.translation_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rotation_encoder = nn.Sequential(
            nn.Linear(self.rotation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pose_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fingertip_encoder = FingertipConvEncoder(hidden_dim)
        self.summary_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.role_embeddings = nn.Parameter(
            torch.empty(1, self.num_component_tokens + 1, hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [
                RotaryTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.summary_token, std=self.hidden_dim ** -0.5)
        nn.init.normal_(self.role_embeddings, std=self.hidden_dim ** -0.5)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _component_tokens(self, proprio: torch.Tensor) -> torch.Tensor:
        if self.separated_layout:
            shaped = proprio.reshape(*proprio.shape[:-1], 5, self.token_dim)

            def encode_pose(token: torch.Tensor) -> torch.Tensor:
                translation = self.translation_encoder(token[..., :3])
                rotation = self.rotation_encoder(token[..., 3 : self.pose_dim])
                return self.pose_fusion(torch.cat([translation, rotation], dim=-1))

            head_pose = encode_pose(shaped[..., 0, :])
            left_eef_pose = encode_pose(shaped[..., 1, :])
            left_fingertips = self.fingertip_encoder(
                shaped[..., 2, :].reshape(*proprio.shape[:-1], 5, 3)
            )
            right_eef_pose = encode_pose(shaped[..., 3, :])
            right_fingertips = self.fingertip_encoder(
                shaped[..., 4, :].reshape(*proprio.shape[:-1], 5, 3)
            )
            return torch.stack(
                [
                    head_pose,
                    left_eef_pose,
                    left_fingertips,
                    right_eef_pose,
                    right_fingertips,
                ],
                dim=-2,
            )

        def encode_pose(token: torch.Tensor) -> torch.Tensor:
            translation = self.translation_encoder(token[..., :3])
            rotation = self.rotation_encoder(token[..., 3 : self.pose_dim])
            return self.pose_fusion(torch.cat([translation, rotation], dim=-1))

        left_offset = self.token_dim
        right_offset = 2 * self.token_dim
        if self.compact_head_chest:
            head_pose = encode_pose(proprio[..., : self.pose_dim])
            chest_pose = encode_pose(
                proprio[..., self.pose_dim : 2 * self.pose_dim]
            )
            left_eef = encode_pose(
                proprio[..., left_offset : left_offset + self.pose_dim]
            )
            left_fingertips = self.fingertip_encoder(
                proprio[
                    ..., left_offset + self.pose_dim : left_offset + self.token_dim
                ].reshape(*proprio.shape[:-1], 5, 3)
            )
            right_eef = encode_pose(
                proprio[..., right_offset : right_offset + self.pose_dim]
            )
            right_fingertips = self.fingertip_encoder(
                proprio[
                    ..., right_offset + self.pose_dim : right_offset + self.token_dim
                ].reshape(*proprio.shape[:-1], 5, 3)
            )
            return torch.stack(
                (
                    head_pose,
                    chest_pose,
                    left_eef,
                    left_fingertips,
                    right_eef,
                    right_fingertips,
                ),
                dim=-2,
            )

        head_rotation = self.rotation_encoder(proprio[..., 3 : self.pose_dim])
        left_translation = self.translation_encoder(
            proprio[..., left_offset : left_offset + 3]
        )
        left_rotation = self.rotation_encoder(
            proprio[..., left_offset + 3 : left_offset + self.pose_dim]
        )
        left_fingertips = self.fingertip_encoder(
            proprio[..., left_offset + self.pose_dim : left_offset + self.token_dim].reshape(
                *proprio.shape[:-1], 5, 3
            )
        )
        right_translation = self.translation_encoder(
            proprio[..., right_offset : right_offset + 3]
        )
        right_rotation = self.rotation_encoder(
            proprio[..., right_offset + 3 : right_offset + self.pose_dim]
        )
        right_fingertips = self.fingertip_encoder(
            proprio[..., right_offset + self.pose_dim : right_offset + self.token_dim].reshape(
                *proprio.shape[:-1], 5, 3
            )
        )
        return torch.stack(
            [
                head_rotation,
                left_translation,
                left_rotation,
                left_fingertips,
                right_translation,
                right_rotation,
                right_fingertips,
            ],
            dim=-2,
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.shape[-1] != self.input_dim:
            raise ValueError(
                f"Ego structured proprio expects last dim {self.input_dim}, got "
                f"{proprio.shape[-1]}."
            )
        leading_shape = proprio.shape[:-1]
        flat = proprio.reshape(-1, self.input_dim)
        components = self._component_tokens(flat)
        summary = self.summary_token.expand(flat.shape[0], -1, -1)
        hidden = torch.cat([summary, components], dim=1)
        hidden = hidden + self.role_embeddings.to(dtype=hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.output_proj(self.output_norm(hidden[:, 0]))
        return output.reshape(*leading_shape, -1)


def build_proprio_encoder(
    *,
    encoder_type: str,
    input_dim: int,
    output_dim: int,
    hidden_mult: int,
    structured_hidden_dim: int,
    structured_num_heads: int,
    structured_num_layers: int,
    structured_ff_mult: int,
    structured_dropout: float,
    action_tokens_per_step: int | None = None,
    rotation_representation: str | None = None,
    proprio_delta_policy: dict[str, bool] | None = None,
) -> nn.Module:
    encoder_type = normalize_proprio_encoder_type(encoder_type)
    if encoder_type == PROPRIO_ENCODER_MLP:
        hidden_dim = output_dim * max(1, int(hidden_mult))
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    if input_dim not in EgoStructuredProprioEncoder.supported_input_dims:
        raise ValueError(
            "proprio_encoder_type='ego_structured_rope' requires the canonical "
            "63D compact Euler, 72D compact rotation_6d, or 75D five-token ego layout, got "
            f"proprio_dim={input_dim}."
        )
    return EgoStructuredProprioEncoder(
        output_dim=output_dim,
        input_dim=input_dim,
        hidden_dim=int(structured_hidden_dim),
        num_heads=int(structured_num_heads),
        num_layers=int(structured_num_layers),
        ff_mult=int(structured_ff_mult),
        dropout=float(structured_dropout),
        action_tokens_per_step=action_tokens_per_step,
        rotation_representation=rotation_representation,
        proprio_delta_policy=proprio_delta_policy,
    )


def initialize_proprio_encoder(module: nn.Module) -> None:
    if isinstance(module, FrameSpecificProprioEncoder):
        for encoder in module.encoders.values():
            initialize_proprio_encoder(encoder)
        return
    if isinstance(module, EgoStructuredProprioEncoder):
        module.reset_parameters()
        return
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
    final = module[-1] if isinstance(module, nn.Sequential) else None
    if isinstance(final, nn.Linear):
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


class FrameSpecificProprioEncoder(nn.Module):
    """Independent coordinate-frame encoders with a uniform distributed schedule.

    All modules execute for every batch. This avoids rank-dependent FSDP parameter
    collection order and preserves zero gradients for frames absent on a rank.
    Legacy shared encoder weights can initialize every route losslessly.
    """

    def __init__(self, encoder: nn.Module, frame_names: tuple[str, ...]):
        super().__init__()
        from copy import deepcopy
        if not frame_names or len(set(frame_names)) != len(frame_names):
            raise ValueError("Frame-specific proprio requires unique frame names")
        self.frame_names = tuple(frame_names)
        self.encoders = nn.ModuleDict({name: deepcopy(encoder) for name in frame_names})

    def forward(self, proprio: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        if frame_ids is None or tuple(frame_ids.shape) != tuple(proprio.shape[:-1]):
            raise ValueError("Frame-specific proprio requires explicit IDs matching the input rows")
        if frame_ids.dtype != torch.long:
            raise ValueError("proprio_frame_ids must be int64")
        ids = frame_ids.to(device=proprio.device)
        valid = ((ids >= 0) & (ids < len(self.frame_names))).all()
        if ids.device.type == "cuda":
            torch._assert_async(valid, "Invalid proprio frame ID")
        elif not bool(valid):
            raise ValueError("Invalid proprio frame ID")
        result = None
        for index, encoder in enumerate(self.encoders.values()):
            projected = encoder(proprio)
            selected = torch.where((ids == index).unsqueeze(-1), projected, 0.0)
            result = selected if result is None else result + selected
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Historical checkpoints stored the shared encoder directly at prefix.
        keys = [key for key in state_dict if key.startswith(prefix)
                and not key.startswith(prefix + "encoders.")]
        for key in keys:
            suffix = key[len(prefix):]
            if suffix in self.encoders[self.frame_names[0]].state_dict():
                value = state_dict.pop(key)
                for frame in self.frame_names:
                    state_dict[prefix + "encoders." + frame + "." + suffix] = value.clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
