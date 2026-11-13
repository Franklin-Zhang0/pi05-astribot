"""Camera-token masks and logical positions shared by training and rollout.

Camera slots are tensor storage, not recurrent timesteps. Missing slots are
excluded from attention and from the convolution/linear-attention state update.
"""
from __future__ import annotations

from typing import Any

import torch
from prism_vla.modeling.camera_token_budget import camera_token_counts


def camera_layout_enabled(model: Any) -> bool:
    return getattr(getattr(model, 'config', None), 'compact_token_layout', 'source') == 'camera'


def camera_validity(model, mask, *, batch_size, steps=None, device):
    """Return the present-camera inventory, including storage-only empty slots."""
    cameras = int(model.num_dino_cameras)
    shape = (batch_size, cameras) if steps is None else (batch_size, steps, cameras)
    if mask is None:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if steps is not None:
        mask = mask[:, :steps]
    if tuple(mask.shape) != shape:
        raise ValueError(f'Camera presence must have shape {shape}, got {tuple(mask.shape)}')
    return mask.to(device=device, dtype=torch.bool)


def camera_slot_ids(model, device):
    ids = tuple(getattr(model.config, 'camera_compact_camera_ids', ()))
    if not ids:
        ids = tuple(range(int(model.num_dino_cameras)))
    return torch.tensor(ids, dtype=torch.long, device=device)


def expand_camera_presence(present: torch.Tensor, token_count: int, *, tokens_per_camera=None, layout=None) -> torch.Tensor:
    """Expand camera metadata to flattened camera-major compact slots.

    The original inventory stays ``[..., C]``; consumers choose their compact
    width explicitly, so an already expanded mask is never expanded twice.
    """
    if layout is not None:
        result = layout.camera_mask(present)
        if result.shape[-1] != token_count:
            raise ValueError('Compact token count differs from direct-query layout')
        return result
    cameras = int(present.shape[-1])
    if token_count == cameras:
        return present.bool()
    if tokens_per_camera is not None:
        counts = camera_token_counts(tokens_per_camera, cameras)
        if sum(counts) != token_count:
            raise ValueError('Compact token count differs from configured per-camera inventory')
        # Static concatenation avoids a CUDA repeat_interleave shape readback.
        return torch.cat([present[..., i:i + 1].bool().expand(*present.shape[:-1], count)
            for i, count in enumerate(counts)], dim=-1)
    if cameras == 0 or token_count <= 0 or token_count % cameras:
        raise ValueError('Compact token count must be a positive multiple of the camera inventory')
    return present.bool().repeat_interleave(token_count // cameras, dim=-1)


def camera_step_layout(
    present: torch.Tensor, start_position, *, tokens_per_camera: int | tuple[int, ...] = 1,
    special_tokens: int = 2,
):
    """Build camera-slot and special-token masks with reserved positional gaps.

    The storage width can be fixed for a batch while each row/frame contributes
    a different number of real camera tokens. Absent cameras reserve their
    positional IDs but do not advance convolution or recurrent memory.
    The default suffix is [CLS, Q]; the ACT layout uses [CLS, ACT, Q].
    """
    if present.ndim != 3:
        raise ValueError('Camera inventory must be [B, T, C]')
    if type(special_tokens) is not int or special_tokens <= 0:
        raise ValueError('special_tokens must be a positive integer')
    batch, steps, cameras = present.shape
    counts = camera_token_counts(tokens_per_camera, cameras)
    token_count = sum(counts)
    compact_present = expand_camera_presence(present, token_count, tokens_per_camera=counts)
    valid = torch.cat((compact_present, torch.ones(batch, steps, special_tokens, device=present.device, dtype=torch.bool)), dim=-1)
    start = torch.as_tensor(start_position, device=present.device, dtype=torch.long)
    if start.ndim == 0:
        start = start.expand(batch)
    if tuple(start.shape) != (batch,):
        raise ValueError('Each stream needs one starting logical token position')
    flat = valid.flatten(1)
    positions = start[:, None] + torch.arange(flat.shape[1], device=present.device)[None]
    positions = positions.reshape(batch, steps, token_count + special_tokens)
    return valid, positions, start + flat.shape[1]


def pool_camera_hidden(hidden: torch.Tensor, present: torch.Tensor, *, tokens_per_camera=None, layout=None) -> torch.Tensor:
    """Pool real camera hidden states for the existing global CLS predictor."""
    present = expand_camera_presence(present, hidden.shape[-2], tokens_per_camera=tokens_per_camera, layout=layout)
    weights = present.to(device=hidden.device, dtype=hidden.dtype)
    hidden = torch.where(present.to(hidden.device).unsqueeze(-1), hidden, 0)
    return hidden.sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1)


def mask_camera_tokens(tokens, present, *, tokens_per_camera=None, layout=None):
    """Mask after biased projections too; absent cameras never acquire features."""
    if layout is not None and layout.num_compacts == 1 and tokens.ndim == present.ndim:
        slot_present = layout.camera_mask(present)
        return torch.where(slot_present.to(tokens.device), tokens, 0)
    if present.shape[-1] == 1 and tokens.ndim == present.ndim:
        return torch.where(present.to(tokens.device), tokens, 0)
    present = expand_camera_presence(present, tokens.shape[-2], tokens_per_camera=tokens_per_camera, layout=layout)
    return torch.where(present.to(tokens.device).unsqueeze(-1), tokens, 0)


def camera_compact_loss(model, pred, target, valid, present):
    """Regress ordinary slots; designated global slots have their own target."""
    from prism_vla.modeling.losses import masked_regression_loss
    layout = (getattr(model, 'direct_query_layout', None)
        if getattr(model.config, 'query_design', None) == 'direct_query' else None)
    if (present.shape[-1] == 1 or (layout is not None and layout.num_compacts == 1)) and pred.ndim == present.ndim:
        pred, target = pred.unsqueeze(-2), target.unsqueeze(-2)
    cameras = int(present.shape[-1])
    counts = (tuple(g + t + r for g, t, r in zip(layout.global_counts, layout.task_counts, layout.generated_counts))
        if layout is not None else camera_token_counts(getattr(model.config, 'camera_compact_tokens_per_camera', 1), cameras))
    present = expand_camera_presence(present, pred.shape[-2], tokens_per_camera=counts, layout=layout)
    global_indices = tuple(getattr(model.config, 'global_compact_indices', ()))
    if global_indices:
        if layout is None:
            ordinary_values = [slot not in global_indices for count in counts for slot in range(count)]
        else:
            camera_slot_counts = [0] * cameras
            ordinary_values = []
            for _, camera, _ in layout.slots:
                ordinary_values.append(camera_slot_counts[camera] not in global_indices)
                camera_slot_counts[camera] += 1
        ordinary = torch.tensor(ordinary_values, device=pred.device, dtype=torch.bool)
        present = present.to(pred.device) & ordinary
    mask = present.to(device=pred.device, dtype=pred.dtype) * valid.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    # Mask values before subtraction, so invalid NaNs cannot contaminate loss.
    pred = torch.where(mask.bool().unsqueeze(-1), pred, 0)
    target = torch.where(mask.bool().unsqueeze(-1), target, 0)
    return masked_regression_loss(pred, target, mask=mask,
        loss_type=model.config.loss_type, huber_delta=model.config.huber_delta, mean_dim=-1)


def target_camera_validity(model, mask, valid, frame_start=1):
    if valid.ndim == 1:
        selected = mask[:, frame_start] if mask is not None else None
        return camera_validity(model, selected, batch_size=valid.shape[0], device=valid.device)
    selected = mask[:, frame_start:frame_start + valid.shape[1]] if mask is not None else None
    return camera_validity(model, selected, batch_size=valid.shape[0], steps=valid.shape[1], device=valid.device)


def action_camera_mask(model, mask, valid, frame_start=0, row_indices=None):
    from prism_vla.transition_rows import select_transition_rows
    inventory = target_camera_validity(model, mask, valid, frame_start)
    return select_transition_rows(inventory.reshape(-1, model.num_dino_cameras), valid.reshape(-1).bool(), row_indices)


def _run_qwen_impl(model, *, camera_token_mask=None, main_stream_prefix_tokens=None,
             allow_inactive_observers=False, **kwargs):
    """Call Qwen with an explicit real-token scope for camera-specific layouts."""
    from prism_vla.modeling.action_patch_output import (
        raw_action_token_output_scope, requires_raw_action_token,
    )
    raw_act_output = requires_raw_action_token(model)
    if type(allow_inactive_observers) is not bool:
        raise ValueError('Inactive ACT observer permission must be boolean')
    if not camera_layout_enabled(model):
        if raw_act_output and main_stream_prefix_tokens is not None:
            raise ValueError('Raw ACT output requires the camera-specific main-stream layout')
        if allow_inactive_observers:
            raise ValueError('Inactive ACT timesteps require the camera-specific valid-token scope')
        return model.qwen.model(**kwargs)
    backbone = model.qwen.model
    if getattr(backbone, 'gradient_checkpointing', False) and getattr(model, 'training', False):
        raise ValueError('Camera-specific compact tokens currently require Qwen gradient checkpointing disabled; replay must retain valid-token metadata')
    token_stream = ("main" if main_stream_prefix_tokens is not None else
        "query" if camera_token_mask is not None else "prefix")
    length = int(kwargs['inputs_embeds'].shape[1])
    mask = kwargs.get('attention_mask')
    if camera_token_mask is None:
        if mask is None:
            camera_token_mask = torch.ones(kwargs['inputs_embeds'].shape[:2], device=kwargs['inputs_embeds'].device, dtype=torch.bool)
        elif mask.ndim == 2:
            camera_token_mask = mask[:, -length:].bool()
        elif mask.ndim == 4 and mask.dtype == torch.bool:
            camera_token_mask = mask[..., -length:].diagonal(dim1=-2, dim2=-1).any(dim=1)
        else:
            raise ValueError('Camera-specific Qwen calls need a current-token validity mask')
    if tuple(camera_token_mask.shape) != tuple(kwargs['inputs_embeds'].shape[:2]):
        raise ValueError('Current-token validity does not match Qwen input shape')
    observer_mask = None
    if (main_stream_prefix_tokens is not None
            and bool(getattr(model.config, 'main_stream_action_token', False))):
        prefix = main_stream_prefix_tokens
        if allow_inactive_observers and prefix != 0:
            raise ValueError('Inactive ACT timesteps require a prefix-free cached main-stream call')
        width = model._main_stream_tokens_per_step()
        if (type(prefix) is not int or prefix < 0 or length <= prefix
                or (length - prefix) % width):
            raise ValueError('ACT main-stream calls require a prefix and complete timesteps')
        positions = torch.arange(length)
        action_count = model._main_stream_action_tokens_per_step()
        slot = (positions - prefix) % width
        tail = model._query_tail_tokens_per_step()
        observer_mask = ((positions >= prefix)
            & (slot >= width - tail - action_count) & (slot < width - tail))
        observer_mask = observer_mask.unsqueeze(0).expand(kwargs['inputs_embeds'].shape[0], -1)
        if mask is None or mask.ndim != 4 or mask.dtype != torch.bool:
            raise ValueError('ACT observers require an explicit boolean timestep-causal attention mask')
        from prism_vla.modeling.act_observer import mask_action_observer_attention
        kwargs['attention_mask'] = mask_action_observer_attention(mask, observer_mask)
        mask = kwargs['attention_mask']
    if mask is not None and mask.ndim == 4:
        if mask.dtype != torch.bool:
            raise ValueError('Camera-specific blockwise attention requires a boolean mask')
        prefix = torch.ones(mask.shape[0], mask.shape[-1] - length, device=mask.device, dtype=torch.bool)
        key_valid = torch.cat((prefix, camera_token_mask.to(mask.device).bool()), dim=-1)
        allowed = mask & key_valid[:, None, None, :]
        # Ignored query rows need a finite result too: some eager/fast attention
        # paths turn an all-masked row into NaNs, which can leak at later layers.
        # Give only those unused rows their own diagonal. Real queries continue
        # to exclude every absent camera key.
        diagonal = torch.eye(length, device=mask.device, dtype=torch.bool)
        diagonal = torch.nn.functional.pad(diagonal, (mask.shape[-1] - length, 0))
        kwargs['attention_mask'] = torch.where(
            camera_token_mask.to(mask.device)[:, None, :, None], allowed, diagonal[None, None],
        )
    from prism_vla.modeling.qwen35_deltanet_patch import qwen35_valid_token_scope
    if observer_mask is not None:
        from prism_vla.modeling.act_observer import action_observer_scope
        with action_observer_scope(observer_mask):
            direct = getattr(model.config, 'query_design', None) == 'direct_query'
            with qwen35_valid_token_scope(camera_token_mask, observer_token_mask=observer_mask,
                    observer_tail_tokens=tail, allow_invalid_observer_tail=direct,
                    stream=token_stream, allow_inactive_observers=allow_inactive_observers):
                with raw_action_token_output_scope(backbone, observer_mask, enabled=raw_act_output):
                    from prism_vla.modeling.qwen_activation_offload import qwen_saved_activation_cpu_offload
                    with qwen_saved_activation_cpu_offload(
                            getattr(model.config, 'qwen_saved_activation_cpu_offload_min_bytes', 0),
                            training=getattr(model, 'training', False)):
                        return backbone(**kwargs)
    with qwen35_valid_token_scope(camera_token_mask, stream=token_stream):
        with raw_action_token_output_scope(backbone, None, enabled=raw_act_output):
            from prism_vla.modeling.qwen_activation_offload import qwen_saved_activation_cpu_offload
            with qwen_saved_activation_cpu_offload(
                    getattr(model.config, 'qwen_saved_activation_cpu_offload_min_bytes', 0),
                    training=getattr(model, 'training', False)):
                return backbone(**kwargs)


def run_qwen(model, *, camera_token_mask=None, main_stream_prefix_tokens=None, **kwargs):
    """Preserve legacy execution; opt-in experts receive detached native K/V."""
    if bool(getattr(getattr(model, "config", None), "text_input_deepstack", False)):
        memory = getattr(model, "_text_deepstack_memory", None)
        if memory is None:
            raise RuntimeError("Text input deepstack memory must be encoded before backbone execution")
        # Snapshot the immutable memory in explicit decoder kwargs so gradient
        # checkpoint replay cannot observe a later instruction's active memory.
        kwargs["text_deepstack_memory"] = memory
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            raise ValueError("Text input deepstack requires explicit native position_ids")
        kwargs["position_ids"] = position_ids + int(memory.attention_mask.shape[-1])
        if kwargs["inputs_embeds"].shape[1] == 0:
            from types import SimpleNamespace
            return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"],
                past_key_values=kwargs.get("past_key_values"))
    from prism_vla.modeling.action_expert_cache import needs_action_expert, capture_native_kv
    expert_main = main_stream_prefix_tokens is not None and needs_action_expert(model)
    if expert_main:
        if getattr(model.qwen.model, "gradient_checkpointing", False):
            raise ValueError("Native action-expert K/V requires backbone gradient checkpointing disabled")
        kwargs["use_cache"] = True
    output = _run_qwen_impl(model, camera_token_mask=camera_token_mask,
        main_stream_prefix_tokens=main_stream_prefix_tokens, **kwargs)
    if expert_main:
        mask = kwargs.get("attention_mask")
        if camera_token_mask is None and camera_layout_enabled(model):
            length = kwargs["inputs_embeds"].shape[1]
            if mask is not None and mask.ndim == 4:
                camera_token_mask = mask[..., -length:].diagonal(dim1=-2, dim2=-1).any(dim=1)
            elif mask is not None:
                camera_token_mask = mask[:, -length:].bool()
        if camera_token_mask is not None:
            length = kwargs["inputs_embeds"].shape[1]
            keys = output.past_key_values.get_seq_length()
            present = torch.cat((torch.ones(camera_token_mask.shape[0], keys - length,
                device=camera_token_mask.device, dtype=torch.bool), camera_token_mask.bool()), dim=-1)
            if mask is not None and mask.ndim == 4:
                mask = mask & present[:, None, None, :]
            elif mask is not None:
                mask = mask.bool() & present
            else:
                mask = present
        output = capture_native_kv(output, attention_mask=mask,
            query_length=kwargs["inputs_embeds"].shape[1], prefix_tokens=main_stream_prefix_tokens,
            position_ids=kwargs.get("position_ids"))
    return output
