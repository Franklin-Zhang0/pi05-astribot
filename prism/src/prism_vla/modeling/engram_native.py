"""Opt-in joint-space Engram interfaces over PRISM's existing DiT trunk.

The five 15D tokens are storage slots for a native 75D vector here. They do not
denote camera/EEF poses. Existing geometric heads and proprio encoders remain
separate and retain their checkpoint semantics.
"""
from __future__ import annotations

import torch
from torch import nn

from prism_vla.modeling.condition_mask import ConditionList, condition_mask
from prism_vla.modeling.idm_backends import (
    TemporalCrossAttentionFlowMatchingIDM, _combine_action_chunk_masks,
)
from prism_vla.modeling.losses import masked_regression_loss


ENGRAM_ROUTE_NAMES = (
    "dex_w1_gripper", "dex_w1_linker_l6", "dex_w1_linker_l20", "galbot_inspire_rh56dfx_hand",
)
ENGRAM_NATIVE_WIDTH = 75
ENGRAM_NATIVE_SLOT_WIDTH = 15
ENGRAM_NATIVE_SLOTS = 5


def _assert_tensor(value: torch.Tensor, condition: torch.Tensor, message: str) -> None:
    if value.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _route_ids(ids: torch.Tensor, shape: tuple[int, ...], *, device) -> torch.Tensor:
    if not torch.is_tensor(ids) or ids.dtype != torch.long or tuple(ids.shape) != shape:
        raise ValueError(f"Native Engram embodiment_ids must be int64 with shape {shape}")
    ids = ids.to(device=device)
    _assert_tensor(ids, ((ids >= 0) & (ids < len(ENGRAM_ROUTE_NAMES))).all(), "Unknown native Engram embodiment ID")
    return ids


def native_structural_mask(ids: torch.Tensor) -> torch.Tensor:
    """Return hardware coordinates [B,75], independent of missing annotations."""
    ids = _route_ids(ids, (ids.shape[0],), device=ids.device)
    table = torch.zeros((4, ENGRAM_NATIVE_WIDTH), dtype=torch.bool, device=ids.device)
    table[:3, :20] = True
    table[3, :14] = True
    for route, width in enumerate((2, 12, 32, 12)):
        table[route, 20:20 + width] = True
    return table[ids]


class EngramJointProprioEncoder(nn.Module):
    """Encode normalized joints/hands and explicit availability separately.

    Unknown values (including NaNs) are removed before the first projection.
    Availability bits distinguish an observed zero from an unavailable value;
    the numeric hardware route never depends on the prompt's text format.
    """

    def __init__(self, cond_dim: int, *, hidden_dim: int | None = None):
        super().__init__()
        if type(cond_dim) is not int or cond_dim < 1:
            raise ValueError("cond_dim must be a positive integer")
        hidden_dim = hidden_dim if hidden_dim is not None else 2 * cond_dim
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        self.input_proj = nn.Linear(2 * ENGRAM_NATIVE_WIDTH, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, cond_dim)
        self.route_embeddings = nn.Embedding(len(ENGRAM_ROUTE_NAMES), cond_dim)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor, available: torch.Tensor,
                embodiment_ids: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(values) or values.ndim < 2 or values.shape[-1] != ENGRAM_NATIVE_WIDTH:
            raise ValueError("Native Engram proprio values must have trailing width75")
        if not values.is_floating_point():
            raise ValueError("Native Engram proprio values must be floating point")
        if not torch.is_tensor(available) or available.dtype != torch.bool or available.shape != values.shape:
            raise ValueError("Native Engram proprio availability must be boolean and match values")
        ids = _route_ids(embodiment_ids, tuple(values.shape[:-1]), device=values.device)
        structural = native_structural_mask(ids.flatten()).reshape(values.shape)
        available = available.to(values.device) & structural
        safe = torch.where(available, values, 0)
        _assert_tensor(safe, torch.isfinite(safe).all(), "Available native proprio contains nonfinite values")
        inputs = torch.cat((safe, available.to(values.dtype)), dim=-1)
        return self.output_proj(self.activation(self.input_proj(inputs))) + self.route_embeddings(ids)


class _NativeConditionList(ConditionList):
    def __init__(self, conditions, *, embodiment_ids, action_mask):
        valid = condition_mask(conditions)
        if valid is None:
            valid = torch.ones((conditions[0].shape[0], len(conditions)), dtype=torch.bool,
                               device=conditions[0].device)
        super().__init__(conditions, valid_mask=valid,
            end_effector_type_ids=getattr(conditions, "end_effector_type_ids", None),
            backbone_kv=getattr(conditions, "backbone_kv", None))
        self.embodiment_ids = embodiment_ids
        self.native_action_mask = action_mask


class _RoutedNativeFinalLayer(nn.Module):
    """Time-modulated velocity projections, one per numeric embodiment."""

    def __init__(self, width: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.projections = nn.ModuleDict({name: nn.Linear(width, ENGRAM_NATIVE_SLOT_WIDTH)
                                         for name in ENGRAM_ROUTE_NAMES})
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        for layer in self.projections.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden, time, *, end_effector_type_ids=None, embodiment_ids=None):
        # All ranks execute the same four projection calls. An absent route has
        # exactly zero loss gradient without changing HSDP parameter schedules.
        ids = _route_ids(embodiment_ids, (hidden.shape[0],), device=hidden.device)
        shift, scale = self.adaLN_modulation(time).chunk(2, dim=-1)
        hidden = self.norm_final(hidden) * (1 + scale[:, None]) + shift[:, None]
        result = None
        for index, layer in enumerate(self.projections.values()):
            values = layer(hidden)
            selected = ids[:, None, None] == index
            result = torch.where(selected, values, torch.zeros_like(values) if result is None else result)
        return result


class EngramJointDiT(TemporalCrossAttentionFlowMatchingIDM):
    """Native joint/hands flow with PRISM's temporal cross-attention decoder.

    ``physical_horizon`` is required: this component does not select a recipe.
    Packed actions are [B,physical_horizon*5,15], with flattened physical rows
    storing chunk-relative joints0:20, native actuator deltas20:52 and padding
    52:75. Hand exports retain percentage-point units, not invented radians or
    3D keypoints. Galbot uses14 joint slots.
    """

    def __init__(self, *, physical_horizon: int, flow_qk_norm: bool, **trunk_options):
        if type(physical_horizon) is not int or physical_horizon < 1:
            raise ValueError("physical_horizon must be an explicitly selected positive integer")
        forbidden = {"action_dim", "action_horizon", "action_token_types", "robot_action_layout",
                     "flow_output_projection", "proprio_as_action_token"} & trunk_options.keys()
        if forbidden:
            raise ValueError(f"Native Engram owns its action/proprio packing: {sorted(forbidden)}")
        self.physical_horizon = physical_horizon
        # No geometric sampling layout or per-hand camera/EEF token semantics.
        options = {"action_clip_abs_max": None, "sampled_action_clip_abs_max": None, **trunk_options}
        super().__init__(action_dim=15, action_horizon=physical_horizon * 5,
            action_token_types=(), robot_action_layout="", flow_output_projection="shared",
            proprio_as_action_token=False, flow_qk_norm=flow_qk_norm, **options)
        self.final_layer = _RoutedNativeFinalLayer(self.hidden_dim)
        self.slot_embeddings = nn.Embedding(5, self.hidden_dim)
        nn.init.normal_(self.slot_embeddings.weight, std=0.02)

    def _native_conditions(self, conditions, embodiment_ids, action_mask=None):
        ids = _route_ids(embodiment_ids, (conditions[0].shape[0],), device=conditions[0].device)
        structural = native_structural_mask(ids)[:, None].expand(-1, self.physical_horizon, -1)
        structural = structural.reshape(-1, self.action_horizon, self.action_dim)
        if action_mask is not None:
            if not torch.is_tensor(action_mask) or action_mask.dtype != torch.bool:
                raise ValueError("Native action availability must be a boolean tensor")
            if action_mask.shape == (ids.shape[0], 75):
                action_mask = action_mask[:, None].expand(-1, self.physical_horizon, -1)
                action_mask = action_mask.reshape_as(structural)
            if action_mask.shape != structural.shape:
                raise ValueError("Native action availability must be [B,75] or packed [B,H*5,15]")
            structural = structural & action_mask.to(structural.device)
        return _NativeConditionList(conditions, embodiment_ids=ids, action_mask=structural)

    def _vector_field(self, cond_list, noisy_actions, times, proprio_token=None, static_kv_caches=None):
        if not isinstance(cond_list, _NativeConditionList):
            raise ValueError("Native Engram diffusion requires explicit numeric routing metadata")
        if proprio_token is not None:
            raise ValueError("Native proprio belongs in the pretrained condition stream, not an action-token bypass")
        if noisy_actions.shape != cond_list.native_action_mask.shape:
            raise ValueError("Native noisy action shape differs from the packed horizon")
        # Reuse the real PRISM trunk; the only custom action computation is
        # freshly initialized native IO/slot embeddings and the routed outputs.
        positions = torch.arange(self.action_horizon, device=noisy_actions.device)
        hidden = self.action_proj(noisy_actions)
        hidden = hidden + self.pos_embed(positions).to(hidden.dtype)[None]
        hidden = hidden + self.slot_embeddings(positions % 5).to(hidden.dtype)[None]
        tokens = self._condition_tokens(cond_list).to(hidden.dtype)
        valid = condition_mask(cond_list, device=hidden.device)
        time = self._time_embedding(times).to(hidden.dtype)
        for index, block in enumerate(self.blocks):
            cache = {} if static_kv_caches is None else {"static_kv_cache": static_kv_caches[index]}
            hidden = block(hidden, tokens, time,
                **({} if valid is None else {"cond_key_padding_mask": ~valid}), **cache)
        velocity = self.final_layer(hidden, time, embodiment_ids=cond_list.embodiment_ids)
        return torch.where(cond_list.native_action_mask, velocity, 0)

    def training_step(self, cond_list, *, embodiment_ids, action_chunks,
                      action_chunk_valid_steps=None, action_chunk_loss_mask=None,
                      flow_times=None, flow_noise=None, compute_action_metrics=False,
                      replan_horizon=None, **unsupported):
        if any(value is not None for value in unsupported.values()):
            raise ValueError(f"Unsupported native Engram training inputs: {sorted(unsupported)}")
        if action_chunks is None or action_chunks.shape != (cond_list[0].shape[0], self.action_horizon, 15):
            raise ValueError("Native action_chunks must match [B,physical_horizon*5,15]")
        loss_mask = _combine_action_chunk_masks(action_chunk_valid_steps, action_chunk_loss_mask,
            horizon=self.action_horizon, action_dim=15, device=action_chunks.device, dtype=torch.float32)
        if loss_mask is not None:
            _assert_tensor(loss_mask, torch.isfinite(loss_mask).all() & ((loss_mask == 0) | (loss_mask == 1)).all(),
                           "Native action loss mask must contain only zero/one availability")
        native = self._native_conditions(cond_list, embodiment_ids, None if loss_mask is None else loss_mask.bool())
        mask = native.native_action_mask
        targets = torch.where(mask, action_chunks.to(cond_list[0].dtype), 0)
        _assert_tensor(targets, torch.isfinite(targets).all(), "Available native action targets contain nonfinite values")
        if (flow_times is None) != (flow_noise is None):
            raise ValueError("flow_times and flow_noise must be provided together")
        if flow_times is None:
            times = torch.rand(targets.shape[0], device=targets.device, dtype=targets.dtype)
            noise = torch.randn_like(targets)
        else:
            times, noise = flow_times, flow_noise
            for name, value, shape in (("flow_times", times, (targets.shape[0],)),
                                       ("flow_noise", noise, tuple(targets.shape))):
                if (not torch.is_tensor(value) or tuple(value.shape) != shape
                        or value.device != targets.device or value.dtype != targets.dtype):
                    raise ValueError(f"{name} must match native target shape/device/dtype")
                _assert_tensor(value, torch.isfinite(value).all(), f"{name} must be finite")
            _assert_tensor(times, ((times >= 0) & (times <= 1)).all(), "flow_times must lie in [0,1]")
        bridge = (1 - times[:, None, None]) * noise + times[:, None, None] * targets
        # Unavailable coordinates stay independent noise: no pseudo-labels or
        # masked NaNs can enter observed coordinates through self-attention.
        bridge = torch.where(mask, bridge, noise)
        prediction = self._vector_field(native, bridge, times)
        velocity = torch.where(mask, targets - noise, 0)
        loss = masked_regression_loss(prediction.float(), velocity.float(), mask=mask.float(),
            loss_type=self.loss_type, huber_delta=self.huber_delta)
        metrics = {"loss": loss, "action_loss_weight": mask.float().sum().detach(),
                   "action_acc": loss.detach().new_zeros(())}
        if compute_action_metrics:
            sampled = self.generate_actions(cond_list, embodiment_ids=embodiment_ids, action_mask=mask)
            horizon = min(replan_horizon or self.action_horizon, self.action_horizon)
            error = sampled[:, :horizon] - targets[:, :horizon]
            selected = mask[:, :horizon].float()
            count = selected.sum().clamp_min(1)
            metrics.update(action_chunk_mse=(error.float().square() * selected).sum() / count,
                           action_chunk_l1=(error.float().abs() * selected).sum() / count,
                           action_chunk_metric_weight=selected.sum().detach())
        return metrics

    @torch.no_grad()
    def generate_actions(self, cond_list, *, embodiment_ids, action_mask=None, initial_noise=None,
                         sampling_steps=None):
        native = self._native_conditions(cond_list, embodiment_ids, action_mask)
        steps = self.flow_sampling_steps if sampling_steps is None else sampling_steps
        if type(steps) is not int or steps < 1:
            raise ValueError("Native flow sampling_steps must be a positive integer")
        shape = (cond_list[0].shape[0], self.action_horizon, 15)
        if initial_noise is None:
            initial_noise = torch.randn(shape, device=cond_list[0].device, dtype=cond_list[0].dtype)
        elif not torch.is_tensor(initial_noise) or tuple(initial_noise.shape) != shape:
            raise ValueError("Native initial_noise must match packed action shape")
        state = initial_noise.to(device=cond_list[0].device, dtype=cond_list[0].dtype)
        _assert_tensor(state, torch.isfinite(state).all(), "Native initial noise must be finite")
        noise = state
        dt = 1.0 / steps
        caches = [{} for _ in self.blocks]
        for index in range(steps):
            times = state.new_full((shape[0],), index / steps)
            state = state + dt * self._vector_field(native, state, times, static_kv_caches=caches)
            state = torch.where(native.native_action_mask, state, noise)
        return torch.where(native.native_action_mask, state, 0)


def engram_time_adamw_parameter_names(model: nn.Module, *, prefix="engram_action_head") -> tuple[str, ...]:
    """Return exact native time/modulation matrix FQNs for FP32 AdamW routing.

    Source-run keyword lists contain only ``idm.heads.*`` names. The finetuning
    recipe must append these new FQNs to its AdamW keyword policy; checkpoint
    optimizer migration is neither needed nor allowed for fresh initialization.
    """
    selected = []
    for name, parameter in model.named_parameters():
        full_name = f"{prefix}.{name}" if prefix else name
        if parameter.ndim != 2 or not name.endswith(".weight"):
            continue
        if (name in {"time_proj.0.weight", "time_proj.2.weight"}
                or name == "final_layer.adaLN_modulation.1.weight"
                or name.startswith("blocks.") and name.endswith(".adaLN_modulation.1.weight")):
            selected.append(full_name)
    return tuple(sorted(selected))


def robot_eef_conditions(conditions, embodiment_ids):
    """Retain the shared condition layout and attach numeric hand identities."""
    ids = _route_ids(embodiment_ids, (conditions[0].shape[0],), device=conditions[0].device)
    valid = condition_mask(conditions)
    if valid is None:
        valid = torch.ones((ids.shape[0], len(conditions)), device=ids.device, dtype=torch.bool)
    # This head supervises only EEF pose slots. Its dormant hand/gripper
    # projections still require the source binary routing metadata.
    binary = (ids == 0).long()[:, None].expand(-1, 2)
    return ConditionList(conditions, valid_mask=valid, end_effector_type_ids=binary,
                         backbone_kv=getattr(conditions, "backbone_kv", None))


def robot_eef_action_mask(embodiment_ids, *, horizon, action_mask=None):
    """Allow W1 left/right XYZ+rotation6D pose slots, never hand keypoints."""
    if type(horizon) is not int or horizon < 1 or horizon % 5:
        raise ValueError("Robot EEF horizon must contain complete five-token physical steps")
    ids = _route_ids(embodiment_ids, (embodiment_ids.shape[0],), device=embodiment_ids.device)
    step = torch.zeros((5, 15), dtype=torch.bool, device=ids.device)
    step[[1, 3], :9] = True
    structural = step.repeat(horizon // 5, 1)[None].expand(ids.shape[0], -1, -1)
    structural = structural & (ids < 3)[:, None, None]
    if action_mask is None:
        return structural
    if not torch.is_tensor(action_mask) or action_mask.shape != structural.shape:
        raise ValueError("Robot EEF mask must match [B,physical_horizon*5,15]")
    values = action_mask.to(device=ids.device)
    _assert_tensor(values, torch.isfinite(values).all() & ((values == 0) | (values == 1)).all(),
                   "Robot EEF mask must contain only zero/one availability")
    _assert_tensor(values, (~(values.bool() & ~structural)).all(),
                   "Robot EEF supervision may only label W1 pose slots1/3 XYZ+rotation6D")
    return structural & values.bool()


def hybrid_native_inference_mask(native_head, conditions, *, embodiment_ids, state_mask, eef_mask):
    """Retain body/head0:6 and hand joints; replace only qualified W1 arms."""
    native = native_head._native_conditions(conditions, embodiment_ids, state_mask).native_action_mask
    packed = native.reshape(native.shape[0], native_head.physical_horizon, 75).clone()
    poses = eef_mask.reshape(native.shape[0], native_head.physical_horizon, 5, 15)
    for pose_slot, start, stop in ((1, 6, 13), (3, 13, 20)):
        qualified = poses[:, :, pose_slot, :9].all(dim=-1)
        packed[:, :, start:stop] &= ~qualified[:, :, None]
    return packed.reshape_as(native)


def robot_eef_flow_training_step(head, conditions, *, embodiment_ids, targets, loss_mask,
                                 valid_steps=None, flow_times=None, flow_noise=None,
                                 compute_action_metrics=False, replan_horizon=None):
    """Supervise the existing robot expert with observed XYZ/rotation6D offsets.

    This opt-in boundary sanitizes unknown feedback before interpolation and
    leaves it independent noise. No legacy source expert implementation or
    geometric head weights are replaced or reinterpreted.
    """
    expected = (conditions[0].shape[0], head.action_horizon, 15)
    if not torch.is_tensor(targets) or tuple(targets.shape) != expected or loss_mask is None:
        raise ValueError("Hybrid robot EEF targets and explicit masks must match the packed source horizon")
    mask = robot_eef_action_mask(embodiment_ids, horizon=head.action_horizon, action_mask=loss_mask)
    valid = _combine_action_chunk_masks(valid_steps, None, horizon=head.action_horizon,
        action_dim=15, device=targets.device, dtype=torch.float32)
    if valid is not None:
        mask = mask & valid.bool()
    cond = robot_eef_conditions(conditions, embodiment_ids)
    safe = torch.where(mask, targets.to(conditions[0].dtype), 0)
    _assert_tensor(safe, torch.isfinite(safe).all(), "Available observed robot EEF labels must be finite")
    if (flow_times is None) != (flow_noise is None):
        raise ValueError("Robot EEF flow_times and flow_noise must be provided together")
    if flow_times is None:
        times = torch.rand(safe.shape[0], device=safe.device, dtype=safe.dtype)
        noise = torch.randn_like(safe)
    else:
        times, noise = flow_times, flow_noise
        for name, value, shape in (("flow_times", times, (safe.shape[0],)), ("flow_noise", noise, expected)):
            if (not torch.is_tensor(value) or tuple(value.shape) != shape
                    or value.dtype != safe.dtype or value.device != safe.device):
                raise ValueError(f"Robot EEF {name} must match target shape/device/dtype")
            _assert_tensor(value, torch.isfinite(value).all(), f"Robot EEF {name} must be finite")
        _assert_tensor(times, ((times >= 0) & (times <= 1)).all(), "Robot EEF flow times must lie in [0,1]")
    bridge = (1 - times[:, None, None]) * noise + times[:, None, None] * safe
    bridge = torch.where(mask, bridge, noise)
    prediction = head._vector_field(cond, bridge, times)
    prediction = torch.where(mask, prediction, 0)
    velocity = torch.where(mask, safe - noise, 0)
    loss = masked_regression_loss(prediction.float(), velocity.float(), mask=mask.float(),
        loss_type=head.loss_type, huber_delta=head.huber_delta)
    metrics = {"loss": loss, "action_loss_weight": mask.float().sum().detach()}
    if compute_action_metrics:
        sampled = sample_robot_eef_actions(head, conditions, embodiment_ids=embodiment_ids, action_mask=mask)
        horizon = min(replan_horizon or head.action_horizon, head.action_horizon)
        selected = mask[:, :horizon].float()
        error = sampled[:, :horizon].float() - safe[:, :horizon].float()
        count = selected.sum().clamp_min(1)
        metrics.update(action_chunk_mse=(error.square() * selected).sum() / count,
                       action_chunk_l1=(error.abs() * selected).sum() / count,
                       action_chunk_metric_weight=selected.sum().detach())
    return metrics


@torch.no_grad()
def sample_robot_eef_actions(head, conditions, *, embodiment_ids, action_mask=None,
                             initial_noise=None, sampling_steps=None):
    """Sample normalized XYZ+rotation6D offsets with masked coordinates inert."""
    mask = robot_eef_action_mask(embodiment_ids, horizon=head.action_horizon, action_mask=action_mask)
    cond = robot_eef_conditions(conditions, embodiment_ids)
    shape = (conditions[0].shape[0], head.action_horizon, 15)
    if initial_noise is None:
        initial_noise = torch.randn(shape, device=conditions[0].device, dtype=conditions[0].dtype)
    elif not torch.is_tensor(initial_noise) or tuple(initial_noise.shape) != shape:
        raise ValueError("Robot EEF initial noise must match packed source horizon")
    state = initial_noise.to(device=conditions[0].device, dtype=conditions[0].dtype)
    _assert_tensor(state, torch.isfinite(state).all(), "Robot EEF initial noise must be finite")
    noise = state
    steps = head.flow_sampling_steps if sampling_steps is None else sampling_steps
    if type(steps) is not int or steps < 1:
        raise ValueError("Robot EEF sampling steps must be a positive integer")
    caches = [{} for _ in head.blocks]
    for index in range(steps):
        times = state.new_full((shape[0],), index / steps)
        velocity = head._vector_field(cond, state, times, static_kv_caches=caches)
        state = state + torch.where(mask, velocity, 0) / steps
        state = torch.where(mask, state, noise)
    return torch.where(mask, state, 0)
