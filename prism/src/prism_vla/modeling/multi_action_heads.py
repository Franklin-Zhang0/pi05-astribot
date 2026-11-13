"""Joint action objectives with one shared, differentiable condition list."""
from __future__ import annotations

from collections.abc import Mapping
import inspect

import torch

from prism_vla.transition_rows import select_transition_rows
from torch import nn

from prism_vla.modeling.action_head_config import normalize_action_heads
from prism_vla.modeling.idm_backends import build_idm_backend


class MultiActionHeadIDM(nn.Module):
    """Own independent decoders under the existing ``idm`` optimizer boundary.

    The primary head retains the legacy inference/metric interface. Named
    inference returns each head's own representation; outputs are never blended
    across coordinate systems. Every decoder receives the identical tensors.
    """

    def __init__(self, specs, *, primary_head: str = "", native_kv_config=None,
                 auxiliary_sampled_metrics: bool = True,
                 pi05_block_gradient_checkpointing: bool = False, **backend_kwargs):
        super().__init__()
        self.specs = normalize_action_heads(specs)
        if not self.specs:
            raise ValueError("MultiActionHeadIDM requires at least one head")
        self.primary_head = primary_head or self.specs[0]["name"]
        if type(auxiliary_sampled_metrics) is not bool:
            raise ValueError("auxiliary_sampled_metrics must be a bool")
        self.auxiliary_sampled_metrics = auxiliary_sampled_metrics
        if self.primary_head not in {spec["name"] for spec in self.specs}:
            raise ValueError(f"Unknown primary action head: {self.primary_head}")
        self.action_dim = backend_kwargs["action_dim"]
        self.action_horizon = backend_kwargs["action_horizon"]
        self.heads = nn.ModuleDict()
        valid_options = set(inspect.signature(build_idm_backend).parameters) - {"idm_type"}
        for spec in self.specs:
            options = dict(backend_kwargs)
            if spec["type"] == "pi05":
                from prism_vla.modeling.pi05_action_expert import Pi05ActionExpert
                if native_kv_config is None:
                    raise ValueError("pi05 construction requires actual backbone K/V dimensions")
                overrides = spec["overrides"]
                allowed = {"d_model", "nhead", "ff_mult", "flow_output_projection", "flow_sampling_steps"}
                if set(overrides) - allowed:
                    raise ValueError("pi05 overrides may configure width/heads/MLP/output/sampling; depth covers all native K/V layers")
                options.update(overrides)
                self.heads[spec["name"]] = Pi05ActionExpert(**native_kv_config,
                    pi05_block_gradient_checkpointing=pi05_block_gradient_checkpointing, **options)
                continue
            if spec["type"] != "fast_ar":
                options["fast_ffn_gradient_checkpointing"] = False
                options["fast_layer_gradient_checkpointing"] = False
                options["fast_broadcast_attention"] = False
            if spec["type"] != "flow_matching":
                # DiT-specific inherited settings cannot be applied to other
                # architectures. Explicit incompatible overrides still fail.
                for key in options:
                    if key.endswith("_limit") and key.startswith("idm_") and key != "idm_qk_logit_limit":
                        options[key] = None
                options["idm_qk_control_mode"] = "none"
                options["flow_output_projection"] = "shared"
                options["flow_qk_norm"] = False
            if spec["type"] == "fast_ar":
                options["action_clip_abs_max"] = 1.0
            overrides = spec["overrides"]
            if set(overrides) - valid_options:
                raise ValueError(f"Head {spec['name']}: unknown backend overrides {sorted(set(overrides)-valid_options)}")
            options.update(overrides)
            # A global flow execution option applies only to compatible heads;
            # retain explicit incompatible overrides for factory validation.
            if "flow_cross_block_gradient_checkpointing" not in overrides and not (
                spec["type"] == "flow_matching"
                and str(options.get("flow_idm_arch", "mlp")).lower()
                in {"temporal_transformer", "transformer"}
                and str(options.get("flow_conditioning", "adaln_zero")).lower().replace("-", "_")
                in {"cross_attn", "cross_attention", "crossattn"}
            ):
                options["flow_cross_block_gradient_checkpointing"] = False
            self.heads[spec["name"]] = build_idm_backend(spec["type"], **options)
        self.idm_type = self.heads[self.primary_head].idm_type

    @property
    def requires_raw_action_token(self) -> bool:
        return any(getattr(head, "requires_raw_action_token", False)
                   for head in self.heads.values())

    @property
    def action_clip_abs_max(self):
        return getattr(self.heads[self.primary_head], "action_clip_abs_max", None)

    @property
    def sampled_action_clip_abs_max(self):
        return getattr(self.heads[self.primary_head], "sampled_action_clip_abs_max", None)

    def consume_training_rng(self, *, action_head_targets, dtype, **kwargs):
        """Consume flow bridge draws in the same deterministic head order."""
        for spec in self.specs:
            name = spec["name"]
            self.heads[name].consume_training_rng(
                action_chunks=action_head_targets[name].get("action_chunks"), dtype=dtype)

    def training_step(self, cond_list, *, action_head_targets=None,
                      compute_action_metrics=False, replan_horizon=None, **legacy):
        if not isinstance(action_head_targets, Mapping) or set(action_head_targets) != set(self.heads):
            raise ValueError("Multi-head training requires complete named action_head_targets")
        if legacy.get("ego_weak_supervision") is not None:
            raise ValueError("Multi-head training requires explicit pose labels; weak supervision is not supported")
        if self.requires_raw_action_token and legacy.get("proprio_token") is not None:
            raise ValueError("act_patch requires condition-placed proprio fused into ACT")
        total, metrics, primary_metrics = None, {}, {}
        for spec in self.specs:
            name, backend = spec["name"], self.heads[spec["name"]]
            target = action_head_targets[name]
            chunks = target.get("action_chunks")
            if not torch.is_tensor(chunks) or chunks.shape != (cond_list[0].shape[0], self.action_horizon, self.action_dim):
                raise ValueError(f"Head {name}: action_chunks must match shared batch/horizon/dimension")
            kwargs = {key: target.get(key) for key in (
                "fast_tokens", "action_chunks", "action_chunk_valid_steps", "action_chunk_loss_mask")}
            if backend.idm_type == "fast_ar":
                from prism_vla.data.fast_targets import FAST_TOKEN_VALIDATION_KEY
                kwargs[FAST_TOKEN_VALIDATION_KEY] = target.get(FAST_TOKEN_VALIDATION_KEY)
                kwargs["action_semantic_mask"] = target.get("action_semantic_mask")
            if "proprio_token" in legacy and backend.idm_type == "flow_matching":
                kwargs["proprio_token"] = legacy["proprio_token"]
            if backend.idm_type == "flow_matching":
                kwargs.update({key: target[key] for key in ("flow_times", "flow_noise")
                               if key in target})
            output = backend.training_step(cond_list, **kwargs,
                compute_action_metrics=(compute_action_metrics
                    and (self.auxiliary_sampled_metrics or name == self.primary_head)),
                replan_horizon=replan_horizon)
            weighted = spec["loss_weight"] * output["loss"]
            total = weighted if total is None else total + weighted
            for key, value in output.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    metrics[f"action_heads/{name}/{key}"] = value.detach()
            if name == self.primary_head:
                primary_metrics = output
        # A mixture of MSE/L1/token objectives has no common scalar/token
        # denominator. Weight the joint diagnostic by shared transitions;
        # preserve each head's own denominator in its named metrics.
        weight = (primary_metrics["action_loss_weight"] if self.requires_raw_action_token
                  else total.new_full((), cond_list[0].shape[0]).detach())
        return {**primary_metrics, **metrics, "loss": total, "action_loss_weight": weight}

    def tokenizer_contracts(self):
        """Bind decoder token vocabularies and physical layouts to checkpoints."""
        return {name: backend.fast_tokenizer.provenance
                for name, backend in self.heads.items() if backend.idm_type == "fast_ar"}

    @torch.no_grad()
    def generate_actions(self, cond_list, *, head_name=None, **kwargs):
        name = head_name or self.primary_head
        if name not in self.heads:
            raise ValueError(f"Unknown action head: {name}")
        backend = self.heads[name]
        if getattr(backend, "requires_raw_action_token", False) and kwargs.get("proprio_token") is not None:
            raise ValueError("act_patch requires condition-placed proprio fused into ACT")
        if backend.idm_type != "flow_matching":
            kwargs = {key: value for key, value in kwargs.items() if key not in {
                "initial_noise", "guidance_scale", "proprio_token", "sampling_steps",
                "uncond_cond_list", "uncond_proprio_token", "cache_static_kv"}}
        return backend.generate_actions(cond_list, **kwargs)

    @torch.no_grad()
    def generate_all_actions(self, cond_list):
        return {name: self.generate_actions(cond_list, head_name=name) for name in self.heads}

    @torch.no_grad()
    def generate(self, cond_list, **kwargs):
        return self.heads[self.primary_head].generate(cond_list, **kwargs)


def select_action_head_targets(targets, valid_mask, *, timestep=None, row_indices=None):
    """Select exactly the transitions whose shared VLM conditions are used."""
    if targets is None:
        return None
    if timestep is not None and row_indices is not None:
        raise ValueError("precomputed row indices require full-chunk targets")
    result = {}
    for name, payload in targets.items():
        selected = {}
        from prism_vla.data.fast_targets import FAST_TOKEN_VALIDATION_KEY
        if FAST_TOKEN_VALIDATION_KEY in payload:
            selected[FAST_TOKEN_VALIDATION_KEY] = payload[FAST_TOKEN_VALIDATION_KEY]
        for key in ("action_chunks", "action_chunk_valid_steps", "action_chunk_loss_mask", "fast_tokens", "action_semantic_mask"):
            value = payload.get(key)
            if value is not None:
                if not torch.is_tensor(value) or value.ndim < 2:
                    raise ValueError(f"Head {name}: {key} requires batch and transition dimensions")
                if timestep is None:
                    value = value.reshape(-1, *value.shape[2:])
                else:
                    value = value[:, timestep]
                if value.shape[0] != valid_mask.numel():
                    raise ValueError(f"Head {name}: target transitions differ from shared conditions")
                selected[key] = select_transition_rows(value, valid_mask, row_indices)
        result[name] = selected
    return result


def blended_head_metrics(gt, pred, gamma):
    """Return detached diagnostics without keeping previous graphs on modules."""
    gt, pred = gt or {}, pred or {}
    result = {}
    for key in sorted(set(gt) | set(pred)):
        if key.startswith("action_heads/"):
            if gamma <= 0:
                result[key] = gt[key]
            elif gamma >= 1:
                result[key] = pred[key]
            elif key.endswith("action_loss_weight"):
                result[key] = gt[key]
            else:
                result[key] = (1-gamma)*gt[key] + gamma*pred[key]
    return result
