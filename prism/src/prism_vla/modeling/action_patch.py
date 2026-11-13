"""Fixed action targets and decoding for one raw ACT output per action step.

The repetition/zero-fill interface follows PatchWAM, Sec. 3.2:
https://arxiv.org/html/2609.25961v1#S3.SS2 . Its Apache-2.0 reference codec is
https://github.com/TianhengW/PatchWAM/blob/main/src/patchwam/models/codec.py .
This independent implementation uses the same per-step token layout and
directly regresses clean actions, rather than joint flow-matching velocities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from prism_vla.modeling.action_layout import robot_action_sampling_mask
from prism_vla.modeling.condition_mask import condition_mask
from prism_vla.modeling.losses import masked_regression_loss, normalize_regression_loss_type


@dataclass(frozen=True)
class RepeatedActionCodec:
    """Map normalized action steps ``[..., A]`` to tokens without parameters."""

    action_dim: int
    token_dim: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (
            self.action_dim, self.token_dim,
        )):
            raise ValueError("Action-patch action dimension and token width must be positive integers")
        if self.action_dim > self.token_dim:
            raise ValueError(
                "Each ACT token requires action_dim <= backbone width; "
                f"got {self.action_dim} > {self.token_dim}"
            )

    @property
    def copies(self) -> int:
        return self.token_dim // self.action_dim

    @property
    def occupied(self) -> int:
        return self.copies * self.action_dim

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Repeat each component of each action step, then append exact zeros."""
        if actions.shape[-1] != self.action_dim:
            raise ValueError("Action-patch target must end in the configured action dimension")
        values = actions.repeat_interleave(self.copies, dim=-1)
        return torch.nn.functional.pad(values, (0, self.token_dim - self.occupied))

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Average each repeated group in FP32; ignore fill coordinates."""
        if tokens.shape[-1] != self.token_dim:
            raise ValueError("Action-patch output width differs from the backbone token width")
        groups = tokens[..., :self.occupied].float().reshape(
            *tokens.shape[:-1], self.action_dim, self.copies,
        )
        return groups.mean(dim=-1)

    def supervision_mask(self, action_mask: torch.Tensor) -> torch.Tensor:
        """Supervise fill coordinates iff that action step has a real label."""
        if action_mask.shape[-1] != self.action_dim:
            raise ValueError("Action-patch mask must end in the configured action dimension")
        repeated = action_mask.repeat_interleave(self.copies, dim=-1)
        fill = (action_mask > 0).any(dim=-1, keepdim=True).to(action_mask.dtype)
        fill = fill.expand(*action_mask.shape[:-1], self.token_dim - self.occupied)
        return torch.cat((repeated, fill), dim=-1)


class ActionPatchIDM(nn.Module):
    """Regress one raw ACT token for each physical action step.

    This module deliberately owns no trainable parameters or projections.
    Dataset action normalization is external, as for the existing decoders.
    Serialized body-part rows are grouped before encoding and restored after
    decoding, preserving the external ``[N, action_horizon, action_dim]`` API.
    """

    idm_type = "act_patch"
    requires_raw_action_token = True

    def __init__(
        self, *, cond_dim: int, action_dim: int, action_horizon: int,
        n_cond_tokens: int = 1, action_tokens_per_step: int = 1,
        loss_type: str = "mse", huber_delta: float = 0.1,
        action_clip_abs_max: float | None = None,
        sampled_action_clip_abs_max: float | None = None,
        robot_action_layout: str = "",
    ) -> None:
        super().__init__()
        if n_cond_tokens != 1:
            raise ValueError("act_patch requires exactly one raw ACT condition block")
        if type(action_horizon) is not int or action_horizon < 1:
            raise ValueError("Action-patch horizon must be a positive integer")
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError("Action-patch action dimension must be a positive integer")
        if type(action_tokens_per_step) is not int or action_tokens_per_step < 1:
            raise ValueError("action_tokens_per_step must be a positive integer")
        if action_horizon % action_tokens_per_step:
            raise ValueError("Action-patch action_horizon must be divisible by action_tokens_per_step")
        self.codec = RepeatedActionCodec(action_tokens_per_step * action_dim, cond_dim)
        self.cond_dim, self.n_cond_tokens = cond_dim, n_cond_tokens
        self.action_dim, self.action_horizon = action_dim, action_horizon
        self.action_tokens_per_step = action_tokens_per_step
        self.action_step_horizon = action_horizon // action_tokens_per_step
        self.loss_type = normalize_regression_loss_type(loss_type)
        self.huber_delta = float(huber_delta)
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        for name, value in (("action_clip_abs_max", action_clip_abs_max),
                            ("sampled_action_clip_abs_max", sampled_action_clip_abs_max)):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{name} must be finite and positive or null")
            setattr(self, name, None if value is None else float(value))
        sampling_mask = robot_action_sampling_mask(
            robot_action_layout, action_horizon=action_horizon, action_dim=action_dim,
        )
        self.register_buffer("sampling_active_mask", sampling_mask, persistent=False)

    def consume_training_rng(self, **kwargs) -> None:
        """A deterministic ACT readout draws no training noise."""
        return None

    def forward(self, cond_list, *, return_action_token: bool = False) -> torch.Tensor:
        if len(cond_list) != 1 or not torch.is_tensor(cond_list[0]):
            raise ValueError("act_patch requires one raw ACT block [N, physical_horizon, backbone_width]")
        token = cond_list[0]
        if token.ndim != 3 or tuple(token.shape[1:]) != (self.action_step_horizon, self.cond_dim):
            raise ValueError("act_patch ACT condition must have shape [N, physical_horizon, backbone_width]")
        if condition_mask(cond_list) is not None:
            raise ValueError("The ACT condition must be valid for every action row")
        return token if return_action_token else self._decode_actions(token)

    def _decode_actions(self, token: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(token).reshape(token.shape[0], self.action_horizon, self.action_dim)

    def _sampled_actions(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.sampling_active_mask is not None:
            prediction = prediction.masked_fill(~self.sampling_active_mask.to(prediction.device), 0.0)
        bound = self.sampled_action_clip_abs_max
        if bound is None:
            bound = self.action_clip_abs_max
        return prediction if bound is None else prediction.clamp(-bound, bound)

    def training_step(
        self, cond_list, *, action_chunks: torch.Tensor | None,
        fast_tokens: torch.Tensor | None = None,
        action_chunk_valid_steps: torch.Tensor | None = None,
        action_chunk_loss_mask: torch.Tensor | None = None,
        replan_horizon: int | None = None, compute_action_metrics: bool = False,
        ego_weak_supervision: torch.Tensor | None = None,
        proprio_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ego_weak_supervision is not None:
            raise ValueError("act_patch requires explicit action labels; weak supervision is not supported")
        if proprio_token is not None:
            raise ValueError("act_patch requires condition-placed proprio fused into ACT")
        token = self(cond_list, return_action_token=True).float()
        expected = (token.shape[0], self.action_horizon, self.action_dim)
        if not torch.is_tensor(action_chunks) or tuple(action_chunks.shape) != expected:
            raise ValueError("act_patch action_chunks must match the configured batch/horizon/dimension")
        target = action_chunks.to(device=token.device, dtype=torch.float32)
        # Import lazily to keep the factory and this parameter-free backend
        # independent while sharing the established horizon/mask convention.
        from prism_vla.modeling.idm_backends import _combine_action_chunk_masks
        mask = _combine_action_chunk_masks(
            action_chunk_valid_steps, action_chunk_loss_mask,
            horizon=self.action_horizon, action_dim=self.action_dim,
            device=token.device, dtype=torch.float32,
        )
        if mask is None:
            mask = torch.ones_like(target)
        if tuple(mask.shape) != expected:
            raise ValueError("act_patch supervision mask differs from the action chunk shape")
        valid_mask = torch.isfinite(mask) & (mask >= 0)
        if mask.device.type == "cpu":
            if not bool(valid_mask.all()):
                raise ValueError("act_patch supervision weights must be finite and nonnegative")
        else:
            torch._assert_async(valid_mask.all(), "act_patch supervision weights must be finite and nonnegative")
        target = torch.where(mask > 0, target, 0.0)
        physical_shape = (token.shape[0], self.action_step_horizon, self.codec.action_dim)
        encoded_target = self.codec.encode(target.reshape(physical_shape))
        encoded_mask = self.codec.supervision_mask(mask.reshape(physical_shape))
        # Mask before subtraction, including completely unlabeled rows, so
        # invalid NaN targets or predictions cannot contaminate the objective.
        safe_token = torch.where(encoded_mask > 0, token, 0.0)
        loss = masked_regression_loss(safe_token, encoded_target, mask=encoded_mask,
            loss_type=self.loss_type, huber_delta=self.huber_delta)
        metrics = {"loss": loss, "action_loss_weight": encoded_mask.sum().detach(),
                   "action_acc": loss.new_zeros(())}
        if compute_action_metrics:
            horizon = min(self.action_horizon, replan_horizon or self.action_horizon)
            prediction = self._sampled_actions(self._decode_actions(safe_token))[:, :horizon]
            for kind in ("l1", "mse"):
                metrics[f"action_chunk_{kind}"] = masked_regression_loss(
                    prediction, target[:, :horizon], mask=mask[:, :horizon], loss_type=kind,
                )
        return metrics

    @torch.no_grad()
    def generate_actions(self, cond_list) -> torch.Tensor:
        """Decode normalized commands; the existing pipeline denormalizes them."""
        return self._sampled_actions(self(cond_list))
