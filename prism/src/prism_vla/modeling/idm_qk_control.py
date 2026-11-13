"""Parameter-free Q/K activation controls for existing packed IDM attention.

The logit-rescaling equation is independently implemented from the MuonClip
description at https://www.kimi.com/en/blog/kimi-k2. Unlike MuonClip's
post-optimizer projection-weight scaling, this acts on every forward's projected
activations, including bias and changing conditioning, without mutating optimizer
masters or moments. The QK-normalization alternative is inspired by
https://arxiv.org/abs/2010.04245; its temperature is fixed, rather than learned,
to retain legacy checkpoint parameter names, shapes and optimizer ownership.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from prism_vla.modeling.config import validate_idm_recovery_controls


class ControlledIDMSelfAttention(nn.MultiheadAttention):
    """Packed self-attention with opt-in per-sample, per-head Q/K control.

    The inherited packed parameters and their registration order are unchanged.
    Statistics/scales remain detached device tensors; there are no CPU reads,
    synchronization calls, distributed collectives or persistent extra buffers.
    Enabled controls intentionally fail closed for unsupported attention forms.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qk_control_mode: str = "none",
        qk_logit_limit: float = 100.0,
        qk_norm_temperature: float = 8.0,
    ) -> None:
        controls = validate_idm_recovery_controls(
            idm_qk_control_mode=qk_control_mode,
            idm_qk_logit_limit=qk_logit_limit,
            idm_qk_norm_temperature=qk_norm_temperature,
        )
        super().__init__(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.qk_control_mode = controls["idm_qk_control_mode"]
        self.qk_logit_limit = controls["idm_qk_logit_limit"]
        self.qk_norm_temperature = controls["idm_qk_norm_temperature"]
        self.last_qk_control: dict[str, object] = {}

    def _control_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply control after packed projection, with FP32 statistics/scales."""
        dtype = q.dtype
        eps = torch.finfo(dtype).eps
        # FP32 statistics must not be silently lowered by surrounding BF16 autocast.
        with torch.autocast(device_type=q.device.type, enabled=False):
            q32, k32 = q.float(), k.float()
            if self.qk_control_mode == "qk_norm":
                # SDPA divides by sqrt(d), so multiplying each unit vector by
                # sqrt(temperature * sqrt(d)) gives temperature * cosine(q,k).
                effective_temperature = self.qk_norm_temperature * (1.0 - 4.0 * eps)
                magnitude = math.sqrt(effective_temperature * math.sqrt(self.head_dim))
                q_scale = magnitude / torch.linalg.vector_norm(q32, dim=-1, keepdim=True).clamp_min(1e-12)
                k_scale = magnitude / torch.linalg.vector_norm(k32, dim=-1, keepdim=True).clamp_min(1e-12)
                self.last_qk_control = {
                    "mode": "qk_norm",
                    "q_scale": q_scale.detach(),
                    "k_scale": k_scale.detach(),
                    "norm_temperature": self.qk_norm_temperature,
                    "roundoff_margin": 4.0 * eps,
                }
            elif self.qk_control_mode == "logit_rescale":
                with torch.no_grad():
                    logits = torch.matmul(q32.detach(), k32.detach().transpose(-2, -1)) / math.sqrt(self.head_dim)
                    observed = logits.abs().amax(dim=(-2, -1), keepdim=True)
                    del logits
                    # BF16 rounding can alter cancellations: a relative margin
                    # on the observed dot product alone is not a universal bound.
                    # Cauchy bounds the absolute dot-product rounding error.
                    norm_product = (
                        torch.linalg.vector_norm(q32.detach(), dim=-1).amax(-1, keepdim=True)
                        * torch.linalg.vector_norm(k32.detach(), dim=-1).amax(-1, keepdim=True)
                    ).unsqueeze(-1) / math.sqrt(self.head_dim)
                    target = self.qk_logit_limit * (1.0 - 4.0 * eps)
                    denominator = observed + (2.0 * eps + eps * eps) * norm_product
                    eta = (target / denominator.clamp_min(1e-12)).clamp(max=1.0)
                    eta = torch.where(observed <= target, torch.ones_like(eta), eta)
                    q_scale = eta.sqrt()
                    k_scale = q_scale
                self.last_qk_control = {
                    "mode": "logit_rescale",
                    "q_scale": q_scale.detach(),
                    "k_scale": k_scale.detach(),
                    "pre_logit_absmax": observed.squeeze(-1).squeeze(-1),
                    "logit_limit": self.qk_logit_limit,
                    "roundoff_margin": 4.0 * eps,
                    "roundoff_norm_product": norm_product.squeeze(-1).squeeze(-1),
                    "roundoff_guard_coefficient": 2.0 * eps + eps * eps,
                    "target_logit_limit": target,
                }
            else:
                return q, k
            return (q32 * q_scale).to(dtype=dtype), (k32 * k_scale).to(dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if self.qk_control_mode == "none":
            return super().forward(
                query, key, value, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                average_attn_weights=average_attn_weights, is_causal=is_causal,
            )
        if (
            query.ndim != 3 or query.is_nested or key.is_nested or value.is_nested
            or query.shape != key.shape or query.shape != value.shape
            or key_padding_mask is not None or attn_mask is not None or is_causal
            or need_weights or not self.batch_first or not self._qkv_same_embed_dim
            or self.bias_k is not None or self.bias_v is not None or self.add_zero_attn
            or self.dropout != 0.0
        ):
            raise ValueError(
                "Enabled IDM Q/K controls require unmasked, noncausal packed "
                "batch-first aligned attention with need_weights=False and dropout=0."
            )
        batch, length, _ = query.shape
        if query is key and key is value:
            projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = projected.reshape(batch, length, 3, self.num_heads, self.head_dim).unbind(2)
            projection_layout = "packed"
        else:
            # FSDP's input precision boundary may independently cast the three
            # shared inputs. Preserve their semantics without equality reads or
            # GPU synchronization; parameter slices retain packed ownership.
            weights = self.in_proj_weight.chunk(3, dim=0)
            biases = (None, None, None) if self.in_proj_bias is None else self.in_proj_bias.chunk(3)
            q, k, v = (
                F.linear(tensor, weight, bias).reshape(batch, length, self.num_heads, self.head_dim)
                for tensor, weight, bias in zip((query, key, value), weights, biases)
            )
            projection_layout = "separate"
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        projection_dtype = str(q.dtype)
        q, k = self._control_qk(q, k)
        self.last_qk_control.update(
            projection_layout=projection_layout,
            projection_dtype=projection_dtype,
            input_dtype=str(query.dtype),
            statistics_dtype="torch.float32",
        )
        # Keep BF16 compute and the fused SDPA path for the attention itself.
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().reshape(batch, length, self.embed_dim)
        # Match native MHA's functional out_proj consumption: its mixed-precision
        # parameter boundary is the parent MHA, not the child Linear.forward.
        return F.linear(output, self.out_proj.weight, self.out_proj.bias), None
