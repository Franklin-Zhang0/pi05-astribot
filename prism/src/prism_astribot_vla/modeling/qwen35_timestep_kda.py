"""One simultaneous KDA memory transition per observed visual timestep.

This opt-in path keeps token-specific queries/keys/values. All keys at a frame
correct the same decayed previous state; their masked-mean correction is
published once, and all queries read that new state. Spatial ordering and the
number of identical copies of a token therefore do not change memory decay.
Instruction-prefix calls remain on the native tokenwise implementation.

The one-token delta rule follows FLA's MIT-licensed KDA reference semantics:
https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
The simultaneous masked-mean frame update and framewise convolution are UVA
extensions; no sequential FLA kernel is presented as a framewise kernel.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TimestepKDALayout:
    valid: torch.Tensor
    valid_cpu: torch.Tensor
    elapsed_ratios: torch.Tensor | None = None
    write_scaling: bool = False


_TIMESTEP_LAYOUT: ContextVar[TimestepKDALayout | None] = ContextVar(
    "prism_qwen35_timestep_kda_layout", default=None)


def get_qwen35_timestep_layout() -> TimestepKDALayout | None:
    """Return the visual frame layout without consulting mutable caches."""
    return _TIMESTEP_LAYOUT.get()


@contextmanager
def qwen35_timestep_scope(valid: torch.Tensor, *, valid_cpu: torch.Tensor | None = None,
        elapsed_ratios: torch.Tensor | None = None, write_scaling: bool = False
        ) -> Iterator[None]:
    """Scope complete [B,T,N] frames for once-per-frame KDA execution.

    Supply the runner's existing CPU validity metadata to avoid a device sync
    per layer. ``elapsed_ratios`` optionally scales activated log decay once
    per frame. Its default is one, independently of the spatial token count.
    This scope must not wrap instruction-prefix or partial-frame calls.
    """
    if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.ndim != 3 or min(valid.shape) < 1:
        raise ValueError("Timestep KDA requires nonempty boolean [B,T,N] validity")
    host = valid.detach().cpu() if valid_cpu is None else valid_cpu
    if (not torch.is_tensor(host) or host.device.type != "cpu" or host.dtype != torch.bool
            or tuple(host.shape) != tuple(valid.shape)):
        raise ValueError("Timestep KDA CPU validity must match [B,T,N]")
    if type(write_scaling) is not bool:
        raise ValueError("Timestep KDA write scaling must be boolean")
    if elapsed_ratios is not None:
        if (not torch.is_tensor(elapsed_ratios) or tuple(elapsed_ratios.shape) != tuple(valid.shape[:2])
                or not elapsed_ratios.is_floating_point()):
            raise ValueError("Timestep KDA elapsed ratios require floating [B,T]")
        host_ratios = elapsed_ratios.detach().cpu()
        if not bool(torch.isfinite(host_ratios).all()) or not bool((host_ratios > 0).all()):
            raise ValueError("Timestep KDA elapsed ratios must be finite and positive")
    token = _TIMESTEP_LAYOUT.set(TimestepKDALayout(valid, host, elapsed_ratios, write_scaling))
    try:
        yield
    finally:
        _TIMESTEP_LAYOUT.reset(token)


def timestep_kda_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        log_decay: torch.Tensor, beta: torch.Tensor, valid: torch.Tensor, *,
        initial_state: torch.Tensor | None = None, scale: float | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable simultaneous delta rule, using activated gates.

    Inputs are [B,T,N,H,K/V], decay [B,T,N,H,K], beta [B,T,N,H]
    and validity [B,T,N]. Query/key L2 normalization is performed internally.
    The correction operator is a convex mean of PSD rank-one key operators,
    with spectrum in [0,1] for native sigmoid beta in [0,1]. A missing frame
    publishes neither decay nor a write and has zero output.
    """
    if (q.ndim != 5 or v.ndim != 5 or min(q.shape) < 1
            or k.shape != q.shape or v.shape[:3] != q.shape[:3]):
        raise ValueError("Timestep KDA requires q/k/v [B,T,N,H,K/V]")
    if (log_decay.shape != (*q.shape[:3], v.shape[3], q.shape[-1])
            or beta.shape != (*q.shape[:3], v.shape[3]) or valid.shape != q.shape[:3]
            or valid.dtype != torch.bool):
        raise ValueError("Timestep KDA gate/validity shapes disagree")
    heads, key_dim, value_dim = v.shape[-2], q.shape[-1], v.shape[-1]
    if heads % q.shape[-2]:
        raise ValueError("KDA value heads must be divisible by query/key heads")
    if initial_state is not None and initial_state.shape != (q.shape[0], heads, key_dim, value_dim):
        raise ValueError("Timestep KDA initial state must match [B,HV,K,V]")
    repeat = heads // q.shape[-2]
    if repeat != 1:
        q, k = q.repeat_interleave(repeat, -2), k.repeat_interleave(repeat, -2)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    state = (torch.zeros(q.shape[0], heads, key_dim, value_dim, dtype=dtype, device=q.device)
        if initial_state is None else initial_state.to(dtype))
    scale = key_dim ** -0.5 if scale is None else scale
    outputs = []
    for frame in range(q.shape[1]):
        mask = valid[:, frame, :, None, None]
        query = torch.where(mask, q[:, frame], 0).to(dtype)
        key = torch.where(mask, k[:, frame], 0).to(dtype)
        value = torch.where(mask, v[:, frame], 0).to(dtype)
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        count = valid[:, frame].sum(-1).clamp_min(1).to(dtype)
        decay = torch.where(mask, log_decay[:, frame], 0).to(dtype).sum(1) / count[:, None, None]
        decayed = state * decay.exp().unsqueeze(-1)
        key, query, value = (x.transpose(1, 2) for x in (key, query, value))
        predicted = torch.matmul(key, decayed)
        strength = torch.where(valid[:, frame, :, None], beta[:, frame], 0).to(dtype).transpose(1, 2)
        residual = (value - predicted) * (strength / count[:, None, None]).unsqueeze(-1)
        state = decayed + torch.matmul(key.transpose(-1, -2), residual)
        outputs.append((torch.matmul(query, state) * scale).transpose(1, 2))
    return torch.stack(outputs, 1).to(v.dtype), state


# Explicit correctness entrypoint retained for component tests. Production uses
# the same GPU-compatible batched matmuls, with stateless activation replay.
timestep_kda_reference = timestep_kda_delta_rule


def timestep_kda_execution_settings(module=None) -> dict:
    """Resolve implementation-only switches without changing model contracts."""
    def flag(name, attribute):
        if module is not None and hasattr(module, attribute):
            value = getattr(module, attribute)
            if type(value) is not bool:
                raise ValueError(f"{attribute} must be boolean")
            return value
        value = os.environ.get(name, "0").strip().lower()
        if value not in {"0", "1", "false", "true"}:
            raise ValueError(f"{name} must be 0/1 or false/true")
        return value in {"1", "true"}
    optimized = flag("PRISM_TIMESTEP_KDA_OPTIMIZED", "_prism_timestep_kda_optimized")
    use_compile = flag("PRISM_TIMESTEP_KDA_COMPILE", "_prism_timestep_kda_compile")
    frames = getattr(module, "_prism_timestep_kda_prepare_frames", None) if module is not None else None
    if frames is None:
        frames = int(os.environ.get("PRISM_TIMESTEP_KDA_PREPARE_FRAMES", "4"))
    if type(frames) is not int or not 1 <= frames <= 32:
        raise ValueError("Timestep KDA preparation block must contain 1..32 frames")
    if use_compile and not optimized:
        raise ValueError("Compiled timestep KDA requires its optimized implementation")
    return dict(optimized=optimized, use_compile=use_compile, prepare_frames=frames)


def timestep_convolution(mixed: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor | None = None, initial_state: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise temporal convolution with one pooled history entry per frame.

    A token sees the last K-1 frame means plus its own current projection.
    The current frame mean is appended once after all token outputs are formed.
    One token per frame is the ordinary native causal convolution exactly.
    Empty frames leave history untouched; padded tokens never enter the mean.
    """
    if mixed.ndim != 4 or valid.shape != mixed.shape[:3]:
        raise ValueError("Timestep convolution requires [B,T,N,C] and [B,T,N]")
    channels = mixed.shape[-1]
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight.squeeze(1)
    if weight.ndim != 2 or weight.shape[0] != channels:
        raise ValueError("Timestep convolution weight requires [C,K]")
    kernel = weight.shape[-1]
    history = (mixed.new_zeros(mixed.shape[0], channels, kernel) if initial_state is None
        else initial_state.to(device=mixed.device, dtype=mixed.dtype))
    if history.shape != (mixed.shape[0], channels, kernel):
        raise ValueError("Timestep convolution cache has incompatible shape")
    dtype = torch.float64 if mixed.dtype == torch.float64 else torch.float32
    weights = weight.to(dtype)
    outputs = []
    for frame in range(mixed.shape[1]):
        mask = valid[:, frame, :, None]
        current = torch.where(mask, mixed[:, frame], 0)
        count = valid[:, frame].sum(-1).clamp_min(1).to(dtype)
        pooled = current.to(dtype).sum(1) / count[:, None]
        past = (history[..., -(kernel - 1):].to(dtype) * weights[:, :-1]).sum(-1) if kernel > 1 else 0.
        convolved = current.to(dtype) * weights[:, -1] + (past[:, None] if torch.is_tensor(past) else past)
        if bias is not None:
            convolved = convolved + bias.to(dtype)
        outputs.append(torch.where(mask, F.silu(convolved).to(mixed.dtype), 0))
        proposed = torch.cat((history[..., 1:], pooled.to(mixed.dtype).unsqueeze(-1)), -1)
        history = torch.where(valid[:, frame].any(-1)[:, None, None], proposed, history)
    return torch.stack(outputs, 1), history


def _activated_gates(raw_gate, beta_logits, a_log, dt_bias, elapsed_ratios,
        *, value_heads, lower_bound, write_scaling):
    dtype = torch.float64 if raw_gate.dtype == torch.float64 else torch.float32
    adjusted = raw_gate.to(dtype) + dt_bias.to(dtype).reshape(value_heads, -1)
    rate = a_log.to(dtype).exp().reshape(value_heads, 1)
    decay = (-rate * F.softplus(adjusted) if lower_bound is None
        else float(lower_bound) * torch.sigmoid(rate * adjusted))
    beta = beta_logits.to(dtype).sigmoid()
    if elapsed_ratios is not None:
        ratio = elapsed_ratios.to(dtype=dtype, device=raw_gate.device)
        decay = decay * ratio[:, :, None, None, None]
        if write_scaling:
            # log(1-sigmoid(x))=-softplus(x) remains stable even when beta
            # rounds to one, and avoids the infinite derivative of log1p(-1).
            beta = -torch.expm1(-F.softplus(beta_logits.to(dtype)) * ratio[:, :, None, None])
    return decay, beta


@lru_cache(maxsize=1)
def _compiled_gates():
    from prism_astribot_vla.modeling.qwen35_timestep_kda_kernels import compile_frame_function
    return compile_frame_function(_activated_gates)


def _projected_core(mixed, raw_gate, beta_logits, conv_weight, conv_bias, a_log,
        dt_bias, valid, initial_conv, initial_recurrent, elapsed_ratios,
        *, key_dim, value_dim, key_heads, value_heads, lower_bound, write_scaling,
        optimized=False, use_compile=False, prepare_frames=4):
    if optimized:
        from prism_astribot_vla.modeling.qwen35_timestep_kda_conv import timestep_convolution_vectorized
        convolved, conv = timestep_convolution_vectorized(mixed, valid, conv_weight,
            conv_bias, initial_conv, block_frames=prepare_frames, compile_pointwise=use_compile)
    else:
        convolved, conv = timestep_convolution(mixed, valid, conv_weight, conv_bias, initial_conv)
    query, key, value = torch.split(convolved, (key_dim, key_dim, value_dim), -1)
    query = query.reshape(*query.shape[:3], key_heads, key_dim // key_heads)
    key = key.reshape_as(query)
    value = value.reshape(*value.shape[:3], value_heads, value_dim // value_heads)
    gate_function = _compiled_gates() if optimized and use_compile else _activated_gates
    decay, beta = gate_function(raw_gate, beta_logits, a_log, dt_bias, elapsed_ratios,
        value_heads=value_heads, lower_bound=lower_bound, write_scaling=write_scaling)
    if optimized:
        from prism_astribot_vla.modeling.qwen35_timestep_kda_kernels import optimized_timestep_kda_delta_rule
        output, recurrent = optimized_timestep_kda_delta_rule(query, key, value, decay, beta, valid,
            initial_state=initial_recurrent, use_compile=use_compile, prepare_frames=prepare_frames)
    else:
        output, recurrent = timestep_kda_delta_rule(query, key, value, decay, beta, valid,
            initial_state=initial_recurrent)
    return output, conv, recurrent


def timestep_kda_forward(module, hidden_states, cache_params, layout: TimestepKDALayout):
    """Run projected frame updates under the existing linear module boundary.

    Cache mutation is outside the optional stateless checkpoint closure. Replay
    receives immutable tensors and neither re-enters child modules nor consults
    a dynamic token scope, so it cannot advance recurrent/conv state twice.
    """
    if not bool(getattr(module, "_prism_kda_enabled", False)):
        raise ValueError("Once-per-timestep updates require the KDA variant")
    execution = timestep_kda_execution_settings(module)
    batch, frames, tokens = layout.valid.shape
    if tuple(hidden_states.shape[:2]) != (batch, frames * tokens):
        raise ValueError("Timestep KDA scope must cover complete current visual frames")
    valid = layout.valid.to(hidden_states.device)
    indices = (None if bool(layout.valid_cpu.all()) else
        layout.valid_cpu.flatten().nonzero().flatten().to(hidden_states.device))
    clean = (hidden_states if indices is None else
        hidden_states.flatten(0, 1).index_select(0, indices))
    def unpack(value):
        if indices is None:
            return value
        return value.new_zeros(batch * frames * tokens, value.shape[-1]).index_copy_(0, indices, value)
    # Retain the existing valid-row projection optimization: camera padding is
    # not sent through any learned linear map, including its gradient path.
    mixed = module.in_proj_qkv(clean)
    z = module.in_proj_z(clean)
    b = module.in_proj_b(clean)
    a = module.in_proj_a(clean)
    gate_delta = module.kda_gate_up(module.kda_gate_down(clean))
    raw_gate = (a.unsqueeze(-1) + gate_delta.unflatten(-1,(module.num_v_heads, module.head_k_dim))).flatten(-2)

    def expand_inputs(mixed, raw_gate, b):
        return (unpack(mixed).reshape(batch, frames, tokens, -1),
                unpack(raw_gate).reshape(batch, frames, tokens, module.num_v_heads, module.head_k_dim),
                unpack(b).reshape(batch, frames, tokens, module.num_v_heads))

    # Delay scattering until replay so missing camera slots consume no saved
    # projection storage during the rest of the backbone and action forwards.
    packed_replay = indices is not None and os.getenv("PRISM_KDA_PACKED_REPLAY") == "1"
    if not packed_replay:
        mixed, raw_gate, b = expand_inputs(mixed, raw_gate, b)
    bias = module.dt_bias.repeat_interleave(module.head_k_dim) + module.kda_dt_bias_delta
    initial_conv = initial_recurrent = None
    if cache_params is not None and cache_params.has_previous_state:
        initial_conv = cache_params.conv_states[module.layer_idx]
        initial_recurrent = cache_params.recurrent_states[module.layer_idx]
        if initial_conv is not None:
            initial_conv = initial_conv.detach().clone()
        if initial_recurrent is not None:
            initial_recurrent = initial_recurrent.detach().clone()
    def run(mixed, raw_gate, b, weight, conv_bias, a_log, bias, valid, conv, recurrent, ratios):
        if packed_replay:
            mixed, raw_gate, b = expand_inputs(mixed, raw_gate, b)
        return _projected_core(mixed, raw_gate, b, weight, conv_bias, a_log, bias,
            valid, conv, recurrent, ratios, key_dim=module.key_dim,
            value_dim=module.value_dim, key_heads=module.num_k_heads,
            value_heads=module.num_v_heads,
            lower_bound=module._prism_kda_lower_bound if module._prism_kda_safe_gate else None,
            write_scaling=layout.write_scaling, **execution)
    inputs = (mixed, raw_gate, b, module.conv1d.weight, module.conv1d.bias,
        module.A_log, bias, valid, initial_conv, initial_recurrent, layout.elapsed_ratios)
    replay = bool(getattr(module, "_prism_timestep_kda_checkpoint", True))
    first_layers = os.environ.get("PRISM_KDA_CHECKPOINT_FIRST_LAYERS")
    if first_layers is not None:
        limit = int(first_layers)
        if limit < 0:
            raise ValueError("PRISM_KDA_CHECKPOINT_FIRST_LAYERS must be nonnegative")
        replay = replay and module.layer_idx < limit
    if module.training and torch.is_grad_enabled() and replay:
        core, conv, recurrent = checkpoint(run, *inputs, use_reentrant=False,
            preserve_rng_state=False, early_stop=False)
    else:
        core, conv, recurrent = run(*inputs)
    if cache_params is not None:
        cache_params.conv_states[module.layer_idx] = conv.detach().clone()
        cache_params.recurrent_states[module.layer_idx] = recurrent.detach().clone()
    real_core = core.reshape(batch * frames * tokens, -1)
    if indices is not None:
        real_core = real_core.index_select(0, indices)
    normalized = module.norm(real_core.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
    output = unpack(module.out_proj(normalized.reshape(-1, module.value_dim)))
    return output.reshape(batch, frames * tokens, -1)
