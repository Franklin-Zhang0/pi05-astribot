"""Runtime patch for Qwen3.5 DeltaNet cached chunk forwards."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


_RESEARCHBOOST_DISPATCH_KEYS = (
    "packed_row_split_calls", "ordinary_kda_varlen_calls",
    "observer_kda_varlen_calls", "observer_uniform_calls",
)
_RESEARCHBOOST_DISPATCH_COUNTS = dict.fromkeys(_RESEARCHBOOST_DISPATCH_KEYS, 0)


def _record_researchboost_dispatch(name: str) -> None:
    if os.environ.get("RB_OPT_VERIFY_DISPATCH", "0") != "1":
        return
    _RESEARCHBOOST_DISPATCH_COUNTS[name] += 1


def researchboost_dispatch_counts(*, reset: bool = False) -> dict[str, int]:
    """Read opt-in CPU counters without inspecting or synchronizing GPU tensors.

    Set ``RB_OPT_VERIFY_DISPATCH=1`` before a qualification forward. Counts
    aggregate all linear-attention calls in this rank, outside checkpoint state.
    ``reset=True`` returns the old totals and clears them for the next window.
    """
    result = dict(_RESEARCHBOOST_DISPATCH_COUNTS)
    if reset:
        _RESEARCHBOOST_DISPATCH_COUNTS.update(dict.fromkeys(_RESEARCHBOOST_DISPATCH_KEYS, 0))
    return result


@dataclass
class _ValidTokenLayout:
    shape: tuple[int, int]
    lengths: tuple[int, ...]
    flat_indices: torch.Tensor
    device_indices: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    varlen_cu_cpu: torch.Tensor | None = None
    varlen_cu_devices: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    stream: str = "unspecified"

    def cumulative_lengths_for(self, device: torch.device):
        """Describe only nonempty streams; empty streams keep their old state."""
        if self.varlen_cu_cpu is None:
            cumulative = [0]
            for length in self.lengths:
                if length:
                    cumulative.append(cumulative[-1] + length)
            self.varlen_cu_cpu = torch.tensor(cumulative, dtype=torch.long)
        if device not in self.varlen_cu_devices:
            self.varlen_cu_devices[device] = self.varlen_cu_cpu.to(device=device)
        return self.varlen_cu_devices[device], self.varlen_cu_cpu

    def indices_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or tuple(hidden_states.shape[:2]) != self.shape:
            raise ValueError("Scoped Qwen valid-token mask must match the current [B, T] query")
        device = hidden_states.device
        if device not in self.device_indices:
            self.device_indices[device] = self.flat_indices.to(device=device)
        return self.device_indices[device]


@dataclass
class _ObserverTokenLayout(_ValidTokenLayout):
    """Packed order places ACT after its timestep's persistent query tail."""

    observer_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.bool))
    packed_observers: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.bool))
    uniform_layout: bool = False
    device_row_metadata: dict = field(default_factory=dict)

    def row_metadata(self, row: int, *, device: torch.device):
        key = (row, device)
        if key not in self.device_row_metadata:
            start = sum(self.lengths[:row])
            flags = self.packed_observers[start:start + self.lengths[row]]
            persistent = (~flags).nonzero().flatten()
            observers = flags.nonzero().flatten()
            preceding = (~flags).long().cumsum(0)[observers]
            self.device_row_metadata[key] = tuple(
                value.to(device=device) for value in (persistent, observers, preceding)
            )
        return self.device_row_metadata[key]


_VALID_TOKEN_LAYOUT: ContextVar[_ValidTokenLayout | None] = ContextVar(
    "prism_qwen35_valid_token_layout", default=None
)


def get_qwen35_valid_token_layout() -> _ValidTokenLayout | None:
    """Read the current shared layout without rebuilding masks or indices.

    Consumers must treat this object as read-only, except for its existing
    device-index cache. Checkpoint closures must capture gathered tensors, not
    consult this scope again during backward after the caller has exited it.
    ``None`` means an all-valid or an unscoped call and requires native fallback.
    """
    return _VALID_TOKEN_LAYOUT.get()


@contextmanager
def qwen35_valid_token_scope(
    valid_token_mask: torch.Tensor, observer_token_mask: torch.Tensor | None = None,
    *, observer_tail_tokens: int = 1, allow_invalid_observer_tail: bool = False,
    stream: str = "unspecified", allow_inactive_observers: bool = False,
):
    """Exclude invalid *current-query* tokens from GDN/KDA state updates.

    HF discards its linear-attention mask on cached calls, so the caller must
    supply this independent boolean [B, T] mask on every relevant Qwen call.
    The layout is copied once per scope and reused by every linear layer.
    All-valid calls retain the ordinary implementation exactly. Individual
    empty streams are supported; a wholly empty query is rejected before any
    model forward. Activation-checkpoint recomputation must restore this scope
    explicitly, or the caller must reject that unsupported configuration.

    An optional observer mask marks valid ACT rows. Each consecutive group
    moves after the specified query tail only inside the recurrent scan. Every ACT
    reads a private self update from the same persistent prefix,
    and leave persistent convolution/recurrent state unchanged. Output rows
    are scattered back to the original [camera slots, CLS, ACT, Q] layout.
    """
    if stream not in {"unspecified", "prefix", "main", "query"}:
        raise ValueError("Qwen token stream must be unspecified, prefix, main, or query")
    if type(allow_inactive_observers) is not bool:
        raise ValueError("Inactive ACT observer permission must be boolean")
    if (not torch.is_tensor(valid_token_mask) or valid_token_mask.dtype != torch.bool
            or valid_token_mask.ndim != 2 or min(valid_token_mask.shape) < 1):
        raise ValueError("Qwen valid-token scope requires a nonempty boolean [B, T] mask")
    mask = valid_token_mask.detach().to(device="cpu").clone()
    lengths = tuple(int(value) for value in mask.sum(dim=1).tolist())
    if not any(lengths):
        raise ValueError("Qwen valid-token scope requires at least one valid current-query token")
    layout = None
    if observer_token_mask is not None:
        if (not torch.is_tensor(observer_token_mask) or observer_token_mask.dtype != torch.bool
                or tuple(observer_token_mask.shape) != tuple(mask.shape)):
            raise ValueError("Observer token mask must match the boolean [B, T] valid-token mask")
        observers = observer_token_mask.detach().to(device="cpu").clone()
        if not torch.equal(observers, observers[:1].expand_as(observers)):
            raise ValueError("Observer positions must be identical across batch rows")
        if not allow_inactive_observers and bool((observers & ~mask).any()):
            raise ValueError("Observer tokens must be valid current-query tokens")
        if type(observer_tail_tokens) is not int or observer_tail_tokens < 0:
            raise ValueError("Observer query-tail width must be a nonnegative integer")
        locations = observers[0].nonzero().flatten()
        if locations.numel():
            # Rotate [ACT_0, ..., ACT_H-1, Q_0, ..., Q_R-1] to
            # [Q_0, ..., Q_R-1, ACT_0, ..., ACT_H-1].
            # The recurrent scan publishes valid Qs first; all ACTs then read the
            # same state without publishing an update of their own. Moving
            # one observer at a time would accidentally read another ACT.
            starts = locations[torch.cat((torch.ones(1, dtype=torch.bool), locations.diff() != 1))]
            ends = locations[torch.cat((locations.diff() != 1, torch.ones(1, dtype=torch.bool)))]
            order = torch.arange(mask.shape[1]).expand(mask.shape[0], -1).clone()
            previous_end = 0
            for start, end in zip(starts.tolist(), ends.tolist()):
                tail_end = end + 1 + observer_tail_tokens
                active_rows = mask[:, start:end + 1].all(dim=1)
                if allow_inactive_observers:
                    if (bool((mask[:, start:end + 1].any(dim=1) != active_rows).any())
                            or bool((mask[:, previous_end:tail_end].any(dim=1) & ~active_rows).any())):
                        raise ValueError("Inactive ACT observers require a wholly invalid timestep")
                if (tail_end > mask.shape[1]
                        or bool(observers[0, end + 1:tail_end].any())
                        or (not allow_invalid_observer_tail
                            and not bool(mask[active_rows, end + 1:tail_end].all()))):
                    raise ValueError("Every ACT observer group must be followed by its valid non-observer query tail")
                order[:, start:tail_end] = torch.cat((
                    torch.arange(end + 1, tail_end), torch.arange(start, end + 1),
                ))
                previous_end = tail_end
            ordered_valid = mask.gather(1, order)
            flat_indices = (order + torch.arange(mask.shape[0])[:, None] * mask.shape[1])[ordered_valid]
            packed_observers = observers.gather(1, order)[ordered_valid]
            layout = _ObserverTokenLayout(tuple(mask.shape), lengths, flat_indices,
                observer_mask=observers, packed_observers=packed_observers,
                uniform_layout=torch.equal(mask, mask[:1].expand_as(mask)))
    if layout is None and any(length != mask.shape[1] for length in lengths):
        layout = _ValidTokenLayout(tuple(mask.shape), lengths, mask.flatten().nonzero().flatten())
    if layout is not None:
        layout.stream = stream
    token = _VALID_TOKEN_LAYOUT.set(layout)
    try:
        yield
    finally:
        _VALID_TOKEN_LAYOUT.reset(token)


def _observer_convolution(
    mixed: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    previous: torch.Tensor | None, persistent: torch.Tensor,
    observers: torch.Tensor, preceding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve persistent inputs once and gather private ACT windows in parallel.

    ``mixed`` contains packed streams [B, channels, tokens]. Their ACT
    observers are ordered after Q. The returned tail contains only persistent
    raw inputs; no observer can enter another token's convolution window.
    """
    kernel = int(weight.shape[-1])
    raw = mixed.index_select(2, persistent)
    if previous is None:
        previous = mixed.new_zeros(mixed.shape[0], mixed.shape[1], kernel)
    history = torch.cat((previous.to(mixed), raw), dim=-1)
    convolved = F.conv1d(history.to(weight.dtype), weight.unsqueeze(1), bias,
        groups=mixed.shape[1])[:, :, -raw.shape[-1]:]
    convolved = F.silu(convolved).to(mixed.dtype)
    result = mixed.new_zeros(mixed.shape).index_copy(2, persistent, convolved)
    if observers.numel():
        batch, channels = mixed.shape[:2]
        count = observers.numel()
        window_indices = preceding[:, None] + torch.arange(1, kernel, device=mixed.device)[None]
        windows = history.transpose(1, 2).index_select(1, window_indices.flatten())
        windows = windows.reshape(batch, count, kernel - 1, channels)
        windows = windows.transpose(-1, -2).reshape(batch * count, channels, kernel - 1)
        current = mixed.index_select(2, observers).transpose(1, 2).reshape(batch * count, channels, 1)
        windows = torch.cat((windows, current), dim=-1)
        ephemeral = F.conv1d(windows.to(weight.dtype), weight.unsqueeze(1), bias,
            groups=mixed.shape[1])
        ephemeral = F.silu(ephemeral).to(mixed.dtype).reshape(batch, count, channels).transpose(1, 2)
        result = result.index_copy(2, observers, ephemeral)
    return result, history[:, :, -kernel:]


def _ephemeral_delta_rule_queries(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    log_decay: torch.Tensor, beta: torch.Tensor, observers: torch.Tensor,
    *, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read a private delta-rule update while the native scan keeps its state.

    Inputs q/k are already L2 normalized. For an observer, expand
    q^T[G S + beta k(v-k^T G S)] into q_eff^T S + private_bias.
    Only q_eff enters the scan, with zero update and zero log decay. Its
    private bias is added to that observer's output, never to persistent state.
    """
    repeat = int(value.shape[2]) // int(query.shape[2])
    if repeat * int(query.shape[2]) != int(value.shape[2]):
        raise ValueError("Observer delta-rule value heads must be a multiple of Q/K heads")
    if repeat != 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    accumulation_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    q = query.index_select(1, observers).to(accumulation_dtype)
    k = key.index_select(1, observers).to(accumulation_dtype)
    v = value.index_select(1, observers).to(accumulation_dtype)
    b = beta.index_select(1, observers).to(accumulation_dtype)
    g = log_decay.index_select(1, observers).to(accumulation_dtype)
    if g.ndim == q.ndim - 1:
        g = g.unsqueeze(-1)
    dot = (q * k).sum(dim=-1)
    effective_query = (q - (b * dot).unsqueeze(-1) * k) * g.exp()
    bias = (scale * b * dot).unsqueeze(-1) * v
    query = query.index_copy(1, observers, effective_query.to(query.dtype))
    private_bias = value.new_zeros(value.shape, dtype=accumulation_dtype).index_copy(1, observers, bias)
    return (query, key, log_decay.index_fill(1, observers, 0.),
        beta.index_fill(1, observers, 0.), private_bias)


def _merge_linear_cache_rows(rows: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Join row caches, using a zero initial state for previously empty rows."""
    template = next((value for value in rows if value is not None), None)
    if template is None:
        return None
    shape = (1, *template.shape[1:])
    if any(value is not None and tuple(value.shape) != shape for value in rows):
        raise RuntimeError("Qwen linear-attention row cache shapes differ")
    return torch.cat([
        value if value is not None else template.new_zeros(shape) for value in rows
    ], dim=0)


def normalize_linear_attention_variant(value: str | None) -> str:
    """Normalize the Qwen linear-attention implementation selector."""
    normalized = str(value or "gated_deltanet").strip().lower().replace("-", "_")
    aliases = {
        "gdn": "gated_deltanet",
        "gated_delta_net": "gated_deltanet",
        "gated_deltanet": "gated_deltanet",
        "kda": "kda",
        "kimi_delta_attention": "kda",
    }
    if normalized not in aliases:
        raise ValueError(
            "linear_attention_variant must be 'gated_deltanet' or 'kda', "
            f"got {value!r}."
        )
    return aliases[normalized]


def _make_kda_linear(
    in_features: int,
    out_features: int,
    *,
    reference: torch.Tensor,
    seed: int,
    zero: bool,
    initializer_std: float,
) -> nn.Linear:
    """Create a deterministic adapter without perturbing caller RNG state."""
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        layer = nn.Linear(in_features, out_features, bias=False)
        if zero:
            nn.init.zeros_(layer.weight)
        else:
            nn.init.normal_(layer.weight, mean=0.0, std=initializer_std)
    finally:
        torch.random.set_rng_state(rng_state)
    return layer.to(device=reference.device, dtype=reference.dtype)


def convert_qwen35_deltanet_to_kda(
    model: nn.Module,
    *,
    gate_rank: int = 128,
    chunk_size: int = 64,
    disable_recompute: bool = False,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
) -> int:
    """Add function-preserving KDA gates to Qwen3.5 Gated DeltaNet layers.

    The pretrained scalar gate remains the base gate and is broadcast across
    key dimensions. A zero-initialized low-rank residual supplies KDA's
    per-key-dimension degrees of freedom, so conversion starts in the exact GDN
    subspace while retaining every pretrained parameter.
    """
    if gate_rank <= 0:
        raise ValueError(f"kda_gate_rank must be positive, got {gate_rank}.")
    if chunk_size not in {32, 64}:
        raise ValueError(f"kda_chunk_size must be 32 or 64, got {chunk_size}.")
    if safe_gate and not (-5.0 <= lower_bound < 0.0):
        raise ValueError(
            "kda_lower_bound must be in [-5, 0) when kda_safe_gate=true, "
            f"got {lower_bound}."
        )

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
    except Exception as exc:
        raise RuntimeError(
            "KDA conversion requires a compatible Qwen3.5 Transformers "
            "implementation."
        ) from exc

    layer_cls = getattr(qwen35, "Qwen3_5GatedDeltaNet", None)
    if layer_cls is None:
        raise RuntimeError("Transformers does not expose Qwen3_5GatedDeltaNet.")

    hf_config = getattr(model, "config", None)
    text_config = getattr(hf_config, "text_config", hf_config)
    initializer_std = float(getattr(text_config, "initializer_range", 0.02))
    converted = 0
    for module in model.modules():
        if not isinstance(module, layer_cls):
            continue
        if bool(getattr(module, "_prism_kda_enabled", False)):
            if int(module._prism_kda_gate_rank) != gate_rank:
                raise ValueError(
                    "Qwen3.5 KDA layer was already converted with a different "
                    f"gate rank: {module._prism_kda_gate_rank} != {gate_rank}."
                )
            converted += 1
            continue

        reference = module.in_proj_a.weight
        layer_seed = 0x4B444100 + int(getattr(module, "layer_idx", converted))
        module.kda_gate_down = _make_kda_linear(
            module.hidden_size,
            gate_rank,
            reference=reference,
            seed=layer_seed,
            zero=False,
            initializer_std=initializer_std,
        )
        module.kda_gate_up = _make_kda_linear(
            gate_rank,
            module.num_v_heads * module.head_k_dim,
            reference=reference,
            seed=layer_seed + 1,
            zero=True,
            initializer_std=initializer_std,
        )
        module.register_parameter(
            "kda_dt_bias_delta",
            nn.Parameter(
                torch.zeros(
                    module.num_v_heads * module.head_k_dim,
                    device=module.dt_bias.device,
                    dtype=module.dt_bias.dtype,
                )
            ),
        )
        module._prism_kda_enabled = True
        module._prism_kda_gate_rank = gate_rank
        module._prism_kda_chunk_size = chunk_size
        module._prism_kda_disable_recompute = disable_recompute
        module._prism_kda_safe_gate = safe_gate
        module._prism_kda_lower_bound = lower_bound
        converted += 1
    return converted


def patch_qwen35_deltanet_chunk_cache() -> bool:
    """Allow cached multi-token Qwen3.5 DeltaNet chunks to keep memory.

    The upstream Qwen3.5 implementation only uses the cached DeltaNet recurrent
    state when ``seq_len == 1``. Sliding-window training needs cached chunks
    such as ``[CS_t, CLS_t, QS_t]`` while preserving both the convolution state
    and the DeltaNet recurrent state across chunks.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
    except Exception:
        return False

    cls = getattr(qwen35, "Qwen3_5GatedDeltaNet", None)
    if cls is None:
        return False
    cache_cls = getattr(qwen35, "Qwen3_5DynamicCache", None)
    if bool(getattr(cls, "_prism_chunk_cache_patch", False)) and (
        cache_cls is None or bool(getattr(cache_cls, "_prism_layer_skip_cache_patch", False))
    ):
        return True

    apply_mask_to_padding_states = qwen35.apply_mask_to_padding_states
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except Exception:
        chunk_kda = None
        fused_recurrent_kda = None

    if cache_cls is not None and not bool(
        getattr(cache_cls, "_prism_layer_skip_cache_patch", False)
    ):
        original_cache_init = cache_cls.__init__

        def cache_init(self, config):
            try:
                original_cache_init(self, config)
                return
            except ValueError as exc:
                if "linear_attention" not in str(exc):
                    raise

            self.layer_types = config.layer_types
            self.transformer_layers = [
                i
                for i in range(config.num_hidden_layers)
                if self.layer_types[i] == "full_attention"
            ]
            linear_layers = [
                i
                for i in range(config.num_hidden_layers)
                if self.layer_types[i] == "linear_attention"
            ]
            self.last_linear_layer = linear_layers[-1] if linear_layers else -1
            self.conv_states = [None for _ in range(config.num_hidden_layers)]
            self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
            self.key_cache = [None for _ in range(config.num_hidden_layers)]
            self.value_cache = [None for _ in range(config.num_hidden_layers)]

        def get_seq_length(self, layer_idx: int | None = 0) -> int:
            transformer_layers = getattr(self, "transformer_layers", [])
            if not transformer_layers:
                return 0
            layer_idx = transformer_layers[0] if layer_idx not in transformer_layers else layer_idx
            if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
                return 0
            return self.key_cache[layer_idx].shape[-2]

        @property
        def has_previous_state(self):
            last_linear_layer = getattr(self, "last_linear_layer", -1)
            if last_linear_layer < 0:
                return False
            return self.conv_states[last_linear_layer] is not None

        cache_cls.__init__ = cache_init
        cache_cls.get_seq_length = get_seq_length
        cache_cls.has_previous_state = has_previous_state
        cache_cls._prism_layer_skip_cache_patch = True

    def _cache_state(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None:
            return None
        return state.detach().clone()

    def _conv_with_cached_state(self, mixed_qkv: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        weight = self.conv1d.weight.squeeze(1)
        hidden_states_new = torch.cat([conv_state.to(dtype=mixed_qkv.dtype), mixed_qkv], dim=-1)
        cache_tail = _cache_state(hidden_states_new[:, :, -self.conv_kernel_size :])
        conv_out = F.conv1d(
            hidden_states_new.to(dtype=weight.dtype),
            weight.unsqueeze(1),
            self.conv1d.bias,
            padding=0,
            groups=self.conv_dim,
        )
        conv_out = F.silu(conv_out[:, :, -mixed_qkv.shape[-1] :]).to(dtype=mixed_qkv.dtype)
        return conv_out, cache_tail

    def _stateful_core(
        self,
        hidden_states: torch.Tensor,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        cache_params=None,
        *,
        precomputed_kda_gate: torch.Tensor | None = None,
        functional_convolution: bool = False,
        prepare_kda_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run the cache-mutating convolution and DeltaNet kernel eagerly."""
        batch_size, seq_len = z.shape[:2]
        has_cache_state = cache_params is not None and cache_params.has_previous_state
        conv_state = None
        recurrent_state = None
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        if has_cache_state and conv_state is not None:
            if seq_len == 1 and not (functional_convolution and self.training):
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                )
            else:
                mixed_qkv, conv_state_next = _conv_with_cached_state(self, mixed_qkv, conv_state)
                cache_params.conv_states[self.layer_idx] = conv_state_next
        else:
            if cache_params is not None:
                if seq_len >= self.conv_kernel_size:
                    conv_state_next = mixed_qkv[:, :, -self.conv_kernel_size :]
                else:
                    conv_state_next = F.pad(mixed_qkv, (self.conv_kernel_size - seq_len, 0))
                cache_params.conv_states[self.layer_idx] = _cache_state(conv_state_next)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                if functional_convolution:
                    # The outer GDN/FSDP boundary is already entered once.
                    # A per-stream kernel must not re-enter child modules.
                    convolved = F.conv1d(mixed_qkv, self.conv1d.weight, self.conv1d.bias,
                        padding=self.conv_kernel_size - 1, groups=self.conv_dim)
                else:
                    convolved = self.conv1d(mixed_qkv)
                mixed_qkv = F.silu(convolved[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        use_kda = bool(getattr(self, "_prism_kda_enabled", False))
        if use_kda:
            if chunk_kda is None or fused_recurrent_kda is None:
                raise RuntimeError(
                    "linear_attention_variant='kda' requires fla.ops.kda in the active runtime."
                )
            gate_delta = (self.kda_gate_up(self.kda_gate_down(hidden_states))
                if precomputed_kda_gate is None else precomputed_kda_gate)
            gate_delta = gate_delta.reshape(
                batch_size,
                seq_len,
                self.num_v_heads,
                self.head_k_dim,
            )
            g = a.unsqueeze(-1) + gate_delta
            beta = b
            if prepare_kda_only:
                # Ordinary packed streams only; ACT observers keep private state updates.
                return query, key, value, g, beta
            kda_dt_bias = self.dt_bias.repeat_interleave(self.head_k_dim)
            kda_dt_bias = kda_dt_bias + self.kda_dt_bias_delta
        else:
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if not use_kda and self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        use_recurrent_kernel = (
            has_cache_state
            and recurrent_state is not None
            and seq_len == 1
            and not self.training
        )
        if use_kda:
            kda_kwargs = {
                "q": query,
                "k": key,
                "v": value,
                "g": g,
                "beta": beta,
                "A_log": self.A_log,
                "dt_bias": kda_dt_bias,
                "initial_state": recurrent_state if has_cache_state else None,
                "output_final_state": cache_params is not None,
                "use_qk_l2norm_in_kernel": True,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "state_v_first": False,
            }
            if use_recurrent_kernel:
                core_attn_out, last_recurrent_state = fused_recurrent_kda(
                    **kda_kwargs,
                    lower_bound=(
                        self._prism_kda_lower_bound
                        if self._prism_kda_safe_gate
                        else None
                    ),
                )
            else:
                core_attn_out, last_recurrent_state = chunk_kda(
                    **kda_kwargs,
                    safe_gate=self._prism_kda_safe_gate,
                    lower_bound=(
                        self._prism_kda_lower_bound
                        if self._prism_kda_safe_gate
                        else None
                    ),
                    chunk_size=self._prism_kda_chunk_size,
                    disable_recompute=self._prism_kda_disable_recompute,
                )
        elif use_recurrent_kernel:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            # FLA's fused recurrent DeltaNet path is inference-only: its
            # autograd backward raises NotImplementedError. Even cached
            # single-token chunks must stay on the chunk kernel in training.
            initial_state = recurrent_state if has_cache_state else None
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            # Keep DeltaNet memory values across cached chunks, but truncate
            # gradients at chunk boundaries. Full-episode BPTT through mutable
            # cache tensors is both memory-heavy and prone to version-counter
            # failures during backward.
            cache_params.recurrent_states[self.layer_idx] = _cache_state(last_recurrent_state)

        return core_attn_out

    def _observer_stateful_core(
        self, hidden_states, mixed_qkv, z, b, a, cache_params, *,
        persistent, observers, preceding, precomputed_kda_gate=None,
        prepare_kda_only=False,
    ):
        """Parallel ACT readouts use private delta updates and persistent conv."""
        batch_size, seq_len = z.shape[:2]
        previous_conv = (cache_params.conv_states[self.layer_idx]
            if cache_params is not None else None)
        previous_state = (cache_params.recurrent_states[self.layer_idx]
            if cache_params is not None else None)
        convolved, conv_tail = _observer_convolution(
            mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias,
            previous_conv, persistent, observers, preceding,
        )
        mixed = convolved.transpose(1, 2)
        query, key, value = torch.split(mixed,
            [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        if query.is_cuda:
            from fla.modules.l2norm import l2norm

            # This is the same public FLA kernel used inside chunk_kda's
            # use_qk_l2norm_in_kernel=True path, including its autograd rule.
            query, key = l2norm(query), l2norm(key)
        else:
            # CPU reference only; production CUDA retains the official kernel.
            query = query * torch.rsqrt(query.float().square().sum(-1, keepdim=True) + 1e-6).to(query)
            key = key * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6).to(key)

        use_kda = bool(getattr(self, "_prism_kda_enabled", False))
        if use_kda:
            if chunk_kda is None:
                raise RuntimeError("KDA ACT observers require fla.ops.kda in the active runtime")
            from fla.ops.common.gate import fused_beta_sigmoid
            from fla.ops.kda.gate import fused_kda_gate

            gate_delta = precomputed_kda_gate.reshape(
                batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            raw_gate = a.unsqueeze(-1) + gate_delta
            bias = self.dt_bias.repeat_interleave(self.head_k_dim) + self.kda_dt_bias_delta
            log_decay = fused_kda_gate(raw_gate, self.A_log, bias,
                lower_bound=self._prism_kda_lower_bound if self._prism_kda_safe_gate else None,
                output_dtype=torch.float32)
            beta = fused_beta_sigmoid(b)
        else:
            beta = b.sigmoid()
            log_decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        scale = self.head_k_dim ** -0.5
        query, key, log_decay, beta, private_bias = _ephemeral_delta_rule_queries(
            query, key, value, log_decay, beta, observers, scale=scale)
        if prepare_kda_only:
            if not use_kda:
                raise ValueError("Batched observer preparation requires KDA")
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx] = _cache_state(conv_tail)
            # ACT rows have zero decay/update in the scan. Their transformed
            # queries and private bias retain the original self-read gradients;
            # a cu_seqlens boundary must still isolate each persistent stream.
            return query, key, value, log_decay, beta, private_bias
        if use_kda:
            core, final_state = chunk_kda(
                q=query, k=key, v=value, g=log_decay, beta=beta, scale=scale,
                initial_state=previous_state, output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False, state_v_first=False,
                safe_gate=self._prism_kda_safe_gate,
                lower_bound=self._prism_kda_lower_bound if self._prism_kda_safe_gate else None,
                chunk_size=self._prism_kda_chunk_size,
                disable_recompute=self._prism_kda_disable_recompute,
            )
        else:
            core, final_state = self.chunk_gated_delta_rule(
                query, key, value, g=log_decay, beta=beta,
                initial_state=previous_state, output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=False,
            )
        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = _cache_state(conv_tail)
            cache_params.recurrent_states[self.layer_idx] = _cache_state(final_state)
        return (core.to(private_bias.dtype) + private_bias).to(core.dtype)

    def _ordinary_valid_token_forward(self, hidden_states, cache_params, layout):
        indices = layout.indices_for(hidden_states)
        batch_size, seq_len, hidden_size = hidden_states.shape
        packed = hidden_states.reshape(-1, hidden_size).index_select(0, indices).unsqueeze(0)
        total = packed.shape[1]
        # Each projection is called once, on real tokens only. In particular,
        # neither padded inputs nor projection biases enter a recurrent kernel.
        mixed = self.in_proj_qkv(packed).transpose(1, 2)
        z = self.in_proj_z(packed).reshape(1, total, -1, self.head_v_dim)
        b, a = self.in_proj_b(packed), self.in_proj_a(packed)
        gate = (self.kda_gate_up(self.kda_gate_down(packed))
            if bool(getattr(self, "_prism_kda_enabled", False)) else None)
        previous_conv = previous_recurrent = None
        has_previous = False
        if cache_params is not None:
            previous_conv = cache_params.conv_states[self.layer_idx]
            previous_recurrent = cache_params.recurrent_states[self.layer_idx]
            has_previous = cache_params.has_previous_state
            for value in (previous_conv, previous_recurrent):
                if value is not None and value.shape[0] != batch_size:
                    raise ValueError("Qwen cached state batch differs from the valid-token scope")
        outputs, next_conv, next_recurrent = [], [], []
        batch_kda = (
            self.training and bool(getattr(self, "_prism_kda_enabled", False))
            and os.environ.get("RB_OPT_KDA_VARLEN", "0") == "1"
        )
        prepared, initial_rows, nonempty_rows = [], [], []
        row_inputs = None
        if os.environ.get("RB_OPT_PACKED_ROW_SPLIT", "0") == "1":
            _record_researchboost_dispatch("packed_row_split_calls")
            # These rows partition the packed token axis. A single split lets
            # backward concatenate their gradients, avoiding one full-sized
            # zero/scatter and accumulation per independent row slice.
            lengths = layout.lengths
            row_inputs = (
                packed.split(lengths, dim=1), mixed.split(lengths, dim=2),
                z.split(lengths, dim=1), b.split(lengths, dim=1),
                a.split(lengths, dim=1),
                None if gate is None else gate.split(lengths, dim=1),
            )
        offset = 0
        for row, length in enumerate(layout.lengths):
            conv = None if previous_conv is None else previous_conv[row:row + 1].clone()
            recurrent = (None if previous_recurrent is None
                else previous_recurrent[row:row + 1].clone())
            row_cache = None
            if cache_params is not None:
                row_cache = SimpleNamespace(
                    conv_states=[None] * self.layer_idx + [conv],
                    recurrent_states=[None] * self.layer_idx + [recurrent],
                    has_previous_state=has_previous,
                )
            if length:
                end = offset + length
                if row_inputs is None:
                    row_packed, row_mixed, row_z = (
                        packed[:, offset:end], mixed[:, :, offset:end], z[:, offset:end]
                    )
                    row_b, row_a = b[:, offset:end], a[:, offset:end]
                    row_gate = None if gate is None else gate[:, offset:end]
                else:
                    row_packed, row_mixed, row_z, row_b, row_a = (
                        parts[row] for parts in row_inputs[:5]
                    )
                    row_gate = None if row_inputs[5] is None else row_inputs[5][row]
                result = self._prism_qwen35_stateful_core(
                    row_packed, row_mixed, row_z, row_b, row_a, row_cache,
                    precomputed_kda_gate=row_gate,
                    functional_convolution=True,
                    prepare_kda_only=batch_kda,
                )
                if batch_kda:
                    prepared.append(result)
                    nonempty_rows.append(row)
                    initial_rows.append(
                        row_cache.recurrent_states[self.layer_idx]
                        if has_previous and row_cache is not None else None
                    )
                else:
                    outputs.append(result)
                offset = end
            if row_cache is not None:
                next_conv.append(row_cache.conv_states[self.layer_idx])
                next_recurrent.append(row_cache.recurrent_states[self.layer_idx])
        if batch_kda:
            _record_researchboost_dispatch("ordinary_kda_varlen_calls")
            query, key, value, g, beta = (
                torch.cat([parts[index] for parts in prepared], dim=1)
                for index in range(5)
            )
            cu_seqlens, cu_seqlens_cpu = layout.cumulative_lengths_for(hidden_states.device)
            kda_dt_bias = self.dt_bias.repeat_interleave(self.head_k_dim)
            kda_dt_bias = kda_dt_bias + self.kda_dt_bias_delta
            core, last_recurrent = chunk_kda(
                q=query, k=key, v=value, g=g, beta=beta,
                A_log=self.A_log, dt_bias=kda_dt_bias,
                initial_state=_merge_linear_cache_rows(initial_rows),
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True, state_v_first=False,
                safe_gate=self._prism_kda_safe_gate,
                lower_bound=(self._prism_kda_lower_bound if self._prism_kda_safe_gate else None),
                chunk_size=self._prism_kda_chunk_size,
                disable_recompute=self._prism_kda_disable_recompute,
                cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
            )
            outputs.append(core)
            if cache_params is not None:
                for index, row in enumerate(nonempty_rows):
                    next_recurrent[row] = _cache_state(last_recurrent[index:index + 1])
        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = _merge_linear_cache_rows(next_conv)
            cache_params.recurrent_states[self.layer_idx] = _merge_linear_cache_rows(next_recurrent)
        core = torch.cat(outputs, dim=1).reshape(-1, self.head_v_dim)
        normalized = self.norm(core, z.reshape(-1, self.head_v_dim)).reshape(1, total, -1)
        valid_output = self.out_proj(normalized).squeeze(0)
        output = valid_output.new_zeros(batch_size * seq_len, hidden_size)
        return output.index_copy(0, indices, valid_output).reshape(batch_size, seq_len, hidden_size)

    def _valid_token_forward(self, hidden_states, cache_params, layout):
        # Keep every ordinary stream on main's original implementation,
        # including its opt-in packed-Qwen optimizations. ACT observers use
        # private convolution/delta updates and reordered rows instead.
        if not isinstance(layout, _ObserverTokenLayout):
            return _ordinary_valid_token_forward(self, hidden_states, cache_params, layout)
        indices = layout.indices_for(hidden_states)
        batch_size, seq_len, hidden_size = hidden_states.shape
        packed = hidden_states.reshape(-1, hidden_size).index_select(0, indices).unsqueeze(0)
        total = packed.shape[1]
        # Each projection is called once, on real tokens only. In particular,
        # neither padded inputs nor projection biases enter a recurrent kernel.
        mixed = self.in_proj_qkv(packed).transpose(1, 2)
        z = self.in_proj_z(packed).reshape(1, total, -1, self.head_v_dim)
        b, a = self.in_proj_b(packed), self.in_proj_a(packed)
        gate = (self.kda_gate_up(self.kda_gate_down(packed))
            if bool(getattr(self, "_prism_kda_enabled", False)) else None)
        previous_conv = previous_recurrent = None
        has_previous = False
        if cache_params is not None:
            previous_conv = cache_params.conv_states[self.layer_idx]
            previous_recurrent = cache_params.recurrent_states[self.layer_idx]
            has_previous = cache_params.has_previous_state
            for value in (previous_conv, previous_recurrent):
                if value is not None and value.shape[0] != batch_size:
                    raise ValueError("Qwen cached state batch differs from the valid-token scope")
        if isinstance(layout, _ObserverTokenLayout) and layout.uniform_layout:
            _record_researchboost_dispatch("observer_uniform_calls")
            # All-present camera batches keep one batched native scan, just
            # like the legacy all-valid fast path. Only genuinely different
            # per-row inventories need the existing ragged fallback below.
            length = layout.lengths[0]
            persistent, observers, preceding = layout.row_metadata(0, device=packed.device)
            core = _observer_stateful_core(
                self, packed.reshape(batch_size, length, hidden_size),
                mixed.transpose(1, 2).reshape(batch_size, length, -1).transpose(1, 2),
                z.reshape(batch_size, length, self.num_v_heads, self.head_v_dim),
                b.reshape(batch_size, length, self.num_v_heads),
                a.reshape(batch_size, length, self.num_v_heads), cache_params,
                persistent=persistent, observers=observers, preceding=preceding,
                precomputed_kda_gate=None if gate is None else gate.reshape(batch_size, length, -1),
            ).reshape(-1, self.head_v_dim)
        else:
            outputs, next_conv, next_recurrent = [], [], []
            batch_observer_kda = (
                isinstance(layout, _ObserverTokenLayout) and self.training
                and bool(getattr(self, "_prism_kda_enabled", False))
                and os.environ.get("RB_OPT_KDA_VARLEN", "0") == "1"
            )
            prepared, initial_rows, nonempty_rows = [], [], []
            row_inputs = None
            if os.environ.get("RB_OPT_PACKED_ROW_SPLIT", "0") == "1":
                _record_researchboost_dispatch("packed_row_split_calls")
                lengths = layout.lengths
                row_inputs = (
                    packed.split(lengths, dim=1), mixed.split(lengths, dim=2),
                    z.split(lengths, dim=1), b.split(lengths, dim=1),
                    a.split(lengths, dim=1),
                    None if gate is None else gate.split(lengths, dim=1),
                )
            offset = 0
            for row, length in enumerate(layout.lengths):
                conv = None if previous_conv is None else previous_conv[row:row + 1].clone()
                recurrent = (None if previous_recurrent is None
                    else previous_recurrent[row:row + 1].clone())
                row_cache = None
                if cache_params is not None:
                    row_cache = SimpleNamespace(
                        conv_states=[None] * self.layer_idx + [conv],
                        recurrent_states=[None] * self.layer_idx + [recurrent],
                        has_previous_state=has_previous,
                    )
                if length:
                    end = offset + length
                    if row_inputs is None:
                        row_packed, row_mixed, row_z = packed[:, offset:end], mixed[:, :, offset:end], z[:, offset:end]
                        row_b, row_a = b[:, offset:end], a[:, offset:end]
                        row_gate = None if gate is None else gate[:, offset:end]
                    else:
                        row_packed, row_mixed, row_z, row_b, row_a = (parts[row] for parts in row_inputs[:5])
                        row_gate = None if row_inputs[5] is None else row_inputs[5][row]
                    if isinstance(layout, _ObserverTokenLayout):
                        persistent, observers, preceding = layout.row_metadata(row, device=packed.device)
                        result = _observer_stateful_core(
                            self, row_packed, row_mixed, row_z,
                            row_b, row_a, row_cache,
                            persistent=persistent, observers=observers, preceding=preceding,
                            precomputed_kda_gate=row_gate, prepare_kda_only=batch_observer_kda,
                        )
                        if batch_observer_kda:
                            prepared.append(result)
                            nonempty_rows.append(row)
                            initial_rows.append(None if row_cache is None
                                else row_cache.recurrent_states[self.layer_idx])
                        else:
                            outputs.append(result)
                    else:
                        outputs.append(self._prism_qwen35_stateful_core(
                            row_packed, row_mixed, row_z, row_b, row_a, row_cache,
                            precomputed_kda_gate=row_gate,
                            functional_convolution=True,
                        ))
                    offset = end
                if row_cache is not None:
                    next_conv.append(row_cache.conv_states[self.layer_idx])
                    next_recurrent.append(row_cache.recurrent_states[self.layer_idx])
            if batch_observer_kda:
                _record_researchboost_dispatch("observer_kda_varlen_calls")
                query, key, value, log_decay, beta, private_bias = (
                    torch.cat([parts[index] for parts in prepared], dim=1)
                    for index in range(6)
                )
                cu_seqlens, cu_seqlens_cpu = layout.cumulative_lengths_for(hidden_states.device)
                output, final_state = chunk_kda(
                    q=query, k=key, v=value, g=log_decay, beta=beta,
                    scale=self.head_k_dim ** -0.5,
                    initial_state=_merge_linear_cache_rows(initial_rows),
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False, state_v_first=False,
                    safe_gate=self._prism_kda_safe_gate,
                    lower_bound=self._prism_kda_lower_bound if self._prism_kda_safe_gate else None,
                    chunk_size=self._prism_kda_chunk_size,
                    disable_recompute=self._prism_kda_disable_recompute,
                    cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                )
                outputs.append((output.to(private_bias.dtype) + private_bias).to(output.dtype))
                if cache_params is not None:
                    for index, row in enumerate(nonempty_rows):
                        next_recurrent[row] = _cache_state(final_state[index:index + 1])
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx] = _merge_linear_cache_rows(next_conv)
                cache_params.recurrent_states[self.layer_idx] = _merge_linear_cache_rows(next_recurrent)
            core = torch.cat(outputs, dim=1).reshape(-1, self.head_v_dim)
        normalized = self.norm(core, z.reshape(-1, self.head_v_dim)).reshape(1, total, -1)
        valid_output = self.out_proj(normalized).squeeze(0)
        output = valid_output.new_zeros(batch_size * seq_len, hidden_size)
        return output.index_copy(0, indices, valid_output).reshape(batch_size, seq_len, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
    ):
        layout = _VALID_TOKEN_LAYOUT.get()
        if layout is not None:
            return _valid_token_forward(self, hidden_states, cache_params, layout)
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        core_attn_out = self._prism_qwen35_stateful_core(
            hidden_states,
            mixed_qkv,
            z,
            b,
            a,
            cache_params,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output

    cls._prism_qwen35_stateful_core = _stateful_core
    cls.forward = forward
    cls._prism_chunk_cache_patch = True
    return True
