"""Explicit full-attention windows, independently of global DeltaNet memory.

Old configurations omit ``full_attention_window_policy``. Their historical
booleans keep their exact semantics, including the cache-only legacy mode.
Rolling is an explicit opt-in: eight means seven previous timesteps plus the
current timestep, with a permanently visible instruction prefix. Both main and
query streams use this temporal lower bound.
"""
from __future__ import annotations

from typing import Any


def _get(config: Any, key: str, default: Any) -> Any:
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def resolved_full_attention_window_policy(config: Any) -> str:
    """Resolve a policy without silently reinterpreting legacy checkpoints."""
    requested = str(_get(config, 'full_attention_window_policy', 'legacy')).strip().lower()
    if requested not in {'legacy', 'rolling', 'blockwise'}:
        raise ValueError('full_attention_window_policy must be legacy, rolling, or blockwise')
    sliding = bool(_get(config, 'sliding_window_attention', False))
    blockwise = bool(_get(config, 'blockwise_full_attention', False))
    if requested == 'legacy':
        return 'blockwise' if blockwise else ('legacy_sliding_cache' if sliding else 'full_causal')
    if not sliding:
        raise ValueError(f'full_attention_window_policy={requested!r} requires sliding_window_attention=true')
    if blockwise != (requested == 'blockwise'):
        raise ValueError(f'full_attention_window_policy={requested!r} requires '
            f'blockwise_full_attention={requested == "blockwise"}')
    window = _get(config, 'attention_window_timesteps', 16)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError('attention_window_timesteps must be a positive integer')
    if str(_get(config, 'attn_implementation', 'sdpa')) != 'sdpa':
        raise ValueError(f'full_attention_window_policy={requested!r} requires attn_implementation=sdpa')
    if not bool(_get(config, 'pass_deltanet_state_between_chunks', True)):
        raise ValueError(f'full_attention_window_policy={requested!r} requires global DeltaNet state passing')
    if str(_get(config, 'blockwise_full_attention_backend', 'dense_sdpa')) != 'dense_sdpa':
        raise ValueError('Windowed full attention requires blockwise_full_attention_backend=dense_sdpa')
    return requested


def attention_window_policy_contract(config: Any) -> dict:
    """Bind behavior while equating implicit and explicit legacy block mode."""
    return dict(policy=resolved_full_attention_window_policy(config),
        window_timesteps=int(_get(config, 'attention_window_timesteps', 16)),
        main_timestep_causal=bool(_get(config, 'main_stream_timestep_causal', False)))


def attention_window_policy_description(config: Any) -> dict:
    """Return serialized startup evidence for both streams and their caches."""
    mode = resolved_full_attention_window_policy(config)
    window = int(_get(config, 'attention_window_timesteps', 16))
    timestep_causal = bool(_get(config, 'main_stream_timestep_causal', False))
    return dict(requested_policy=_get(config, 'full_attention_window_policy', 'legacy'),
        resolved_policy=mode, window_timesteps=window,
        current_timestep_included=True,
        main_stream=('same-timestep mutual' if timestep_causal else 'token causal'),
        query_stream='token causal; CLS through t generates query for t+1',
        temporal_visibility=('max(0,t-W+1)..t' if mode == 'rolling' else
            ('floor(t/W)*W..t' if mode == 'blockwise' else 'historical behavior')),
        persistent_history_timesteps=(window - 1 if mode == 'rolling' else
            ('current fixed block' if mode == 'blockwise' else window)),
        batched_chunk_policy=('per-query rolling lower bound inside every chunk' if mode == 'rolling'
            else ('fixed non-overlapping blocks' if mode == 'blockwise'
                else 'legacy causal mask without a per-query rolling lower bound')),
        incremental_policy=('retain W-1 previous timesteps after each call' if mode == 'rolling'
            else ('clear temporal full-attention KV before multiples of W' if mode == 'blockwise'
                else 'historical cache behavior')),
        instruction_prefix='always retained and visible',
        act_keys=('self only; never persisted' if _get(config, 'main_stream_action_token', False)
            else 'not enabled'),
        kda_state=('legacy per-transition W=1 restores prefix state' if
            mode == 'legacy_sliding_cache' and window == 1 else
            ('global; never reset or cropped at attention-window boundaries' if
                bool(_get(config, 'pass_deltanet_state_between_chunks', True)) else
                'legacy state passing disabled')))
