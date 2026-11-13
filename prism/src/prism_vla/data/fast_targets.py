"""Head-local FAST+ labels with explicit physical time and supervision coverage."""

from __future__ import annotations

import os
from typing import Sequence

import torch

from prism_vla.data.fast_tokenizer import FASTActionTokenizer
from prism_vla.modeling.action_layout import robot_action_sampling_mask


FAST_TOKEN_VALIDATION_KEY = "fast_token_validation"


def trust_dataset_fast_targets() -> bool:
    """Opt in to carrying CPU validation through the immutable target path."""
    return os.environ.get("PRISM_FAST_TRUST_DATASET_TARGETS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def fast_token_validation_contract(*, vocab_size: int, eos_token_id: int,
                                   max_fast_len: int) -> dict[str, int | str]:
    """Describe the token invariants checked before device transfer.

    This is internal provenance, not a security boundary. Target selection and
    right padding preserve the invariants; the collator revalidates every final
    CPU token before attaching the contract used by the model.
    """
    if (type(vocab_size) is not int or type(eos_token_id) is not int
            or type(max_fast_len) is not int or vocab_size <= 0
            or not 0 <= eos_token_id < vocab_size or max_fast_len <= 0):
        raise ValueError("Invalid FAST token validation contract")
    return {"schema": "prism_cpu_fast_tokens_v1", "vocab_size": vocab_size,
            "eos_token_id": eos_token_id, "max_fast_len": max_fast_len}


def validate_fast_token_ids(tokens: torch.Tensor, *, vocab_size: int,
                            eos_token_id: int, max_fast_len: int,
                            require_cpu: bool = False) -> dict[str, int | str]:
    """Check IDs, suffix padding and exactly one terminal EOS per nonempty row."""
    contract = fast_token_validation_contract(vocab_size=vocab_size,
        eos_token_id=eos_token_id, max_fast_len=max_fast_len)
    if not torch.is_tensor(tokens) or tokens.ndim < 2 or tokens.dtype != torch.long:
        raise ValueError("FAST targets must be a torch.long tensor with token rows")
    if require_cpu and tokens.device.type != "cpu":
        raise ValueError("Dataset FAST validation requires CPU target tensors")
    if not 1 <= tokens.shape[-1] <= max_fast_len:
        raise ValueError(f"FAST target length must be in [1, {max_fast_len}]")
    rows = tokens.reshape(-1, tokens.shape[-1])
    if bool(((rows < -1) | (rows >= vocab_size)).any()):
        raise ValueError("FAST target IDs must be -1 or inside the tokenizer vocabulary")
    valid = rows != -1
    if bool((valid & (~valid).long().cumsum(1).bool()).any()):
        raise ValueError("FAST padding must follow all valid tokens")
    lengths = valid.sum(1)
    last = rows.gather(1, (lengths - 1).clamp_min(0)[:, None]).squeeze(1)
    eos_count = (rows == eos_token_id).sum(1)
    if bool(((lengths > 0) & ((last != eos_token_id) | (eos_count != 1))).any()):
        raise ValueError("Each nonempty FAST target must end with exactly one EOS; do not truncate labels")
    return contract


def fast_semantic_mask(
    action_horizon: int,
    action_dim: int,
    action_token_types: Sequence[str] = (),
    robot_action_layout: str = "",
) -> torch.Tensor:
    """Distinguish canonical schema padding from missing semantic coordinates."""
    robot_mask = robot_action_sampling_mask(
        robot_action_layout, action_horizon=action_horizon, action_dim=action_dim,
        action_token_types=tuple(action_token_types),
    )
    if robot_mask is not None:
        return robot_mask[0]
    if not action_token_types:
        return torch.ones(action_horizon, action_dim, dtype=torch.bool)
    if action_dim != 15 or tuple(action_token_types) != ("head", "eef", "hand", "eef", "hand"):
        raise ValueError("Typed FAST targets require the declared five-token rotation-6D 15D layout")
    if action_horizon % len(action_token_types):
        raise ValueError("FAST action horizon must contain complete typed physical steps")
    step = torch.zeros(len(action_token_types), action_dim, dtype=torch.bool)
    for index, token_type in enumerate(action_token_types):
        step[index, :15 if token_type == "hand" else 9] = True
    return step.repeat(action_horizon // len(action_token_types), 1)


@torch.no_grad()
def prepare_fast_targets(
    action_chunks: torch.Tensor,
    *,
    tokenizer: FASTActionTokenizer,
    max_fast_len: int,
    action_chunk_valid_steps: torch.Tensor | None = None,
    action_chunk_loss_mask: torch.Tensor | None = None,
    action_semantic_mask: torch.Tensor | None = None,
    action_token_types: Sequence[str] = (),
    robot_action_layout: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one head's normalized labels, returning tokens and valid rows.

    DCT and BPE mix coordinates: missing semantic labels cannot be masked at the
    token level. Incomplete semantic chunks are therefore unsupervised (-1).
    Schema padding is known zero, not a missing label. The typed ego layout is
    reshaped from [physical_time*5, 15] to [physical_time, 75] so DCT operates
    over physical time rather than interleaved body parts. No label is truncated.
    """
    if action_chunks.ndim != 3:
        raise ValueError(f"FAST expected action_chunks [B, H, A], got {tuple(action_chunks.shape)}")
    batch, horizon, width = action_chunks.shape
    token_types = tuple(action_token_types)
    physical_horizon = horizon // len(token_types) if token_types else horizon
    physical_width = width * len(token_types) if token_types else width
    if (tokenizer.action_horizon, tokenizer.action_dim) != (physical_horizon, physical_width):
        raise ValueError(
            "FAST tokenizer shape differs from physical-time action layout: "
            f"tokenizer=({tokenizer.action_horizon}, {tokenizer.action_dim}), "
            f"labels=({physical_horizon}, {physical_width})"
        )
    canonical_semantic = fast_semantic_mask(horizon, width, token_types, robot_action_layout).to(action_chunks.device)
    if action_semantic_mask is None:
        semantic = canonical_semantic[None].expand_as(action_chunks)
    else:
        if (not torch.is_tensor(action_semantic_mask) or action_semantic_mask.dtype != torch.bool
                or action_semantic_mask.shape != action_chunks.shape):
            raise ValueError("FAST action_semantic_mask must be boolean [B,H,A] matching action_chunks")
        semantic = action_semantic_mask.to(action_chunks.device)
        if bool((semantic & ~canonical_semantic[None]).any()):
            raise ValueError("FAST structural semantics cannot add coordinates outside the declared action schema")
        # Hardware/schema padding is constant over physical time. A caller
        # cannot convert missing endpoints or right-padded time into padding.
        physical_semantic = semantic.reshape(batch, physical_horizon, physical_width)
        if not torch.equal(physical_semantic, physical_semantic[:, :1].expand_as(physical_semantic)):
            raise ValueError("FAST structural semantics must be constant across the full physical horizon")
    observed = torch.ones_like(action_chunks, dtype=torch.bool)
    if action_chunk_loss_mask is not None:
        if action_chunk_loss_mask.shape != action_chunks.shape:
            raise ValueError("FAST action_chunk_loss_mask must match action_chunks exactly")
        observed &= action_chunk_loss_mask.to(action_chunks.device) > 0
    if action_chunk_valid_steps is not None:
        if action_chunk_valid_steps.shape != (batch,):
            raise ValueError("FAST action_chunk_valid_steps must be [B]")
        observed &= torch.arange(horizon, device=action_chunks.device)[None, :, None] < action_chunk_valid_steps.to(action_chunks.device)[:, None, None]
    observed &= torch.isfinite(action_chunks)
    supervised_rows = (observed | ~semantic).flatten(1).all(1)
    supervised_rows &= semantic.flatten(1).any(1)
    if action_semantic_mask is not None and action_chunk_valid_steps is not None:
        supervised_rows &= action_chunk_valid_steps.to(action_chunks.device) == horizon
    canonical = torch.where(semantic, action_chunks, 0.0)
    canonical = canonical.reshape(batch, physical_horizon, physical_width).detach().float().cpu().numpy()
    chunks: list[list[int]] = []
    max_length = 1
    for index, is_valid in enumerate(supervised_rows.cpu().tolist()):
        tokens = tokenizer.encode(canonical[index]) if is_valid else []
        if len(tokens) > max_fast_len:
            raise ValueError(
                f"FAST chunk requires {len(tokens)} tokens including EOS, exceeds max_fast_len={max_fast_len}; "
                "increase max_fast_len rather than truncating training labels"
            )
        chunks.append(tokens)
        max_length = max(max_length, len(tokens))
    targets = torch.full((batch, max_length), -1, dtype=torch.long, device=action_chunks.device)
    for index, tokens in enumerate(chunks):
        if tokens:
            targets[index, :len(tokens)] = torch.tensor(tokens, device=targets.device)
    return targets, supervised_rows
