"""Fail-closed validation shared by full-checkpoint loaders and unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prism_astribot_vla.data.instruction_conditioning import (
    normalize_instruction_conditioning_config,
    normalize_instruction_format,
)

INSTRUCTION_ENCODER_DEFAULTS = {
    "instruction_encoder_model_name": None,
    "instruction_encoder_revision": None,
    "instruction_encoder_nhead": 8,
    "instruction_encoder_dim_feedforward": None,
    "instruction_encoder_dropout": 0.0,
}


def instruction_conditioning_config(config: Any) -> dict[str, Any]:
    """Return the prompt contract, interpreting legacy checkpoints as plain/64."""
    get = config.get if isinstance(config, dict) else lambda key, default: getattr(config, key, default)
    instruction_format = normalize_instruction_format(get("instruction_format", "plain"))
    max_inst_len = get("max_inst_len", 64)
    if type(max_inst_len) is not int or max_inst_len <= 0:
        raise ValueError("max_inst_len must be a positive integer")
    raw_conditioning = get("instruction_conditioning", None)
    conditioning = normalize_instruction_conditioning_config(
        raw_conditioning if raw_conditioning is not None else {"format": instruction_format}
    )
    if conditioning["format"] != instruction_format:
        raise ValueError("instruction_format must match instruction_conditioning.format")
    return {"instruction_format": instruction_format, "max_inst_len": max_inst_len,
            "instruction_conditioning": conditioning}


def assert_instruction_conditioning_config_compatible(
    source: Any, target: Any, *, allow_astribot_source_extension: bool = False,
) -> None:
    """Bind saved text semantics and admit one new profile for Astribot warm starts."""
    original, requested = instruction_conditioning_config(source), instruction_conditioning_config(target)
    if allow_astribot_source_extension:
        original_profiles = original["instruction_conditioning"]["by_source"]
        requested_profiles = requested["instruction_conditioning"]["by_source"]
        if "astribot_mobile" not in original_profiles and "astribot_mobile" in requested_profiles:
            requested_profiles.pop("astribot_mobile")
    if original != requested:
        raise ValueError(
            "Instruction conditioning checkpoint contract mismatch: "
            f"checkpoint={original}, requested={requested}"
        )


def instruction_encoder_config(config: Any) -> dict[str, Any]:
    """Return the serialized instruction vocabulary and adapter contract."""
    get = config.get if isinstance(config, dict) else lambda key, default: getattr(config, key, default)
    result = {key: get(key, default) for key, default in INSTRUCTION_ENCODER_DEFAULTS.items()}
    # Keep legacy disabled manifests byte-for-byte compatible.
    if get("text_input_deepstack", False):
        result.update(
            text_input_deepstack=True,
            text_input_deepstack_layer_map=tuple(tuple(pair) for pair in get("text_input_deepstack_layer_map", ())),
            text_input_deepstack_position_policy=get("text_input_deepstack_position_policy", "reserved_text_prefix_v1"),
            text_input_deepstack_geometry=tuple(tuple(item) for item in get("text_input_deepstack_geometry", ())),
        )
    if get("vlm_image_input", False):
        result.update(vlm_image_input=True, vlm_image_update_hz=get("vlm_image_update_hz", 1 / 3.2),
                      vlm_image_age_embedding=get("vlm_image_age_embedding", False),
                      vlm_image_size=get("vlm_image_size", 256))
    return result


def assert_cotrain_model_config_compatible(source: Any, target: Any) -> None:
    """Bind source labels, cadence, and TRAIN statistics for resume/evaluation."""
    get = lambda config: config.get("cotrain_contract") if isinstance(config, dict) else getattr(config, "cotrain_contract", None)
    saved, requested = get(source), get(target)
    if saved is None and requested is None:
        return
    if saved is None or requested is None:
        raise ValueError("cotrain checkpoint contract mismatch: source contract is missing")
    from prism_astribot_vla.data.datasets.cotrain_contract import assert_cotrain_contract_compatible
    assert_cotrain_contract_compatible(saved, requested)
    from prism_astribot_vla.modeling.action_head_config import normalize_action_heads
    read = lambda config, key, default: config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)
    if (normalize_action_heads(read(source, "action_heads", None)) != normalize_action_heads(read(target, "action_heads", None))
            or read(source, "action_primary_head", "") != read(target, "action_primary_head", "")):
        raise ValueError("cotrain checkpoint action_heads mismatch; frame-specific decoders cannot be reinterpreted")


def assert_instruction_encoder_config_compatible(
    source: dict[str, Any], target: Any, *, allow_new_encoder: bool = False,
) -> bool:
    """Allow a fresh adapter only for explicit weights-only initialization.

    Return whether this is a legacy initialization with a newly enabled encoder.
    Equal-shaped tensors alone cannot detect a changed tokenizer or revision.
    """
    original, requested = instruction_encoder_config(source), instruction_encoder_config(target)
    introducing = not original["instruction_encoder_model_name"] and bool(requested["instruction_encoder_model_name"])
    if bool(original.get("text_input_deepstack")) != bool(requested.get("text_input_deepstack")):
        raise ValueError("Text input deepstack checkpoint architecture mismatch")
    if allow_new_encoder and introducing:
        return True
    if not original["instruction_encoder_model_name"] and not requested["instruction_encoder_model_name"]:
        return False
    if original != requested:
        raise ValueError(
            "Instruction encoder checkpoint contract mismatch: "
            f"checkpoint={original}, requested={requested}"
        )
    return False


def assert_full_checkpoint_compatible(incompatible: Any, checkpoint: Path) -> None:
    """Reject missing or unexpected parameters in a full model checkpoint."""
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Full PRISM-VLA checkpoint is incompatible with this code/worktree: "
            f"checkpoint={checkpoint}, missing={missing}, unexpected={unexpected}"
        )
