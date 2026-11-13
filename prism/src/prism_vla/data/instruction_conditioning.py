"""Shared setup-conditioned text for dataset loading and policy inference.

Setup values come from explicit source/episode annotations or a caller's
deployment setup. Neither frame rate nor tracking validity supplies speed,
motion quality, or the number of physical end effectors.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


INSTRUCTION_CONDITIONING_VERSION = 1
CONDITIONING_FIELDS = (
    "task instruction",
    "subtask instruction",
    "end effector type",
    "end effector number",
    "embodiment",
    "setting",
    "speed",
    "motion quality",
)
_FIELD_ALIASES = {key.replace(" ", "_"): key for key in CONDITIONING_FIELDS}
_CATEGORIES = {
    "end effector type": frozenset({"hand", "gripper"}),
    "end effector number": frozenset({"single", "dual", "none"}),
    "embodiment": frozenset({"human", "robot"}),
    "speed": frozenset({"fast", "moderate", "slow"}),
    "motion quality": frozenset({"good", "normal", "bad"}),
}


class InstructionOnlyText(str):
    """Explicit training augmentation: preserve task text through JSON collators.

    The marker avoids guessing from the text itself: a literal task beginning
    with ``{`` is still plain text. Ordinary strings retain legacy formatting.
    """


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate instruction conditioning field: {key!r}")
        result[key] = value
    return result


def normalize_instruction_format(value: object) -> str:
    """Validate the checkpoint-owned text format without guessing aliases."""
    if not isinstance(value, str) or value not in {"plain", "json"}:
        raise ValueError("instruction_format must be 'plain' or 'json'")
    return value


def validate_conditioning_metadata(
    metadata: Mapping[str, Any] | None, *, allow_task: bool = False
) -> dict[str, str]:
    """Validate explicit annotations and omit unknown (null/empty) fields.

    Snake-case input keys are accepted for YAML/Python convenience. Output keys
    always use the user's JSON schema and canonical order. In particular, the
    end-effector count ``'none'`` is a known category and is never omitted.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Instruction conditioning metadata must be a mapping")
    result: dict[str, str] = {}
    seen: set[str] = set()
    for name, raw in metadata.items():
        key = _FIELD_ALIASES.get(name, name) if isinstance(name, str) else name
        if key not in CONDITIONING_FIELDS:
            raise ValueError(f"Unknown instruction conditioning field: {name!r}")
        if key in seen:
            raise ValueError(f"Duplicate instruction conditioning field: {key!r}")
        seen.add(key)
        if key in {"task instruction", "subtask instruction"} and not allow_task:
            raise ValueError("Task instruction must come from the sample, not setup metadata")
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise ValueError(f"Instruction conditioning field {key!r} must be a string or null")
        value = raw.strip()
        if not value:
            continue
        if key in _CATEGORIES and value not in _CATEGORIES[key]:
            raise ValueError(
                f"Invalid {key!r}: {value!r}; expected one of {sorted(_CATEGORIES[key])}"
            )
        result[key] = value
    return {key: result[key] for key in CONDITIONING_FIELDS if key in result}


def normalize_instruction_conditioning_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve formatting and explicit source defaults, with no inferred labels."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("data.instruction_conditioning must be a mapping")
    unknown = set(config) - {"format", "defaults", "by_source", "include_subtask_instruction", "instruction_only_probability"}
    if unknown:
        raise ValueError(f"Unknown instruction_conditioning settings: {sorted(unknown)}")
    instruction_format = normalize_instruction_format(config.get("format", "plain"))
    probability = config.get("instruction_only_probability", 0.0)
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0.0 <= probability <= 1.0):
        raise ValueError("instruction_only_probability must be a finite number in [0, 1]")
    if probability and instruction_format != "json":
        raise ValueError("instruction_only_probability requires instruction_conditioning.format=json")
    include_subtask = config.get("include_subtask_instruction", False)
    if type(include_subtask) is not bool:
        raise ValueError("include_subtask_instruction must be a boolean")
    if include_subtask and instruction_format != "json":
        raise ValueError("include_subtask_instruction requires instruction_conditioning.format=json")
    defaults = validate_conditioning_metadata(config.get("defaults"))
    raw_sources = config.get("by_source", {})
    if not isinstance(raw_sources, Mapping):
        raise ValueError("instruction_conditioning.by_source must be a mapping")
    by_source = {}
    for source, values in raw_sources.items():
        if not isinstance(source, str) or not source.strip() or source != source.strip():
            raise ValueError("Instruction conditioning source names must be nonempty, unpadded strings")
        validated = validate_conditioning_metadata(values)
        # Preserve explicit omissions in source overrides, so a source can
        # remove a global default rather than silently inheriting it.
        by_source[source] = {
            _FIELD_ALIASES.get(key, key): validated.get(_FIELD_ALIASES.get(key, key))
            for key in (values or {})
        }
    if instruction_format == "plain" and (defaults or by_source):
        raise ValueError("Setup metadata requires instruction_conditioning.format=json")
    result = {"format": instruction_format, "defaults": defaults, "by_source": by_source}
    if include_subtask:
        result["include_subtask_instruction"] = True
    if probability:
        result["instruction_only_probability"] = float(probability)
    return result


def merge_conditioning_metadata(*layers: Mapping[str, Any] | None) -> dict[str, str]:
    """Merge defaults, source, and episode annotations, honoring omissions."""
    result: dict[str, str] = {}
    for layer in layers:
        validated = validate_conditioning_metadata(layer)
        for key in layer or {}:
            result.pop(_FIELD_ALIASES.get(key, key), None)
        result.update(validated)
    return {key: result[key] for key in CONDITIONING_FIELDS if key in result}


def _instruction_payload(instruction: object) -> Mapping[str, Any]:
    if isinstance(instruction, Mapping):
        return instruction
    if isinstance(instruction, str):
        if instruction.lstrip().startswith("{"):
            try:
                return json.loads(instruction, object_pairs_hook=_unique_json_object)
            except (ValueError, TypeError) as error:
                raise ValueError("Malformed JSON instruction conditioning") from error
        return {"task instruction": instruction}
    if instruction is None:
        return {}
    raise ValueError("JSON instructions require task text or a conditioning mapping")


def prepare_source_instruction(
    instruction: object, *, source: str, instruction_format: str = "plain",
    include_subtask_instruction: bool = False,
) -> object:
    """Separate a known source annotation envelope from conditioning JSON.

    Xperience10M stores its caption as ``config`` plus ordered ``segments``.
    The release also has envelopes with an optional ``global_summary`` field.
    Only these known source envelopes are accepted here, and only explicit
    nonempty string ``Sub Task`` labels enter the prompt. Configuration,
    timestamps, API usage, objects, and current-action descriptions stay out of
    conditioning. Repeated task labels are kept once in first-seen order.

    JSON results are explicit schema mappings so literal task text beginning
    with ``{`` is not reparsed as a conditioning object after augmentation.
    Other serialized objects retain strict schema validation. Plain mode keeps
    the legacy text unchanged, including annotation-envelope text.
    """
    if normalize_instruction_format(instruction_format) == "plain":
        return instruction
    payload = _instruction_payload(instruction)
    if (source == "xperience10m" and isinstance(payload, Mapping)
            and set(payload) in ({"config", "segments"},
                                 {"config", "segments", "global_summary"})):
        segments = payload["segments"]
        if (not isinstance(payload["config"], Mapping)
                or not isinstance(segments, list)
                or any(not isinstance(segment, Mapping) for segment in segments)):
            raise ValueError("Malformed Xperience10M caption envelope")
        if include_subtask_instruction:
            # A segment label is only known when its interval is observed. Do
            # not leak all future segment labels into the global task prompt.
            summary = payload.get("global_summary")
            return {"task instruction": summary.strip()} if isinstance(summary, str) and summary.strip() else {}
        tasks: dict[str, None] = {}
        for segment in segments:
            label = segment.get("Sub Task")
            if isinstance(label, str) and label.strip():
                tasks.setdefault(label.strip(), None)
        return {"task instruction": "\n".join(tasks)} if tasks else {}
    validate_conditioning_metadata(payload, allow_task=True)
    return payload


def format_instruction(
    instruction: object,
    *,
    instruction_format: str = "plain",
    metadata: Mapping[str, Any] | None = None,
    include_subtask_instruction: bool = False,
) -> str:
    """Serialize one deterministic prompt shared by training and inference.

    JSON mode accepts raw task text, an already serialized schema object, or a
    schema mapping. Augment raw task text before calling this function. Explicit
    fields in a complete prompt override supplied setup defaults. Empty/unknown
    fields are omitted rather than represented as invented labels.
    """
    instruction_format = normalize_instruction_format(instruction_format)
    if isinstance(instruction, InstructionOnlyText):
        return str(instruction)
    if instruction_format == "plain":
        # Preserve the legacy collator's str conversion and byte-for-byte text.
        return str(instruction)
    setup = validate_conditioning_metadata(metadata)
    raw_payload = _instruction_payload(instruction)
    payload = validate_conditioning_metadata(raw_payload, allow_task=True)
    # An explicitly unknown per-request value also removes a source default.
    for key in raw_payload:
        setup.pop(_FIELD_ALIASES.get(key, key), None)
    setup.update(payload)
    # Preserve the explicit empty field through repeated canonicalization by
    # dataset, collator and inference. Other unknown fields remain omitted.
    has_subtask = any(_FIELD_ALIASES.get(key, key) == "subtask instruction" for key in raw_payload)
    if include_subtask_instruction or has_subtask:
        setup.setdefault("subtask instruction", "")
    ordered = {key: setup[key] for key in CONDITIONING_FIELDS if key in setup}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def tokenize_instructions(
    tokenizer: Any,
    instructions: Sequence[object],
    *,
    max_inst_len: int,
    instruction_format: str = "plain",
    include_subtask_instruction: bool = False,
) -> Any:
    """Tokenize complete prompts; JSON overflow always fails before truncation.

    Plain mode retains the legacy tokenizer call. JSON mode checks the complete
    tokenized object before padding, then pads those same IDs: it neither cuts
    JSON syntax nor discards setup fields to make a sample fit.
    """
    if type(max_inst_len) is not int or max_inst_len < 1:
        raise ValueError("max_inst_len must be a positive integer")
    instruction_format = normalize_instruction_format(instruction_format)
    if isinstance(instructions, (str, bytes)) or not isinstance(instructions, Sequence):
        raise ValueError("instructions must be a sequence of individual prompts")
    texts = [format_instruction(value, instruction_format=instruction_format,
                                include_subtask_instruction=include_subtask_instruction)
             for value in instructions]
    if instruction_format == "plain":
        return tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_inst_len,
            return_tensors="pt", return_attention_mask=True,
        )
    encoded = tokenizer(texts, padding=False, truncation=False, return_attention_mask=True)
    token_ids = encoded["input_ids"]
    if len(token_ids) != len(texts):
        raise ValueError("Tokenizer returned a different number of instruction rows")
    overflow = [(index, len(row)) for index, row in enumerate(token_ids) if len(row) > max_inst_len]
    if overflow:
        required = max(length for _, length in overflow)
        raise ValueError(
            f"JSON instruction exceeds max_inst_len={max_inst_len}: rows={overflow}; "
            f"requires at least {required} tokens. Increase model.max_inst_len; "
            "structured conditioning is never silently truncated."
        )
    return tokenizer.pad(
        encoded, padding="max_length", max_length=max_inst_len, return_tensors="pt",
    )
