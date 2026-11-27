"""Strict, append-only initialization of native Engram action interfaces.

This is weights-only initialization, never optimizer resume or a reinterpretation
of the pretrained geometric action heads. Decoder and physical horizon choices
belong to the caller's explicit ``engram_native`` configuration.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

import torch
from torch import nn


_NEW_NAMESPACES = ("engram_action_head.", "engram_proprio_encoder.")
_TRUNK_ROOTS = ("blocks.", "time_proj.", "condition_role_embeddings", "final_layer.norm_final.",
                "final_layer.adaLN_modulation.")


def validate_engram_native_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the opt-in model extension without choosing a run recipe."""
    fields = {"enabled", "source_head", "physical_horizon", "route_names"}
    optional = {"mode", "robot_eef_head", "robot_eef_frame_provenance"}
    if (not isinstance(value, Mapping) or not fields <= set(value)
            or set(value) - fields - optional or value.get("enabled") is not True):
        raise ValueError(f"engram_native must explicitly enable these fields: {sorted(fields)}")
    source = value["source_head"]
    if not isinstance(source, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", source) is None:
        raise ValueError("engram_native.source_head must explicitly select a pretrained decoder")
    horizon = value["physical_horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("engram_native.physical_horizon must be a positive integer")
    routes = ("dex_w1_gripper", "dex_w1_linker_l6", "dex_w1_linker_l20", "galbot_inspire_rh56dfx_hand")
    if not isinstance(value["route_names"], (tuple, list)) or tuple(value["route_names"]) != routes:
        raise ValueError("engram_native.route_names must retain the canonical numeric route ordering")
    result = {"enabled": True, "source_head": source, "physical_horizon": horizon, "route_names": list(routes)}
    mode = value.get("mode", "joint")
    if mode not in {"joint", "joint_and_robot_eef"}:
        raise ValueError("Unknown native Engram action mode")
    if "mode" in value:
        result["mode"] = mode
    if mode == "joint_and_robot_eef":
        head = value.get("robot_eef_head")
        if not isinstance(head, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", head) is None:
            raise ValueError("Hybrid Engram requires an explicit pretrained robot_eef_head")
        result["robot_eef_head"] = head
        if "robot_eef_frame_provenance" in value:
            provenance = value["robot_eef_frame_provenance"]
            if not isinstance(provenance, Mapping) or not provenance:
                raise ValueError("Robot EEF frame provenance must be a nonempty mapping")
            result["robot_eef_frame_provenance"] = _json_value(dict(provenance))
    elif "robot_eef_head" in value or "robot_eef_frame_provenance" in value:
        raise ValueError("Robot EEF configuration requires joint_and_robot_eef mode")
    return result


@dataclass(frozen=True)
class EngramInitializationPlan:
    """Declare every new tensor and the optional pretrained decoder trunk.

    ``new_tensor_names`` is the exact inventory of newly registered parameters
    and persistent buffers, not a permissive missing-key prefix. New joint
    input/output and proprio projections retain constructor initialization.
    Only explicitly compatible hidden/time decoder tensors may be cloned.
    """

    new_tensor_names: tuple[str, ...]
    source_head: str | None = None
    trunk_tensor_map: tuple[tuple[str, str], ...] = ()
    allow_lossless_fp32_promotion: bool = False


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def assert_engram_inherited_config(source: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    """Keep every inherited model field; permit only the native extension.

    Compare JSON values so serialized lists and in-memory tuples have the same
    meaning. The sole text-policy addition is an explicitly declared 50/50
    instruction-only mixture; its existing schema/defaults remain unchanged.
    Native horizon and routing belong to the extension, not inherited fields.
    """
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise TypeError("Engram initialization requires source and target model config mappings")
    if "engram_native" in source:
        raise ValueError("Expected a geometric pretraining source, not an Engram resume")
    if not isinstance(target.get("engram_native"), Mapping) or not target["engram_native"]:
        raise ValueError("Engram initialization requires an explicit nonempty engram_native config")
    inherited = {key: value for key, value in target.items() if key != "engram_native"}
    original, requested = _json_value(dict(source)), _json_value(inherited)
    # Older public checkpoints predate default-off feature fields. Their
    # absence is equivalent only to the exact disabled-mode defaults, never to
    # activating or changing the token-reduction architecture.
    if original.get("query_design", "legacy") != "token_reduction" and requested.get("query_design", "legacy") != "token_reduction":
        from dataclasses import fields
        from prism_astribot_vla.modeling.config import PrismVLAConfig
        for field in fields(PrismVLAConfig):
            if (field.name.startswith("token_reduction_") and field.name not in original
                    and field.name in requested and requested[field.name] == _json_value(field.default)):
                requested.pop(field.name)
    old_text, new_text = original.get("instruction_conditioning"), requested.get("instruction_conditioning")
    if (isinstance(old_text, dict) and isinstance(new_text, dict)
            and "instruction_only_probability" not in old_text
            and "instruction_only_probability" in new_text):
        probability = new_text["instruction_only_probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or probability != 0.5:
            raise ValueError("Engram initialization permits only the declared 50/50 instruction mixture")
        new_text.pop("instruction_only_probability")
    changed = sorted(key for key in original.keys() | requested.keys()
                     if key not in original or key not in requested or original[key] != requested[key])
    if changed:
        raise ValueError(f"Engram initialization changed inherited model configuration: {changed}")


def build_engram_trunk_map(
    source_names: Sequence[str], target_names: Sequence[str], *, source_head: str,
) -> tuple[tuple[str, str], ...]:
    """Match existing native decoder trunk tensors without copying geometric IO.

    This does not choose a decoder. Instantiate the explicitly selected native
    decoder first; a different topology fails the complete trunk check below.
    """
    if not isinstance(source_head, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", source_head) is None:
        raise ValueError("A valid explicitly selected source_head is required")
    prefix = f"idm.heads.{source_head}."
    source = set(source_names)
    result = []
    for name in target_names:
        if name.startswith("engram_action_head."):
            suffix = name[len("engram_action_head."):]
            if suffix.startswith(_TRUNK_ROOTS):
                original = prefix + suffix
                if original not in source:
                    raise ValueError(f"Selected pretrained decoder lacks native trunk tensor: {original}")
                result.append((name, original))
    if not result:
        raise ValueError("Native decoder has no compatible time/transformer trunk tensors")
    return tuple(sorted(result))


def initialize_engram_from_pretrained(
    model: nn.Module, source_state: Mapping[str, torch.Tensor], *,
    source_config: Mapping[str, Any], target_config: Mapping[str, Any],
    plan: EngramInitializationPlan,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the complete transfer before loading any model tensor.

    Existing source tensors are retained by exact name, shape and dtype. An
    explicit policy can promote BF16/FP16 to FP32 without losing any value. Only
    the declared new native modules can be absent. A decoder trunk clone is
    optional, explicit, complete for the target trunk and dimension preserving;
    all native joint IO/proprio projections stay freshly initialized.
    """
    assert_engram_inherited_config(source_config, target_config)
    if not isinstance(source_state, Mapping) or not source_state:
        raise ValueError("Pretrained state must be a nonempty tensor mapping")
    target = model.state_dict()
    if any(not isinstance(name, str) or not torch.is_tensor(value)
           for name, value in source_state.items()):
        raise TypeError("Pretrained state must contain only named tensors")
    if type(plan.allow_lossless_fp32_promotion) is not bool:
        raise ValueError("Lossless FP32 promotion must be an explicit boolean")
    if source_manifest is not None:
        _validate_manifest_state(model, source_state, source_manifest)
    declared = plan.new_tensor_names
    if (not isinstance(declared, tuple) or not declared or len(set(declared)) != len(declared)
            or any(not isinstance(name, str) or not name.startswith(_NEW_NAMESPACES) for name in declared)):
        raise ValueError("Declare unique new tensor names exclusively in the native Engram modules")
    if set(source_state) & set(declared):
        raise ValueError("New native tensor declaration overlaps inherited pretrained state")
    unexpected = sorted(set(source_state) - set(target))
    missing = sorted(set(target) - set(source_state) - set(declared))
    nonexistent = sorted(set(declared) - set(target))
    inherited_native = sorted(name for name in source_state if name.startswith(_NEW_NAMESPACES))
    if unexpected or missing or nonexistent or inherited_native:
        raise ValueError("Engram pretrained tensor inventory mismatch: "
                         f"discarded={unexpected}, undeclared_new={missing}, "
                         f"nonexistent_new={nonexistent}, inherited_native={inherited_native}")
    promoted = []

    def compatible(name, value):
        expected = target[name]
        if value.shape != expected.shape:
            return False
        if value.dtype == expected.dtype:
            return True
        if (plan.allow_lossless_fp32_promotion and value.dtype in (torch.bfloat16, torch.float16)
                and expected.dtype == torch.float32
                and torch.equal(value, value.float().to(value.dtype))):
            promoted.append(name)
            return True
        return False

    mismatched = [name for name, value in source_state.items() if not compatible(name, value)]
    if mismatched:
        raise ValueError(f"Engram inherited tensor shape/dtype mismatch: {sorted(mismatched)}")
    pairs = plan.trunk_tensor_map
    if not isinstance(pairs, tuple) or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
        raise ValueError("Decoder trunk mapping must be explicit target/source pairs")
    copies = dict(pairs)
    if len(copies) != len(pairs):
        raise ValueError("Duplicate native decoder trunk destination")
    if plan.source_head is None:
        if copies:
            raise ValueError("Trunk copies require an explicitly selected source_head")
    else:
        expected = dict(build_engram_trunk_map(tuple(source_state), declared, source_head=plan.source_head))
        if copies != expected:
            raise ValueError("Native decoder trunk map must exactly match the selected pretrained head; geometric IO cannot be copied")
    for destination, original in copies.items():
        if destination not in declared or original not in source_state:
            raise ValueError("Decoder trunk copy must reference declared new tensors and existing source tensors")
        if not compatible(destination, source_state[original]):
            raise ValueError(f"Engram decoder trunk shape/dtype mismatch: {original} -> {destination}")
    # All semantic, inventory and shape checks above run before the first copy.
    # Clone new entries to prevent aliasing a constructor tensor during load.
    complete = {**source_state, **{name: target[name].clone() for name in declared}}
    complete.update({destination: source_state[original].clone() for destination, original in copies.items()})
    model.load_state_dict(complete, strict=True)
    return {
        "schema": "prism_engram_pretrained_initialization_v1",
        "inherited_tensor_count": len(source_state),
        "new_tensor_names": sorted(declared),
        "source_head": plan.source_head,
        "trunk_tensor_map": [list(pair) for pair in sorted(copies.items())],
        "fresh_tensor_names": sorted(set(declared) - copies.keys()),
        "lossless_fp32_promoted_tensor_names": sorted(promoted),
        "source_manifest_validated": source_manifest is not None,
        "optimizer_restored": False,
    }


def _validate_manifest_state(model, source_state, manifest):
    """Bind saved parameter values and persistent buffer slots, including None."""
    topology = manifest.get("topology", {}) if isinstance(manifest, Mapping) else {}
    required = {"parameter_fqns", "parameter_storage_dtypes", "persistent_buffer_slots",
                "persistent_buffer_tensor_fqns"}
    if not isinstance(topology, Mapping) or not required <= topology.keys():
        raise ValueError("Pretrained manifest lacks parameter/persistent-buffer topology")
    parameters, buffers, slots = (topology[key] for key in (
        "parameter_fqns", "persistent_buffer_tensor_fqns", "persistent_buffer_slots"))
    for names in (parameters, buffers, slots):
        if (not isinstance(names, (tuple, list)) or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Pretrained manifest tensor/slot names must be unique lists")
    if set(parameters) & set(buffers) or not set(buffers) <= set(slots):
        raise ValueError("Pretrained manifest parameter/buffer inventories overlap or have absent slots")
    if set(source_state) != set(parameters) | set(buffers):
        raise ValueError("Pretrained payload differs from manifest parameter/buffer inventory")
    dtypes = topology["parameter_storage_dtypes"]
    if not isinstance(dtypes, Mapping) or set(dtypes) != set(parameters):
        raise ValueError("Pretrained manifest parameter dtype inventory differs")
    for name in parameters:
        if str(source_state[name].dtype).removeprefix("torch.") != dtypes[name]:
            raise ValueError(f"Pretrained payload storage dtype differs from manifest: {name}")
    live_slots = {}
    for prefix, module in model.named_modules():
        for name, value in module._buffers.items():
            if name not in module._non_persistent_buffers_set:
                qualified = f"{prefix}.{name}" if prefix else name
                if not qualified.startswith(_NEW_NAMESPACES):
                    live_slots[qualified] = value
    if set(live_slots) != set(slots):
        raise ValueError("Inherited persistent buffer registration/None slots differ from manifest")
    # A saved tensor cannot silently become an unregistered/None constructor
    # slot; loading it would otherwise be unexpected or ignored by PyTorch.
    if {name for name, value in live_slots.items() if value is not None} != set(buffers):
        raise ValueError("Inherited persistent buffer tensor/None slots differ from manifest")
