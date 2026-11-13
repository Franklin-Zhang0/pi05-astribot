"""Serializable contracts for jointly trained action heads."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re

from prism_vla.data.datasets.ego_action_representation import normalize_action_representation


def normalize_action_heads(value: Sequence[Mapping] | None) -> tuple[dict, ...]:
    """Validate named heads without allowing a private VLM condition layout.

    Head overrides are backend constructor options, e.g. ``d_model`` and
    ``n_layers``. Horizons, dimensions and condition slots belong to the shared
    model contract and cannot be overridden by a head.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError("action_heads must be an ordered list of named head specifications")
    names, result = set(), []
    allowed = {"name", "type", "loss_weight", "action_representation", "overrides"}
    shared = {"cond_dim", "n_cond_tokens", "action_dim", "action_horizon",
              "action_token_types", "action_tokens_per_step", "robot_action_layout", "proprio_as_action_token"}
    aliases = {"dit": "flow_matching", "dit_flow_matching": "flow_matching",
               "fast+": "fast_ar", "fast_plus": "fast_ar", "oft": "oft",
               "patchwam": "act_patch"}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) - allowed:
            raise ValueError(f"Invalid action head specification; allowed keys: {sorted(allowed)}")
        if not {"name", "type", "action_representation"} <= set(raw):
            raise ValueError("Each action head requires name, type and action_representation")
        name = str(raw["name"])
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError(f"Action head names must be unique identifiers, got {name!r}")
        names.add(name)
        kind = str(raw["type"]).strip().lower()
        kind = aliases.get(kind, kind)
        if kind not in {"flow_matching", "oft", "fast_ar", "act_patch", "pi05"}:
            raise ValueError(f"Unsupported action head type: {kind}")
        weight = raw.get("loss_weight", 1.0)
        if isinstance(weight, bool) or not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError(f"Head {name}: loss_weight must be finite and positive")
        representation = dict(raw["action_representation"])
        hand = dict(representation.get("hand", {}))
        if hand.get("frame") == "camera" and representation.get("frame") == "camera":
            # Keep the established camera-parent alias stable for checkpoints.
            # With a different parent, camera is an independent hand reference.
            hand["frame"] = "pose_frame"
            representation["hand"] = hand
        overrides = raw.get("overrides") or {}
        if not isinstance(overrides, Mapping) or shared & set(overrides):
            raise ValueError(f"Head {name}: overrides cannot replace shared action/condition dimensions")
        result.append({"name": name, "type": kind, "loss_weight": float(weight),
                       "action_representation": normalize_action_representation(representation),
                       "overrides": dict(overrides)})
    if len(result) > 1 and any(spec["type"] == "act_patch" for spec in result):
        raise ValueError("act_patch requires a singleton action head; mixed or multiple fixed-codec heads are not supported")
    return tuple(result)
