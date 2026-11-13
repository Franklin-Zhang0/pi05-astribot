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
_NEW_NAMESPACES = ('engram_action_head.', 'engram_proprio_encoder.')
_TRUNK_ROOTS = ('blocks.', 'time_proj.', 'condition_role_embeddings', 'final_layer.norm_final.', 'final_layer.adaLN_modulation.')

def validate_engram_native_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the opt-in model extension without choosing a run recipe."""
    fields = {'enabled', 'source_head', 'physical_horizon', 'route_names'}
    optional = {'mode', 'robot_eef_head', 'robot_eef_frame_provenance'}
    if not isinstance(value, Mapping) or not fields <= set(value) or set(value) - fields - optional or (value.get('enabled') is not True):
        raise ValueError(f'engram_native must explicitly enable these fields: {sorted(fields)}')
    source = value['source_head']
    if not isinstance(source, str) or re.fullmatch('[A-Za-z][A-Za-z0-9_-]*', source) is None:
        raise ValueError('engram_native.source_head must explicitly select a pretrained decoder')
    horizon = value['physical_horizon']
    if type(horizon) is not int or horizon < 1:
        raise ValueError('engram_native.physical_horizon must be a positive integer')
    routes = ('dex_w1_gripper', 'dex_w1_linker_l6', 'dex_w1_linker_l20', 'galbot_inspire_rh56dfx_hand')
    if not isinstance(value['route_names'], (tuple, list)) or tuple(value['route_names']) != routes:
        raise ValueError('engram_native.route_names must retain the canonical numeric route ordering')
    result = {'enabled': True, 'source_head': source, 'physical_horizon': horizon, 'route_names': list(routes)}
    mode = value.get('mode', 'joint')
    if mode not in {'joint', 'joint_and_robot_eef'}:
        raise ValueError('Unknown native Engram action mode')
    if 'mode' in value:
        result['mode'] = mode
    if mode == 'joint_and_robot_eef':
        head = value.get('robot_eef_head')
        if not isinstance(head, str) or re.fullmatch('[A-Za-z][A-Za-z0-9_-]*', head) is None:
            raise ValueError('Hybrid Engram requires an explicit pretrained robot_eef_head')
        result['robot_eef_head'] = head
        if 'robot_eef_frame_provenance' in value:
            provenance = value['robot_eef_frame_provenance']
            if not isinstance(provenance, Mapping) or not provenance:
                raise ValueError('Robot EEF frame provenance must be a nonempty mapping')
            result['robot_eef_frame_provenance'] = _json_value(dict(provenance))
    elif 'robot_eef_head' in value or 'robot_eef_frame_provenance' in value:
        raise ValueError('Robot EEF configuration requires joint_and_robot_eef mode')
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
