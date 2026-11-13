"""Command-first labels and checkpoint-owned contracts for source co-training.

EEF geometry is common to a head; source controllers, actuator auxiliaries and
TRAIN normalizers remain explicit. A missing command conversion is never a
license to substitute observed future poses for an available command.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import json
import math
from typing import Any
from prism_vla.data.datasets.ego_action_representation import normalize_action_representation
COTRAIN_CONTRACT_SCHEMA = 'cotrain_action_contract_v1'
COMMAND_FIRST_POLICY = 'command_if_available_else_pose_change_v1'
DEFAULT_ACTION_REPRESENTATION = {'frame': 'camera', 'frame_anchor': 'chunk_start', 'target': 'chunk_offset', 'hand': {'frame': 'per_frame_local', 'target': 'chunk_offset'}}

def default_cotrain_action_heads() -> list[dict[str, Any]]:
    """Independent frame experts; equal nominal weights are configurable."""
    specs = (('eef_local', 'self', 'previous_step', 'step_delta'), ('camera', 'camera', 'chunk_start', 'chunk_offset'), ('robot', 'robot', 'previous_step', 'step_delta'))
    return [{'name': name, 'type': 'flow_matching', 'loss_weight': 1.0, 'action_representation': {'frame': frame, 'frame_anchor': anchor, 'target': target, 'hand': {'frame': 'per_frame_local', 'target': target}}} for name, frame, anchor, target in specs]
DEFAULT_ACTION_HEADS = default_cotrain_action_heads()

def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Co-training contracts must contain finite JSON values') from error

def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    return float(value)

def validate_cotrain_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate source routing without pooling, refitting, or changing statistics."""
    if not isinstance(metadata, Mapping):
        raise ValueError('Co-training requires an explicit source contract mapping')
    result = _json_copy(dict(metadata))
    if result.get('schema') != COTRAIN_CONTRACT_SCHEMA or result.get('target_policy') != COMMAND_FIRST_POLICY:
        raise ValueError('Unsupported co-training schema or command-first target policy')
    for key, expected in (('action_dim', 15), ('action_tokens_per_step', 5), ('proprio_dim', 75)):
        if type(result.get(key)) is not int or result[key] != expected:
            raise ValueError(f'Co-training {key} must equal {expected}')
    horizon = result.get('horizon')
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Co-training horizon must be a positive integer')
    if result.get('rotation_representation') != 'rotation_6d' or result.get('pose_representation') != 'umi_local':
        raise ValueError('The shared co-training head requires 3D translation plus 6D EEF rotation')
    result['representation'] = normalize_action_representation(result.get('representation'))
    result['position_scale'] = _positive(result.get('position_scale'), 'position_scale')
    if result.get('cadence_policy', 'native_action_ticks_v1') != 'native_action_ticks_v1':
        raise ValueError('Unsupported co-training action cadence policy')
    result['cadence_policy'] = 'native_action_ticks_v1'
    sources = result.get('sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Co-training requires at least one explicit source')
    for name, entry in sources.items():
        if not isinstance(name, str) or not name or name != name.strip() or (not isinstance(entry, dict)):
            raise ValueError('Co-training source names and contracts must be explicit')
        origin = entry.get('label_origin')
        if origin not in {'command', 'pose_change', 'mixed', 'none'}:
            raise ValueError(f'Source {name!r} lacks a qualified label_origin')
        supervised = entry.get('action_supervision', origin != 'none')
        if type(supervised) is not bool or supervised != (origin != 'none'):
            raise ValueError(f'Source {name!r} action supervision conflicts with label_origin')
        entry['action_supervision'] = supervised
        available = entry.get('command_available', origin in {'command', 'mixed'})
        if type(available) is not bool or (available and origin == 'pose_change'):
            raise ValueError(f'Source {name!r} may not replace available commands with pose changes')
        reason = entry.get('unavailable_reason')
        if available and origin == 'none' and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f'Source {name!r} must explain why this head cannot supervise available commands')
        if origin in {'command', 'mixed'} and (not available):
            raise ValueError(f'Source {name!r} command labels require available commands')
        entry['command_available'] = available
        entry['action_hz'] = _positive(entry.get('action_hz'), f'{name}.action_hz')
        if 'representation' in entry and normalize_action_representation(entry['representation']) != result['representation']:
            raise ValueError(f'Source {name!r} action representation differs from its head')
        if entry.get('rotation_representation', 'rotation_6d') != 'rotation_6d':
            raise ValueError(f'Source {name!r} EEF rotation encoding differs from its head')
        if 'position_scale' in entry and _positive(entry['position_scale'], f'{name}.position_scale') != result['position_scale']:
            raise ValueError(f'Source {name!r} position scale differs from its head')
        normalization = entry.get('action_normalization')
        if not supervised and normalization is not None:
            raise ValueError(f'Visual-only source {name!r} must not fit fictitious action statistics')
        if normalization is None:
            entry['action_normalization'] = None
            continue
        mode = normalization.get('mode') if isinstance(normalization, Mapping) else None
        if mode == 'mean_std_v1':
            from prism_vla.data.datasets.robot_action_normalization import validate_action_normalization_metadata
            checked = validate_action_normalization_metadata(normalization)
            if checked.get('stats_horizon', horizon) != horizon:
                raise ValueError(f'Source {name!r} TRAIN statistics use a different horizon')
        elif mode == 'completed_chunk_masked_mean_std_v1':
            from prism_vla.data.datasets.ego_action_normalization import validate_ego_action_normalization
            checked = validate_ego_action_normalization(normalization)
            if checked['horizon'] != horizon or checked['representation'] != result['representation'] or checked['position_scale'] != result['position_scale']:
                raise ValueError(f'Source {name!r} TRAIN statistics use different action geometry')
            dataset = entry.get('dataset_name')
            if not isinstance(dataset, str) or dataset not in checked['by_dataset']:
                raise ValueError(f'Source {name!r} must explicitly select its original dataset statistics')
        else:
            raise ValueError(f'Source {name!r} has unsupported action normalization')
        entry['action_normalization'] = checked
    heads = result.get('heads')
    if heads is not None:
        primary = result.get('primary_head')
        if not isinstance(heads, dict) or not heads or primary not in heads:
            raise ValueError('Multi-head co-training requires a named primary head')
        validated = {}
        for name, head in heads.items():
            if not isinstance(name, str) or not name or (not isinstance(head, dict)) or ('heads' in head):
                raise ValueError('Co-training head contracts must be named, non-recursive mappings')
            validated[name] = validate_cotrain_contract(head)
            if set(validated[name]['sources']) != set(sources) or validated[name]['horizon'] != horizon:
                raise ValueError('Co-training heads must share the source cohort and action horizon')
        top = {key: value for key, value in result.items() if key not in {'heads', 'primary_head'}}
        if top != validated[primary]:
            raise ValueError('Top-level co-training contract must match the primary head')
        result['heads'] = validated
    elif 'primary_head' in result:
        raise ValueError('primary_head requires explicit co-training head contracts')
    return result

class CotrainActionDenormalizer:
    """Invert checkpoint-owned statistics independently per mixed batch row."""

    def __init__(self, metadata: Mapping[str, Any], *, head_name: str | None=None) -> None:
        checked = validate_cotrain_contract(metadata)
        if head_name is not None:
            if head_name not in checked.get('heads', {}):
                raise ValueError(f'Unknown co-training head {head_name!r}')
            checked = checked['heads'][head_name]
        self.metadata = {key: value for key, value in checked.items() if key not in {'heads', 'primary_head'}}
        self._statistics: dict[Any, Any] = {}

    def validate_source_names(self, names: Sequence[str] | None, batch_size: int) -> tuple[str, ...]:
        if not isinstance(names, (list, tuple)) or len(names) != batch_size or any((not isinstance(name, str) or not name for name in names)):
            raise ValueError('Co-training evaluation requires one source selector per batch row')
        missing = set(names) - set(self.metadata['sources'])
        if missing:
            raise ValueError(f'Checkpoint lacks co-training sources: {sorted(missing)}')
        return tuple(names)

    def _scales(self, names: Sequence[str] | None, values: Any) -> tuple[Any, Any]:
        import torch
        names = self.validate_source_names(names, values.shape[0])
        means, scales = ([], [])
        for name in names:
            key = (name, values.device)
            if key not in self._statistics:
                entry = self.metadata['sources'][name]
                normalization = entry['action_normalization']
                if normalization is None:
                    mean, scale = ([[0.0] * 15 for _ in range(5)], [[1.0] * 15 for _ in range(5)])
                else:
                    stats = normalization['by_dataset'][entry['dataset_name']] if 'by_dataset' in normalization else normalization
                    mean, scale = (stats['mean'], stats['std'])
                self._statistics[key] = (torch.tensor(mean, device=values.device, dtype=torch.float32), torch.tensor(scale, device=values.device, dtype=torch.float32))
            means.append(self._statistics[key][0])
            scales.append(self._statistics[key][1])
        return (torch.stack(means), torch.stack(scales))

    def denormalize_pair(self, predicted: Any, target: Any, *, dataset_names: Sequence[str] | None, loss_mask: Any=None) -> tuple[Any, Any]:
        import torch
        if not torch.is_tensor(predicted) or not torch.is_tensor(target) or predicted.shape != target.shape or (target.ndim < 3) or (target.shape[-2:] != (self.metadata['horizon'] * 5, 15)):
            raise ValueError('Co-training physical evaluation requires matching [B,...,H*5,15] tensors')
        if predicted.device != target.device or (loss_mask is not None and loss_mask.shape != target.shape):
            raise ValueError('Co-training evaluation devices or masks differ')
        mean, scale = self._scales(dataset_names, target)
        broadcast_shape = (target.shape[0], *(1,) * (target.ndim - 2), 5, 15)
        mean, scale = (mean.reshape(broadcast_shape), scale.reshape(broadcast_shape))
        grouped_shape = (*target.shape[:-2], self.metadata['horizon'], 5, 15)

        def inverse(values: Any) -> Any:
            result = (values.float().reshape(grouped_shape) * scale + mean).reshape(values.shape)
            return torch.where(loss_mask.to(values.device) > 0, result, 0) if loss_mask is not None else result
        return (inverse(predicted), inverse(target))
