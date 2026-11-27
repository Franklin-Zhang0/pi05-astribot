"""Blockwise frame KDA with batched readout and optional affine prefix evaluation."""
from __future__ import annotations

import torch


def _deltas_and_decay(key: torch.Tensor, value: torch.Tensor, strength: torch.Tensor,
                      decay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one frame's normalized keys and strengths into its affine state map."""
    # The previous state is decayed along its K axis before the mean delta update.
    # Matrices are in [..., K, K] and additive updates in [..., K, V].
    weighted = strength * torch.cat((key, value), dim=-1)
    kv = torch.matmul(key.transpose(-1, -2), weighted).to(decay.dtype)
    gram, update = kv.split((key.shape[-1], value.shape[-1]), dim=-1)
    eye = torch.eye(key.shape[-1], device=key.device, dtype=decay.dtype)
    matrix = (eye - gram) * decay.transpose(-1, -2)
    return matrix, update


def _prefix_states(matrix: torch.Tensor, offset: torch.Tensor,
                   initial: torch.Tensor) -> torch.Tensor:
    """Evaluate all inclusive affine prefixes in logarithmic depth and linear work."""
    n = matrix.shape[1]
    if n == 1:
        return torch.matmul(matrix, initial[:, None]) + offset
    pairs = n // 2
    left, right = matrix[:, :2*pairs:2], matrix[:, 1::2]
    # Compose adjacent affine maps. The K and V columns share the left operand.
    composed = torch.matmul(right, torch.cat((left, offset[:, :2*pairs:2]), dim=-1))
    composed_matrix, composed_offset = composed.split((matrix.shape[-1], offset.shape[-1]), dim=-1)
    right_states = _prefix_states(composed_matrix, composed_offset + offset[:, 1::2], initial)
    previous = torch.cat((initial[:, None], right_states), dim=1)[:, :n-pairs]
    left_states = torch.matmul(matrix[:, ::2], previous) + offset[:, ::2]
    result = torch.stack((left_states[:, :pairs], right_states), dim=2).flatten(1,2)
    return torch.cat((result, left_states[:, -1:]), dim=1) if n % 2 else result


def _batched_readout(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     decay: torch.Tensor, strength: torch.Tensor, initial: torch.Tensor,
                     scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Fully preserve the original sequence of state operations and cast points.
    # Only read-only query projections for completed states are batched.
    states = []
    state = initial
    for k, v, gate, weight in zip(key.unbind(1), value.unbind(1), decay.unbind(1), strength.unbind(1)):
        decayed = state * gate
        prediction = torch.matmul(k, decayed)
        correction = torch.matmul(k.transpose(-1, -2), (v - prediction) * weight)
        state = decayed + correction
        states.append(state)
    output = torch.matmul(query, torch.stack(states, dim=1)) * scale
    return output.transpose(2, 3), state


def scan_prepared_frames(query, key, value, decay, strength, initial, scale, mode):
    if mode == "readout":
        return _batched_readout(query,key,value,decay,strength,initial,scale)
    if mode != "affine":
        raise ValueError(f"Unsupported KDA scan mode {mode!r}")
    matrix, offset = _deltas_and_decay(key,value,strength,decay)
    # Decay and recurrent state are FP32 in the existing recurrence. Do not
    # allow BF16 autocast to round the composed identity/decay matrices.
    with torch.autocast(device_type=initial.device.type, enabled=False):
        states = _prefix_states(matrix,offset,initial)
    output = torch.matmul(query, states) * scale
    return output.transpose(2, 3), states[:, -1]
