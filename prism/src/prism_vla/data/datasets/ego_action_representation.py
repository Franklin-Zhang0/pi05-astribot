"""Independent coordinate frame, frame anchor, and target contracts for ego actions."""
from collections.abc import Mapping

def normalize_action_representation(value: Mapping) -> dict:
    """Validate a complete explicit contract and return its canonical JSON form."""
    if not isinstance(value, Mapping):
        raise ValueError('action_representation must be a mapping')
    required = {'frame', 'frame_anchor', 'target', 'hand'}
    if set(value) != required:
        raise ValueError(f'action_representation requires exactly {sorted(required)}')
    hand = value['hand']
    if not isinstance(hand, Mapping) or set(hand) != {'frame', 'target'}:
        raise ValueError('action_representation.hand requires exactly frame and target')

    def choice(raw, allowed, name):
        result = str(raw).strip().lower()
        if result not in allowed:
            raise ValueError(f'{name} must be one of {sorted(allowed)}, got {raw!r}')
        return result
    targets = {'absolute', 'chunk_offset', 'step_delta'}
    return {'frame': choice(value['frame'], {'camera', 'self', 'world', 'robot'}, 'frame'), 'frame_anchor': choice(value['frame_anchor'], {'chunk_start', 'previous_step'}, 'frame_anchor'), 'target': choice(value['target'], targets, 'target'), 'hand': {'frame': choice(hand['frame'], {'per_frame_local', 'pose_frame', 'camera'}, 'hand.frame'), 'target': choice(hand['target'], targets, 'hand.target')}}
