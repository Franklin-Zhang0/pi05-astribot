"""Pure metadata helpers for camera-major compact-token inventories."""
from __future__ import annotations


def camera_token_counts(value: int | tuple[int, ...] | list[int], cameras: int) -> tuple[int, ...]:
    """Resolve a shared count or one positive count for each stable camera slot."""
    if type(cameras) is not int or cameras <= 0:
        raise ValueError("Camera inventory must contain a positive integer number of cameras")
    if type(value) is int:
        counts = (value,) * cameras
    elif isinstance(value, (tuple, list)):
        counts = tuple(value)
        if len(counts) != cameras:
            raise ValueError("camera_compact_tokens_per_camera must contain one count per camera")
    else:
        raise ValueError("camera_compact_tokens_per_camera must be a positive integer or per-camera tuple/list")
    if any(type(count) is not int or count <= 0 for count in counts):
        raise ValueError("camera_compact_tokens_per_camera counts must be positive integers")
    return counts
