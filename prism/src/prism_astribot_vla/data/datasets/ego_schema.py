"""Canonical schema helpers for ego-centric dataset loaders."""

from __future__ import annotations

from typing import Optional

import numpy as np

IMAGE_KEY = "image"
LEFT_WRIST_IMAGE_KEY = "left_wrist_image"
RIGHT_WRIST_IMAGE_KEY = "right_wrist_image"

LEFT_EEF = "left_eef"
RIGHT_EEF = "right_eef"
LEFT_HAND_3D_KEYPOINTS = "left_hand_3D_keypoints"
RIGHT_HAND_3D_KEYPOINTS = "right_hand_3D_keypoints"
HEAD_MOTION = "head_motion"
CHEST_MOTION = "chest_motion"
WHOLE_BODY_KEYPOINTS = "whole_body_keypoints"

ACTION_FRAME_WORLD = "world"
ACTION_FRAME_HEAD = "head"
ACTION_FRAME_LOCAL = "local"
ACTION_FRAMES = (ACTION_FRAME_WORLD, ACTION_FRAME_HEAD, ACTION_FRAME_LOCAL)

ACTION_MODALITIES = (
    LEFT_EEF,
    RIGHT_EEF,
    LEFT_HAND_3D_KEYPOINTS,
    RIGHT_HAND_3D_KEYPOINTS,
    HEAD_MOTION,
    CHEST_MOTION,
    WHOLE_BODY_KEYPOINTS,
)

MEDIAPIPE_HAND_NUM_KEYPOINTS = 21
MEDIAPIPE_FINGERTIP_INDICES = (4, 8, 12, 16, 20)

ROTATION_REPRESENTATION_EULER = "euler"
ROTATION_REPRESENTATION_6D = "rotation_6d"
ROTATION_REPRESENTATIONS = (
    ROTATION_REPRESENTATION_EULER,
    ROTATION_REPRESENTATION_6D,
)

EGO_POSE_REPRESENTATION_STANDARD = "standard"
EGO_POSE_REPRESENTATION_UMI_LOCAL = "umi_local"
EGO_POSE_REPRESENTATIONS = (
    EGO_POSE_REPRESENTATION_STANDARD,
    EGO_POSE_REPRESENTATION_UMI_LOCAL,
)
EGO_DEFAULT_POSE_REPRESENTATION = EGO_POSE_REPRESENTATION_UMI_LOCAL
EGO_DEFAULT_ROTATION_REPRESENTATION = ROTATION_REPRESENTATION_6D
EGO_COMPACT_ACTION_TOKENS_PER_STEP = 3
EGO_SEPARATED_ACTION_TOKENS_PER_STEP = 5
EGO_DEFAULT_ACTION_TOKENS_PER_STEP = EGO_SEPARATED_ACTION_TOKENS_PER_STEP
EGO_ACTION_TOKENS_PER_STEP_OPTIONS = (
    EGO_COMPACT_ACTION_TOKENS_PER_STEP,
    EGO_SEPARATED_ACTION_TOKENS_PER_STEP,
)


def normalize_rotation_representation(value: object) -> str:
    """Return the canonical ego action/proprio rotation representation."""

    normalized = str(value or EGO_DEFAULT_ROTATION_REPRESENTATION).strip().lower()
    normalized = normalized.replace("-", "_")
    aliases = {
        "euler_xyz": ROTATION_REPRESENTATION_EULER,
        "rpy": ROTATION_REPRESENTATION_EULER,
        "6d": ROTATION_REPRESENTATION_6D,
        "rot6d": ROTATION_REPRESENTATION_6D,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ROTATION_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported rotation representation {value!r}; expected one of "
            f"{ROTATION_REPRESENTATIONS}."
        )
    return normalized


def rotation_representation_dim(value: object) -> int:
    representation = normalize_rotation_representation(value)
    return 3 if representation == ROTATION_REPRESENTATION_EULER else 6


def normalize_ego_pose_representation(value: object) -> str:
    normalized = str(value or EGO_DEFAULT_POSE_REPRESENTATION).strip().lower()
    normalized = normalized.replace("-", "_")
    aliases = {
        "default": EGO_DEFAULT_POSE_REPRESENTATION,
        "umi": EGO_POSE_REPRESENTATION_UMI_LOCAL,
        "umi_local_delta": EGO_POSE_REPRESENTATION_UMI_LOCAL,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in EGO_POSE_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported ego pose representation {value!r}; expected one of "
            f"{EGO_POSE_REPRESENTATIONS}."
        )
    return normalized


def ego_action_tokens_per_step(value: object = None) -> int:
    """Return the independent compact (3) or separated (5) token layout."""

    if value is None:
        return EGO_DEFAULT_ACTION_TOKENS_PER_STEP
    aliases = {
        "compact": EGO_COMPACT_ACTION_TOKENS_PER_STEP,
        "compact_3": EGO_COMPACT_ACTION_TOKENS_PER_STEP,
        "separated": EGO_SEPARATED_ACTION_TOKENS_PER_STEP,
        "separate": EGO_SEPARATED_ACTION_TOKENS_PER_STEP,
        "separated_5": EGO_SEPARATED_ACTION_TOKENS_PER_STEP,
    }
    normalized = aliases.get(str(value).strip().lower().replace("-", "_"), value)
    try:
        count = int(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported ego action token layout {value!r}; expected 3 or 5."
        ) from exc
    if count not in EGO_ACTION_TOKENS_PER_STEP_OPTIONS:
        raise ValueError(
            f"Unsupported ego action token layout {value!r}; expected one of "
            f"{EGO_ACTION_TOKENS_PER_STEP_OPTIONS}."
        )
    return count


def ego_action_token_dim(
    value: object,
    action_tokens_per_step: object = None,
) -> int:
    """Return pose plus five-fingertip width for one ego action token."""

    token_count = ego_action_tokens_per_step(action_tokens_per_step)
    if token_count == EGO_SEPARATED_ACTION_TOKENS_PER_STEP:
        # Pose tokens use 3D translation + 6D rotation and zero padding. Hand
        # tokens use five 3D fingertips, so 15 is the minimum shared width.
        return max(3 + rotation_representation_dim(value), 5 * 3)
    return 3 + rotation_representation_dim(value) + 5 * 3


def ego_proprio_dim(value: object, action_tokens_per_step: object = None) -> int:
    token_count = ego_action_tokens_per_step(action_tokens_per_step)
    return token_count * ego_action_token_dim(
        value,
        token_count,
    )


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Encode rotations using UMI's first-two-matrix-rows convention."""

    mat = np.asarray(matrix)
    if mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [..., 3, 3], got {mat.shape}.")
    return mat[..., :2, :].copy().reshape(mat.shape[:-2] + (6,))


def rotation_6d_to_matrix(value: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    """Decode UMI-style 6D rotations with Gram-Schmidt orthogonalization."""

    d6 = np.asarray(value)
    if d6.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotations [..., 6], got {d6.shape}.")
    first, second = d6[..., :3], d6[..., 3:]
    first_norm = np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), eps)
    first_unit = first / first_norm
    second = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_norm = np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), eps)
    second_unit = second / second_norm
    third_unit = np.cross(first_unit, second_unit, axis=-1)
    return np.stack((first_unit, second_unit, third_unit), axis=-2)


def action_entry(
    value: Optional[np.ndarray],
    validity: Optional[np.ndarray] = None,
    *,
    name: Optional[str] = None,
    frame: str = ACTION_FRAME_WORLD,
) -> dict:
    """Return one canonical action annotation entry."""

    normalized_frame = str(frame).strip().lower()
    if normalized_frame not in ACTION_FRAMES:
        raise ValueError(f"Unknown ego action frame {frame!r}; expected one of {ACTION_FRAMES}")
    if value is None:
        return {
            "available": False,
            "value": None,
            "validity": None,
            "frame": normalized_frame,
        }
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return {
            "available": False,
            "value": None,
            "validity": None,
            "frame": normalized_frame,
        }

    finite = np.isfinite(arr)
    if validity is None:
        valid_arr = finite
        if name in {LEFT_HAND_3D_KEYPOINTS, RIGHT_HAND_3D_KEYPOINTS} and arr.ndim >= 2:
            point_valid = finite.all(axis=-1, keepdims=True) & (
                np.abs(np.nan_to_num(arr, copy=True)).sum(axis=-1, keepdims=True) > 0.0
            )
            valid_arr = np.broadcast_to(point_valid, arr.shape).copy()
    else:
        valid_arr = np.asarray(validity, dtype=bool)
        if valid_arr.shape == arr.shape:
            pass
        elif arr.ndim > 0 and valid_arr.shape == arr.shape[:-1]:
            valid_arr = np.expand_dims(valid_arr, axis=-1)
        elif arr.ndim > 0 and valid_arr.ndim == 1 and valid_arr.shape[0] == arr.shape[0]:
            valid_arr = valid_arr.reshape((arr.shape[0],) + (1,) * (arr.ndim - 1))
        elif valid_arr.ndim == 0:
            valid_arr = valid_arr.reshape((1,) * arr.ndim)
        try:
            valid_arr = np.broadcast_to(valid_arr, arr.shape).copy()
        except ValueError as exc:
            raise ValueError(
                f"Action validity shape {valid_arr.shape} is not broadcastable to value shape {arr.shape}"
            ) from exc
        valid_arr &= finite
    return {
        "available": True,
        "value": arr,
        "validity": np.asarray(valid_arr, dtype=bool),
        "frame": normalized_frame,
    }


def build_actions(
    *,
    frames: Optional[dict[str, str]] = None,
    validity: Optional[dict[str, np.ndarray]] = None,
    **values: Optional[np.ndarray],
) -> dict[str, dict]:
    """Build a complete action dict with all required modalities."""

    unknown = sorted(set(values).difference(ACTION_MODALITIES))
    if unknown:
        raise KeyError(f"Unknown ego action modalities: {unknown}")
    frames = dict(frames or {})
    validity = dict(validity or {})
    unknown_frames = sorted(set(frames).difference(ACTION_MODALITIES))
    unknown_validity = sorted(set(validity).difference(ACTION_MODALITIES))
    if unknown_frames or unknown_validity:
        raise KeyError(
            "Unknown ego action metadata modalities: "
            f"frames={unknown_frames}, validity={unknown_validity}"
        )
    actions = {name: action_entry(None, name=name) for name in ACTION_MODALITIES}
    for name, value in values.items():
        actions[name] = action_entry(
            value,
            validity.get(name),
            name=name,
            frame=frames.get(name, ACTION_FRAME_WORLD),
        )
    return actions


def hand_keypoints_21x3(value: object) -> Optional[np.ndarray]:
    """Normalize MediaPipe-style 21x3 hand keypoints to [T, 21, 3]."""

    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 2 and arr.shape == (MEDIAPIPE_HAND_NUM_KEYPOINTS, 3):
        arr = arr.reshape(1, MEDIAPIPE_HAND_NUM_KEYPOINTS, 3)
    elif arr.ndim == 2 and arr.shape[-1] == 63:
        arr = arr.reshape(arr.shape[0], MEDIAPIPE_HAND_NUM_KEYPOINTS, 3)
    elif arr.ndim == 3 and arr.shape[-2:] == (MEDIAPIPE_HAND_NUM_KEYPOINTS, 3):
        pass
    else:
        return None
    return arr.astype(np.float32, copy=False)


def partial_hand_keypoints_to_mediapipe(
    value: object,
    *,
    source_indices: Optional[tuple[int, ...]] = None,
) -> Optional[np.ndarray]:
    """Zero-fill partial hand points into a 21x3 MediaPipe-layout tensor.

    If exactly five source points are provided and no explicit mapping is given,
    they are interpreted as thumb/index/middle/ring/pinky fingertips.
    Otherwise the source points are copied into the leading MediaPipe slots to
    preserve the available coordinates while making missing joints explicit.
    """

    full = hand_keypoints_21x3(value)
    if full is not None:
        return full

    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 2 and arr.shape[-1] == 3:
        arr = arr.reshape(1, arr.shape[0], 3)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None

    num_points = int(arr.shape[-2])
    if num_points <= 0:
        return None
    if source_indices is None:
        if num_points == len(MEDIAPIPE_FINGERTIP_INDICES):
            source_indices = MEDIAPIPE_FINGERTIP_INDICES
        else:
            source_indices = tuple(range(min(num_points, MEDIAPIPE_HAND_NUM_KEYPOINTS)))
    if len(source_indices) < num_points:
        num_points = len(source_indices)
        arr = arr[:, :num_points]

    out = np.zeros(
        (arr.shape[0], MEDIAPIPE_HAND_NUM_KEYPOINTS, 3),
        dtype=np.float32,
    )
    for src_idx, dst_idx in enumerate(source_indices[:num_points]):
        if 0 <= int(dst_idx) < MEDIAPIPE_HAND_NUM_KEYPOINTS:
            out[:, int(dst_idx)] = arr[:, src_idx]
    return out


def pose7_to_xyzrpy(value: object, *, quat_order: str = "xyzw") -> Optional[np.ndarray]:
    """Convert [x, y, z, quaternion] poses to [x, y, z, roll, pitch, yaw]."""

    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0 or arr.shape[-1] != 7:
        return None
    flat = arr.reshape(-1, 7)
    trans = flat[:, :3]
    quat = flat[:, 3:7]
    euler = quat_to_euler_xyz(quat, order=quat_order)
    out = np.concatenate([trans, euler], axis=-1).astype(np.float32, copy=False)
    return out.reshape(arr.shape[:-1] + (6,))


def pose_matrix_to_xyzrpy(value: object) -> Optional[np.ndarray]:
    """Convert flattened 3x4/4x4 rigid matrices to [x, y, z, roll, pitch, yaw]."""

    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None

    leading_shape = arr.shape[:-1] if arr.ndim >= 2 and arr.shape[-1] in (12, 16) else arr.shape[:-2]
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 and arr.shape[-1] in (12, 16) else arr.reshape(-1, *arr.shape[-2:])
    matrices = []
    for matrix in flat:
        mat_arr = np.asarray(matrix, dtype=np.float32)
        if mat_arr.shape == (12,):
            mat = mat_arr.reshape(3, 4)
        elif mat_arr.shape == (16,):
            mat = mat_arr.reshape(4, 4)[:3, :4]
        elif mat_arr.shape == (3, 4):
            mat = mat_arr
        elif mat_arr.shape == (4, 4):
            mat = mat_arr[:3, :4]
        else:
            return None
        matrices.append(mat)
    if not matrices:
        return None

    mats = np.stack(matrices, axis=0)
    trans = mats[:, :3, 3]
    euler = rotation_matrix_to_euler_xyz(mats[:, :3, :3])
    out = np.concatenate([trans, euler], axis=-1).astype(np.float32, copy=False)
    return out.reshape(leading_shape + (6,))


def quat_to_euler_xyz(quat: np.ndarray, *, order: str = "xyzw") -> np.ndarray:
    """Convert quaternions to XYZ Euler angles in radians."""

    q = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    if order == "xyzw":
        x, y, z, w = [q[:, i] for i in range(4)]
    elif order == "wxyz":
        w, x, y, z = [q[:, i] for i in range(4)]
    else:
        raise ValueError(f"Unsupported quaternion order: {order!r}")

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    norm = np.where(norm > 0.0, norm, 1.0)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32, copy=False)


def rotation_matrix_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to XYZ Euler angles in radians."""

    r = np.asarray(rotation, dtype=np.float32).reshape(-1, 3, 3)
    sy = np.sqrt(r[:, 0, 0] * r[:, 0, 0] + r[:, 1, 0] * r[:, 1, 0])
    singular = sy < 1e-6

    roll = np.where(
        singular,
        np.arctan2(-r[:, 1, 2], r[:, 1, 1]),
        np.arctan2(r[:, 2, 1], r[:, 2, 2]),
    )
    pitch = np.where(singular, np.arctan2(-r[:, 2, 0], sy), np.arctan2(-r[:, 2, 0], sy))
    yaw = np.where(singular, 0.0, np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32, copy=False)
