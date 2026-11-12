"""Pinned physical layouts and typed statistics for Engram embodiments."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.shared import normalize

ACTION_DIM = 64
HAND_START = 20
STATS_FORMAT = "engram-route-stats-v1"


@dataclasses.dataclass(frozen=True)
class Embodiment:
    name: str
    robot_type: str
    end_effector_type: str
    joint_dim: int
    hand_dim: int
    hand_units: str


EMBODIMENTS = (
    Embodiment("dex_w1_gripper", "dex_w1", "gripper", 20, 2, "percent"),
    Embodiment("dex_w1_linker_l6", "dex_w1", "linker_l6", 20, 12, "percent"),
    Embodiment("dex_w1_linker_l20", "dex_w1", "linker_l20", 20, 32, "signed_percent"),
    Embodiment("galbot_inspire_rh56dfx_hand", "galbot", "inspire_rh56dfx_hand", 14, 12, "unavailable"),
)
ROUTE_NAMES = tuple(item.name for item in EMBODIMENTS)
W1_JOINT_NAMES = (
    "ANKLE",
    "KNEE",
    "BUTTOCK",
    "WAIST",
    "NECK1",
    "NECK2",
    *(f"LEFT_J{i}" for i in range(1, 8)),
    *(f"RIGHT_J{i}" for i in range(1, 8)),
)
GALBOT_JOINT_NAMES = (*(f"right_arm_joint{i}" for i in range(1, 8)), *(f"left_arm_joint{i}" for i in range(1, 8)))
ROUTES = {
    item.name: {
        "embodiment_id": index,
        **dataclasses.asdict(item),
        "joint_names": list(W1_JOINT_NAMES if item.robot_type == "dex_w1" else GALBOT_JOINT_NAMES),
        "hand_width": item.hand_dim // 2,
        "hand_start": HAND_START,
        "default_hand_supervision": item.hand_units != "unavailable",
    }
    for index, item in enumerate(EMBODIMENTS)
}


def route_id(robot_type: str, end_effector_type: str) -> int:
    for i, item in enumerate(EMBODIMENTS):
        if (robot_type, end_effector_type) == (item.robot_type, item.end_effector_type):
            return i
    raise ValueError(f"Unsupported Engram embodiment: {robot_type}/{end_effector_type}")


def physical_mask(index: int, has_hand_labels: bool) -> np.ndarray:  # noqa: FBT001
    item = EMBODIMENTS[index]
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[: item.joint_dim] = True
    if has_hand_labels:
        mask[HAND_START : HAND_START + item.hand_dim] = True
    return mask


@dataclasses.dataclass(frozen=True)
class RouteStats:
    state: normalize.NormStats
    actions: normalize.NormStats
    supervised_dims: np.ndarray
    counts: dict[str, int] = dataclasses.field(default_factory=dict)


def save_route_stats(path: str | Path, stats: dict[str, RouteStats]) -> None:
    """Write route-owned state/action statistics without merging embodiments."""
    routes = {}
    for name, value in stats.items():
        routes[name] = {
            "norm_stats": json.loads(normalize.serialize_json({"state": value.state, "actions": value.actions}))[
                "norm_stats"
            ],
            "supervised_dims": np.asarray(value.supervised_dims, bool).tolist(),
            "counts": value.counts,
        }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "format": STATS_FORMAT,
                "model_action_dim": ACTION_DIM,
                "route_names": list(ROUTE_NAMES),
                "routes": routes,
            },
            indent=2,
        )
        + "\n"
    )


def load_route_stats(path: str | Path) -> dict[str, RouteStats]:
    """Validate saved route names, dimensions, finite ranges, and label masks."""
    value = json.loads(Path(path).read_text())
    if (
        value.get("format") != STATS_FORMAT
        or value.get("model_action_dim") != ACTION_DIM
        or value.get("route_names") != list(ROUTE_NAMES)
        or set(value.get("routes", {})) != set(ROUTE_NAMES)
    ):
        raise ValueError("Typed Engram normalization has an incompatible route/layout contract")
    result = {}
    for index, name in enumerate(ROUTE_NAMES):
        entry = value["routes"][name]
        raw = normalize.deserialize_json(json.dumps({"norm_stats": entry["norm_stats"]}))
        mask = np.asarray(entry["supervised_dims"])
        if mask.dtype != np.dtype(bool) or mask.shape != (ACTION_DIM,):
            raise ValueError(f"Invalid supervised-dimension mask for {name}")
        allowed = physical_mask(index, has_hand_labels=True)
        if not mask[: EMBODIMENTS[index].joint_dim].all() or np.any(mask & ~allowed):
            raise ValueError(f"Statistics supervise unavailable/padded dimensions for {name}")
        if EMBODIMENTS[index].hand_units == "unavailable" and mask[HAND_START:].any():
            raise ValueError("The unlabeled Galbot hand cannot be declared supervised")
        for key in ("state", "actions"):
            stat = raw[key]
            arrays = [np.asarray(getattr(stat, field)) for field in ("mean", "std", "q01", "q99")]
            if any(array.shape != (ACTION_DIM,) or not np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Invalid {key} statistics for {name}")
            if np.any(arrays[1] < 0) or np.any(arrays[3] <= arrays[2]):
                raise ValueError(f"Invalid normalization ranges for {name}")
            minimum = np.zeros(ACTION_DIM)
            minimum[: EMBODIMENTS[index].joint_dim] = 0.1
            minimum[HAND_START : HAND_START + EMBODIMENTS[index].hand_dim] = 5
            if np.any((arrays[3] - arrays[2])[mask] < minimum[mask] - 1e-6):
                raise ValueError(f"Normalization quantile span is below the physical floor for {name}")
        result[name] = RouteStats(raw["state"], raw["actions"], mask, entry.get("counts", {}))
    return result
