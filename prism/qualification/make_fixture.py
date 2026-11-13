"""Generate deterministic valid observations for real-checkpoint graph parity.

The images are synthetic RGB patterns, not recorded demonstrations. This
fixture establishes runtime equivalence, not task success or model quality.
No training dataset is opened. Joint states use the bundle's recorded means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from prism_engram.adapter import ROUTES, EngramAdapter


def image(height: int, width: int, seed: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        ((x + seed * 17) % 256, (y * 2 + seed * 29) % 256, (x // 3 + y // 2 + seed * 37) % 256), axis=-1
    ).astype(np.uint8)


def make_fixture(normalization: dict) -> tuple[dict, dict]:
    adapter = EngramAdapter(normalization)
    arrays = {}
    cases = []
    for route_index, route in enumerate(ROUTES):
        mean = np.asarray(
            normalization["route_statistics"]["routes"][route.name]["norm_stats"]["state"]["mean"], dtype=np.float32
        )
        head_hw = (360, 640) if route.robot_type == "dex_w1" else (360, 480)
        wrist_hw = (480, 640) if route.robot_type == "dex_w1" else (360, 640)
        for name, hw in (("head", head_hw), ("left_wrist", wrist_hw), ("right_wrist", wrist_hw)):
            arrays[f"{route.name}/{name}"] = image(*hw, route_index + len(name))
        # A nonidentity anchor is essential: identity quaternions cannot detect
        # accidentally reversed relative-rotation composition.
        half_angle = np.pi / 8
        arrays[f"{route.name}/eef"] = np.array(
            [
                [0.45, 0.20, 0.35, np.cos(half_angle), 0, 0, np.sin(half_angle)],
                [0.40, -0.25, 0.30, np.cos(half_angle), np.sin(half_angle), 0, 0],
            ],
            dtype=np.float64,
        )
        scenarios = [
            ("head_plain", (), "plain", True, 9),
            ("both_wrists_json", ("left_wrist", "right_wrist"), "json", True, 9),
            ("left_wrist_json", ("left_wrist",), "json", True, 2),
            ("right_wrist_plain", ("right_wrist",), "plain", True, 2),
        ]
        if route.robot_type == "dex_w1":
            scenarios.append(("hand_unavailable_json", (), "json", False, 2))
        for scenario, wrists, prompt_format, hands_available, observations in scenarios:
            requests = []
            for tick in range(observations):
                joint_key = f"{route.name}/joints{tick}"
                arrays[joint_key] = mean[: route.joint_dim] + np.float32(0.001 * tick)
                request = {
                    "timestamp_s": tick / 10,
                    "robot_type": route.robot_type,
                    "end_effector_type": route.end_effector_type,
                    "prompt_format": prompt_format,
                    "prompt": "pick up the cup",
                    "observation/head_rgbd": {"array": f"{route.name}/head"},
                    "observation/joint_position": {"array": joint_key},
                }
                if prompt_format == "json":
                    request["prompt"] = {
                        "task instruction": "pick up the cup",
                        "end effector type": "gripper" if route_index == 0 else "hand",
                        "end effector number": "dual",
                        "embodiment": "robot",
                        "setting": f"engram_{route.name}",
                    }
                for wrist in wrists:
                    request[f"observation/{wrist}_rgbd"] = {"array": f"{route.name}/{wrist}"}
                if route.robot_type == "dex_w1":
                    request["observation/ee_poses"] = {"array": f"{route.name}/eef"}
                    request["end_effector_state_available"] = hands_available
                    if hands_available:
                        key = f"{route.name}/hands"
                        arrays[key] = np.clip(
                            mean[20 : 20 + route.hand_dim], -100 if route.hand_units == "signed_percent" else 0, 100
                        ).astype(np.float32)
                        request["observation/hand_position"] = {"array": key}
                # Verify actual routing, units and masks before publishing a fixture.
                from capture import resolve_arrays

                adapter.prepare(resolve_arrays(request, arrays))
                requests.append(request)
            cases.append({"name": f"{route.name}_{scenario}", "seed": route_index, "requests": requests})
    return {
        "schema": "prism_engram_parity_fixture_v1",
        "cases": cases,
        "image_origin": "synthetic_deterministic_rgb_patterns",
        "state_origin": "checkpoint_normalization_means_plus_small_known_offsets",
        "policy_quality_evaluated": False,
    }, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture, arrays = make_fixture(json.loads((args.bundle / "normalization.json").read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(
        json.dumps(
            {
                "cases": len(fixture["cases"]),
                "observations": sum(len(case["requests"]) for case in fixture["cases"]),
                "arrays": len(arrays),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
