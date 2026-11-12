#!/usr/bin/env python3
"""Serve a checkpoint-owned multi-embodiment Engram pi0.5 policy."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import base_policy

from openpi.policies import policy_config
from openpi.policies.engram_multi_policy import camera_size
from openpi.policies.engram_multi_policy import prepare_camera
from openpi.serving import websocket_policy_server
from openpi.shared.engram_multi import ACTION_DIM
from openpi.shared.engram_multi import EMBODIMENTS
from openpi.shared.engram_multi import HAND_START
from openpi.shared.engram_multi import ROUTE_NAMES
from openpi.shared.engram_multi import ROUTES
from openpi.shared.engram_multi import load_route_stats
from openpi.shared.engram_multi import route_id
from openpi.training import config as train_config

ASSET_ID = "engram/multi_embodiment_train"


def _canonical_robot(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("robot_type is required")
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    return {"w1": "dex_w1", "dex": "dex_w1"}.get(key, key)


def _canonical_effector(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("end_effector_type is required")
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    return {"l6": "linker_l6", "l20": "linker_l20", "inspire": "inspire_rh56dfx_hand"}.get(key, key)


def parse_prompt(request: dict) -> tuple[str, str, str, str, bool | None]:
    """Resolve JSON/plain instructions and reject conflicting route metadata."""
    prompt = request.get("prompt", request.get("instruction"))
    structured = isinstance(prompt, dict)
    if isinstance(prompt, str) and prompt.lstrip().startswith("{"):
        try:
            prompt = json.loads(prompt)
        except json.JSONDecodeError as exc:
            raise ValueError("The JSON prompt is malformed") from exc
        structured = True
    if structured:
        if not isinstance(prompt, dict) or set(prompt) - {"instruction", "robot_type", "end_effector_type", "end_effector_state_available"}:
            raise ValueError("JSON prompts support instruction, robot_type, end_effector_type and end_effector_state_available")
        instruction = prompt.get("instruction")
        robot = _canonical_robot(prompt.get("robot_type", request.get("robot_type")))
        effector = _canonical_effector(prompt.get("end_effector_type", request.get("end_effector_type")))
        availability = prompt.get("end_effector_state_available")
    else:
        instruction = prompt
        robot = _canonical_robot(request.get("robot_type"))
        effector = _canonical_effector(request.get("end_effector_type"))
        availability = request.get("end_effector_state_available")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    if availability is not None and type(availability) is not bool:
        raise ValueError("end_effector_state_available must be boolean")
    if "robot_type" in request and _canonical_robot(request["robot_type"]) != robot:
        raise ValueError("Conflicting robot_type in prompt and request")
    if "end_effector_type" in request and _canonical_effector(request["end_effector_type"]) != effector:
        raise ValueError("Conflicting end_effector_type in prompt and request")
    if "instruction" in request and request["instruction"] != instruction:
        raise ValueError("Conflicting instruction and prompt")
    if "end_effector_state_available" in request and availability is not None and request["end_effector_state_available"] != availability:
        raise ValueError("Conflicting end-effector availability")
    if availability is None:
        availability = request.get("end_effector_state_available")
    if availability is not None and type(availability) is not bool:
        raise ValueError("end_effector_state_available must be boolean")
    route_id(robot, effector)
    return instruction, robot, effector, "json" if structured else "plain", availability


def prepare_observation(request: dict, route_stats: dict, *, default_head_layout: str = "left") -> tuple[dict, int, bool]:
    if "action_mask" in request:
        raise ValueError("action_mask is internal; declare end_effector_state_available instead")
    instruction, robot, effector, prompt_format, declared_available = parse_prompt(request)
    index = route_id(robot, effector)
    spec = EMBODIMENTS[index]
    joints = np.asarray(request.get("observation/joint_position"), dtype=np.float32)
    hands = request.get("observation/hand_position")
    if joints.shape == (spec.joint_dim + spec.hand_dim,):
        if hands is not None:
            raise ValueError("Supply hand positions either appended to joints or separately")
        hands = joints[spec.joint_dim:]
        joints = joints[:spec.joint_dim]
    if joints.shape != (spec.joint_dim,) or not np.isfinite(joints).all():
        raise ValueError(f"{spec.name} requires {spec.joint_dim} finite measured joint positions in radians")
    if declared_available is False:
        hands = None
    available = hands is not None
    if declared_available is True and not available:
        raise ValueError("The prompt declares hand feedback available but no measurement was supplied")
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[:spec.joint_dim] = True
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    state[:spec.joint_dim] = joints
    if available:
        values = np.asarray(hands, dtype=np.float32)
        if values.shape != (spec.hand_dim,) or not np.isfinite(values).all():
            raise ValueError(f"Expected {spec.hand_dim} finite hand measurements, left hand followed by right hand")
        if not route_stats[spec.name].supervised_dims[HAND_START:HAND_START + spec.hand_dim].all():
            raise ValueError("This checkpoint has no supervised hand channels for the requested route; omit hand feedback")
        minimum = -100 if spec.hand_units == "signed_percent" else 0
        if np.any((values < minimum) | (values > 100)):
            raise ValueError(f"Hand feedback for {spec.name} must be in [{minimum}, 100]")
        state[HAND_START:HAND_START + spec.hand_dim] = values
        mask[HAND_START:HAND_START + spec.hand_dim] = True
    layout = request.get("head_camera_layout", default_head_layout)
    if layout not in ("left", "stereo"):
        raise ValueError("head_camera_layout must be left or stereo")
    if "observation/head_rgbd" not in request:
        raise ValueError("observation/head_rgbd is required")
    image = {"base_0_rgb": prepare_camera(request["observation/head_rgbd"], robot, "head", stereo=layout == "stereo")}
    image_mask = {"base_0_rgb": np.ones((), dtype=bool)}
    for side in ("left", "right"):
        key, role = f"observation/{side}_wrist_rgbd", f"{side}_wrist"
        camera_key = f"{side}_wrist_0_rgb"
        present = key in request
        width, height = camera_size(robot, role)
        image[camera_key] = (prepare_camera(request[key], robot, role) if present
                             else np.zeros((height, width, 3), dtype=np.uint8))
        image_mask[camera_key] = np.bool_(present)
    return {"state": state, "action_mask": mask, "embodiment_id": np.int32(index),
            "image": image, "image_mask": image_mask, "prompt": instruction,
            "robot_type": robot, "end_effector_type": effector, "prompt_format": prompt_format}, index, available


def read_checkpoint_assets(checkpoint: Path) -> tuple[dict, dict, dict]:
    checkpoint = Path(checkpoint)
    if any(".orbax-checkpoint-tmp-" in part for part in (*checkpoint.parts, *checkpoint.resolve().parts)):
        raise ValueError("Refusing an incomplete Orbax temporary checkpoint")
    metadata = json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text())
    if type(metadata.get("commit_timestamp_nsecs")) is not int or metadata["commit_timestamp_nsecs"] <= 0:
        raise ValueError("Checkpoint has not completed its Orbax commit")
    if not (checkpoint / "params").is_dir() or (checkpoint / "model.safetensors").exists():
        raise ValueError("Expected unambiguous Orbax model parameters")
    assets = checkpoint / "assets" / ASSET_ID
    contract = json.loads((assets / "engram_multi_contract.json").read_text())
    manifest = json.loads((assets / "engram_multi_manifest.json").read_text())
    stats_file = assets / "engram_multi_stats.json"
    digest = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    if (contract.get("format") != "engram-multi-policy-v1" or contract.get("action_dim") != ACTION_DIM
            or contract.get("action_horizon") != 64 or contract.get("action_head_names") != list(ROUTE_NAMES)
            or manifest.get("action_head_names") != list(ROUTE_NAMES)
            or contract.get("stats_sha256") != digest or manifest.get("stats_sha256") != digest
            or manifest.get("max_frames") is not None or manifest.get("split") != "train"
            or type(manifest.get("total_anchors")) is not int or manifest["total_anchors"] < 2
            or manifest.get("used_anchors") != manifest["total_anchors"]
            or manifest.get("sampled_anchors") != manifest["total_anchors"]
            or contract.get("routes") != ROUTES or contract.get("preserve_prompt_text") is not True
            or type(contract.get("max_token_len")) is not int or contract["max_token_len"] <= 0
            or contract.get("max_token_len") != manifest.get("max_token_len")):
        raise ValueError("Checkpoint normalization, layout, or full-training contract is incompatible")
    video = contract.get("video", {})
    if video.get("profiling_subset_only") is not False or video.get("mode") != "proxy_640":
        raise ValueError("Expected the complete validated multi-camera proxy training input")
    for robot in ("dex_w1", "galbot"):
        for role in ("head", "left_wrist", "right_wrist"):
            if contract.get("camera_sizes", {}).get(robot, {}).get(role) != list(camera_size(robot, role)):
                raise ValueError("Checkpoint camera preprocessing differs from this server")
    stats = load_route_stats(stats_file)
    counts = manifest.get("routes", {})
    if set(counts) != set(ROUTE_NAMES):
        raise ValueError("Checkpoint does not cover every end-effector route")
    anchors = 0
    for index, name in enumerate(ROUTE_NAMES):
        entry = counts[name]
        if not isinstance(entry, dict) or entry != stats[name].counts:
            raise ValueError(f"Checkpoint sample counts disagree for {name}")
        count, hands = entry.get("anchors"), entry.get("labeled_hand_anchors")
        if (type(count) is not int or count <= 0 or type(hands) is not int
                or hands < 0 or hands > count):
            raise ValueError(f"Invalid checkpoint sample coverage for {name}")
        expected_hand = EMBODIMENTS[index].hand_units != "unavailable"
        if expected_hand != (hands > 0) or expected_hand != bool(stats[name].supervised_dims[HAND_START:].any()):
            raise ValueError(f"Checkpoint hand supervision is incomplete for {name}")
        anchors += count
    if anchors != manifest["total_anchors"]:
        raise ValueError("Checkpoint per-route counts do not sum to full training coverage")
    return stats, contract, manifest


class MultiEngramServingPolicy(base_policy.BasePolicy):
    def __init__(self, policy: Any, route_stats: dict, contract: dict, *, head_layout: str = "left"):
        self._policy, self._stats, self._contract, self._head_layout = policy, route_stats, contract, head_layout

    @property
    def metadata(self):
        return self._contract

    def infer(self, request: dict) -> dict:
        sample, index, has_hand = prepare_observation(request, self._stats, default_head_layout=self._head_layout)
        result = self._policy.infer(sample)
        packed = np.asarray(result["actions"])
        spec = EMBODIMENTS[index]
        if packed.shape != (64, ACTION_DIM) or not np.isfinite(packed).all():
            raise ValueError("Expected finite absolute action targets with shape (64, 64)")
        joints = packed[:, :spec.joint_dim].copy()
        actions = joints
        output = {"joint_actions": joints, "joint_names": ROUTES[spec.name]["joint_names"],
                  "robot_type": spec.robot_type, "end_effector_type": spec.end_effector_type,
                  "end_effector_available": has_hand}
        if has_hand:
            hands = packed[:, HAND_START:HAND_START + spec.hand_dim].copy()
            output["end_effector_actions"] = hands
            actions = np.concatenate([joints, hands], axis=-1)
        output["actions"] = actions
        if "policy_timing" in result:
            output["policy_timing"] = result["policy_timing"]
        return output


def create_policy(checkpoint: Path, *, head_layout: str = "left") -> MultiEngramServingPolicy:
    stats, contract, _ = read_checkpoint_assets(checkpoint)
    tokenizer = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser() / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError("Set OPENPI_DATA_HOME to a cache containing big_vision/paligemma_tokenizer.model")
    config = train_config.get_config("pi05_engram_multi")
    model = dataclasses.replace(config.model, max_token_len=contract["max_token_len"])
    assets = dataclasses.replace(config.data.assets, assets_dir=str(checkpoint / "assets"), asset_id=ASSET_ID)
    data = dataclasses.replace(config.data, assets=assets, split="val")
    config = dataclasses.replace(config, model=model, data=data)
    policy = policy_config.create_trained_policy(config, checkpoint, norm_stats={})
    return MultiEngramServingPolicy(policy, stats, contract, head_layout=head_layout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--head-layout", choices=("left", "stereo"), default="left")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    policy = create_policy(args.checkpoint_dir, head_layout=args.head_layout)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                                                          metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
