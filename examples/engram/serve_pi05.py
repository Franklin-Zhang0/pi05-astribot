#!/usr/bin/env python3
"""Serve a strict Engram pi0.5 checkpoint with the OpenPI WebSocket protocol."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

import av
import numpy as np
from openpi_client import base_policy

from openpi import transforms
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as train_config
from openpi.training import engram_dataset

ASSET_ID = "engram/dex_w1_sandwich_train"
ACTION_HORIZON = 64
PHYSICAL_ACTION_DIM = 22
MODEL_ACTION_DIM = 32
CameraLayout = Literal["left", "stereo"]


class _RejectTruncationEncoder:
    """Count the exact SentencePiece sequence before OpenPI truncates it."""

    def __init__(self, encoder: Any, max_tokens: int):
        self._encoder = encoder
        self._max_tokens = max_tokens

    def encode(self, *args, **kwargs):
        tokens = self._encoder.encode(*args, **kwargs)
        if len(tokens) > self._max_tokens:
            raise ValueError(
                f"Task prompt and normalized state require {len(tokens)} tokens, exceeding "
                f"the {self._max_tokens}-token model limit; shorten the task prompt"
            )
        return tokens

    def __getattr__(self, name):
        return getattr(self._encoder, name)


def reject_truncated_prompts(policy: Any, expected_max_tokens: int) -> None:
    """Wrap OpenPI's existing tokenizer without changing its prompt formatting."""
    pipeline = getattr(policy, "_input_transform", None)
    if not isinstance(pipeline, transforms.CompositeTransform):
        raise ValueError("Expected the OpenPI policy input transform pipeline")
    inputs = list(pipeline.transforms)
    matches = [(index, item) for index, item in enumerate(inputs) if isinstance(item, transforms.TokenizePrompt)]
    if len(matches) != 1 or not matches[0][1].discrete_state_input:
        raise ValueError("Expected exactly one discrete-state PaliGemma prompt tokenizer")
    index, token_transform = matches[0]
    tokenizer = token_transform.tokenizer
    if (getattr(tokenizer, "_max_len", None) != expected_max_tokens
            or not hasattr(getattr(tokenizer, "_tokenizer", None), "encode")):
        raise ValueError("PaliGemma tokenizer does not match the expected model token limit")
    strict_tokenizer = copy.copy(tokenizer)
    strict_tokenizer._tokenizer = _RejectTruncationEncoder(tokenizer._tokenizer, expected_max_tokens)  # noqa: SLF001
    inputs[index] = dataclasses.replace(token_transform, tokenizer=strict_tokenizer)
    policy._input_transform = transforms.compose(inputs)  # noqa: SLF001


def _required_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint file is missing: {path}")


def read_checkpoint_assets(
    checkpoint: Path, *, expected_source_manifest_sha256: str | None = None
) -> tuple[dict[str, normalize.NormStats], dict[str, Any]]:
    """Validate checkpoint-owned assets without accessing the training dataset."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if any(".orbax-checkpoint-tmp-" in part for part in (*checkpoint.parts, *checkpoint.resolve().parts)):
        raise ValueError("Refusing an Orbax temporary checkpoint directory")
    if (checkpoint / "model.safetensors").exists():
        raise ValueError("Unexpected model.safetensors would override the Engram Orbax checkpoint parameters")
    checkpoint_metadata_path = checkpoint / "_CHECKPOINT_METADATA"
    _required_file(checkpoint_metadata_path)
    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text())
    commit_time = checkpoint_metadata.get("commit_timestamp_nsecs") if isinstance(checkpoint_metadata, dict) else None
    if type(commit_time) is not int or commit_time <= 0:
        raise ValueError("Checkpoint metadata does not contain a completed Orbax commit timestamp")
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint parameters are missing: {checkpoint / 'params'}")
    assets = checkpoint / "assets" / ASSET_ID
    stats_path = assets / "norm_stats.json"
    manifest_path = assets / "engram_stats_manifest.json"
    video_path = assets / "engram_video_contract.json"
    for path in (stats_path, manifest_path, video_path):
        _required_file(path)
    manifest = json.loads(manifest_path.read_text())
    video = json.loads(video_path.read_text())
    stats_digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    total = manifest.get("total_training_anchors")
    labeled = manifest.get("labeled_gripper_anchors")
    source_digest = manifest.get("source_manifest_sha256")
    if not isinstance(source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ValueError("Checkpoint must contain a valid multisource manifest SHA-256")
    if expected_source_manifest_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_source_manifest_sha256):
            raise ValueError("Expected source manifest SHA-256 must be 64 lowercase hexadecimal characters")
        if expected_source_manifest_sha256 != source_digest:
            raise ValueError("Checkpoint source manifest does not match the expected SHA-256")
    if (manifest.get("contract") != engram_dataset.STATS_CONTRACT or manifest.get("split") != "train"
            or manifest.get("action_horizon") != ACTION_HORIZON or manifest.get("max_frames", 0) is not None
            or manifest.get("prune_threshold_rad") != 0.01 or manifest.get("gripper_threshold_percent") != 0.0
            or manifest.get("joint_quantile_span_floor_radians") != 0.1
            or manifest.get("gripper_quantile_span_floor_percent") != 5.0
            or type(total) is not int or total < 2 or manifest.get("sampled_anchors") != total
            or type(labeled) is not int or not 2 <= labeled <= total
            or manifest.get("norm_stats_sha256") != stats_digest):
        raise ValueError("Checkpoint statistics do not match the full strict Engram training contract or hash")
    if (video.get("version") != 1 or video.get("mode") != "proxy_640"
            or video.get("profiling_subset_only") is not False
            or not isinstance(video.get("proxy_identity"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", video["proxy_identity"])):
        raise ValueError("Checkpoint video contract does not match the full 640-pixel training corpus")
    stats = normalize.load(assets)
    if set(stats) != {"state", "actions"}:
        raise ValueError("Checkpoint must contain state and action normalization statistics")
    for name, value in stats.items():
        arrays = [np.asarray(getattr(value, key)) for key in ("mean", "std", "q01", "q99")]
        if any(array.shape != (PHYSICAL_ACTION_DIM,) or not np.isfinite(array).all() for array in arrays):
            raise ValueError(f"Checkpoint {name} statistics must have 22 finite dimensions")
        if np.any(arrays[1] < 0) or np.any(arrays[3] - arrays[2] < np.array([0.1] * 20 + [5.0] * 2) - 1e-6):
            raise ValueError(f"Checkpoint {name} statistics have an invalid normalization range")
    return stats, {
        "checkpoint": str(checkpoint.resolve()),
        "normalization_asset_id": ASSET_ID,
        "normalization_stats_sha256": stats_digest,
        "normalization_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training_source_manifest_sha256": source_digest,
        "training_proxy_identity": video["proxy_identity"],
    }


def prepare_camera(image: Any, layout: CameraLayout) -> np.ndarray:
    """Select the specified RGB eye and area-resize to the training proxy size."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Head camera must be an HWC RGB image with three channels")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError("Floating camera values must be finite and in [0, 1]")
        array = (array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError("Camera image must be uint8 RGB or floating RGB in [0, 1]")
    if layout == "stereo":
        if array.shape[1] % 2:
            raise ValueError("Side-by-side stereo width must be even")
        array = array[:, : array.shape[1] // 2]
    elif layout != "left":
        raise ValueError(f"Unknown camera layout: {layout}")
    height, width = array.shape[:2]
    if width * 9 != height * 16:
        raise ValueError("Selected left camera image must have 16:9 aspect ratio; check camera layout")
    array = np.ascontiguousarray(array)
    if (height, width) != (360, 640):
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        array = frame.reformat(width=640, height=360, format="rgb24", interpolation="AREA").to_ndarray(format="rgb24")
    return array


def prepare_observation(observation: dict[str, Any], layout: CameraLayout) -> tuple[dict[str, Any], bool]:
    """Build the original Engram policy input and its end-effector availability."""
    if "observation/left_wrist_rgbd" in observation or "observation/right_wrist_rgbd" in observation:
        raise ValueError("This single-camera policy cannot consume wrist images")
    prompt = observation.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty task prompt is required")
    if "observation/head_rgbd" not in observation or "observation/joint_position" not in observation:
        raise ValueError("Head camera and current joint position are required")
    state = np.asarray(observation["observation/joint_position"])
    if state.shape not in ((20,), (22,)) or state.dtype.kind not in "fiu":
        raise ValueError("Current state must have 20 joints or 20 joints plus two grippers")
    state = state.astype(np.float32)
    if not np.isfinite(state[:20]).all():
        raise ValueError("Joint state must be finite radians")
    mask = np.asarray(observation.get("action_mask", [True] * 20 + [len(state) == 22] * 2))
    if (mask.shape != (22,) or mask.dtype != np.dtype(bool) or not mask[:20].all() or mask[20] != mask[21]
            or (len(state) == 20 and mask[20])):
        raise ValueError("action_mask must mark all joints available and both grippers jointly available or absent")
    has_gripper = bool(mask[20])
    if has_gripper:
        if not np.isfinite(state[20:]).all() or np.any((state[20:] < 0) | (state[20:] > 100)):
            raise ValueError("Measured gripper positions must be finite percentages in [0, 100]")
    else:
        # These placeholders are masked in the real Engram transform stack,
        # before tokenization, and are never exposed as physical output.
        state = np.concatenate((state[:20], np.zeros(2, dtype=np.float32)))
    return {
        "observation/head_rgbd": prepare_camera(observation["observation/head_rgbd"], layout),
        "observation/joint_position": state,
        "action_mask": mask,
        "prompt": prompt,
    }, has_gripper


class EngramServingPolicy(base_policy.BasePolicy):
    """Adapt OpenPI requests and hide unsupervised end-effector outputs."""

    def __init__(self, policy: Any, *, camera_layout: CameraLayout, metadata: dict[str, Any]):
        if camera_layout not in ("left", "stereo"):
            raise ValueError("camera_layout must be left or stereo")
        self._policy = policy
        self._camera_layout = camera_layout
        self._metadata = metadata

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        sample, has_gripper = prepare_observation(observation, self._camera_layout)
        output = self._policy.infer(sample)
        actions = np.asarray(output["actions"])
        if (actions.shape != (ACTION_HORIZON, PHYSICAL_ACTION_DIM) or actions.dtype.kind not in "fiu"
                or not np.isfinite(actions).all()):
            raise ValueError(f"Expected finite physical targets with shape (64, 22), got {actions.shape}")
        # OpenPI has already unnormalized and converted joint deltas back to
        # absolute targets. The missing dimensions are not valid commands.
        result = {"actions": actions[:, :22 if has_gripper else 20].copy(), "gripper_available": has_gripper}
        if "policy_timing" in output:
            result["policy_timing"] = output["policy_timing"]
        return result


def create_policy(
    checkpoint: Path, *, camera_layout: CameraLayout, expected_source_manifest_sha256: str | None = None
) -> EngramServingPolicy:
    """Build the existing OpenPI inference pipeline with checkpoint statistics."""
    stats, metadata = read_checkpoint_assets(checkpoint, expected_source_manifest_sha256=expected_source_manifest_sha256)
    tokenizer = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser() / "big_vision/paligemma_tokenizer.model"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"PaliGemma tokenizer is missing: {tokenizer}; set OPENPI_DATA_HOME to its cache")
    base = train_config.get_config("pi05_engram_w1_sandwich")
    if (base.model.action_horizon != ACTION_HORIZON or base.model.action_dim != MODEL_ACTION_DIM
            or base.model.max_token_len != 200):
        raise ValueError("Engram model configuration must have horizon 64, action head 32 and 200 prompt tokens")
    assets = dataclasses.replace(base.data.assets, assets_dir=str(checkpoint / "assets"), asset_id=ASSET_ID)
    data = dataclasses.replace(base.data, assets=assets, split="val", single_camera=True, video_mode="proxy_640")
    config = dataclasses.replace(base, data=data)
    policy = policy_config.create_trained_policy(config, checkpoint, norm_stats=stats)
    reject_truncated_prompts(policy, base.model.max_token_len)
    metadata.update({
        "model": "pi05_engram_w1_sandwich", "action_horizon": ACTION_HORIZON, "action_rate_hz": 30,
        "joint_names": list(engram_dataset.JOINT_NAMES), "joint_units": "radians", "gripper_units": "percent",
        "camera_layout": camera_layout, "camera_processing": "left RGB area-resize 640x360; OpenPI pad to 224x224",
        "single_camera": True,
        "single_camera_provenance": "imposed from the Engram run configuration; not encoded in the video contract",
        "outputs": "unbounded absolute 20-joint targets; two unbounded gripper targets only when feedback is available",
    })
    return EngramServingPolicy(policy, camera_layout=camera_layout, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Completed Engram checkpoint directory")
    parser.add_argument("--host", default="127.0.0.1", help="WebSocket bind host")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket bind port")
    parser.add_argument("--camera-layout", choices=("left", "stereo"), default="left")
    parser.add_argument("--expected-source-manifest-sha256", help="Require a specific 64-hex training source identity")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    policy = create_policy(args.checkpoint_dir, camera_layout=args.camera_layout,
                           expected_source_manifest_sha256=args.expected_source_manifest_sha256)
    print(json.dumps({"event": "ENGRAM_POLICY_LOADED", "host": args.host, "port": args.port, **policy.metadata}), flush=True)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                                                            metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
