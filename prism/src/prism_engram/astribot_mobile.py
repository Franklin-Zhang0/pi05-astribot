"""Serve the full Astribot mobile PRISM checkpoint over OpenPI WebSocket."""

from __future__ import annotations

import argparse
import json
import logging
import math
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from prism_engram.adapter import rotation_6d_to_matrix
from prism_engram.bundle import CRADIO_REVISION, FAST_REVISION, QWEN_REVISION, restore_model_weights
from prism_engram.coffee import vector
from prism_engram.runtime import training_image

SOURCE = "astribot_mobile"
HEAD = "robot"
HORIZON = 32
OBSERVATION_HZ, ACTION_HZ = 10, 20
ARM_SLOTS = (1, 3)
METADATA = {
    "model": "astribot_mobile_prism",
    "protocol_version": 1,
    "stateful": True,
    "reset": {"op": "reset", "seed": 0},
    "observation_hz": OBSERVATION_HZ,
    "action_horizon": HORIZON,
    "action_hz": ACTION_HZ,
    "camera_order": ["stereo_left or head_rgb", "left_wrist", "right_wrist"],
    "action_head": HEAD,
    "arm_order": ["left", "right"],
    "eef_pose_order": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    "chassis_units": ["m", "m", "rad"],
}


def checkpoint_statistics(config: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Bind the named robot head, output geometry, route and fitted TRAIN stats."""
    contract = config.get("cotrain_contract") or {}
    source = contract.get("heads", {}).get(HEAD, {}).get("sources", {}).get(SOURCE) or {}
    representation = source.get("representation") or {}
    geometry = tuple(config.get(key) for key in
                     ("action_step_horizon", "action_tokens_per_step", "action_dim", "proprio_dim"))
    chassis = source.get("chassis_action") or {}
    head = next((head for head in config.get("action_heads", []) if head["name"] == HEAD), {})
    if (
        geometry != (HORIZON, 5, 15, 75)
        or config.get("action_horizon") != HORIZON * 5
        or config.get("query_design") != "token_reduction"
        or config.get("instruction_format") != "json"
        or not config.get("proprio_encoder_by_frame")
        or config.get("vlm_image_input") or config.get("engram_native")
        or config.get("num_dino_cameras") != 3
        or config.get("dino_camera_fusion") != "concat"
        or config.get("visual_encoder_type") != "cradio"
        or config.get("visual_encoder_model_name") != "nvidia/C-RADIOv4-SO400M"
        or config.get("visual_encoder_revision") != CRADIO_REVISION
        or config.get("instruction_encoder_revision") != QWEN_REVISION
        or head.get("type") != "flow_matching"
        or representation != head.get("action_representation")
        or (representation.get("frame"), representation.get("frame_anchor"), representation.get("target"))
        != ("robot", "chunk_start", "chunk_offset")
        or source.get("rotation_representation") != "rotation_6d"
        or source.get("proprio_contract") != "cotrain_native_observed_eef6d_gripper75_v1"
        or source.get("native_frame_kinds") != ["astribot_chassis_relative_controller_axes"]
        or (source.get("observation_hz"), source.get("action_hz")) != (OBSERVATION_HZ, ACTION_HZ)
        or chassis != {"layout": "astribot_sdk_local_offsets_v1", "slot": 0, "coordinates": [0, 1, 2],
                       "units": ["m", "m", "rad"], "anchor": "observed_chunk_start"}
    ):
        raise ValueError("Expected the full Astribot mobile checkpoint with its trained robot/chassis head")
    position_scale = float(source["position_scale"])
    if not math.isfinite(position_scale) or position_scale <= 0 or position_scale != config["chunk_frame_position_scale"]:
        raise ValueError("Invalid checkpoint position scale")
    stats = source.get("action_normalization") or {}
    if stats.get("stats_horizon") != HORIZON:
        raise ValueError("Expected 32-step robot-head TRAIN normalization")
    mean, std, count = (
        vector(np.asarray(stats[key]).reshape(-1), 75, key).reshape(5, 15) for key in ("mean", "std", "count")
    )
    active = np.zeros((5, 15), dtype=bool)
    active[0, :3] = True
    active[list(ARM_SLOTS), :9] = True
    active[[2, 4], 0] = True
    if np.any(std <= 0) or not np.array_equal(count > 0, active):
        raise ValueError("Expected fitted statistics for both arms, both grippers and chassis X/Y/yaw")
    return mean, std, position_scale


def model_assets(local_path: Path | None, repo: str, revision: str, files: tuple[str, ...]) -> str:
    """Use explicit local assets or resolve the released pinned Hub metadata."""
    if local_path is not None:
        return str(local_path.expanduser().resolve(strict=True))
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_OFFLINE

    return snapshot_download(repo, revision=revision, allow_patterns=files, local_files_only=HF_HUB_OFFLINE)


def camera_observation(request):
    """Return RGB views in fixed main/left/right slots; absent wrists stay absent."""
    main = request.get("stereo_left")
    if main is None:
        main = request.get("head_rgb")
    if main is None:
        raise ValueError("Supply stereo_left or head_rgb")
    views = (main, request.get("left_wrist"), request.get("right_wrist"))
    for name, view in zip(("main", "left_wrist", "right_wrist"), views, strict=True):
        if view is not None and (
            not isinstance(view, np.ndarray) or view.dtype != np.uint8
            or view.ndim != 3 or view.shape[-1] != 3 or min(view.shape[:2]) == 0
        ):
            raise ValueError(f"{name} must be an HWC uint8 RGB array")
    return views


def pack_proprio(request, position_scale: float, *, first: bool):
    """Use measured controller EEFs and the immediately preceding 20 Hz chassis state."""
    eef = np.asarray(request["eef_pose"])
    if eef.shape != (2, 7):
        raise ValueError("eef_pose requires shape [2, 7]")
    eef = vector(eef.reshape(-1), 14, "eef_pose", dtype=np.float64).reshape(2, 7)
    if not np.allclose(np.linalg.norm(eef[:, 3:], axis=1), 1, atol=0.02, rtol=0):
        raise ValueError("eef_pose requires left/right XYZ and unit XYZW quaternions")
    gripper = vector(request["gripper_percent"], 2, "gripper_percent")
    if np.any((gripper < 0) | (gripper > 100)):
        raise ValueError("gripper_percent must lie in [0, 100]")
    chassis, previous = request.get("chassis_xyyaw"), request.get("previous_chassis_xyyaw")
    if (chassis is None and previous is not None) or (chassis is not None and previous is None and not first):
        raise ValueError("Supply current and previous 20 Hz chassis states together, except at episode start")
    rotations = Rotation.from_quat(eef[:, 3:]).as_matrix()
    packed = np.zeros((5, 15), dtype=np.float32)
    packed[list(ARM_SLOTS), :3] = eef[:, :3] / position_scale
    packed[list(ARM_SLOTS), 3:9] = rotations[:, :2].reshape(2, 6)
    packed[[2, 4], 0] = gripper / 100
    if chassis is not None:
        chassis = vector(chassis, 3, "chassis_xyyaw", dtype=np.float64)
    if previous is not None:
        packed[0, :3] = chassis - vector(previous, 3, "previous_chassis_xyyaw", dtype=np.float64)
        if not np.isfinite(packed[0, :3]).all():
            raise ValueError("Nonfinite chassis difference")
    return packed.reshape(75), eef[:, :3], rotations, chassis


def decode_actions(prediction, mean, std, position_scale, xyz, rotations, chassis):
    """Convert normalized chunk offsets into absolute controller-coordinate plans."""
    if prediction.shape != (1, HORIZON * 5, 15):
        raise RuntimeError(f"Unexpected robot-head output shape: {prediction.shape}")
    values = prediction[0].float().cpu().numpy().reshape(HORIZON, 5, 15).astype(np.float64)
    values = values * std + mean
    if not np.isfinite(values).all():
        raise RuntimeError("Robot head produced nonfinite actions")
    offsets = values[:, ARM_SLOTS, :9]
    xyz_target = xyz + offsets[..., :3] * position_scale
    matrices = rotation_6d_to_matrix(offsets[..., 3:9]) @ rotations
    quat = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(HORIZON, 2, 4)
    # Keep quaternion signs continuous from the same measured orientation.
    measured = Rotation.from_matrix(rotations).as_quat()
    prior = np.concatenate((measured[None], quat[:-1]))
    signs = np.cumprod(np.where(np.sum(quat * prior, axis=-1) < 0, -1, 1), axis=0)
    eef_pose = np.concatenate((xyz_target, quat * signs[..., None]), axis=-1)
    return {
        "eef_pose": eef_pose,
        "gripper_percent": np.clip(values[:, (2, 4), 0] * 100, 0, 100),
        "chassis_xyyaw": None if chassis is None else chassis + values[:, 0, :3],
    }


class AstribotRuntime:
    """Strict full-checkpoint load with one serialized model and per-client state."""

    def __init__(self, model, tokenizer, encoder, config: dict, device: torch.device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.mean, self.std, self.position_scale = checkpoint_statistics(config)
        self.model.idm.primary_head = HEAD
        self.encoder = encoder
        if encoder.embed_dim != config["dino_dim"]:
            raise ValueError("C-RADIO embedding width differs from the checkpoint")
        self.output_hw = tuple(config["visual_encoder_eval_image_hw"])
        self.robot_frame_id = config["proprio_frame_names"].index("robot")
        instructions = config["instruction_conditioning"]
        if not instructions.get("include_subtask_instruction") or SOURCE not in instructions["by_source"]:
            raise ValueError("Checkpoint must contain the Astribot JSON/subtask profile")
        self.prompt_setup = instructions["defaults"] | instructions["by_source"][SOURCE]
        self.lock = RLock()

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, device: str = "cuda:0", *, qwen_assets=None, fast_assets=None):
        config = json.loads((checkpoint / "config.json").read_text())
        checkpoint_statistics(config)
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer

        from prism_astribot_vla.modeling.checkpoint_compat import instruction_encoder_config
        from prism_astribot_vla.modeling.config import PrismVLAConfig
        from prism_astribot_vla.modeling.model import PrismVLA
        from prism_astribot_vla.perception.encoders import build_visual_encoder

        qwen = model_assets(
            qwen_assets, "Qwen/Qwen3.5-2B-Base", QWEN_REVISION,
            ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
             "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"),
        )
        config["qwen_model_name"] = config["instruction_encoder_model_name"] = qwen
        config["fast_pretrained_path"] = model_assets(
            fast_assets, "physical-intelligence/fast", FAST_REVISION, ("tokenizer.json", "processor_config.json"),
        )
        config.update(stage1_checkpoint="", gradient_checkpointing=False)
        with init_empty_weights(include_buffers=False):
            model = PrismVLA(PrismVLAConfig.from_dict(config))
        if instruction_encoder_config(model.config) != instruction_encoder_config(config):
            raise ValueError("Constructed instruction encoder differs from the checkpoint")
        torch_device = torch.device(device)
        restore_model_weights(model, checkpoint / "model.pt", device=torch_device)
        if model.share_dino_cls_encoder_for_patches or config.get("token_reduction_frozen_vision_encoder"):
            encoder = model.dino_cls_encoder
        elif config.get("train_dino_backbone"):
            encoder = model.dino_encoder
        else:
            # The Astribot run used a frozen encoder outside the model state.
            encoder = build_visual_encoder(
                "cradio", model_name=config["visual_encoder_model_name"],
                revision=config["visual_encoder_revision"], image_size=config["visual_encoder_image_size"],
                summary_pooling=config["visual_encoder_summary_pooling"],
                dynamic_resolution=config["visual_encoder_dynamic_resolution"],
            ).eval().requires_grad_(False).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(qwen, local_files_only=True)
        return cls(model, tokenizer, encoder, config, torch_device)

    def new_session(self):
        return AstribotSession(self)

    def prompt_tokens(self, task: str, subtask: str):
        from prism_astribot_vla.data.instruction_conditioning import format_instruction, tokenize_instructions

        prompt = format_instruction(
            {"task instruction": task, "subtask instruction": subtask},
            instruction_format="json", metadata=self.prompt_setup, include_subtask_instruction=True,
        )
        tokens = tokenize_instructions(self.tokenizer, [prompt],
                                       max_inst_len=self.model.max_inst_len, instruction_format="json")
        return tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)

    def visual_features(self, views):
        valid = [i for i, view in enumerate(views) if view is not None]
        images = np.stack([training_image(views[i], self.output_hw, crop_area=1.0) for i in valid])
        pixels = self.encoder.preprocess_image_tensor(torch.from_numpy(images), device=self.device)
        _, patches = self.encoder(pixels)
        indices = torch.tensor(valid, device=self.device)
        dense = patches.new_zeros((3, *patches.shape[1:])).index_copy_(0, indices, patches)
        mask = torch.tensor([[view is not None for view in views]], device=self.device)
        patch_size = self.encoder.patch_size
        ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        return dense.reshape(1, -1, patches.shape[-1]), mask, (pixels.shape[-2] // ph, pixels.shape[-1] // pw)


class AstribotSession:
    def __init__(self, runtime):
        self.runtime = runtime
        self.reset()

    def reset(self, seed=0):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.episode_generation = getattr(self, "episode_generation", -1) + 1
        self.generator = torch.Generator(device=self.runtime.device).manual_seed(seed)
        self.cache, self.last_timestamp, self.prompt = None, None, None
        return {"reset": True, "seed": seed, "observation_hz": OBSERVATION_HZ}

    @torch.inference_mode()
    def infer(self, request, *, decode=True):
        runtime = self.runtime
        timestamp = request.get("timestamp_s")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp):
            raise ValueError("Supply a finite timestamp_s")
        if self.last_timestamp is not None and not 0.05 - 1e-6 <= timestamp - self.last_timestamp <= 0.15 + 1e-6:
            raise ValueError("Stream observations at 10 Hz (50–150 ms tolerance); reset after a gap")
        task, subtask = request.get("prompt", ""), request.get("subtask", "")
        if not isinstance(task, str) or not isinstance(subtask, str):
            raise ValueError("prompt and subtask must be strings")
        prompt = task.strip(), subtask.strip()
        views = camera_observation(request)
        proprio, xyz, rotations, chassis = pack_proprio(request, runtime.position_scale,
                                                        first=self.last_timestamp is None)
        tokens = runtime.prompt_tokens(*prompt) if prompt != self.prompt else None
        device = runtime.device
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        try:
            with runtime.lock, precision:
                patches, mask, grid = runtime.visual_features(views)
                refresh = {}
                if self.cache is None:
                    self.cache = runtime.model.inference_reset(*tokens, None, None)
                elif tokens is not None:
                    refresh = dict(instruction_ids=tokens[0], instruction_mask=tokens[1],
                                   prefix_refresh_mask=torch.ones(1, dtype=torch.bool, device=device))
                noise = torch.randn(
                    (1, HORIZON * 5, 15), device=device, dtype=runtime.model._model_dtype(), generator=self.generator,
                ) if decode else None
                output, self.cache = runtime.model.inference_step(
                    self.cache, None, patches,
                    dino_camera_mask_new=mask, dino_patch_grid_hw_new=grid,
                    proprio_new=torch.as_tensor(proprio[None], device=device),
                    proprio_frame_ids=torch.tensor([runtime.robot_frame_id], device=device),
                    end_effector_type_ids=torch.ones((1, 2), dtype=torch.long, device=device),
                    decode_actions=decode, flow_initial_noise=noise, **refresh,
                )
            result = decode_actions(output, runtime.mean, runtime.std, runtime.position_scale,
                                    xyz, rotations, chassis) if decode else {"observed": True}
        except Exception:
            # A failed forward may have partially updated the recurrent state.
            self.reset()
            raise
        self.last_timestamp, self.prompt = float(timestamp), prompt
        return {**result, "timestamp_s": self.last_timestamp}


def main():
    from prism_engram.server import serve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Full Astribot checkpoint directory")
    parser.add_argument("--qwen-assets", type=Path, help="Local Qwen configuration and tokenizer (optional)")
    parser.add_argument("--fast-assets", type=Path, help="Local FAST tokenizer (optional)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    runtime = AstribotRuntime.from_checkpoint(
        args.checkpoint, args.device, qwen_assets=args.qwen_assets, fast_assets=args.fast_assets,
    )
    serve(runtime, args.host, args.port, metadata=METADATA)


if __name__ == "__main__":
    main()
