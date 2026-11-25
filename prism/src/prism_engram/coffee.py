"""Serve the Shanghai Coffee PRISM adapter over the OpenPI WebSocket protocol."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from contextlib import nullcontext
from importlib import import_module
from numbers import Integral, Real
from pathlib import Path
from threading import Lock

import numpy as np
import torch

CAMERAS = ("main", "left_wrist", "right_wrist")
CAMERA_KEYS = (
    "observation.images.kfc_compressed",
    "observation.images.camera_l",
    "observation.images.camera_r",
)
STATE_KEY = "observation.state"
STATE_DIM = ACTION_DIM = HORIZON = 32
ACTION_TOKENS, TOKEN_WIDTH = 5, 7
CONTEXT_FRAMES = 2
OBSERVATION_HZ, ACTION_HZ = 10.0, 30.0
METADATA = {
    "model": "coffee_prism",
    "protocol_version": 1,
    "stateful": True,
    "reset": {"op": "reset", "seed": 0},
    "observation_hz": OBSERVATION_HZ,
    "action_hz": ACTION_HZ,
    "action_horizon": HORIZON,
    "action_dim": ACTION_DIM,
    "camera_order": list(CAMERAS),
    "camera_keys": list(CAMERA_KEYS),
    "state_key": STATE_KEY,
    "output_contract": "actions in the original Coffee LeRobot action order and units",
}


def vector(value, width: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (width,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector of length {width}")
    with np.errstate(over="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite in float32")
    return array


def load_finetune_code(directory: Path):
    """Load the model and preprocessing modules from Coffee training."""
    directory = directory.resolve()
    for name in ("policy.py", "data.py", "prism_vla/modeling/action_context.py"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    for name in ("prism_vla", "policy", "data"):
        cached = sys.modules.get(name)
        if cached is not None and not any(
            Path(location).resolve().is_relative_to(directory)
            for location in [getattr(cached, "__file__", None), *getattr(cached, "__path__", ())]
            if location is not None
        ):
            raise RuntimeError(f"{name} is already loaded from a different checkout; start a fresh process")
    sys.path.insert(0, str(directory))
    return import_module("policy"), import_module("data")


class CoffeeRuntime:
    def __init__(self, policy, adapter: dict, data):
        dataset = adapter["config"]["dataset"]
        if (
            (adapter["mode"], adapter["state_dim"], adapter["action_dim"], adapter["token_width"])
            != ("lerobot", STATE_DIM, ACTION_DIM, TOKEN_WIDTH)
            or dataset["context_frames"] != CONTEXT_FRAMES
            or dataset["observation_hz"] != OBSERVATION_HZ
            or dataset["state_key"] != STATE_KEY
            or dataset["action_key"] != "action"
            or dataset["cameras"] != dict(zip(CAMERAS, CAMERA_KEYS, strict=True))
        ):
            raise ValueError("Expected the Shanghai Coffee LeRobot checkpoint contract")
        self.state_mean = vector(adapter["state_mean"], STATE_DIM, "state_mean")
        self.state_std = vector(adapter["state_std"], STATE_DIM, "state_std")
        self.action_mean = vector(adapter["action_mean"], ACTION_DIM, "action_mean")
        self.action_std = vector(adapter["action_std"], ACTION_DIM, "action_std")
        if (self.state_std <= 0).any() or (self.action_std <= 0).any():
            raise ValueError("Checkpoint normalization standard deviations must be positive")
        self.flip_horizontal = dataset["flip_horizontal"]
        self.policy = policy
        self.data = data
        self.lock = Lock()

    @classmethod
    def from_checkpoint(
        cls, *, finetune_dir: Path, base_checkpoint: Path, checkpoint: Path, qwen_assets: Path, device: str
    ) -> CoffeeRuntime:
        policy_module, data = load_finetune_code(finetune_dir)
        source = policy_module.read_checkpoint(base_checkpoint)
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise ValueError("Coffee PRISM requires a CUDA device")
        if torch_device.index is None:
            torch_device = torch.device("cuda", torch.cuda.current_device())

        policy = policy_module.Policy(
            base_checkpoint,
            source,
            str(qwen_assets),
            mode="lerobot",
            token_width=TOKEN_WIDTH,
            state_dim=STATE_DIM,
            device=torch_device,
        )
        # load_adapter restores the saved Qwen decoder in FP32. Policy.sample
        # selects its trained windowed path based on that restored dtype.
        adapter = policy.load_adapter(checkpoint)
        policy.model.eval()
        policy.state_adapter.eval()
        return cls(policy, adapter, data)

    def new_session(self) -> CoffeeSession:
        return CoffeeSession(self)

    @torch.inference_mode()
    def sample(self, frames: torch.Tensor, state: np.ndarray, prompt: str, seed: int) -> np.ndarray:
        batch = {
            "prompt": [
                self.data.format_instruction(
                    prompt,
                    instruction_format="json",
                    metadata={"embodiment": "robot"},
                    include_subtask_instruction=True,
                )
            ],
            "frames": frames[None],
            "camera_mask": torch.ones((1, len(CAMERAS)), dtype=torch.bool),
            "state": torch.from_numpy(state[None]),
            "proprio": torch.zeros((1, CONTEXT_FRAMES, 75), dtype=torch.float32),
        }
        device = self.policy.device
        devices = [device.index] if device.type == "cuda" else []
        cuda_device = torch.cuda.device(device) if devices else nullcontext()
        with self.lock, torch.random.fork_rng(devices=devices), cuda_device:
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed(seed)
            sample = self.policy.sample(batch)
        expected = (1, HORIZON * ACTION_TOKENS, TOKEN_WIDTH)
        if tuple(sample.shape) != expected:
            raise RuntimeError(f"Expected action token shape {expected}, got {tuple(sample.shape)}")
        normalized = sample.float().cpu().numpy().reshape(HORIZON, -1)[:, :ACTION_DIM]
        actions = normalized * self.action_std + self.action_mean
        if not np.isfinite(actions).all():
            raise RuntimeError("Coffee policy produced nonfinite actions")
        return actions


class CoffeeSession:
    """A two-observation window and independent random stream for one client."""

    def __init__(self, runtime: CoffeeRuntime):
        self.runtime = runtime
        self.reset()

    def reset(self, seed: int = 0) -> dict:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        self.episode_generation = getattr(self, "episode_generation", -1) + 1
        self.last_timestamp = None
        self.last_frames = None
        self.prompt = None
        self.rng = np.random.default_rng(int(seed))
        return {"reset": True, "seed": int(seed), "observation_hz": OBSERVATION_HZ}

    def infer(self, request: dict, *, decode: bool = True) -> dict:
        timestamp = request.get("timestamp_s")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp):
            raise ValueError("Supply a finite timestamp_s")
        if self.last_timestamp is not None and not 0.05 - 1e-6 <= timestamp - self.last_timestamp <= 0.15 + 1e-6:
            raise ValueError("Observations must advance at 10 Hz (50–150 ms tolerance); reset after gaps")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Supply a nonempty Coffee task string in prompt")
        if self.prompt is not None and self.prompt != prompt:
            raise ValueError("Reset before changing the task prompt")

        runtime = self.runtime
        state = vector(request[STATE_KEY], STATE_DIM, STATE_KEY)
        with np.errstate(over="ignore", invalid="ignore"):
            state = (state - runtime.state_mean) / runtime.state_std
        if not np.isfinite(state).all():
            raise ValueError("Normalized Coffee state values must be finite")
        images = []
        for key in CAMERA_KEYS:
            image = np.asarray(request[key])
            if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1 or image.dtype != np.uint8:
                raise ValueError(f"{key} must be a nonempty HWC uint8 RGB image")
            pixels = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
            images.append(runtime.data.image_tensor(pixels, flip_horizontal=runtime.flip_horizontal)[0])
        frames = torch.stack(images)

        if decode:
            previous = frames if self.last_frames is None else self.last_frames
            try:
                actions = runtime.sample(torch.stack((previous, frames)), state, prompt, int(self.rng.integers(2**63)))
            except Exception:
                self.reset()
                raise
            response = {"actions": actions}
        else:
            response = {"observed": True}
        self.last_frames = frames
        self.last_timestamp = timestamp
        self.prompt = prompt
        return response


def main():
    from prism_engram.server import serve

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-tune-dir", type=Path, required=True, help="Checkpoint-compatible UVA_Finetune checkout")
    parser.add_argument("--base-checkpoint", type=Path, required=True, help="Original PRISM checkpoint-50000 directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Coffee step-*.pt adapter file")
    parser.add_argument("--qwen-assets", type=Path, required=True, help="Local Qwen3.5-2B-Base snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    runtime = CoffeeRuntime.from_checkpoint(
        finetune_dir=args.fine_tune_dir,
        base_checkpoint=args.base_checkpoint,
        checkpoint=args.checkpoint,
        qwen_assets=args.qwen_assets,
        device=args.device,
    )
    serve(runtime, args.host, args.port, metadata=METADATA)


if __name__ == "__main__":
    main()
