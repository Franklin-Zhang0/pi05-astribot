#!/usr/bin/env python3
"""Run one checkpoint-backed inference for the dexterous-hand candle policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from openpi_client import websocket_client_policy

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

DEFAULT_CHECKPOINT = Path(
    "/data2/202608/astribot/checkpoints/pi05_astribot_light_candle/astribot-light-candle-20260827-235217/20000"
)
DEFAULT_CONFIG = "pi05_astribot_light_candle"
DEFAULT_REPO_ID = "astribot/astribot_light_candle_lerobot"
DEFAULT_PROMPT = "Light the candle"
EXPECTED_ACTION_SHAPE = (64, 52)


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def run(
    *,
    checkpoint_dir: Path,
    config_name: str,
    sample_index: int,
    prompt: str,
    policy_host: str | None,
    policy_port: int,
) -> dict[str, object]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_dir}")

    train_cfg = _config.get_config(config_name)
    if train_cfg.data.repo_id != DEFAULT_REPO_ID:
        raise ValueError(f"Unexpected dataset for {config_name}: {train_cfg.data.repo_id}")

    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.asset_id is None:
        raise ValueError("Astribot serving config must define an asset_id")
    norm_stats = checkpoint_dir / "assets" / data_cfg.asset_id / "norm_stats.json"
    if not norm_stats.is_file():
        raise FileNotFoundError(f"checkpoint-owned norm stats are missing: {norm_stats}")

    dataset = _data_loader.create_torch_dataset(data_cfg, train_cfg.model.action_horizon, train_cfg.model)
    raw = dict(dataset[int(sample_index)])
    obs = {
        "observation/head_rgbd": _as_numpy(raw["images.head"]),
        "observation/left_wrist_rgbd": _as_numpy(raw["images.left_wrist"]),
        "observation/right_wrist_rgbd": _as_numpy(raw["images.right_wrist"]),
        "observation/joint_position": _as_numpy(raw["state"]),
        "prompt": str(raw.get("prompt", prompt)),
    }
    if obs["observation/joint_position"].shape != (52,):
        raise AssertionError(f"Expected a 52-D state, got {obs['observation/joint_position'].shape}")

    load_started = time.monotonic()
    if policy_host is None:
        policy = _policy_config.create_trained_policy(train_cfg, checkpoint_dir, default_prompt=prompt)
        transport = "direct"
    else:
        policy = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)
        transport = "websocket"
    load_seconds = time.monotonic() - load_started

    infer_started = time.monotonic()
    out = policy.infer(obs)
    wall_infer_seconds = time.monotonic() - infer_started
    actions = np.asarray(out["actions"])
    if actions.shape != EXPECTED_ACTION_SHAPE:
        raise AssertionError(f"Expected actions shape {EXPECTED_ACTION_SHAPE}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise AssertionError("Non-finite actions found (NaN/Inf)")

    return {
        "status": "passed",
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "config_name": config_name,
        "repo_id": data_cfg.repo_id,
        "asset_id": data_cfg.asset_id,
        "sample_index": int(sample_index),
        "prompt": obs["prompt"],
        "transport": transport,
        "state_shape": list(obs["observation/joint_position"].shape),
        "action_shape": list(actions.shape),
        "actions_finite": True,
        "actions_min": float(actions.min()),
        "actions_max": float(actions.max()),
        "checkpoint_load_seconds": load_seconds,
        "wall_infer_seconds": wall_infer_seconds,
        "model_infer_ms": float(out["policy_timing"]["infer_ms"]),
        "server_infer_ms": float(out.get("server_timing", {}).get("infer_ms", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--policy-host", help="Use the websocket server at this host instead of loading locally")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()

    result = run(
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        sample_index=args.sample_index,
        prompt=args.prompt,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.result_json.with_suffix(args.result_json.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, args.result_json)


if __name__ == "__main__":
    main()
