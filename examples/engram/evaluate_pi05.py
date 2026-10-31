"""Offline validation of a trained Engram OpenPI checkpoint in physical units."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as train_config
from openpi.training import engram_dataset


def evaluate_samples(dataset: Any, policy: Any, max_samples: int, seed: int) -> dict:
    """Compute held-out action errors on random anchors, not robot rollout success."""
    if max_samples < 1 or len(dataset) == 0:
        raise ValueError("max_samples and validation dataset size must be positive")
    count = min(max_samples, len(dataset))
    indices = np.random.default_rng(seed).choice(len(dataset), size=count, replace=False)
    squared_joints = 0.0
    squared_grippers = np.zeros(2, dtype=np.float64)
    action_steps = 0
    for index in indices:
        sample = dataset[int(index)]
        target = sample["actions"]
        observation = {key: value for key, value in sample.items() if key != "actions"}
        predicted = np.asarray(policy.infer(observation)["actions"])
        if predicted.shape != target.shape or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid policy output: {predicted.shape}, expected {target.shape}")
        error = predicted.astype(np.float64) - target
        squared_joints += float(np.sum(np.square(error[:, :20])))
        squared_grippers += np.sum(np.square(error[:, 20:]), axis=0)
        action_steps += len(error)
    return {
        "num_anchors": count,
        "action_steps": action_steps,
        "joint_rmse_rad": float(np.sqrt(squared_joints / (action_steps * 20))),
        "left_gripper_rmse_percent": float(np.sqrt(squared_grippers[0] / action_steps)),
        "right_gripper_rmse_percent": float(np.sqrt(squared_grippers[1] / action_steps)),
        "selection_seed": seed,
    }


def main(
    checkpoint_dir: str,
    data_root: str = train_config.EngramDataConfig().data_root,
    prune_threshold_rad: float = 0.01,
    gripper_threshold_percent: float = 0.0,
    max_samples: int = 256,
    seed: int = 42,
) -> None:
    checkpoint = Path(checkpoint_dir)
    config = train_config.get_config("pi05_engram_w1_sandwich")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            data_root=data_root,
            split="train",
            prune_threshold_rad=prune_threshold_rad,
            gripper_threshold_percent=gripper_threshold_percent,
        ),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    manifest = checkpoint / "assets" / data_config.asset_id / "engram_stats_manifest.json"
    engram_dataset.check_stats_identity(
        manifest, data_root, "val", config.model.action_horizon, prune_threshold_rad, gripper_threshold_percent
    )
    dataset = engram_dataset.EngramDataset(
        data_root,
        config.model.action_horizon,
        split="val",
        prune_threshold_rad=prune_threshold_rad,
        gripper_threshold_percent=gripper_threshold_percent,
    )
    try:
        policy = policy_config.create_trained_policy(config, checkpoint)
        result = evaluate_samples(dataset, policy, max_samples, seed)
        result.update(
            {
                "checkpoint": str(checkpoint),
                "split": "validation",
                "episodes": len(dataset.episodes),
                "eligible_anchors": len(dataset),
                "metric": "offline physical action RMSE; not rollout success",
            }
        )
        print(json.dumps(result, indent=2))
    finally:
        dataset.close()


if __name__ == "__main__":
    tyro.cli(main)
