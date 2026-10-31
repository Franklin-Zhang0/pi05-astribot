"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import hashlib
import json

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    seed: int = 0,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        seed=seed,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    engram_data_root: str | None = None,
    engram_split: str | None = None,
    engram_prune_threshold_rad: float | None = None,
    engram_gripper_threshold_percent: float | None = None,
    assets_base_dir: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    config = _config.get_config(config_name)
    if (
        engram_data_root is not None
        or engram_split is not None
        or engram_prune_threshold_rad is not None
        or engram_gripper_threshold_percent is not None
    ):
        if not isinstance(config.data, _config.EngramDataConfig):
            raise ValueError("Engram dataset overrides require an Engram config")
        data_kwargs = {}
        if engram_data_root is not None:
            data_kwargs["data_root"] = engram_data_root
        if engram_split is not None:
            if engram_split not in ("all", "train"):
                raise ValueError("Normalization statistics may only be fit on train or all, never validation")
            data_kwargs["split"] = engram_split
        if engram_prune_threshold_rad is not None:
            data_kwargs["prune_threshold_rad"] = engram_prune_threshold_rad
        if engram_gripper_threshold_percent is not None:
            data_kwargs["gripper_threshold_percent"] = engram_gripper_threshold_percent
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, **data_kwargs))
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        config = dataclasses.replace(config, batch_size=batch_size)
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        config = dataclasses.replace(config, num_workers=num_workers)
    if max_frames is not None and max_frames < config.batch_size:
        raise ValueError("max_frames must be at least batch_size")
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.engram_data_dir is not None and data_config.engram_split == "val":
        raise ValueError("Normalization statistics may not be fit on validation")

    if data_config.engram_data_dir is not None:
        # Only proprioception and actions contribute to these statistics. Avoid
        # decoding the 4K stereo video while retaining exactly the training
        # anchors, physical-time action chunks and input transforms.
        data_config = dataclasses.replace(data_config, engram_decode_images=False)
        np.random.seed(config.seed)

        # Use a vectorized exact anchor traversal. In particular, placeholders
        # from joint-only recordings must not alter gripper statistics, and the
        # final partial batch must not be silently dropped.
        from openpi.training.engram_dataset import compute_training_stats, stats_identity

        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        norm_stats, counts = compute_training_stats(dataset, seed=config.seed, max_frames=max_frames)
        output_path = config.assets_dirs / data_config.asset_id
        normalize.save(output_path, norm_stats)
        manifest = stats_identity(
            data_config.engram_data_dir, data_config.engram_split, config.model.action_horizon,
            data_config.engram_prune_threshold_rad, data_config.engram_gripper_threshold_percent,
        )
        manifest.update({
            "source_root": str(data_config.engram_data_dir),
            "max_frames": max_frames,
            "sampled_anchors": counts["used_anchors"],
            "labeled_gripper_anchors": counts["labeled_gripper_anchors"],
            "total_training_anchors": counts["total_anchors"],
            "seed": config.seed,
            "joint_quantile_span_floor_radians": 0.1,
            "gripper_quantile_span_floor_percent": 5.0,
            "norm_stats_sha256": hashlib.sha256((output_path / "norm_stats.json").read_bytes()).hexdigest(),
        })
        (output_path / "engram_stats_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote Engram stats to {output_path}: {counts}")
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
            seed=config.seed if data_config.engram_data_dir is not None else 0,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.engram_data_dir is not None:
        from openpi.training.engram_dataset import floor_quantile_ranges

        norm_stats = floor_quantile_ranges(norm_stats)

    output_path = config.assets_dirs / (
        data_config.asset_id if data_config.engram_data_dir is not None else data_config.repo_id
    )
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    if data_config.engram_data_dir is not None:
        from openpi.training.engram_dataset import stats_identity

        manifest = stats_identity(
            data_config.engram_data_dir,
            data_config.engram_split,
            config.model.action_horizon,
            data_config.engram_prune_threshold_rad,
            data_config.engram_gripper_threshold_percent,
        )
        manifest.update(
            {
                "source_root": str(data_config.engram_data_dir),
                "max_frames": max_frames,
                "sampled_anchors": num_batches * config.batch_size,
                "seed": config.seed,
                "joint_quantile_span_floor_radians": 0.1,
                "gripper_quantile_span_floor_percent": 5.0,
                "norm_stats_sha256": hashlib.sha256((output_path / "norm_stats.json").read_bytes()).hexdigest(),
            }
        )
        (output_path / "engram_stats_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    tyro.cli(main)
