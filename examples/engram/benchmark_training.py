"""Bounded real π0.5 updates and separate loader/compute timing, no checkpoints."""

import argparse
import dataclasses
import functools
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

# Also work when invoked by path without a project-root PYTHONPATH entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import sharding
from scripts import train


def _dataset_inventory(loader):
    """Report the actual dataset behind this benchmark's Torch loader."""
    wrapped = loader._data_loader.torch_loader.dataset  # noqa: SLF001
    if not isinstance(wrapped, data_loader.TransformedDataset):
        raise TypeError("Expected the standard transformed Engram dataset")
    dataset = wrapped._dataset  # noqa: SLF001
    parts = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
    sources = [{"root": str(part.root), "episodes": len(part.episodes), "anchors": len(part)} for part in parts]
    if sum(part["anchors"] for part in sources) != len(dataset):
        raise ValueError("Reported source anchor counts do not add up")
    return {"sources": sources, "episodes": sum(part["episodes"] for part in sources), "anchors": len(dataset)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--proxy-root", default=None,
                        help="Optional scratch proxy root; omit to use adjacent *_640.mp4 files")
    parser.add_argument("--mode", choices=("original", "proxy_640"), required=True)
    parser.add_argument("--full-corpus", action="store_true",
                        help="Benchmark all training episodes instead of the completed-proxy profiling subset")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    devices = jax.devices()
    if len(devices) != 2 or not all("H100" in device.device_kind for device in devices):
        raise ValueError(f"This benchmark requires exactly two H100s, got {devices}")
    if args.steps < 1 or args.warmup < 1 or args.workers < 0 or args.prefetch_factor < 1:
        raise ValueError("Need positive warmup/measured steps and nonnegative workers")
    base = config_lib.get_config("pi05_engram_w1_sandwich")
    config = dataclasses.replace(
        base, exp_name=f"engram-benchmark-{args.mode}", batch_size=256, gradient_accumulation_steps=32,
        fsdp_devices=2, num_workers=args.workers, prefetch_factor=args.prefetch_factor,
        seed=args.seed, assets_base_dir=args.assets_base_dir,
        wandb_enabled=False,
        data=dataclasses.replace(base.data, data_root=args.data_root, split="train", single_camera=True,
                                 video_mode=args.mode, proxy_root=args.proxy_root,
                                 proxy_subset_only=not args.full_corpus),
    )
    mesh = sharding.make_mesh(2)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    start = time.monotonic()
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    inventory = _dataset_inventory(loader)
    print(json.dumps({"event": "dataset", "mode": args.mode, "full_corpus": args.full_corpus,
                      "proxy_identity": loader.data_config().engram_video_identity, **inventory}), flush=True)
    iterator = iter(loader)
    batch = next(iterator)
    jax.block_until_ready(batch)
    print(json.dumps({"event": "initial_batch", "mode": args.mode, "seconds": time.monotonic() - start,
                      "proxy_identity": loader.data_config().engram_video_identity}), flush=True)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    start = time.monotonic()
    with sharding.set_mesh(mesh):
        state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
        jax.block_until_ready(state)
        step_fn = jax.jit(functools.partial(train.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        print(json.dumps({"event": "pretrained_init", "seconds": time.monotonic() - start}), flush=True)
        results = []
        for iteration in range(args.warmup + args.steps):
            start = time.monotonic()
            state, info = step_fn(train_rng, state, batch)
            jax.block_until_ready((state, info))
            compute = time.monotonic() - start
            metrics = jax.device_get(info)
            if not all(np.isfinite(np.asarray(value)).all() for value in metrics.values()):
                raise ValueError(f"Nonfinite training metrics at step {iteration}")
            start = time.monotonic()
            batch = next(iterator)
            jax.block_until_ready(batch)
            load = time.monotonic() - start
            row = {"iteration": iteration + 1, "warmup": iteration < args.warmup,
                   "compute_s": compute, "next_batch_s": load,
                   "loss": float(metrics["loss"]), "grad_norm": float(metrics["grad_norm"])}
            results.append(row)
            print(json.dumps({"event": "step", **row}), flush=True)
    measured = [item for item in results if not item["warmup"]]
    print(json.dumps({"event": "complete", "mode": args.mode, "batch_size": 256, "accumulation": 32,
                      "full_corpus": args.full_corpus, "episodes": inventory["episodes"], "anchors": inventory["anchors"],
                      "warmup_steps": args.warmup, "measured_steps": args.steps,
                      "workers": args.workers, "prefetch_factor": args.prefetch_factor,
                      "mean_compute_s": float(np.mean([r["compute_s"] for r in measured])),
                      "mean_next_batch_s": float(np.mean([r["next_batch_s"] for r in measured])),
                      "checkpoint_saved": False}), flush=True)


if __name__ == "__main__":
    main()
