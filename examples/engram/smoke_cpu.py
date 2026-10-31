"""Run a random tiny π0.5 optimizer step on real data; never loads a checkpoint."""

import argparse
import dataclasses
import functools
from pathlib import Path
import sys

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.models import siglip
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import sharding
from openpi.training import weight_loaders

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import train

parser = argparse.ArgumentParser()
parser.add_argument("--assets-base-dir", default="/tmp/engram-pi-smoke-final")
parser.add_argument("--data-root", default=config_lib.EngramDataConfig().data_root)
args = parser.parse_args()
base = config_lib.get_config("pi05_engram_w1_sandwich")
tiny_model = pi0_config.Pi0Config(
    pi05=True, action_dim=32, action_horizon=64, paligemma_variant="dummy", action_expert_variant="dummy"
)
config = dataclasses.replace(
    base,
    model=tiny_model,
    batch_size=1,
    num_workers=0,
    assets_base_dir=args.assets_base_dir,
    weight_loader=weight_loaders.NoOpWeightLoader(),
    data=dataclasses.replace(base.data, allow_sampled_stats=True, data_root=args.data_root),
    ema_decay=None,
    fsdp_devices=1,
    wandb_enabled=False,
)
for split in ("train", "val"):
    current = dataclasses.replace(config, data=dataclasses.replace(config.data, split=split))
    loader = data_loader.create_data_loader(current, num_batches=1)
    obs, actions = next(iter(loader))
    print(
        "real_batch",
        split,
        "state",
        obs.state.shape,
        "actions",
        actions.shape,
        "images",
        {k: v.shape for k, v in obs.images.items()},
        "masks",
        {k: np.asarray(v).tolist() for k, v in obs.image_masks.items()},
        "finite_actions",
        bool(np.isfinite(np.asarray(actions)).all()),
        flush=True,
    )
    if split == "train":
        train_batch = (obs, actions)

original_module = siglip.Module


def tiny_vision(*args, **kwargs):
    kwargs["variant"] = "mu/14"
    return original_module(*args, **kwargs)


siglip.Module = tiny_vision
print("initializing random dummy π0.5 and tiny SigLIP", flush=True)
mesh = sharding.make_mesh(1)
key = jax.random.key(1)
with sharding.set_mesh(mesh):
    state, _ = train.init_train_state(config, key, mesh, resume=False)
    print("running CPU optimizer step", flush=True)
    fn = jax.jit(functools.partial(train.train_step, config))
    state, metrics = fn(key, state, train_batch)
    metrics = jax.device_get(metrics)
print("step", int(state.step), "loss", float(metrics["loss"]), "grad_norm", float(metrics["grad_norm"]), flush=True)
assert int(state.step) == 1
assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics))
