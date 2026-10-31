"""One FSDP-sharded, accumulated π0.5 update on two virtual CPU devices."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config, siglip
from openpi.training import config as config_lib, sharding, weight_loaders
from scripts import train

if jax.device_count() != 2:
    raise ValueError("Set XLA_FLAGS=--xla_force_host_platform_device_count=2")
original_module = siglip.Module


def tiny_vision(*args, **kwargs):
    kwargs["variant"] = "mu/14"
    return original_module(*args, **kwargs)


siglip.Module = tiny_vision
model_config = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=2, max_token_len=16,
                                    paligemma_variant="dummy", action_expert_variant="dummy")
base = config_lib.get_config("pi05_engram_w1_sandwich")
config = dataclasses.replace(base, model=model_config, batch_size=4, gradient_accumulation_steps=2,
                             fsdp_devices=2, weight_loader=weight_loaders.NoOpWeightLoader(),
                             ema_decay=None, wandb_enabled=False)
mask = jnp.ones((4, 32), bool).at[2:, 20:22].set(False)
obs = dataclasses.replace(model_config.fake_obs(batch_size=4), action_mask=mask)
actions = model_config.fake_act(batch_size=4)
mesh = sharding.make_mesh(2)
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
key = jax.random.key(2)
with sharding.set_mesh(mesh):
    state, state_sharding = train.init_train_state(config, key, mesh, resume=False)
    batch = jax.device_put((obs, actions), data_sharding)
    update = jax.jit(functools.partial(train.train_step, config),
                     in_shardings=(replicated, state_sharding, data_sharding),
                     out_shardings=(state_sharding, replicated))
    state, metrics = update(key, state, batch)
    metrics = jax.device_get(metrics)
assert int(state.step) == 1 and all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
print("two_cpu_fsdp_accumulated_optimizer_step", int(state.step), "loss", float(metrics["loss"]))
