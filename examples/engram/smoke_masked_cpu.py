"""Exercise masked-action gradients through a small real π0.5 on CPU."""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config, siglip

original_module = siglip.Module


def tiny_vision(*args, **kwargs):
    kwargs["variant"] = "mu/14"
    return original_module(*args, **kwargs)


siglip.Module = tiny_vision
config = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=2, max_token_len=16,
                              paligemma_variant="dummy", action_expert_variant="dummy")
model = config.create(jax.random.key(1))
obs = dataclasses.replace(config.fake_obs(batch_size=1), action_mask=jnp.ones((1, 32), bool).at[:, 20:22].set(False))
actions = config.fake_act(batch_size=1)
key = jax.random.key(7)
objective = lambda values: jnp.mean(model.compute_loss(key, obs, values, train=False))
evaluate = jax.jit(jax.value_and_grad(objective))
loss, grad = evaluate(actions)
other_loss, other_grad = evaluate(actions.at[:, :, 20:22].set(1000.0))
np.testing.assert_allclose(loss, other_loss, rtol=1e-6)
np.testing.assert_allclose(grad, other_grad, rtol=1e-6, atol=1e-6)
np.testing.assert_array_equal(np.asarray(grad)[:, :, 20:22], 0)
assert np.any(np.asarray(grad)[:, :, :20] != 0)
print("tiny_real_pi05_masked_loss", float(loss), "missing_grad_max", float(jnp.max(jnp.abs(grad[:, :, 20:22]))))

# Compare the optional base-only path with the ordinary three-camera input
# when both wrist slots are absent. Both invoke the real π0.5 transformer.
full_obs = dataclasses.replace(
    obs,
    image_masks={**obs.image_masks, "left_wrist_0_rgb": jnp.zeros((1,), bool), "right_wrist_0_rgb": jnp.zeros((1,), bool)},
)
base_obs = dataclasses.replace(
    full_obs, images={"base_0_rgb": full_obs.images["base_0_rgb"]},
    image_masks={"base_0_rgb": full_obs.image_masks["base_0_rgb"]},
)


def compare(observation, train=False):
    objective = lambda values: jnp.mean(model.compute_loss(key, observation, values, train=train))
    return jax.jit(jax.value_and_grad(objective))(actions)


full_loss, full_grad = compare(full_obs)
base_loss, base_grad = compare(base_obs)
np.testing.assert_allclose(full_loss, base_loss, rtol=2e-5, atol=2e-5)
np.testing.assert_allclose(full_grad, base_grad, rtol=2e-4, atol=2e-5)
print("single_camera_loss_delta", float(jnp.abs(full_loss - base_loss)),
      "max_action_gradient_delta", float(jnp.max(jnp.abs(full_grad - base_grad))))
full_train_loss, full_train_grad = compare(full_obs, train=True)
base_train_loss, base_train_grad = compare(base_obs, train=True)
np.testing.assert_allclose(full_train_loss, base_train_loss, rtol=2e-5, atol=2e-5)
np.testing.assert_allclose(full_train_grad, base_train_grad, rtol=2e-4, atol=2e-5)
print("single_camera_training_loss_delta", float(jnp.abs(full_train_loss - base_train_loss)),
      "max_action_gradient_delta", float(jnp.max(jnp.abs(full_train_grad - base_train_grad))))


def parameter_loss(module, observation):
    return jnp.mean(module.compute_loss(key, observation, actions, train=True))


parameter_grad = nnx.jit(nnx.value_and_grad(parameter_loss, argnums=nnx.DiffState(0, nnx.Param)))
full_value, full_parameter_grad = parameter_grad(model, full_obs)
base_value, base_parameter_grad = parameter_grad(model, base_obs)
np.testing.assert_allclose(full_value, base_value, rtol=2e-5, atol=2e-5)
full_leaves, base_leaves = jax.tree.leaves(full_parameter_grad), jax.tree.leaves(base_parameter_grad)
assert len(full_leaves) == len(base_leaves) and full_leaves
max_diff = 0.0
for first, second in zip(full_leaves, base_leaves, strict=True):
    first, second = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    np.testing.assert_allclose(first, second, rtol=2e-3, atol=2e-4)
    max_diff = max(max_diff, float(np.max(np.abs(first - second))))
print("single_camera_training_parameter_gradient_max_delta", max_diff)
