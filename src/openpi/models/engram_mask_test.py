"""The real pi0 flow loss must not consume or supervise absent dimensions."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0


def test_pi05_flow_loss_and_gradient_ignore_missing_actions(monkeypatch):
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, observation, train: observation)

    class Expert:
        action_horizon = 2
        PaliGemma = SimpleNamespace(llm=lambda inputs, **kwargs: ([None, inputs[1]], None))
        action_out_proj = staticmethod(lambda x: x)

        def embed_prefix(self, observation):
            return jnp.zeros((1, 1, 32)), jnp.ones((1, 1), bool), jnp.zeros((1,), bool)

        def embed_suffix(self, observation, actions, time):
            return actions, jnp.ones((1, 2), bool), jnp.ones((2,), bool), None

    mask = jnp.ones((1, 32), bool).at[:, 20:22].set(False)
    observation = SimpleNamespace(action_mask=mask, images={"base_0_rgb": None, "left_wrist_0_rgb": None})
    rng = jax.random.key(5)
    actions = jnp.zeros((1, 2, 32)).at[:, :, :20].set(0.2)
    objective = lambda value: jnp.sum(pi0.Pi0.compute_loss(Expert(), rng, observation, value))
    loss, grad = jax.value_and_grad(objective)(actions)
    changed = actions.at[:, :, 20:22].set(10000)
    changed_loss, changed_grad = jax.value_and_grad(objective)(changed)
    np.testing.assert_allclose(loss, changed_loss)
    np.testing.assert_allclose(grad, changed_grad)
    np.testing.assert_array_equal(np.asarray(grad)[:, :, 20:22], 0)
    assert np.any(np.asarray(grad)[:, :, :20] != 0)


def test_accumulated_gradients_equal_explicit_microbatch_average():
    from scripts.train import _accumulate_microbatch_grads

    x = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 10
    targets = jnp.arange(8, dtype=jnp.float32)[:, None]
    params = jnp.array([[0.1], [-0.2], [0.3], [0.5]])

    def get_grad(key, inputs, labels):
        return jax.value_and_grad(lambda p: jnp.mean((inputs @ p - labels) ** 2))(params)

    key = jax.random.key(3)
    if jax.device_count() >= 2:
        mesh = jax.make_mesh((1, jax.device_count()), ("batch", "fsdp"))
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("batch", "fsdp")))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        x = jax.device_put(x, data_sharding)
        targets = jax.device_put(targets, data_sharding)
        fn = jax.jit(lambda a, b: _accumulate_microbatch_grads(get_grad, (a, b), key, 4),
                     in_shardings=(data_sharding, data_sharding), out_shardings=(replicated, replicated))
    else:
        fn = jax.jit(lambda a, b: _accumulate_microbatch_grads(get_grad, (a, b), key, 4))
    loss, gradient = fn(x, targets)
    expected_loss, expected_gradient = jax.value_and_grad(lambda p: jnp.mean((x @ p - targets) ** 2))(params)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6)


def test_checkpoint_labels_include_exact_completed_updates_and_resume():
    from scripts.train import _checkpoint_save_step

    legacy = [_checkpoint_save_step(i, 0, 20_001, 10_000, False) for i in range(20_001)]
    assert [step for step in legacy if step is not None] == [10_000, 20_000]
    completed = [_checkpoint_save_step(i, 0, 30_000, 10_000, True) for i in range(30_000)]
    assert [step for step in completed if step is not None] == [10_000, 20_000, 30_000]
    resumed = [_checkpoint_save_step(i, 10_000, 30_000, 10_000, True) for i in range(10_000, 30_000)]
    assert [step for step in resumed if step is not None] == [20_000, 30_000]
