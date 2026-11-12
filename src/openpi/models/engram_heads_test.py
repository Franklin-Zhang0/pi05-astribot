"""Routed projections must obey the same contract in flow loss and sampling."""

import dataclasses
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import pi0_config

NAMES = ("dex_w1_gripper", "dex_w1_linker_l6", "dex_w1_linker_l20", "galbot_inspire_rh56dfx_hand")


class IdentityLLM(nnx.Module):
    def __call__(self, inputs, **kwargs):
        return [None, inputs[1]], None


class RoutedExpert(nnx.Module):
    """A small expert exercising the production routing/loss/sampler methods."""

    def __init__(self):
        self.action_head_names = NAMES
        self.action_dim = 4
        self.action_horizon = 2
        self.action_in_projs = nnx.Dict({name: nnx.Linear(4, 4, rngs=nnx.Rngs(i)) for i, name in enumerate(NAMES)})
        self.action_out_projs = nnx.Dict({name: nnx.Linear(4, 4, rngs=nnx.Rngs(i)) for i, name in enumerate(NAMES)})
        for i, name in enumerate(NAMES):
            self.action_in_projs[name].kernel.value = jnp.eye(4)
            self.action_in_projs[name].bias.value = jnp.full((4,), i + 10.0)
            self.action_out_projs[name].kernel.value = jnp.zeros((4, 4))
            self.action_out_projs[name].bias.value = jnp.full((4,), i + 1.0)
        self.PaliGemma = nnx.Dict(llm=IdentityLLM())

    def embed_prefix(self, observation):
        b = observation.state.shape[0]
        return jnp.zeros((b, 1, 4)), jnp.ones((b, 1), bool), jnp.zeros((1,), bool)

    def embed_suffix(self, observation, actions, time):
        values = pi0.Pi0._project_actions(self, actions, observation)  # noqa: SLF001
        return values, jnp.ones(values.shape[:2], bool), jnp.ones((values.shape[1],), bool), None


def observation(ids, mask=None):
    ids = jnp.asarray(ids, dtype=jnp.int32)
    b = ids.shape[0]
    return model_lib.Observation(
        images={}, image_masks={}, state=jnp.zeros((b, 4)),
        action_mask=jnp.ones((b, 4), bool) if mask is None else mask, embodiment_id=ids,
    )


def test_projection_routes_mixed_batch_without_prompt_metadata():
    model = RoutedExpert()
    obs = observation([0, 1, 2, 3])
    values = jnp.zeros((4, 2, 4))
    for output, offset in ((False, 10), (True, 1)):
        actual = jax.jit(lambda x, output=output: pi0.Pi0._project_actions(model, x, obs, output=output))(values)  # noqa: SLF001
        expected = np.broadcast_to(np.arange(4)[:, None, None] + offset, (4, 2, 4))
        np.testing.assert_array_equal(actual, expected)


def test_only_selected_projection_receives_gradient():
    model = RoutedExpert()
    obs = observation([2, 2])
    values = jnp.ones((2, 2, 4))
    def objective(m):
        return jnp.sum(pi0.Pi0._project_actions(m, values, obs, output=True))  # noqa: SLF001

    grads = nnx.grad(objective)(model).to_pure_dict()
    for i, name in enumerate(NAMES):
        leaves = jax.tree.leaves(grads["action_out_projs"][name])
        if i == 2:
            assert all(np.any(np.asarray(leaf) != 0) for leaf in leaves)
        else:
            assert all(np.all(np.asarray(leaf) == 0) for leaf in leaves)
        assert all(np.all(np.asarray(leaf) == 0) for leaf in jax.tree.leaves(grads["action_in_projs"][name]))


def test_routed_flow_loss_ignores_missing_and_padded_targets(monkeypatch):
    monkeypatch.setattr(pi0, "_preprocess_observation", lambda rng, observation, train: observation)
    model = RoutedExpert()
    mask = jnp.array([[True, True, False, False], [True, False, True, False]])
    obs = observation([0, 3], mask)
    actions = jnp.full((2, 2, 4), 0.2)
    rng = jax.random.key(4)
    def objective(a):
        return jnp.sum(pi0.Pi0.compute_loss(model, rng, obs, a))

    original_loss, original_grad = jax.value_and_grad(objective)(actions)
    changed = jnp.where(mask[:, None, :], actions, jnp.nan)
    changed_loss, changed_grad = jax.value_and_grad(objective)(changed)
    assert np.isfinite(changed_loss)
    np.testing.assert_array_equal(original_loss, changed_loss)
    np.testing.assert_array_equal(original_grad, changed_grad)
    np.testing.assert_array_equal(np.asarray(changed_grad)[~np.broadcast_to(mask[:, None, :], actions.shape)], 0)


def test_flow_loss_only_updates_heads_used_by_the_batch(monkeypatch):
    monkeypatch.setattr(pi0, "_preprocess_observation", lambda rng, observation, train: observation)
    model = RoutedExpert()
    for name in NAMES:
        model.action_out_projs[name].kernel.value = jnp.eye(4)
    obs = observation([0, 3])
    actions = jnp.full((2, 2, 4), 0.2)

    def objective(m):
        return jnp.sum(pi0.Pi0.compute_loss(m, jax.random.key(5), obs, actions))

    grads = nnx.grad(objective)(model).to_pure_dict()
    for i, name in enumerate(NAMES):
        for projection in ("action_in_projs", "action_out_projs"):
            leaves = jax.tree.leaves(grads[projection][name])
            if i in (0, 3):
                assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)
            else:
                assert all(np.all(np.asarray(leaf) == 0) for leaf in leaves)


def test_sampler_uses_selected_head_at_every_step_and_zeros_unavailable_channels(monkeypatch):
    monkeypatch.setattr(pi0, "_preprocess_observation", lambda rng, observation, train: observation)
    model = RoutedExpert()
    mask = jnp.ones((4, 4), bool).at[:, 3].set(False)
    obs = observation([0, 1, 2, 3], mask)
    values = pi0.Pi0.sample_actions(model, jax.random.key(3), obs, num_steps=4, noise=jnp.zeros((4, 2, 4)))
    expected = np.broadcast_to(-np.arange(1, 5)[:, None, None], (4, 2, 4)).copy()
    expected[:, :, 3] = 0
    np.testing.assert_allclose(values, expected)


def test_routes_reject_missing_metadata_and_do_not_wrap_invalid_ids():
    model = RoutedExpert()
    obs = observation([0])
    values = jnp.zeros((1, 2, 4))
    with pytest.raises(ValueError, match="requires embodiment_id"):
        pi0.Pi0._project_actions(model, values, dataclasses.replace(obs, embodiment_id=None))  # noqa: SLF001
    with pytest.raises(ValueError, match="requires action_mask"):
        pi0.Pi0._project_actions(model, values, dataclasses.replace(obs, action_mask=None))  # noqa: SLF001
    malformed = SimpleNamespace(action_mask=jnp.ones((1, 3), bool), embodiment_id=obs.embodiment_id)
    with pytest.raises(ValueError, match="every packed action dimension"):
        pi0.Pi0._project_actions(model, values, malformed)  # noqa: SLF001
    for route in (-1, 4):
        result = pi0.Pi0._project_actions(model, values, observation([route]))  # noqa: SLF001
        assert np.isnan(result).all()


def test_observation_route_survives_round_trip_and_preprocessing():
    obs = model_lib.Observation(
        images={"base_0_rgb": jnp.zeros((2, 224, 224, 3))}, image_masks={"base_0_rgb": jnp.ones((2,), bool)},
        state=jnp.zeros((2, 64)), embodiment_id=jnp.array([1, 3], dtype=jnp.int32),
    )
    recovered = model_lib.Observation.from_dict(obs.to_dict())
    prepared = model_lib.preprocess_observation(None, recovered, image_keys=("base_0_rgb",))
    np.testing.assert_array_equal(prepared.embodiment_id, [1, 3])


def test_config_defaults_preserve_legacy_and_multi_head_spec_is_explicit():
    legacy = pi0_config.Pi0Config(pi05=True)
    assert legacy.action_head_names == ()
    assert legacy.inputs_spec()[0].embodiment_id is None
    multi = pi0_config.Pi0Config(pi05=True, action_dim=64, max_token_len=320, action_head_names=NAMES)
    spec, action = multi.inputs_spec(batch_size=2)
    assert spec.embodiment_id.shape == (2,)
    assert spec.action_mask.shape == (2, 64)
    assert action.shape == (2, multi.action_horizon, 64)
    np.testing.assert_array_equal(multi.fake_obs(batch_size=2).embodiment_id, 0)
    with pytest.raises(ValueError, match="require pi05"):
        pi0_config.Pi0Config(action_head_names=NAMES)
    with pytest.raises(ValueError, match="unique lowercase"):
        pi0_config.Pi0Config(pi05=True, action_head_names=("same", "same"))
