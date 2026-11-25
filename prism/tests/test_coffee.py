"""Coffee normalization, history, and wire protocol regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prism_engram import codec
from prism_engram.coffee import CAMERA_KEYS, CAMERAS, METADATA, STATE_KEY, CoffeeRuntime
from prism_engram.server import connection_handler


class Policy:
    device = torch.device("cpu")

    def __init__(self):
        self.batches = []

    def sample(self, batch):
        self.batches.append(batch)
        result = torch.rand(1, 160, 7)
        result.reshape(32, 35)[:, 32:] = 100
        return result


@pytest.fixture
def runtime():
    return CoffeeRuntime(
        Policy(),
        {
            "mode": "lerobot",
            "state_dim": 32,
            "action_dim": 32,
            "token_width": 7,
            "state_mean": [1] * 32,
            "state_std": [2] * 32,
            "action_mean": [10] * 32,
            "action_std": [3] * 32,
            "config": {
                "dataset": {
                    "context_frames": 2,
                    "observation_hz": 10,
                    "state_key": STATE_KEY,
                    "action_key": "action",
                    "cameras": dict(zip(CAMERAS, CAMERA_KEYS, strict=True)),
                    "flip_horizontal": False,
                }
            },
        },
        SimpleNamespace(
            image_tensor=lambda image, *, flip_horizontal: image.unsqueeze(0),
            format_instruction=lambda prompt, **kwargs: (prompt, kwargs),
        ),
    )


def observation(timestamp, pixel=0):
    return {
        "timestamp_s": timestamp,
        "prompt": "make coffee",
        STATE_KEY: [3] * 32,
        **{key: np.full((2, 3, 3), pixel + i, dtype=np.uint8) for i, key in enumerate(CAMERA_KEYS)},
    }


def test_coffee_batch_and_action_denormalization(runtime):
    session = runtime.new_session()
    before = torch.random.get_rng_state().clone()
    actions = session.infer(observation(0, 10))["actions"]
    assert torch.equal(torch.random.get_rng_state(), before)
    assert actions.shape == (32, 32)
    assert actions.dtype == np.float32
    assert np.all((actions >= 10) & (actions <= 13))
    (batch,) = runtime.policy.batches
    assert batch["prompt"] == [
        (
            "make coffee",
            {
                "instruction_format": "json",
                "metadata": {"embodiment": "robot"},
                "include_subtask_instruction": True,
            },
        )
    ]
    assert tuple(batch["frames"].shape) == (1, 2, 3, 3, 2, 3)
    assert batch["frames"][0, 0, :, 0, 0, 0].tolist() == [10, 11, 12]
    assert torch.equal(batch["frames"][0, 0], batch["frames"][0, 1])
    np.testing.assert_array_equal(batch["state"], np.ones((1, 32)))
    assert batch["camera_mask"].all()
    assert not batch["proprio"].any()


def test_coffee_history_reset_and_client_isolation(runtime):
    first, second = runtime.new_session(), runtime.new_session()
    initial = first.infer(observation(0))["actions"]
    assert first.infer(observation(0.1, 20), decode=False) == {"observed": True}
    assert len(runtime.policy.batches) == 1
    with pytest.raises(ValueError, match="advance"):
        first.infer(observation(0.1, 50))
    with pytest.raises(ValueError, match="changing"):
        first.infer({**observation(0.2, 50), "prompt": "stop"})
    with pytest.raises(ValueError, match="finite"):
        first.infer({**observation(0.2, 50), STATE_KEY: [float("nan")] * 32})
    succeeding = first.infer(observation(0.2, 30))["actions"]
    assert runtime.policy.batches[-1]["frames"][0, :, 0, 0, 0, 0].tolist() == [20, 30]
    np.testing.assert_array_equal(second.infer(observation(0, 50))["actions"], initial)
    first.reset()
    np.testing.assert_array_equal(first.infer(observation(0, 50))["actions"], initial)
    assert not np.array_equal(succeeding, initial)


def test_coffee_protocol_metadata_and_actions(runtime):
    class Connection:
        replies = []

        def __iter__(self):
            yield codec.packb({"op": "reset", "seed": 12})
            yield codec.packb({"op": "observe", **observation(0)})
            yield codec.packb(observation(0.1))

        def send(self, payload):
            self.replies.append(codec.unpackb(payload))

    connection = Connection()
    connection_handler(runtime, connection, metadata=METADATA)
    metadata, reset, observe, infer = connection.replies
    assert metadata["model"] == "coffee_prism"
    assert metadata["camera_keys"] == list(CAMERA_KEYS)
    assert reset["seed"] == 12
    assert observe["observed"] is True
    assert infer["actions"].shape == (32, 32)
