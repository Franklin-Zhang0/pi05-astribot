"""Astribot inference geometry and stateful WebSocket contract."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from prism_engram.astribot_mobile import (
    AstribotSession,
    camera_observation,
    decode_actions,
    pack_proprio,
)


def measurement(t=0.0):
    quat = Rotation.from_euler("x", 0.5).as_quat()
    return {
        "timestamp_s": t,
        "stereo_left": np.ones((3, 4, 3), dtype=np.uint8),
        "head_rgb": np.zeros((3, 4, 3), dtype=np.uint8),
        "right_wrist": np.ones((3, 4, 3), dtype=np.uint8) * 2,
        "eef_pose": np.array([[1, 2, 3, *quat], [3, 4, 5, 0, 0, 0, 1]]),
        "gripper_percent": [20, 90],
        "chassis_xyyaw": [4, 3, 2],
        "previous_chassis_xyyaw": [3, 2, 1],
        "subtask": "pick up the cup",
    }


def test_stereo_priority_head_fallback_and_wrist_slots():
    sample = measurement()
    main, left, right = camera_observation(sample)
    assert main is sample["stereo_left"]
    assert left is None and right is sample["right_wrist"]
    sample["stereo_left"] = None
    assert camera_observation(sample)[0] is sample["head_rgb"]
    sample["stereo_left"] = sample["head_rgb"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        camera_observation(sample)


def test_proprio_and_denormalization_match_training_slots():
    sample = measurement()
    packed, xyz, rotation, base = pack_proprio(sample, 2, first=False)
    packed = packed.reshape(5, 15)
    np.testing.assert_allclose(packed[0, :3], [1, 1, 1])
    np.testing.assert_allclose(packed[[1, 3], :3], xyz / 2)
    np.testing.assert_allclose(packed[[1, 3], 3:9], rotation[:, :2].reshape(2, 6))
    np.testing.assert_allclose(packed[[2, 4], 0], [0.2, 0.9])

    mean = np.zeros((5, 15), dtype=np.float32)
    delta = Rotation.from_euler("z", 0.4).as_matrix()
    mean[[1, 3], 3:9] = delta[:2].reshape(6)
    mean[0, :3] = [0.25, -0.1, 0.05]
    mean[[2, 4], 0] = [0.2, 1.2]
    std = np.full((5, 15), 0.5, dtype=np.float32)
    pred = torch.zeros(1, 32, 5, 15)
    pred[..., 1, 0] = 0.2
    pred[..., 3, 2] = -0.2
    result = decode_actions(pred.reshape(1, 160, 15), mean, std, 2, xyz, rotation, base)
    assert result["eef_pose"].shape == (32, 2, 7)
    np.testing.assert_allclose(result["eef_pose"][0, :, :3], xyz + [[0.2, 0, 0], [0, 0, -0.2]])
    np.testing.assert_allclose(Rotation.from_quat(result["eef_pose"][0, :, 3:]).as_matrix(),
                               delta @ rotation, atol=1e-7)
    np.testing.assert_allclose(result["chassis_xyyaw"][0], [4.25, 2.9, 2.05])
    np.testing.assert_allclose(result["gripper_percent"][0], [20, 100])


def test_chassis_requires_the_previous_20_hz_state_when_enabled():
    sample = measurement()
    sample.pop("previous_chassis_xyyaw")
    np.testing.assert_array_equal(pack_proprio(sample, 1, first=True)[0][:3], [0, 0, 0])
    with pytest.raises(ValueError, match="20 Hz"):
        pack_proprio(sample, 1, first=False)
    sample.pop("chassis_xyyaw")
    assert pack_proprio(sample, 1, first=False)[-1] is None


def test_observe_refresh_and_rng_are_session_local():
    class Runtime:
        device = torch.device("cpu")
        position_scale = 1
        robot_frame_id = 2

        def __init__(self):
            from threading import RLock
            self.lock = RLock()
            self.model = self
            self.mean = np.zeros((5, 15))
            self.mean[[1, 3], 3:9] = [1, 0, 0, 0, 1, 0]
            self.std = np.ones((5, 15))
            self.calls = []

        def prompt_tokens(self, task, subtask):
            return (task, subtask)

        def visual_features(self, views):
            return None, tuple(view is not None for view in views), (16, 16)

        def inference_reset(self, *_):
            return 0

        def _model_dtype(self):
            return torch.float32

        def inference_step(self, cache, *_args, **kwargs):
            self.calls.append((cache, kwargs))
            prediction = torch.zeros(1, 32, 5, 15)
            if kwargs["decode_actions"]:
                prediction[..., 0, :3] = kwargs["flow_initial_noise"].reshape(1, 32, 5, 15)[..., 0, :3]
            return prediction.reshape(1, 160, 15), cache + 1

    runtime = Runtime()
    first, second = AstribotSession(runtime), AstribotSession(runtime)
    assert first.infer(measurement(), decode=False)["observed"] is True
    assert runtime.calls[-1][1]["flow_initial_noise"] is None
    result = first.infer({**measurement(0.1), "subtask": "place the cup"})
    cache, args = runtime.calls[-1]
    assert cache == 1 and args["prefix_refresh_mask"].item()
    assert args["instruction_mask"] == "place the cup"
    assert args["dino_camera_mask_new"] == (True, False, True)
    np.testing.assert_array_equal(result["chassis_xyyaw"], second.infer(measurement())["chassis_xyyaw"])
    with pytest.raises(ValueError, match="10 Hz"):
        first.infer(measurement(0.5))
