import numpy as np

from examples.engram.evaluate_pi05 import evaluate_samples


class _Dataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        target = np.zeros((4, 22), dtype=np.float32)
        target[:, :20] = 0.5
        target[:, 20:] = [20, 40]
        return {"actions": target, "observation/joint_position": np.zeros(22), "prompt": "sandwich"}


class _ZeroPolicy:
    def infer(self, observation):
        assert "actions" not in observation
        return {"actions": np.zeros((4, 22), dtype=np.float32)}


def test_validation_reports_action_errors_in_physical_units():
    result = evaluate_samples(_Dataset(), _ZeroPolicy(), max_samples=2, seed=42)
    assert result["num_anchors"] == 2
    assert result["action_steps"] == 8
    assert result["joint_rmse_rad"] == 0.5
    assert result["left_gripper_rmse_percent"] == 20
    assert result["right_gripper_rmse_percent"] == 40
