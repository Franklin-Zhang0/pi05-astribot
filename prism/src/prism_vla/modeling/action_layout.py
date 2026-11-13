"""Structural sampling constraints for explicitly declared action layouts."""

import torch

from prism_vla.data.datasets.ego_schema import (
    EGO_SEPARATED_ACTION_TOKENS_PER_STEP,
    ROTATION_REPRESENTATION_6D,
    ego_action_token_dim,
    rotation_representation_dim,
)


def robot_action_sampling_mask(
    robot_action_layout: str, *, action_horizon: int, action_dim: int,
    action_token_types: tuple[str, ...] = (),
) -> torch.Tensor | None:
    """Return active coordinates, not a per-example supervision/validity mask.

    The v1 robot extension retains the factorized ego packing (head, left EEF,
    left hand, right EEF, right hand). RobotEgoSource supplies head/right EEF;
    EgoRobotTrainingBackend adds gripper and optional RoboCasa base controls.
    All other targets are structurally zero, including absent body parts whose
    output regressors may be shared with present parts. Unspecified layouts
    intentionally retain legacy sampling, even for the same tensor dimensions.
    The camera-masked v3 command layout additionally removes the entire head
    token while retaining the EEF command and gripper coordinates.
    """
    if not robot_action_layout:
        return None
    if robot_action_layout not in {
        "ego_robot_controls_v1:libero", "ego_robot_controls_v1:robocasa365",
        "ego_robot_command_trajectory_v2:libero",
        "ego_robot_command_trajectory_v3_camera_masked:libero",
        "ego_robot_command_trajectory_v4_world:libero",
        "ego_robot_command_trajectory_v5_camera_hand_gripper:libero",
        "ego_robot_absolute_trajectory_v1_camera_hand_gripper:robodojo",
    }:
        raise ValueError(f"Unsupported robot_action_layout: {robot_action_layout!r}")
    tokens = EGO_SEPARATED_ACTION_TOKENS_PER_STEP
    width = ego_action_token_dim(ROTATION_REPRESENTATION_6D, tokens)
    if action_dim != width or action_horizon < 1 or action_horizon % tokens:
        raise ValueError("robot_action_layout requires complete five-token 15D action steps")
    if action_token_types and tuple(action_token_types) != ("head", "eef", "hand", "eef", "hand"):
        raise ValueError("robot_action_layout requires head/eef/hand/eef/hand token order")
    pose_dim = 3 + rotation_representation_dim(ROTATION_REPRESENTATION_6D)
    active = torch.zeros(tokens, width, dtype=torch.bool)
    if robot_action_layout == "ego_robot_absolute_trajectory_v1_camera_hand_gripper:robodojo":
        active[[1, 3], :pose_dim] = True
        active[[2, 4], 0] = True
        return active.repeat(action_horizon // tokens, 1).unsqueeze(0)
    if robot_action_layout not in {"ego_robot_command_trajectory_v3_camera_masked:libero",
                                    "ego_robot_command_trajectory_v4_world:libero",
                                    "ego_robot_command_trajectory_v5_camera_hand_gripper:libero"}:
        active[0, :pose_dim] = True
    active[3, :pose_dim] = True
    if robot_action_layout == "ego_robot_command_trajectory_v5_camera_hand_gripper:libero":
        active[4, 0] = True  # Right hand token carries gripper command.
    else:
        active[3, pose_dim] = True  # Historical layouts use EEF padding.
    if robot_action_layout == "ego_robot_controls_v1:robocasa365":
        active[0, pose_dim:pose_dim + 5] = True
    return active.repeat(action_horizon // tokens, 1).unsqueeze(0)
