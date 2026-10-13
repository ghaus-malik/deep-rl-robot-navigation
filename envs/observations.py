"""Observation construction for the navigation task (pure PyTorch).

Layout of the 133-dimensional observation vector:

    [0:120]    LiDAR, 120 sectors of 3 deg (min of 3 consecutive 1-deg rays), metres
    [120:125]  heading error to the goal, radians in (-pi, pi], repeated 5 times
    [125:130]  distance to the goal, metres, repeated 5 times
    [130:132]  robot linear velocity x, y in the WORLD frame, m/s
    [132]      robot angular velocity about z in the WORLD frame, rad/s

The heading/distance repetition matters together with the per-episode observation noise: every copy
receives an independent noise offset, so the policy can average them.
"""

from __future__ import annotations

import torch

NUM_LIDAR_SECTORS = 120
GOAL_REPEATS = 5

LIDAR_SLICE = slice(0, NUM_LIDAR_SECTORS)
HEADING_SLICE = slice(NUM_LIDAR_SECTORS, NUM_LIDAR_SECTORS + GOAL_REPEATS)
DISTANCE_SLICE = slice(HEADING_SLICE.stop, HEADING_SLICE.stop + GOAL_REPEATS)
LIN_VEL_SLICE = slice(DISTANCE_SLICE.stop, DISTANCE_SLICE.stop + 2)
ANG_VEL_INDEX = LIN_VEL_SLICE.stop
OBS_DIM = ANG_VEL_INDEX + 1  # 133


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def goal_distance(robot_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
    """Planar Euclidean distance robot -> goal, shape (N,)."""
    return torch.linalg.norm(goal_xy - robot_xy, dim=-1)


def heading_to_goal(robot_xy: torch.Tensor, robot_yaw: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
    """Signed angle the robot must turn to face the goal, in (-pi, pi].

    Positive = goal is to the left (counter-clockwise), negative = goal is to the right.
    """
    delta = goal_xy - robot_xy
    bearing = torch.atan2(delta[:, 1], delta[:, 0])
    return wrap_to_pi(bearing - robot_yaw)


def build_observation(
    lidar_sectors: torch.Tensor,
    heading: torch.Tensor,
    distance: torch.Tensor,
    lin_vel_xy_w: torch.Tensor,
    ang_vel_z_w: torch.Tensor,
) -> torch.Tensor:
    """Concatenate all terms in the fixed order documented at the top of this module."""
    if lidar_sectors.shape[-1] != NUM_LIDAR_SECTORS:
        raise ValueError(f"Expected {NUM_LIDAR_SECTORS} LiDAR sectors, got {lidar_sectors.shape[-1]}.")
    return torch.cat(
        (
            lidar_sectors,
            heading.unsqueeze(-1).repeat(1, GOAL_REPEATS),
            distance.unsqueeze(-1).repeat(1, GOAL_REPEATS),
            lin_vel_xy_w,
            ang_vel_z_w.unsqueeze(-1),
        ),
        dim=-1,
    )
