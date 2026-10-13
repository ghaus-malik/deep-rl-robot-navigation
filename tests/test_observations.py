import math

import pytest
import torch

from envs.lidar_processing import clean_ranges, downsample_min, planar_ray_angles
from envs.observations import (
    ANG_VEL_INDEX,
    DISTANCE_SLICE,
    HEADING_SLICE,
    LIDAR_SLICE,
    LIN_VEL_SLICE,
    OBS_DIM,
    build_observation,
    goal_distance,
    heading_to_goal,
    wrap_to_pi,
)


def test_observation_dimension_is_133():
    assert OBS_DIM == 133
    obs = build_observation(torch.ones(4, 120), torch.zeros(4), torch.zeros(4), torch.zeros(4, 2), torch.zeros(4))
    assert obs.shape == (4, 133)


def test_observation_ordering_and_values():
    lidar = torch.linspace(0.15, 2.0, 120).unsqueeze(0)
    obs = build_observation(lidar, torch.tensor([0.7]), torch.tensor([2.5]), torch.tensor([[0.3, -0.1]]), torch.tensor([0.4]))
    assert torch.equal(obs[0, LIDAR_SLICE], lidar[0])
    assert torch.allclose(obs[0, HEADING_SLICE], torch.full((5,), 0.7))
    assert torch.allclose(obs[0, DISTANCE_SLICE], torch.full((5,), 2.5))
    assert torch.allclose(obs[0, LIN_VEL_SLICE], torch.tensor([0.3, -0.1]))
    assert obs[0, ANG_VEL_INDEX].item() == pytest.approx(0.4)


def test_build_observation_rejects_wrong_lidar_size():
    with pytest.raises(ValueError):
        build_observation(torch.ones(1, 360), torch.zeros(1), torch.zeros(1), torch.zeros(1, 2), torch.zeros(1))


@pytest.mark.parametrize(
    "goal, yaw, expected",
    [
        ((1.0, 0.0), 0.0, 0.0),  # straight ahead
        ((0.0, 1.0), 0.0, math.pi / 2),  # to the left -> positive
        ((0.0, -1.0), 0.0, -math.pi / 2),  # to the right -> negative
        ((1.0, 0.0), math.pi / 2, -math.pi / 2),  # robot faces +y, goal on its right
    ],
)
def test_heading_sign_convention(goal, yaw, expected):
    h = heading_to_goal(torch.zeros(1, 2), torch.tensor([yaw]), torch.tensor([goal]))
    assert h.item() == pytest.approx(expected, abs=1e-6)


def test_heading_wraps_across_pi():
    # robot yaw 3.0 rad, goal bearing -3.0 rad: the short way is +0.283 rad, not -6.0 rad
    goal = torch.tensor([[math.cos(-3.0), math.sin(-3.0)]])
    h = heading_to_goal(torch.zeros(1, 2), torch.tensor([3.0]), goal)
    assert h.item() == pytest.approx(2 * math.pi - 6.0, abs=1e-5)
    assert -math.pi < h.item() <= math.pi


def test_wrap_to_pi_range():
    angles = torch.linspace(-20, 20, 1001)
    wrapped = wrap_to_pi(angles)
    assert torch.all(wrapped > -math.pi - 1e-6) and torch.all(wrapped <= math.pi + 1e-6)
    assert torch.allclose(torch.sin(wrapped), torch.sin(angles), atol=1e-5)


def test_goal_distance_is_planar_metres():
    d = goal_distance(torch.tensor([[1.0, 1.0]]), torch.tensor([[4.0, 5.0]]))
    assert d.item() == pytest.approx(5.0)


def test_ray_angles_ordering():
    a = planar_ray_angles(360, -180.0, 1.0)
    assert a.shape == (360,)
    assert a[0].item() == pytest.approx(-math.pi)  # backwards
    assert a[180].item() == pytest.approx(0.0)  # forwards
    assert a[270].item() == pytest.approx(math.pi / 2)  # left
    assert torch.allclose(a[1:] - a[:-1], torch.full((359,), math.radians(1.0)), atol=1e-6)


def test_clean_ranges_invalid_and_limits():
    raw = torch.tensor([[float("inf"), float("nan"), 0.05, 1.0, 5.0]])
    cleaned = clean_ranges(raw, 0.15, 2.0)
    assert torch.allclose(cleaned, torch.tensor([[2.0, 2.0, 0.15, 1.0, 2.0]]))


def test_downsample_min_groups_of_three():
    ranges = torch.arange(360, dtype=torch.float32).flip(0).unsqueeze(0)  # 359 ... 0
    sectors = downsample_min(ranges, 3)
    assert sectors.shape == (1, 120)
    assert sectors[0, 0].item() == 357.0  # min of 359, 358, 357
    assert sectors[0, -1].item() == 0.0
    with pytest.raises(ValueError):
        downsample_min(torch.ones(1, 10), 3)
