import pytest
import torch

from robots.diff_drive import scale_actions, unicycle_to_wheel_velocities, wheel_to_unicycle_velocities

LOW, HIGH = (0.1, -0.5), (0.5, 0.5)
R, B = 0.0325, 0.1125


def test_action_scaling_endpoints_and_midpoint():
    a = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    cmd = scale_actions(a, LOW, HIGH)
    assert torch.allclose(cmd, torch.tensor([[0.1, -0.5], [0.5, 0.5], [0.3, 0.0]]))


def test_action_scaling_clamps_out_of_range_outputs():
    cmd = scale_actions(torch.tensor([[5.0, -7.0]]), LOW, HIGH)
    assert torch.allclose(cmd, torch.tensor([[0.5, -0.5]]))


def test_forward_motion_equal_wheel_speeds():
    w = unicycle_to_wheel_velocities(torch.tensor([0.3]), torch.tensor([0.0]), R, B)
    assert w[0, 0].item() == pytest.approx(0.3 / R)
    assert w[0, 1].item() == pytest.approx(0.3 / R)


def test_counter_clockwise_rotation_right_wheel_faster():
    w = unicycle_to_wheel_velocities(torch.tensor([0.1]), torch.tensor([0.5]), R, B)
    assert w[0, 1] > w[0, 0]


def test_clockwise_rotation_left_wheel_faster():
    w = unicycle_to_wheel_velocities(torch.tensor([0.1]), torch.tensor([-0.5]), R, B)
    assert w[0, 0] > w[0, 1]


def test_pure_rotation_opposite_wheels():
    w = unicycle_to_wheel_velocities(torch.tensor([0.0]), torch.tensor([0.5]), R, B)
    assert w[0, 0].item() == pytest.approx(-w[0, 1].item())
    assert w[0, 1].item() == pytest.approx(0.5 * B / (2 * R))


def test_wheel_kinematics_round_trip():
    v, om = torch.tensor([0.1, 0.27, 0.5]), torch.tensor([-0.5, 0.13, 0.5])
    v2, om2 = wheel_to_unicycle_velocities(unicycle_to_wheel_velocities(v, om, R, B), R, B)
    assert torch.allclose(v, v2, atol=1e-6) and torch.allclose(om, om2, atol=1e-6)


def test_limits_bound_wheel_speeds():
    corners = scale_actions(torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]), LOW, HIGH)
    w = unicycle_to_wheel_velocities(corners[:, 0], corners[:, 1], R, B)
    max_expected = (2 * 0.5 + 0.5 * B) / (2 * R)
    assert w.abs().max().item() == pytest.approx(max_expected)
    assert torch.all(corners[:, 0] > 0)  # the robot can never drive backwards
