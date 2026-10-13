"""Differential-drive (unicycle) kinematics.

Pure PyTorch so it can be unit-tested without Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def scale_actions(actions: torch.Tensor, low: Sequence[float], high: Sequence[float]) -> torch.Tensor:
    """Map policy outputs from [-1, 1] to the physical command range [low, high], per dimension.

    Identical to the affine rescaling RL-Games applies to a bounded Box action space:
    ``a * (high - low) / 2 + (high + low) / 2`` after clamping ``a`` to [-1, 1].
    """
    low_t = torch.as_tensor(low, dtype=actions.dtype, device=actions.device)
    high_t = torch.as_tensor(high, dtype=actions.dtype, device=actions.device)
    clamped = actions.clamp(-1.0, 1.0)
    return low_t + 0.5 * (clamped + 1.0) * (high_t - low_t)


def unicycle_to_wheel_velocities(
    linear: torch.Tensor, angular: torch.Tensor, wheel_radius: float, wheel_base: float
) -> torch.Tensor:
    """Convert body velocity commands to wheel angular velocities.

    Args:
        linear: Forward speed v in m/s, shape (N,).
        angular: Yaw rate w in rad/s (positive = counter-clockwise), shape (N,).
        wheel_radius: Wheel radius r in m.
        wheel_base: Distance b between the two wheels in m.

    Returns:
        Wheel angular velocities in rad/s, shape (N, 2), ordered [left, right]:
        ``w_left = (2v - w*b) / 2r`` and ``w_right = (2v + w*b) / 2r``.
    """
    left = (2.0 * linear - angular * wheel_base) / (2.0 * wheel_radius)
    right = (2.0 * linear + angular * wheel_base) / (2.0 * wheel_radius)
    return torch.stack((left, right), dim=-1)


def wheel_to_unicycle_velocities(
    wheel_velocities: torch.Tensor, wheel_radius: float, wheel_base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`unicycle_to_wheel_velocities`; returns (linear, angular)."""
    left, right = wheel_velocities[..., 0], wheel_velocities[..., 1]
    linear = wheel_radius * (left + right) / 2.0
    angular = wheel_radius * (right - left) / wheel_base
    return linear, angular
