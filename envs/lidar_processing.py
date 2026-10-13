"""Planar 2D LiDAR post-processing (pure PyTorch).

Ray ordering convention used throughout the project:
ray ``i`` points at body-frame azimuth ``-180 deg + i * resolution``, i.e. index 0 looks straight
backwards and indices increase counter-clockwise (left side of the robot comes before the front).
"""

from __future__ import annotations

import torch


def planar_ray_angles(num_rays: int, start_angle_deg: float = -180.0, resolution_deg: float = 1.0) -> torch.Tensor:
    """Body-frame azimuth of every ray in radians, shape (num_rays,)."""
    return torch.deg2rad(start_angle_deg + resolution_deg * torch.arange(num_rays, dtype=torch.float32))


def clean_ranges(distances: torch.Tensor, min_range: float, max_range: float) -> torch.Tensor:
    """Replace invalid readings and clamp to the sensor's valid interval.

    Misses (inf) and NaN become ``max_range``; everything is then clamped to [min_range, max_range].
    """
    cleaned = torch.where(torch.isfinite(distances), distances, torch.full_like(distances, max_range))
    return cleaned.clamp(min_range, max_range)


def downsample_min(ranges: torch.Tensor, group_size: int) -> torch.Tensor:
    """Keep the closest reading in every block of ``group_size`` consecutive rays.

    (N, R) -> (N, R // group_size). Taking the minimum (not the mean) keeps thin obstacles visible.
    """
    num_envs, num_rays = ranges.shape
    if num_rays % group_size != 0:
        raise ValueError(f"Number of rays ({num_rays}) must be divisible by group size ({group_size}).")
    return ranges.view(num_envs, num_rays // group_size, group_size).min(dim=-1).values
