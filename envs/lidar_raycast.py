"""Exact planar ray casting against the arena walls, circles and rotated boxes (pure PyTorch, batched).

Obstacles are vertical prisms that stand on the ground and are taller than the scan plane, so a horizontal
LiDAR sees their 2D cross-sections: the arena's inner wall square, circles (cylinders) and rotated rectangles
(boxes). Because obstacles change every episode, they are passed in as tensors instead of a static mesh.

All positions are in each environment's local frame (arena centre = origin).
"""

from __future__ import annotations

import torch

_EPS = 1e-9


def _safe(d: torch.Tensor) -> torch.Tensor:
    """Replace (near-)zero direction components so divisions give +-huge instead of nan."""
    return torch.where(d.abs() < _EPS, torch.full_like(d, _EPS), d)


def ray_square_interior(origin: torch.Tensor, dirs: torch.Tensor, half: float) -> torch.Tensor:
    """Distance from points inside the square |x|, |y| <= half to its boundary along each ray.

    origin (N, 2), dirs (N, R, 2) -> (N, R)
    """
    o = origin[:, None, :]
    d = _safe(dirs)
    t = torch.where(d > 0, (half - o) / d, (-half - o) / d)  # exit distance per axis
    return t.min(dim=-1).values.clamp(min=0.0)


def ray_circles(origin: torch.Tensor, dirs: torch.Tensor, circles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Nearest hit with any active circle (inf if none). 0 if the origin is inside a circle.

    origin (N, 2), dirs (N, R, 2), circles (N, K, 3) = (cx, cy, r), mask (N, K) -> (N, R)
    """
    oc = circles[:, None, :, :2] - origin[:, None, None, :]  # (N, 1, K, 2) origin -> centre
    proj = (oc * dirs[:, :, None, :]).sum(-1)  # (N, R, K) distance along the ray to the closest approach
    perp2 = (oc * oc).sum(-1) - proj**2
    disc = circles[:, None, :, 2] ** 2 - perp2
    root = disc.clamp(min=0.0).sqrt()
    t_in, t_out = proj - root, proj + root
    hit = (disc >= 0) & (t_out >= 0) & mask[:, None, :]
    t = torch.where(hit, t_in.clamp(min=0.0), torch.full_like(t_in, float("inf")))
    return t.min(dim=-1).values


def ray_boxes(origin: torch.Tensor, dirs: torch.Tensor, boxes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Nearest hit with any active rotated box (inf if none), slab method in each box's own frame.

    origin (N, 2), dirs (N, R, 2), boxes (N, K, 5) = (cx, cy, half_x, half_y, yaw), mask (N, K) -> (N, R)
    """
    c, s = torch.cos(boxes[..., 4]), torch.sin(boxes[..., 4])  # (N, K)
    rel = origin[:, None, :] - boxes[..., :2]  # (N, K, 2) box centre -> origin
    # rotate by -yaw into the box frame
    o_x = c * rel[..., 0] + s * rel[..., 1]
    o_y = -s * rel[..., 0] + c * rel[..., 1]
    d_x = c[:, None] * dirs[..., 0:1] + s[:, None] * dirs[..., 1:2]  # (N, R, K)
    d_y = -s[:, None] * dirs[..., 0:1] + c[:, None] * dirs[..., 1:2]
    d_x, d_y = _safe(d_x), _safe(d_y)
    hx, hy = boxes[:, None, :, 2], boxes[:, None, :, 3]
    tx1, tx2 = (-hx - o_x[:, None]) / d_x, (hx - o_x[:, None]) / d_x
    ty1, ty2 = (-hy - o_y[:, None]) / d_y, (hy - o_y[:, None]) / d_y
    t_near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
    t_far = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
    hit = (t_near <= t_far) & (t_far >= 0) & mask[:, None, :]
    t = torch.where(hit, t_near.clamp(min=0.0), torch.full_like(t_near, float("inf")))
    return t.min(dim=-1).values


def cast_planar_lidar(
    origin: torch.Tensor,
    yaw: torch.Tensor,
    ray_angles: torch.Tensor,
    arena_inner_half: float,
    circles: torch.Tensor,
    circle_mask: torch.Tensor,
    boxes: torch.Tensor,
    box_mask: torch.Tensor,
) -> torch.Tensor:
    """Distance along every ray to the nearest surface.

    Args:
        origin: Sensor position in the env frame, (N, 2).
        yaw: Sensor heading, (N,). Ray ``i`` points at ``yaw + ray_angles[i]``.
        ray_angles: Body-frame azimuths (R,), see :func:`envs.lidar_processing.planar_ray_angles`.

    Returns:
        (N, R) distances in metres (not clipped to the sensor range).
    """
    angles = yaw[:, None] + ray_angles[None, :]
    dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    dist = ray_square_interior(origin, dirs, arena_inner_half)
    dist = torch.minimum(dist, ray_circles(origin, dirs, circles, circle_mask))
    return torch.minimum(dist, ray_boxes(origin, dirs, boxes, box_mask))
