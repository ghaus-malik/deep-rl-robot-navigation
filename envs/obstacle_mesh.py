"""Triangle meshes for the static parts of the scene.

Only the arena walls are baked into a mesh (they never change). Per-episode obstacles are drawn with visual
markers and seen by the analytic LiDAR instead (see ``envs/nav_env.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import trimesh

from .layouts import Box, Layout, arena_walls


def _box_mesh(box: Box, origin_xy: Sequence[float]) -> trimesh.Trimesh:
    transform = trimesh.transformations.rotation_matrix(box.yaw, (0.0, 0.0, 1.0))
    transform[:3, 3] = (box.center[0] + origin_xy[0], box.center[1] + origin_xy[1], box.height / 2.0)
    return trimesh.creation.box(extents=(box.size[0], box.size[1], box.height), transform=transform)


def layout_to_mesh(
    layout: Layout, origin_xy: Sequence[float] = (0.0, 0.0), cylinder_sections: int = 32, walls: bool = True
) -> trimesh.Trimesh:
    """Triangle mesh of one layout (optionally with the walls); obstacles stand on the ground."""
    parts = [_box_mesh(box, origin_xy) for box in (arena_walls() if walls else []) + list(layout.boxes)]
    for cyl in layout.cylinders:
        transform = trimesh.transformations.translation_matrix(
            (cyl.center[0] + origin_xy[0], cyl.center[1] + origin_xy[1], cyl.height / 2.0)
        )
        parts.append(
            trimesh.creation.cylinder(
                radius=cyl.radius, height=cyl.height, sections=cylinder_sections, transform=transform
            )
        )
    return trimesh.util.concatenate(parts)


def build_walls_mesh(env_origins: np.ndarray) -> trimesh.Trimesh:
    """One mesh with the four arena walls of every environment."""
    return trimesh.util.concatenate(
        [_box_mesh(wall, origin[:2]) for origin in env_origins for wall in arena_walls()]
    )
