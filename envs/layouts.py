"""Scene layouts (pure NumPy).

All coordinates are in metres, in the local frame of one environment (its origin at the arena centre).
Every layout lives in the same walled 4 m x 4 m arena; the walls are implied and not stored in the layout.

* :func:`baseline_layout` is the original fixed scene: four cylinders, fixed start and goal.
* :func:`random_layout` samples a new scene: 3-6 cylinders / rotated boxes of random size, a random goal and
  (by default) the fixed start. Every sampled layout is checked to be reachable.
* :func:`make_layout_bank` pre-samples many random layouts and packs them into arrays so the environment can
  draw a new layout for every episode on the GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

ARENA_HALF_SIZE = 2.0  # wall centre-lines at x, y = +-2 m
WALL_THICKNESS = 0.2
ARENA_INNER_HALF = ARENA_HALF_SIZE - WALL_THICKNESS / 2.0  # inner wall faces at +-1.9 m
OBSTACLE_HEIGHT = 0.5
BASELINE_START_XY = (-1.4, 0.0)
BASELINE_START_YAW = 0.0


@dataclass
class Box:
    """Box obstacle, rotated by ``yaw`` (rad) about its centre."""

    center: tuple[float, float]
    size: tuple[float, float]  # full extent along the box's own x and y axes
    yaw: float = 0.0
    height: float = OBSTACLE_HEIGHT


@dataclass
class Cylinder:
    center: tuple[float, float]
    radius: float
    height: float = OBSTACLE_HEIGHT


@dataclass
class Layout:
    name: str
    boxes: list[Box] = field(default_factory=list)  # obstacles only (walls are implied)
    cylinders: list[Cylinder] = field(default_factory=list)
    start_xy: tuple[float, float] = BASELINE_START_XY
    start_yaw: float = BASELINE_START_YAW
    goal_xy: tuple[float, float] = (0.0, 0.0)


def arena_walls(half_size: float = ARENA_HALF_SIZE, thickness: float = WALL_THICKNESS) -> list[Box]:
    """Four walls whose centre-lines sit at +-half_size; each wall is 2*half_size + thickness long."""
    length = 2.0 * half_size + thickness
    return [
        Box(center=(half_size, 0.0), size=(thickness, length)),
        Box(center=(-half_size, 0.0), size=(thickness, length)),
        Box(center=(0.0, half_size), size=(length, thickness)),
        Box(center=(0.0, -half_size), size=(length, thickness)),
    ]


def baseline_layout() -> Layout:
    """Original fixed scene.

    Four 0.3 m-radius cylinders at (+-0.75, +-0.75). Start (-1.4, 0.0) facing +x and goal (1.4, -1.4) both keep
    0.5 m from the walls and > 0.6 m from the cylinders, so the robot never starts inside the proximity-penalty
    zone (< 0.40 m).
    """
    cylinders = [Cylinder(center=(sx * 0.75, sy * 0.75), radius=0.3) for sx in (1, -1) for sy in (1, -1)]
    return Layout(name="baseline", cylinders=cylinders, goal_xy=(1.4, -1.4))


# --------------------------------------------------------------------------------------------- geometry
def clearance_many(points: np.ndarray, layout: Layout) -> np.ndarray:
    """Distance from each point (M, 2) to the nearest surface, walls included (negative inside an obstacle)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    best = ARENA_INNER_HALF - np.abs(p).max(axis=1)  # inner wall faces
    for box in layout.boxes:
        c, s = np.cos(box.yaw), np.sin(box.yaw)
        d = p - np.asarray(box.center)
        local = np.stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]), axis=1)  # into the box frame
        offset = np.abs(local) - np.asarray(box.size) / 2.0
        outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
        inside = np.minimum(offset.max(axis=1), 0.0)
        best = np.minimum(best, outside + inside)
    for cyl in layout.cylinders:
        best = np.minimum(best, np.linalg.norm(p - np.asarray(cyl.center), axis=1) - cyl.radius)
    return best


def clearance(point, layout: Layout) -> float:
    """Distance from one point to the nearest surface, walls included (negative inside an obstacle)."""
    return float(clearance_many(np.asarray(point)[None], layout)[0])


def is_reachable(layout: Layout, robot_radius: float = 0.30, resolution: float = 0.05) -> bool:
    """Flood fill from start to goal over grid cells with clearance > ``robot_radius``."""
    n = int(round(2 * ARENA_HALF_SIZE / resolution)) + 1
    coords = np.linspace(-ARENA_HALF_SIZE, ARENA_HALF_SIZE, n)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    free = (clearance_many(np.stack((xx.ravel(), yy.ravel()), axis=1), layout) > robot_radius).reshape(n, n)

    def to_cell(xy):
        return tuple(int(round((v + ARENA_HALF_SIZE) / resolution)) for v in xy)

    start, goal = to_cell(layout.start_xy), to_cell(layout.goal_xy)
    if not (free[start] and free[goal]):
        return False
    reached = np.zeros_like(free)
    reached[start] = True
    while True:  # grow the reached region by one cell (4-neighbourhood) until it stops changing
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if grown[goal]:
            return True
        if (grown == reached).all():
            return False
        reached = grown


# --------------------------------------------------------------------------------------------- sampling
def _bounding_radius(obstacle: Box | Cylinder) -> float:
    if isinstance(obstacle, Cylinder):
        return obstacle.radius
    return float(np.hypot(*obstacle.size) / 2.0)


def random_layout(
    rng: np.random.Generator,
    num_obstacles_range: tuple[int, int] = (3, 6),
    cylinder_probability: float = 0.5,
    cylinder_radius_range: tuple[float, float] = (0.15, 0.35),
    box_side_range: tuple[float, float] = (0.2, 0.8),
    obstacle_gap: float = 0.1,
    randomize_start: bool = False,
    randomize_goal: bool = True,
    randomize_obstacles: bool = True,
    min_start_goal_distance: float = 2.0,
    start_clearance: float = 0.5,
    goal_clearance: float = 0.45,
    robot_radius: float = 0.30,
    max_tries: int = 1000,
) -> Layout:
    """Sample a reachable layout. Anything not randomised is taken from :func:`baseline_layout`.

    Args:
        num_obstacles_range: Inclusive range for the number of obstacles.
        cylinder_probability: Chance that an obstacle is a cylinder (otherwise a rotated box).
        obstacle_gap: Minimum gap between the bounding circles of two obstacles (no overlapping shapes).
        start_clearance / goal_clearance: Minimum distance from start / goal to any surface (walls included).
        robot_radius: Free-space margin used by the reachability check (> 0.25 m collision distance).
    """
    base = baseline_layout()
    # drawn once, so rejected attempts do not bias the layouts towards fewer obstacles
    num_obstacles = int(rng.integers(num_obstacles_range[0], num_obstacles_range[1] + 1))
    for _ in range(max_tries):
        if randomize_start:
            lim = ARENA_INNER_HALF - start_clearance
            start = tuple(float(v) for v in rng.uniform(-lim, lim, size=2))
            start_yaw = float(rng.uniform(-np.pi, np.pi))
        else:
            start, start_yaw = base.start_xy, base.start_yaw

        boxes, cylinders = base.boxes, base.cylinders
        if randomize_obstacles:
            boxes, cylinders, placed = [], [], []
            for _ in range(num_obstacles):
                for _ in range(50):  # place this obstacle without overlapping the others
                    if rng.uniform() < cylinder_probability:
                        obs = Cylinder(center=(0.0, 0.0), radius=float(rng.uniform(*cylinder_radius_range)))
                    else:
                        size = tuple(float(v) for v in rng.uniform(*box_side_range, size=2))
                        obs = Box(center=(0.0, 0.0), size=size, yaw=float(rng.uniform(0.0, np.pi)))
                    r = _bounding_radius(obs)
                    lim = ARENA_INNER_HALF - r
                    obs.center = tuple(float(v) for v in rng.uniform(-lim, lim, size=2))
                    if all(np.hypot(*np.subtract(obs.center, q.center)) > r + _bounding_radius(q) + obstacle_gap
                           for q in placed):
                        placed.append(obs)
                        (cylinders if isinstance(obs, Cylinder) else boxes).append(obs)
                        break
            if len(placed) < num_obstacles:
                continue

        if randomize_goal:
            lim = ARENA_INNER_HALF - goal_clearance
            goal = tuple(float(v) for v in rng.uniform(-lim, lim, size=2))
        else:
            goal = base.goal_xy

        layout = Layout("random", boxes, cylinders, start, start_yaw, goal)
        if np.hypot(*np.subtract(goal, start)) < min_start_goal_distance:
            continue
        if clearance(start, layout) < start_clearance or clearance(goal, layout) < goal_clearance:
            continue
        if is_reachable(layout, robot_radius=robot_radius):
            return layout
    raise RuntimeError("Could not sample a valid layout; relax the constraints.")


# ------------------------------------------------------------------------------------------------- bank
def layouts_to_arrays(layouts: list[Layout], max_obstacles: int) -> dict[str, np.ndarray]:
    """Pack layouts into fixed-size arrays (unused slots are masked out).

    Returns (L = number of layouts, K = max_obstacles):
        circles (L, K, 3): cx, cy, radius             circle_mask (L, K)
        boxes   (L, K, 5): cx, cy, half_x, half_y, yaw box_mask    (L, K)
        start (L, 2), start_yaw (L,), goal (L, 2)
    """
    n = len(layouts)
    arrays = {
        "circles": np.zeros((n, max_obstacles, 3), np.float32),
        "circle_mask": np.zeros((n, max_obstacles), bool),
        "boxes": np.zeros((n, max_obstacles, 5), np.float32),
        "box_mask": np.zeros((n, max_obstacles), bool),
        "start": np.zeros((n, 2), np.float32),
        "start_yaw": np.zeros(n, np.float32),
        "goal": np.zeros((n, 2), np.float32),
    }
    for i, lay in enumerate(layouts):
        if len(lay.cylinders) > max_obstacles or len(lay.boxes) > max_obstacles:
            raise ValueError(f"Layout {i} has more than {max_obstacles} cylinders or boxes.")
        for k, c in enumerate(lay.cylinders):
            arrays["circles"][i, k] = (*c.center, c.radius)
            arrays["circle_mask"][i, k] = True
        for k, b in enumerate(lay.boxes):
            arrays["boxes"][i, k] = (*b.center, b.size[0] / 2.0, b.size[1] / 2.0, b.yaw)
            arrays["box_mask"][i, k] = True
        arrays["start"][i] = lay.start_xy
        arrays["start_yaw"][i] = lay.start_yaw
        arrays["goal"][i] = lay.goal_xy
    return arrays


def make_layout_bank(
    size: int, seed: int, cache_dir: str | None = None, **random_layout_kwargs
) -> dict[str, np.ndarray]:
    """``size`` random layouts from ``seed`` packed with :func:`layouts_to_arrays` (cached on disk if asked)."""
    max_obstacles = random_layout_kwargs.get("num_obstacles_range", (3, 6))[1]
    path = None
    if cache_dir is not None:
        key = json.dumps({"size": size, "seed": seed, **random_layout_kwargs}, sort_keys=True, default=str)
        path = os.path.join(cache_dir, f"layout_bank_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz")
        if os.path.exists(path):
            with np.load(path) as data:
                return {k: data[k] for k in data.files}
    rng = np.random.default_rng(seed)
    bank = layouts_to_arrays([random_layout(rng, **random_layout_kwargs) for _ in range(size)], max_obstacles)
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, **bank)
    return bank


def array_layout(bank: dict[str, np.ndarray], index: int) -> Layout:
    """Rebuild one :class:`Layout` from a bank (for plots, checks and tests)."""
    cylinders = [
        Cylinder(center=(float(c[0]), float(c[1])), radius=float(c[2]))
        for c, m in zip(bank["circles"][index], bank["circle_mask"][index]) if m
    ]
    boxes = [
        Box(center=(float(b[0]), float(b[1])), size=(2 * float(b[2]), 2 * float(b[3])), yaw=float(b[4]))
        for b, m in zip(bank["boxes"][index], bank["box_mask"][index]) if m
    ]
    return Layout(
        f"bank_{index}", boxes, cylinders,
        tuple(float(v) for v in bank["start"][index]), float(bank["start_yaw"][index]),
        tuple(float(v) for v in bank["goal"][index]),
    )
