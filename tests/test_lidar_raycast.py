import math

import numpy as np
import pytest
import torch

from envs.layouts import ARENA_INNER_HALF, array_layout, clearance_many, layouts_to_arrays, make_layout_bank
from envs.layouts import baseline_layout
from envs.lidar_processing import planar_ray_angles
from envs.lidar_raycast import cast_planar_lidar

ANGLES = planar_ray_angles(360)


def _cast(origin, yaw, layout_arrays):
    a = {k: torch.as_tensor(v) for k, v in layout_arrays.items()}
    return cast_planar_lidar(
        torch.tensor([origin], dtype=torch.float32), torch.tensor([yaw], dtype=torch.float32), ANGLES,
        ARENA_INNER_HALF, a["circles"], a["circle_mask"], a["boxes"], a["box_mask"],
    )[0]


def _empty(max_obstacles=2):
    lay = baseline_layout()
    lay.cylinders = []
    return layouts_to_arrays([lay], max_obstacles)


def test_walls_only_from_centre():
    d = _cast((0.0, 0.0), 0.0, _empty())
    assert d[180].item() == pytest.approx(1.9, abs=1e-5)  # front
    assert d[0].item() == pytest.approx(1.9, abs=1e-5)  # back
    assert d[225].item() == pytest.approx(1.9 * math.sqrt(2), abs=1e-4)  # front-left corner


def test_baseline_start_matches_scene_geometry():
    d = _cast((-1.4, 0.0), 0.0, layouts_to_arrays([baseline_layout()], 4))
    assert d[0].item() == pytest.approx(0.5, abs=1e-5)  # back wall
    assert d[90].item() == pytest.approx(1.9, abs=1e-5)  # right wall
    assert d[270].item() == pytest.approx(1.9, abs=1e-5)  # left wall
    # cylinder at (-0.75, 0.75), r = 0.3: closest surface 0.6925 m away at ~49 deg
    assert d[226:232].min().item() == pytest.approx(math.hypot(0.65, 0.75) - 0.3, abs=2e-3)


def test_yaw_rotates_the_scan():
    arrays = layouts_to_arrays([baseline_layout()], 4)
    d0 = _cast((-1.4, 0.0), 0.0, arrays)
    d90 = _cast((-1.4, 0.0), math.pi / 2, arrays)
    # facing +y, the back ray (index 0) now looks along -y to the bottom wall
    assert d90[0].item() == pytest.approx(1.9, abs=1e-5)
    assert d90[90].item() == pytest.approx(d0[180].item(), abs=1e-5)  # right side now looks along +x


def test_rotated_box_hit_distance():
    arrays = _empty()
    arrays["boxes"][0, 0] = (1.0, 0.0, 0.5, 0.1, math.pi / 2)  # 1.0 x 0.2 box turned to lie along y
    arrays["box_mask"][0, 0] = True
    d = _cast((0.0, 0.0), 0.0, arrays)
    assert d[180].item() == pytest.approx(0.9, abs=1e-5)  # front face at x = 0.9 after rotation
    arrays["box_mask"][0, 0] = False  # masked slots are ignored
    assert _cast((0.0, 0.0), 0.0, arrays)[180].item() == pytest.approx(1.9, abs=1e-5)


def test_origin_inside_obstacle_reads_zero():
    arrays = _empty()
    arrays["circles"][0, 0] = (0.0, 0.0, 0.3)
    arrays["circle_mask"][0, 0] = True
    assert _cast((0.1, 0.0), 0.0, arrays).max().item() == pytest.approx(0.0)


def test_min_range_equals_clearance_on_random_layouts():
    bank = make_layout_bank(20, seed=7)
    rng = np.random.default_rng(0)
    for i in range(20):
        lay = array_layout(bank, i)
        point = rng.uniform(-1.5, 1.5, size=2)
        true_clearance = clearance_many(point[None], lay)[0]
        if true_clearance <= 0.05:
            continue
        single = {k: v[i:i + 1] for k, v in bank.items()}
        d = _cast(tuple(point), 0.3, single)
        # 1 deg rays can only miss the true closest point by a little
        assert true_clearance - 1e-4 <= d.min().item() <= true_clearance + 0.02
