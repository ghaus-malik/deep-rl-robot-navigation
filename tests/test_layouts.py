import numpy as np
import pytest

from envs.layouts import (
    BASELINE_START_XY,
    Box,
    Cylinder,
    array_layout,
    baseline_layout,
    clearance,
    is_reachable,
    layouts_to_arrays,
    make_layout_bank,
    random_layout,
)


def test_baseline_start_and_goal_are_collision_free():
    lay = baseline_layout()
    assert clearance(lay.start_xy, lay) >= 0.45
    assert clearance(lay.goal_xy, lay) >= 0.45


def test_baseline_start_outside_proximity_penalty_zone():
    lay = baseline_layout()
    assert clearance(lay.start_xy, lay) > 0.40


def test_baseline_is_reachable():
    assert is_reachable(baseline_layout())


def test_baseline_obstacles():
    lay = baseline_layout()
    assert len(lay.boxes) == 0 and len(lay.cylinders) == 4
    assert all(c.radius == pytest.approx(0.3) for c in lay.cylinders)


def test_clearance_includes_walls_and_obstacles():
    lay = baseline_layout()
    assert clearance((0.75, 0.75), lay) < 0  # inside a cylinder
    assert clearance((1.95, 0.0), lay) < 0  # inside the wall
    assert clearance((-1.4, 0.0), lay) == pytest.approx(0.5)  # back wall face at x = -1.9


def test_clearance_rotated_box():
    lay = baseline_layout()
    lay.cylinders = []
    lay.boxes = [Box(center=(0.0, 0.0), size=(1.0, 0.2), yaw=np.pi / 2)]  # long side along y
    assert clearance((0.0, 0.45), lay) < 0  # inside after rotation
    assert clearance((0.3, 0.0), lay) == pytest.approx(0.2)


def test_unreachable_layout_detected():
    lay = baseline_layout()
    lay.boxes = [Box(center=(0.0, 0.0), size=(0.2, 4.0))]  # wall splitting the arena
    assert not is_reachable(lay)


def test_random_layouts_are_valid_and_seeded():
    a = random_layout(np.random.default_rng(3))
    b = random_layout(np.random.default_rng(3))
    assert a.goal_xy == b.goal_xy
    assert a.start_xy == BASELINE_START_XY  # start stays fixed by default
    assert 3 <= len(a.cylinders) + len(a.boxes) <= 6
    assert clearance(a.start_xy, a) >= 0.5 and clearance(a.goal_xy, a) >= 0.45
    assert np.hypot(*np.subtract(a.goal_xy, a.start_xy)) >= 2.0
    assert is_reachable(a)


def test_random_layouts_mix_shapes_and_goals():
    rng = np.random.default_rng(0)
    layouts = [random_layout(rng) for _ in range(30)]
    assert sum(len(lay.boxes) for lay in layouts) > 0 and sum(len(lay.cylinders) for lay in layouts) > 0
    assert len({lay.goal_xy for lay in layouts}) == 30
    counts = {len(lay.boxes) + len(lay.cylinders) for lay in layouts}
    assert counts <= {3, 4, 5, 6} and len(counts) > 1


def test_bank_round_trip_and_cache(tmp_path):
    bank = make_layout_bank(5, seed=1, cache_dir=str(tmp_path))
    again = make_layout_bank(5, seed=1, cache_dir=str(tmp_path))  # served from the cache file
    assert len(list(tmp_path.iterdir())) == 1
    for key in bank:
        assert np.array_equal(bank[key], again[key])
    assert bank["circles"].shape == (5, 6, 3) and bank["boxes"].shape == (5, 6, 5)
    lay = array_layout(bank, 2)
    assert clearance(lay.goal_xy, lay) >= 0.45 - 1e-5
    assert is_reachable(lay)


def test_layouts_to_arrays_masks_unused_slots():
    arr = layouts_to_arrays([baseline_layout()], max_obstacles=6)
    assert arr["circle_mask"][0].sum() == 4 and arr["box_mask"][0].sum() == 0
    assert arr["circles"][0, 0, 2] == pytest.approx(0.3)


def test_walls_mesh_offsets_each_env():
    pytest.importorskip("trimesh")
    from envs.obstacle_mesh import build_walls_mesh, layout_to_mesh

    origins = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    lo, hi = build_walls_mesh(origins).bounds
    assert lo[0] == pytest.approx(-2.1) and hi[0] == pytest.approx(7.1)
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(0.5)
    lay = baseline_layout()
    lay.cylinders = [Cylinder(center=(0.0, 0.0), radius=0.2)]
    lo, hi = layout_to_mesh(lay, walls=False).bounds
    assert lo[0] == pytest.approx(-0.2, abs=1e-3) and hi[0] == pytest.approx(0.2, abs=1e-3)
