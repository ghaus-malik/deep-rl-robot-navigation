import pytest
import torch

from rewards.navigation_reward import TERM_NAMES, RewardCfg, compute_reward_terms, termination_flags

CFG = RewardCfg()


def step(min_range=1.0, distance=2.0, prev_distance=2.0, step_idx=10):
    """Evaluate one step for a single env and return (terms as floats, total, flags)."""
    t = lambda x: torch.tensor([x], dtype=torch.float32)  # noqa: E731
    min_r, d, prev, k = t(min_range), t(distance), t(prev_distance), torch.tensor([step_idx])
    goal, col, to = termination_flags(min_r, d, k, CFG)
    terms = compute_reward_terms(min_r, d, prev, k, goal, col, to, CFG)
    values = {name: v.item() for name, v in terms.items()}
    return values, sum(values.values()), (goal.item(), col.item(), to.item())


def test_all_terms_present():
    terms, _, _ = step()
    assert set(terms) == set(TERM_NAMES)


def test_neutral_step_is_zero():
    _, total, flags = step()
    assert total == pytest.approx(0.0)
    assert flags == (False, False, False)


def test_goal_reached_reward_and_time_bonus():
    terms, _, flags = step(distance=0.29, prev_distance=0.29, step_idx=200)
    assert flags[0]
    assert terms["goal"] == pytest.approx(30.0)
    assert terms["time_bonus"] == pytest.approx(0.02 * (1200 - 200))


def test_goal_radius_boundary_is_strict():
    _, _, flags = step(distance=0.30, prev_distance=0.30)
    assert not flags[0]


def test_collision_penalty_includes_proximity():
    terms, _, flags = step(min_range=0.2)
    assert flags[1]
    assert terms["collision"] == pytest.approx(-30.0)
    assert terms["proximity"] == pytest.approx(-5.0)


def test_timeout_penalty_on_last_step():
    terms, _, flags = step(step_idx=1199)
    assert flags[2]
    assert terms["timeout"] == pytest.approx(-30.0)
    _, _, flags_before = step(step_idx=1198)
    assert not flags_before[2]


def test_moving_closer_is_rewarded():
    terms, total, _ = step(distance=1.99, prev_distance=2.0)
    assert terms["progress"] == pytest.approx(10 * 0.01, rel=1e-4)
    assert terms["potential"] == pytest.approx(0.1 * 0.01 * 60, rel=1e-3)
    assert total > 0


def test_moving_away_is_penalised():
    terms, total, _ = step(distance=2.01, prev_distance=2.0)
    assert terms["progress"] < 0 and terms["potential"] < 0 and total < 0


@pytest.mark.parametrize(
    "min_range, expected",
    [(0.41, 0.0), (0.39, -1.0), (0.36, -1.0), (0.34, -3.0), (0.31, -3.0), (0.29, -5.0), (0.26, -5.0)],
)
def test_proximity_thresholds_are_cumulative(min_range, expected):
    terms, _, flags = step(min_range=min_range)
    assert terms["proximity"] == pytest.approx(expected)
    assert not flags[1]  # still above the 0.25 m collision range


@pytest.mark.parametrize("min_range", [0.41, 0.39, 0.30, 0.26, 0.20])
def test_exponential_proximity_penalty(min_range):
    cfg = RewardCfg(proximity_mode="exponential")
    r = torch.tensor([min_range])
    k = torch.tensor([10])
    flags = termination_flags(r, torch.tensor([2.0]), k, cfg)
    terms = compute_reward_terms(r, torch.tensor([2.0]), torch.tensor([2.0]), k, *flags, cfg)
    expected = -torch.exp(-r).item() if min_range < 0.40 else 0.0
    assert terms["proximity"].item() == pytest.approx(expected)


def test_exponential_proximity_grows_closer_to_obstacles():
    cfg = RewardCfg(proximity_mode="exponential")
    r = torch.tensor([0.39, 0.30, 0.26])
    k = torch.full((3,), 10)
    d = torch.full((3,), 2.0)
    terms = compute_reward_terms(r, d, d, k, *termination_flags(r, d, k, cfg), cfg)
    p = terms["proximity"]
    assert p[0] > p[1] > p[2]
    assert p.min() > -1.0  # softer than the -1/-3/-5 step penalty


def test_unknown_proximity_mode_raises():
    with pytest.raises(ValueError):
        step_cfg = RewardCfg(proximity_mode="bogus")
        t = torch.tensor([1.0])
        compute_reward_terms(t, t, t, torch.tensor([1]), *termination_flags(t, t, torch.tensor([1]), step_cfg), step_cfg)


def test_simultaneous_goal_collision_and_timeout_all_count():
    terms, total, flags = step(min_range=0.2, distance=0.1, prev_distance=0.1, step_idx=1199)
    assert flags == (True, True, True)
    expected = 30.0 + 0.02 * 1 - 30.0 - 30.0 - 5.0
    assert total == pytest.approx(expected)


def test_first_step_after_reset_has_small_progress():
    # With prev_distance initialised to the start distance, the first step only sees the real displacement.
    start = 3.13
    terms, _, _ = step(distance=start - 0.005, prev_distance=start, step_idx=1)
    assert abs(terms["progress"]) < 0.1
