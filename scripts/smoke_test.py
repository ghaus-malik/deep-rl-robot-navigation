"""Scripted end-to-end check of the navigation environment inside Isaac Sim (no learning).

Four environments are driven by hand-written controllers so that every code path is exercised:

    env 0  straight ahead                -> speed check at 0.3 m/s, then collides with the far wall
    env 1  waypoint follower             -> counter-clockwise check, then reaches the goal
    env 2  straight at 0.5 m/s command   -> maximum-speed check
    env 3  slow clockwise circle         -> clockwise check, then times out after 1200 steps

Checks: observation shape/values at the start pose, LiDAR geometry, measured vs commanded speed,
rotation direction, the three termination types with their rewards, and the per-episode noise.

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/smoke_test.py --headless
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/smoke_test.py --headless --layout random
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--output", default=None, help="Default: results/smoke_test[_random].json")
parser.add_argument("--layout", choices=("baseline", "random"), default="baseline",
                    help="baseline: full scripted checks; random: checks of the per-episode random layouts.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from envs.layouts import array_layout, baseline_layout, clearance_many  # noqa: E402
from envs.nav_env import OUTCOME_COLLISION, OUTCOME_GOAL, OUTCOME_NONE, OUTCOME_TIMEOUT, NavEnv  # noqa: E402
from envs.nav_env_cfg import NavEnvCfg  # noqa: E402
from envs.observations import DISTANCE_SLICE, HEADING_SLICE, OBS_DIM, build_observation  # noqa: E402

results: list[dict] = []


def check(name: str, passed: bool, detail: str = ""):
    results.append({"check": name, "passed": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def to_policy_action(v: torch.Tensor, w: torch.Tensor, cfg: NavEnvCfg) -> torch.Tensor:
    """Inverse of the env's action scaling: physical (v, w) -> policy output in [-1, 1]."""
    low, high = torch.tensor(cfg.action_low, device=v.device), torch.tensor(cfg.action_high, device=v.device)
    cmd = torch.stack((v, w), dim=-1)
    return (2.0 * (cmd - low) / (high - low) - 1.0).clamp(-1.0, 1.0)


def random_layout_checks():
    """Every env gets its own layout; resets draw new ones; LiDAR and goal observations follow the layout."""
    cfg = NavEnvCfg()
    cfg.scene.num_envs = 16
    cfg.enable_noise = False
    cfg.layout = "random"
    cfg.layout_bank_size = 2000
    cfg.seed = 0
    env = NavEnv(cfg)
    dev = env.device
    origins = env.scene.env_origins[:, :2]

    obs = env.reset()[0]["policy"]
    s = env.state
    first_idx = env.layout_idx.clone()
    check("envs use different layouts", len(set(first_idx.tolist())) > 12, f"{len(set(first_idx.tolist()))}/16 distinct")
    goals_local = env._goal_xy_w - origins
    check("goals differ between envs", torch.unique(goals_local.round(decimals=3), dim=0).shape[0] > 12)
    exp_dist = torch.linalg.norm(env.bank["goal"][first_idx] - env.bank["start"][first_idx], dim=-1)
    check("goal distance observation", torch.allclose(obs[:, DISTANCE_SLICE.start], exp_dist, atol=0.03),
          f"max error {(obs[:, DISTANCE_SLICE.start] - exp_dist).abs().max().item():.4f} m")
    robot_local = (s["robot_xy"] - origins).cpu().numpy()
    errors = []
    for i in range(16):
        true_clear = clearance_many(robot_local[i:i + 1], array_layout({k: v.cpu().numpy() for k, v in env.bank.items()},
                                                                      int(first_idx[i])))[0]
        errors.append(s["ranges"][i].min().item() - min(true_clear, cfg.lidar_max_range))
    check("LiDAR minimum = distance to nearest surface", all(-1e-3 < e < 0.02 for e in errors),
          f"errors {min(errors):+.4f}..{max(errors):+.4f} m")
    check("start outside proximity zone", bool((s["min_range"] > 0.40).all()), f"min={s['min_range'].min().item():.3f}")

    # Force a reset of envs 0-7 by driving into obstacles/walls; their layouts must change, the others must not.
    changed = torch.zeros(16, dtype=torch.bool, device=dev)
    for _ in range(400):
        _, _, terminated, truncated, _ = env.step(torch.tensor([[1.0, 0.0]], device=dev).repeat(16, 1))
        changed |= terminated | truncated
        if changed[:8].all():
            break
    new_idx = env.layout_idx
    check("reset draws a new layout", bool((new_idx[changed] != first_idx[changed]).float().mean() > 0.9),
          f"{int(changed.sum())} envs reset")
    check("envs that did not reset keep their layout", bool((new_idx[~changed] == first_idx[~changed]).all()))
    write_results()
    env.close()


def main():
    if args.layout == "random":
        random_layout_checks()
        return
    cfg = NavEnvCfg()
    cfg.scene.num_envs = 4
    cfg.enable_noise = False  # deterministic checks first
    cfg.seed = 0
    env = NavEnv(cfg)
    dev = env.device
    lay = baseline_layout()

    obs = env.reset()[0]["policy"]
    s = env.state

    # ---------------------------------------------------------------- start pose
    check("observation shape", obs.shape == (4, OBS_DIM), str(tuple(obs.shape)))
    check("observation finite", bool(torch.isfinite(obs).all()))
    exp_dist = math.dist(lay.start_xy, lay.goal_xy)
    exp_head = math.atan2(lay.goal_xy[1] - lay.start_xy[1], lay.goal_xy[0] - lay.start_xy[0]) - lay.start_yaw
    check("goal distance at start", torch.allclose(obs[:, DISTANCE_SLICE], torch.tensor(exp_dist, device=dev), atol=0.03),
          f"obs={obs[0, DISTANCE_SLICE.start].item():.3f} expected={exp_dist:.3f}")
    check("goal heading at start", torch.allclose(obs[:, HEADING_SLICE], torch.tensor(exp_head, device=dev), atol=0.03),
          f"obs={obs[0, HEADING_SLICE.start].item():.3f} expected={exp_head:.3f}")

    # LiDAR geometry from the start (-1.4, 0) facing +x: back wall face at x=-1.9 -> 0.5 m,
    # side walls 1.9 m, front free up to 2 m (clipped), cylinder (-0.75, 0.75) surface ~0.69 m at ~49 deg.
    ranges = s["ranges"][0]
    print(f"[INFO] robot root (local) = {(s['robot_xy'][0] - env.scene.env_origins[0, :2]).tolist()}")
    for label, idx, expected in (("back", 0, 0.5), ("front", 180, 2.0), ("left", 270, 1.9), ("right", 90, 1.9)):
        check(f"lidar {label} ray", abs(ranges[idx].item() - expected) < 0.06,
              f"ray {idx} = {ranges[idx].item():.3f} m, expected ~{expected}")
    bearing = math.degrees(math.atan2(0.75 - lay.start_xy[1], -0.75 - lay.start_xy[0]))
    idx = int(round(bearing + 180))
    exp_cyl = math.dist(lay.start_xy, (-0.75, 0.75)) - 0.3
    check("lidar sees cylinder", abs(ranges[idx - 2: idx + 3].min().item() - exp_cyl) < 0.06,
          f"rays {idx - 2}..{idx + 2} min = {ranges[idx - 2: idx + 3].min().item():.3f} m, expected ~{exp_cyl:.3f}")
    check("120 sectors", s["sectors"].shape == (4, 120), str(tuple(s["sectors"].shape)))
    check("start outside proximity zone", bool((s["min_range"] > 0.40).all()), f"min={s['min_range'].min().item():.3f}")

    # ---------------------------------------------------------- kinematics, 1 s
    start_xy = s["robot_xy"].clone()
    start_yaw = s["yaw"].clone()
    # env 0: 0.3 m/s -> wheel-radius check
    # env 2: 0.5 m/s -> the maximum command must be reachable with the raised wheel velocity limit
    v = torch.tensor([0.3, 0.1, 0.5, 0.1], device=dev)
    w = torch.tensor([0.0, 0.5, 0.0, -0.5], device=dev)
    steps = 80  # 80 policy steps x 0.025 s = 2 s; speed is measured over the second half (steady state)
    for k in range(steps):
        env.step(to_policy_action(v, w, cfg))
        if k == steps // 2 - 1:
            mid_xy, mid_yaw = env.state["robot_xy"].clone(), env.state["yaw"].clone()
    s = env.state
    speed = torch.linalg.norm(s["robot_xy"] - mid_xy, dim=-1) / 1.0
    yaw_rate = torch.atan2(torch.sin(s["yaw"] - mid_yaw), torch.cos(s["yaw"] - mid_yaw)) / 1.0
    speed_ratio = speed[0].item() / 0.3
    check("forward speed (0.3 m/s)", 0.9 < speed_ratio < 1.1, f"measured {speed[0].item():.3f} m/s (ratio {speed_ratio:.3f})")
    check("forward speed (0.5 m/s, max command)", 0.9 < speed[2].item() / 0.5 < 1.1,
          f"measured {speed[2].item():.3f} m/s")
    check("counter-clockwise turn", yaw_rate[1].item() > 0.35, f"yaw rate {yaw_rate[1].item():+.3f} rad/s for w=+0.5")
    check("clockwise turn", yaw_rate[3].item() < -0.35, f"yaw rate {yaw_rate[3].item():+.3f} rad/s for w=-0.5")
    measured_radius = cfg.wheel_radius * speed_ratio
    print(f"[INFO] implied wheel radius = {measured_radius:.4f} m (config {cfg.wheel_radius})")

    # ----------------------------------------------- run until each env terminates
    waypoints = torch.tensor([[0.0, 0.0], [1.4, 0.0], [1.4, -1.4]], device=dev) + env.scene.env_origins[1, :2]
    wp_idx = 0
    first = {i: None for i in range(4)}
    for _ in range(1300):
        s = env.state
        # env 1: drive towards the current waypoint
        delta = waypoints[wp_idx] - s["robot_xy"][1]
        if torch.linalg.norm(delta) < 0.15 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            delta = waypoints[wp_idx] - s["robot_xy"][1]
        bearing_err = math.atan2(delta[1].item(), delta[0].item()) - s["yaw"][1].item()
        err = math.atan2(math.sin(bearing_err), math.cos(bearing_err))
        v = torch.tensor([0.5, 0.3 if abs(err) < 0.4 else 0.1, 0.5, 0.1], device=dev)
        w = torch.tensor([0.0, max(-0.5, min(0.5, 1.5 * err)), 0.0, -0.5], device=dev)

        _, rew, terminated, truncated, extras = env.step(to_policy_action(v, w, cfg))
        done = terminated | truncated
        outcome = env.last_outcome
        for i in range(4):
            if first[i] is None and done[i]:
                first[i] = {"outcome": int(outcome[i]), "reward": rew[i].item(), "steps": None}
        if all(first.values()):
            break
    names = {OUTCOME_NONE: "none", OUTCOME_GOAL: "goal", OUTCOME_COLLISION: "collision", OUTCOME_TIMEOUT: "timeout"}
    for i, f in first.items():
        print(f"[INFO] env {i}: {names[f['outcome']] if f else 'not finished'} reward={f['reward'] if f else None}")
    check("env 0 collides", first[0] is not None and first[0]["outcome"] == OUTCOME_COLLISION,
          f"terminal reward {first[0]['reward']:.2f}" if first[0] else "no termination")
    check("collision reward", first[0] is not None and first[0]["reward"] < -30.0,
          "expected <= -35 (-30 collision, -5 proximity, small progress)")
    check("env 1 reaches goal", first[1] is not None and first[1]["outcome"] == OUTCOME_GOAL,
          f"terminal reward {first[1]['reward']:.2f}" if first[1] else "no termination")
    check("goal reward", first[1] is not None and first[1]["reward"] > 30.0, "expected > +30 (+30 goal + time bonus)")
    check("env 3 times out", first[3] is not None and first[3]["outcome"] == OUTCOME_TIMEOUT,
          f"terminal reward {first[3]['reward']:.2f}" if first[3] else "no termination")
    check("timeout reward", first[3] is not None and first[3]["reward"] <= -29.0, "expected ~ -30")

    # ------------------------------------------------- per-episode noise behaviour
    env.cfg.enable_noise = True
    env.reset()
    offsets = []
    for _ in range(3):
        noisy = env.step(torch.zeros(4, 2, device=dev))[0]["policy"]
        s = env.state
        clean = build_observation(
            s["sectors"], s["heading"], s["distance"],
            env.robot.data.root_lin_vel_w[:, :2], env.robot.data.root_ang_vel_w[:, 2],
        )
        offsets.append((noisy - clean).clone())
    check("observation noise held for the episode",
          all(torch.allclose(offsets[0], o, atol=1e-5) for o in offsets[1:]),
          f"max change between steps {max((o - offsets[0]).abs().max().item() for o in offsets):.2e}")
    std = env._obs_bias.std().item()
    check("observation noise scale", 0.1 < std < 0.3, f"std of offsets = {std:.3f} (config 0.2)")
    heading_copies = env._obs_bias[:, HEADING_SLICE]
    check("heading copies get independent noise", bool((heading_copies.std(dim=-1) > 0).all()))

    write_results({"implied_wheel_radius": measured_radius, "top_speed_at_0.5_cmd": speed[2].item()})
    env.close()


def write_results(extra: dict | None = None):
    name = "smoke_test.json" if args.layout == "baseline" else "smoke_test_random.json"
    output = args.output or os.path.join(PROJECT_ROOT, "results", name)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump({**(extra or {}), "checks": results}, f, indent=2)
    n_fail = sum(not r["passed"] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed -> {output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
