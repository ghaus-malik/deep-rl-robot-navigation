"""Test a trained policy on layouts that were never used in training (deterministic actions).

Each test episode ends when the robot reaches the goal, collides, or reaches the step limit (1200 steps, 30 s).
Test layouts come from a separate layout bank (``--layout_seed`` differs from the training seed 0), so every
obstacle set and goal is new to the policy.

Metrics over many episodes (64 envs, first ``--episodes_per_env`` episodes of each env, no rendering):

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --headless --checkpoint runs/<exp>/nn/<file>.pth

Videos of complete test episodes (1 env, top-down camera):

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --headless --checkpoint ... --videos 5
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser(description="Evaluate a trained navigation policy.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "ppo_static_baseline.yaml"))
parser.add_argument("--layout", choices=("baseline", "random"), default="random")
parser.add_argument("--layout_seed", type=int, default=1000, help="Test bank seed (training uses 0).")
parser.add_argument("--layout_bank_size", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes_per_env", type=int, default=8)
parser.add_argument("--videos", type=int, default=0, help="If > 0: film this many complete episodes (1 env).")
parser.add_argument("--video_dir", default=os.path.join(PROJECT_ROOT, "videos", "testing"))
parser.add_argument("--no_noise", action="store_true", help="Disable per-episode observation/action noise.")
parser.add_argument("--proximity", choices=("step", "exponential"), default="step",
                    help="Proximity penalty used for the logged returns (outcomes do not depend on it).")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--output", default=None, help="Default: results/eval_<layout>_seed<layout_seed>[_videos].json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.videos > 0:
    args.enable_cameras = True

simulation_app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

from envs.nav_env import NavEnv  # noqa: E402
from envs.nav_env_cfg import NavEnvCfg  # noqa: E402
from scripts.episode_recording import OUTCOME_NAMES  # noqa: E402

CHECKPOINT_NUMPY_GLOBALS = [np.core.multiarray.scalar, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType]


class EpisodeFilm:
    """Writes one MP4 per episode of env 0, from its start pose to the step before it ends."""

    def __init__(self, env: NavEnv, video_dir: str, warmup_renders: int = 5):
        self.env, self.video_dir, self.warmup = env, video_dir, warmup_renders
        self.fps = round(1.0 / env.step_dt)
        self.count = 0
        self.writer = None
        self.path = None
        os.makedirs(video_dir, exist_ok=True)

    def _frame(self):
        return np.ascontiguousarray(self.env.render()[:, :, :3])

    def start(self):
        for _ in range(self.warmup):  # let the renderer settle on the new layout
            self.env.render()
        self.count += 1
        self.path = os.path.join(self.video_dir, f"test_{self.count}.mp4")
        self.writer = imageio.get_writer(self.path, fps=self.fps, codec="libx264", pixelformat="yuv420p", quality=8)
        self.writer.append_data(self._frame())

    def add(self):
        self.writer.append_data(self._frame())

    def finish(self, outcome: str) -> str:
        """Close the video and add the outcome to its file name."""
        self.writer.close()
        self.writer = None
        final = self.path.replace(".mp4", f"_{outcome}.mp4")
        os.replace(self.path, final)
        return final


def main():
    with open(args.config) as f:
        agent_cfg = yaml.safe_load(f)
    num_envs = 1 if args.videos > 0 else args.num_envs
    episodes_per_env = args.videos if args.videos > 0 else args.episodes_per_env

    env_cfg = NavEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.layout = args.layout
    env_cfg.layout_seed = args.layout_seed
    env_cfg.layout_bank_size = args.layout_bank_size
    env_cfg.enable_noise = not args.no_noise
    env_cfg.reward.proximity_mode = args.proximity
    env_cfg.seed = args.seed
    nav_env = NavEnv(cfg=env_cfg, render_mode="rgb_array" if args.videos > 0 else None)
    env = RlGamesVecEnvWrapper(
        nav_env,
        agent_cfg["params"]["config"]["device"],
        agent_cfg["params"]["env"].get("clip_observations", math.inf),
        agent_cfg["params"]["env"].get("clip_actions", math.inf),
    )
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    # RL-Games player = the trained network + its observation normaliser, restored from the checkpoint
    agent_cfg["params"]["config"]["num_actors"] = num_envs
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = os.path.abspath(args.checkpoint)
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    # torch >= 2.6 loads weights-only by default; RL-Games checkpoints also hold a few numpy scalars
    with torch.serialization.safe_globals(CHECKPOINT_NUMPY_GLOBALS):
        agent.restore(os.path.abspath(args.checkpoint))
    agent.reset()

    film = EpisodeFilm(nav_env, args.video_dir) if args.videos > 0 else None
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if film:
        film.start()

    done_count = np.zeros(num_envs, dtype=int)
    episodes: list[dict] = []
    while (done_count < episodes_per_env).any():
        with torch.inference_mode():
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
        finished = nav_env.finished_episodes
        if finished is None:
            if film:
                film.add()
            continue
        values = {k: v.tolist() for k, v in finished.items()}
        for i, env_id in enumerate(values["env_ids"]):
            if done_count[env_id] >= episodes_per_env:
                continue  # only the first episodes of each env count (no bias towards short episodes)
            done_count[env_id] += 1
            length = values["length"][i]
            outcome = OUTCOME_NAMES[values["outcome"][i]]
            path = values["path_length"][i]
            episode = {
                "env_id": env_id,
                "layout_idx": values["layout_idx"][i],
                "outcome": outcome,
                "steps": length,
                "time_s": length * nav_env.step_dt,
                "return": values["return"][i],
                "path_length_m": path,
                "start_goal_distance_m": values["start_goal_distance"][i],
                # straight-line distance / driven distance, only meaningful for successful episodes
                "path_efficiency": values["start_goal_distance"][i] / path if outcome == "goal" and path > 0 else None,
                "min_clearance_m": values["min_clearance"][i],
            }
            if film:
                episode["video"] = os.path.relpath(film.finish(outcome), PROJECT_ROOT)
                print(f"[TEST] episode {film.count}: {outcome} after {length} steps -> {episode['video']}")
                if done_count[0] < episodes_per_env:
                    film.start()  # the env has already been reset to the next test layout
            episodes.append(episode)
        if film and film.writer is not None and 0 not in values["env_ids"]:
            film.add()

    summary = summarize(episodes)
    output = args.output or os.path.join(
        PROJECT_ROOT, "results", f"eval_{args.layout}_seed{args.layout_seed}{'_videos' if film else ''}.json"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "layout": args.layout, "layout_seed": args.layout_seed,
                   "noise": not args.no_noise, "summary": summary, "episodes": episodes}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {output}")
    env.close()


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    rate = lambda o: 100.0 * sum(e["outcome"] == o for e in episodes) / n  # noqa: E731
    success = [e for e in episodes if e["outcome"] == "goal"]
    mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
    return {
        "episodes": n,
        "success_%": rate("goal"),
        "collision_%": rate("collision"),
        "timeout_%": rate("timeout"),
        "navigation_time_s (successes)": mean([e["time_s"] for e in success]),
        "path_length_m (successes)": mean([e["path_length_m"] for e in success]),
        "path_efficiency (successes)": mean([e["path_efficiency"] for e in success]),
        "min_clearance_m (all)": mean([e["min_clearance_m"] for e in episodes]),
    }


if __name__ == "__main__":
    main()
    simulation_app.close()
