"""Train the PPO navigation policy with RL-Games.

Examples (on the GPU server, GPU 0 only):

    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --headless
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --headless --num_envs 64 --max_iterations 200 --seed 42
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --headless --checkpoint runs/<exp>/nn/<file>.pth   # resume
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --headless --video   # + videos/training/iter_XXXX.mp4
    # new obstacles and goal every episode, 1000 iterations, log every episode (to pick videos afterwards)
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --headless --layout random --max_iterations 1000 --seed 42 \\
        --episode_log logs/episodes_random_seed42.csv
    # same, with the stability fixes: linear lr, stronger action bound, gamma 0.995, softer proximity
    # penalty and a curriculum from the fixed scene to random layouts over the first 300 iterations
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --headless --layout random --max_iterations 1000 --seed 42 \\
        --config configs/ppo_random_obstacles.yaml --proximity exponential --curriculum_iterations 300
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser(description="Train PPO for static-obstacle navigation.")
parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "ppo_static_baseline.yaml"))
parser.add_argument("--experiment", default=None, help="Run name (default: <config name>_seed<seed>).")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", default=None, help="Checkpoint to resume from.")
parser.add_argument("--no_noise", action="store_true", help="Disable per-episode observation/action noise.")
parser.add_argument("--layout", choices=("baseline", "random"), default="baseline",
                    help="baseline: fixed scene; random: new obstacles + goal every episode.")
parser.add_argument("--layout_seed", type=int, default=0, help="Seed of the random layout bank (training).")
parser.add_argument("--layout_bank_size", type=int, default=None)
parser.add_argument("--proximity", choices=("step", "exponential"), default="step",
                    help="Proximity penalty: step (-1/-3/-5) or exponential (-exp(-min_range)).")
parser.add_argument("--curriculum_iterations", type=int, default=0,
                    help="Random layouts only: share of random layouts grows 0 -> 100%% over this many iterations "
                         "(the rest use the fixed baseline scene). 0 = no curriculum.")
parser.add_argument("--runs_dir", default=os.path.join(PROJECT_ROOT, "runs"))
parser.add_argument("--video", action="store_true", help="Record top-down videos of one env at selected iterations.")
parser.add_argument("--video_iterations", default="1,25,50,100,150,200", help="Comma-separated iterations to record.")
parser.add_argument("--video_dir", default=os.path.join(PROJECT_ROOT, "videos", "training"))
parser.add_argument("--episode_log", default=None, help="CSV file listing every finished training episode.")
parser.add_argument("--record_episodes", default=None, help="JSON schedule from select_episodes.py to film.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video or args.record_episodes:
    args.enable_cameras = True  # headless rendering is required for the viewport camera

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Everything below needs the simulator to be running.
import json  # noqa: E402
import math  # noqa: E402

import yaml  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

from envs.nav_env import NavEnv  # noqa: E402
from envs.nav_env_cfg import NavEnvCfg  # noqa: E402
from scripts.episode_recording import EpisodeLogger, EpisodeVideoRecorder  # noqa: E402
from scripts.nav_observer import NavObserver  # noqa: E402
from scripts.training_video import TrainingVideoRecorder  # noqa: E402


def main():
    with open(args.config) as f:
        agent_cfg = yaml.safe_load(f)
    params = agent_cfg["params"]
    if args.seed is not None:
        params["seed"] = args.seed
    if args.max_iterations is not None:
        params["config"]["max_epochs"] = args.max_iterations

    env_cfg = NavEnvCfg()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.no_noise:
        env_cfg.enable_noise = False
    env_cfg.layout = args.layout
    env_cfg.layout_seed = args.layout_seed
    if args.layout_bank_size is not None:
        env_cfg.layout_bank_size = args.layout_bank_size
    env_cfg.reward.proximity_mode = args.proximity
    env_cfg.curriculum_steps = args.curriculum_iterations * params["config"]["horizon_length"]
    env_cfg.seed = params["seed"]
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device

    if args.layout == "random" and params["config"]["name"] == "static_baseline":
        params["config"]["name"] = "random_obstacles"  # only used in run and checkpoint names
    experiment = args.experiment or f"{params['config']['name']}_seed{params['seed']}"
    params["config"]["train_dir"] = os.path.abspath(args.runs_dir)
    params["config"]["full_experiment_name"] = experiment
    params["config"]["num_actors"] = env_cfg.scene.num_envs
    if args.checkpoint:
        params["load_checkpoint"] = True
        params["load_path"] = os.path.abspath(args.checkpoint)

    run_dir = os.path.join(params["config"]["train_dir"], experiment)
    dump_yaml(os.path.join(run_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(run_dir, "params", "agent.yaml"), agent_cfg)

    env = NavEnv(cfg=env_cfg, render_mode="rgb_array" if (args.video or args.record_episodes) else None)
    if args.episode_log:
        env = EpisodeLogger(env, args.episode_log)
    if args.record_episodes:
        with open(args.record_episodes) as f:
            env = EpisodeVideoRecorder(env, json.load(f), args.video_dir)
    if args.video:
        env = TrainingVideoRecorder(
            env,
            iterations=[int(i) for i in args.video_iterations.split(",")],
            horizon_length=params["config"]["horizon_length"],
            video_dir=args.video_dir,
            seed=params["seed"],
        )
    env = RlGamesVecEnvWrapper(
        env,
        params["config"]["device"],
        params["env"].get("clip_observations", math.inf),
        params["env"].get("clip_actions", math.inf),
    )
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    runner = Runner(NavObserver())
    runner.load(agent_cfg)
    runner.reset()
    run_args = {"train": True, "play": False, "sigma": None}
    if args.checkpoint:
        run_args["checkpoint"] = params["load_path"]
    runner.run(run_args)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
