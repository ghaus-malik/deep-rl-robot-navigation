"""Configuration of the obstacle-avoidance navigation environment."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rewards.navigation_reward import RewardCfg
from robots.jetbot import JETBOT_CFG, JETBOT_WHEEL_BASE, JETBOT_WHEEL_RADIUS

from .observations import OBS_DIM

WALLS_MESH_PATH = "/World/walls"


@configclass
class NavEnvCfg(DirectRLEnvCfg):
    # --- timing: physics at 120 Hz, policy at 40 Hz, 1200 policy steps (30 s) per episode
    decimation = 3
    episode_length_s = 30.0
    # --- spaces
    action_space = 2
    observation_space = OBS_DIM
    state_space = 0
    # Timeouts are treated as terminal (no value bootstrapping), like the baseline.
    is_finite_horizon = True

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
        physx=PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=5.0, replicate_physics=True)

    robot_cfg: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- 2D LiDAR: 360 horizontal rays at 1 deg from the robot's centre, ray 0 pointing backwards and
    # angles increasing counter-clockwise. Ranges are computed exactly against the walls and the current
    # episode's obstacles (envs/lidar_raycast.py), then clipped to [min, max] and reduced to 120 sectors.
    lidar_num_rays: int = 360
    lidar_start_angle_deg: float = -180.0
    lidar_resolution_deg: float = 1.0
    lidar_min_range: float = 0.15
    lidar_max_range: float = 2.0
    lidar_group_size: int = 3  # 360 rays -> 120 sectors

    # --- actions: policy output in [-1, 1] is rescaled to [v_min, v_max] x [w_min, w_max]
    action_low: tuple[float, float] = (0.1, -0.5)  # (m/s, rad/s)
    action_high: tuple[float, float] = (0.5, 0.5)
    wheel_radius: float = JETBOT_WHEEL_RADIUS
    wheel_base: float = JETBOT_WHEEL_BASE

    # --- per-episode noise: one Gaussian offset per env and per dimension, drawn at reset, held all episode
    enable_noise: bool = True
    observation_noise_std: float = 0.2
    action_noise_std: float = 0.05

    # --- layout
    # "baseline": the fixed four-cylinder scene in every episode.
    # "random":   every episode draws a new layout (obstacles + goal) from a bank of ``layout_bank_size``
    #             reachable layouts generated from ``layout_seed``. Use a different seed for testing so the
    #             test layouts are never seen during training.
    layout: str = "baseline"
    layout_seed: int = 0
    layout_bank_size: int = 10000
    random_layout_kwargs: dict = {}  # forwarded to envs.layouts.random_layout
    layout_cache_dir: str = os.path.expanduser("~/.cache/drl_nav")
    # Curriculum (random layouts only): at reset an env gets the fixed baseline scene with probability
    # 1 - step / curriculum_steps, otherwise a random layout. "step" counts env.step calls (128 per PPO
    # iteration), so the share of random layouts grows linearly from 0 to 1. 0 disables the curriculum.
    curriculum_steps: int = 0

    reward: RewardCfg = RewardCfg(max_episode_length=1200)

    # --- viewport camera (rendering only; used for videos). Top-down over one env's arena: with the default
    # 60 deg field of view and a square image, 4.6 m height shows +-2.66 m at ground level, i.e. the whole
    # 4.2 m arena but not the neighbouring arenas (5 m spacing). The tiny y-offset avoids a degenerate
    # look-at when looking straight down along the z (up) axis; image up = +y, image right = +x.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, -0.001, 4.6), lookat=(0.0, 0.0, 0.0), origin_type="env", env_index=0, resolution=(720, 720)
    )
