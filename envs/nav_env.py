"""Goal navigation with obstacle avoidance for a differential-drive robot with a 2D LiDAR (Isaac Lab DirectRLEnv).

Every environment is a walled 4 m x 4 m arena. Its obstacles and goal come from a layout bank: the fixed
baseline scene, or (``cfg.layout = "random"``) a new random layout drawn at every episode reset.

Obstacles are static within an episode and only exist as tensors (for the LiDAR) and as visual markers (for
videos); they have no physics collider. Collisions are detected from the LiDAR (closest reading < 0.25 m),
which ends the episode before the robot body (~0.07 m half-width) could touch an obstacle.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from rewards.navigation_reward import TERM_NAMES, compute_reward_terms, termination_flags
from robots.diff_drive import scale_actions, unicycle_to_wheel_velocities
from robots.jetbot import JETBOT_WHEEL_JOINTS

from .layouts import ARENA_INNER_HALF, OBSTACLE_HEIGHT, baseline_layout, layouts_to_arrays, make_layout_bank
from .lidar_processing import clean_ranges, downsample_min, planar_ray_angles
from .lidar_raycast import cast_planar_lidar
from .nav_env_cfg import WALLS_MESH_PATH, NavEnvCfg
from .obstacle_mesh import build_walls_mesh
from .observations import OBS_DIM, build_observation, goal_distance, heading_to_goal

# Outcome codes, in priority order when several conditions fire on the same step.
OUTCOME_NONE, OUTCOME_GOAL, OUTCOME_COLLISION, OUTCOME_TIMEOUT = 0, 1, 2, 3

OBSTACLE_COLOR = (0.55, 0.55, 0.6)


class NavEnv(DirectRLEnv):
    cfg: NavEnvCfg

    def __init__(self, cfg: NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.max_episode_length != cfg.reward.max_episode_length:
            raise ValueError(
                f"Episode length mismatch: env has {self.max_episode_length} steps, "
                f"reward expects {cfg.reward.max_episode_length}."
            )
        self._wheel_ids, _ = self.robot.find_joints(list(JETBOT_WHEEL_JOINTS), preserve_order=True)

        n, dev = self.num_envs, self.device
        self._origins_xy = self.scene.env_origins[:, :2]
        self._ray_angles = planar_ray_angles(
            cfg.lidar_num_rays, cfg.lidar_start_angle_deg, cfg.lidar_resolution_deg
        ).to(dev)

        # layout bank (all layouts, on the GPU) and the layout currently used by each env
        self.bank = {k: torch.as_tensor(v, device=dev) for k, v in self._make_bank().items()}
        self.bank_size = self.bank["goal"].shape[0]  # layouts drawn at random (the curriculum entry excluded)
        self._baseline_idx = None
        if cfg.layout == "random" and cfg.curriculum_steps > 0:
            # the baseline scene is appended as one extra bank entry, only used by the curriculum
            extra = layouts_to_arrays([baseline_layout()], max_obstacles=self.bank["circles"].shape[1])
            self.bank = {k: torch.cat((v, torch.as_tensor(extra[k], device=dev))) for k, v in self.bank.items()}
            self._baseline_idx = self.bank_size
        k = self.bank["circles"].shape[1]
        self.layout_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self._circles = torch.zeros(n, k, 3, device=dev)
        self._circle_mask = torch.zeros(n, k, dtype=torch.bool, device=dev)
        self._boxes = torch.zeros(n, k, 5, device=dev)
        self._box_mask = torch.zeros(n, k, dtype=torch.bool, device=dev)
        self._start_xy_w = torch.zeros(n, 2, device=dev)
        self._start_yaw = torch.zeros(n, device=dev)
        self._goal_xy_w = torch.zeros(n, 2, device=dev)
        self._start_goal_distance = torch.zeros(n, device=dev)
        self._assign_layouts(self.robot._ALL_INDICES)

        # actuation buffers
        self._commands = torch.zeros(n, 2, device=dev)  # applied (v, w) after scaling + noise
        self._wheel_targets = torch.zeros(n, 2, device=dev)
        # per-episode noise offsets
        self._obs_bias = torch.zeros(n, OBS_DIM, device=dev)
        self._action_bias = torch.zeros(n, 2, device=dev)
        # reward / termination state
        self._prev_distance = self._start_goal_distance.clone()
        self._goal_reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._collided = torch.zeros(n, dtype=torch.bool, device=dev)
        self._timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_outcome = torch.zeros(n, dtype=torch.long, device=dev)
        # episode statistics
        self._episode_sums = {name: torch.zeros(n, device=dev) for name in (*TERM_NAMES, "total")}
        self._path_length = torch.zeros(n, device=dev)
        self._min_clearance = torch.full((n,), float("inf"), device=dev)
        self._prev_xy = self._start_xy_w.clone()
        self.state: dict[str, torch.Tensor] = {}
        self.finished_episodes: dict[str, torch.Tensor] | None = None

        # Visual markers are only updated when something is rendered (videos / GUI).
        self._render_visuals = self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=OBSTACLE_COLOR)
        self._obstacle_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Obstacles",
                markers={
                    # unit shapes, scaled per instance to the obstacle's size
                    "cylinder": sim_utils.CylinderCfg(radius=1.0, height=OBSTACLE_HEIGHT, visual_material=material),
                    "box": sim_utils.CuboidCfg(size=(1.0, 1.0, OBSTACLE_HEIGHT), visual_material=material),
                },
            )
        )
        self._goal_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/GoalMarkers",
                markers={
                    "goal": sim_utils.SphereCfg(
                        radius=0.06, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.2))
                    )
                },
            )
        )
        self._update_visuals()

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground", WALLS_MESH_PATH])
        self.scene.articulations["robot"] = self.robot

        # The arena walls never change: one static mesh (visual + collider) for all envs.
        create_prim_from_mesh(
            WALLS_MESH_PATH,
            build_walls_mesh(self.scene.env_origins.cpu().numpy()),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.88)),
            physics_material=self.cfg.sim.physics_material,
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _make_bank(self) -> dict[str, np.ndarray]:
        if self.cfg.layout == "baseline":
            return layouts_to_arrays([baseline_layout()], max_obstacles=4)
        if self.cfg.layout == "random":
            return make_layout_bank(
                self.cfg.layout_bank_size, self.cfg.layout_seed, self.cfg.layout_cache_dir,
                **self.cfg.random_layout_kwargs,
            )
        raise ValueError(f"Unknown layout '{self.cfg.layout}'.")

    def _assign_layouts(self, env_ids: torch.Tensor):
        """Give each env in ``env_ids`` a layout from the bank (a random one unless the bank has one entry)."""
        if self.bank_size == 1:
            idx = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        else:
            idx = torch.randint(self.bank_size, (len(env_ids),), device=self.device)
        if self._baseline_idx is not None:
            p_baseline = max(0.0, 1.0 - self.common_step_counter / self.cfg.curriculum_steps)
            use_baseline = torch.rand(len(env_ids), device=self.device) < p_baseline
            idx = torch.where(use_baseline, self._baseline_idx, idx)
        self.layout_idx[env_ids] = idx
        self._circles[env_ids] = self.bank["circles"][idx]
        self._circle_mask[env_ids] = self.bank["circle_mask"][idx]
        self._boxes[env_ids] = self.bank["boxes"][idx]
        self._box_mask[env_ids] = self.bank["box_mask"][idx]
        self._start_xy_w[env_ids] = self.bank["start"][idx] + self._origins_xy[env_ids]
        self._start_yaw[env_ids] = self.bank["start_yaw"][idx]
        self._goal_xy_w[env_ids] = self.bank["goal"][idx] + self._origins_xy[env_ids]
        self._start_goal_distance[env_ids] = goal_distance(self._start_xy_w[env_ids], self._goal_xy_w[env_ids])

    def _update_visuals(self):
        """Move the obstacle and goal markers to the current layouts (only when rendering)."""
        if not self._render_visuals:
            return
        n, k, dev = self.num_envs, self._circles.shape[1], self.device
        origins = self._origins_xy[:, None, :]
        z = torch.full((n, k, 1), OBSTACLE_HEIGHT / 2.0, device=dev)
        hidden = torch.tensor([0.0, 0.0, -10.0], device=dev)  # unused slots are moved below the ground
        # cylinders: scale the unit cylinder by the radius
        c_pos = torch.cat((self._circles[..., :2] + origins, z), dim=-1)
        c_pos = torch.where(self._circle_mask[..., None], c_pos, hidden)
        c_scale = torch.cat((self._circles[..., 2:3].expand(n, k, 2), torch.ones(n, k, 1, device=dev)), dim=-1)
        c_quat = torch.zeros(n, k, 4, device=dev)
        c_quat[..., 0] = 1.0
        # boxes: scale the unit cube by the full side lengths, rotate by yaw
        b_pos = torch.cat((self._boxes[..., :2] + origins, z), dim=-1)
        b_pos = torch.where(self._box_mask[..., None], b_pos, hidden)
        b_scale = torch.cat((2.0 * self._boxes[..., 2:4], torch.ones(n, k, 1, device=dev)), dim=-1)
        zeros = torch.zeros(n * k, device=dev)
        b_quat = quat_from_euler_xyz(zeros, zeros, self._boxes[..., 4].reshape(-1)).view(n, k, 4)

        indices = torch.cat((torch.zeros(n, k, dtype=torch.long), torch.ones(n, k, dtype=torch.long)), dim=1)
        self._obstacle_markers.visualize(
            translations=torch.cat((c_pos, b_pos), dim=1).reshape(-1, 3),
            orientations=torch.cat((c_quat, b_quat), dim=1).reshape(-1, 4),
            scales=torch.cat((c_scale, b_scale), dim=1).reshape(-1, 3),
            marker_indices=indices.reshape(-1),
        )
        goal_pos = torch.cat((self._goal_xy_w, torch.full((n, 1), 0.06, device=dev)), dim=-1)
        self._goal_markers.visualize(translations=goal_pos)

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        # Episode logs are only meaningful on the step an episode ends.
        self.extras.pop("log", None)
        self.finished_episodes = None

        commands = scale_actions(actions, self.cfg.action_low, self.cfg.action_high)
        if self.cfg.enable_noise:
            commands = commands + self._action_bias
        # Env-side clip to |a| <= 1 after the noise: only matters if noise pushes v above 1 m/s.
        self._commands = commands.clamp(-1.0, 1.0)
        self._wheel_targets = unicycle_to_wheel_velocities(
            self._commands[:, 0], self._commands[:, 1], self.cfg.wheel_radius, self.cfg.wheel_base
        )

    def _apply_action(self):
        self.robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)

    # ------------------------------------------------------------------ state
    def _compute_state(self) -> dict[str, torch.Tensor]:
        """Noise-free quantities shared by terminations, rewards and observations."""
        robot_xy = self.robot.data.root_pos_w[:, :2]
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        distances = cast_planar_lidar(
            robot_xy - self._origins_xy, yaw, self._ray_angles, ARENA_INNER_HALF,
            self._circles, self._circle_mask, self._boxes, self._box_mask,
        )
        ranges = clean_ranges(distances, self.cfg.lidar_min_range, self.cfg.lidar_max_range)
        sectors = downsample_min(ranges, self.cfg.lidar_group_size)
        self.state = {
            "ranges": ranges,
            "sectors": sectors,
            "min_range": sectors.min(dim=-1).values,
            "robot_xy": robot_xy,
            "yaw": yaw,
            "distance": goal_distance(robot_xy, self._goal_xy_w),
            "heading": heading_to_goal(robot_xy, yaw, self._goal_xy_w),
        }
        return self.state

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Isaac Lab calls this before _get_rewards, so the shared state is computed here.
        s = self._compute_state()
        goal, collided, timed_out = termination_flags(
            s["min_range"], s["distance"], self.episode_length_buf, self.cfg.reward
        )
        self._goal_reached[:], self._collided[:], self._timed_out[:] = goal, collided, timed_out
        # Kept until the next step, so callers can still read it after the automatic reset.
        self.last_outcome = self.outcome()
        return goal | collided, timed_out

    def _get_rewards(self) -> torch.Tensor:
        s = self.state
        terms = compute_reward_terms(
            s["min_range"],
            s["distance"],
            self._prev_distance,
            self.episode_length_buf,
            self._goal_reached,
            self._collided,
            self._timed_out,
            self.cfg.reward,
        )
        total = torch.stack(list(terms.values()), dim=0).sum(dim=0)
        # per-step diagnostics read by the video recorders (taken before any automatic reset)
        self.last_reward_terms = terms
        self.last_min_range = s["min_range"]

        for name, value in terms.items():
            self._episode_sums[name] += value
        self._episode_sums["total"] += total
        self._path_length += torch.linalg.norm(s["robot_xy"] - self._prev_xy, dim=-1)
        self._prev_xy = s["robot_xy"].clone()
        self._min_clearance = torch.minimum(self._min_clearance, s["min_range"])
        self._prev_distance = s["distance"].clone()
        return total

    def _get_observations(self) -> dict:
        # Recomputed because environments that were just reset have a new start pose and layout.
        s = self._compute_state()
        obs = build_observation(
            s["sectors"],
            s["heading"],
            s["distance"],
            self.robot.data.root_lin_vel_w[:, :2],
            self.robot.data.root_ang_vel_w[:, 2],
        )
        if self.cfg.enable_noise:
            obs = obs + self._obs_bias
        return {"policy": obs}

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self._log_finished_episodes(env_ids)
        super()._reset_idx(env_ids)

        # new obstacles + goal for this episode
        self._assign_layouts(env_ids)

        num = len(env_ids)
        root_pos = torch.zeros(num, 3, device=self.device)
        root_pos[:, :2] = self._start_xy_w[env_ids]
        root_pos[:, 2] = self.scene.env_origins[env_ids, 2] + self.robot.data.default_root_state[env_ids, 2]
        zeros = torch.zeros(num, device=self.device)
        root_quat = quat_from_euler_xyz(zeros, zeros, self._start_yaw[env_ids])
        self.robot.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(num, 6, device=self.device), env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), None, env_ids)

        # Progress reward of the first step must be measured from the start, not from the previous episode.
        self._prev_distance[env_ids] = self._start_goal_distance[env_ids]
        self._prev_xy[env_ids] = self._start_xy_w[env_ids]
        self._path_length[env_ids] = 0.0
        self._min_clearance[env_ids] = float("inf")
        for buf in self._episode_sums.values():
            buf[env_ids] = 0.0
        self._goal_reached[env_ids] = False
        self._collided[env_ids] = False
        self._timed_out[env_ids] = False
        self._wheel_targets[env_ids] = 0.0

        if self.cfg.enable_noise:
            self._obs_bias[env_ids] = self.cfg.observation_noise_std * torch.randn(num, OBS_DIM, device=self.device)
            self._action_bias[env_ids] = self.cfg.action_noise_std * torch.randn(num, 2, device=self.device)

        self._update_visuals()

    # ---------------------------------------------------------------- metrics
    def outcome(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Outcome code of the current step (goal > collision > timeout priority)."""
        ids = slice(None) if env_ids is None else env_ids
        code = torch.full_like(self.episode_length_buf[ids], OUTCOME_NONE)
        code = torch.where(self._timed_out[ids], OUTCOME_TIMEOUT, code)
        code = torch.where(self._collided[ids], OUTCOME_COLLISION, code)
        return torch.where(self._goal_reached[ids], OUTCOME_GOAL, code)

    def _log_finished_episodes(self, env_ids: torch.Tensor):
        """Store the episodes ending now: means in ``extras['log']``, per-episode values in ``finished_episodes``."""
        ids = torch.as_tensor(env_ids, device=self.device)
        ids = ids[self.episode_length_buf[ids] > 0]  # skip the initial reset
        if len(ids) == 0:
            return
        outcome = self.outcome(ids)
        self.finished_episodes = {
            "env_ids": ids,
            "layout_idx": self.layout_idx[ids].clone(),
            "outcome": outcome,
            "length": self.episode_length_buf[ids].clone(),
            "return": self._episode_sums["total"][ids].clone(),
            "path_length": self._path_length[ids].clone(),
            "min_clearance": self._min_clearance[ids].clone(),
            "start_goal_distance": self._start_goal_distance[ids].clone(),
        }
        log = {
            "num_episodes": float(len(ids)),
            "success_rate": (outcome == OUTCOME_GOAL).float().mean(),
            "collision_rate": (outcome == OUTCOME_COLLISION).float().mean(),
            "timeout_rate": (outcome == OUTCOME_TIMEOUT).float().mean(),
            "episode_length": self.episode_length_buf[ids].float().mean(),
            "episode_return": self._episode_sums["total"][ids].mean(),
            "distance_traveled": self._path_length[ids].mean(),
            "min_clearance": self._min_clearance[ids].mean(),
        }
        for name in TERM_NAMES:
            log[f"reward/{name}"] = self._episode_sums[name][ids].mean()
        self.extras["log"] = log
