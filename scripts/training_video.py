"""Record short top-down videos of one environment during selected PPO training iterations.

RL-Games collects exactly ``horizon_length`` environment steps per iteration (after a single initial reset),
so the current iteration is ``step_count // horizon_length + 1``. At the first step of a selected iteration
one environment is drawn at random and the top-down viewport camera is moved over it. Its frames are
recorded for that iteration's rollout only and recording stops early if its episode ends.

The policy is not touched: frames show the stochastic actions PPO samples during training. Rendering is
only triggered while recording, so the other iterations run at full speed.
"""

from __future__ import annotations

import json
import os

import gymnasium as gym
import imageio
import numpy as np

from rewards.navigation_reward import TERM_NAMES

OUTCOME_NAMES = {0: "none", 1: "goal", 2: "collision", 3: "timeout"}


class TrainingVideoRecorder(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        iterations: list[int],
        horizon_length: int,
        video_dir: str,
        seed: int,
        max_steps: int = 1200,
        warmup_renders: int = 10,
    ):
        """
        Args:
            env: The navigation environment, created with ``render_mode="rgb_array"``.
            iterations: Training iterations (1-based) to record.
            horizon_length: Environment steps per PPO iteration (``horizon_length`` in the agent config).
            video_dir: Output folder for ``iter_XXXX.mp4`` files and ``index.json``.
            seed: Seed for choosing which environment to record (separate RNG, training RNG untouched).
            max_steps: Upper bound on recorded steps (an episode is at most 1200 steps).
            warmup_renders: Renders discarded after moving the camera so the renderer can settle.
        """
        super().__init__(env)
        self.iterations = set(iterations)
        self.horizon_length = horizon_length
        self.max_steps = min(max_steps, horizon_length)
        self.video_dir = video_dir
        self.warmup_renders = warmup_renders
        self.fps = round(1.0 / self.unwrapped.step_dt)  # real time: one frame per policy step
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._writer = None
        self._index: list[dict] = []
        os.makedirs(video_dir, exist_ok=True)

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        iteration = self._step_count // self.horizon_length + 1
        if self._step_count % self.horizon_length == 0 and iteration in self.iterations:
            self._start(iteration)

        obs, rew, terminated, truncated, extras = self.env.step(action)
        self._step_count += 1

        if self._writer is not None:
            i = self._env_id
            self._info["steps"] += 1
            self._info["return"] += float(rew[i])
            for name, value in self.unwrapped.last_reward_terms.items():
                self._info["reward_terms"][name] += float(value[i])
            self._info["min_lidar_range"] = min(self._info["min_lidar_range"], float(self.unwrapped.last_min_range[i]))
            if bool(terminated[i] | truncated[i]):
                # the env has already been reset inside step(), so this frame would show the start pose again
                self._info["outcome"] = OUTCOME_NAMES[int(self.unwrapped.last_outcome[i])]
                self._finish()
            else:
                self._writer.append_data(self._grab())
                if self._info["steps"] >= self.max_steps or self._step_count % self.horizon_length == 0:
                    self._info["outcome"] = "in progress (end of iteration)"
                    self._finish()
        return obs, rew, terminated, truncated, extras

    def close(self):
        if self._writer is not None:
            self._info["outcome"] = "in progress (training ended)"
            self._finish()
        return self.env.close()

    # ---------------------------------------------------------------- recording
    def _grab(self) -> np.ndarray:
        return np.ascontiguousarray(self.env.render()[:, :, :3])

    def _start(self, iteration: int):
        env = self.unwrapped
        self._env_id = int(self._rng.integers(env.num_envs))
        env.viewport_camera_controller.set_view_env_index(self._env_id)
        for _ in range(self.warmup_renders):
            env.render()
        path = os.path.join(self.video_dir, f"iter_{iteration:04d}.mp4")
        # yuv420p H.264 plays in every common player; 720x720 is a multiple of the 16-px macro block
        self._writer = imageio.get_writer(path, fps=self.fps, codec="libx264", pixelformat="yuv420p", quality=8)
        self._info = {
            "iteration": iteration,
            "env_id": self._env_id,
            "episode_step_at_start": int(env.episode_length_buf[self._env_id]),
            "steps": 0,
            "return": 0.0,
            "reward_terms": {name: 0.0 for name in TERM_NAMES},
            "min_lidar_range": float("inf"),
            "outcome": None,
            "file": os.path.basename(path),
        }
        self._writer.append_data(self._grab())  # state before the first action of the iteration
        print(f"[VIDEO] iteration {iteration}: recording env {self._env_id} -> {path}")

    def _finish(self):
        self._writer.close()
        self._writer = None
        self._info["frames"] = self._info["steps"] + (1 if self._info["outcome"].startswith("in progress") else 0)
        self._index.append(self._info)
        with open(os.path.join(self.video_dir, "index.json"), "w") as f:
            json.dump(self._index, f, indent=2)
        print(f"[VIDEO] iteration {self._info['iteration']}: {self._info['steps']} steps, "
              f"outcome={self._info['outcome']}, return={self._info['return']:.2f}")
