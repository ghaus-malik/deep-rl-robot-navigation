"""Log every training episode, and film chosen training episodes from the start point to their end.

Training with a fixed seed is reproducible, so complete episodes with a given outcome are filmed in two runs:

1. ``EpisodeLogger`` (no rendering) writes one CSV row per finished episode: which env, the global step at
   which it started and ended, its outcome and return.
2. A schedule is built from that CSV (``scripts/select_episodes.py``) and the same training is run again with
   ``EpisodeVideoRecorder``, which moves the top-down camera over the scheduled env when that episode starts
   and records until it ends. Only one camera exists, so scheduled episodes must not overlap in time.

"Global step" = number of ``env.step`` calls made so far (RL-Games makes ``horizon_length`` calls per iteration).
"""

from __future__ import annotations

import csv
import json
import os

import gymnasium as gym
import imageio
import numpy as np
import torch

from rewards.navigation_reward import TERM_NAMES

OUTCOME_NAMES = {0: "none", 1: "goal", 2: "collision", 3: "timeout"}
CSV_FIELDS = (
    "env_id", "start_step", "end_step", "length", "outcome", "return", "layout_idx", "path_length", "min_clearance",
)


class EpisodeLogger(gym.Wrapper):
    """Record when each episode of each env starts and ends, which layout it used and how it ended."""

    def __init__(self, env: gym.Env, path: str):
        super().__init__(env)
        self.path = path
        self._step_count = 0
        self._start = torch.zeros(self.unwrapped.num_envs, dtype=torch.long)  # all envs start at step 0
        self._rows: list[dict] = []
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def step(self, action):
        obs, rew, terminated, truncated, extras = self.env.step(action)
        self._step_count += 1
        finished = self.unwrapped.finished_episodes  # filled by the env for the episodes that ended this step
        if finished is not None:
            values = {k: v.tolist() for k, v in finished.items()}
            for i, env_id in enumerate(values["env_ids"]):
                start = int(self._start[env_id])
                self._rows.append({
                    "env_id": env_id,
                    "start_step": start,
                    "end_step": self._step_count,
                    "length": self._step_count - start,
                    "outcome": OUTCOME_NAMES[values["outcome"][i]],
                    "return": round(values["return"][i], 4),
                    "layout_idx": values["layout_idx"][i],
                    "path_length": round(values["path_length"][i], 4),
                    "min_clearance": round(values["min_clearance"][i], 4),
                })
                self._start[env_id] = self._step_count  # the next episode starts after this step
        return obs, rew, terminated, truncated, extras

    def close(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[EPISODES] {len(self._rows)} finished episodes -> {self.path}")
        return self.env.close()


class EpisodeVideoRecorder(gym.Wrapper):
    """Film scheduled episodes (env id + start step) from the start pose until the episode ends."""

    def __init__(self, env: gym.Env, schedule: list[dict], video_dir: str, warmup_renders: int = 10):
        """
        Args:
            env: The navigation environment, created with ``render_mode="rgb_array"``.
            schedule: Items with ``name``, ``env_id``, ``start_step`` and ``expected_outcome``.
            video_dir: Output folder for ``<name>.mp4`` and ``episodes_index.json``.
            warmup_renders: Renders discarded after moving the camera so the renderer can settle.
        """
        super().__init__(env)
        self.schedule = {item["start_step"]: item for item in schedule}
        if len(self.schedule) != len(schedule):
            raise ValueError("Two scheduled episodes start at the same step; only one camera is available.")
        self.video_dir = video_dir
        self.warmup_renders = warmup_renders
        self.fps = round(1.0 / self.unwrapped.step_dt)  # real time: one frame per policy step
        self.max_steps = self.unwrapped.max_episode_length
        self._step_count = 0
        self._writer = None
        os.makedirs(video_dir, exist_ok=True)
        # several runs may film into the same folder (overlapping episodes need separate runs): keep their entries
        self._index_path = os.path.join(video_dir, "episodes_index.json")
        names = {item["name"] for item in schedule}
        self._index: list[dict] = []
        if os.path.exists(self._index_path):
            with open(self._index_path) as f:
                self._index = [e for e in json.load(f) if e["name"] not in names]

    def step(self, action):
        item = self.schedule.get(self._step_count)
        if item is not None:
            if self._writer is None:
                self._start(item)
            else:
                print(f"[VIDEO] skipped '{item['name']}': camera busy with '{self._info['name']}'")

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
                # the env is already reset inside step(), so the last frame kept is the one before termination
                self._info["outcome"] = OUTCOME_NAMES[int(self.unwrapped.last_outcome[i])]
                self._finish()
            elif self._info["steps"] >= self.max_steps:
                self._info["outcome"] = "not finished"
                self._finish()
            else:
                self._writer.append_data(self._grab())
        return obs, rew, terminated, truncated, extras

    def close(self):
        if self._writer is not None:
            self._info["outcome"] = "not finished (training ended)"
            self._finish()
        return self.env.close()

    def _grab(self) -> np.ndarray:
        return np.ascontiguousarray(self.env.render()[:, :, :3])

    def _start(self, item: dict):
        env = self.unwrapped
        self._env_id = int(item["env_id"])
        env.viewport_camera_controller.set_view_env_index(self._env_id)
        for _ in range(self.warmup_renders):
            env.render()
        path = os.path.join(self.video_dir, f"{item['name']}.mp4")
        self._writer = imageio.get_writer(path, fps=self.fps, codec="libx264", pixelformat="yuv420p", quality=8)
        self._info = {
            **item,
            "file": os.path.basename(path),
            "episode_step_at_start": int(env.episode_length_buf[self._env_id]),  # 0 = at the start point
            "steps": 0,
            "return": 0.0,
            "reward_terms": {name: 0.0 for name in TERM_NAMES},
            "min_lidar_range": float("inf"),
            "outcome": None,
        }
        self._writer.append_data(self._grab())  # start pose, before the first action
        print(f"[VIDEO] '{item['name']}': recording env {self._env_id} from step {self._step_count} -> {path}")

    def _finish(self):
        self._writer.close()
        self._writer = None
        self._info["matches_expected"] = self._info["outcome"] == self._info.get("expected_outcome")
        self._index.append(self._info)
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)
        print(f"[VIDEO] '{self._info['name']}': {self._info['steps']} steps, outcome={self._info['outcome']}, "
              f"return={self._info['return']:.2f}")
