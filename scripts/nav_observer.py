"""RL-Games observer that logs navigation episode statistics and GPU memory to TensorBoard."""

from __future__ import annotations

from collections import defaultdict

import torch
from rl_games.common.algo_observer import AlgoObserver


class NavObserver(AlgoObserver):
    """Aggregates ``infos['episode']`` (filled by the env when episodes end) over each PPO iteration.

    Every log entry is a mean over the ``num_episodes`` episodes that ended on that step, so values are
    weighted by that count to obtain the exact mean over all episodes finished during the iteration.
    """

    def after_init(self, algo):
        self.algo = algo
        self.writer = algo.writer
        self._sums: dict[str, float] = defaultdict(float)
        self._count = 0.0

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict):
            return
        episode = infos.get("episode")
        if not episode:
            return
        n = float(episode.get("num_episodes", 0.0))
        if n <= 0:
            return
        for key, value in episode.items():
            if key == "num_episodes":
                continue
            self._sums[key] += float(value) * n
        self._count += n

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None:
            return
        if self._count > 0:
            for key, total in self._sums.items():
                self.writer.add_scalar(f"Episode/{key}", total / self._count, epoch_num)
            self.writer.add_scalar("Episode/episodes_finished", self._count, epoch_num)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            # whole-device usage, i.e. it also includes other users' processes on the same GPU
            self.writer.add_scalar("GPU/device_memory_used_all_processes_GB", (total - free) / 1e9, epoch_num)
            self.writer.add_scalar("GPU/torch_max_allocated_GB", torch.cuda.max_memory_allocated(device) / 1e9, epoch_num)
        self._sums.clear()
        self._count = 0.0
        # Isaac Sim ends the process abruptly on close, so unflushed events would be lost.
        self.writer.flush()
