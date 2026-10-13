"""Per-step reward for goal-reaching with static obstacle avoidance (pure PyTorch).

Total reward = sum of the components returned by :func:`compute_reward_terms`:

    proximity  proximity_mode "step" (default):
                   -1 if min_range < 0.40, additional -2 if < 0.35, additional -2 if < 0.30 (cumulative: -1/-3/-5)
               proximity_mode "exponential" (softer, grows smoothly as the robot gets closer):
                   -exp(-min_range) if min_range < 0.40   (about -0.67 at 0.40 m, -0.78 at 0.25 m)
    collision  -30 when min_range < 0.25 (collision is detected from the LiDAR, not from contacts)
    timeout    -30 on the last allowed step of the episode
    goal       +30 when the planar goal distance < 0.30
    time_bonus +0.02 * (max_episode_length - step) on the step the goal is reached
    progress   +10 * (d_prev - d)
    potential  +0.1 * (phi - phi_prev), phi = -d / potential_dt, i.e. +0.1 * (d_prev - d) / potential_dt

All terms are evaluated independently, so several can fire in the same step (e.g. goal + timeout).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TERM_NAMES = ("proximity", "collision", "timeout", "goal", "time_bonus", "progress", "potential")


@dataclass
class RewardCfg:
    goal_radius: float = 0.30
    collision_range: float = 0.25
    proximity_thresholds: tuple[float, ...] = (0.40, 0.35, 0.30)
    proximity_penalties: tuple[float, ...] = (1.0, 2.0, 2.0)
    # "step": cumulative penalties above; "exponential": -exp(-min_range) below the first threshold
    proximity_mode: str = "step"
    goal_reward: float = 30.0
    collision_penalty: float = 30.0
    timeout_penalty: float = 30.0
    time_bonus_scale: float = 0.02
    progress_scale: float = 10.0
    potential_scale: float = 0.1
    # The potential is divided by a fixed 1/60 s even though the control step is 1/40 s.
    # Kept as a constant (not tied to the env step) to preserve the baseline's reward magnitudes.
    potential_dt: float = 1.0 / 60.0
    max_episode_length: int = 1200


def termination_flags(
    min_range: torch.Tensor, distance: torch.Tensor, episode_step: torch.Tensor, cfg: RewardCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return boolean (goal_reached, collided, timed_out), each shape (N,).

    ``episode_step`` counts steps already taken in the episode (1 on the first step).
    """
    goal_reached = distance < cfg.goal_radius
    collided = min_range < cfg.collision_range
    timed_out = episode_step >= cfg.max_episode_length - 1
    return goal_reached, collided, timed_out


def compute_reward_terms(
    min_range: torch.Tensor,
    distance: torch.Tensor,
    prev_distance: torch.Tensor,
    episode_step: torch.Tensor,
    goal_reached: torch.Tensor,
    collided: torch.Tensor,
    timed_out: torch.Tensor,
    cfg: RewardCfg,
) -> dict[str, torch.Tensor]:
    """Compute every reward component, each of shape (N,). Use ``sum(terms.values())`` for the total."""
    if cfg.proximity_mode == "step":
        proximity = torch.zeros_like(distance)
        for threshold, penalty in zip(cfg.proximity_thresholds, cfg.proximity_penalties):
            proximity = proximity - penalty * (min_range < threshold).float()
    elif cfg.proximity_mode == "exponential":
        near = (min_range < cfg.proximity_thresholds[0]).float()
        proximity = -torch.exp(-min_range) * near
    else:
        raise ValueError(f"Unknown proximity_mode '{cfg.proximity_mode}'.")

    remaining_steps = (cfg.max_episode_length - episode_step).float()
    delta = prev_distance - distance

    return {
        "proximity": proximity,
        "collision": -cfg.collision_penalty * collided.float(),
        "timeout": -cfg.timeout_penalty * timed_out.float(),
        "goal": cfg.goal_reward * goal_reached.float(),
        "time_bonus": cfg.time_bonus_scale * remaining_steps * goal_reached.float(),
        "progress": cfg.progress_scale * delta,
        "potential": cfg.potential_scale * delta / cfg.potential_dt,
    }
