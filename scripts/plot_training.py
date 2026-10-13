"""Plot episode outcome rates over PPO iterations from episode logs written by ``EpisodeLogger``.

    python scripts/plot_training.py --log results/episodes_random_obstacles_v2_seed42.csv --label "stabilised" \\
        --log results/episodes_random_obstacles_seed42.csv --label "adaptive lr" --out docs/media/training_outcomes.png

The first log is drawn with all three outcomes; further logs only add their success rate (dashed) for comparison.
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"goal": "tab:green", "collision": "tab:red", "timeout": "tab:orange"}


def outcome_rates(path: str, horizon: int, window: int, iterations: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Share of each outcome among the episodes ending in consecutive windows of ``window`` iterations."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    end_iter = np.array([(int(r["end_step"]) - 1) // horizon + 1 for r in rows])
    outcome = np.array([r["outcome"] for r in rows])
    edges = np.arange(0, iterations + window, window)
    centers, rates = [], {o: [] for o in COLORS}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (end_iter > lo) & (end_iter <= hi)
        if sel.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        for o in COLORS:
            rates[o].append(100.0 * (outcome[sel] == o).mean())
    return np.array(centers), {o: np.array(v) for o, v in rates.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=128, help="env steps per PPO iteration")
    parser.add_argument("--window", type=int, default=20, help="iterations per point")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--curriculum", type=int, default=0, help="shade the curriculum phase (iterations)")
    args = parser.parse_args()
    if len(args.log) != len(args.label):
        raise ValueError("Give one --label per --log.")

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    if args.curriculum:
        ax.axvspan(0, args.curriculum, color="0.92", zorder=0)
        ax.text(args.curriculum / 2, 101, "curriculum", ha="center", va="bottom", fontsize=8, color="0.4")
    x, rates = outcome_rates(args.log[0], args.horizon, args.window, args.iterations)
    for o, color in COLORS.items():
        ax.plot(x, rates[o], color=color, lw=2, label=f"{o} ({args.label[0]})")
    for path, label in zip(args.log[1:], args.label[1:]):
        x, rates = outcome_rates(path, args.horizon, args.window, args.iterations)
        ax.plot(x, rates["goal"], color="tab:green", lw=1.2, ls="--", alpha=0.7, label=f"goal ({label})")
    ax.set_xlim(0, args.iterations)
    ax.set_ylim(0, 100)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(f"episodes ending in window [%]  (window = {args.window} it.)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
