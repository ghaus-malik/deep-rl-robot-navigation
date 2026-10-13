"""Pick training episodes to film from an episode log written by ``EpisodeLogger``.

Chooses ``--per_outcome`` episodes for each outcome. There is only one camera, so the chosen episodes must not
overlap in time; rare outcomes are chosen first so they get priority. Strategies:

* ``latest``: the latest episodes in training (most trained policy).
* ``spread``: episodes spread evenly over the part of training where that outcome occurs, so the videos show
  how the behaviour changes during training.

    python scripts/select_episodes.py --log logs/episodes_seed42.csv --out logs/video_schedule.json
    python scripts/select_episodes.py --log logs/episodes_random_seed42.csv --out logs/schedule.json \\
        --per_outcome 6 --strategy spread
    # episodes left out because of overlaps -> a second schedule, filmed in a second run
    python scripts/select_episodes.py --log logs/episodes_seed42.csv --out logs/video_schedule_2.json \\
        --exclude logs/video_schedule.json --outcomes goal --per_outcome 1
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np


def overlaps(a: dict, b: dict) -> bool:
    # an episode occupies the camera for steps [start_step, end_step)
    return a["start_step"] < b["end_step"] and b["start_step"] < a["end_step"]


def _candidate_orders(candidates: list[dict], per_outcome: int, strategy: str) -> list[list[dict]]:
    """One preference-ordered candidate list per video slot."""
    if strategy == "latest":
        latest = sorted(candidates, key=lambda r: -r["end_step"])
        return [latest] * per_outcome
    ends = np.array([r["end_step"] for r in candidates])
    targets = np.linspace(ends.min(), ends.max(), per_outcome)
    return [sorted(candidates, key=lambda r: abs(r["end_step"] - t)) for t in targets]


def select(
    rows: list[dict],
    outcomes: list[str],
    per_outcome: int,
    horizon: int,
    exclude: list[dict] | None = None,
    strategy: str = "latest",
) -> list[dict]:
    """``exclude``: a previous schedule; its episodes are skipped and the numbering continues after it."""
    exclude = exclude or []
    taken = {(e["env_id"], e["start_step"]) for e in exclude}
    counts = {o: sum(r["outcome"] == o for r in rows) for o in outcomes}
    chosen: list[dict] = []
    for outcome in sorted(outcomes, key=lambda o: counts[o]):  # rarest first
        candidates = [r for r in rows if r["outcome"] == outcome and (r["env_id"], r["start_step"]) not in taken]
        offset = sum(e["expected_outcome"] == outcome for e in exclude)
        picked = []
        if candidates:
            for order in _candidate_orders(candidates, per_outcome, strategy):
                for r in order:
                    if r not in picked and not any(overlaps(r, c) for c in chosen):
                        picked.append(r)
                        chosen.append(r)
                        break
        if len(picked) < per_outcome:
            print(f"[WARN] only {len(picked)} non-overlapping '{outcome}' episodes available")
        for k, r in enumerate(sorted(picked, key=lambda r: r["end_step"]), start=1):
            end_iteration = (r["end_step"] - 1) // horizon + 1
            r["schedule"] = {
                "name": f"{outcome}_{offset + k}_iter{end_iteration:04d}",
                "env_id": r["env_id"],
                "start_step": r["start_step"],
                "end_step": r["end_step"],
                "length": r["length"],
                "start_iteration": r["start_step"] // horizon + 1,
                "end_iteration": end_iteration,
                "layout_idx": r.get("layout_idx"),
                "expected_outcome": outcome,
                "expected_return": r["return"],
            }
    return sorted((r["schedule"] for r in chosen), key=lambda c: c["start_step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per_outcome", type=int, default=2)
    parser.add_argument("--outcomes", default="goal,collision,timeout")
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--strategy", choices=("latest", "spread"), default="latest")
    parser.add_argument("--exclude", default=None, help="Earlier schedule whose episodes should not be picked again.")
    args = parser.parse_args()
    exclude = None
    if args.exclude:
        with open(args.exclude) as f:
            exclude = json.load(f)

    ints = ("env_id", "start_step", "end_step", "length", "layout_idx")
    with open(args.log) as f:
        rows = [
            {**r, **{k: int(r[k]) for k in ints if k in r}, "return": float(r["return"])}
            for r in csv.DictReader(f)
        ]
    schedule = select(rows, args.outcomes.split(","), args.per_outcome, args.horizon, exclude, args.strategy)
    with open(args.out, "w") as f:
        json.dump(schedule, f, indent=2)
    for s in schedule:
        print(f"{s['name']:30s} env {s['env_id']:2d}  steps {s['start_step']:7d}-{s['end_step']:7d} "
              f"(iterations {s['start_iteration']}-{s['end_iteration']}, {s['length']} steps)")


if __name__ == "__main__":
    main()
