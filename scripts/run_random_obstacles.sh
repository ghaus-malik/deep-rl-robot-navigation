#!/bin/bash
# Random-obstacle experiment on the GPU server (GPU 0 only), one stage at a time:
#
#   scripts/run_random_obstacles.sh check   # unit tests + Isaac Sim smoke tests (baseline and random layouts)
#   scripts/run_random_obstacles.sh train   # 1000 PPO iterations, new obstacles + goal every episode, episode log
#   scripts/run_random_obstacles.sh film    # re-run the same seeded training and film 6 goal / collision / timeout
#                                           # episodes spread over training -> videos/training/
#   scripts/run_random_obstacles.sh test    # unseen test layouts: metrics (512 episodes) + 5 full test videos
#                                           # -> results/, videos/testing/
set -e
cd "$(dirname "$0")/.."
source ~/envs/isaaclab/bin/activate
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0
mkdir -p logs results

SEED=42
ITERATIONS=1000
EXPERIMENT=random_obstacles_seed${SEED}
EPISODE_LOG=logs/episodes_${EXPERIMENT}.csv

case "$1" in
  check)
    python -m pytest -q tests
    python scripts/smoke_test.py --headless > logs/smoke_test.log 2>&1
    grep -E "checks passed" logs/smoke_test.log
    python scripts/smoke_test.py --headless --layout random > logs/smoke_test_random.log 2>&1
    grep -E "^\[(PASS|FAIL)\]|checks passed" logs/smoke_test_random.log
    ;;
  train)
    python scripts/train.py --headless --layout random --seed $SEED --max_iterations $ITERATIONS \
      --experiment $EXPERIMENT --episode_log $EPISODE_LOG > logs/train_${EXPERIMENT}.log 2>&1
    ;;
  film)
    python scripts/select_episodes.py --log $EPISODE_LOG --out logs/schedule_${EXPERIMENT}.json \
      --per_outcome 6 --strategy spread
    rm -rf runs_film
    python scripts/train.py --headless --layout random --seed $SEED --max_iterations $ITERATIONS \
      --experiment $EXPERIMENT --runs_dir runs_film --record_episodes logs/schedule_${EXPERIMENT}.json \
      --video_dir videos/training > logs/film_${EXPERIMENT}.log 2>&1
    rm -rf runs_film  # identical copy of the training run
    grep -E "\[VIDEO\].*outcome" logs/film_${EXPERIMENT}.log
    ;;
  test)
    CKPT=$(ls -t runs/${EXPERIMENT}/nn/last_*ep_${ITERATIONS}_*.pth | head -1)
    echo "checkpoint: $CKPT"
    python scripts/evaluate.py --headless --checkpoint "$CKPT" > logs/eval_${EXPERIMENT}.log 2>&1
    python scripts/evaluate.py --headless --checkpoint "$CKPT" --videos 5 --video_dir videos/testing \
      > logs/eval_videos_${EXPERIMENT}.log 2>&1
    grep -E "\[TEST\]" logs/eval_videos_${EXPERIMENT}.log
    ;;
  *)
    echo "usage: $0 {check|train|film|test}"; exit 1 ;;
esac
