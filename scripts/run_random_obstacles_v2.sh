#!/bin/bash
# Random-obstacle experiment with the stability fixes (GPU server, GPU 0 only), one stage at a time:
#
#   - configs/ppo_random_obstacles.yaml: linear learning-rate decay, stronger action bound loss, gamma 0.995
#   - exponential (softer) proximity penalty
#   - curriculum: fixed baseline scene -> random layouts, share of random layouts 0 -> 100% over 300 iterations
#
#   scripts/run_random_obstacles_v2.sh train   # 1000 PPO iterations + episode log
#   scripts/run_random_obstacles_v2.sh film    # re-run the same seeded training, film 6 goal / collision / timeout
#                                              # episodes spread over training -> videos/v2/training/
#   scripts/run_random_obstacles_v2.sh test    # unseen test layouts, final checkpoint: metrics (512 episodes)
#                                              # + 5 full test videos -> results/, videos/v2/testing/
set -e
cd "$(dirname "$0")/.."
source ~/envs/isaaclab/bin/activate
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0
mkdir -p logs results

SEED=42
ITERATIONS=1000
EXPERIMENT=random_obstacles_v2_seed${SEED}
EPISODE_LOG=logs/episodes_${EXPERIMENT}.csv
TRAIN_ARGS="--headless --layout random --seed $SEED --max_iterations $ITERATIONS --experiment $EXPERIMENT \
  --config configs/ppo_random_obstacles.yaml --proximity exponential --curriculum_iterations 300"

case "$1" in
  train)
    python scripts/train.py $TRAIN_ARGS --episode_log $EPISODE_LOG > logs/train_${EXPERIMENT}.log 2>&1
    ;;
  film)
    python scripts/select_episodes.py --log $EPISODE_LOG --out logs/schedule_${EXPERIMENT}.json \
      --per_outcome 6 --strategy spread
    rm -rf runs_film_v2
    python scripts/train.py $TRAIN_ARGS --runs_dir runs_film_v2 --record_episodes logs/schedule_${EXPERIMENT}.json \
      --video_dir videos/v2/training > logs/film_${EXPERIMENT}.log 2>&1
    rm -rf runs_film_v2  # identical copy of the training run
    grep -E "\[VIDEO\].*outcome" logs/film_${EXPERIMENT}.log
    ;;
  test)
    CKPT=$(ls -t runs/${EXPERIMENT}/nn/last_*ep_${ITERATIONS}_*.pth | head -1)
    echo "final checkpoint: $CKPT"
    python scripts/evaluate.py --headless --checkpoint "$CKPT" --config configs/ppo_random_obstacles.yaml \
      --proximity exponential --output results/eval_${EXPERIMENT}_final.json > logs/eval_${EXPERIMENT}_final.log 2>&1
    python scripts/evaluate.py --headless --checkpoint "$CKPT" --config configs/ppo_random_obstacles.yaml \
      --proximity exponential --videos 5 --video_dir videos/v2/testing \
      --output results/eval_${EXPERIMENT}_final_videos.json > logs/eval_videos_${EXPERIMENT}_final.log 2>&1
    grep -E "\[TEST\]" logs/eval_videos_${EXPERIMENT}_final.log
    ;;
  *)
    echo "usage: $0 {train|film|test}"; exit 1 ;;
esac
