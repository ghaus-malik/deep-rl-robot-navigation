#!/usr/bin/env bash
# User-level installation of Isaac Sim 4.5.0 + Isaac Lab v2.1.1 on the shared GPU server.
#
# Constraints of the shared server:
#   - no sudo, no system packages, no driver changes
#   - everything lives under $HOME (venv + Isaac Lab checkout)
#   - driver 570.207 / Python 3.10 / glibc 2.35 -> Isaac Sim 4.5 is the newest compatible release
#
# Usage (on the server):
#   setsid nohup bash scripts/install_isaaclab.sh > ~/drl_nav/logs/install.log 2>&1 &

set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/envs/isaaclab}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
ISAACSIM_VERSION="4.5.0"
ISAACLAB_TAG="v2.1.1"

# Keep the user-site packages (system torch/numpy of other projects) out of this env.
export PYTHONNOUSERSITE=1
# Never touch GPU 1 during the install (shader/extension warm-up can touch the GPU).
export CUDA_VISIBLE_DEVICES=0

echo "[$(date)] creating venv at $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    python3.10 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# PyTorch must be pinned BEFORE isaacsim: otherwise pip resolves "torch>=2.5.1" to the newest
# wheel built for CUDA 13, which requires a newer driver than 570.x.
# torch 2.7.0+cu128 is the version Isaac Lab v2.1.1 itself installs and CUDA 12.8 runs on driver 570.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
CONSTRAINTS="$VENV_DIR/constraints.txt"
printf "torch==2.7.0+cu128\ntorchvision==0.22.0+cu128\n" > "$CONSTRAINTS"

echo "[$(date)] installing torch 2.7.0+cu128"
pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 --index-url "$TORCH_INDEX"

echo "[$(date)] installing isaacsim==$ISAACSIM_VERSION (large download)"
# --no-cache-dir avoids keeping a second ~20 GB copy of the wheels in ~/.cache/pip.
pip install --no-cache-dir -c "$CONSTRAINTS" "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com --extra-index-url "$TORCH_INDEX"

echo "[$(date)] cloning Isaac Lab $ISAACLAB_TAG"
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone --branch "$ISAACLAB_TAG" --depth 1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

echo "[$(date)] installing Isaac Lab extensions + rl_games"
cd "$ISAACLAB_DIR"
# The constraints file keeps every later pip call from swapping the CUDA 12.8 torch build.
export PIP_CONSTRAINT="$CONSTRAINTS"
export PIP_EXTRA_INDEX_URL="$TORCH_INDEX"
# flatdict (an Isaac Lab dependency) only builds with a setuptools that still ships pkg_resources.
pip install --no-cache-dir "setuptools<81" wheel
pip install --no-cache-dir --no-build-isolation flatdict==4.0.1
./isaaclab.sh --install rl_games

# MP4 encoding without a system ffmpeg.
pip install --no-cache-dir imageio imageio-ffmpeg tensorboard pytest trimesh

echo "[$(date)] installed versions:"
pip list 2>/dev/null | grep -iE "^(isaacsim |isaaclab|rl.games|torch |numpy |gymnasium|imageio)" || true
echo "[$(date)] INSTALL DONE"
