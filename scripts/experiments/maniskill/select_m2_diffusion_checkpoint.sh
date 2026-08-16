#!/usr/bin/env bash
#
# M2 native LeRobot Diffusion Policy checkpoint selection.
#
# Each checkpoint is evaluated on the fixed development state bank by
# eval_stackcube_gpu, then the best checkpoint is copied under
# <selection-dir>/selected/pretrained_model.  The test state bank is not used.
#
# Example:
#   bash scripts/experiments/maniskill/select_m2_diffusion_checkpoint.sh \
#     --run-dir=outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000 \
#     --checkpoint-steps=10000,20000,30000,40000,50000 \
#     --num-gpus=8 --num-envs=64

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.select_m2_diffusion_checkpoint "$@"
