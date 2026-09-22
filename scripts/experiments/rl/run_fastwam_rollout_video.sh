#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
checkpoint="${1:?usage: bash $0 CHECKPOINT EVAL_ROOT NEW_OUTPUT}"
eval_root="${2:?evaluation root required}"
output="${3:?new output directory required}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
exec conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.fastwam_rollout_video \
  --checkpoint="$checkpoint" --eval-root="$eval_root" --output="$output" \
  --gpus="${GPUS:-0,1,2,3,4,5}"
