#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
checkpoint="${1:?usage: bash $0 CHECKPOINT EVAL_ROOT NEW_OUTPUT}"
eval_root="${2:?completed FastWAM multi-GPU evaluation directory required}"
output="${3:?new output directory required}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS="${CPU_THREADS:-4}"
exec conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.fastwam_square_diagnostic \
  --checkpoint="$checkpoint" --eval-root="$eval_root" --output="$output" \
  --trajectory-ids=0,10,20,30,40,50,60,70 --sample-seed=0 \
  --video-frames=0,40,80 --eval-seed=10001 --eval-chunk=10
