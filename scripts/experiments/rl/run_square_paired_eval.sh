#!/usr/bin/env bash
# One checkpoint per invocation; same random-reset seed list for both policies.
set -euo pipefail
cd "$(dirname "$0")/../../.."
policy="${1:?usage: bash $0 lightwam|dp CHECKPOINT OUTPUT}"
checkpoint="${2:?checkpoint required}"
output="${3:?new output directory required}"
case "$policy" in lightwam) policy_env=lightwam ;; dp) policy_env=dppo ;; *) exit 2 ;; esac
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "START policy=$policy gpu=$CUDA_VISIBLE_DEVICES episodes=80 control_steps=600 RESULT_DIR=$output"
exec conda run --no-capture-output -n "$policy_env" python -u \
  -m robot_practice.experiments.rl.square_paired_eval \
  --policy="$policy" --checkpoint="$checkpoint" --output="$output" \
  --episodes=80 --batch-size=8 --seed=10001
