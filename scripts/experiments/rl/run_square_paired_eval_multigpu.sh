#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
policy="${1:?usage: bash $0 lightwam|dp|fastwam CHECKPOINT OUTPUT GPUS}"
checkpoint="${2:?checkpoint required}"
output="${3:?new output directory required}"
gpus="${4:?comma-separated GPU ids required}"
case "$policy" in
  lightwam) policy_env=lightwam ;;
  dp) policy_env=dppo ;;
  fastwam) policy_env=fastwam-upstream ;;
  *) exit 2 ;;
esac
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec conda run --no-capture-output -n "$policy_env" python -u \
  -m robot_practice.experiments.rl.square_paired_eval_multigpu \
  --policy="$policy" --checkpoint="$checkpoint" --output="$output" --gpus="$gpus" \
  --episodes=80 --batch-size=8 --seed=10001 --cpu-threads="${CPU_THREADS:-2}"
