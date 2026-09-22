#!/usr/bin/env bash
# From-scratch supervised DPPO baseline; does not download assets or run PPO.
# Usage: GPU=0 bash scripts/experiments/rl/run_dppo_square_bc_control.sh --smoke --envs=1
# Pilot: GPU=0 bash scripts/experiments/rl/run_dppo_square_bc_control.sh --stop-after=1000
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
output="${OUTPUT_ROOT:-outputs/dppo_square_bc_control_$(date +%Y%m%d_%H%M%S)_$$}"
exec conda run --no-capture-output -n dppo \
  python -u -m robot_practice.experiments.rl.dppo_square_bc_control \
  --output="$output" "$@"
