#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${CPU_THREADS:-4}"
OUT="${OUT:-outputs/fastwam_square_benchmark_$(date +%Y%m%d_%H%M%S)}"
echo "START benchmark gpu=$CUDA_VISIBLE_DEVICES output=$OUT"
conda run --no-capture-output -n "${FASTWAM_ENV:-fastwam-upstream}" \
  python -u -m robot_practice.experiments.rl.fastwam_square_benchmark \
  --output="$OUT" "$@"
echo "RESULT_DIR=$OUT"
