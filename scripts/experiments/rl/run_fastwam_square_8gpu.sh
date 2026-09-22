#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
OUT="${OUT:-outputs/fastwam_square_8gpu_$(date +%Y%m%d_%H%M%S)}"
mkdir "$OUT"
echo "START output=$OUT gpus=$CUDA_VISIBLE_DEVICES phase=benchmark"
conda run --no-capture-output -n "${FASTWAM_ENV:-fastwam-upstream}" \
  torchrun --standalone --nproc_per_node=8 \
  -m robot_practice.experiments.rl.fastwam_square_distributed \
  --mode=benchmark --output="$OUT/benchmark" \
  --microbatch=4 --accumulation=8 \
  --warmup-updates=5 --measure-updates=20 "$@"
echo "START output=$OUT phase=train hours=${TRAIN_HOURS:-12} fresh_pretrained=true"
conda run --no-capture-output -n "${FASTWAM_ENV:-fastwam-upstream}" \
  torchrun --standalone --nproc_per_node=8 \
  -m robot_practice.experiments.rl.fastwam_square_distributed \
  --mode=train --output="$OUT/train" \
  --microbatch=4 --accumulation=8 \
  --hours="${TRAIN_HOURS:-12}" --save-hours=2 \
  --lr=1e-4 --min-lr=1e-6 --warmup-fraction=0.01 "$@"
echo "DONE phase=all completed=2 total=2 RESULT_DIR=$OUT"
