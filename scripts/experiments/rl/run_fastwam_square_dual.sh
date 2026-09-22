#!/usr/bin/env bash
# Usage: bash scripts/experiments/rl/run_fastwam_square_dual.sh export|benchmark|train
# Defaults: native dual224, predict32/execute4, video9 stride4; fresh initialization; save every 4h.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
mode="${1:?Expected export, benchmark or train}"
shift
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS="${CPU_THREADS:-2}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
data="${DUAL_DATA:-datasets/derived/square_dual224}"
if [[ "$mode" == export ]]; then
  exec conda run --no-capture-output -n dppo python -u \
    -m robot_practice.experiments.rl.square_export_dual --output "$data" "$@"
fi
if [[ "$mode" != benchmark && "$mode" != train ]]; then
  echo "Expected export, benchmark or train" >&2
  exit 2
fi
out="${OUT:-outputs/fastwam_square_dual_${mode}_$(date +%Y%m%d_%H%M%S)}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "START mode=$mode output=$out protocol=32/4/9/4 fresh_pretrained=true"
exec conda run --no-capture-output -n fastwam-upstream \
  torchrun --standalone --nproc_per_node=8 \
  -m robot_practice.experiments.rl.fastwam_square_distributed \
  --mode "$mode" --output "$out" --dual-data "$data" \
  --microbatch "${MICROBATCH:-2}" --accumulation "${ACCUMULATION:-16}" \
  --workers 2 --checkpointing --warmup-updates 5 --measure-updates 20 \
  --hours "${TRAIN_HOURS:-12}" --save-hours 4 \
  --lr 1e-4 --min-lr 1e-6 --warmup-fraction 0.01 "$@"
