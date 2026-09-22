#!/usr/bin/env bash
# Pinned official BC defaults; fresh training then official model-weight evaluation.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
revision=cc7234ad7ff39a8f32de3af903606723a16f0648
source_root="$PWD/third_party/dppo_official"
if [[ ! -e "$source_root" ]]; then
  git clone https://github.com/irom-princeton/dppo.git "$source_root"
  git -C "$source_root" checkout --detach "$revision"
fi
output="${OUTPUT_ROOT:-outputs/dppo_square_official_$(date +%Y%m%d_%H%M%S)_$$}"
extra=()
epochs=(500 1000 1500 2000)
if [[ "${1:-}" == --smoke ]]; then
  extra=(--smoke)
  epochs=(1)
elif [[ $# -ne 0 ]]; then
  echo 'Usage: run_dppo_square_official.sh [--smoke]' >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "Output exists; choose a fresh OUTPUT_ROOT: $output" >&2
  exit 2
fi
mkdir -p "$output"
echo "RESULT_DIR=$output"
conda run --no-capture-output -n dppo python -u \
  -m robot_practice.experiments.rl.dppo_square_official train \
  --root="$source_root" --output="$output/train" "${extra[@]}"
for epoch in "${epochs[@]}"; do
  conda run --no-capture-output -n dppo python -u \
    -m robot_practice.experiments.rl.dppo_square_official eval \
    --root="$source_root" --output="$output/eval_epoch_$epoch" \
    --checkpoint="$output/train/checkpoint/state_$epoch.pt" "${extra[@]}"
done
echo "DONE phase=all checkpoints=${#epochs[@]} RESULT_DIR=$output"
