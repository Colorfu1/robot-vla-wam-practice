#!/usr/bin/env bash
# Offline-only: sync both pinned source checkouts from the local machine first.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
# These were re-downloaded from the official folder and matched the existing local files.
sha256sum --check <<'CHECKSUMS'
7c101e5ffcb1e094b3b2b44f088c92391119a9e9438395b632b3391bd474e400  datasets/raw/dppo/robomimic/square-img/train.npz
15eb8c018169a73dfaa8d108cbc692535215c65ed79a33bda29a3ca3ce25be62  datasets/raw/dppo/robomimic/square-img/normalization.npz
CHECKSUMS
for source_root in third_party/dppo_v06 third_party/dppo_official; do
  if [[ ! -d "$source_root/.git" ]]; then
    echo "Missing $source_root; rsync the local pinned checkout (including .git)." >&2
    exit 2
  fi
done
output="${OUTPUT_ROOT:-outputs/dppo_square_v06_$(date +%Y%m%d_%H%M%S)_$$}"
extra=()
epochs=(1000 2000 3000 4000)
if [[ "${1:-}" == --smoke ]]; then
  extra=(--smoke)
  epochs=(1)
elif [[ $# -ne 0 ]]; then
  echo 'Usage: run_dppo_square_v06.sh [--smoke]' >&2
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
  --source-version=v0.6 --root=third_party/dppo_v06 \
  --output="$output/train" "${extra[@]}"
for epoch in "${epochs[@]}"; do
  # Historical SL loader preferred EMA; raw weights are a separate final-checkpoint control.
  variants=(ema)
  if [[ "$epoch" == 4000 || "${1:-}" == --smoke ]]; then
    variants+=(model)
  fi
  for variant in "${variants[@]}"; do
    conda run --no-capture-output -n dppo python -u \
      -m robot_practice.experiments.rl.dppo_square_official eval \
      --root=third_party/dppo_official --weight-variant="$variant" \
      --output="$output/eval_epoch_${epoch}_${variant}" \
      --checkpoint="$output/train/checkpoint/state_$epoch.pt" "${extra[@]}"
  done
done
echo "DONE phase=all checkpoints=${#epochs[@]} final_model_control=true RESULT_DIR=$output"
