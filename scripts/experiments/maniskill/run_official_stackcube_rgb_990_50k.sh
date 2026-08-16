#!/usr/bin/env bash
# Official StackCube RGB Diffusion Policy: 990 demos, 50k iterations.
# This is a fixed-budget wrapper around run_official_stackcube_rgb_anchor.sh.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_official_stackcube_rgb_990_50k.sh [options]

Options:
  --phase=all|prepare|train|verify
  --official-repo=DIR       ManiSkill checkout; default is bundled offline copy
  --output-root=DIR         Default: outputs/maniskill/official_stackcube_rgb_990_50k
  --gpu=N                   Policy GPU, default 0; simulator remains physx_cpu
  --gpus=LIST               Optional DDP training GPUs, e.g. 4,5,6,7
  --model-device=cuda|cpu   Policy device, default cuda
  --capture-video           Save videos during final verification
EOF
}

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
runner="$script_dir/run_official_stackcube_rgb_anchor.sh"
phase="all"
official_repo=""
output_root="outputs/maniskill/official_stackcube_rgb_990_50k"
gpu="0"
gpus=""
model_device="cuda"
capture_video=false

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --phase=*) phase="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --official-repo=*) official_repo="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --output-root=*) output_root="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --gpu=*) gpu="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --gpus=*) gpus="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --model-device=*) model_device="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --capture-video) capture_video=true; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 64 ;;
  esac
done

case "$phase" in
  all|prepare|train|verify) ;;
  *) echo "error: invalid --phase: $phase" >&2; exit 64 ;;
esac

run_phase() {
  local requested_phase="$1"
  local -a args=(
    --phase="$requested_phase"
    --output-root="$output_root"
    --num-demos=990 --total-iters=50000
    --eval-freq=10000 --save-freq=10000
    --gpu="$gpu" --model-device="$model_device"
    --exp-name=official_stackcube_rgb_990_50k
  )
  [[ -n "$official_repo" ]] && args+=(--official-repo="$official_repo")
  [[ -n "$gpus" ]] && args+=(--gpus="$gpus")
  [[ "$capture_video" == true && "$requested_phase" == verify ]] && args+=(--capture-video)
  bash "$runner" "${args[@]}"
}

case "$phase" in
  prepare) run_phase prepare ;;
  train) run_phase train ;;
  verify) run_phase verify ;;
  all)
    run_phase prepare
    run_phase train
    run_phase verify
    ;;
esac
