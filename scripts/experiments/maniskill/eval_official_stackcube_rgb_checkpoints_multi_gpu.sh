#!/usr/bin/env bash
# Evaluate all numeric checkpoints in one official StackCube RGB run directory.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  eval_official_stackcube_rgb_checkpoints_multi_gpu.sh \
    --run-dir=DIR --output-root=DIR --gpus=0,1,2,3

Options:
  --run-dir=DIR             Directory containing checkpoints/*.pt
  --output-root=DIR         Root for per-checkpoint evaluation results
  --gpus=LIST               Physical GPU ids, comma-separated
  --official-repo=DIR       ManiSkill checkout; default is bundled offline copy
  --num-episodes=N          Default: 100
  --max-episode-steps=N     Default: 200
  --seed-start=N            Environment reset seed start; default: 3000
  --overwrite               Recompute existing checkpoint results
EOF
}

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
evaluator="$script_dir/eval_official_stackcube_rgb_multi_gpu.sh"
practice_root="$(cd -- "$script_dir/../../.." && pwd)"
run_dir=""
output_root=""
gpus=""
official_repo=""
num_episodes="100"
max_episode_steps="200"
seed_start="3000"
overwrite=false

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --run-dir=*) run_dir="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --output-root=*) output_root="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --gpus=*) gpus="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --official-repo=*) official_repo="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --num-episodes=*) num_episodes="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --max-episode-steps=*) max_episode_steps="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --seed-start=*) seed_start="$(printf '%s' "$1" | cut -d= -f2-)" ; shift ;;
    --overwrite) overwrite=true; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 64 ;;
  esac
done

[[ -n "$run_dir" && -n "$output_root" && -n "$gpus" ]] || {
  echo "error: --run-dir, --output-root, and --gpus are required" >&2
  exit 64
}
[[ -d "$run_dir/checkpoints" ]] || {
  echo "error: checkpoint directory does not exist: $run_dir/checkpoints" >&2
  exit 66
}
resolved_repo="$official_repo"
[[ -n "$resolved_repo" ]] || resolved_repo="$practice_root/third_party/maniskill_official"
mkdir -p "$output_root"

found=false
for checkpoint in "$run_dir"/checkpoints/*.pt; do
  [[ -f "$checkpoint" ]] || continue
  step="$(basename "$checkpoint" .pt)"
  [[ "$step" =~ ^[1-9][0-9]*$ ]] || continue
  found=true
  result_dir="$output_root/checkpoint_$step"
  if [[ -f "$result_dir/metrics.json" && "$overwrite" != true ]]; then
    echo "[official-eval] REUSE checkpoint=$step result=$result_dir"
    continue
  fi
  if [[ "$overwrite" == true ]]; then
    bash "$evaluator" \
      --official-repo="$resolved_repo" \
      --checkpoint="$checkpoint" \
      --output-dir="$result_dir" \
      --gpus="$gpus" \
      --num-episodes="$num_episodes" \
      --max-episode-steps="$max_episode_steps" \
      --seed-start="$seed_start" \
      --overwrite
  else
    bash "$evaluator" \
      --official-repo="$resolved_repo" \
      --checkpoint="$checkpoint" \
      --output-dir="$result_dir" \
      --gpus="$gpus" \
      --num-episodes="$num_episodes" \
      --max-episode-steps="$max_episode_steps" \
      --seed-start="$seed_start"
  fi
done

[[ "$found" == true ]] || {
  echo "error: no numeric checkpoint *.pt found under $run_dir/checkpoints" >&2
  exit 66
}
