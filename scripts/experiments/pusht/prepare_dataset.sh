#!/usr/bin/env bash
# Purpose: download/validate PushT, create smoke/pilot train-stat views, padding summaries,
# and paired BCChunk/Diffusion step-zero checkpoints.
# Entry points: dataset_views, padding_stats, and initialization Python modules.
# Arguments: --train-seed=N, --device=cpu|cuda, --no-download, --overwrite, --print-command.
# Example: bash scripts/experiments/pusht/prepare_dataset.sh
# Output: outputs/pusht_diffusion/{dataset_views,initialization}.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: prepare_dataset.sh [--train-seed=N] [--device=cpu|cuda] [--no-download] [--overwrite] [--print-command]
EOF
}

train_seed="1000"
device="cuda"
download=true
overwrite=false
print_command=false
seen=""
for argument in "$@"; do
  key="${argument%%=*}"
  if [[ " ${seen} " == *" ${key} "* ]]; then
    echo "error: duplicate argument: ${key}" >&2
    exit 64
  fi
  case "${argument}" in
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --device=cpu|--device=cuda) device="${argument#*=}"; seen+=" ${key}" ;;
    --no-download) download=false; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; usage >&2; exit 64 ;;
  esac
done
[[ "${train_seed}" =~ ^[0-9]+$ ]] || { echo "error: invalid --train-seed" >&2; exit 64; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
source_root="${practice_root}/datasets/lerobot/lerobot/pusht"
views_root="${practice_root}/outputs/pusht_diffusion/dataset_views"
split_file="${practice_root}/configs/experiments/pusht_diffusion/split.json"
view_command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.dataset_views
  "--source-root=${source_root}"
  "--views-root=${views_root}"
  "--split-file=${split_file}"
)
[[ "${download}" == true ]] && view_command+=(--download)
[[ "${overwrite}" == true ]] && view_command+=(--overwrite)

run_mode_commands() {
  local mode="$1"
  padding_command=(
    conda run --no-capture-output -n robot-practice
    python -m robot_practice.experiments.pusht_diffusion.padding_stats
    "--dataset-root=${views_root}/${mode}"
    "--mode=${mode}"
    "--split-file=${split_file}"
    "--output=${views_root}/${mode}/padding_stats.json"
  )
  initialization_command=(
    conda run --no-capture-output -n robot-practice
    python -m robot_practice.experiments.pusht_diffusion.initialization
    "--mode=${mode}"
    "--dataset-root=${views_root}/${mode}"
    "--output-dir=${practice_root}/outputs/pusht_diffusion/initialization/${mode}/seed_${train_seed}"
    "--split-file=${split_file}"
    "--train-seed=${train_seed}"
    "--device=${device}"
  )
  [[ "${overwrite}" == true ]] && initialization_command+=(--overwrite)
  if [[ "${print_command}" == true ]]; then
    printf '%q ' "${padding_command[@]}"
    printf '\n'
    printf '%q ' "${initialization_command[@]}"
    printf '\n'
  else
    "${padding_command[@]}"
    "${initialization_command[@]}"
  fi
}

if [[ "${print_command}" == true ]]; then
  printf '%q ' "${view_command[@]}"
  printf '\n'
  run_mode_commands smoke
  run_mode_commands pilot
  exit 0
fi
"${view_command[@]}"
run_mode_commands smoke
run_mode_commands pilot
