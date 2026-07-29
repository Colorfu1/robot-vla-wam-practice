#!/usr/bin/env bash
# Purpose: E0-lite qualitative inspection and held-out two-frame candidate export.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.data_inspection
# Arguments: --num-episodes=N, --num-candidates=N, --dataset-root=DIR,
# --output-dir=DIR, --overwrite, --print-command.
# Example: bash scripts/experiments/pusht/inspect_demonstrations.sh --num-episodes=206 --num-candidates=10

set -euo pipefail
usage() { echo "Usage: inspect_demonstrations.sh [--num-episodes=N] [--num-candidates=N] [--dataset-root=DIR] [--output-dir=DIR] [--overwrite] [--print-command]"; }
num_episodes=206; num_candidates=10; dataset_root="datasets/lerobot/lerobot/pusht"
output_dir="outputs/pusht_diffusion/data_inspection"; overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --num-episodes=*) num_episodes="${argument#*=}"; seen+=" ${key}" ;;
    --num-candidates=*) num_candidates="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ "${num_episodes}" =~ ^[1-9][0-9]*$ && "${num_candidates}" =~ ^[1-9][0-9]*$ ]] || { echo "error: counts must be positive" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${dataset_root}" == /* ]] || dataset_root="${practice_root}/${dataset_root}"
[[ "${output_dir}" == /* ]] || output_dir="${practice_root}/${output_dir}"
command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.data_inspection "--dataset-root=${dataset_root}" "--num-demonstrations=${num_episodes}" "--num-candidates=${num_candidates}" "--output-dir=${output_dir}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
