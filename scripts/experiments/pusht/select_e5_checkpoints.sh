#!/usr/bin/env bash
# Rank both E5 model families on all 20 development episodes with one oracle mode each.

set -euo pipefail

usage() {
  echo "Usage: select_e5_checkpoints.sh [--bc-run=DIR] [--diffusion-run=DIR] [--prototype-manifest=FILE] [--dataset-root=DIR] [--num-gpus=N] [--train-seed=N] [--output-dir=DIR] [--overwrite] [--print-command]"
}

bc_run="outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_bc_chunk"
diffusion_run="outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_diffusion"
prototype_manifest="outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json"
dataset_root="datasets/lerobot/lerobot/pusht"
num_gpus=1
train_seed=1000
output_dir="outputs/pusht_diffusion/pilot/e5_oracle_checkpoint_selection"
overwrite=false
print_command=false
seen=""

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || {
    echo "error: duplicate ${key}" >&2
    exit 64
  }
  case "${argument}" in
    --bc-run=*) bc_run="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-run=*) diffusion_run="${argument#*=}"; seen+=" ${key}" ;;
    --prototype-manifest=*) prototype_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --num-gpus=*) num_gpus="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --num-gpus must be a positive integer" >&2
  exit 64
}
[[ "${train_seed}" =~ ^[0-9]+$ ]] || {
  echo "error: --train-seed must be non-negative" >&2
  exit 64
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_run diffusion_run prototype_manifest dataset_root output_dir; do
  value="${!variable}"
  [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.e5_checkpoint_selection
  "--bc-run=${bc_run}"
  "--diffusion-run=${diffusion_run}"
  "--prototype-manifest=${prototype_manifest}"
  "--dataset-root=${dataset_root}"
  "--num-gpus=${num_gpus}"
  "--train-seed=${train_seed}"
  "--output-dir=${output_dir}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
