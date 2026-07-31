#!/usr/bin/env bash
# Evaluate P0/P1/P2/NULL for both selected E5 models on all 21 held-out episodes.

set -euo pipefail

usage() {
  echo "Usage: eval_e5_oracle_audit.sh [--selection-dir=DIR] [--prototype-manifest=FILE] [--dataset-root=DIR] [--num-gpus=N] [--device=DEVICE] [--open-loop-batch-size=N] [--output-dir=DIR] [--overwrite] [--print-command]"
}

selection_dir="outputs/pusht_diffusion/pilot/e5_oracle_checkpoint_selection"
prototype_manifest="outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json"
dataset_root="datasets/lerobot/lerobot/pusht"
num_gpus=1
device="cuda:0"
device_explicit=false
open_loop_batch_size=32
output_dir="outputs/pusht_diffusion/pilot/e5_oracle_audit"
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
    --selection-dir=*) selection_dir="${argument#*=}"; seen+=" ${key}" ;;
    --prototype-manifest=*) prototype_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --num-gpus=*) num_gpus="${argument#*=}"; seen+=" ${key}" ;;
    --device=*) device="${argument#*=}"; device_explicit=true; seen+=" ${key}" ;;
    --open-loop-batch-size=*) open_loop_batch_size="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ "${open_loop_batch_size}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --open-loop-batch-size must be a positive integer" >&2
  exit 64
}
[[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --num-gpus must be a positive integer" >&2
  exit 64
}
if ((num_gpus > 1)) && [[ "${device_explicit}" == true && "${device}" != "cuda:0" ]]; then
  echo "error: --device cannot be set with --num-gpus>1" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in selection_dir prototype_manifest dataset_root output_dir; do
  value="${!variable}"
  [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.e5_oracle_audit
  "--selection-dir=${selection_dir}"
  "--prototype-manifest=${prototype_manifest}"
  "--dataset-root=${dataset_root}"
  "--num-gpus=${num_gpus}"
  "--open-loop-batch-size=${open_loop_batch_size}"
  "--output-dir=${output_dir}"
)
[[ "${num_gpus}" == 1 ]] && command+=("--device=${device}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
