#!/usr/bin/env bash
# Purpose: create paired step-zero Mode-BCChunk and Mode-Diffusion checkpoints.

set -euo pipefail

usage() {
  echo "Usage: initialize_e5_models.sh --mode=smoke|pilot [--prototype-manifest=FILE] [--dataset-root=DIR] [--output-dir=DIR] [--split-file=FILE] [--train-seed=N] [--device=auto|cpu|cuda|mps] [--overwrite] [--print-command]"
}

mode=""
prototype_manifest="outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json"
dataset_root=""
output_dir=""
split_file="configs/experiments/pusht_diffusion/split.json"
train_seed=1000
device=auto
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
    --mode=smoke|--mode=pilot) mode="${argument#*=}"; seen+=" ${key}" ;;
    --prototype-manifest=*) prototype_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --split-file=*) split_file="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --device=auto|--device=cpu|--device=cuda|--device=mps)
      device="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ -n "${mode}" ]] || { echo "error: --mode is required" >&2; exit 64; }
[[ "${train_seed}" =~ ^[0-9]+$ ]] || { echo "error: --train-seed must be non-negative" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ -n "${dataset_root}" ]] || dataset_root="outputs/pusht_diffusion/dataset_views/${mode}"
[[ -n "${output_dir}" ]] || output_dir="outputs/pusht_diffusion/initialization/${mode}/seed_${train_seed}/e5_mode_conditioned"
[[ "${prototype_manifest}" == /* ]] || prototype_manifest="${practice_root}/${prototype_manifest}"
[[ "${dataset_root}" == /* ]] || dataset_root="${practice_root}/${dataset_root}"
[[ "${output_dir}" == /* ]] || output_dir="${practice_root}/${output_dir}"
[[ "${split_file}" == /* ]] || split_file="${practice_root}/${split_file}"

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.e5_initialization
  "--mode=${mode}"
  "--prototype-manifest=${prototype_manifest}"
  "--dataset-root=${dataset_root}"
  "--output-dir=${output_dir}"
  "--split-file=${split_file}"
  "--train-seed=${train_seed}"
  "--device=${device}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
