#!/usr/bin/env bash
# Purpose: fit train-only E5 K-medoids prototypes and write sidecar assignments.

set -euo pipefail

usage() {
  echo "Usage: fit_e5_prototypes.sh [--trajectories=FILE] [--output-dir=DIR] [--split-file=FILE] [--num-prototypes=2|3|4] [--selection-rationale=TEXT] [--overwrite] [--print-command]"
}

trajectories="outputs/pusht_diffusion/data_inspection/trajectories.parquet"
output_dir="outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes"
split_file="configs/experiments/pusht_diffusion/split.json"
num_prototypes=""
selection_rationale=""
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
    --trajectories=*) trajectories="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --split-file=*) split_file="${argument#*=}"; seen+=" ${key}" ;;
    --num-prototypes=2|--num-prototypes=3|--num-prototypes=4)
      num_prototypes="${argument#*=}"; seen+=" ${key}" ;;
    --selection-rationale=*) selection_rationale="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${trajectories}" == /* ]] || trajectories="${practice_root}/${trajectories}"
[[ "${output_dir}" == /* ]] || output_dir="${practice_root}/${output_dir}"
[[ "${split_file}" == /* ]] || split_file="${practice_root}/${split_file}"

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.e5_prototypes
  "--trajectories=${trajectories}"
  "--output-dir=${output_dir}"
  "--split-file=${split_file}"
)
[[ -n "${num_prototypes}" ]] && command+=("--num-prototypes=${num_prototypes}")
[[ -n "${selection_rationale}" ]] && command+=("--selection-rationale=${selection_rationale}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)

if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
