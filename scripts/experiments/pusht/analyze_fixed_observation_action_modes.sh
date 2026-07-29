#!/usr/bin/env bash
# Purpose: analyze fixed-observation E1 action chunks for exploratory modes.

set -euo pipefail

usage() {
  echo "Usage: analyze_fixed_observation_action_modes.sh [--input-dir=DIR] [--output-dir=DIR] [--overwrite] [--print-command]"
}

input_dir="outputs/pusht_diffusion/pilot/e1_fixed_observation"
output_dir="outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis"
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
    --input-dir=*) input_dir="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ -n "${input_dir}" ]] || {
  echo "error: --input-dir must be non-empty" >&2
  exit 64
}
[[ -n "${output_dir}" ]] || {
  echo "error: --output-dir must be non-empty" >&2
  exit 64
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${input_dir}" == /* ]] || input_dir="${practice_root}/${input_dir}"
[[ "${output_dir}" == /* ]] || output_dir="${practice_root}/${output_dir}"

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.fixed_observation_action_modes
  "--input-dir=${input_dir}"
  "--output-dir=${output_dir}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
