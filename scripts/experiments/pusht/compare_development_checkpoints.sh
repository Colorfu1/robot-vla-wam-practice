#!/usr/bin/env bash
# Purpose: compare the selected BCChunk and Diffusion checkpoints on all development resets.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.development_comparison
# Optional: --selection=FILE, --device=cuda:0, --output-dir=DIR, --overwrite,
# --print-command.
# Result: Closed-loop / Formal comparison plus recordings/{bc_chunk,diffusion}/env_*.rrd.

set -euo pipefail
usage() {
  echo "Usage: compare_development_checkpoints.sh [--selection=FILE] [--device=DEVICE]"
  echo "       [--output-dir=DIR] [--overwrite] [--print-command]"
}
selection="outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"
device="cuda:0"
output_dir="outputs/pusht_diffusion/pilot/development_comparison"
overwrite=false
print_command=false
seen=""
for argument in "$@"; do
  key="${argument%%=*}"
  if [[ " ${seen} " == *" ${key} "* ]]; then
    echo "error: duplicate ${key}" >&2
    exit 64
  fi
  case "${argument}" in
    --selection=*) selection="${argument#*=}"; seen+=" ${key}" ;;
    --device=*) device="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
if [[ -z "${selection}" || -z "${device}" || -z "${output_dir}" ]]; then
  echo "error: path and device arguments must be non-empty" >&2
  exit 64
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in selection output_dir; do
  value="${!variable}"
  if [[ "${value}" != /* ]]; then
    printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
  fi
done
command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.development_comparison
  "--selection=${selection}"
  "--device=${device}"
  "--output-dir=${output_dir}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
