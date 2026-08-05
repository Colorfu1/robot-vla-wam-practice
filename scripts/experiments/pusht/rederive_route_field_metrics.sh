#!/usr/bin/env bash
# Re-derive corrected RF7 metrics from an existing held-out run without running
# any new rollout. Reads the source run read-only and writes a new directory.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.route_field_metrics_rederive

set -euo pipefail

usage() {
  echo "Usage: rederive_route_field_metrics.sh --source-dir=DIR [--output-dir=DIR] [--overwrite] [--print-command]"
}

source_dir="outputs/pusht_diffusion/pilot/route_field_held_out_eval/route_field_held_out_eval"
output_dir="outputs/pusht_diffusion/pilot/route_field_held_out_eval_corrected"
overwrite=false; print_command=false; seen=""

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --source-dir=*) source_dir="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ -n "${source_dir}" ]] || { echo "error: --source-dir is required" >&2; exit 64; }
[[ -n "${output_dir}" ]] || { echo "error: --output-dir must not be empty" >&2; exit 64; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in source_dir output_dir; do
  value="${!variable}"
  [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.route_field_metrics_rederive
  "--source-dir=${source_dir}"
  "--output-dir=${output_dir}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
