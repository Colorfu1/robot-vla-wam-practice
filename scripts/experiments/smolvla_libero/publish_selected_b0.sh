#!/usr/bin/env bash
# Publish a completed B0 selection to checkpoints/selected without running any
# rollout or ranking. The selection JSON is validated before the source
# checkpoint is copied; the target is never overwritten without --overwrite.
# Entry point: python -m robot_practice.experiments.smolvla_libero.b0_checkpoint_selection --publish-only

set -euo pipefail

usage() {
  echo "Usage: publish_selected_b0.sh --run-dir=DIR --selection-file=FILE [--selected-dir=DIR]"
  echo "       [--overwrite] [--print-command]"
}

run_dir="outputs/smolvla_libero/pilot/b0_domain_adaptation_offline"
selection_file=""
selected_dir=""
overwrite=false
print_command=false
seen=""

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --run-dir=*) run_dir="${argument#*=}"; seen+=" ${key}" ;;
    --selection-file=*) selection_file="${argument#*=}"; seen+=" ${key}" ;;
    --selected-dir=*) selected_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ -n "${selection_file}" ]] || {
  echo "error: --selection-file is required" >&2
  exit 64
}
[[ -n "${run_dir}" ]] || { echo "error: --run-dir must not be empty" >&2; exit 64; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in run_dir selection_file selected_dir; do
  value="${!variable}"
  [[ -z "${value}" || "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done
if [[ -z "${selected_dir}" ]]; then
  selected_dir="${run_dir}/checkpoints/selected"
fi

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.smolvla_libero.b0_checkpoint_selection
  --publish-only
  "--run-dir=${run_dir}"
  "--selection-file=${selection_file}"
  "--selected-dir=${selected_dir}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
