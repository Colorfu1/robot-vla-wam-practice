#!/usr/bin/env bash
# Publish a completed C1/C2/C3 target checkpoint selection to checkpoints/selected
# without running any rollout or ranking. The selection JSON is validated before
# the source checkpoint is copied; supports both full-weight (model.safetensors)
# and PEFT/LoRA (adapter_model.safetensors) checkpoints.
# Entry point: python -m robot_practice.experiments.smolvla_libero.target_checkpoint_selection --publish-only

set -euo pipefail

usage() {
  echo "Usage: publish_selected_target.sh --run-dir=DIR --selection-file=FILE [--selected-dir=DIR]"
  echo "       [--overwrite] [--print-command]"
}

run_dir=""
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

[[ -n "${run_dir}" ]] || { echo "error: --run-dir is required" >&2; exit 64; }
[[ -n "${selection_file}" ]] || { echo "error: --selection-file is required" >&2; exit 64; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in run_dir selection_file selected_dir; do
  value="${!variable}"
  [[ -z "${value}" || "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.smolvla_libero.target_checkpoint_selection
  --publish-only
  "--run-dir=${run_dir}"
  "--selection-file=${selection_file}"
)
if [[ -n "${selected_dir}" ]]; then
  command+=("--selected-dir=${selected_dir}")
fi
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
