#!/usr/bin/env bash
# Open E5 recordings ordered by environment, condition, model, then policy seed.

set -euo pipefail

usage() {
  echo "Usage: view_e5_rollouts.sh --recordings-dir=DIR [--port=9878] [--print-command]"
}

recordings_dir=""; port=9878; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --recordings-dir=*) recordings_dir="${argument#*=}"; seen+=" ${key}" ;;
    --port=*) port="${argument#*=}"; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${recordings_dir}" ]] || { echo "error: --recordings-dir is required" >&2; exit 64; }
[[ "${port}" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || {
  echo "error: --port must be in [1,65535]" >&2
  exit 64
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${recordings_dir}" == /* ]] || recordings_dir="${practice_root}/${recordings_dir}"
[[ -d "${recordings_dir}" ]] || {
  echo "error: recordings directory does not exist: ${recordings_dir}" >&2
  exit 66
}
shopt -s nullglob
all_rrds=("${recordings_dir}"/mode_*/*/*.rrd "${recordings_dir}"/mode_*/*/*/*.rrd)
shopt -u nullglob
((${#all_rrds[@]} > 0)) || { echo "error: no E5 recordings found" >&2; exit 66; }

rrd_paths=()
for environment_seed in {2000..2019}; do
  for condition in P0 P1 P2 P3 NULL; do
    bc="${recordings_dir}/mode_bc_chunk/condition_${condition}/env_${environment_seed}.rrd"
    [[ -s "${bc}" ]] && rrd_paths+=("${bc}")
    for policy_seed in {00..09}; do
      dp="${recordings_dir}/mode_diffusion/condition_${condition}/policy_seed_${policy_seed}/env_${environment_seed}.rrd"
      [[ -s "${dp}" ]] && rrd_paths+=("${dp}")
    done
  done
done
((${#rrd_paths[@]} == ${#all_rrds[@]})) || {
  echo "error: one or more E5 recordings do not follow the registered layout" >&2
  exit 66
}
command=(
  conda run --no-capture-output -n robot-practice
  robot-rerun --persist-state --detach-process "--port=${port}"
  "${rrd_paths[@]}"
)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
