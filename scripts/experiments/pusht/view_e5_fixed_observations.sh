#!/usr/bin/env bash
# Open E5 fixed-observation recordings in prototype-ID order.

set -euo pipefail

usage() {
  echo "Usage: view_e5_fixed_observations.sh --recordings-dir=DIR [--port=9878] [--print-command]"
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
rrd_paths=()
for condition in P0 P1 P2 P3 NULL; do
  path="${recordings_dir}/condition_${condition}.rrd"
  [[ -s "${path}" ]] && rrd_paths+=("${path}")
done
((${#rrd_paths[@]} >= 3)) || {
  echo "error: expected prototype and NULL recordings in ${recordings_dir}" >&2
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
