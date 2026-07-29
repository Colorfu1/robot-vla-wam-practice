#!/usr/bin/env bash
# Purpose: open all scene-level E3 sampling-step comparison Recordings.
# Required inputs: none.
# Optional: --recordings-dir=DIR, --port=N, --print-command.

set -euo pipefail

usage() {
  echo "Usage: view_e3_rollouts.sh [--recordings-dir=DIR] [--port=N] [--print-command]"
}

recordings_dir="outputs/pusht_diffusion/pilot/e3_sampling_steps/recordings"
port=9878
print_command=false
seen=""

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || {
    echo "error: duplicate ${key}" >&2
    exit 64
  }
  case "${argument}" in
    --recordings-dir=*) recordings_dir="${argument#*=}"; seen+=" ${key}" ;;
    --port=*) port="${argument#*=}"; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ "${port}" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || {
  echo "error: --port must be an integer in [1,65535]" >&2
  exit 64
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
if [[ "${recordings_dir}" != /* ]]; then
  recordings_dir="${practice_root}/${recordings_dir}"
fi
[[ -d "${recordings_dir}" ]] || {
  echo "error: E3 recordings directory does not exist: ${recordings_dir}" >&2
  exit 66
}

shopt -s nullglob
rrd_paths=("${recordings_dir}"/env_*.rrd)
shopt -u nullglob
if ((${#rrd_paths[@]} == 0)); then
  echo "error: no E3 scene Recordings found in ${recordings_dir}" >&2
  exit 66
fi
if ((${#rrd_paths[@]} != 50)); then
  echo "error: expected exactly 50 E3 scene Recordings in ${recordings_dir}" >&2
  exit 66
fi
rrd_paths=()
for environment_seed in {2000..2049}; do
  path="${recordings_dir}/env_${environment_seed}.rrd"
  [[ -s "${path}" ]] || {
    echo "error: expected non-empty E3 Recording: ${path}" >&2
    exit 66
  }
  rrd_paths+=("${path}")
done

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
