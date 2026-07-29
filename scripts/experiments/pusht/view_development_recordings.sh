#!/usr/bin/env bash
# Purpose: open one complete model directory of PushT development task Recordings.
# Required: --recordings-dir=DIR.
# Optional: --port=9876, --print-command.
# Result: one Rerun Viewer containing env_1000 through env_1019 as selectable Recordings.

set -euo pipefail

usage() {
  echo "Usage: view_development_recordings.sh --recordings-dir=DIR [--port=PORT]"
  echo "       [--print-command]"
}

recordings_dir=""
port="9876"
print_command=false
seen=""
for argument in "$@"; do
  key="${argument%%=*}"
  if [[ " ${seen} " == *" ${key} "* ]]; then
    echo "error: duplicate ${key}" >&2
    exit 64
  fi
  case "${argument}" in
    --recordings-dir=*)
      recordings_dir="${argument#*=}"
      seen+=" ${key}"
      ;;
    --port=*)
      port="${argument#*=}"
      seen+=" ${key}"
      ;;
    --print-command)
      print_command=true
      seen+=" ${key}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: ${argument}" >&2
      exit 64
      ;;
  esac
done

if [[ -z "${recordings_dir}" ]]; then
  echo "error: --recordings-dir must be non-empty" >&2
  exit 64
fi
if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "error: --port must be an integer from 1 to 65535" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
if [[ "${recordings_dir}" != /* ]]; then
  recordings_dir="${practice_root}/${recordings_dir}"
fi
if [[ ! -d "${recordings_dir}" ]]; then
  echo "error: recordings directory does not exist: ${recordings_dir}" >&2
  exit 66
fi

shopt -s nullglob
observed_rrds=("${recordings_dir}"/*.rrd)
shopt -u nullglob
if ((${#observed_rrds[@]} != 20)); then
  echo "error: expected exactly 20 RRD files in ${recordings_dir}" >&2
  exit 66
fi

rrd_paths=()
for seed in {1000..1019}; do
  path="${recordings_dir}/env_${seed}.rrd"
  if [[ ! -s "${path}" ]]; then
    echo "error: expected non-empty Recording: ${path}" >&2
    exit 66
  fi
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
