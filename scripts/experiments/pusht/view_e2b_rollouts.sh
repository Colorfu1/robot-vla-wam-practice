#!/usr/bin/env bash
# Purpose: load the complete E2b BC + 10x20 DP rollout set as selectable Recordings.
# Ordering: environment seed first, then model/policy seed, so identical scenes are adjacent.

set -euo pipefail

usage() {
  echo "Usage: view_e2b_rollouts.sh --recordings-dir=DIR [--port=9878] [--print-command]"
}

recordings_dir=""
port="9878"
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

[[ -n "${recordings_dir}" ]] || {
  echo "error: --recordings-dir must be non-empty" >&2
  exit 64
}
if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
  echo "error: --port must be an integer from 1 to 65535" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${recordings_dir}" == /* ]] || recordings_dir="${practice_root}/${recordings_dir}"
[[ -d "${recordings_dir}" ]] || {
  echo "error: recordings directory does not exist: ${recordings_dir}" >&2
  exit 66
}
bc_recordings_dir="$(dirname -- "${recordings_dir}")/bc_recordings"
[[ -d "${bc_recordings_dir}" ]] || {
  echo "error: BC recordings directory does not exist: ${bc_recordings_dir}" >&2
  exit 66
}

shopt -s nullglob
observed_seed_dirs=("${recordings_dir}"/policy_seed_*)
observed_rrds=("${recordings_dir}"/policy_seed_*/*.rrd)
observed_bc_rrds=("${bc_recordings_dir}"/env_*.rrd)
shopt -u nullglob
if ((${#observed_seed_dirs[@]} != 10)); then
  echo "error: expected exactly 10 policy_seed directories in ${recordings_dir}" >&2
  exit 66
fi
if ((${#observed_rrds[@]} != 200)); then
  echo "error: expected exactly 200 E2b RRD files in ${recordings_dir}" >&2
  exit 66
fi
if ((${#observed_bc_rrds[@]} != 20)); then
  echo "error: expected exactly 20 E2b BC RRD files in ${bc_recordings_dir}" >&2
  exit 66
fi

rrd_paths=()
for environment_seed in {2000..2019}; do
  bc_path="${bc_recordings_dir}/env_${environment_seed}.rrd"
  if [[ ! -s "${bc_path}" ]]; then
    echo "error: expected non-empty BC Recording: ${bc_path}" >&2
    exit 66
  fi
  rrd_paths+=("${bc_path}")
  for policy_seed in {00..09}; do
    path="${recordings_dir}/policy_seed_${policy_seed}/env_${environment_seed}.rrd"
    if [[ ! -s "${path}" ]]; then
      echo "error: expected non-empty Recording: ${path}" >&2
      exit 66
    fi
    rrd_paths+=("${path}")
  done
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
