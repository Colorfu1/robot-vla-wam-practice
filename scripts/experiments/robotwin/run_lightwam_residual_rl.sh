#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

phase=""
for argument in "$@"; do
  case "${argument}" in
    --phase=*) phase="${argument#*=}"; break ;;
    --phase) phase="next" ;;
  esac
done
if [[ "${phase}" == "next" ]]; then
  arguments=("$@")
  for index in "${!arguments[@]}"; do
    if [[ "${arguments[index]}" == "--phase" ]]; then
      next_index=$((index + 1))
      phase="${arguments[next_index]:-}"
      break
    fi
  done
fi
[[ -n "${phase}" && "${phase}" != "next" ]] || {
  echo "--phase is required" >&2
  exit 2
}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-python}"
exec "${python_bin}" -m robot_practice.experiments.rl.robotwin_residual "$@"

