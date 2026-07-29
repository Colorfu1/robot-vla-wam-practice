#!/usr/bin/env bash
# Purpose: combine measured E0/E1/E2/E3 outputs and answer the six Pilot questions.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.report
# Required: --e0=DIR --e1=DIR --e2=DIR --e3=DIR.
# Optional: --output-dir=DIR, --overwrite, --print-command.
# Smoke-labeled inputs are rejected.

set -euo pipefail
usage() { echo "Usage: make_report.sh --e0=DIR --e1=DIR --e2=DIR --e3=DIR [--output-dir=DIR] [--overwrite] [--print-command]"; }
e0=""; e1=""; e2=""; e3=""; output_dir="outputs/pusht_diffusion/pilot/report"
overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --e0=*) e0="${argument#*=}"; seen+=" ${key}" ;;
    --e1=*) e1="${argument#*=}"; seen+=" ${key}" ;;
    --e2=*) e2="${argument#*=}"; seen+=" ${key}" ;;
    --e3=*) e3="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${e0}" && -n "${e1}" && -n "${e2}" && -n "${e3}" ]] || { echo "error: --e0/--e1/--e2/--e3 are required" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in e0 e1 e2 e3 output_dir; do value="${!variable}"; [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"; done
command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.report "--e0=${e0}" "--e1=${e1}" "--e2=${e2}" "--e3=${e3}" "--output-dir=${output_dir}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
