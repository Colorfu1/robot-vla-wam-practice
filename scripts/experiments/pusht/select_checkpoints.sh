#!/usr/bin/env bash
# Purpose: rank every numeric BCChunk/Diffusion checkpoint on development-only evidence.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.checkpoint_selection
# Required: --bc-run=DIR and --diffusion-run=DIR.
# Optional: --dataset-root=DIR, --num-development-seeds=20, --train-seed=1000,
# --output-dir=DIR, --overwrite, --print-command.

set -euo pipefail
usage() { echo "Usage: select_checkpoints.sh --bc-run=DIR --diffusion-run=DIR [--dataset-root=DIR] [--num-development-seeds=N] [--output-dir=DIR] [--overwrite] [--print-command]"; }
bc_run=""; diffusion_run=""; dataset_root="datasets/lerobot/lerobot/pusht"; seeds=20; train_seed=1000
output_dir="outputs/pusht_diffusion/pilot/checkpoint_selection"; overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-run=*) bc_run="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-run=*) diffusion_run="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --num-development-seeds=*) seeds="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${bc_run}" && -n "${diffusion_run}" ]] || { echo "error: both run roots are required" >&2; exit 64; }
[[ "${seeds}" =~ ^[1-9][0-9]*$ && "${train_seed}" =~ ^[0-9]+$ ]] || { echo "error: invalid numeric argument" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_run diffusion_run dataset_root output_dir; do value="${!variable}"; [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"; done
command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.checkpoint_selection "--bc-run=${bc_run}" "--diffusion-run=${diffusion_run}" "--dataset-root=${dataset_root}" "--num-development-seeds=${seeds}" "--train-seed=${train_seed}" "--output-dir=${output_dir}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
