#!/usr/bin/env bash
# Purpose: generate paired E1a initial-noise-only and E1b full-sampling-seed results.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.fixed_observation_eval
# Required: --bc-policy=DIR and --diffusion-policy=DIR.
# Optional: --candidates=FILE, --dataset-root=DIR, --num-action-samples=20,
# --train-seed=N, --output-dir=DIR, --overwrite, --print-command.
# One invocation shares observations/noises and writes two protocol-specific RRD files.
# This launcher runs label=Offline, whose formal E1 protocol requires 20 samples per protocol.

set -euo pipefail
usage() { echo "Usage: eval_fixed_observations.sh --bc-policy=DIR --diffusion-policy=DIR [--candidates=FILE] [--dataset-root=DIR] [--num-action-samples=N] [--train-seed=N] [--output-dir=DIR] [--overwrite] [--print-command]"; }
bc_policy=""; diffusion_policy=""; candidates="outputs/pusht_diffusion/data_inspection/candidates.json"
dataset_root="datasets/lerobot/lerobot/pusht"; samples=20; train_seed=1000
output_dir="outputs/pusht_diffusion/pilot/e1_fixed_observation"; overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-policy=*) bc_policy="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-policy=*) diffusion_policy="${argument#*=}"; seen+=" ${key}" ;;
    --candidates=*) candidates="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --num-action-samples=*) samples="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${bc_policy}" && -n "${diffusion_policy}" ]] || { echo "error: both policy paths are required" >&2; exit 64; }
[[ "${samples}" =~ ^[0-9]+$ ]] || { echo "error: --num-action-samples must be numeric" >&2; exit 64; }
(( samples == 20 )) || { echo "error: formal E1 requires --num-action-samples=20" >&2; exit 64; }
[[ "${train_seed}" =~ ^[0-9]+$ ]] || { echo "error: --train-seed must be numeric" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_policy diffusion_policy candidates dataset_root output_dir; do
  value="${!variable}"; [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done
command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.fixed_observation_eval "--bc-policy=${bc_policy}" "--diffusion-policy=${diffusion_policy}" "--candidates=${candidates}" "--dataset-root=${dataset_root}" "--num-action-samples=${samples}" "--train-seed=${train_seed}" "--output-dir=${output_dir}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
