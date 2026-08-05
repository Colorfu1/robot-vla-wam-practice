#!/usr/bin/env bash
# Rank RouteField-BCChunk / RouteField-Diffusion checkpoints on development episodes
# with each episode's exact macro route as the commanded condition.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.route_field_checkpoint_selection
# At least one of --bc-run / --diffusion-run is required; the other may be omitted.

set -euo pipefail

usage() {
  echo "Usage: select_route_field_checkpoints.sh [--bc-run=DIR] [--diffusion-run=DIR] --route-manifest=FILE [--dataset-root=DIR] [--train-seed=N] [--num-gpus=N] [--device=DEVICE] [--output-dir=DIR] [--overwrite] [--print-command]"
}

bc_run=""; diffusion_run=""; route_manifest="outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json"
dataset_root="datasets/lerobot/lerobot/pusht"; train_seed=1000
num_gpus=1; device=""; output_dir="outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection"
overwrite=false; print_command=false; seen=""

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-run=*) bc_run="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-run=*) diffusion_run="${argument#*=}"; seen+=" ${key}" ;;
    --route-manifest=*) route_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --num-gpus=*) num_gpus="${argument#*=}"; seen+=" ${key}" ;;
    --device=*) device="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ -n "${bc_run}" || -n "${diffusion_run}" ]] || {
  echo "error: at least one of --bc-run / --diffusion-run is required" >&2
  exit 64
}
[[ -n "${route_manifest}" ]] || { echo "error: --route-manifest is required" >&2; exit 64; }
[[ "${train_seed}" =~ ^[0-9]+$ && "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: invalid numeric argument" >&2
  exit 64
}
if ((num_gpus > 1)) && [[ -n "${device}" ]]; then
  echo "error: --device cannot be set with --num-gpus>1" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_run diffusion_run route_manifest dataset_root output_dir; do
  value="${!variable}"
  [[ -z "${value}" || "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.route_field_checkpoint_selection
  "--route-manifest=${route_manifest}"
  "--dataset-root=${dataset_root}"
  "--train-seed=${train_seed}"
  "--num-gpus=${num_gpus}"
  "--output-dir=${output_dir}"
)
[[ -n "${bc_run}" ]] && command+=("--bc-run=${bc_run}")
[[ -n "${diffusion_run}" ]] && command+=("--diffusion-run=${diffusion_run}")
[[ -n "${device}" ]] && command+=("--device=${device}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
