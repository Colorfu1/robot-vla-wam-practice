#!/usr/bin/env bash
# One-click offline RF7 route-condition ablation across every checkpoint in a run:
# fixed dataset middle-frame observation, fixed DDPM initial noise and reverse
# stream, only the commanded route changes. No rollout, no model selection.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.route_field_condition_ablation
# --num-gpus>1 distributes one checkpoint per GPU; --device is only allowed on a single GPU.
# At least one of --bc-run / --diffusion-run is required; the other may be omitted.

set -euo pipefail

usage() {
  echo "Usage: ablate_route_field_condition.sh [--bc-run=DIR] [--diffusion-run=DIR] --route-manifest=FILE [--dataset-root=DIR] [--split=development|audit] [--num-inference-steps=N] [--num-gpus=N] [--device=DEVICE] [--output-dir=DIR] [--overwrite] [--print-command]"
}

bc_run=""; diffusion_run=""; route_manifest="outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json"
dataset_root="datasets/lerobot/lerobot/pusht"; split="development"; num_inference_steps=100; num_gpus=1; device="cuda:0"; device_explicit=false
output_dir="outputs/pusht_diffusion/pilot/route_field_condition_ablation"
overwrite=false; print_command=false; seen=""

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-run=*) bc_run="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-run=*) diffusion_run="${argument#*=}"; seen+=" ${key}" ;;
    --route-manifest=*) route_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --split=*) split="${argument#*=}"; seen+=" ${key}" ;;
    --num-inference-steps=*) num_inference_steps="${argument#*=}"; seen+=" ${key}" ;;
    --num-gpus=*) num_gpus="${argument#*=}"; seen+=" ${key}" ;;
    --device=*) device="${argument#*=}"; device_explicit=true; seen+=" ${key}" ;;
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
[[ "${split}" == "development" || "${split}" == "audit" ]] || {
  echo "error: --split must be development or audit" >&2
  exit 64
}
[[ "${num_inference_steps}" =~ ^[0-9]+$ ]] && ((num_inference_steps >= 1)) || {
  echo "error: --num-inference-steps must be a positive integer" >&2
  exit 64
}
[[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --num-gpus must be a positive integer" >&2
  exit 64
}
if ((num_gpus > 1)) && [[ "${device_explicit}" == true ]]; then
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
  python -m robot_practice.experiments.pusht_diffusion.route_field_condition_ablation
  "--route-manifest=${route_manifest}"
  "--dataset-root=${dataset_root}"
  "--split=${split}"
  "--num-inference-steps=${num_inference_steps}"
  "--num-gpus=${num_gpus}"
  "--output-dir=${output_dir}"
)
[[ -n "${bc_run}" ]] && command+=("--bc-run=${bc_run}")
[[ -n "${diffusion_run}" ]] && command+=("--diffusion-run=${diffusion_run}")
[[ "${num_gpus}" == 1 ]] && command+=("--device=${device}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
