#!/usr/bin/env bash
# Run E5 fixed-observation prototype-ID swaps with paired complete DDPM RNG.

set -euo pipefail

usage() {
  echo "Usage: eval_e5_fixed_observations.sh --bc-policy=DIR --diffusion-policy=DIR --prototype-manifest=FILE [--candidates=FILE] [--dataset-root=DIR] [--num-policy-seeds=N] [--num-inference-steps=N] [--label=Smoke|Offline] [--output-dir=DIR] [--overwrite] [--print-command]"
}

bc_policy=""; diffusion_policy=""; prototype_manifest=""
candidates="outputs/pusht_diffusion/data_inspection_e0_full/candidates.json"
dataset_root="datasets/lerobot/lerobot/pusht"
num_policy_seeds=20; num_inference_steps=100; label="Offline"
output_dir="outputs/pusht_diffusion/pilot/e5_mode_conditioned/fixed_observation"
overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-policy=*) bc_policy="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-policy=*) diffusion_policy="${argument#*=}"; seen+=" ${key}" ;;
    --prototype-manifest=*) prototype_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --candidates=*) candidates="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --num-policy-seeds=*) num_policy_seeds="${argument#*=}"; seen+=" ${key}" ;;
    --num-inference-steps=*) num_inference_steps="${argument#*=}"; seen+=" ${key}" ;;
    --label=*) label="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${bc_policy}" && -n "${diffusion_policy}" && -n "${prototype_manifest}" ]] || {
  echo "error: both policy paths and --prototype-manifest are required" >&2
  exit 64
}
for variable in num_policy_seeds num_inference_steps; do
  value="${!variable}"
  [[ "${value}" =~ ^[0-9]+$ ]] && ((value >= 1)) || {
    echo "error: --${variable//_/-} must be a positive integer" >&2
    exit 64
  }
done
[[ "${label}" == "Smoke" || "${label}" == "Offline" ]] || {
  echo "error: --label must be Smoke or Offline" >&2
  exit 64
}
if [[ "${label}" == "Offline" ]] && (
  ((num_policy_seeds != 20)) || ((num_inference_steps != 100))
); then
  echo "error: Offline E5 fixed-observation evaluation requires 20 seeds and DDPM100" >&2
  exit 64
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_policy diffusion_policy prototype_manifest candidates dataset_root output_dir; do
  value="${!variable}"
  [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done
command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.e5_fixed_observation
  "--bc-policy=${bc_policy}"
  "--diffusion-policy=${diffusion_policy}"
  "--prototype-manifest=${prototype_manifest}"
  "--candidates=${candidates}"
  "--dataset-root=${dataset_root}"
  "--num-policy-seeds=${num_policy_seeds}"
  "--num-inference-steps=${num_inference_steps}"
  "--label=${label}"
  "--output-dir=${output_dir}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
