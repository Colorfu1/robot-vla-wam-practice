#!/usr/bin/env bash
# One-click RF7 held-out validation for selected RouteField checkpoints:
# ground-truth route rollouts + occupancy-matched reversed/permuted probes +
# the fixed-observation forward-vs-reversed gate.
# Use --gate-only to run just the offline gate before spending any rollout budget.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.route_field_eval

set -euo pipefail

usage() {
  echo "Usage: eval_route_field.sh [--selection-dir=DIR] [--route-manifest=FILE] [--dataset-root=DIR] [--num-gpus=N] [--device=DEVICE] [--split=audit|development] [--max-episodes=N] [--episode-length=N] [--num-inference-steps=N] [--fixed-observation-episodes=N] [--gate-only] [--output-dir=DIR] [--no-figures] [--overwrite] [--print-command]"
}

selection_dir="outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection"
route_manifest="outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json"
dataset_root="datasets/lerobot/lerobot/pusht"
num_gpus=1; device="cuda:0"; device_explicit=false
split="audit"; max_episodes=""; episode_length=300; num_inference_steps=100; fixed_observation_episodes=5
output_dir=""
overwrite=false; print_command=false; seen=""
no_figures=false; gate_only=false

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --selection-dir=*) selection_dir="${argument#*=}"; seen+=" ${key}" ;;
    --route-manifest=*) route_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --num-gpus=*) num_gpus="${argument#*=}"; seen+=" ${key}" ;;
    --device=*) device="${argument#*=}"; device_explicit=true; seen+=" ${key}" ;;
    --split=*) split="${argument#*=}"; seen+=" ${key}" ;;
    --max-episodes=*) max_episodes="${argument#*=}"; seen+=" ${key}" ;;
    --episode-length=*) episode_length="${argument#*=}"; seen+=" ${key}" ;;
    --num-inference-steps=*) num_inference_steps="${argument#*=}"; seen+=" ${key}" ;;
    --fixed-observation-episodes=*) fixed_observation_episodes="${argument#*=}"; seen+=" ${key}" ;;
    --gate-only) gate_only=true; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --no-figures) no_figures=true; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

# The gate and the full validation write different schemas, so they get separate
# default directories and can never silently overwrite each other.
if [[ -z "${output_dir}" ]]; then
  if [[ "${gate_only}" == true ]]; then
    output_dir="outputs/pusht_diffusion/pilot/route_field_condition_gate"
  else
    output_dir="outputs/pusht_diffusion/pilot/route_field_held_out_eval"
  fi
fi

[[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || { echo "error: --num-gpus must be a positive integer" >&2; exit 64; }
[[ "${split}" == "audit" || "${split}" == "development" ]] || {
  echo "error: --split must be audit or development" >&2
  exit 64
}
for variable in episode_length num_inference_steps; do
  value="${!variable}"
  [[ "${value}" =~ ^[0-9]+$ ]] && ((value >= 1)) || {
    echo "error: --${variable//_/-} must be a positive integer" >&2
    exit 64
  }
done
# 0 disables the fixed-observation gate; any other value caps the episodes used.
[[ "${fixed_observation_episodes}" =~ ^[0-9]+$ ]] || {
  echo "error: --fixed-observation-episodes must be a non-negative integer" >&2
  exit 64
}
if [[ "${gate_only}" == true ]]; then
  ((fixed_observation_episodes >= 1)) || {
    echo "error: --gate-only requires --fixed-observation-episodes>=1" >&2
    exit 64
  }
  ((num_gpus == 1)) || { echo "error: --gate-only requires --num-gpus=1" >&2; exit 64; }
fi
if [[ -n "${max_episodes}" ]]; then
  [[ "${max_episodes}" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: --max-episodes must be a positive integer" >&2
    exit 64
  }
fi
if ((num_gpus > 1)) && [[ "${device_explicit}" == true && "${device}" != "cuda:0" ]]; then
  echo "error: --device cannot be set with --num-gpus>1" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in selection_dir route_manifest dataset_root output_dir; do
  value="${!variable}"
  [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.route_field_eval
  "--selection-dir=${selection_dir}"
  "--route-manifest=${route_manifest}"
  "--dataset-root=${dataset_root}"
  "--num-gpus=${num_gpus}"
  "--split=${split}"
  "--episode-length=${episode_length}"
  "--num-inference-steps=${num_inference_steps}"
  "--fixed-observation-episodes=${fixed_observation_episodes}"
  "--output-dir=${output_dir}"
)
[[ -n "${max_episodes}" ]] && command+=("--max-episodes=${max_episodes}")
[[ "${num_gpus}" == 1 ]] && command+=("--device=${device}")
[[ "${gate_only}" == true ]] && command+=(--gate-only)
[[ "${no_figures}" == true ]] && command+=(--no-figures)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
