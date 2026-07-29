#!/usr/bin/env bash
# Run E5 prototype-controlled and NULL closed-loop rollouts on an explicit GPU count.

set -euo pipefail

usage() {
  echo "Usage: eval_e5_rollouts.sh --bc-policy=DIR --diffusion-policy=DIR --prototype-manifest=FILE --num-gpus=N [--num-environment-seeds=N] [--num-policy-seeds=N] [--episode-length=N] [--num-inference-steps=N] [--label=Smoke|Closed-loop] [--output-dir=DIR] [--overwrite] [--print-command]"
}

bc_policy=""; diffusion_policy=""; prototype_manifest=""; num_gpus=""
num_environment_seeds=20; num_policy_seeds=10; episode_length=300; num_inference_steps=100
label="Closed-loop"
output_dir="outputs/pusht_diffusion/pilot/e5_mode_conditioned/closed_loop"
overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-policy=*) bc_policy="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-policy=*) diffusion_policy="${argument#*=}"; seen+=" ${key}" ;;
    --prototype-manifest=*) prototype_manifest="${argument#*=}"; seen+=" ${key}" ;;
    --num-gpus=*) num_gpus="${argument#*=}"; seen+=" ${key}" ;;
    --num-environment-seeds=*) num_environment_seeds="${argument#*=}"; seen+=" ${key}" ;;
    --num-policy-seeds=*) num_policy_seeds="${argument#*=}"; seen+=" ${key}" ;;
    --episode-length=*) episode_length="${argument#*=}"; seen+=" ${key}" ;;
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
[[ "${num_gpus}" =~ ^[0-9]+$ ]] && ((num_gpus >= 1)) || {
  echo "error: --num-gpus must be an explicit positive integer" >&2
  exit 64
}
for variable in num_environment_seeds num_policy_seeds episode_length num_inference_steps; do
  value="${!variable}"
  [[ "${value}" =~ ^[0-9]+$ ]] && ((value >= 1)) || {
    echo "error: --${variable//_/-} must be a positive integer" >&2
    exit 64
  }
done
[[ "${label}" == "Smoke" || "${label}" == "Closed-loop" ]] || {
  echo "error: --label must be Smoke or Closed-loop" >&2
  exit 64
}
if [[ "${label}" == "Closed-loop" ]] && (
  ((num_environment_seeds != 20)) ||
  ((num_policy_seeds != 10)) ||
  ((episode_length != 300)) ||
  ((num_inference_steps != 100))
); then
  echo "error: Closed-loop E5 requires 20 environments, 10 policy seeds, 300 steps, and DDPM100" >&2
  exit 64
fi
((num_gpus <= num_environment_seeds)) || {
  echo "error: --num-gpus cannot exceed --num-environment-seeds" >&2
  exit 64
}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_policy diffusion_policy prototype_manifest output_dir; do
  value="${!variable}"
  [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
done
command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.e5_rollout_eval
  "--bc-policy=${bc_policy}"
  "--diffusion-policy=${diffusion_policy}"
  "--prototype-manifest=${prototype_manifest}"
  "--num-gpus=${num_gpus}"
  "--num-environment-seeds=${num_environment_seeds}"
  "--num-policy-seeds=${num_policy_seeds}"
  "--episode-length=${episode_length}"
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
