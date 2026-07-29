#!/usr/bin/env bash
# Purpose: run one registered E2 protocol entirely as fresh closed-loop episodes.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.rollout_eval

set -euo pipefail

usage() {
  echo "Usage: eval_multi_seed_rollouts.sh --diffusion-policy=DIR [--bc-policy=DIR] [--protocol=initial-noise-sensitivity|full-ddpm-sampling] [--label=Closed-loop|Smoke] [--num-environment-seeds=N] [--num-policy-seeds=N] [--num-gpus=N] [--output-dir=DIR] [--overwrite] [--print-command]"
}

protocol="full-ddpm-sampling"
bc_policy=""
diffusion_policy=""
env_count=""
policy_count=""
num_gpus=1
train_seed=1000
episode_length=300
label="Closed-loop"
output_dir=""
overwrite=false
print_command=false
seen=""

for argument in "$@"; do
  key="${argument%%=*}"
  [[ " ${seen} " != *" ${key} "* ]] || {
    echo "error: duplicate ${key}" >&2
    exit 64
  }
  case "${argument}" in
    --protocol=*) protocol="${argument#*=}"; seen+=" ${key}" ;;
    --bc-policy=*) bc_policy="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-policy=*) diffusion_policy="${argument#*=}"; seen+=" ${key}" ;;
    --num-environment-seeds=*) env_count="${argument#*=}"; seen+=" ${key}" ;;
    --num-policy-seeds=*) policy_count="${argument#*=}"; seen+=" ${key}" ;;
    --num-gpus=*) num_gpus="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --episode-length=*) episode_length="${argument#*=}"; seen+=" ${key}" ;;
    --label=*) label="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done

[[ "${protocol}" == "initial-noise-sensitivity" || "${protocol}" == "full-ddpm-sampling" ]] || {
  echo "error: invalid --protocol=${protocol}" >&2
  exit 64
}
[[ "${label}" == "Closed-loop" || "${label}" == "Smoke" ]] || {
  echo "error: invalid --label=${label}" >&2
  exit 64
}
[[ -n "${diffusion_policy}" ]] || {
  echo "error: --diffusion-policy is required" >&2
  exit 64
}
if [[ "${protocol}" == "full-ddpm-sampling" && -z "${bc_policy}" ]]; then
  echo "error: --bc-policy is required for full-ddpm-sampling" >&2
  exit 64
fi
for value in "${train_seed}" "${episode_length}" "${num_gpus}" ${env_count:+"${env_count}"} ${policy_count:+"${policy_count}"}; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "error: numeric argument required" >&2
    exit 64
  }
done
(( episode_length >= 1 )) || {
  echo "error: episode length must be positive" >&2
  exit 64
}
(( num_gpus >= 1 )) || {
  echo "error: GPU count must be positive" >&2
  exit 64
}
[[ -z "${env_count}" || "${env_count}" -ge 1 ]] || {
  echo "error: environment count must be positive" >&2
  exit 64
}
[[ -z "${policy_count}" || "${policy_count}" -ge 1 ]] || {
  echo "error: policy count must be positive" >&2
  exit 64
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
if [[ -z "${output_dir}" ]]; then
  if [[ "${protocol}" == "initial-noise-sensitivity" ]]; then
    output_dir="outputs/pusht_diffusion/pilot/e2a_initial_noise_sensitivity"
  else
    output_dir="outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling"
  fi
fi
for variable in bc_policy diffusion_policy output_dir; do
  value="${!variable}"
  if [[ -n "${value}" && "${value}" != /* ]]; then
    printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"
  fi
done

command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.rollout_eval
  "--protocol=${protocol}"
  "--diffusion-policy=${diffusion_policy}"
  "--train-seed=${train_seed}"
  "--episode-length=${episode_length}"
  "--num-gpus=${num_gpus}"
  "--label=${label}"
  "--output-dir=${output_dir}"
)
[[ -n "${bc_policy}" ]] && command+=("--bc-policy=${bc_policy}")
[[ -n "${env_count}" ]] && command+=("--num-environment-seeds=${env_count}")
[[ -n "${policy_count}" ]] && command+=("--num-policy-seeds=${policy_count}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)

if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

echo "E2 protocol: ${protocol}"
echo "Evidence label: ${label}"
echo "GPU workers: ${num_gpus}"
echo "Output: ${output_dir}"
echo "Execution: environment-seed shards; fresh environment and policy reset for every sample; no resume"
exec "${command[@]}"
