#!/usr/bin/env bash
# Purpose: E2 independent same-initial-state BCChunk and multi-seed Diffusion rollouts.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.rollout_eval
# Required: --bc-policy=DIR and --diffusion-policy=DIR.
# Optional: --num-environment-seeds=N, --num-policy-seeds=N, --train-seed=N,
# --episode-length=N, --output-dir=DIR, --overwrite, --print-command.

set -euo pipefail
usage() { echo "Usage: eval_multi_seed_rollouts.sh --bc-policy=DIR --diffusion-policy=DIR [--num-environment-seeds=N] [--num-policy-seeds=N] [--output-dir=DIR] [--overwrite] [--print-command]"; }
bc_policy=""; diffusion_policy=""; env_count=10; policy_count=5; train_seed=1000; episode_length=300
output_dir="outputs/pusht_diffusion/pilot/e2_rollouts"; overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-policy=*) bc_policy="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-policy=*) diffusion_policy="${argument#*=}"; seen+=" ${key}" ;;
    --num-environment-seeds=*) env_count="${argument#*=}"; seen+=" ${key}" ;;
    --num-policy-seeds=*) policy_count="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --episode-length=*) episode_length="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${bc_policy}" && -n "${diffusion_policy}" ]] || { echo "error: both policy paths are required" >&2; exit 64; }
for value in "${env_count}" "${policy_count}" "${train_seed}" "${episode_length}"; do [[ "${value}" =~ ^[0-9]+$ ]] || { echo "error: numeric argument required" >&2; exit 64; }; done
(( env_count >= 1 && policy_count >= 1 && episode_length >= 1 )) || { echo "error: counts must be positive" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_policy diffusion_policy output_dir; do value="${!variable}"; [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"; done
command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.rollout_eval "--bc-policy=${bc_policy}" "--diffusion-policy=${diffusion_policy}" "--num-environment-seeds=${env_count}" "--num-policy-seeds=${policy_count}" "--train-seed=${train_seed}" "--episode-length=${episode_length}" "--output-dir=${output_dir}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
