#!/usr/bin/env bash
# Purpose: reduced E1 (three noises) plus two-environment/two-noise closed-loop checks.
# Entry points: fixed_observation_eval and rollout_eval Python modules.
# Required: --bc-policy=DIR and --diffusion-policy=DIR.
# Optional: --candidates=FILE, --dataset-root=DIR, --output-dir=DIR.
# Output is explicitly labeled Smoke and cannot be used by make_report.sh.

set -euo pipefail
usage() { echo "Usage: smoke_eval.sh --bc-policy=DIR --diffusion-policy=DIR [--candidates=FILE] [--dataset-root=DIR] [--output-dir=DIR] [--overwrite] [--print-command]"; }
bc_policy=""; diffusion_policy=""; output_dir="outputs/pusht_diffusion/smoke/evaluation"
candidates="outputs/pusht_diffusion/data_inspection/candidates.json"
dataset_root="datasets/lerobot/lerobot/pusht"
overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --bc-policy=*) bc_policy="${argument#*=}"; seen+=" ${key}" ;;
    --diffusion-policy=*) diffusion_policy="${argument#*=}"; seen+=" ${key}" ;;
    --candidates=*) candidates="${argument#*=}"; seen+=" ${key}" ;;
    --dataset-root=*) dataset_root="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${bc_policy}" && -n "${diffusion_policy}" ]] || { echo "error: both policy paths are required" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
for variable in bc_policy diffusion_policy candidates dataset_root output_dir; do value="${!variable}"; [[ "${value}" == /* ]] || printf -v "${variable}" '%s/%s' "${practice_root}" "${value}"; done
if [[ ! -f "${candidates}" ]]; then
  echo "error: Smoke Eval candidates file does not exist: ${candidates}" >&2
  echo "Generate it before running Smoke Eval:" >&2
  echo "  bash scripts/experiments/pusht/inspect_demonstrations.sh \\" >&2
  echo "    --num-episodes=50 \\" >&2
  echo "    --num-candidates=10 \\" >&2
  echo "    --output-dir=outputs/pusht_diffusion/data_inspection" >&2
  exit 66
fi
e1_command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.fixed_observation_eval "--bc-policy=${bc_policy}" "--diffusion-policy=${diffusion_policy}" "--candidates=${candidates}" "--dataset-root=${dataset_root}" "--num-action-samples=3" "--label=Smoke" "--output-dir=${output_dir}/e1_fixed_observation")
e2_command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.rollout_eval "--bc-policy=${bc_policy}" "--diffusion-policy=${diffusion_policy}" "--num-environment-seeds=2" "--num-policy-seeds=2" "--label=Smoke" "--output-dir=${output_dir}/e2_rollouts")
if [[ "${overwrite}" == true ]]; then e1_command+=(--overwrite); e2_command+=(--overwrite); fi
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${e1_command[@]}"; printf '\n'
  printf '%q ' "${e2_command[@]}"; printf '\n'
  exit 0
fi
"${e1_command[@]}"
"${e2_command[@]}"
