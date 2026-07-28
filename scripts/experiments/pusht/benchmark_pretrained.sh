#!/usr/bin/env bash
# Purpose: verify the public pretrained PushT policy and evaluator before local training.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.pretrained_benchmark
# Arguments: --policy=ID, --revision=REV, --n-episodes=N, --output-dir=DIR,
# --overwrite, --print-command.
# Result label: external pretrained reference — not split-comparable.

set -euo pipefail
usage() { echo "Usage: benchmark_pretrained.sh [--policy=ID] [--revision=REV] [--n-episodes=N] [--output-dir=DIR] [--overwrite] [--print-command]"; }
policy="lerobot/diffusion_pusht"
revision="84a7c23178445c6bbf7e1a884ff497017910f653"
n_episodes=10
output_dir="outputs/pusht_diffusion/benchmark_pretrained"
overwrite=false
print_command=false
seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --policy=*) policy="${argument#*=}"; seen+=" ${key}" ;;
    --revision=*) revision="${argument#*=}"; seen+=" ${key}" ;;
    --n-episodes=*) n_episodes="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; usage >&2; exit 64 ;;
  esac
done
[[ "${n_episodes}" =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid --n-episodes" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${output_dir}" == /* ]] || output_dir="${practice_root}/${output_dir}"
command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.pretrained_benchmark "--policy=${policy}" "--revision=${revision}" "--n-episodes=${n_episodes}" "--output-dir=${output_dir}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
