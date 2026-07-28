#!/usr/bin/env bash
# Purpose: E4 paired DDIM/DDPM closed-loop success and synchronized chunk latency.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.sampling_steps_eval
# Required: --diffusion-policy=DIR.
# Optional: --schedulers=ddim,ddpm, --ddim-steps=5,10,20,50,100, --ddpm-steps=100,
# --n-episodes=50, --latency-iterations=1000, --latency-warmup-iterations=50,
# --videos-per-condition=5, --output-dir=DIR, --overwrite, --print-command.

set -euo pipefail
usage() { echo "Usage: benchmark_sampling_steps.sh --diffusion-policy=DIR [--schedulers=ddim,ddpm] [--ddim-steps=LIST] [--ddpm-steps=LIST] [--n-episodes=N] [--latency-iterations=N] [--output-dir=DIR] [--overwrite] [--print-command]"; }
policy=""; schedulers="ddim,ddpm"; ddim_steps="5,10,20,50,100"; ddpm_steps="100"; episodes=50
latency_iterations=1000; latency_warmup=50; videos=5; train_seed=1000; episode_length=300
output_dir="outputs/pusht_diffusion/pilot/e4_sampling_steps"; overwrite=false; print_command=false; seen=""
for argument in "$@"; do
  key="${argument%%=*}"; [[ " ${seen} " != *" ${key} "* ]] || { echo "error: duplicate ${key}" >&2; exit 64; }
  case "${argument}" in
    --diffusion-policy=*) policy="${argument#*=}"; seen+=" ${key}" ;;
    --schedulers=*) schedulers="${argument#*=}"; seen+=" ${key}" ;;
    --ddim-steps=*) ddim_steps="${argument#*=}"; seen+=" ${key}" ;;
    --ddpm-steps=*) ddpm_steps="${argument#*=}"; seen+=" ${key}" ;;
    --n-episodes=*) episodes="${argument#*=}"; seen+=" ${key}" ;;
    --latency-iterations=*) latency_iterations="${argument#*=}"; seen+=" ${key}" ;;
    --latency-warmup-iterations=*) latency_warmup="${argument#*=}"; seen+=" ${key}" ;;
    --videos-per-condition=*) videos="${argument#*=}"; seen+=" ${key}" ;;
    --train-seed=*) train_seed="${argument#*=}"; seen+=" ${key}" ;;
    --episode-length=*) episode_length="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; exit 64 ;;
  esac
done
[[ -n "${policy}" ]] || { echo "error: --diffusion-policy is required" >&2; exit 64; }
for value in "${episodes}" "${latency_iterations}" "${latency_warmup}" "${videos}" "${train_seed}" "${episode_length}"; do [[ "${value}" =~ ^[0-9]+$ ]] || { echo "error: numeric argument required" >&2; exit 64; }; done
(( episodes >= 1 && latency_iterations >= 1 && episode_length >= 1 )) || { echo "error: positive counts required" >&2; exit 64; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${policy}" == /* ]] || policy="${practice_root}/${policy}"; [[ "${output_dir}" == /* ]] || output_dir="${practice_root}/${output_dir}"
command=(conda run --no-capture-output -n robot-practice python -m robot_practice.experiments.pusht_diffusion.sampling_steps_eval "--diffusion-policy=${policy}" "--schedulers=${schedulers}" "--ddim-steps=${ddim_steps}" "--ddpm-steps=${ddpm_steps}" "--n-episodes=${episodes}" "--latency-warmup-iterations=${latency_warmup}" "--latency-iterations=${latency_iterations}" "--videos-per-condition=${videos}" "--train-seed=${train_seed}" "--episode-length=${episode_length}" "--output-dir=${output_dir}")
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
exec "${command[@]}"
