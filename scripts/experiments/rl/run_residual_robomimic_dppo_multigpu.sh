#!/usr/bin/env bash
# Schedule formal seeds across a user-selected set of single-GPU workers.
# The default mode runs directly inside the already-started DPPO container.
# Use --launch-mode=docker only from a Docker host that exposes a daemon.
# Usage: bash "$0" --stage=train --gpus=0,1,2 [--seeds=1,2,3]
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
stage=""
gpus_csv="0"
seeds_csv="1,2,3"
config="${repo_root}/configs/experiments/rl/residual_robomimic_dppo_square.yaml"
experiment_root="${repo_root}/outputs/rl_residual_robomimic_20260901"
launch_mode="${ROBOT_PRACTICE_DPPO_LAUNCH_MODE:-direct}"
direct_launcher="${repo_root}/scripts/experiments/rl/run_residual_robomimic_dppo.sh"
docker_launcher="${repo_root}/scripts/experiments/rl/run_residual_robomimic_dppo_docker.sh"

while (($#)); do
  case "$1" in
    --stage=*) stage="${1#*=}" ;;
    --gpus=*) gpus_csv="${1#*=}" ;;
    --seeds=*) seeds_csv="${1#*=}" ;;
    --config=*) config="${1#*=}" ;;
    --experiment-root=*) experiment_root="${1#*=}" ;;
    --launch-mode=*) launch_mode="${1#*=}" ;;
    -h|--help)
      sed -n '1,100p' "$0"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "${stage}" in
  base|timing|train|zero|learned|random) ;;
  *) echo "--stage must be one of: base, timing, train, zero, learned, random" >&2; exit 2 ;;
esac

case "${launch_mode}" in
  direct|docker) ;;
  *) echo "--launch-mode must be direct or docker" >&2; exit 2 ;;
esac

IFS=',' read -r -a gpus <<<"${gpus_csv}"
IFS=',' read -r -a seeds <<<"${seeds_csv}"
if ((${#gpus[@]} == 0)); then
  echo "--gpus must select at least one GPU" >&2
  exit 2
fi
if ((${#seeds[@]} == 0)); then
  echo "--seeds must select at least one seed" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ -z "${gpu}" || "${gpu}" == "all" || -n "${seen_gpus[$gpu]:-}" ]]; then
    echo "--gpus must contain unique, explicit GPU identifiers" >&2
    exit 2
  fi
  seen_gpus["${gpu}"]=1
done
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ || -n "${seen_seeds[$seed]:-}" ]]; then
    echo "--seeds must contain unique, non-negative integers" >&2
    exit 2
  fi
  seen_seeds["${seed}"]=1
done

case "${stage}" in
  base) stage_root="${experiment_root}/base" ;;
  timing) stage_root="${experiment_root}/timing" ;;
  train) stage_root="${experiment_root}/residual" ;;
  zero) stage_root="${experiment_root}/zero_eval" ;;
  learned) stage_root="${experiment_root}/residual_eval" ;;
  random) stage_root="${experiment_root}/random_residual" ;;
esac
mkdir -p "${stage_root}"
multigpu_log="${stage_root}/multigpu.log"

log_status() {
  local message="$1"
  printf '[%s] %s\n' "$(date -Iseconds)" "${message}" | tee -a "${multigpu_log}"
}

run_worker() {
  local gpu="$1"
  shift
  if [[ "${launch_mode}" == direct ]]; then
    # Each process sees only its assigned physical GPU. Inside the process it
    # is therefore CUDA device 0, while the caller can still use any host GPU ID.
    CUDA_VISIBLE_DEVICES="${gpu}" ROBOT_PRACTICE_DPPO_GPU="${gpu}" \
      bash "${direct_launcher}" "$@"
  else
    ROBOT_PRACTICE_DPPO_GPU="${gpu}" \
      bash "${docker_launcher}" "$@"
  fi
}

if [[ "${stage}" == zero || "${stage}" == learned || "${stage}" == random ]]; then
  manifest="${experiment_root}/base/episode_manifest.jsonl"
  if [[ ! -s "${manifest}" ]]; then
    echo "paired base manifest is missing: ${manifest}" >&2
    exit 3
  fi
fi

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap 'cleanup; exit 130' INT TERM

failures=0
completed=0
total=${#seeds[@]}
gpu_count=${#gpus[@]}
for ((batch_start = 0; batch_start < total; batch_start += gpu_count)); do
  pids=()
  batch_seeds=()
  batch_gpus=()
  for slot in "${!gpus[@]}"; do
    seed_index=$((batch_start + slot))
    if ((seed_index >= total)); then
      break
    fi
    seed="${seeds[$seed_index]}"
    gpu="${gpus[$slot]}"
    run_root="${stage_root}/seed_${seed}"
    mkdir -p "${run_root}"

    worker_args=("--config=${config}" "--output-root=${run_root}" "--seed=${seed}")
    case "${stage}" in
      base)
        worker_args+=(--phase=base-eval)
        ;;
      timing)
        worker_args+=(--phase=train --max-iterations=1 --time-stages)
        ;;
      train)
        worker_args+=(--phase=train)
        ;;
      zero)
        worker_args+=(--phase=eval --residual-mode=zero "--manifest=${manifest}")
        ;;
      learned)
        worker_args+=(
          --phase=eval --residual-mode=learned "--manifest=${manifest}"
          "--residual-root=${experiment_root}/residual/seed_${seed}"
        )
        ;;
      random)
        worker_args+=(--phase=eval --residual-mode=random-bounded "--manifest=${manifest}")
        ;;
    esac

    (
      run_worker "${gpu}" "${worker_args[@]}"
    ) > >(tee -a "${run_root}/launcher.log") 2>&1 &
    pids+=("$!")
    batch_seeds+=("${seed}")
    batch_gpus+=("${gpu}")
    log_status "STATUS stage=${stage} seed=${seed} gpu=${gpu} state=started"
  done

  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      completed=$((completed + 1))
      log_status "STATUS stage=${stage} seed=${batch_seeds[$index]} gpu=${batch_gpus[$index]} completed=${completed} total=${total}"
    else
      failures=$((failures + 1))
      log_status "ERROR stage=${stage} seed=${batch_seeds[$index]} gpu=${batch_gpus[$index]} worker_failed=true"
    fi
  done
  if ((failures)); then
    break
  fi
done
trap - INT TERM

if ((failures)); then
  log_status "DONE stage=${stage} completed=${completed} total=${total} failures=${failures}"
  exit 1
fi

if [[ "${stage}" == base ]]; then
  run_worker "${gpus[0]}" \
    --phase=base-eval-merge \
    "--config=${config}" \
    "--output-root=${stage_root}" 2>&1 | tee -a "${multigpu_log}"
fi

log_status "DONE stage=${stage} completed=${completed} total=${total} failures=0"
