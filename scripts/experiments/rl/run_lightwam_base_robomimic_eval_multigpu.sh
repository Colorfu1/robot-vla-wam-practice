#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${repo_root}"

gpus="0,1,2"
seeds="1,2,3"
episodes=50
max_episode_steps=400
config="configs/experiments/rl/lightwam_base_robomimic_square.yaml"
checkpoint=""
output_root="outputs/lightwam_base_robomimic_square/eval"
resume=0
forward_args=()

for arg in "$@"; do
  case "${arg}" in
    --gpus=*) gpus="${arg#*=}" ;;
    --seeds=*) seeds="${arg#*=}" ;;
    --episodes=*) episodes="${arg#*=}" ;;
    --max-episode-steps=*) max_episode_steps="${arg#*=}" ;;
    --config=*) config="${arg#*=}" ;;
    --checkpoint=*) checkpoint="${arg#*=}" ;;
    --output-root=*) output_root="${arg#*=}" ;;
    --resume) resume=1 ;;
    *) forward_args+=("${arg}") ;;
  esac
done

if [[ -z "${checkpoint}" ]]; then
  echo "--checkpoint is required" >&2
  exit 2
fi
if [[ ! -s "${checkpoint}" ]]; then
  echo "missing/empty Light-WAM checkpoint: ${checkpoint}" >&2
  exit 2
fi
if [[ ! "${episodes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--episodes must be a positive integer" >&2
  exit 2
fi
if [[ ! "${max_episode_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-episode-steps must be a positive integer" >&2
  exit 2
fi

IFS=',' read -r -a gpu_list <<< "${gpus}"
IFS=',' read -r -a seed_list <<< "${seeds}"
if ((${#gpu_list[@]} != ${#seed_list[@]})); then
  echo "--gpus and --seeds must contain the same number of entries" >&2
  exit 2
fi
if ((${#seed_list[@]} == 0)); then
  echo "at least one GPU/seed pair is required" >&2
  exit 2
fi

mkdir -p "${output_root}"
launcher="scripts/experiments/rl/run_lightwam_base_robomimic.sh"
pids=()
worker_labels=()
stop_workers() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap 'stop_workers; exit 130' HUP INT TERM

for index in "${!seed_list[@]}"; do
  gpu="${gpu_list[index]}"
  seed="${seed_list[index]}"
  worker_root="${output_root}/seed_${seed}"
  worker_log="${output_root}/seed_${seed}.launch.log"
  args=(
    --phase=eval
    "--config=${config}"
    "--checkpoint=${checkpoint}"
    "--eval-seeds=${seed}"
    "--episodes=${episodes}"
    "--max-episode-steps=${max_episode_steps}"
    "--output-root=${worker_root}"
  )
  if ((resume)); then
    args+=(--resume)
  fi
  args+=("${forward_args[@]}")
  CUDA_VISIBLE_DEVICES="${gpu}" bash "${launcher}" "${args[@]}" \
    > "${worker_log}" 2>&1 &
  pids+=("$!")
  worker_labels+=("seed=${seed} gpu=${gpu} log=${worker_log}")
  echo "STATUS phase=eval state=launched ${worker_labels[index]} pid=${pids[index]}" >&2
done

failures=0
completed=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    completed=$((completed + 1))
    echo "STATUS phase=eval state=done ${worker_labels[index]}" >&2
  else
    status=$?
    failures=$((failures + 1))
    echo "ERROR phase=eval state=failed status=${status} ${worker_labels[index]}" >&2
  fi
done
if ((failures)); then
  echo "DONE phase=eval completed=${completed} total=${#seed_list[@]} failures=${failures}" >&2
  exit 1
fi
trap - HUP INT TERM

bash "${launcher}" \
  --phase=eval-merge \
  "--config=${config}" \
  "--checkpoint=${checkpoint}" \
  "--eval-seeds=${seeds}" \
  "--episodes=${episodes}" \
  "--max-episode-steps=${max_episode_steps}" \
  "--output-root=${output_root}"

echo "DONE phase=eval completed=${#seed_list[@]} total=${#seed_list[@]} failures=0" >&2
