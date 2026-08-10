#!/usr/bin/env bash
#
# Remote M0 visual checkpoint candidate rollout.
#
# This script only runs the frozen development rollouts and merges worker
# outputs into one metrics.json per candidate. It does not rank checkpoints or
# publish a winner; run select_checkpoint.sh afterward for that comparison.
# It must not be run locally.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  rollout_checkpoint_selection.sh [options]

Options:
  --run-dir=DIR              Default: outputs/maniskill/m0_probe/train_main
  --checkpoint-steps=LIST    Comma-separated steps, e.g. 010000,020000
  --selection-dir=DIR        Default: outputs/maniskill/m0_probe/checkpoint_selection
  --dataset-repo-id=ID       Default: local/maniskill_pickcube_rl_7d_full
  --dataset-root=DIR         Default: datasets/lerobot/maniskill/pickcube_rl_7d_full
  --split-file=FILE          Default: configs/experiments/maniskill/split.json
  --leakage-report=FILE      Default: outputs/maniskill/m0_probe/split_leakage_report.json
  --num-gpus=N               Default: 1; one evaluation worker per visible GPU
  --num-envs=N               Default: 200; full frozen reset batch per worker
  --max-episodes=N           Default: 200
  --max-steps=N              Default: 50
  --sim-backend=BACKEND      Default: physx_cuda
  --reuse-completed          Skip candidates whose aggregate metrics.json exists
  -h, --help                 Show this help.

Outputs:
  <selection-dir>/evaluations/checkpoint_<step>/development/metrics.json
  <selection-dir>/evaluations/checkpoint_<step>/development/episodes.parquet
  <selection-dir>/logs/checkpoint_<step>_gpu_<gpu>.log
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

run_dir="outputs/maniskill/m0_probe/train_main"
checkpoint_steps=""
selection_dir="outputs/maniskill/m0_probe/checkpoint_selection"
dataset_repo_id="local/maniskill_pickcube_rl_7d_full"
dataset_root="datasets/lerobot/maniskill/pickcube_rl_7d_full"
split_file="configs/experiments/maniskill/split.json"
leakage_report="outputs/maniskill/m0_probe/split_leakage_report.json"
num_gpus=1
num_envs=200
max_episodes=200
max_steps=50
sim_backend="physx_cuda"
reuse_completed=false

while (($# > 0)); do
  argument="$1"
  case "${argument}" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-dir=*) run_dir="${argument#*=}"; shift ;;
    --checkpoint-steps=*) checkpoint_steps="${argument#*=}"; shift ;;
    --selection-dir=*) selection_dir="${argument#*=}"; shift ;;
    --dataset-repo-id=*) dataset_repo_id="${argument#*=}"; shift ;;
    --dataset-root=*) dataset_root="${argument#*=}"; shift ;;
    --split-file=*) split_file="${argument#*=}"; shift ;;
    --leakage-report=*) leakage_report="${argument#*=}"; shift ;;
    --num-gpus=*) num_gpus="${argument#*=}"; shift ;;
    --num-envs=*) num_envs="${argument#*=}"; shift ;;
    --max-episodes=*) max_episodes="${argument#*=}"; shift ;;
    --max-steps=*) max_steps="${argument#*=}"; shift ;;
    --sim-backend=*) sim_backend="${argument#*=}"; shift ;;
    --reuse-completed)
      reuse_completed=true
      shift
      ;;
    *)
      echo "error: unknown argument: ${argument}" >&2
      echo "Run with --help for supported options." >&2
      exit 64
      ;;
  esac
done

if [[ -z "${checkpoint_steps}" ]]; then
  echo "error: --checkpoint-steps is required; do not silently choose a checkpoint set" >&2
  exit 64
fi

for value in "${num_gpus}" "${num_envs}" "${max_episodes}" "${max_steps}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: numeric arguments must be positive integers" >&2
    exit 64
  }
done

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
mkdir -p "${selection_dir}/logs"
selection_log="${selection_dir}/selection.log"

log_line() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${selection_log}"
}

visible_gpus="$({
  "${conda_bin}" run --no-capture-output -n robot-practice \
    python -c 'import torch; print(torch.cuda.device_count())'
} 2>>"${selection_log}")"
if [[ ! "${visible_gpus}" =~ ^[1-9][0-9]*$ ]] || ((visible_gpus < num_gpus)); then
  echo "error: requested ${num_gpus} GPU(s), but ${visible_gpus:-0} CUDA device(s) are visible; see ${selection_log}" >&2
  exit 69
fi

# Preserve a pre-existing CUDA_VISIBLE_DEVICES mask. Child processes receive
# one entry from that mask, so CUDA device 0 inside a worker means exactly one
# physical GPU even when the parent has a restricted mask.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#visible_gpu_ids[@]} < num_gpus )); then
    echo "error: CUDA_VISIBLE_DEVICES exposes fewer GPUs than --num-gpus" >&2
    exit 69
  fi
else
  visible_gpu_ids=()
  for ((gpu = 0; gpu < num_gpus; gpu++)); do
    visible_gpu_ids[gpu]="${gpu}"
  done
fi

IFS=',' read -r -a requested_steps <<< "${checkpoint_steps}"
for raw_step in "${requested_steps[@]}"; do
  [[ "${raw_step}" =~ ^[0-9]+$ ]] || {
    echo "error: invalid checkpoint step: ${raw_step}" >&2
    exit 64
  }
done

progress_total=${#requested_steps[@]}
progress_index=0
printf '[M0 checkpoint rollout] checkpoint 0/%d, workers 0/%d' "${progress_total}" "${num_gpus}"

for raw_step in "${requested_steps[@]}"; do
  step_number=$((10#${raw_step}))
  printf -v step_id '%06d' "${step_number}"
  policy_path="${run_dir}/checkpoints/${step_id}/pretrained_model"
  evaluation_dir="${selection_dir}/evaluations/checkpoint_${step_id}/development"
  workers_root="${evaluation_dir}/workers"
  [[ -f "${policy_path}/config.json" && -f "${policy_path}/model.safetensors" ]] || {
    printf '\nerror: incomplete checkpoint: %s\n' "${policy_path}" >&2
    exit 66
  }

  if [[ -f "${evaluation_dir}/metrics.json" ]]; then
    if [[ "${reuse_completed}" == true ]]; then
      log_line "reuse checkpoint=${step_id} metrics=${evaluation_dir}/metrics.json"
      progress_index=$((progress_index + 1))
      printf '\r[M0 checkpoint rollout] checkpoint %d/%d, workers %d/%d' \
        "${progress_index}" "${progress_total}" "${num_gpus}" "${num_gpus}"
      continue
    fi
    printf '\nerror: metrics already exists for checkpoint %s; use --reuse-completed or run the comparison script\n' \
      "${step_id}" >&2
    exit 73
  fi

  log_line "rollout checkpoint=${step_id} policy=${policy_path} num_gpus=${num_gpus}"
  mkdir -p "${workers_root}"
  pids=()
  worker_logs=()
  worker_outputs=()
  for ((worker = 0; worker < num_gpus; worker++)); do
    worker_gpu="${visible_gpu_ids[worker]}"
    worker_output="${workers_root}/gpu_${worker}"
    worker_log="${selection_dir}/logs/checkpoint_${step_id}_gpu_${worker}.log"
    worker_logs[worker]="${worker_log}"
    worker_outputs[worker]="${worker_output}"
    log_line "launch checkpoint=${step_id} worker=${worker} cuda=${worker_gpu} output=${worker_output}"
    CUDA_VISIBLE_DEVICES="${worker_gpu}" \
      bash "${script_dir}/eval_closed_loop.sh" \
        --system=visual \
        --policy-path="${policy_path}" \
        --task=PickCube-v1 \
        --control-mode=pd_ee_delta_pose \
        --split-file="${split_file}" \
        --seed-set=development \
        --sim-backend="${sim_backend}" \
        --num-envs="${num_envs}" \
        --max-episodes="${max_episodes}" \
        --max-steps="${max_steps}" \
        --policy-seed=2000 \
        --device=cuda \
        --leakage-report="${leakage_report}" \
        --dataset-repo-id="${dataset_repo_id}" \
        --dataset-root="${dataset_root}" \
        --worker-index="${worker}" \
        --worker-count="${num_gpus}" \
        --output-dir="${worker_outputs[worker]}" \
        > "${worker_logs[worker]}" 2>&1 &
    pids[worker]=$!
  done

  completed=0
  for ((worker = 0; worker < num_gpus; worker++)); do
    if wait "${pids[worker]}"; then
      completed=$((completed + 1))
      log_line "finished checkpoint=${step_id} worker=${worker}"
    else
      status=$?
      log_line "failed checkpoint=${step_id} worker=${worker} status=${status} log=${worker_logs[worker]}"
      for ((kill_worker = 0; kill_worker < num_gpus; kill_worker++)); do
        if [[ -n "${pids[kill_worker]:-}" ]] && kill -0 "${pids[kill_worker]}" 2>/dev/null; then
          kill "${pids[kill_worker]}" 2>/dev/null || true
        fi
      done
      printf '\nerror: checkpoint %s worker %s failed; inspect %s\n' \
        "${step_id}" "${worker}" "${worker_logs[worker]}" >&2
      exit "${status}"
    fi
    printf '\r[M0 checkpoint rollout] checkpoint %d/%d, workers %d/%d' \
      "$((progress_index + 1))" "${progress_total}" "${completed}" "${num_gpus}"
  done

  merge_log="${selection_dir}/logs/checkpoint_${step_id}_merge.log"
  log_line "merge checkpoint=${step_id} workers_root=${workers_root}"
  "${conda_bin}" run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.maniskill.merge_closed_loop_workers \
    --workers-root="${workers_root}" \
    --output-dir="${evaluation_dir}" \
    --expected-episodes="${max_episodes}" \
    --expected-workers="${num_gpus}" \
    > "${merge_log}" 2>&1 || {
      printf '\nerror: failed to merge checkpoint %s; inspect %s\n' "${step_id}" "${merge_log}" >&2
      exit 1
    }
  progress_index=$((progress_index + 1))
  printf '\r[M0 checkpoint rollout] checkpoint %d/%d, workers %d/%d' \
    "${progress_index}" "${progress_total}" "${num_gpus}" "${num_gpus}"
done

printf '\n[M0 checkpoint rollout] complete; run select_checkpoint.sh to compare candidates\n'
