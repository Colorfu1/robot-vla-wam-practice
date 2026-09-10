#!/usr/bin/env bash
# Run the official DPPO Square pixel fine-tuning baseline and its evaluation.
#
# This is an orchestration wrapper only: every worker calls the upstream
# /opt/robot-practice/dppo/script/run.py with the upstream configs. One worker
# occupies one selected GPU for one seed; all training finishes before eval starts.
#
# Usage (inside the DPPO worker container, from this repository):
#   bash scripts/experiments/rl/run_official_dppo_baseline.sh \
#     --gpus=0,1,2 --seeds=1,2,3
#   bash scripts/experiments/rl/run_official_dppo_baseline.sh --smoke --gpus=0 --seeds=1
#   bash scripts/experiments/rl/run_official_dppo_baseline.sh --resume \
#     --gpus=0,1,2 --seeds=1,2,3
set -euo pipefail

# Resolve the repository from this script's path, not from git metadata.  Some
# remote workers expose the same checkout through a git worktree whose
# `rev-parse --show-toplevel` can point at a different checkout mount; that
# would make all repository-relative data/output paths resolve incorrectly.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
dppo_root="${DPPO_ROOT:-/opt/robot-practice/dppo}"
data_root="${DPPO_DATA_DIR:-${repo_root}/datasets/raw/dppo}"
experiment_root="${DPPO_EXPERIMENT_ROOT:-${repo_root}/outputs/rl_residual_robomimic_20260901}"
output_root="${DPPO_OFFICIAL_ROOT:-}"
dppo_log_root="${DPPO_LOG_DIR:-}"
gpus_csv="${GPUS:-0,1,2}"
seeds_csv="${SEEDS:-1,2,3}"
base_checkpoint=""
normalization=""
output_root_explicit=0
dppo_log_root_explicit=0
smoke=0
resume=0

usage() {
  sed -n '1,45p' "$0"
}

while (($#)); do
  case "$1" in
    --gpus=*) gpus_csv="${1#*=}" ;;
    --seeds=*) seeds_csv="${1#*=}" ;;
    --dppo-root=*) dppo_root="${1#*=}" ;;
    --data-root=*) data_root="${1#*=}" ;;
    --experiment-root=*) experiment_root="${1#*=}" ;;
    --output-root=*) output_root="${1#*=}"; output_root_explicit=1 ;;
    --dppo-log-root=*) dppo_log_root="${1#*=}"; dppo_log_root_explicit=1 ;;
    --base-checkpoint=*) base_checkpoint="${1#*=}" ;;
    --normalization=*) normalization="${1#*=}" ;;
    --smoke) smoke=1 ;;
    --resume) resume=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ((output_root_explicit == 0)) && [[ -z "${output_root}" ]]; then
  if ((smoke)); then
    output_root="${experiment_root}/official_dppo_baseline_smoke"
  else
    output_root="${experiment_root}/official_dppo_baseline"
  fi
fi
if ((dppo_log_root_explicit == 0)) && [[ -z "${dppo_log_root}" ]]; then
  dppo_log_root="${experiment_root}/dppo_logs"
fi
if [[ -z "${base_checkpoint}" ]]; then
  base_checkpoint="${dppo_log_root}/robomimic-pretrain/square/"
  base_checkpoint+="square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/"
  base_checkpoint+="checkpoint/state_4000.pt"
fi
if [[ -z "${normalization}" ]]; then
  normalization="${data_root}/robomimic/square-img/normalization.npz"
fi

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${repo_root}" "$1" ;;
  esac
}

dppo_root="$(resolve_path "${dppo_root}")"
data_root="$(resolve_path "${data_root}")"
experiment_root="$(resolve_path "${experiment_root}")"
output_root="$(resolve_path "${output_root}")"
dppo_log_root="$(resolve_path "${dppo_log_root}")"
base_checkpoint="$(resolve_path "${base_checkpoint}")"
normalization="$(resolve_path "${normalization}")"

if ((smoke)); then
  train_iterations=1
  train_envs=1
  train_steps=100
  train_checkpoint="state_0.pt"
  eval_envs=1
  eval_steps=100
  episodes_per_seed=1
else
  train_iterations=301
  train_envs=50
  train_steps=400
  train_checkpoint="state_300.pt"
  eval_envs=50
  eval_steps=100
  episodes_per_seed=50
fi

IFS=',' read -r -a gpus <<<"${gpus_csv}"
IFS=',' read -r -a seeds <<<"${seeds_csv}"
if ((${#gpus[@]} == 0)) || [[ -z "${gpus[0]}" ]]; then
  echo "--gpus must select at least one explicit GPU" >&2
  exit 2
fi
if ((${#seeds[@]} == 0)) || [[ -z "${seeds[0]}" ]]; then
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

if [[ ! -d "${dppo_root}" || ! -f "${dppo_root}/script/run.py" ]]; then
  echo "missing upstream DPPO checkout: ${dppo_root}" >&2
  exit 3
fi
if [[ ! -f "${dppo_root}/cfg/robomimic/finetune/square/ft_ppo_diffusion_mlp_img.yaml" ]]; then
  echo "missing upstream fine-tuning config under ${dppo_root}" >&2
  exit 3
fi
if [[ ! -f "${dppo_root}/cfg/robomimic/eval/square/eval_diffusion_mlp_img.yaml" ]]; then
  echo "missing upstream evaluation config under ${dppo_root}" >&2
  exit 3
fi
test -s "${normalization}" || {
  echo "missing/empty normalization file: ${normalization}" >&2
  exit 3
}
test -s "${base_checkpoint}" || {
  echo "missing/empty base checkpoint: ${base_checkpoint}" >&2
  exit 3
}
if ((resume)); then
  if [[ ! -d "${output_root}" ]]; then
    echo "cannot resume: output root does not exist: ${output_root}" >&2
    exit 4
  fi
  if [[ ! -f "${repo_root}/scripts/experiments/rl/resume_official_dppo.py" ]]; then
    echo "missing repository resume shim: ${repo_root}/scripts/experiments/rl/resume_official_dppo.py" >&2
    exit 3
  fi
elif [[ -e "${output_root}" ]]; then
  echo "refusing to reuse existing output root: ${output_root}" >&2
  echo "use --resume to continue from the latest completed checkpoint" >&2
  exit 4
fi

finetune_root="${output_root}/finetune"
eval_root="${output_root}/eval"
metrics_path="${output_root}/metrics.json"
run_log="${output_root}/run.log"
mkdir -p "${finetune_root}" "${eval_root}"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "${run_log}"
}

latest_checkpoint() {
  local checkpoint_dir="$1"
  local best_itr=-1
  local best_path=""
  local path name itr
  shopt -s nullglob
  for path in "${checkpoint_dir}"/state_*.pt; do
    # A SIGTERM during torch.save can leave an empty destination behind.  Do
    # not mistake that partial artifact for the newest resumable checkpoint.
    [[ -s "${path}" ]] || continue
    name="${path##*/state_}"
    name="${name%.pt}"
    [[ "${name}" =~ ^[0-9]+$ ]] || continue
    itr=$((10#${name}))
    if ((itr > best_itr)); then
      best_itr="${itr}"
      best_path="${path}"
    fi
  done
  shopt -u nullglob
  if ((best_itr >= 0)); then
    printf '%d\t%s\n' "${best_itr}" "${best_path}"
  fi
}

latest_itr_from_log() {
  local log_path="$1"
  local start_bytes="${2:-0}"
  local contents
  [[ -s "${log_path}" ]] || return 0
  if ((start_bytes > 0)); then
    contents="$(tail -c "+$((start_bytes + 1))" "${log_path}" 2>/dev/null || true)"
  else
    contents="$(cat "${log_path}" 2>/dev/null || true)"
  fi
  grep -Eo '[0-9]+: step' <<<"${contents}" 2>/dev/null \
    | tail -n 1 | cut -d: -f1 || true
}

latest_rollout_from_log() {
  local log_path="$1"
  local start_bytes="${2:-0}"
  local contents
  [[ -s "${log_path}" ]] || return 0
  if ((start_bytes > 0)); then
    contents="$(tail -c "+$((start_bytes + 1))" "${log_path}" 2>/dev/null || true)"
  else
    contents="$(cat "${log_path}" 2>/dev/null || true)"
  fi
  grep -Eo 'Processed step [0-9]+ of [0-9]+' <<<"${contents}" 2>/dev/null \
    | tail -n 1 | sed -E 's/Processed step ([0-9]+) of ([0-9]+)/\1\/\2/' || true
}

pid_is_running() {
  local pid="$1"
  local state
  state="$(ps -o stat= -p "${pid}" 2>/dev/null | awk '{print $1}')"
  [[ -n "${state}" && "${state}" != Z* ]]
}

monitor_train_progress() {
  local seed="$1"
  local gpu="$2"
  local train_root="$3"
  local worker_pid="$4"
  local starting_completed="$5"
  local start_bytes="${6:-0}"
  local progress_log="${train_root}/run.log"
  local started_at now last_report_at elapsed latest_itr rollout completed
  local observed eta remaining percent
  local report_interval="${DPPO_PROGRESS_INTERVAL_SECONDS:-60}"

  started_at="$(date +%s)"
  last_report_at=0
  # ``kill -0`` remains true for a zombie until its parent reaps it.  The
  # worker is the parent and calls ``wait`` only after this monitor returns,
  # so also inspect the process state to avoid waiting forever after a fast
  # child (or a failed child) exits.
  while pid_is_running "${worker_pid}"; do
    now="$(date +%s)"
    latest_itr="$(latest_itr_from_log "${progress_log}" "${start_bytes}")"
    completed="${starting_completed}"
    if [[ "${latest_itr}" =~ ^[0-9]+$ ]]; then
      completed=$((10#${latest_itr} + 1))
      ((completed < starting_completed)) && completed="${starting_completed}"
      ((completed > train_iterations)) && completed="${train_iterations}"
    fi
    if ((completed > 0)); then
      percent=$((completed * 100 / train_iterations))
    else
      percent=0
    fi
    elapsed=$((now - started_at))
    observed=$((completed - starting_completed))
    remaining=$((train_iterations - completed))
    if ((now - last_report_at >= report_interval)); then
      if ((observed > 0 && elapsed > 0)); then
        eta=$((elapsed * remaining / observed))
        echo "STATUS phase=train seed=${seed} gpu=${gpu} itr=${completed}/${train_iterations} progress_pct=${percent} elapsed_s=${elapsed} eta_s=${eta} rollout_step=$(latest_rollout_from_log "${progress_log}" "${start_bytes}")"
      else
        echo "STATUS phase=train seed=${seed} gpu=${gpu} itr=${completed}/${train_iterations} progress_pct=${percent} elapsed_s=${elapsed} eta_s=unknown rollout_step=$(latest_rollout_from_log "${progress_log}" "${start_bytes}")"
      fi
      last_report_at="${now}"
    fi
    sleep "${DPPO_PROGRESS_POLL_SECONDS:-30}"
  done
}

active_pids=()
cleanup() {
  local pid
  for pid in "${active_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap 'cleanup; exit 130' INT TERM

run_train_seed() {
  local gpu="$1"
  local seed="$2"
  local train_root="${finetune_root}/seed_${seed}"
  local scheduler_overrides=()
  local train_overrides=()
  local resume_checkpoint=""
  local resume_itr=-1
  local starting_completed=0
  local worker_pid
  local worker_status=0
  local progress_start_bytes=0

  # The upstream config ties first_cycle_steps to train.n_train_itr and keeps
  # warmup_steps=10.  A one-iteration smoke run would therefore violate the
  # upstream scheduler's warmup_steps < first_cycle_steps assertion.  Keep the
  # upstream code/config untouched and disable warmup only for this smoke path.
  if ((smoke)); then
    scheduler_overrides+=(
      train.actor_lr_scheduler.warmup_steps=0
      train.critic_lr_scheduler.warmup_steps=0
    )
  fi

  mkdir -p "${train_root}"
  if ((resume)); then
    local checkpoint_info
    checkpoint_info="$(latest_checkpoint "${train_root}/checkpoint")"
    if [[ -n "${checkpoint_info}" ]]; then
      IFS=$'\t' read -r resume_itr resume_checkpoint <<<"${checkpoint_info}"
      starting_completed=$((resume_itr + 1))
      if ((resume_itr >= train_iterations - 1)); then
        echo "STATUS seed=${seed} gpu=${gpu} phase=train state=already_complete checkpoint=state_${resume_itr}.pt"
        return 0
      fi
      echo "STATUS seed=${seed} gpu=${gpu} phase=train state=resume_from checkpoint=state_${resume_itr}.pt next_itr=${starting_completed}/${train_iterations}"
    else
      echo "STATUS seed=${seed} gpu=${gpu} phase=train state=resume_no_checkpoint action=start_from_base"
    fi
  fi

  train_overrides=(
    "base_policy_path=${base_checkpoint}"
    "normalization_path=${normalization}"
    "logdir=${train_root}"
    "seed=${seed}"
    "device=cuda:0"
    "ft_denoising_steps=5"
    "env.n_envs=${train_envs}"
    "train.n_steps=${train_steps}"
    "train.n_train_itr=${train_iterations}"
    "${scheduler_overrides[@]}"
    "wandb=null"
  )

  if [[ -f "${train_root}/run.log" ]]; then
    progress_start_bytes="$(wc -c < "${train_root}/run.log")"
  fi
  echo "STATUS seed=${seed} gpu=${gpu} phase=train state=started"
  cd "${dppo_root}"
  if ((resume && resume_itr >= 0)); then
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy \
    PYTHONNOUSERSITE=1 DPPO_DATA_DIR="${data_root}" DPPO_LOG_DIR="${dppo_log_root}" \
    conda run --no-capture-output -n dppo \
      python "${repo_root}/scripts/experiments/rl/resume_official_dppo.py" \
        --config-name=ft_ppo_diffusion_mlp_img \
        --config-dir=cfg/robomimic/finetune/square \
        --resume-checkpoint="${resume_checkpoint}" -- \
        "${train_overrides[@]}" &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy \
    PYTHONNOUSERSITE=1 DPPO_DATA_DIR="${data_root}" DPPO_LOG_DIR="${dppo_log_root}" \
    conda run --no-capture-output -n dppo \
      python script/run.py \
        --config-name=ft_ppo_diffusion_mlp_img \
        --config-dir=cfg/robomimic/finetune/square \
        "${train_overrides[@]}" &
  fi
  worker_pid="$!"
  # The training command is backgrounded so the monitor can emit ETA.  Make
  # sure an interrupt reaches that child before this per-seed worker exits;
  # otherwise a killed wrapper could leave a training process behind.
  trap 'kill "${worker_pid:-}" 2>/dev/null || true; exit 130' INT TERM HUP
  monitor_train_progress "${seed}" "${gpu}" "${train_root}" "${worker_pid}" "${starting_completed}" "${progress_start_bytes}"
  wait "${worker_pid}" || worker_status="$?"
  trap - INT TERM HUP
  if ((worker_status != 0)); then
    return "${worker_status}"
  fi
  test -s "${train_root}/checkpoint/${train_checkpoint}"
  echo "STATUS seed=${seed} gpu=${gpu} phase=train state=done checkpoint=${train_checkpoint}"
}

run_eval_seed() {
  local gpu="$1"
  local seed="$2"
  local train_root="${finetune_root}/seed_${seed}"
  local seed_eval_root="${eval_root}/seed_${seed}"

  mkdir -p "${seed_eval_root}"
  echo "STATUS seed=${seed} gpu=${gpu} phase=eval state=started"
  test -s "${train_root}/checkpoint/${train_checkpoint}"
  cd "${dppo_root}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy \
  PYTHONNOUSERSITE=1 DPPO_DATA_DIR="${data_root}" DPPO_LOG_DIR="${dppo_log_root}" \
  conda run --no-capture-output -n dppo \
    python script/run.py \
      --config-name=eval_diffusion_mlp_img \
      --config-dir=cfg/robomimic/eval/square \
      base_policy_path="${train_root}/checkpoint/${train_checkpoint}" \
      normalization_path="${normalization}" \
      logdir="${seed_eval_root}" \
      seed="${seed}" device=cuda:0 \
      ft_denoising_steps=5 ddim_steps=5 \
      n_steps="${eval_steps}" env.n_envs="${eval_envs}" render_num=0
  test -s "${seed_eval_root}/result.npz"
  echo "STATUS seed=${seed} gpu=${gpu} phase=eval state=done episodes=${episodes_per_seed}"
  touch "${seed_eval_root}/DONE"
  echo "DONE seed=${seed} gpu=${gpu}"
}

run_phase() {
  local phase="$1"
  local worker="$2"
  local phase_root
  if [[ "${phase}" == train ]]; then
    phase_root="${finetune_root}"
  else
    phase_root="${eval_root}"
  fi
  local completed=0
  local failures=0
  local total="${#seeds[@]}"
  local gpu_count="${#gpus[@]}"

  log "START phase=${phase} seeds=${seeds_csv} gpus=${gpus_csv}"
  for ((batch_start = 0; batch_start < total; batch_start += gpu_count)); do
    active_pids=()
    batch_seeds=()
    batch_gpus=()
    for ((slot = 0; slot < gpu_count && batch_start + slot < total; slot++)); do
      index=$((batch_start + slot))
      seed="${seeds[$index]}"
      gpu="${gpus[$slot]}"
      seed_root="${phase_root}/seed_${seed}"
      seed_log="${seed_root}/run.log"
      mkdir -p "${seed_root}"
      "${worker}" "${gpu}" "${seed}" \
        > >(tee -a "${run_log}" "${seed_log}") 2>&1 &
      active_pids+=("$!")
      batch_seeds+=("${seed}")
      batch_gpus+=("${gpu}")
      log "STATUS phase=${phase} seed=${seed} gpu=${gpu} state=started"
    done

    for index in "${!active_pids[@]}"; do
      if wait "${active_pids[$index]}"; then
        completed=$((completed + 1))
        log "STATUS phase=${phase} seed=${batch_seeds[$index]} gpu=${batch_gpus[$index]} completed=${completed} total=${total}"
      else
        failures=$((failures + 1))
        log "ERROR phase=${phase} seed=${batch_seeds[$index]} gpu=${batch_gpus[$index]} worker_failed=true"
      fi
    done
    active_pids=()
    if ((failures > 0)); then
      log "DONE phase=${phase} completed=${completed} total=${total} failures=${failures}"
      return 1
    fi
  done
  log "DONE phase=${phase} completed=${completed} total=${total} failures=0"
}

log "START official DPPO fine-tuning + eval smoke=${smoke} resume=${resume} seeds=${seeds_csv} gpus=${gpus_csv}"
log "CONFIG train=ft_ppo_diffusion_mlp_img eval=eval_diffusion_mlp_img n_train_itr=${train_iterations} train_envs=${train_envs} train_steps=${train_steps} eval_episodes=${episodes_per_seed}"
run_phase train run_train_seed
run_phase eval run_eval_seed

PYTHONNOUSERSITE=1 conda run --no-capture-output -n dppo \
  python - "${eval_root}" "${metrics_path}" "${seeds_csv}" "${episodes_per_seed}" "${smoke}" "${resume}" <<'PY' \
  2>&1 | tee -a "${run_log}"
import json
import sys
from pathlib import Path

import numpy as np

eval_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])
expected_seeds = {int(value) for value in sys.argv[3].split(",") if value}
expected_episodes = int(sys.argv[4])
smoke = sys.argv[5] == "1"
resume = sys.argv[6] == "1"
rows = []
for result_path in sorted(eval_root.glob("seed_*/result.npz")):
    seed = int(result_path.parent.name.split("_")[-1])
    # A resumed run may intentionally request only a subset of seeds.  Ignore
    # stale eval artifacts for seeds outside this invocation's manifest.
    if seed not in expected_seeds:
        continue
    with np.load(result_path) as result:
        episodes = int(result["num_episode"])
        success_rate = float(result["eval_success_rate"])
        avg_episode_reward = float(result["eval_episode_reward"])
    rows.append({
        "seed": seed,
        "episodes": episodes,
        "success_rate": success_rate,
        "avg_episode_reward": avg_episode_reward,
    })

actual_seeds = {row["seed"] for row in rows}
if actual_seeds != expected_seeds or any(row["episodes"] != expected_episodes for row in rows):
    raise SystemExit(
        f"expected seeds={sorted(expected_seeds)} with {expected_episodes} episodes each, got {rows}"
    )
episodes = sum(row["episodes"] for row in rows)
successes = sum(row["episodes"] * row["success_rate"] for row in rows)
summary = {
    "schema_version": 1,
    "result_label": "Official DPPO pixel full-parameter fine-tuning",
    "phase": "official-dppo-baseline",
    "smoke": smoke,
    "resume": resume,
    "task_suite": "robomimic",
    "task": "square",
    "episodes": episodes,
    "successes": int(round(successes)),
    "success_rate": successes / episodes,
    "per_seed": rows,
    "seeds": sorted(expected_seeds),
    "ft_denoising_steps": 5,
    "base_checkpoint": "state_4000.pt",
    "finetuned_checkpoint": "state_0.pt" if smoke else "state_300.pt",
    "train_iterations": 1 if smoke else 301,
    "eval_episodes_per_seed": expected_episodes,
    "official_train_config": "cfg/robomimic/finetune/square/ft_ppo_diffusion_mlp_img.yaml",
    "official_eval_config": "cfg/robomimic/eval/square/eval_diffusion_mlp_img.yaml",
}
output_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

log "DONE official DPPO fine-tuning + eval smoke=${smoke} completed=${#seeds[@]} total=${#seeds[@]} failures=0 metrics=${metrics_path}"
