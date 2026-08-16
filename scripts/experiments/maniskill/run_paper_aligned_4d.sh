#!/usr/bin/env bash
# End-to-end, resumable paper-aligned PickCube 4D baseline.
#
# The first stage prepares the CPU replay/data if needed.  The three requested
# stages then remain independent: train, official checkpoint selection, and
# fixed holdout test.  Existing final artifacts are reused; no run directory is
# deleted by this entry point.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_paper_aligned_4d.sh [options]

Options:
  --num-gpus=N          Training GPUs; default 8 (32 per GPU => global 256)
  --eval-device=DEV     Policy device during CPU-simulator eval; default cuda
  --eval-gpu=ID         One GPU used for policy inference; default 0 (legacy)
  --eval-gpus=LIST      GPUs assigned one-per-checkpoint, e.g. 0,1,2,3
  --eval-parallelism=N  Concurrent checkpoint evaluations; default 4
  --holdout-parallelism=N  Concurrent fixed-seed holdout workers; default 8
  --skip-data           Do not run replay/conversion; require the 4D dataset
  --skip-train          Do not train; require both train_1000/train_100 outputs
  --skip-select         Do not run official checkpoint selection
  --skip-test           Do not run holdout/reference tests
  -h, --help            Show this help

The default invocation runs every missing stage and is safe to rerun after a
failure.  It does not delete partial output directories.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
num_gpus=8
eval_device="cuda"
eval_gpu=0
eval_gpus_value=""
eval_parallelism=4
holdout_parallelism=8
skip_data=false
skip_train=false
skip_select=false
skip_test=false

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --num-gpus=*) num_gpus="${1#*=}"; shift ;;
    --eval-device=*) eval_device="${1#*=}"; shift ;;
    --eval-gpu=*) eval_gpu="${1#*=}"; shift ;;
    --eval-gpus=*) eval_gpus_value="${1#*=}"; shift ;;
    --eval-parallelism=*) eval_parallelism="${1#*=}"; shift ;;
    --holdout-parallelism=*) holdout_parallelism="${1#*=}"; shift ;;
    --skip-data) skip_data=true; shift ;;
    --skip-train) skip_train=true; shift ;;
    --skip-select) skip_select=true; shift ;;
    --skip-test) skip_test=true; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 64 ;;
  esac
done

[[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || { echo "error: --num-gpus must be positive" >&2; exit 64; }
[[ "${eval_gpu}" =~ ^[0-9]+$ ]] || { echo "error: --eval-gpu must be non-negative" >&2; exit 64; }
[[ "${eval_parallelism}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --eval-parallelism must be positive" >&2
  exit 64
}
[[ "${holdout_parallelism}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --holdout-parallelism must be positive" >&2
  exit 64
}
((holdout_parallelism <= 400)) || {
  echo "error: --holdout-parallelism must not exceed 400" >&2
  exit 64
}
eval_gpus=()
if [[ -n "${eval_gpus_value}" ]]; then
  IFS=',' read -r -a eval_gpus <<< "${eval_gpus_value}"
else
  eval_gpus=("${eval_gpu}")
fi
declare -A seen_eval_gpus=()
for gpu in "${eval_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "error: --eval-gpus must be a comma-separated list of non-negative GPU IDs" >&2
    exit 64
  }
  [[ -z "${seen_eval_gpus[${gpu}]+present}" ]] || {
    echo "error: --eval-gpus contains duplicate GPU ID: ${gpu}" >&2
    exit 64
  }
  seen_eval_gpus["${gpu}"]=1
done
if [[ "${eval_device}" == cuda* && "${eval_parallelism}" -gt "${#eval_gpus[@]}" ]]; then
  echo "error: CUDA evaluation parallelism (${eval_parallelism}) exceeds --eval-gpus (${#eval_gpus[@]})" >&2
  exit 64
fi
if [[ "${eval_device}" == cuda* && "${holdout_parallelism}" -gt "${#eval_gpus[@]}" ]]; then
  echo "error: CUDA holdout parallelism (${holdout_parallelism}) exceeds --eval-gpus (${#eval_gpus[@]})" >&2
  exit 64
fi

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi
cd -- "${practice_root}"

root="outputs/maniskill/paper_aligned_4d"
logs="${root}/logs"
mkdir -p "${logs}"
pipeline_log="${root}/pipeline.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${pipeline_log}"
}

run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  log "START ${label}; log=${log_file}"
  if "$@" >"${log_file}" 2>&1; then
    log "DONE ${label}"
  else
    local status=$?
    log "FAILED ${label} status=${status}; inspect ${log_file}"
    return "${status}"
  fi
}

replay_source="datasets/raw/maniskill_demos/PickCube-v1/motionplanning/trajectory.h5"
replay_h5="datasets/raw/maniskill_demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
dataset_root="datasets/lerobot/maniskill/pickcube_mp_4d"
dataset_repo_id="local/maniskill_pickcube_mp_4d"
audit_output="${root}/conversion_audit.json"
action_stats_output="${root}/action_channel_stats.json"
survival_report="${root}/replay_survival.json"
dataset_info="${dataset_root}/meta/info.json"
dataset_manifest="${dataset_root}/conversion_manifest.json"

if [[ "${skip_data}" == false || \
      ("${skip_test}" == false && ! -f "${root}/leakage_report.json") ]]; then
  [[ -f "${replay_source}" ]] || {
    echo "error: source motion-planning trajectory is missing: ${replay_source}" >&2
    exit 66
  }
fi
if [[ "${skip_test}" == false && \
      ! -f "datasets/raw/maniskill_demos/PickCube-v1/rl/ppo_pd_ee_delta_pos_ckpt.pt" ]]; then
  echo "error: 4D PPO reference checkpoint is missing" >&2
  exit 66
fi

evaluation_complete() {
  local directory="$1"
  [[ -f "${directory}/metrics.json" && \
     -f "${directory}/episodes.parquet" && \
     -f "${directory}/run_manifest.json" ]]
}

conversion_complete() {
  [[ -f "${audit_output}" && -f "${action_stats_output}" && \
     -f "${survival_report}" && -f "${dataset_info}" && \
     -f "${dataset_manifest}" ]] || return 1
  "${conda_bin}" run --no-capture-output -n robot-practice \
    python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); c=d.get("contract", {}); raise SystemExit(0 if d.get("gate_pass") is True and c.get("control_mode") == "pd_ee_delta_pos" and c.get("sim_backend") == "physx_cpu" and c.get("action_dim") == 4 else 1)' \
    "${audit_output}" >/dev/null
}

selection_complete() {
  local directory="$1"
  [[ -f "${directory}/official_best.json" && \
     -f "${directory}/selected_checkpoint.json" && \
     -f "${directory}/selected/pretrained_model/config.json" && \
     -f "${directory}/selected/pretrained_model/model.safetensors" ]] || return 1
  for repeat in 00 01 02 03 04; do
    evaluation_complete "${directory}/repeats/repeat_${repeat}" || return 1
  done
  "${conda_bin}" run --no-capture-output -n robot-practice \
    python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if d.get("official_best_repeats", {}).get("count") == 5 else 1)' \
    "${directory}/official_best.json" >/dev/null
}

if [[ "${skip_data}" == false ]]; then
  if [[ ! -f "${replay_h5}" ]]; then
    run_logged "official CPU replay" "${logs}/00_replay.log" \
      env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
      "${conda_bin}" run --no-capture-output -n robot-practice \
      python -m mani_skill.trajectory.replay_trajectory \
      --traj-path "${replay_source}" \
      --use-first-env-state -c pd_ee_delta_pos -o rgb \
      --save-traj --num-envs 10 -b physx_cpu
  else
    log "REUSE official replay ${replay_h5}"
  fi
  if ! conversion_complete; then
    run_logged "audit and convert 4D dataset" "${logs}/01_convert_mp_4d.log" \
      bash scripts/experiments/maniskill/convert_mp_4d.sh \
      --replay-h5="${replay_h5}" \
      --source-h5="${replay_source}" \
      --output-dir="${dataset_root}" \
      --repo-id="${dataset_repo_id}" \
      --audit-output="${audit_output}" \
      --action-stats-output="${action_stats_output}" \
      --survival-report="${survival_report}"
  else
    log "REUSE converted dataset ${dataset_root}"
  fi
else
  conversion_complete || {
    echo "error: --skip-data requires a complete, gate-passing 4D data bundle" >&2
    exit 66
  }
fi

train_arm() {
  local demo_count="$1"
  local output_dir="${root}/train_${demo_count}"
  local final_policy="${output_dir}/checkpoints/030000/pretrained_model"
  if [[ -f "${final_policy}/config.json" && -f "${final_policy}/model.safetensors" ]]; then
    log "REUSE completed train_${demo_count}"
    return 0
  fi
  run_logged "train ${demo_count} demos" "${logs}/10_train_${demo_count}.log" \
    bash scripts/experiments/maniskill/train_paper_aligned_4d.sh \
    --demo-count="${demo_count}" --num-gpus="${num_gpus}" \
    --output-dir="${output_dir}" --job-name="paper4d_${demo_count}"
}

if [[ "${skip_train}" == false ]]; then
  train_arm 1000
  train_arm 100
else
  for demo_count in 1000 100; do
    [[ -f "${root}/train_${demo_count}/checkpoints/030000/pretrained_model/config.json" ]] || {
      echo "error: --skip-train requires completed train_${demo_count}" >&2
      exit 66
    }
  done
fi

checkpoint_steps="002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000"

eval_prefix=()
if [[ "${eval_device}" == cuda* ]]; then
  eval_prefix=(env "CUDA_VISIBLE_DEVICES=${eval_gpus[0]}")
fi

select_arm() {
  local demo_count="$1"
  local train_dir="${root}/train_${demo_count}"
  local evaluation_root="${root}/official/${demo_count}"
  mkdir -p "${evaluation_root}"
  IFS=',' read -r -a steps <<< "${checkpoint_steps}"
  local -a eval_pids=()
  local -a eval_labels=()
  local -a eval_logs=()
  local -a eval_resources=()
  local -a available_resources=()
  local eval_failures=0

  if [[ "${eval_device}" == cuda* ]]; then
    available_resources=("${eval_gpus[@]}")
  else
    for ((slot = 0; slot < eval_parallelism; slot++)); do
      available_resources+=("cpu_${slot}")
    done
  fi

  wait_one_official_eval() {
    local pid="${eval_pids[0]}"
    local label="${eval_labels[0]}"
    local log_file="${eval_logs[0]}"
    local resource="${eval_resources[0]}"
    local status=0
    if wait "${pid}"; then
      log "DONE ${label}; resource=${resource}"
    else
      status=$?
      log "FAILED ${label} status=${status}; resource=${resource}; inspect ${log_file}"
      eval_failures=1
    fi
    eval_pids=("${eval_pids[@]:1}")
    eval_labels=("${eval_labels[@]:1}")
    eval_logs=("${eval_logs[@]:1}")
    eval_resources=("${eval_resources[@]:1}")
    available_resources+=("${resource}")
  }

  for step in "${steps[@]}"; do
    local step_id
    printf -v step_id '%06d' "$((10#${step}))"
    local output_dir="${evaluation_root}/checkpoint_${step_id}"
    if evaluation_complete "${output_dir}"; then
      log "REUSE official ${demo_count} checkpoint ${step_id}"
      continue
    fi
    while ((${#available_resources[@]} == 0)); do
      wait_one_official_eval
    done
    local resource="${available_resources[0]}"
    available_resources=("${available_resources[@]:1}")
    local label="official ${demo_count} checkpoint ${step_id}"
    local log_file="${logs}/20_official_${demo_count}_${step_id}.log"
    local -a child_prefix=()
    if [[ "${eval_device}" == cuda* ]]; then
      child_prefix=(env "CUDA_VISIBLE_DEVICES=${resource}")
    fi
    log "START ${label}; log=${log_file}; resource=${resource}; progress=official ${output_dir}"
    (
      "${child_prefix[@]}" bash scripts/experiments/maniskill/eval_official_protocol.sh \
        --policy-path="${train_dir}/checkpoints/${step_id}/pretrained_model" \
        --dataset-repo-id="${dataset_repo_id}" \
        --dataset-root="${dataset_root}" \
        --output-dir="${output_dir}" \
        --num-envs=10 --num-episodes=100 --max-steps=100 \
        --device="${eval_device}"
    ) >"${log_file}" 2>&1 &
    eval_pids+=("$!")
    eval_labels+=("${label}")
    eval_logs+=("${log_file}")
    eval_resources+=("${resource}")
    while ((${#eval_pids[@]} >= eval_parallelism)); do
      wait_one_official_eval
    done
  done
  while ((${#eval_pids[@]} > 0)); do
    wait_one_official_eval
  done
  if ((eval_failures != 0)); then
    return 1
  fi
  if selection_complete "${evaluation_root}"; then
    log "REUSE completed official selection ${demo_count}"
    return 0
  fi
  run_logged "select and repeat official best ${demo_count}" \
    "${logs}/21_official_best_${demo_count}.log" \
    "${eval_prefix[@]}" bash scripts/experiments/maniskill/eval_official_protocol.sh \
    --summarize-and-repeat \
    --run-dir="${train_dir}" \
    --evaluation-root="${evaluation_root}" \
    --selection-dir="${evaluation_root}" \
    --checkpoint-steps="${checkpoint_steps}" \
    --dataset-repo-id="${dataset_repo_id}" \
    --dataset-root="${dataset_root}" \
    --num-envs=10 --num-episodes=100 --max-steps=100 \
    --repeat-count=5 --device="${eval_device}"
}

if [[ "${skip_select}" == false ]]; then
  select_arm 1000
  select_arm 100
else
  for demo_count in 1000 100; do
    selection_complete "${root}/official/${demo_count}" || {
      echo "error: --skip-select requires complete official selection and 5 repeats for ${demo_count}" >&2
      exit 66
    }
  done
fi

if [[ "${skip_test}" == false ]]; then
  leakage_report="${root}/leakage_report.json"
  if [[ ! -f "${leakage_report}" ]]; then
    run_logged "CPU seed leakage audit" "${logs}/30_leakage_audit.log" \
      bash scripts/experiments/maniskill/eval_cpu_holdout.sh \
      --audit-source-h5="${replay_source}" \
      --split-file=configs/experiments/maniskill/split_cpu.json \
      --output-report="${leakage_report}"
  else
    leakage_status="$("${conda_bin}" run --no-capture-output -n robot-practice \
      python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
      "${leakage_report}")"
    [[ "${leakage_status}" == "no_exact_duplicate" ]] || {
      echo "error: existing leakage report is not clean: ${leakage_report}" >&2
      exit 67
    }
    log "REUSE clean leakage report ${leakage_report}"
  fi

  run_holdout() {
    local label="$1"
    local system="$2"
    local output_dir="$3"
    shift 3
    if evaluation_complete "${output_dir}"; then
      log "REUSE holdout ${label}"
      return 0
    fi
    local shard_root="${output_dir}/shards/parallel_${holdout_parallelism}"
    mkdir -p "${shard_root}"
    local -a holdout_pids=()
    local -a holdout_labels=()
    local -a holdout_logs=()
    local -a holdout_resources=()
    local -a available_resources=()
    local holdout_failures=0

    if [[ "${eval_device}" == cuda* ]]; then
      available_resources=("${eval_gpus[@]:0:${holdout_parallelism}}")
    else
      for ((slot = 0; slot < holdout_parallelism; slot++)); do
        available_resources+=("cpu_${slot}")
      done
    fi

    wait_one_holdout() {
      local pid="${holdout_pids[0]}"
      local shard_label="${holdout_labels[0]}"
      local log_file="${holdout_logs[0]}"
      local resource="${holdout_resources[0]}"
      local status=0
      if wait "${pid}"; then
        log "DONE ${shard_label}; resource=${resource}"
      else
        status=$?
        log "FAILED ${shard_label} status=${status}; resource=${resource}; inspect ${log_file}"
        holdout_failures=1
      fi
      holdout_pids=("${holdout_pids[@]:1}")
      holdout_labels=("${holdout_labels[@]:1}")
      holdout_logs=("${holdout_logs[@]:1}")
      holdout_resources=("${holdout_resources[@]:1}")
      available_resources+=("${resource}")
    }

    for ((worker = 0; worker < holdout_parallelism; worker++)); do
      printf -v worker_id '%02d' "${worker}"
      local shard_dir="${shard_root}/worker_${worker_id}"
      if evaluation_complete "${shard_dir}"; then
        log "REUSE holdout ${label} shard ${worker_id}"
        continue
      fi
      while ((${#available_resources[@]} == 0)); do
        wait_one_holdout
      done
      local resource="${available_resources[0]}"
      available_resources=("${available_resources[@]:1}")
      local shard_label="holdout ${label} shard ${worker_id}"
      local log_file="${logs}/31_holdout_${label}_worker_${worker_id}.log"
      local -a child_prefix=()
      if [[ "${eval_device}" == cuda* ]]; then
        child_prefix=(env "CUDA_VISIBLE_DEVICES=${resource}")
      else
        child_prefix=(env "CUDA_VISIBLE_DEVICES=")
      fi
      log "START ${shard_label}; log=${log_file}; resource=${resource}; progress=${shard_dir}"
      (
        "${child_prefix[@]}" bash scripts/experiments/maniskill/eval_cpu_holdout.sh \
          --system="${system}" \
          --split-file=configs/experiments/maniskill/split_cpu.json \
          --output-dir="${shard_dir}" \
          --device="${eval_device}" \
          --shard-index="${worker}" \
          --shard-count="${holdout_parallelism}" "$@"
      ) >"${log_file}" 2>&1 &
      holdout_pids+=("$!")
      holdout_labels+=("${shard_label}")
      holdout_logs+=("${log_file}")
      holdout_resources+=("${resource}")
      while ((${#holdout_pids[@]} >= holdout_parallelism)); do
        wait_one_holdout
      done
    done
    while ((${#holdout_pids[@]} > 0)); do
      wait_one_holdout
    done
    if ((holdout_failures != 0)); then
      return 1
    fi
    run_logged "merge holdout ${label}" "${logs}/31_merge_holdout_${label}.log" \
      bash scripts/experiments/maniskill/eval_cpu_holdout.sh \
      --merge-shards \
      --shards-root="${shard_root}" \
      --split-file=configs/experiments/maniskill/split_cpu.json \
      --output-dir="${output_dir}"
  }

  run_holdout "diffusion_1000" visual "${root}/holdout/diffusion_1000" \
    --policy-path="${root}/official/1000/selected/pretrained_model" \
    --dataset-repo-id="${dataset_repo_id}" --dataset-root="${dataset_root}"
  run_holdout "diffusion_100" visual "${root}/holdout/diffusion_100" \
    --policy-path="${root}/official/100/selected/pretrained_model" \
    --dataset-repo-id="${dataset_repo_id}" --dataset-root="${dataset_root}"
  run_holdout "random" random "${root}/holdout/random"
  run_holdout "ppo_privileged" ppo_privileged "${root}/holdout/ppo_privileged" \
    --ppo-checkpoint=datasets/raw/maniskill_demos/PickCube-v1/rl/ppo_pd_ee_delta_pos_ckpt.pt

  if [[ ! -f "${root}/comparison.json" ]]; then
    run_logged "write paper-aligned comparison" "${logs}/32_comparison.log" \
      bash scripts/experiments/maniskill/eval_cpu_holdout.sh \
      --compare \
      --official-best-1000="${root}/official/1000/official_best.json" \
      --official-best-100="${root}/official/100/official_best.json" \
      --holdout-1000="${root}/holdout/diffusion_1000/metrics.json" \
      --holdout-100="${root}/holdout/diffusion_100/metrics.json" \
      --random-metrics="${root}/holdout/random/metrics.json" \
      --ppo-metrics="${root}/holdout/ppo_privileged/metrics.json" \
      --output-dir="${root}/comparison.json"
  else
    log "REUSE comparison ${root}/comparison.json"
  fi
fi

log "PIPELINE COMPLETE; root=${root}"
