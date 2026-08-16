#!/usr/bin/env bash
# Train the vendored official ManiSkill Diffusion Policy on frozen M1-W live
# RGB/7D data. The local adapter is selected explicitly by dataset_format.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_official_stackcube_m1w_7d.sh [options]

Options:
  --phase=all|prepare|train|eval
  --official-repo=DIR
  --dataset-h5=PATH
  --split-manifest=PATH
  --output-root=DIR
  --gpus=LIST               default: 0,1,2,3
  --total-iters=N           default: 50000
  --eval-freq=N             default: 10000
  --save-freq=N             default: 10000
  --batch-size=N            default: 256 (global)
  --num-eval-episodes=N     default: 100
  --num-eval-envs=N         default: 64
  --eval-split=development|test  default: development
  --eval-episodes=N         default: 200
  --eval-envs=N             default: 64 per GPU
  --eval-steps=N            default: 500
  --state-bank=PATH         default: frozen M1-W state bank for eval-split
  --exp-name=NAME
  --capture-video
EOF
}

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
practice_root="$(cd -- "$script_dir/../../.." && pwd)"
phase="all"
official_repo_arg=""
dataset_h5="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5"
split_manifest="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/split_manifest.json"
output_root="outputs/maniskill/official_dp_stackcube_m1w_7d"
gpus="0,1,2,3"
total_iters="50000"
eval_freq="10000"
save_freq="10000"
batch_size="256"
num_eval_episodes="100"
num_eval_envs="64"
eval_split="development"
eval_episodes="200"
eval_envs="64"
eval_steps="500"
state_bank_arg=""
exp_name=""
capture_video=false

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --phase=*) phase="${1#*=}"; shift ;;
    --official-repo=*) official_repo_arg="${1#*=}"; shift ;;
    --dataset-h5=*) dataset_h5="${1#*=}"; shift ;;
    --split-manifest=*) split_manifest="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --gpus=*) gpus="${1#*=}"; shift ;;
    --total-iters=*) total_iters="${1#*=}"; shift ;;
    --eval-freq=*) eval_freq="${1#*=}"; shift ;;
    --save-freq=*) save_freq="${1#*=}"; shift ;;
    --batch-size=*) batch_size="${1#*=}"; shift ;;
    --num-eval-episodes=*) num_eval_episodes="${1#*=}"; shift ;;
    --num-eval-envs=*) num_eval_envs="${1#*=}"; shift ;;
    --eval-split=*) eval_split="${1#*=}"; shift ;;
    --eval-episodes=*) eval_episodes="${1#*=}"; shift ;;
    --eval-envs=*) eval_envs="${1#*=}"; shift ;;
    --eval-steps=*) eval_steps="${1#*=}"; shift ;;
    --state-bank=*) state_bank_arg="${1#*=}"; shift ;;
    --exp-name=*) exp_name="${1#*=}"; shift ;;
    --capture-video) capture_video=true; shift ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

case "$phase" in
  all|prepare|train|eval) ;;
  *) echo "error: invalid --phase=$phase" >&2; exit 64 ;;
esac
case "$eval_split" in
  development|test) ;;
  *) echo "error: --eval-split must be development or test" >&2; exit 64 ;;
esac
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "error: --gpus must be a comma-separated list such as 0,1,2,3" >&2
  exit 64
}
IFS=',' read -r -a gpu_ids <<< "$gpus"
gpu_count="${#gpu_ids[@]}"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ -z "${seen_gpu_ids[$gpu_id]+x}" ]] || {
    echo "error: duplicate GPU in --gpus: $gpu_id" >&2
    exit 64
  }
  seen_gpu_ids["$gpu_id"]=1
done
for numeric_option in total_iters eval_freq save_freq batch_size num_eval_episodes num_eval_envs eval_episodes eval_envs eval_steps; do
  numeric_value="${!numeric_option}"
  [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: --${numeric_option//_/-} must be a positive integer" >&2
    exit 64
  }
done
(( batch_size % gpu_count == 0 )) || {
  echo "error: global batch-size=$batch_size is not divisible by GPU count=$gpu_count" >&2
  exit 64
}

conda_bin="$(command -v conda)" || {
  echo "error: conda is not available on PATH" >&2
  exit 69
}
cd -- "$practice_root"

to_absolute() {
  local value="$1"
  [[ "$value" == /* ]] || value="$practice_root/$value"
  if [[ -d "$value" ]]; then
    (cd -- "$value" && pwd)
  else
    printf '%s/%s\n' \
      "$(cd -- "$(dirname -- "$value")" && pwd)" \
      "$(basename -- "$value")"
  fi
}

root="$(to_absolute "$output_root")"
dataset_h5="$(to_absolute "$dataset_h5")"
split_manifest="$(to_absolute "$split_manifest")"
if [[ -n "$state_bank_arg" ]]; then
  state_bank="$(to_absolute "$state_bank_arg")"
else
  state_bank="$(to_absolute "outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/state_banks/$eval_split.jsonl")"
fi
prepare_dir="$root/00_prepare"
train_dir="$root/10_train"
eval_root="$root/20_eval"
eval_dir="$eval_root/$eval_split"
logs="$root/logs"
mkdir -p "$prepare_dir" "$train_dir" "$eval_root" "$logs"
pipeline_log="$root/pipeline.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$pipeline_log"
}

run_logged() {
  local label="$1" log_file="$2"
  shift 2
  log "START $label; log=$log_file"
  if "$@" >"$log_file" 2>&1; then
    log "DONE $label"
  else
    local status=$?
    log "FAILED $label status=$status; inspect $log_file"
    return "$status"
  fi
}

resolve_official_repo() {
  local candidate="$official_repo_arg"
  [[ -n "$candidate" ]] || candidate="${MANISKILL_OFFICIAL_REPO:-}"
  [[ -n "$candidate" ]] || candidate="$practice_root/third_party/maniskill_official"
  candidate="$(to_absolute "$candidate")"
  official_repo="$candidate"
  baseline_dir="$candidate/examples/baselines/diffusion_policy"
  train_file="$baseline_dir/train_rgbd.py"
  [[ -f "$train_file" ]] || {
    echo "error: missing official train_rgbd.py: $train_file" >&2
    exit 66
  }
  [[ -d "$baseline_dir/diffusion_policy" ]] || {
    echo "error: missing official diffusion_policy package: $baseline_dir" >&2
    exit 66
  }
  official_pythonpath="$practice_root/src:$baseline_dir"
}

prepare_data() {
  [[ -s "$dataset_h5" ]] || { echo "error: missing M1-W H5: $dataset_h5" >&2; exit 66; }
  [[ -s "$split_manifest" ]] || { echo "error: missing split manifest: $split_manifest" >&2; exit 66; }
  run_logged "validate M1-W H5 and write adapter sidecars" "$logs/00_prepare.log" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$official_pythonpath" \
    "$conda_bin" run --no-capture-output -n robot-practice python -m \
    robot_practice.experiments.maniskill.official_m1w_adapter \
    --h5="$dataset_h5" --split-manifest="$split_manifest" --output-dir="$prepare_dir"
}

train_official() {
  local train_ids="$prepare_dir/train_episode_ids.json"
  [[ -s "$train_ids" ]] || {
    echo "error: adapter sidecars are missing; run --phase=prepare first" >&2
    exit 67
  }
  [[ -n "$exp_name" ]] || exp_name="official_dp_stackcube_m1w_7d_$total_iters"
  local run_dir="$train_dir/runs/$exp_name"
  if [[ -e "$train_dir/train_complete.json" || -e "$run_dir" ]]; then
    echo "error: training output already exists: $train_dir" >&2
    echo "       choose a new --output-root/--exp-name; official train has no resume flag" >&2
    exit 66
  fi

  local -a train_args=(
    --env-id=StackCube-v1
    --demo-path="$dataset_h5"
    --dataset_format=m1w_live_rgb
    --episode_ids_file="$train_ids"
    --obs-mode=rgb
    --control-mode=pd_ee_delta_pose
    --sim-backend=gpu
    --max_episode_steps=500
    --num-demos=787
    --total_iters="$total_iters"
    --batch_size="$batch_size"
    --obs_horizon=2
    --act_horizon=8
    --pred_horizon=16
    --eval_freq="$eval_freq"
    --save_freq="$save_freq"
    --log_freq=200
    --num_eval_episodes="$num_eval_episodes"
    --num_eval_envs="$num_eval_envs"
    --num_dataload_workers=0
    --exp-name="$exp_name"
    --seed=1000
    --no-track
  )
  if [[ "$capture_video" == true ]]; then
    train_args+=(--capture-video)
  else
    train_args+=(--no-capture-video)
  fi

  local -a train_env=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    PYTHONPATH="$official_pythonpath" CUDA_VISIBLE_DEVICES="$gpus"
  )
  local -a train_command=(
    "$conda_bin" run --no-capture-output -n robot-practice
    python -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node="$gpu_count" "$train_file"
  )
  log "DDP training: physical_gpus=$gpus world_size=$gpu_count global_batch=$batch_size local_batch=$((batch_size / gpu_count))"
  run_logged "official DP training on M1-W live RGB/7D" "$logs/10_train.log" \
    bash -c 'cd "$1" && shift && exec "$@"' bash "$train_dir" \
    "${train_env[@]}" "${train_command[@]}" "${train_args[@]}"

  printf '%s\n' \
    '{' \
    '  "status": "complete",' \
    "  \"run_dir\": \"$run_dir\"" \
    '}' > "$train_dir/train_complete.json"
}

resolve_checkpoint() {
  local run_dir="$train_dir/runs/$exp_name"
  local candidate
  for candidate in \
    "$run_dir/checkpoints/best_eval_success_once.pt" \
    "$run_dir/checkpoints/best_eval_success_at_end.pt"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local best_path=""
  local best_step=-1
  local basename stem step
  for candidate in "$run_dir"/checkpoints/*.pt; do
    [[ -f "$candidate" ]] || continue
    basename="${candidate##*/}"
    stem="${basename%.pt}"
    [[ "$stem" =~ ^[0-9]+$ ]] || continue
    step=$((10#$stem))
    if (( step > best_step )); then
      best_step="$step"
      best_path="$candidate"
    fi
  done
  [[ -n "$best_path" ]] || {
    echo "error: no checkpoint found under $run_dir/checkpoints" >&2
    return 66
  }
  printf '%s\n' "$best_path"
}

eval_official() {
  [[ -s "$state_bank" ]] || {
    echo "error: missing frozen state bank: $state_bank" >&2
    exit 66
  }
  local checkpoint
  checkpoint="$(resolve_checkpoint)"
  [[ -s "$checkpoint" ]] || {
    echo "error: selected checkpoint is missing: $checkpoint" >&2
    exit 66
  }
  if [[ -s "$eval_dir/metrics.json" ]]; then
    echo "error: evaluation output already exists: $eval_dir/metrics.json" >&2
    echo "       choose a new --output-root or remove only this stale eval directory" >&2
    exit 66
  fi

  local -a eval_env=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    PYTHONPATH="$official_pythonpath" CUDA_VISIBLE_DEVICES="$gpus"
  )
  local -a eval_command=(
    "$conda_bin" run --no-capture-output -n robot-practice
    python -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node="$gpu_count"
    --module robot_practice.experiments.maniskill.eval_official_stackcube_m1w
    --official-repo="$official_repo"
    --checkpoint="$checkpoint"
    --state-bank="$state_bank"
    --output-dir="$eval_dir"
    --max-episodes="$eval_episodes"
    --num-envs="$eval_envs"
    --max-steps="$eval_steps"
  )
  log "exact-state evaluation: checkpoint=$checkpoint split=$eval_split physical_gpus=$gpus workers=$gpu_count episodes=$eval_episodes"
  run_logged "official DP exact-state $eval_split evaluation" "$logs/20_eval_${eval_split}.log" \
    bash -c 'cd "$1" && shift && exec "$@"' bash "$practice_root" \
    "${eval_env[@]}" "${eval_command[@]}"
}

resolve_official_repo
[[ -n "$exp_name" ]] || exp_name="official_dp_stackcube_m1w_7d_$total_iters"
log "official_repo=$official_repo"
log "protocol=StackCube-v1 RGB qpos+qvel(18D) pd_ee_delta_pose(7D) physx_cuda horizon=500"
log "dataset=$dataset_h5 train_split=787 total_iters=$total_iters gpus=$gpus"

case "$phase" in
  prepare) prepare_data ;;
  train) prepare_data; train_official ;;
  eval) eval_official ;;
  all) prepare_data; train_official; eval_official ;;
esac

log "DONE phase=$phase"
echo "M1-W official DP artifacts: $root"
echo "Training log: $logs/10_train.log"
echo "Training run: $train_dir/runs/$exp_name"
if [[ "$phase" == "all" || "$phase" == "eval" ]]; then
  echo "Exact evaluation: $eval_dir/metrics.json"
fi
