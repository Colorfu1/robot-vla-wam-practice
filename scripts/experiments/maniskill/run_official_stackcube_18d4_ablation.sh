#!/usr/bin/env bash
# Matched StackCube RGB 18D-state/4D-action ablation runner.
# official: native official pd_ee_delta_pos; m1w: 7D labels projected to [dx,dy,dz,gripper].
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_official_stackcube_18d4_ablation.sh --dataset-kind=official|m1w|native4d [options]
Options:
  --phase=all|prepare|train|eval       default: all
  --gpus=LIST                          default: 0,1,2,3
  --num-demos=N                        default: 787
  --total-iters=N                      default: 50000
  --eval-freq=N                        default: 10000 (training-side eval frequency)
  --save-freq=N                        default: 10000
  --batch-size=N                       default: 256 (global)
  --eval-split=development|test        default: development
  --eval-episodes=N                    default: 200
  --eval-envs=N                        default: 64 per GPU for state-bank routes
  --eval-steps=N                       default: 200 for official CPU protocol
  --eval-seed-start=N                  default: 20000
  --dataloader-workers=N               default: 4 per GPU rank
  --resume-from=PATH                   resume training from a checkpoint
  --dataset-h5=PATH
  --split-manifest=PATH
  --state-bank=PATH
  --output-root=DIR
  --official-repo=DIR
  --exp-name=NAME
EOF
}

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
practice_root="$(cd -- "$script_dir/../../.." && pwd)"
dataset_kind=""
phase="all"
official_repo_arg=""
dataset_h5=""
split_manifest="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/split_manifest.json"
output_root=""
gpus="0,1,2,3"
num_demos="787"
total_iters="50000"
eval_freq="10000"
save_freq="10000"
batch_size="256"
eval_split="development"
state_bank_arg=""
eval_episodes="200"
eval_envs="64"
eval_steps="200"
eval_seed_start="20000"
dataloader_workers="4"
resume_from=""
exp_name=""

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dataset-kind=*) dataset_kind="${1#*=}"; shift ;;
    --phase=*) phase="${1#*=}"; shift ;;
    --official-repo=*) official_repo_arg="${1#*=}"; shift ;;
    --dataset-h5=*) dataset_h5="${1#*=}"; shift ;;
    --split-manifest=*) split_manifest="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --gpus=*) gpus="${1#*=}"; shift ;;
    --num-demos=*) num_demos="${1#*=}"; shift ;;
    --total-iters=*) total_iters="${1#*=}"; shift ;;
    --eval-freq=*) eval_freq="${1#*=}"; shift ;;
    --save-freq=*) save_freq="${1#*=}"; shift ;;
    --batch-size=*) batch_size="${1#*=}"; shift ;;
    --eval-split=*) eval_split="${1#*=}"; shift ;;
    --state-bank=*) state_bank_arg="${1#*=}"; shift ;;
    --eval-episodes=*) eval_episodes="${1#*=}"; shift ;;
    --eval-envs=*) eval_envs="${1#*=}"; shift ;;
    --eval-steps=*) eval_steps="${1#*=}"; shift ;;
    --eval-seed-start=*) eval_seed_start="${1#*=}"; shift ;;
    --dataloader-workers=*) dataloader_workers="${1#*=}"; shift ;;
    --resume-from=*) resume_from="${1#*=}"; shift ;;
    --exp-name=*) exp_name="${1#*=}"; shift ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

case "$dataset_kind" in
  official)
    dataset_format="official_rgb_18d_4d"
    default_h5="datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
    default_root="outputs/maniskill/official_dp_stackcube_18d4_official"
    label="Official DP StackCube official RGB 18D/4D exact-state evaluation"
    ;;
  m1w)
    dataset_format="m1w_projected_rgb_4d"
    default_h5="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5"
    default_root="outputs/maniskill/official_dp_stackcube_18d4_m1w_projected"
    label="Official DP StackCube M1-W projected RGB 18D/4D exact-state evaluation"
    ;;
  native4d)
    # The native M1-W 4D route has the same on-disk contract as the official
    # native 4D RGB H5: qpos+qvel, two RGB cameras, and pd_ee_delta_pos.
    # Keep it separate from `official` so provenance cannot be confused.
    dataset_format="official_rgb_18d_4d"
    default_h5="outputs/maniskill/m1_wam_native_4d/StackCube-v1/06_final_live_rgb/replay.h5"
    default_root="outputs/maniskill/official_dp_stackcube_native_4d"
    label="Official DP StackCube M1-W native 4D RGB 18D/4D exact-state evaluation"
    ;;
  *) echo "error: --dataset-kind=official, --dataset-kind=m1w, or --dataset-kind=native4d is required" >&2; usage >&2; exit 64 ;;
esac
case "$phase" in all|prepare|train|eval) ;; *) echo "error: invalid --phase=$phase" >&2; exit 64 ;; esac
case "$eval_split" in development|test) ;; *) echo "error: invalid --eval-split=$eval_split" >&2; exit 64 ;; esac
[[ -n "$dataset_h5" ]] || dataset_h5="$default_h5"
[[ -n "$output_root" ]] || output_root="$default_root"
[[ -n "$exp_name" ]] || exp_name="official_dp_stackcube_18d4_${dataset_kind}_${total_iters}"
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "error: invalid --gpus=$gpus" >&2; exit 64; }
IFS=',' read -r -a gpu_ids <<< "$gpus"
gpu_count="${#gpu_ids[@]}"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ -z "${seen_gpu_ids[$gpu_id]+x}" ]] || { echo "error: duplicate GPU: $gpu_id" >&2; exit 64; }
  seen_gpu_ids["$gpu_id"]=1
done
for numeric_option in num_demos total_iters save_freq batch_size eval_episodes eval_envs eval_steps eval_seed_start dataloader_workers; do
  numeric_value="${!numeric_option}"
  [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || { echo "error: --${numeric_option//_/-} must be positive" >&2; exit 64; }
done
[[ "$eval_freq" =~ ^[0-9]+$ ]] || { echo "error: --eval-freq must be non-negative" >&2; exit 64; }
(( batch_size % gpu_count == 0 )) || { echo "error: batch size not divisible by GPU count" >&2; exit 64; }

conda_bin="$(command -v conda)" || { echo "error: conda is not on PATH" >&2; exit 69; }
cd -- "$practice_root"
to_absolute() {
  local value="$1"
  [[ "$value" == /* ]] || value="$practice_root/$value"
  if [[ -d "$value" ]]; then (cd -- "$value" && pwd)
  else printf '%s/%s\n' "$(cd -- "$(dirname -- "$value")" && pwd)" "$(basename -- "$value")"; fi
}
official_repo="$(to_absolute "${official_repo_arg:-third_party/maniskill_official}")"
dataset_h5="$(to_absolute "$dataset_h5")"
split_manifest="$(to_absolute "$split_manifest")"
output_root="$(to_absolute "$output_root")"
if [[ -n "$resume_from" ]]; then
  resume_from="$(to_absolute "$resume_from")"
fi
if [[ -n "$state_bank_arg" ]]; then
  state_bank="$(to_absolute "$state_bank_arg")"
else
  state_bank="$(to_absolute "outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/state_banks/${eval_split}.jsonl")"
fi

baseline_dir="$official_repo/examples/baselines/diffusion_policy"
train_file="$baseline_dir/train_rgbd.py"
[[ -f "$train_file" ]] || { echo "error: missing $train_file" >&2; exit 66; }
[[ -d "$baseline_dir/diffusion_policy" ]] || { echo "error: missing official package" >&2; exit 66; }
official_pythonpath="$practice_root/src:$baseline_dir"
prepare_dir="$output_root/00_prepare"
train_dir="$output_root/10_train"
eval_dir="$output_root/20_eval/$eval_split"
logs="$output_root/logs"
mkdir -p "$prepare_dir" "$train_dir" "$eval_dir" "$logs"
pipeline_log="$output_root/pipeline.log"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$pipeline_log"; }
run_logged() {
  local label_text="$1" log_file="$2"; shift 2
  log "START $label_text; log=$log_file"
  if "$@" >"$log_file" 2>&1; then log "DONE $label_text"
  else local status=$?; log "FAILED $label_text status=$status; inspect $log_file"; return "$status"; fi
}

prepare_data() {
  [[ -s "$dataset_h5" ]] || { echo "error: missing dataset H5: $dataset_h5" >&2; exit 66; }
  [[ "$dataset_kind" != m1w || -s "$split_manifest" ]] || { echo "error: missing split manifest: $split_manifest" >&2; exit 66; }
  local -a adapter_args=(
    "--dataset-format=$dataset_format" "--h5=$dataset_h5"
    "--output-dir=$prepare_dir" "--num-demos=$num_demos"
  )
  [[ "$dataset_kind" == m1w ]] && adapter_args+=("--split-manifest=$split_manifest")
  run_logged "validate $dataset_kind 18D/4D input and write sidecars" "$logs/00_prepare.log" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$official_pythonpath" \
    "$conda_bin" run --no-capture-output -n robot-practice python -m \
    robot_practice.experiments.maniskill.official_stackcube_18d4_adapter "${adapter_args[@]}"
}

train_official() {
  local train_ids="$prepare_dir/train_episode_ids.json" run_dir="$train_dir/runs/$exp_name"
  [[ -s "$train_ids" ]] || { echo "error: missing train sidecar; run --phase=prepare" >&2; exit 67; }
  if [[ -n "$resume_from" ]]; then
    [[ -s "$resume_from" ]] || { echo "error: resume checkpoint does not exist: $resume_from" >&2; exit 66; }
    [[ ! -e "$train_dir/train_complete.json" ]] || { echo "error: training is already complete: $train_dir" >&2; exit 66; }
  else
    [[ ! -e "$train_dir/train_complete.json" && ! -e "$run_dir" ]] || { echo "error: training output already exists: $train_dir" >&2; exit 66; }
  fi
  local eval_sim_backend="physx_cpu"
  # Keep the historical M1-W projected comparison protocol unchanged.  The
  # official-data branch follows the native official CPU rollout contract.
  [[ "$dataset_kind" != official ]] && eval_sim_backend="physx_cuda"
  local train_max_steps="200"
  [[ "$dataset_kind" != official ]] && train_max_steps="500"
  local -a train_args=(
    --env-id=StackCube-v1 "--demo-path=$dataset_h5" "--dataset_format=$dataset_format"
    "--episode_ids_file=$train_ids" "--num-demos=$num_demos" --obs-mode=rgb
    # Official-data evaluation contract: policy inference may use all DDP
    # GPUs, but the rollout simulator is always the native physx_cpu path.
    "--control-mode=pd_ee_delta_pos" "--sim-backend=$eval_sim_backend" "--max_episode_steps=$train_max_steps"
    "--total_iters=$total_iters" "--batch_size=$batch_size" --obs_horizon=2 --act_horizon=8 --pred_horizon=16
    # Each DDP rank infers actions on its own GPU; the official evaluator then
    # transfers actions to CPU PhysX.  The training-side eval uses fixed,
    # disjoint seeds; phase=eval repeats every checkpoint independently.
    "--eval_freq=$eval_freq" "--eval_seed=$eval_seed_start" "--save_freq=$save_freq" --log_freq=200
    --num_eval_episodes=100 --num_eval_envs=64 "--num_dataload_workers=$dataloader_workers"
    "--exp-name=$exp_name" --seed=1000 --no-track --no-capture-video
  )
  [[ -n "$resume_from" ]] && train_args+=("--resume-from=$resume_from")
  local -a train_env=(env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "PYTHONPATH=$official_pythonpath" "CUDA_VISIBLE_DEVICES=$gpus")
  local -a train_command=("$conda_bin" run --no-capture-output -n robot-practice python -m torch.distributed.run
    --standalone --nnodes=1 "--nproc_per_node=$gpu_count" "$train_file")
  log "DDP training: kind=$dataset_kind physical_gpus=$gpus world_size=$gpu_count global_batch=$batch_size local_batch=$((batch_size / gpu_count))"
  run_logged "official DP training $dataset_kind RGB 18D/4D" "$logs/10_train.log" \
    bash -c 'cd "$1" && shift && exec "$@"' bash "$train_dir" "${train_env[@]}" "${train_command[@]}" "${train_args[@]}"
  printf '{\n  "status": "complete",\n  "dataset_kind": "%s",\n  "run_dir": "%s"\n}\n' "$dataset_kind" "$run_dir" > "$train_dir/train_complete.json"
}

eval_official() {
  local run_dir="$train_dir/runs/$exp_name"
  [[ -d "$run_dir/checkpoints" ]] || { echo "error: missing checkpoint directory: $run_dir/checkpoints" >&2; exit 66; }
  local -a eval_env=(env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "PYTHONPATH=$official_pythonpath" "CUDA_VISIBLE_DEVICES=$gpus")
  local checkpoint checkpoint_name checkpoint_step checkpoint_dir
  local evaluated=0 skipped=0
  local checkpoint_eval_steps="$eval_steps"

  if [[ "$dataset_kind" != official ]]; then
    [[ -s "$state_bank" ]] || { echo "error: missing state bank: $state_bank" >&2; exit 66; }
    checkpoint_eval_steps="500"
  fi

  for checkpoint in "$run_dir"/checkpoints/*.pt; do
    [[ -f "$checkpoint" ]] || continue
    checkpoint_name="${checkpoint##*/}"
    [[ "$checkpoint_name" =~ ^([0-9]+)\.pt$ ]] || continue
    checkpoint_step="${BASH_REMATCH[1]}"
    checkpoint_dir="$eval_dir/checkpoint_$(printf '%06d' "$((10#$checkpoint_step))")"
    if [[ -s "$checkpoint_dir/metrics.json" ]]; then
      skipped=$((skipped + 1))
      continue
    fi
    local -a eval_command
    local eval_label
    if [[ "$dataset_kind" == official ]]; then
      eval_command=(
        "$conda_bin" run --no-capture-output -n robot-practice python -m
        robot_practice.experiments.maniskill.eval_official_stackcube_rgb_multi_gpu
        "--official-repo=$official_repo" "--checkpoint=$checkpoint"
        "--output-dir=$checkpoint_dir" "--gpus=$gpus"
        "--num-episodes=$eval_episodes" "--max-episode-steps=$checkpoint_eval_steps"
        "--seed-start=$eval_seed_start"
      )
      eval_label="official DP exact CPU $dataset_kind checkpoint $checkpoint_step"
      log "exact CPU development evaluation: kind=$dataset_kind checkpoint=$checkpoint split=$eval_split physical_gpus=$gpus episodes=$eval_episodes seed_start=$eval_seed_start"
    else
      eval_command=(
        "$conda_bin" run --no-capture-output -n robot-practice python -m torch.distributed.run
        --standalone --nnodes=1 "--nproc_per_node=$gpu_count"
        --module robot_practice.experiments.maniskill.eval_official_stackcube_m1w
        "--official-repo=$official_repo" "--checkpoint=$checkpoint" "--state-bank=$state_bank"
        "--output-dir=$checkpoint_dir" "--max-episodes=$eval_episodes" "--num-envs=$eval_envs"
        "--max-steps=$checkpoint_eval_steps" --control-mode=pd_ee_delta_pos --action-dim=4
        "--label=$label" --rollout-sim-backend=physx_cuda
      )
      eval_label="official DP state-bank $dataset_kind checkpoint $checkpoint_step"
      log "state-bank evaluation: kind=$dataset_kind checkpoint=$checkpoint split=$eval_split physical_gpus=$gpus workers=$gpu_count episodes=$eval_episodes max_steps=$checkpoint_eval_steps"
    fi
    run_logged "$eval_label" "$logs/20_eval_${eval_split}_${checkpoint_step}.log" \
      bash -c 'cd "$1" && shift && exec "$@"' bash "$practice_root" "${eval_env[@]}" "${eval_command[@]}"
    evaluated=$((evaluated + 1))
  done
  (( evaluated + skipped > 0 )) || { echo "error: no numeric checkpoints found under $run_dir/checkpoints" >&2; exit 66; }
  run_logged "select official DP checkpoint from exact CPU development metrics" "$logs/21_select_${eval_split}.log" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$practice_root/src" \
    "$conda_bin" run --no-capture-output -n robot-practice python -m \
    robot_practice.experiments.maniskill.select_official_stackcube_checkpoint \
    "--train-run=$run_dir" "--evaluation-root=$eval_dir" "--output-dir=$eval_dir" \
    "--selection-source=$([[ "$dataset_kind" == official ]] && echo 'exact fixed CPU development evaluation' || echo 'fixed state-bank development evaluation')"
}

log "protocol=StackCube RGB state18(qpos+qvel) action4(pd_ee_delta_pos) kind=$dataset_kind"
log "dataset=$dataset_h5 demos=$num_demos total_iters=$total_iters gpus=$gpus"
case "$phase" in
  prepare) prepare_data ;;
  train) prepare_data; train_official ;;
  eval) eval_official ;;
  all) prepare_data; train_official; eval_official ;;
esac
log "DONE phase=$phase"
echo "Ablation artifacts: $output_root"
echo "Training log: $logs/10_train.log"
echo "Training run: $train_dir/runs/$exp_name"
if [[ "$phase" == all || "$phase" == eval ]]; then echo "Exact evaluation: $eval_dir/metrics.json"; fi
