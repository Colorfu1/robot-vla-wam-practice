#!/usr/bin/env bash
# Official ManiSkill StackCube RGB Diffusion Policy golden anchor.
# Replay, training-time evaluation, and final verification all use physx_cpu.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_official_stackcube_rgb_anchor.sh [options]

Options:
  --phase=all|prepare|train|verify
  --official-repo=DIR       ManiSkill checkout; default is bundled offline copy
  --output-root=DIR         Default: outputs/maniskill/official_stackcube_rgb_anchor
  --gpu=N                   Policy GPU, default 0; simulator remains physx_cpu
  --gpus=LIST               Optional DDP training GPUs, e.g. 4,5,6,7
  --model-device=cuda|cpu   Policy device, default cuda
  --num-demos=N             Training demos, default 100
  --total-iters=N           Training iterations, default 100000
  --eval-freq=N             Internal evaluation interval, default 5000
  --save-freq=N             Checkpoint interval, default 10000
  --exp-name=NAME           Official run name; derived from the budget by default
  --capture-video           Save videos during final verification
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
phase="all"
official_repo_arg=""
output_root="outputs/maniskill/official_stackcube_rgb_anchor"
gpu="0"
gpus=""
model_device="cuda"
num_demos="100"
total_iters="100000"
eval_freq="5000"
save_freq="10000"
exp_name=""
capture_video=false

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --phase=*) phase="${1#*=}"; shift ;;
    --official-repo=*) official_repo_arg="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --gpu=*) gpu="${1#*=}"; shift ;;
    --gpus=*) gpus="${1#*=}"; shift ;;
    --model-device=*) model_device="${1#*=}"; shift ;;
    --num-demos=*) num_demos="${1#*=}"; shift ;;
    --total-iters=*) total_iters="${1#*=}"; shift ;;
    --eval-freq=*) eval_freq="${1#*=}"; shift ;;
    --save-freq=*) save_freq="${1#*=}"; shift ;;
    --exp-name=*) exp_name="${1#*=}"; shift ;;
    --capture-video) capture_video=true; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 64 ;;
  esac
done

case "${phase}" in
  all|prepare|train|verify) ;;
  *) echo "error: invalid --phase: ${phase}" >&2; exit 64 ;;
esac
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "error: --gpu must be non-negative" >&2; exit 64; }
gpu_count=1
eval_gpu="${gpu}"
if [[ -n "${gpus}" ]]; then
  [[ "${gpus}" =~ ^[0-9]+(,[0-9]+)+$ ]] || {
    echo "error: --gpus must be a comma-separated list such as 4,5,6,7" >&2
    exit 64
  }
  IFS=',' read -r -a gpu_ids <<< "${gpus}"
  gpu_count="${#gpu_ids[@]}"
  (( gpu_count >= 2 )) || { echo "error: --gpus requires at least two GPUs" >&2; exit 64; }
  declare -A seen_gpu_ids=()
  for gpu_id in "${gpu_ids[@]}"; do
    [[ -z "${seen_gpu_ids[${gpu_id}]+x}" ]] || {
      echo "error: duplicate GPU in --gpus: ${gpu_id}" >&2
      exit 64
    }
    seen_gpu_ids[${gpu_id}]=1
  done
  eval_gpu="${gpu_ids[0]}"
fi
for numeric_option in num_demos total_iters eval_freq save_freq; do
  numeric_value="${!numeric_option}"
  [[ "${numeric_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "error: --${numeric_option//_/-} must be a positive integer" >&2
    exit 64
  }
done
[[ "${model_device}" == cuda || "${model_device}" == cpu ]] || {
  echo "error: --model-device must be cuda or cpu" >&2
  exit 64
}
if [[ -n "${gpus}" && "${model_device}" != cuda ]]; then
  echo "error: --gpus is only supported with --model-device=cuda" >&2
  exit 64
fi

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi
cd -- "${practice_root}"

root="${output_root}"
if [[ "${root}" != /* ]]; then
  root="${practice_root}/${root}"
fi
root="$(cd -- "$(dirname -- "${root}")" && pwd)/$(basename -- "${root}")"
logs="${root}/logs"
replay_dir="${root}/00_replay"
train_dir="${root}/10_train"
verify_dir="${root}/20_verify"
mkdir -p "${logs}" "${replay_dir}" "${train_dir}" "${verify_dir}"
pipeline_log="${root}/pipeline.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${pipeline_log}"
}

run_logged() {
  local label="$1" log_file="$2"
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

resolve_official_repo() {
  local candidate="${official_repo_arg}"
  [[ -n "${candidate}" ]] || candidate="${MANISKILL_OFFICIAL_REPO:-}"
  [[ -n "${candidate}" ]] || candidate="${practice_root}/third_party/maniskill_official"
  if [[ "${candidate}" != /* ]]; then
    candidate="${practice_root}/${candidate}"
  fi
  candidate="$(cd -- "$(dirname -- "${candidate}")" && pwd)/$(basename -- "${candidate}")"
  local train_candidate="${candidate}/examples/baselines/diffusion_policy/train_rgbd.py"
  if [[ ! -f "${train_candidate}" ]]; then
    echo "error: bundled/offline official ManiSkill baseline is missing: ${train_candidate}" >&2
    echo "       sync third_party/maniskill_official or pass --official-repo=/absolute/path/to/ManiSkill" >&2
    exit 66
  fi
  [[ -f "${train_candidate}" ]] || { echo "error: missing ${train_candidate}" >&2; exit 66; }
  official_repo="${candidate}"
  baseline_dir="${official_repo}/examples/baselines/diffusion_policy"
  train_file="${baseline_dir}/train_rgbd.py"
  baselines_file="${baseline_dir}/baselines.sh"
  [[ -f "${baselines_file}" ]] || { echo "error: missing official baselines.sh: ${baselines_file}" >&2; exit 66; }
  grep -Fq 'python train_rgbd.py --env-id StackCube-v1' "${baselines_file}" || {
    echo "error: official baselines.sh has no StackCube RGB entry" >&2
    exit 66
  }
  grep -Fq -- '--max_episode_steps 200' "${baselines_file}" || {
    echo "error: official StackCube RGB entry is not max_episode_steps=200" >&2
    exit 66
  }
  grep -Fq -- '--total_iters 100000' "${baselines_file}" || {
    echo "error: official StackCube RGB entry is not total_iters=100000" >&2
    exit 66
  }
  official_pythonpath="${practice_root}/src:${baseline_dir}"
}

write_manifest() {
  local revision
  revision="$(git -C "${official_repo}" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "${revision}" && -f "${official_repo}/SOURCE_REVISION" ]]; then
    revision="$(tr -d '[:space:]' < "${official_repo}/SOURCE_REVISION")"
  fi
env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${official_pythonpath}" ANCHOR_MODEL_DEVICE="${model_device}" "${conda_bin}" run --no-capture-output -n robot-practice python - "${root}/anchor_manifest.json" "${official_repo}" "${revision}" "${train_file}" "${num_demos}" "${total_iters}" "${eval_freq}" "${save_freq}" "${exp_name}" "${gpus}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
official_repo = Path(sys.argv[2])
revision = sys.argv[3]
train_file = Path(sys.argv[4])
num_demos = int(sys.argv[5])
total_iters = int(sys.argv[6])
eval_freq = int(sys.argv[7])
save_freq = int(sys.argv[8])
exp_name = sys.argv[9]
gpus = sys.argv[10]
evaluate_file = official_repo / "examples" / "baselines" / "diffusion_policy" / "diffusion_policy" / "evaluate.py"
payload = {
    "schema_version": 2,
    "experiment": "official_stackcube_rgb_anchor",
    "task": "StackCube-v1",
    "source": "official ManiSkill motionplanning demos",
    "official_repo": str(official_repo),
    "official_git_revision": revision if revision else None,
    "official_train_rgbd_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
    "official_evaluate_sha256": hashlib.sha256(evaluate_file.read_bytes()).hexdigest(),
    "replay": {"control_mode": "pd_ee_delta_pos", "action_dim": 4, "obs_mode": "rgb", "sim_backend": "physx_cpu", "num_envs": 10},
    "training": {"num_demos": num_demos, "total_iters": total_iters, "eval_freq": eval_freq, "save_freq": save_freq, "batch_size": 256, "global_batch_size": 256, "obs_horizon": 2, "act_horizon": 8, "pred_horizon": 16, "max_episode_steps": 200, "sim_backend": "physx_cpu", "model_device": os.environ.get("ANCHOR_MODEL_DEVICE", "cuda"), "gpus": gpus.split(",") if gpus else [], "world_size": len(gpus.split(",")) if gpus else 1, "exp_name": exp_name},
    "evaluation": {"num_episodes": 100, "num_envs": 10, "max_episode_steps": 200, "partial_reset": False, "ignore_terminations": True, "reconfiguration_freq": 1, "primary_metric": "success_once", "sim_backend": "physx_cpu"},
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

source_h5="${practice_root}/datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5"
replay_h5="${practice_root}/datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
replay_json="${replay_h5%.h5}.json"
audit_json="${replay_dir}/audit.json"
if [[ -z "${exp_name}" ]]; then
  exp_name="official_stackcube_rgb_${num_demos}_${total_iters}"
fi
official_run_dir="${train_dir}/runs/${exp_name}"
best_checkpoint="${official_run_dir}/checkpoints/best_eval_success_once.pt"

audit_passed() {
  [[ -f "${audit_json}" ]] || return 1
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${official_pythonpath}" "${conda_bin}" run --no-capture-output -n robot-practice python - "${audit_json}" <<'PY' >/dev/null
import json
import sys
data = json.loads(open(sys.argv[1], encoding="utf-8").read())
raise SystemExit(0 if data.get("gate_pass") is True else 1)
PY
}

prepare_data() {
  [[ -f "${source_h5}" ]] || { echo "error: missing ${source_h5}" >&2; exit 66; }
  if [[ ! -f "${replay_h5}" || ! -f "${replay_json}" ]]; then
    run_logged "official CPU replay to RGB/4D" "${logs}/00_replay.log" env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${official_pythonpath}" "${conda_bin}" run --no-capture-output -n robot-practice python -m mani_skill.trajectory.replay_trajectory --traj-path "${source_h5}" --use-first-env-state --target-control-mode pd_ee_delta_pos --obs-mode rgb --save-traj --num-envs 10 --sim-backend physx_cpu
  else
    log "REUSE official replay ${replay_h5}"
  fi
  if audit_passed; then
    log "REUSE replay audit ${audit_json}"
  else
    run_logged "audit official RGB replay" "${logs}/01_audit.log" env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${official_pythonpath}" "${conda_bin}" run --no-capture-output -n robot-practice python -m robot_practice.experiments.maniskill.official_stackcube_rgb_anchor audit --replay-h5 "${replay_h5}" --output "${audit_json}" --num-demos "${num_demos}"
  fi
  audit_passed || { echo "error: replay audit failed: ${audit_json}" >&2; exit 67; }
}

train_official() {
  [[ -f "${replay_h5}" ]] || { echo "error: replay H5 is missing" >&2; exit 66; }
  audit_passed || { echo "error: replay audit must pass before training" >&2; exit 67; }
  if [[ -f "${train_dir}/train_complete.json" && -f "${best_checkpoint}" ]]; then
    log "REUSE completed official training ${official_run_dir}"
    return 0
  fi
  if [[ -d "${official_run_dir}" && -n "$(find "${official_run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "error: incomplete official training directory: ${official_run_dir}" >&2
    echo "       train_rgbd.py has no resume flag; choose a new output-root" >&2
    exit 66
  fi
  mkdir -p "${train_dir}"
  local -a train_args=(
    --env-id StackCube-v1 --demo-path "${replay_h5}"
    --control-mode pd_ee_delta_pos --sim-backend physx_cpu --num-demos "${num_demos}"
    --max_episode_steps 200 --total_iters "${total_iters}" --obs-mode rgb
    --batch_size 256 --obs_horizon 2 --act_horizon 8 --pred_horizon 16
    --eval_freq "${eval_freq}" --save_freq "${save_freq}" --log_freq 200
    --num_eval_episodes 100 --num_eval_envs 10 --num_dataload_workers 0
    --exp-name "${exp_name}" --seed 1 --demo_type motionplanning --no-track
  )
  [[ "${model_device}" == cuda ]] || train_args+=(--no-cuda)
  local -a train_env=(env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${official_pythonpath}")
  if [[ "${model_device}" == cuda && -n "${gpus}" ]]; then
    train_env+=(CUDA_VISIBLE_DEVICES="${gpus}")
    train_env+=("${conda_bin}" run --no-capture-output -n robot-practice python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${gpu_count}" "${train_file}")
    log "DDP training enabled: physical_gpus=${gpus} world_size=${gpu_count} global_batch_size=256 local_batch_size=$((256 / gpu_count))"
  else
    [[ "${model_device}" == cuda ]] && train_env+=(CUDA_VISIBLE_DEVICES="${gpu}")
    train_env+=("${conda_bin}" run --no-capture-output -n robot-practice python "${train_file}")
  fi
  run_logged "official StackCube RGB training + internal success_once evaluation" "${logs}/10_train.log" bash -c 'cd "$1" && shift && exec "$@"' bash "${train_dir}" "${train_env[@]}" "${train_args[@]}"
  [[ -f "${best_checkpoint}" ]] || { echo "error: best_eval_success_once.pt was not produced" >&2; exit 68; }
  env ANCHOR_MODEL_DEVICE="${model_device}" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${official_pythonpath}" "${conda_bin}" run --no-capture-output -n robot-practice python - "${train_dir}/train_complete.json" "${best_checkpoint}" "${official_run_dir}" <<'PY'
import json
import sys
from pathlib import Path
output, checkpoint, run_dir = map(Path, sys.argv[1:])
output.write_text(json.dumps({"status": "complete", "checkpoint": str(checkpoint), "run_dir": str(run_dir)}, indent=2) + "\n", encoding="utf-8")
PY
}

verify_official() {
  [[ -f "${best_checkpoint}" ]] || { echo "error: missing ${best_checkpoint}" >&2; exit 66; }
  [[ -f "${verify_dir}/metrics.json" ]] && { log "REUSE official verification"; return 0; }
  local -a verify_env=(env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${official_pythonpath}")
  local -a verify_args=()
  if [[ "${model_device}" == cuda && -n "${gpus}" ]]; then
    # One policy process per requested GPU; each worker uses the official
    # CPU rollout path and the parent only merges metrics.
    verify_args=(
      -m robot_practice.experiments.maniskill.eval_official_stackcube_rgb_multi_gpu
      --official-repo "${official_repo}" --checkpoint "${best_checkpoint}"
      --output-dir "${verify_dir}" --gpus "${gpus}"
      --num-episodes 100 --max-episode-steps 200
    )
  else
    [[ "${model_device}" == cuda ]] && verify_env+=(CUDA_VISIBLE_DEVICES="${eval_gpu}")
    verify_args=(
      -m robot_practice.experiments.maniskill.official_stackcube_rgb_verify
      --official-repo "${official_repo}" --checkpoint "${best_checkpoint}"
      --output-dir "${verify_dir}" --num-episodes 100 --num-envs 10
      --max-episode-steps 200 --device "${model_device}" --seed 1
    )
  fi
  [[ "${capture_video}" == true ]] && verify_args+=(--capture-video)
  verify_env+=("${conda_bin}" run --no-capture-output -n robot-practice python)
  run_logged "official standalone success_once verification" "${logs}/20_verify.log" "${verify_env[@]}" "${verify_args[@]}"
}

resolve_official_repo
write_manifest
log "official_repo=${official_repo}"
log "protocol=StackCube-v1 RGB pd_ee_delta_pos physx_cpu demos=${num_demos} total_iters=${total_iters} eval_freq=${eval_freq} max_episode_steps=200 train_gpus=${gpus:-single}"

case "${phase}" in
  prepare) prepare_data ;;
  train) prepare_data; train_official ;;
  verify) verify_official ;;
  all) prepare_data; train_official; verify_official ;;
esac

log "DONE phase=${phase}"
echo "Official StackCube RGB anchor artifacts: ${root}"
echo "Training run: ${official_run_dir}"
echo "Best checkpoint: ${best_checkpoint}"
echo "Verification: ${verify_dir}/metrics.json"
