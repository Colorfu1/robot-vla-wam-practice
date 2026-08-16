#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

baseline_h5="outputs/maniskill/stackcube_joint_to_ee_gpu_repair/03_gpu_replay_stable_none/replay.h5"
repair_h5="outputs/maniskill/stackcube_joint_to_ee_gpu_repair/04_gpu_failed_ee_repair/repair.h5"
layout="outputs/maniskill/stackcube_joint_to_ee_gpu_repair/frozen_replay_layout.json"
output_dir="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/final_live_rgb_full"
num_envs="100"

usage() {
  cat <<'EOF'
Usage:
  run_stackcube_live_rgb_pipeline.sh [options]

Freezes the already validated baseline/repair carrier batches, then executes
that exact GPU layout while directly recording RGB/action/env_state/success.

Options:
  --baseline-h5=PATH
  --repair-h5=PATH
  --layout=PATH
  --output-dir=PATH
  --num-envs=N       default: 100
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --baseline-h5=*) baseline_h5="${argument#*=}" ;;
    --repair-h5=*) repair_h5="${argument#*=}" ;;
    --layout=*) layout="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --num-envs=*) num_envs="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

echo "[live-rgb] freezing and executing validated carrier layout" >&2
env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.capture_gpu_ee_live_rgb \
  --baseline-h5 "${baseline_h5}" \
  --repair-h5 "${repair_h5}" \
  --layout "${layout}" \
  --output-dir "${output_dir}" \
  --num-envs "${num_envs}"

echo "[live-rgb] formal training input: ${output_dir}/replay.h5" >&2
