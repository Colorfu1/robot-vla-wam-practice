#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

source_h5="datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5"
output_root="outputs/maniskill/stackcube_joint_to_ee_cpu_gpu_replay"
conversion_num_envs="10"
cpu_num_envs="10"
gpu_num_envs="100"

usage() {
  cat <<'EOF'
Usage:
  run_official_joint_ee_replay_compare.sh [options]

Stages:
  1. Official CPU pd_joint_pos -> pd_ee_delta_pose conversion.
  2. CPU replay of the converted 7D EE actions.
  3. GPU replay of the same converted 7D EE actions.

Options:
  --source-h5=PATH
  --output-root=PATH
  --conversion-num-envs=N   default: 10
  --cpu-num-envs=N          default: 10
  --gpu-num-envs=N          default: 100
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --source-h5=*) source_h5="${argument#*=}" ;;
    --output-root=*) output_root="${argument#*=}" ;;
    --conversion-num-envs=*) conversion_num_envs="${argument#*=}" ;;
    --cpu-num-envs=*) cpu_num_envs="${argument#*=}" ;;
    --gpu-num-envs=*) gpu_num_envs="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

exec env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.official_joint_ee_replay_compare \
  --source-h5="${source_h5}" \
  --output-root="${output_root}" \
  --conversion-num-envs="${conversion_num_envs}" \
  --cpu-num-envs="${cpu_num_envs}" \
  --gpu-num-envs="${gpu_num_envs}"
