#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

source_h5="datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5"
output_dir="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/actual_ee_retarget_probe_pose"
episode_ids="30"
target_control_mode="pd_ee_delta_pose"
substeps_per_source_step="4"
num_workers="1"

usage() {
  cat <<'EOF'
Usage:
  run_stackcube_actual_ee_retarget_probe.sh [options]

Options:
  --source-h5=PATH
  --output-dir=PATH
  --episode-ids=ID[,ID,...]   default: 30
  --target-control-mode=pd_ee_delta_pos|pd_ee_delta_pose default: pd_ee_delta_pose
  --substeps-per-source-step=N default: 4
  --num-workers=N             default: 1
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --source-h5=*) source_h5="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --episode-ids=*) episode_ids="${argument#*=}" ;;
    --target-control-mode=*) target_control_mode="${argument#*=}" ;;
    --substeps-per-source-step=*) substeps_per_source_step="${argument#*=}" ;;
    --num-workers=*) num_workers="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.stackcube_actual_ee_retarget_probe \
  --source-h5 "${source_h5}" \
  --output-dir "${output_dir}" \
  --episode-ids "${episode_ids}" \
  --target-control-mode "${target_control_mode}" \
  --substeps-per-source-step "${substeps_per_source_step}" \
  --num-workers "${num_workers}"
