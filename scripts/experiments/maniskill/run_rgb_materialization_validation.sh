#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

source_h5="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/gpu_native_source/full/trajectory.h5"
d_core_json="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.json"
output_dir="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/rgb_materialization_validation"
count="20"
num_envs="20"

usage() {
  cat <<'EOF'
Usage:
  run_rgb_materialization_validation.sh [options]

Options:
  --source-h5=PATH       GPU-native 7D source H5
  --d-core-json=PATH     Frozen D_core metadata; selects the first N source IDs
  --output-dir=PATH      Validation output directory
  --count=N              Number of audit episodes, 20..50 (default: 20)
  --num-envs=N           GPU envs per batch (default: 20)
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --source-h5=*) source_h5="${argument#*=}" ;;
    --d-core-json=*) d_core_json="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --count=*) count="${argument#*=}" ;;
    --num-envs=*) num_envs="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.rgb_materialization_validation \
  --source-h5="${source_h5}" \
  --d-core-json="${d_core_json}" \
  --output-dir="${output_dir}" \
  --count="${count}" \
  --num-envs="${num_envs}"
