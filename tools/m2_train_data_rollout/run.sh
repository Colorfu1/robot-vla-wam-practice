#!/usr/bin/env bash
set -euo pipefail

tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${tool_dir}/../.." && pwd)"
cd -- "${practice_root}"

checkpoint=""
output_dir=""
sample_count=10
seed=1000
max_steps=500
n_action_steps=8
device="cuda"
d_core_h5="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5"
split_manifest="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/split_manifest.json"

usage() {
  cat <<'EOF'
Usage:
  bash tools/m2_train_data_rollout/run.sh \
    --checkpoint=outputs/.../checkpoint_step_00050000.pt \
    --output-dir=outputs/maniskill/m2_task_base/StackCube-v1/train_probe \
    [--sample-count=10] [--seed=1000] [--n-action-steps=8]
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --checkpoint=*) checkpoint="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --sample-count=*) sample_count="${argument#*=}" ;;
    --seed=*) seed="${argument#*=}" ;;
    --max-steps=*) max_steps="${argument#*=}" ;;
    --n-action-steps=*) n_action_steps="${argument#*=}" ;;
    --device=*) device="${argument#*=}" ;;
    --d-core-h5=*) d_core_h5="${argument#*=}" ;;
    --split-manifest=*) split_manifest="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${checkpoint}" ]] || { echo "--checkpoint is required" >&2; usage >&2; exit 2; }
[[ -n "${output_dir}" ]] || { echo "--output-dir is required" >&2; usage >&2; exit 2; }

exec env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n robot-practice \
  python tools/m2_train_data_rollout/rollout_train_samples.py \
    --checkpoint "${checkpoint}" \
    --d-core-h5 "${d_core_h5}" \
    --split-manifest "${split_manifest}" \
    --output-dir "${output_dir}" \
    --sample-count "${sample_count}" \
    --seed "${seed}" \
    --max-steps "${max_steps}" \
    --n-action-steps "${n_action_steps}" \
    --device "${device}"
