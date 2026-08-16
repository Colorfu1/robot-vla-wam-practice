#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

dataset_h5="${POLICY_DECORATOR_H5:-/tmp/policy_decorator_trajectory.h5}"
dataset_json="${POLICY_DECORATOR_JSON:-/tmp/policy_decorator_trajectory.json}"
output_dir="outputs/maniskill/m1_peg_continue_diagnosis/p1_exact_replay"
count="20"
sample_seed="20260811"
policy_decorator_root="${POLICY_DECORATOR_ROOT:-/tmp/policy_decorator_ref.YGDFTy}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/maniskill/run_policy_decorator_exact_replay.sh [options]

Options:
  --dataset-h5=PATH
  --dataset-json=PATH
  --output-dir=PATH
  --count=N             default 20
  --sample-seed=N       default 20260811
  --policy-decorator-root=PATH
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --dataset-h5=*) dataset_h5="${argument#*=}" ;;
    --dataset-json=*) dataset_json="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --count=*) count="${argument#*=}" ;;
    --sample-seed=*) sample_seed="${argument#*=}" ;;
    --policy-decorator-root=*) policy_decorator_root="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

test -s "${dataset_h5}" || { echo "missing dataset H5: ${dataset_h5}" >&2; exit 66; }
test -s "${dataset_json}" || { echo "missing dataset JSON: ${dataset_json}" >&2; exit 66; }
test -d "${policy_decorator_root}/envs" || {
  echo "missing Policy Decorator repo: ${policy_decorator_root}" >&2
  exit 66
}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src:${policy_decorator_root}${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n maniskill2-pidec \
  python -m robot_practice.experiments.maniskill.policy_decorator_exact_replay \
  --dataset-h5="${dataset_h5}" \
  --dataset-json="${dataset_json}" \
  --output-dir="${output_dir}" \
  --count="${count}" \
  --sample-seed="${sample_seed}"
