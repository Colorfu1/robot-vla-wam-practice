#!/usr/bin/env bash
# Generate Rerun recordings and a contact sheet for randomly selected StackCube train cases.
# Example:
#   bash scripts/experiments/maniskill/visualize_stackcube_train.sh \
#     --output-dir=outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.stackcube_visualization "$@"
