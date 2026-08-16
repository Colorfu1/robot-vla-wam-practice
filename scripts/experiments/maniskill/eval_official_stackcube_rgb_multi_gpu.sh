#!/usr/bin/env bash
# Multi-GPU external evaluator for one official StackCube RGB checkpoint.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$BASH_SOURCE")" && pwd)"
practice_root="$(cd -- "$script_dir/../../.." && pwd)"
cd -- "$practice_root"

exec conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.eval_official_stackcube_rgb_multi_gpu "$@"
