#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.eval_stackcube_gpu "$@"
