#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.m2_base_loss_probe "$@"
