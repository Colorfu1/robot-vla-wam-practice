#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.peg_episode2_diagnostic \
  --source-h5=datasets/raw/maniskill_demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
  --output=outputs/maniskill/m1_peg_continue_diagnosis/p3_episode2_source_b_c.json
