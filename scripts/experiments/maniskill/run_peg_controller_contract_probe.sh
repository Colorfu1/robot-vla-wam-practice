#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/outputs/maniskill/m1_peg_continue_diagnosis/p2_controller_contract"
mkdir -p "${OUTPUT_DIR}"

cd "${ROOT_DIR}"

POLICY_DECORATOR_ROOT="${POLICY_DECORATOR_ROOT:?Set POLICY_DECORATOR_ROOT to the checked-out Policy Decorator repository}"

conda run --no-capture-output -n maniskill2-pidec \
  env PYTHONPATH="${ROOT_DIR}/src:${POLICY_DECORATOR_ROOT:-}" \
  python -m robot_practice.experiments.maniskill.peg_controller_contract_probe \
  --phase=baseline --backend=ms2 \
  --output="${OUTPUT_DIR}/ms2_baseline.json"

conda run --no-capture-output -n robot-practice \
  env PYTHONNOUSERSITE=1 \
  python -m robot_practice.experiments.maniskill.peg_controller_contract_probe \
  --phase=baseline --backend=ms3 \
  --output="${OUTPUT_DIR}/ms3_baseline.json"

conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.peg_controller_contract_probe \
  --phase=calibrate \
  --ms2="${OUTPUT_DIR}/ms2_baseline.json" \
  --ms3="${OUTPUT_DIR}/ms3_baseline.json" \
  --output="${OUTPUT_DIR}/tolerance.json"

conda run --no-capture-output -n maniskill2-pidec \
  env PYTHONPATH="${ROOT_DIR}/src:${POLICY_DECORATOR_ROOT:-}" \
  python -m robot_practice.experiments.maniskill.peg_controller_contract_probe \
  --phase=axis --backend=ms2 \
  --output="${OUTPUT_DIR}/ms2_axis.json"

conda run --no-capture-output -n robot-practice \
  env PYTHONNOUSERSITE=1 \
  python -m robot_practice.experiments.maniskill.peg_controller_contract_probe \
  --phase=axis --backend=ms3 \
  --input="${OUTPUT_DIR}/ms2_axis.json" \
  --output="${OUTPUT_DIR}/ms3_axis.json"

conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.peg_controller_contract_probe \
  --phase=compare \
  --ms2="${OUTPUT_DIR}/ms2_axis.json" \
  --ms3="${OUTPUT_DIR}/ms3_axis.json" \
  --tolerance="${OUTPUT_DIR}/tolerance.json" \
  --output="${OUTPUT_DIR}/p2_controller_contract.json"

echo "P2 controller contract probe complete: ${OUTPUT_DIR}/p2_controller_contract.json"
