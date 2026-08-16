#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/outputs/maniskill/m1_peg_continue_diagnosis/p3_cross_version"
DATASET_H5="${POLICY_DECORATOR_H5:?Set POLICY_DECORATOR_H5 to the Policy Decorator H5 dataset}"
DATASET_JSON="${POLICY_DECORATOR_JSON:?Set POLICY_DECORATOR_JSON to the Policy Decorator JSON metadata}"
POLICY_DECORATOR_ROOT="${POLICY_DECORATOR_ROOT:?Set POLICY_DECORATOR_ROOT to the checked-out Policy Decorator repository}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

conda run --no-capture-output -n maniskill2-pidec \
  env PYTHONPATH="${ROOT_DIR}/src:${POLICY_DECORATOR_ROOT}" \
  python -m robot_practice.experiments.maniskill.peg_cross_version_probe \
  --backend=ms2 --phase=trajectory \
  --dataset-h5="${DATASET_H5}" --dataset-json="${DATASET_JSON}" \
  --output="${OUTPUT_DIR}/ms2_reference.json"

conda run --no-capture-output -n robot-practice \
  env PYTHONNOUSERSITE=1 \
  python -m robot_practice.experiments.maniskill.peg_cross_version_probe \
  --backend=ms3 --phase=trajectory \
  --source="${OUTPUT_DIR}/ms2_reference.json" \
  --output="${OUTPUT_DIR}/ms3_semantic_converted.json"

conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.peg_cross_version_probe \
  --phase=summary \
  --ms2="${OUTPUT_DIR}/ms2_reference.json" \
  --ms3="${OUTPUT_DIR}/ms3_semantic_converted.json" \
  --output="${OUTPUT_DIR}/p3_cross_version_summary.json"

echo "P3 cross-version probe complete: ${OUTPUT_DIR}/p3_cross_version_summary.json"
