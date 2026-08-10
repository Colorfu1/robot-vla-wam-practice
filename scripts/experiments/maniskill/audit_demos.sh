#!/usr/bin/env bash
#
# Offline HDF5 audit plus optional reset-only initial-state fingerprint audit.
# The reset portion creates ManiSkill environments and must be run remotely;
# this wrapper never replays actions.
#
# Example:
#   bash scripts/experiments/maniskill/audit_demos.sh \
#     --source-h5=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
#     --output-dir=outputs/maniskill/m0_probe \
#     --split-file=configs/experiments/maniskill/split.json \
#     --state-set=development --state-set=test

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
  python -m robot_practice.experiments.maniskill.audit_demos "$@"
