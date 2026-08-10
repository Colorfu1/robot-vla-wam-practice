#!/usr/bin/env bash
#
# Create Rerun/MP4 visualizations from an existing ManiSkill replay HDF5.
# This is an offline reader: it does not create an environment or replay actions.
# The generated RRD carries robot_practice_act_offline, so --open-viewer routes
# through robot-rerun to the local Robot Rerun Offline application.
#
# Example:
#   bash scripts/experiments/maniskill/visualize_replay.sh \
#     --replay-h5=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
#     --output-dir=outputs/maniskill/m0_probe/visualization/replay \
#     --episode=0 --episode=1 --include-privileged-overlay --write-video

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
  python -m robot_practice.experiments.maniskill.replay_visualization "$@"
