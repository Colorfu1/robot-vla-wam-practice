#!/usr/bin/env bash
#
# Convert ManiSkill replay outputs into LeRobotDataset roots.  Replay itself is
# deliberately a separate user-run prerequisite; this launcher only performs
# the offline HDF5-to-dataset conversion.
#
# Main RL 7D EEF conversion (run after remote ManiSkill replay):
#   bash scripts/experiments/maniskill/convert_demos.sh \
#     --replay-h5=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
#     --source-h5=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
#     --output-dir=datasets/lerobot/maniskill/pickcube_rl_7d_full \
#     --repo-id=local/maniskill_pickcube_rl_7d \
#     --survival-report=outputs/maniskill/m0_probe/replay_survival.json \
#     --action-stats-output=outputs/maniskill/m0_probe/action_channel_stats.json

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
  python -m robot_practice.experiments.maniskill.convert_demos "$@"
