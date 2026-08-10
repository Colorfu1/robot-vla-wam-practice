#!/usr/bin/env bash
#
# Run the native ManiSkill closed-loop M0 probe.  This is the only M0 launcher
# that steps an environment; run it remotely on the user's ManiSkill machine.
#
# Development visual smoke/formal example (remote ManiSkill process):
#   bash scripts/experiments/maniskill/eval_closed_loop.sh \
#     --policy-path=outputs/maniskill/m0_probe/train_ee/checkpoints/last/pretrained_model \
#     --system=visual \
#     --task=PickCube-v1 \
#     --control-mode=pd_ee_delta_pose \
#     --split-file=configs/experiments/maniskill/split.json \
#     --seed-set=development \
#     --num-envs=200 \
#     --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
#     --output-dir=outputs/maniskill/m0_probe/eval_ee_development

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
  python -m robot_practice.experiments.maniskill.eval_closed_loop "$@"
