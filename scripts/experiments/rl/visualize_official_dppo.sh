#!/usr/bin/env bash
# Convert recorded official DPPO episodes to Rerun/MP4 locally.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
cd -- "${practice_root}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  exec "${PYTHON_BIN}" -m robot_practice.experiments.rl.robomimic_dppo_visualization visualize "$@"
fi

if [[ -x /opt/conda/envs/robot-practice/bin/python ]]; then
  exec /opt/conda/envs/robot-practice/bin/python \
    -m robot_practice.experiments.rl.robomimic_dppo_visualization visualize "$@"
fi

if conda_bin="$(command -v conda)"; then
  exec "${conda_bin}" run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.rl.robomimic_dppo_visualization visualize "$@"
fi

exec python -m robot_practice.experiments.rl.robomimic_dppo_visualization visualize "$@"
