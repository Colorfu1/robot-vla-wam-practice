#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x /opt/conda/envs/robot-practice/bin/python ]]; then
  python_bin=/opt/conda/envs/robot-practice/bin/python
else
  python_bin=python
fi

exec "${python_bin}" -m robot_practice.experiments.rl.policy_decorator_peg_visualization "$@"
