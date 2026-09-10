#!/usr/bin/env bash
# Audit the official DPPO evaluation result.npz files.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
cd -- "${practice_root}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" && -x /opt/conda/envs/robot-practice/bin/python ]]; then
  python_bin=/opt/conda/envs/robot-practice/bin/python
fi
if [[ -z "${python_bin}" ]]; then
  python_bin=python
fi

exec "${python_bin}" -m robot_practice.experiments.rl.robomimic_dppo_visualization audit "$@"
