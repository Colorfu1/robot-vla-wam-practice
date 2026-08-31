#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Keep the official Light-WAM loader from starting an unapproved multi-GB model
# download.  After the model artifact has been prepared and approved, callers
# may explicitly export DIFFSYNTH_SKIP_DOWNLOAD=false.
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x /opt/conda/envs/lightwam-peg/bin/python ]]; then
  python_bin=/opt/conda/envs/lightwam-peg/bin/python
else
  python_bin=python
fi
exec "${python_bin}" -m robot_practice.experiments.rl.maniskill_lightwam_residual "$@"
