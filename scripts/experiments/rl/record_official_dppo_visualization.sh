#!/usr/bin/env bash
# Re-run the official DPPO pixel evaluator and record selected closed-loop episodes.
# Intended to run inside the DPPO worker/container.  It writes only dependency-light
# NPZ bundles and does not require the Rerun SDK in the DPPO environment.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
cd -- "${practice_root}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

if [[ -n "${DPPO_PYTHON:-}" ]]; then
  exec "${DPPO_PYTHON}" -m robot_practice.experiments.rl.robomimic_dppo_visualization record "$@"
fi

if [[ -x /opt/conda/envs/dppo/bin/python ]]; then
  exec /opt/conda/envs/dppo/bin/python \
    -m robot_practice.experiments.rl.robomimic_dppo_visualization record "$@"
fi

if conda_bin="$(command -v conda)"; then
  exec "${conda_bin}" run --no-capture-output -n dppo \
    python -m robot_practice.experiments.rl.robomimic_dppo_visualization record "$@"
fi

echo "error: dppo Python is unavailable; set DPPO_PYTHON explicitly" >&2
exit 69
