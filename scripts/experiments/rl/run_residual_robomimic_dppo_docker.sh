#!/usr/bin/env bash
# Run one residual robomimic phase by launching the pinned DPPO pixel image.
# This wrapper is for a Docker host only; inside an already-started DPPO worker,
# call run_residual_robomimic_dppo.sh directly.
# Usage: ROBOT_PRACTICE_DPPO_GPU=0 bash "$0" --phase=preflight [runner arguments]
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
image="${ROBOT_PRACTICE_DPPO_IMAGE:-practice:robomimic-dppo-pixel}"
gpu="${ROBOT_PRACTICE_DPPO_GPU:-0}"

if [[ -z "${gpu}" || "${gpu}" == "all" || "${gpu}" == *,* ]]; then
  echo "ROBOT_PRACTICE_DPPO_GPU must select exactly one GPU" >&2
  exit 2
fi

exec docker run --rm --gpus "device=${gpu}" --ipc=host --shm-size=16g \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -e SDL_VIDEODRIVER=dummy \
  -v "${repo_root}:/workspace/practice" \
  -w /workspace/practice \
  "${image}" \
  bash scripts/experiments/rl/run_residual_robomimic_dppo.sh "$@"
