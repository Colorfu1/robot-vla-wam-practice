#!/usr/bin/env bash
# Build one Rerun Recording per W0 Fast-WAM LIBERO task and open the merged archive.
#
# The default selects the first failed rollout for each task (or the first successful
# rollout when that task has no failures), so the Rerun left Recording selector shows
# all 40 task-level scenes without pretending that MP4-only output contains 3D state.

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
  python -m robot_practice.experiments.literature.w0_fastwam_visualization "$@"
