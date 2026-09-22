#!/usr/bin/env bash
# One recorded trajectory, unchanged actions; no model inference or training.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source_dir="${1:?usage: bash $0 RECORDED_FORENSICS_DIR}"
output="outputs/lightwam_base_robomimic_square/grasp_probe_$(date +%Y%m%d_%H%M%S)_$$"
export PYTHONNOUSERSITE=1 PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND=osmesa
export ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES=4
export ROBOT_PRACTICE_READPIXELS_PROBE=1 ROBOT_PRACTICE_GRASP_PROBE=1
echo "START grasp probe output=$output worker_selection=0 backend=osmesa"
conda run --no-capture-output -n lightwam \
  python -u -m robot_practice.experiments.rl.lightwam_rollout_replay \
  --source="$source_dir" --output-root="$output" \
  --start-method=spawn --serial-step --worker-index=0
echo "RESULT_DIR=$output"
conda run --no-capture-output -n lightwam \
  python -m robot_practice.experiments.rl.grasp_probe "$output/forensics" \
  | tee "$output/grasp_summary.txt"
