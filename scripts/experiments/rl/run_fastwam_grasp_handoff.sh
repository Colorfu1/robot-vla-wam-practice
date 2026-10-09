#!/usr/bin/env bash
# Usage: GPUS=0,1,2,3 OUT=outputs/NEW bash scripts/experiments/rl/run_fastwam_grasp_handoff.sh
# Four EGL jobs: saved original_execute32 history, switch at 380/412, then execute4 vs execute32.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS=2
visual="${VISUAL_SOURCE:-outputs/fastwam_dual_visual_checks_20260923_102341}"
handoff="${HANDOFF_SOURCE:-outputs/fastwam_execution_window_probe_20261008_063142/original_execute32}"
out="${OUT:-outputs/fastwam_grasp_handoff_$(date +%Y%m%d_%H%M%S)}"
test -s "$visual/metrics.json"
test -s "$handoff/metrics.json"
test -s "$handoff/continuation/control_trace.npz"
test -s "$handoff/plans.npz"
echo "START phase=handoff output=$out switches=380,412 variants=4"
exec conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.fastwam_execution_window_probe \
  --source "$visual" --handoff-source "$handoff" --switch-steps 380,412 \
  --trajectory 0 --max-steps 600 --gpus "${GPUS:-0,1,2,3}" --output "$out"
