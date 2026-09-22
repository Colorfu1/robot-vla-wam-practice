#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
run=outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42
export PYTHONNOUSERSITE=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
backend="${ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND:-egl}"
case "$backend" in
  egl) ;;
  osmesa)
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export ROBOT_PRACTICE_READPIXELS_PROBE=1
    export ROBOT_PRACTICE_ROLLOUT_START_METHOD=spawn
    export ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP=1
    export ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES=4
    ;;
  *) echo "Unsupported diagnostic backend: $backend" >&2; exit 2 ;;
esac
out="${DIAGNOSTIC_ROOT:-$run/diagnostic_vision_closed_loop_${backend}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$out")"
exec > >(tee "${out}.launch.log") 2>&1
echo "START diagnostic output=$out gpu=$CUDA_VISIBLE_DEVICES backend=$backend diagnostic_only=true"
conda run --no-capture-output -n lightwam python -u \
  -m robot_practice.experiments.rl.lightwam_square_diagnostic \
  --checkpoints="$run/checkpoints/weights/step_026350.pt,$run/checkpoints/weights/step_102500.pt" \
  --output-root="$out" "$@"
