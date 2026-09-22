#!/usr/bin/env bash
# Same live-policy contract; record every 20 Hz control step and verify prefix replay.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
checkpoint="${CHECKPOINT:-outputs/fastwam_square_b32_12h_cosine_20260920_182225/checkpoints/step_020262.pt}"
output="${OUTPUT_ROOT:-outputs/fastwam_train_scene_capture_$(date +%Y%m%d_%H%M%S)_$$}"
echo "START phase=scene_capture RESULT_DIR=$output GPU=$CUDA_VISIBLE_DEVICES"
exec conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.lightwam_train_scene_eval \
  --policy fastwam --capture-scene --max-steps 600 --output "$output" \
  --trajectory-ids "${TRAJECTORY_IDS:-0,10,20,30,40,50,60,70}" \
  --checkpoints "$checkpoint" "$@"
