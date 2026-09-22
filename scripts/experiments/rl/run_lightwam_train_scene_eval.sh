#!/usr/bin/env bash
# Verified training-scene restoration, demo replay, then live LightWAM rollouts.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
output="${OUTPUT_ROOT:-outputs/lightwam_base_robomimic_square/train_scene_eval_$(date +%Y%m%d_%H%M%S)_$$}"
checkpoint="outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42/checkpoints/weights/step_102500.pt"
echo "START paired_training_scene_eval RESULT_DIR=$output"
exec conda run --no-capture-output -n "${POLICY_CONDA_ENV:-lightwam}" python -u \
  -m robot_practice.experiments.rl.lightwam_train_scene_eval \
  --output="$output" \
  --hdf5="${HDF5_PATH:-datasets/raw/robomimic/square/mh/demo_v141.hdf5}" \
  --trajectory-ids="${TRAJECTORY_IDS:-0,10,20,30,40,50,60,70}" \
  --checkpoints="${CHECKPOINTS:-$checkpoint}" "$@"
