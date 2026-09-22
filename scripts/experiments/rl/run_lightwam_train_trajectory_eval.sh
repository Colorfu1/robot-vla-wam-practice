#!/usr/bin/env bash
# Training-trajectory open-loop fidelity, NOT closed-loop success.
set -euo pipefail
cd "$(dirname "$0")/../../.."
run=outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42
output="outputs/lightwam_base_robomimic_square/train_trajectory_eval_$(date +%Y%m%d_%H%M%S)_$$"
export PYTHONNOUSERSITE=1 PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$output"
exec > >(tee "$output/launch.log") 2>&1
echo "START training trajectory evaluation output=$output"
echo "CLOSED_LOOP_BLOCKED requires original simulation states and verified NPZ-to-demo mapping"
conda run --no-capture-output -n lightwam \
  python -u -m robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic \
  --checkpoints="$run/checkpoints/weights/step_026350.pt,$run/checkpoints/weights/step_102500.pt" \
  --trajectory-ids="${TRAJECTORY_IDS:-0,10,20,30,40,50,60,70}" \
  --output-root="$output/open_loop" --seed=0
echo "RESULT_DIR=$output"
echo "=== Potential original simulation datasets (identity not yet verified) ==="
find datasets -type f \( -name '*.hdf5' -o -name '*.h5' \) -print
echo "DONE phase=open_loop completed=2 total=2; closed_loop=blocked"
