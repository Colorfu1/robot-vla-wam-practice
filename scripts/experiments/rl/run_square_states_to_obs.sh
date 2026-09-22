#!/usr/bin/env bash
# From repo root: conda run --no-capture-output -n dppo bash \
#   scripts/experiments/rl/run_square_states_to_obs.sh
# Optional environment: OUTPUT_DIR, DATASET, NUM_TRAJECTORIES, EGL_DEVICE_ID.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE_ID:-6}"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
unset CUDA_VISIBLE_DEVICES
output="${OUTPUT_DIR:-outputs/square_egl_cleanup_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$output")"
mkdir "$output"  # Refuse to overwrite a previous run.
output="$(cd "$output" && pwd)"
set +e
python -u -m robot_practice.experiments.rl.robomimic_states_to_obs \
  --dataset "${DATASET:-datasets/raw/robomimic/square/mh/demo_v141.hdf5}" \
  --output_name "$output/observations.hdf5" \
  --n "${NUM_TRAJECTORIES:-2}" \
  --camera_names agentview robot0_eye_in_hand \
  --camera_height 224 --camera_width 224 --done_mode 0 \
  2>&1 | tee "$output/run.log"
statuses=("${PIPESTATUS[@]}")
set -e
rc="${statuses[0]}"
if [[ "$rc" -eq 0 && "${statuses[1]}" -ne 0 ]]; then
  rc="${statuses[1]}"
fi
printf '%s\n' "$rc" > "$output/exit_code.txt"
printf 'EXIT_CODE=%s\n' "$rc" | tee -a "$output/run.log"
exit "$rc"
