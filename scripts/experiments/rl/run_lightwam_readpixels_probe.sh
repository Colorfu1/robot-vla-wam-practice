#!/usr/bin/env bash
# Diagnostic only: same recorded actions, with readPixels boundary evidence.
set -u -o pipefail
cd "$(dirname "$0")/../../.." || exit 1
source_dir="${1:?usage: bash $0 RECORDED_FORENSICS_DIR}"
samples="${ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES:-default}"
backend="${ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND:-egl}"
case "$backend" in egl|osmesa) ;; *) echo "Invalid backend" >&2; exit 2 ;; esac
case "$samples" in
  default|0|4) ;;
  *) echo "Diagnostic offsamples must be 0 or 4" >&2; exit 2 ;;
esac
output="outputs/lightwam_base_robomimic_square/readpixels_probe_${backend}_msaa${samples}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output" || exit 1
exec > >(tee "$output/launch.log") 2>&1
export PYTHONNOUSERSITE=1 PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL="$backend" PYOPENGL_PLATFORM="$backend"
export ROBOT_PRACTICE_READPIXELS_PROBE=1
export ROBOT_PRACTICE_CUDA_CONTEXT_PROBE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ulimit -c 0
# NVML-based monitoring; includes graphics AND compute processes on all visible GPUs.
# Keep monitoring failure visible instead of interpreting absent rows as idle GPUs.
nvidia-smi -q -x -l 1 > "$output/gpu_processes.xml.log" 2>&1 &
monitor_pid=$!
stop_monitor() {
  if kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  else
    echo "WARNING GPU monitor exited early; inspect gpu_processes.xml.log"
  fi
}
trap stop_monitor EXIT
for attempt in 1 2 3; do
  echo "START attempt=$attempt total=3 gpu=0 backend=$backend offsamples=$samples"
  conda run --no-capture-output -n lightwam \
    python -u -m robot_practice.experiments.rl.lightwam_rollout_replay \
    --source="$source_dir" --output-root="$output/attempt_$attempt" \
    --start-method=spawn --serial-step
  rc=$?
  echo "STATUS completed=$attempt total=3 exit_code=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "STOP first_failure=true evidence=$output/attempt_$attempt/forensics"
    break
  fi
done
echo "RESULT_DIR=$output"
echo "GPU_PROCESS_LOG=$output/gpu_processes.xml.log"
exit "$rc"
