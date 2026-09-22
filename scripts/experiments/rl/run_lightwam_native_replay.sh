#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source_dir=""
output_root="outputs/lightwam_base_robomimic_square/native_replay_$(date +%Y%m%d_%H%M%S)"
for arg in "$@"; do
  case "$arg" in
    --source=*) source_dir="${arg#*=}" ;;
    --output-root=*) output_root="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
test -d "$source_dir" || { echo "--source must be an existing forensics directory" >&2; exit 2; }
test ! -e "$output_root" || { echo "Refusing existing output: $output_root" >&2; exit 2; }
command -v gcc >/dev/null || { echo "gcc missing; nothing will be installed" >&2; exit 2; }
mkdir -p "$output_root/native"
exec > >(tee "$output_root/launch.log") 2>&1
export PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export ROBOT_PRACTICE_NATIVE_TRACE_DIR="$(cd "$output_root/native" && pwd)"
library="$ROBOT_PRACTICE_NATIVE_TRACE_DIR/libabort_trace.so"
gcc -shared -fPIC -g -O0 -Wall -Wextra \
  scripts/experiments/rl/native_abort_trace.c -o "$library" -ldl
echo "START native replay source=$source_dir output=$output_root gpu=$CUDA_VISIBLE_DEVICES"
# Limit preload to this Python process tree, not conda or the interactive shell.
# Do not create huge core files; retain the small native stacks instead.
ulimit -c 0
set +e
conda run --no-capture-output -n lightwam env \
  LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u -m robot_practice.experiments.rl.lightwam_rollout_replay \
  --source="$source_dir" --output-root="$output_root/replay" \
  --start-method=spawn --serial-step
status=$?
set -e
echo "DONE native replay exit_code=$status output=$output_root"
find "$output_root/native" -name 'native_*.log' -type f -exec head -n 45 {} \;
exit "$status"
