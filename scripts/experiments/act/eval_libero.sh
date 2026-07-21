#!/usr/bin/env bash
# Purpose:
#   Execute ACT actions closed-loop in fixed LIBERO Object task 8 and measure actual
#   task success, reward, action smoothness, gripper toggles, and inference latency.
#   Save rollout data, synchronized Rerun visualization, and selected episode videos.
#
# Trigger examples (run from the repository root):
#   policy="outputs/act/a_sanity/<run_timestamp>/checkpoints/<step>/pretrained_model"
#   scripts/experiments/act/eval_libero.sh \
#     --policy-path="${policy}" --n-episodes=20 --num-visualized-episodes=1
#   scripts/experiments/act/eval_libero.sh \
#     --policy-path="${policy}" --n-episodes=1 --num-visualized-episodes=1 \
#     --max-steps=1 --no-open-viewer
#
# Accepted arguments:
#   --policy-path PATH              Required complete compatible 7D ACT
#                                   pretrained_model directory, not a .pth or
#                                   .safetensors file. It must contain config, model,
#                                   pre/postprocessor configs, and every referenced
#                                   processor state file.
#   --n-episodes N                  Number of rollouts (default: 20; must be positive).
#   --num-visualized-episodes N     Number of leading episodes saved to Rerun and MP4
#                                   (default: 1; range: 0..--n-episodes).
#   --perturbation NAME             Perturbation mode (default: none); only none is
#                                   currently implemented.
#   --seed N                        Base rollout seed (default: 1000; non-negative).
#                                   Episode i uses seed N+i and init_state_id=i.
#   --output-dir PATH               Optional result directory itself; it must not
#                                   exist. If omitted, --policy-path must use the
#                                   standard run layout and output goes to
#                                   <run>/eval/libero/<YYYYMMDD_HHMMSS>.
#   --max-steps N                   Optional per-episode bound in 1..280. Values below
#                                   280 produce a smoke result; 280 is formal.
#   --no-open-viewer                Save rollout.rrd without opening Rerun.
#   -h, --help                      Print the Python evaluator help and exit.
#
# Fixed simulator settings (not CLI arguments):
#   suite=libero_object, task_id=8, control_frequency=10 Hz, episode_length=280,
#   relative control, and two 256x256 RGB cameras.
#
# Outputs:
#   metrics.json, eval_info.json, rollouts.parquet, rollout.rrd, run_config.json,
#   and videos/episode_000.mp4-style files for visualized episodes.
#   A new result starts with .incomplete; successful completion removes it, while
#   failure retains it for diagnosis. Rerun launch is attempted by default; a missing
#   executable or launch failure only emits a warning.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.evaluation.libero_act "$@"
