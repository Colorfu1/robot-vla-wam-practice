#!/usr/bin/env bash
# Purpose:
#   Execute ACT actions closed-loop in one explicitly selected LIBERO task and
#   measure simulator success, reward, action smoothness, gripper toggles, and
#   latency. Save every rollout row plus Rerun/MP4 for a configurable episode prefix.
#
# Trigger examples (run from the repository root):
#   policy="outputs/act/a_sanity/<run_timestamp>/checkpoints/<step>/pretrained_model"
#   scripts/experiments/act/eval_libero.sh \
#     --policy-path="${policy}" --suite=libero_object --task-id=8 \
#     --n-episodes=20 --num-visualized-episodes=1
#   scripts/experiments/act/eval_libero.sh \
#     --policy-path="${policy}" --suite=libero_goal --task-id=3 \
#     --n-episodes=1 --max-steps=1 --output-dir=outputs/act/libero_debug \
#     --overwrite --no-open-viewer
#   scripts/experiments/act/eval_libero.sh \
#     --policy-path="${policy}" --suite=libero_object --task-id=8 \
#     --n-episodes=20 --perturbation=proprio_gaussian \
#     --proprio-noise-std=0.01 --output-dir=outputs/act/libero_noisy
#
# Accepted arguments:
#   --policy-path PATH              Required complete compatible 7D ACT
#                                   pretrained_model directory, not a .pth or
#                                   .safetensors file. It must contain config, model,
#                                   pre/postprocessor configs, and every referenced
#                                   processor state file.
#   --suite NAME                    Required: libero_spatial, libero_object,
#                                   libero_goal, libero_10, or libero_90.
#   --task-id ID                    Required zero-based task ID within --suite.
#   --n-episodes N                  Number of rollouts (default: 20; must be positive).
#   --num-visualized-episodes N     Number of leading episodes saved to Rerun and MP4
#                                   (default: 1; range: 0..--n-episodes).
#   --perturbation NAME             Policy-input perturbation: none (default) or
#                                   proprio_gaussian.
#   --proprio-noise-std VALUE       Gaussian std in checkpoint-normalized proprioception
#                                   space (default: 0.01). Used only by
#                                   --perturbation=proprio_gaussian.
#   --seed N                        Base rollout seed (default: 1000; non-negative).
#                                   Episode i uses seed N+i and init_state_id=i.
#   --output-dir PATH               Optional result directory itself. It must be new
#                                   unless --overwrite is also supplied. If omitted,
#                                   --policy-path must use the standard run layout and
#                                   output goes to
#                                   <run>/eval/libero/<suite>/task_<ID>/<timestamp>.
#   --overwrite                     Replace an existing recognized evaluator result
#                                   at --output-dir. Requires explicit --output-dir;
#                                   unrelated directories and active runs are refused.
#   --max-steps N                   Optional positive per-episode bound. Its maximum
#                                   is selected by suite (280/300/400/520); a shorter
#                                   bound produces smoke results without formal SR.
#   --no-open-viewer                Save rollout.rrd without opening Rerun.
#   -h, --help                      Print the Python evaluator help and exit.
#
# Fixed simulator settings (not CLI arguments):
#   control_frequency=10 Hz, relative 7D control, 8D state, and two 256x256
#   RGB cameras. Closed-loop Panda geometry uses the simulator's measured 7D arm
#   joints. Episode length is selected from the suite's LeRobot contract.
#
# Outputs:
#   metrics.json, eval_info.json, rollouts.parquet, rollout.rrd, run_config.json,
#   and videos/episode_000.mp4-style files for visualized episodes.
#   Rerun shows measured state plus clean/noisy normalized policy state at t,
#   calibrated 3D measured robot/EEF geometry,
#   executed controller command, measured t->t+1 response, executable ACT queue,
#   discarded chunk tail (or Temporal Ensemble new chunk), post-action agentview,
#   selectable 7D action/8D state curves, reward/success, latency, and metrics.
#   In Free 3D, the expanded Blueprint Panel eye icons independently toggle the
#   coordinate frame, calibrated camera, robot, measured trajectory, command,
#   response, and plan groups. Thin orange is the controller command; the thicker
#   cyan arrow and endpoint are the measured EEF response at t+1.
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
