#!/usr/bin/env bash
#
# Run a SmolVLA policy closed-loop in LIBERO and save synchronized numeric,
# video, and Rerun outputs. --policy-path accepts either a Hugging Face repo ID
# or a local ordinary/PEFT pretrained_model directory.
#
# Example:
#   bash scripts/experiments/smolvla_libero/eval_libero.sh \
#     --policy-path=HuggingFaceVLA/smolvla_libero \
#     --suite=libero_goal \
#     --task-id=1 \
#     --n-episodes=1 \
#     --num-visualized-episodes=1 \
#     --n-action-steps=1 \
#     --policy-seed=2000 \
#     --init-state-ids=0 \
#     --libero-assets-dir=datasets/raw/libero_assets \
#     --output-dir=outputs/smolvla_libero/smoke/public_rerun
#
# Outputs:
#   metrics.json, metrics.md, eval_info.json, rollouts.parquet, rollout.rrd,
#   run_config.json, and one MP4 per visualized episode. Successful episodes
#   stop at the authoritative LIBERO success terminal; unsuccessful episodes
#   run until environment_done or the configured suite/time limit. Every
#   visualized Recording includes a final observation endpoint with both policy
#   cameras, measured joints, Panda pose, state, done, and success.
#   --init-state-ids accepts an explicit comma-separated official-state list
#   whose length equals --n-episodes. --policy-seed resets the policy RNG before
#   every rollout so stochastic policy comparisons remain paired.
#   --episode-seeds optionally freezes the simulator seed paired with each state.
#   --instruction overrides the task text supplied to SmolVLA. The unified
#   target protocol launcher additionally uses --goal-classifier and
#   --expected-goal for simulator-grounded Prompt Swap scoring; direct users
#   should prefer eval_target_protocols.sh for that formal protocol.
#   --libero-assets-dir forces a validated offline asset tree and disables the
#   unreliable runtime download fallback.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.evaluation.libero_smolvla "$@"
