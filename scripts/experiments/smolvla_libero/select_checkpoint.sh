#!/usr/bin/env bash
#
# Shared C1/C2/C3 target checkpoint selection. Every frozen checkpoint is
# evaluated on the three target LIBERO-Goal tasks with 20 validation states
# each (360 rollouts for six checkpoints), plus a fixed-frame, fixed-flow-noise
# offline open-loop action error over the 30 offline validation demonstrations.
# Ranking is three-level: macro closed-loop SR desc, macro normalized offline
# action MSE asc, then earlier step. The winner is published as a real,
# self-contained checkpoints/selected directory (never a symlink).
#
# C1 example:
#   bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
#     --run-dir=outputs/smolvla_libero/pilot/c1_default_non_peft \
#     --dataset-root=datasets/lerobot/lerobot/libero \
#     --suite=libero_goal \
#     --task-ids=1,4,8 \
#     --split-file=configs/experiments/smolvla_libero/target_split.json \
#     --checkpoint-steps=5000,10000,15000,20000,25000,30000 \
#     --policy-seed=1500 \
#     --output-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoint_selection \
#     --selected-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected
#
# C2/C3 use the same flags with their own --run-dir and output paths.
#
# Outputs:
#   checkpoint_selection/checkpoint_results.json
#   checkpoint_selection/selected_checkpoint.json
#   checkpoint_selection/ranking_audit.json
#   checkpoint_selection/metrics.md
#   checkpoint_selection/jobs/.../metrics.json
#   checkpoint_selection/logs/*.log
#   checkpoints/selected/pretrained_model/

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.smolvla_libero.target_checkpoint_selection "$@"
