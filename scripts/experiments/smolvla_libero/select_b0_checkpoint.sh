#!/usr/bin/env bash
#
# Strict B0 checkpoint selection on frozen background LIBERO-Goal validation
# states. The Python coordinator assigns at most one task job to each visible
# GPU, evaluates every 5k checkpoint with full 300-step closed-loop rollouts,
# ranks by seven-task macro success rate, and copies the winner as a real,
# self-contained checkpoints/selected directory (never a symlink).
#
# Fresh 8-GPU run:
#   bash scripts/experiments/smolvla_libero/select_b0_checkpoint.sh \
#     --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline \
#     --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/005000/pretrained_model \
#     --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/010000/pretrained_model \
#     --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/015000/pretrained_model \
#     --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/020000/pretrained_model \
#     --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/025000/pretrained_model \
#     --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/030000/pretrained_model \
#     --split-file=configs/experiments/smolvla_libero/background_closed_loop_split.json \
#     --num-gpus=8 \
#     --env-seed=1000 \
#     --policy-seed=2000 \
#     --libero-assets-dir=datasets/raw/libero_assets \
#     --output-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoint_selection \
#     --selected-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected
#
# The assets directory is validated before any GPU process starts; the selector
# never downloads simulator assets from Hugging Face. Continue after
# interruption by repeating the command with --resume. Complete per-task jobs
# are verified and reused; incomplete jobs are replaced.
#
# Outputs:
#   checkpoint_selection/checkpoint_results.json
#   checkpoint_selection/selected_checkpoint.json
#   checkpoint_selection/metrics.md
#   checkpoint_selection/jobs/.../metrics.json
#   checkpoint_selection/logs/*.log
#   checkpoints/selected/pretrained_model/
#   checkpoints/selected/training_state/

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.smolvla_libero.b0_checkpoint_selection "$@"
