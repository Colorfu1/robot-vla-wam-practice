#!/usr/bin/env bash
#
# Run the same frozen 210-rollout target-task protocol for any selected SmolVLA
# B0/C1/C2/C3 policy. Only --policy-path, --model-label, and --output-dir vary
# across models; split, prompt assignment, seeds, action execution, and Prompt
# Swap classification are shared.
#
# B0 example (8 GPUs):
#   bash scripts/experiments/smolvla_libero/eval_target_protocols.sh \
#     --policy-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/025000/pretrained_model \
#     --model-label=b0_domain_adapted_zero_shot \
#     --num-gpus=8 \
#     --libero-assets-dir=datasets/raw/libero_assets \
#     --output-dir=outputs/smolvla_libero/eval/b0_domain_adapted_zero_shot
#
# Repeat the same command with each selected C1/C2/C3 policy, model label, and
# output directory. Add --resume after interruption. Add --print-command to
# validate the frozen 21-job / 210-rollout workload without launching GPUs.
#
# Default formal workload:
#   canonical:             3 tasks x 30 held-out states = 90
#   held_out_paraphrase:   paired 3 x 30 states         = 90
#   prompt_swap:           10 reference states x 3      = 30
#   total:                                                210
#
# Root outputs:
#   run_manifest.json, metrics.json, metrics.md, episodes.parquet,
#   prompt_goal_confusion.csv, jobs/**/{metrics.json,rollout.rrd,...}, logs/**.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.smolvla_libero.target_protocol_evaluation "$@"
