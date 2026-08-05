#!/usr/bin/env bash
#
# One-shot formal held-out B0 test. It first performs a no-rollout paired
# diagnostic over the existing 20k/25k/30k validation episodes, then reads the
# frozen checkpoint winner from checkpoint_selection/selected_checkpoint.json,
# evaluates the 30 untouched test init states for each of the seven background
# LIBERO-Goal tasks, and writes one aggregate report. The test result must never
# be used to reselect a checkpoint.
#
# Fresh 8-GPU run:
#   bash scripts/experiments/smolvla_libero/eval_b0_test.sh \
#     --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline \
#     --num-gpus=8 \
#     --libero-assets-dir=datasets/raw/libero_assets
#
# Direct policy run (skips B0 selection provenance; used for the background
# regression comparison of an arbitrary finetuned policy):
#   bash scripts/experiments/smolvla_libero/eval_b0_test.sh \
#     --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline \
#     --policy-path=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected/pretrained_model \
#     --model-label=c1_selected \
#     --num-gpus=8 \
#     --libero-assets-dir=datasets/raw/libero_assets \
#     --output-dir=outputs/smolvla_libero/eval/background_regression/c1_selected
#
# Resume after interruption by repeating the same command with --resume.
# Validate the frozen workload without launching GPU jobs with --print-command.
#
# Outputs (default):
#   <run-dir>/checkpoint_selection/validation_paired_analysis.json
#   <run-dir>/checkpoint_selection/validation_paired_analysis.md
#   <output-dir>/test_results.json
#   <output-dir>/episodes.parquet
#   <run-dir>/background_test/test_manifest.json
#   <run-dir>/background_test/metrics.md
#   <run-dir>/background_test/jobs/task_*/metrics.json
#   <run-dir>/background_test/logs/task_*.log

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.smolvla_libero.b0_test_evaluation "$@"
