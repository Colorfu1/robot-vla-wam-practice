#!/usr/bin/env bash
# Purpose:
#   Run teacher-forced ACT prediction on local lerobot/libero demonstrations.
#   Compare predicted and target action chunks; report open-loop error, temporal
#   smoothness, and model-query latency; save synchronized Rerun visualization.
#   This script does not execute actions and cannot measure task success.
#
# Trigger examples (run from the repository root):
#   policy="outputs/act/a_sanity/<run_timestamp>/checkpoints/<step>/pretrained_model"
#   scripts/experiments/act/eval_offline.sh --policy-path="${policy}"
#   scripts/experiments/act/eval_offline.sh \
#     --policy-path="${policy}" --max-frames=1 --no-open-viewer
#
# Accepted arguments:
#   --policy-path PATH         Required complete compatible 7D ACT pretrained_model
#                              directory, not a .pth or .safetensors file. It must
#                              contain config, model, pre/postprocessor configs, and
#                              every referenced processor state file.
#   --episodes INDEX [...]     Dataset episodes (default: 823 826 834 840 841).
#                              Values must be unique, non-negative, present locally,
#                              and all belong to fixed dataset task_index=29.
#   --dataset-root PATH        Dataset root (default under this repository:
#                              datasets/lerobot/lerobot/libero). It must contain
#                              meta/info.json; repo_id remains lerobot/libero.
#   --batch-size N             Model-call batch size (default: 1; must be positive).
#   --output-dir PATH          Optional result directory itself; it must not exist.
#                              If omitted, --policy-path must use the standard run
#                              layout and output goes to
#                              <run>/eval/offline/<YYYYMMDD_HHMMSS>.
#   --max-frames N             Optional positive global frame bound for a non-formal
#                              smoke evaluation.
#   --no-open-viewer           Save prediction.rrd without opening Rerun.
#   -h, --help                 Print the Python evaluator help and exit.
#
# Outputs:
#   metrics.json, predictions.parquet, prediction.rrd, and run_config.json.
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
  python -m robot_practice.evaluation.offline_act "$@"
