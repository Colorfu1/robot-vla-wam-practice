#!/usr/bin/env bash
# Purpose:
#   Run teacher-forced ACT prediction for one explicitly selected LIBERO task.
#   Reuse matching local demonstrations when available; otherwise download that
#   task's single upstream HDF5, replay it at 256x256, and retain both raw and
#   converted data as caches. This does not execute predicted actions and cannot
#   measure simulator task success.
#
# Trigger examples (run from the repository root):
#   policy="outputs/act/a_sanity/<run_timestamp>/checkpoints/<step>/pretrained_model"
#   scripts/experiments/act/eval_offline.sh \
#     --policy-path="${policy}" --suite=libero_object --task-id=8 \
#     --agentview-image-convention=lerobot_180
#   scripts/experiments/act/eval_offline.sh \
#     --policy-path="${policy}" --suite=libero_object --task-id=8 \
#     --episodes 823 826 --max-frames=1 --output-dir=outputs/act/offline_debug \
#     --agentview-image-convention=lerobot_180 --overwrite --no-open-viewer
#   scripts/experiments/act/eval_offline.sh \
#     --policy-path="${policy}" --suite=libero_object --task-id=8 \
#     --dataset-root=datasets/lerobot/lerobot/libero \
#     --episodes 823 826 834 840 841 --num-visualized-episodes=5 \
#     --agentview-image-convention=lerobot_180 \
#     --output-dir=outputs/act/a_sanity/offline_5 --overwrite
#
# Accepted arguments:
#   --policy-path PATH              Required complete compatible 7D ACT
#                                   pretrained_model directory, not one weight file.
#   --suite NAME                    Required: libero_spatial, libero_object,
#                                   libero_goal, libero_10, or libero_90.
#   --task-id ID                    Required zero-based task ID within --suite.
#   --episodes INDEX [...]          Dataset-native episode indices. Omit to evaluate
#                                   every demonstration for the selected task. The
#                                   printed namespace explains whether IDs are global
#                                   combined-dataset IDs or task-cache-local IDs.
#   --dataset-root PATH             Optional exact existing LeRobot dataset override.
#                                   It must unambiguously contain the selected task;
#                                   failure does not fall through to automatic data.
#   --batch-size N                  Model-call batch size (default: 1; positive).
#   --num-visualized-episodes N     Leading selected episodes written to Rerun
#                                   (default: 1; use 0 to skip per-frame Rerun data
#                                   and the auxiliary LIBERO/MuJoCo IK environment).
#   --output-dir PATH               Optional exact result directory. It must be new
#                                   unless --overwrite is also supplied. Otherwise
#                                   output is <run>/eval/offline/<suite>/task_<ID>/
#                                   <YYYYMMDD_HHMMSS>.
#   --overwrite                     Replace an existing recognized evaluator result
#                                   at --output-dir. Requires explicit --output-dir;
#                                   unrelated directories and active runs are refused.
#   --max-frames N                  Positive global smoke/debug frame bound.
#   --rebuild-dataset-cache         Rebuild a generated single-task cache. It cannot
#                                   be combined with --dataset-root and never changes
#                                   the existing combined dataset or raw HDF5.
#   --agentview-image-convention C  Calibrated-overlay image convention: auto,
#                                   lerobot_180, or physical_opencv (default: auto).
#                                   auto requires explicit generated-cache metadata;
#                                   use lerobot_180 for the verified A-Sanity dataset.
#   --no-open-viewer                Save prediction.rrd without opening Rerun.
#   -h, --help                      Print Python evaluator help and exit.
#
# Outputs:
#   metrics.json, predictions.parquet, prediction.rrd, and run_config.json.
#   Every evaluated frame is kept in metrics/Parquet; only the requested episode
#   prefix is written to Rerun. Each visualized dataset episode is an independent
#   Recording inside prediction.rrd; use Rerun's left Recording list to switch the
#   complete synchronized view between episodes. The original two recorded image
#   panes remain unchanged and keep the model-facing LeRobot/OpenVLA orientation. The upper-right
#   pane defaults to a physically calibrated agentview overlay; for lerobot_180 it
#   uses a visualization-only horizontal reflection so projected geometry aligns.
#   its second tab is a free 3D orbit view with the real agentview camera shown as
#   a cyan frustum. Eye-in-hand remains image-only because exact dynamic camera pose
#   requires joint states absent from this dataset. The normalized 3D frame uses the
#   robot base as origin with labeled X, Y Forward, and Z Up axes. Deep blue is recorded EEF
#   history. Each recorded EEF anchors separate light-blue GT and deep-orange Pred
#   first-action command segments; segments from different frames are not joined.
#   Translucent orange is only the current Pred future-command preview. Commands are
#   teacher-forced and not executed; endpoints are controller goals, not measured
#   next poses. A blue Panda skeleton is reconstructed by continuous IK. The action
#   chart exposes 14 independently selectable GT/Pred series in adjacent dimension
#   pairs. Only axis labels remain always visible; select a point or path to inspect
#   its label in Rerun's Selection panel. A new result starts with .incomplete;
#   successful completion removes it. Rerun launch failure only emits a warning.

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
