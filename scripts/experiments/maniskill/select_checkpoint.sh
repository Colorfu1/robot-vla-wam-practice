#!/usr/bin/env bash
#
# Offline M0 visual checkpoint comparison and publication.
#
# This script never creates a ManiSkill environment. Candidate rollout and
# worker merging are handled by rollout_checkpoint_selection.sh. This entry
# reads the completed development metrics, ranks the candidates, and copies
# the winner under <selection-dir>/selected/.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  select_checkpoint.sh [options]

Options:
  --run-dir=DIR              Default: outputs/maniskill/m0_probe/train_main
  --checkpoint-steps=LIST    Comma-separated steps, e.g. 010000,020000
  --selection-dir=DIR        Default: outputs/maniskill/m0_probe/checkpoint_selection
  --expected-episodes=N      Default: 200
  -h, --help                 Show this help.

Inputs:
  <selection-dir>/evaluations/checkpoint_<step>/development/metrics.json

Outputs:
  <selection-dir>/checkpoint_results.json
  <selection-dir>/selected_checkpoint.json
  <selection-dir>/selected/pretrained_model/*  (self-contained winner copy)
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

run_dir="outputs/maniskill/m0_probe/train_main"
checkpoint_steps=""
selection_dir="outputs/maniskill/m0_probe/checkpoint_selection"
expected_episodes=200

while (($# > 0)); do
  argument="$1"
  case "${argument}" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-dir=*) run_dir="${argument#*=}"; shift ;;
    --checkpoint-steps=*) checkpoint_steps="${argument#*=}"; shift ;;
    --selection-dir=*) selection_dir="${argument#*=}"; shift ;;
    --expected-episodes=*) expected_episodes="${argument#*=}"; shift ;;
    *)
      echo "error: unknown argument: ${argument}" >&2
      echo "Run with --help for supported options." >&2
      exit 64
      ;;
  esac
done

if [[ -z "${checkpoint_steps}" ]]; then
  echo "error: --checkpoint-steps is required; do not silently choose a checkpoint set" >&2
  exit 64
fi
[[ "${expected_episodes}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --expected-episodes must be a positive integer" >&2
  exit 64
}

IFS=',' read -r -a requested_steps <<< "${checkpoint_steps}"
for raw_step in "${requested_steps[@]}"; do
  [[ "${raw_step}" =~ ^[0-9]+$ ]] || {
    echo "error: invalid checkpoint step: ${raw_step}" >&2
    exit 64
  }
done

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
mkdir -p "${selection_dir}"
selection_log="${selection_dir}/selection.log"

printf '[M0 checkpoint comparison] ranking %s\n' "${checkpoint_steps}"
"${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.checkpoint_selection \
  --run-dir="${run_dir}" \
  --evaluation-root="${selection_dir}/evaluations" \
  --output-dir="${selection_dir}" \
  --checkpoint-steps="${checkpoint_steps}" \
  --expected-episodes="${expected_episodes}" \
  2>&1 | tee -a "${selection_log}"

printf '[M0 checkpoint comparison] complete; winner copy: %s/selected/pretrained_model\n' \
  "${selection_dir}"
