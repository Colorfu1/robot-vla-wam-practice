#!/usr/bin/env bash
#
# Download/freeze raw LIBERO-Goal sources, scaffold reviewed paraphrases, or
# build the B0 background, target demonstration, and closed-loop splits.
#
# Examples:
#   bash scripts/experiments/smolvla_libero/prepare_data.sh \
#     --stage=fetch-sources \
#     --raw-data-root=datasets/raw/libero \
#     --output-dir=outputs/smolvla_libero/data_audit
#
#   bash scripts/experiments/smolvla_libero/prepare_data.sh \
#     --stage=scaffold-paraphrases \
#     --output-paraphrases=configs/experiments/smolvla_libero/paraphrases.json
#
#   bash scripts/experiments/smolvla_libero/prepare_data.sh \
#     --stage=build-splits \
#     --dataset-root=datasets/lerobot/lerobot/libero \
#     --raw-data-root=datasets/raw/libero \
#     --paraphrases=configs/experiments/smolvla_libero/paraphrases.json \
#     --offline-validation-per-task=10 \
#     --closed-loop-validation-per-task=20 \
#     --closed-loop-test-per-task=30 \
#     --output-dir=outputs/smolvla_libero/data_audit

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.smolvla_libero.data_preparation "$@"
