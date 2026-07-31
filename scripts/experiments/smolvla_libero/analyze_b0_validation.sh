#!/usr/bin/env bash
#
# Reuse the completed 20k/25k/30k B0 validation rollouts for paired analysis.
# This command launches no simulator and does not change the selected checkpoint.
#
#   bash scripts/experiments/smolvla_libero/analyze_b0_validation.sh \
#     --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.smolvla_libero.b0_validation_analysis "$@"
