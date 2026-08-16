#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

phase=""
config="configs/experiments/maniskill/m1_r_policy_decorator_audit.json"
output_root="outputs/policy_decorator_ms2/m1_r_protocol_audit"
count="50"
protocol=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=validate-config
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=source-audit
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=environment-audit
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=replay-smoke --count=50
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=replay-full --count=986
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=trajectory-audit
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=freeze-data
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=state-banks
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=finalize
  bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=validate-frozen --protocol=...
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --phase=*) phase="${argument#*=}" ;;
    --config=*) config="${argument#*=}" ;;
    --output-root=*) output_root="${argument#*=}" ;;
    --count=*) count="${argument#*=}" ;;
    --protocol=*) protocol="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${phase}" ]] || { echo "--phase is required" >&2; usage >&2; exit 2; }

if [[ "${phase}" != "validate-config" ]]; then
  m1w_protocol="outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json"
  test -s "${m1w_protocol}" || {
    echo "M1-R is gated on M1-W frozen_protocol.json: ${m1w_protocol}" >&2
    echo "Run M1-W state-banks, finalize, and validate-frozen first." >&2
    exit 77
  }
fi

common=(--config "${config}" --output-root "${output_root}")
if [[ "${phase}" == "validate-frozen" ]]; then
  [[ -n "${protocol}" ]] || { echo "--protocol is required" >&2; exit 2; }
  common+=(--protocol "${protocol}")
fi
if [[ "${phase}" == "replay-smoke" || "${phase}" == "replay-full" ]]; then
  common+=(--count "${count}")
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PRACTICE_ROOT="${practice_root}"
export PYTHONPATH="${practice_root}/src:${practice_root}/datasets/raw/policy_decorator/reference_repo${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n maniskill2-pidec \
  python -m robot_practice.experiments.maniskill.m1_r_protocol_audit \
  "${phase}" "${common[@]}"
