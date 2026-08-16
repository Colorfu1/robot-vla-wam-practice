#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

mode=""
source_h5="datasets/raw/maniskill_demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
output_dir=""
count="10"
num_workers="1"
max_inner_steps="16"
position_tolerance="0.002"
rotation_tolerance="0.01"

usage() {
  cat <<'EOF'
Usage:
  run_peg_conversion_probe.sh --mode=fixed_delta [options]
  run_peg_conversion_probe.sh --mode=absolute [options]

Options:
  --source-h5=PATH
  --output-dir=PATH
  --count=N
  --num-workers=N
  --max-inner-steps=N       fixed_delta only; default 16
  --position-tolerance=M    fixed_delta only; default 0.002
  --rotation-tolerance=RAD  fixed_delta only; default 0.01
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --mode=*) mode="${argument#*=}" ;;
    --source-h5=*) source_h5="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --count=*) count="${argument#*=}" ;;
    --num-workers=*) num_workers="${argument#*=}" ;;
    --max-inner-steps=*) max_inner_steps="${argument#*=}" ;;
    --position-tolerance=*) position_tolerance="${argument#*=}" ;;
    --rotation-tolerance=*) rotation_tolerance="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${mode}" == "fixed_delta" || "${mode}" == "absolute" ]] || {
  echo "--mode must be fixed_delta or absolute" >&2
  usage >&2
  exit 2
}

if [[ -z "${output_dir}" ]]; then
  output_dir="outputs/maniskill/m1_protocol_audit/PegInsertionSide-v1/conversion_probe/${mode}"
fi

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.peg_conversion_probe \
  --mode "${mode}" \
  --source-h5 "${source_h5}" \
  --output-dir "${output_dir}" \
  --count "${count}" \
  --num-workers "${num_workers}" \
  --max-inner-steps "${max_inner_steps}" \
  --position-tolerance "${position_tolerance}" \
  --rotation-tolerance "${rotation_tolerance}"
