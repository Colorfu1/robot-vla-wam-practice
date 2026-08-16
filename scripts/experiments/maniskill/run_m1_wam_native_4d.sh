#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

phase=""
config="configs/experiments/maniskill/m1_wam_native_4d.json"
num_envs="100"

usage() {
  cat <<'EOF'
Usage:
  run_m1_wam_native_4d.sh --phase=convert
  run_m1_wam_native_4d.sh --phase=gpu-replay --num-envs=100
  run_m1_wam_native_4d.sh --phase=gpu-repair --num-envs=100
  run_m1_wam_native_4d.sh --phase=merge
  run_m1_wam_native_4d.sh --phase=capture --num-envs=100
  run_m1_wam_native_4d.sh --phase=all --num-envs=100
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --phase=*) phase="${argument#*=}" ;;
    --config=*) config="${argument#*=}" ;;
    --num-envs=*) num_envs="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${phase}" ]] || { echo "--phase is required" >&2; usage >&2; exit 2; }

exec env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.native_4d_data_pipeline \
  --phase="${phase}" \
  --config="${config}" \
  --num-envs="${num_envs}"
