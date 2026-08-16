#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

phase=""
task="all"
num_workers="10"
num_envs="100"
rgb_count="20"
# Active M1-W route. The legacy three-task config must be passed explicitly
# and is not a default execution path anymore.
config="configs/experiments/maniskill/m1_wam_protocol_audit.json"
output_root=""
protocol=""

usage() {
  cat <<'EOF'
Usage:
  run_m1_protocol_audit.sh --phase=download
  run_m1_protocol_audit.sh --phase=source-audit|environment-audit|trajectory-audit --task=StackCube-v1
  run_m1_protocol_audit.sh --phase=gpu-source-smoke|gpu-source-full --task=StackCube-v1
  run_m1_protocol_audit.sh --phase=replay-smoke|replay-full --task=StackCube-v1 --num-envs=100
  run_m1_protocol_audit.sh --phase=gpu-baseline-smoke|gpu-baseline-full --task=StackCube-v1
  run_m1_protocol_audit.sh --phase=gpu-repair-smoke|gpu-repair-full --task=StackCube-v1
  run_m1_protocol_audit.sh --phase=gpu-merge-smoke|gpu-merge-full --task=StackCube-v1
  run_m1_protocol_audit.sh --phase=gpu-final-smoke|gpu-final-full --task=StackCube-v1
  run_m1_protocol_audit.sh --phase=freeze-data|state-banks --task=StackCube-v1
  run_m1_protocol_audit.sh --phase=finalize
  run_m1_protocol_audit.sh --phase=validate-frozen --protocol=outputs/.../frozen_protocol.json
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --phase=*) phase="${argument#*=}" ;;
    --task=*) task="${argument#*=}" ;;
    --num-workers=*) num_workers="${argument#*=}" ;;
    --num-envs=*) num_envs="${argument#*=}" ;;
    --count=*) rgb_count="${argument#*=}" ;;
    --config=*) config="${argument#*=}" ;;
    --output-root=*) output_root="${argument#*=}" ;;
    --protocol=*) protocol="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${phase}" ]] || { echo "--phase is required" >&2; usage >&2; exit 2; }

common=(--config "${config}")
if [[ -n "${output_root}" ]]; then
  common+=(--output-root "${output_root}")
fi

run_one() {
  local command="$1"
  shift
  env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.maniskill.m1_protocol_audit \
    "${command}" "${common[@]}" "$@"
}

case "${phase}" in
  validate-config)
    run_one validate-config
    ;;
  download)
    run_one download
    ;;
  source-audit|environment-audit|trajectory-audit|freeze-data|state-banks)
    if [[ "${task}" == "all" ]]; then
      run_one "${phase}" --task all
    else
      run_one "${phase}" --task "${task}"
    fi
    ;;
  replay-smoke)
    run_one replay-smoke --task "${task}" --num-workers "${num_workers}" --num-envs "${num_envs}"
    ;;
  gpu-source-smoke)
    run_one gpu-source-smoke --task "${task}"
    ;;
  gpu-source-full)
    [[ "${task}" != "all" ]] || { echo "gpu-source-full requires one task" >&2; exit 2; }
    run_one gpu-source-full --task "${task}"
    ;;
  replay-full)
    [[ "${task}" != "all" ]] || { echo "replay-full requires one task" >&2; exit 2; }
    run_one replay-full --task "${task}" --num-workers "${num_workers}" --num-envs "${num_envs}"
    ;;
  gpu-baseline-smoke|gpu-baseline-full|gpu-repair-smoke|gpu-repair-full|gpu-merge-smoke|gpu-merge-full|gpu-final-smoke|gpu-final-full)
    [[ "${task}" != "all" ]] || { echo "${phase} requires one task" >&2; exit 2; }
    run_one "${phase}" --task "${task}" --num-envs "${num_envs}"
    ;;
  peg-predicate-audit)
    run_one peg-predicate-audit
    ;;
  freeze-horizon)
    run_one freeze-horizon --task "${task}"
    ;;
  finalize)
    run_one finalize
    ;;
  validate-frozen)
    [[ -n "${protocol}" ]] || {
      echo "validate-frozen requires --protocol=..." >&2
      usage >&2
      exit 2
    }
    env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
      python -m robot_practice.experiments.maniskill.m1_protocol_audit \
      validate-frozen --config "${config}" --protocol "${protocol}"
    ;;
  *)
    echo "Unknown phase: ${phase}" >&2
    usage >&2
    exit 2
    ;;
esac
