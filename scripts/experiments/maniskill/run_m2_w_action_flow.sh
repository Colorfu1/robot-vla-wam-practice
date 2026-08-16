#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

phase=""
config="configs/experiments/maniskill/m2_w_action_only_flow.json"
output_root=""
train_root=""
selection=""
protocol=""
num_gpus=""
num_envs="64"
device=""
overwrite=false

usage() {
  cat <<'EOF'
Usage:
  run_m2_w_action_flow.sh --phase=validate-config --config=...
  run_m2_w_action_flow.sh --phase=smoke --config=... --num-gpus=1 --output-root=... [--overwrite]
  run_m2_w_action_flow.sh --phase=train --config=... --num-gpus=8 --output-root=... [--overwrite]
  run_m2_w_action_flow.sh --phase=evaluate-references --config=... --num-gpus=8 --output-root=... [--overwrite]
  run_m2_w_action_flow.sh --phase=select-development --config=... --num-gpus=8 --train-root=... --output-root=... [--overwrite]
  run_m2_w_action_flow.sh --phase=freeze --config=... --train-root=... --selection=... --output-root=... [--overwrite]
  run_m2_w_action_flow.sh --phase=validate-frozen --config=... --protocol=...
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --phase=*) phase="${argument#*=}" ;;
    --config=*) config="${argument#*=}" ;;
    --output-root=*) output_root="${argument#*=}" ;;
    --train-root=*) train_root="${argument#*=}" ;;
    --selection=*) selection="${argument#*=}" ;;
    --protocol=*) protocol="${argument#*=}" ;;
    --num-gpus=*) num_gpus="${argument#*=}" ;;
    --num-envs=*) num_envs="${argument#*=}" ;;
    --device=*) device="${argument#*=}" ;;
    --overwrite) overwrite=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${phase}" ]] || { echo "--phase is required" >&2; usage >&2; exit 2; }

run_python() {
  env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.maniskill.m2_w_action_flow "$@"
}

case "${phase}" in
  validate-config)
    validate_args=(validate-config --config "${config}")
    [[ -n "${num_gpus}" ]] && validate_args+=(--num-gpus "${num_gpus}")
    run_python "${validate_args[@]}"
    ;;
  smoke)
    [[ -n "${output_root}" ]] || { echo "smoke requires --output-root" >&2; exit 2; }
    [[ -n "${num_gpus}" ]] || num_gpus=1
    smoke_args=(smoke --config "${config}" --num-gpus "${num_gpus}" --output-root "${output_root}")
    [[ "${overwrite}" == true ]] && smoke_args+=(--overwrite)
    run_python "${smoke_args[@]}"
    ;;
  train)
    [[ -n "${output_root}" ]] || { echo "train requires --output-root" >&2; exit 2; }
    [[ -n "${num_gpus}" ]] || { echo "train requires --num-gpus" >&2; exit 2; }
    train_args=(train --config "${config}" --num-gpus "${num_gpus}" --output-root "${output_root}")
    [[ "${overwrite}" == true ]] && train_args+=(--overwrite)
    if [[ "${num_gpus}" -gt 1 ]]; then
      env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
        torchrun --standalone --nproc_per_node="${num_gpus}" \
        -m robot_practice.experiments.maniskill.m2_w_action_flow "${train_args[@]}"
    else
      run_python "${train_args[@]}"
    fi
    ;;
  evaluate-references)
    [[ -n "${output_root}" ]] || { echo "evaluate-references requires --output-root" >&2; exit 2; }
    [[ -n "${num_gpus}" ]] || { echo "evaluate-references requires --num-gpus" >&2; exit 2; }
    reference_args=(evaluate-references --config "${config}" --num-gpus "${num_gpus}" \
      --num-envs "${num_envs}" --output-root "${output_root}")
    [[ "${overwrite}" == true ]] && reference_args+=(--overwrite)
    run_python "${reference_args[@]}"
    ;;
  select-development)
    [[ -n "${output_root}" ]] || { echo "select-development requires --output-root" >&2; exit 2; }
    [[ -n "${train_root}" ]] || { echo "select-development requires --train-root" >&2; exit 2; }
    [[ -n "${num_gpus}" ]] || { echo "select-development requires --num-gpus" >&2; exit 2; }
    select_args=(select-development --config "${config}" --num-gpus "${num_gpus}" \
      --num-envs "${num_envs}" --train-root "${train_root}" --output-root "${output_root}")
    [[ -n "${device}" ]] && select_args+=(--device "${device}")
    [[ "${overwrite}" == true ]] && select_args+=(--overwrite)
    run_python "${select_args[@]}"
    ;;
  freeze)
    [[ -n "${train_root}" && -n "${selection}" && -n "${output_root}" ]] || {
      echo "freeze requires --train-root, --selection and --output-root" >&2
      exit 2
    }
    freeze_args=(freeze --config "${config}" --train-root "${train_root}" \
      --selection "${selection}" --output-root "${output_root}")
    [[ "${overwrite}" == true ]] && freeze_args+=(--overwrite)
    run_python "${freeze_args[@]}"
    ;;
  validate-frozen)
    [[ -n "${protocol}" ]] || { echo "validate-frozen requires --protocol" >&2; exit 2; }
    run_python validate-frozen --config "${config}" --protocol "${protocol}"
    ;;
  *)
    echo "Unknown phase: ${phase}" >&2; usage >&2; exit 2
    ;;
esac
