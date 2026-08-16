#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

phase=""
config="configs/experiments/maniskill/m2_base_official_rgb_4d.json"
output_root=""
num_gpus=""
num_workers="4"
train_root=""
policy_device="cpu"
max_steps="200"
n_action_steps="4,8"
overwrite=false

usage() {
  cat <<'EOF'
Usage:
  run_m2_base_action_flow.sh --phase=validate-config [--config=...]
  run_m2_base_action_flow.sh --phase=smoke --config=... --output-root=... [--num-gpus=1]
  run_m2_base_action_flow.sh --phase=train --config=... --num-gpus=4 --output-root=... [--overwrite]
  run_m2_base_action_flow.sh --phase=select-development --config=... --train-root=... --output-root=... [--num-workers=4]

The train phase is offline training only. It does not run rollout, checkpoint
selection, or test evaluation.
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --phase=*) phase="${argument#*=}" ;;
    --config=*) config="${argument#*=}" ;;
    --output-root=*) output_root="${argument#*=}" ;;
    --num-gpus=*) num_gpus="${argument#*=}" ;;
    --num-workers=*) num_workers="${argument#*=}" ;;
    --train-root=*) train_root="${argument#*=}" ;;
    --policy-device=*) policy_device="${argument#*=}" ;;
    --max-steps=*) max_steps="${argument#*=}" ;;
    --n-action-steps=*) n_action_steps="${argument#*=}" ;;
    --overwrite) overwrite=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${phase}" ]] || { echo "--phase is required" >&2; usage >&2; exit 2; }

run_python() {
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.maniskill.m2_base_action_flow "$@"
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
      env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n robot-practice \
        torchrun --standalone --nproc_per_node="${num_gpus}" \
        -m robot_practice.experiments.maniskill.m2_base_action_flow "${train_args[@]}"
    else
      run_python "${train_args[@]}"
    fi
    ;;
  select-development)
    [[ -n "${train_root}" ]] || { echo "select-development requires --train-root" >&2; exit 2; }
    [[ -n "${output_root}" ]] || { echo "select-development requires --output-root" >&2; exit 2; }
    select_args=(select-development --config "${config}" --train-root "${train_root}" \
      --output-root "${output_root}" --num-workers "${num_workers}" \
      --policy-device "${policy_device}" --max-steps "${max_steps}" \
      --n-action-steps "${n_action_steps}")
    [[ "${overwrite}" == true ]] && select_args+=(--overwrite)
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run --no-capture-output -n robot-practice \
      python -m robot_practice.experiments.maniskill.m2_base_checkpoint_selection \
      "${select_args[@]}"
    ;;
  *)
    echo "Unknown phase: ${phase}" >&2; usage >&2; exit 2
    ;;
esac
