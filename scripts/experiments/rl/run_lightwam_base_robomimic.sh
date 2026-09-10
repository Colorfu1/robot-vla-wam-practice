#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${repo_root}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -x /opt/conda/envs/lightwam/bin/python ]]; then
  python_bin=/opt/conda/envs/lightwam/bin/python
elif [[ -x /opt/conda/envs/lightwam-peg/bin/python ]]; then
  python_bin=/opt/conda/envs/lightwam-peg/bin/python
elif [[ "${ROBOT_PRACTICE_ALLOW_HOST_PYTHON:-0}" == "1" ]]; then
  python_bin=python
else
  echo "Light-WAM runtime is unavailable: expected /opt/conda/envs/lightwam/bin/python" >&2
  echo "Set ROBOT_PRACTICE_ALLOW_HOST_PYTHON=1 only for a deliberately configured " \
    "host environment." >&2
  exit 1
fi

phase=""
remaining_args=("$@")
while ((${#remaining_args[@]} > 0)); do
  arg="${remaining_args[0]}"
  remaining_args=("${remaining_args[@]:1}")
  case "${arg}" in
    --phase=*)
      phase="${arg#--phase=}"
      ;;
    --phase)
      if ((${#remaining_args[@]} == 0)); then
        echo "--phase requires a value" >&2
        exit 2
      fi
      phase="${remaining_args[0]}"
      remaining_args=("${remaining_args[@]:1}")
      ;;
  esac
done

module=("-m" "robot_practice.experiments.rl.lightwam_base")
if [[ "${phase}" == "train" || "${phase}" == "smoke" ]]; then
  # Wan22Trainer currently assumes that Accelerator was initialized with a
  # DeepSpeed plugin.  The official Light-WAM launch path supplies that plugin
  # through Accelerate; a direct Python invocation does not.  The real-training
  # smoke and formal train therefore share the exact official launcher.
  python_dir="$(dirname "${python_bin}")"
  accelerate_bin="${python_dir}/accelerate"
  vendor_root="${LIGHTWAM_VENDOR_ROOT:-${repo_root}/third_party/lightwam/Light-WAM}"
  default_accelerate_config="${vendor_root}/scripts/accelerate_configs/accelerate_zero1_ds.yaml"
  default_deepspeed_config="${vendor_root}/scripts/ds_configs/ds_zero1_config.json"
  accelerate_config="${LIGHTWAM_ACCELERATE_CONFIG:-${default_accelerate_config}}"
  deepspeed_config="${LIGHTWAM_DEEPSPEED_CONFIG:-${default_deepspeed_config}}"
  if [[ ! -x "${accelerate_bin}" ]]; then
    echo "Light-WAM training requires Accelerate in the selected environment: ${accelerate_bin}" >&2
    exit 1
  fi
  if [[ ! -f "${accelerate_config}" ]]; then
    echo "Light-WAM Accelerate config is missing: ${accelerate_config}" >&2
    exit 1
  fi
  if [[ ! -f "${deepspeed_config}" ]]; then
    echo "Light-WAM DeepSpeed config is missing: ${deepspeed_config}" >&2
    exit 1
  fi

  launch_args=(
    launch
    --config_file "${accelerate_config}"
    --deepspeed_config_file "${deepspeed_config}"
  )
  if [[ -n "${LIGHTWAM_NUM_PROCESSES:-}" ]]; then
    launch_args+=(--num_processes "${LIGHTWAM_NUM_PROCESSES}")
  fi
  if [[ -n "${LIGHTWAM_MAIN_PROCESS_PORT:-}" ]]; then
    launch_args+=(--main_process_port "${LIGHTWAM_MAIN_PROCESS_PORT}")
  fi
  echo "[lightwam-base] ${phase} launcher=${accelerate_bin}" >&2
  echo "[lightwam-base] accelerate_config=${accelerate_config}" >&2
  echo "[lightwam-base] deepspeed_config=${deepspeed_config}" >&2
  exec "${accelerate_bin}" "${launch_args[@]}" "${module[@]}" "$@"
fi

exec "${python_bin}" "${module[@]}" "$@"
