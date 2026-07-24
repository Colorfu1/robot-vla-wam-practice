#!/usr/bin/env bash
# Purpose:
#   Launch the fixed ACT action-chunking comparison through one readable entry.
#   chunk_size controls the supervised training horizon. n_action_steps is stored
#   as the checkpoint's default inference execution horizon and is not used by
#   ACTPolicy.forward() to compute the training loss.
#
# Examples (run from the repository root):
#   scripts/experiments/act/action_chunking.sh --variant=a0 --steps=20000
#   scripts/experiments/act/action_chunking.sh --variant=a1 --steps=20000
#   scripts/experiments/act/action_chunking.sh --variant=a0 --steps=0
#   scripts/experiments/act/action_chunking.sh \
#     --variant=a1 --steps=20000 --policy-path=<A1 step-0 pretrained_model>
#   scripts/experiments/act/action_chunking.sh \
#     --variant=a1 --steps=20000 --print-command
#
# Accepted arguments:
#   --variant=a0|a1    Required. A0 is single-step; A1 uses action chunking.
#   --steps=N          Optimizer updates; default 20000. N=0 saves step-0 weights.
#   --policy-path=PATH Compatible same-variant checkpoint for a positive-step run.
#   --print-command    Print the final command without creating output or running it.
#   -h, --help         Show usage.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: action_chunking.sh --variant=a0|a1 [--steps=N] [--policy-path=PATH] [--print-command]

Variants:
  a0  ACT single-step baseline: training chunk_size=1, inference default n_action_steps=1
  a1  ACT action chunking:      training chunk_size=20, inference default n_action_steps=5

Options:
  --steps=0          Save a verified zero-update initialization checkpoint.
  --steps=N          Run N optimizer updates (default: 20000).
  --policy-path=PATH Initialize a positive-step run from a compatible checkpoint.
  --print-command    Validate and print the final command without running it.
EOF
}

variant=""
variant_seen=false
steps="20000"
steps_seen=false
policy_path=""
policy_path_seen=false
print_command=false
print_command_seen=false

for argument in "$@"; do
  case "${argument}" in
    --variant=*)
      if [[ "${variant_seen}" == true ]]; then
        echo "error: --variant may be specified only once" >&2
        usage >&2
        exit 64
      fi
      variant="${argument#--variant=}"
      variant_seen=true
      ;;
    --steps=*)
      if [[ "${steps_seen}" == true ]]; then
        echo "error: --steps may be specified only once" >&2
        usage >&2
        exit 64
      fi
      steps="${argument#--steps=}"
      steps_seen=true
      ;;
    --policy-path=*)
      if [[ "${policy_path_seen}" == true ]]; then
        echo "error: --policy-path may be specified only once" >&2
        usage >&2
        exit 64
      fi
      policy_path="${argument#--policy-path=}"
      policy_path_seen=true
      ;;
    --print-command)
      if [[ "${print_command_seen}" == true ]]; then
        echo "error: --print-command may be specified only once" >&2
        usage >&2
        exit 64
      fi
      print_command=true
      print_command_seen=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: ${argument}" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ "${variant_seen}" != true ]]; then
  echo "error: --variant=a0|a1 is required" >&2
  usage >&2
  exit 64
fi
case "${variant}" in
  a0)
    variant_label="A0 single-step"
    chunk_size="1"
    n_action_steps="1"
    output_name="a0_single_step"
    ;;
  a1)
    variant_label="A1 action chunking"
    chunk_size="20"
    n_action_steps="5"
    output_name="a1_chunking"
    ;;
  *)
    echo "error: --variant must be a0 or a1: ${variant}" >&2
    exit 64
    ;;
esac

if [[ ! "${steps}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "error: --steps must be canonical decimal in [0, 2147483647]: ${steps}" >&2
  exit 64
fi
max_steps="2147483647"
if (( ${#steps} > ${#max_steps} )); then
  echo "error: --steps exceeds 2147483647: ${steps}" >&2
  exit 64
fi
steps_number=$((10#${steps}))
if (( steps_number > 2147483647 )); then
  echo "error: --steps exceeds 2147483647: ${steps}" >&2
  exit 64
fi
if [[ "${policy_path_seen}" == true && -z "${policy_path}" ]]; then
  echo "error: --policy-path must not be empty" >&2
  exit 64
fi
if (( steps_number == 0 )) && [[ "${policy_path_seen}" == true ]]; then
  echo "error: --policy-path cannot be used with --steps=0" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
dataset_root="${practice_root}/datasets/lerobot/lerobot/libero"
output_base="${practice_root}/outputs/act/${output_name}"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${output_base}/${timestamp}"
episodes="[823,826,834,840,841,845,847,869,875,878,910,918,920,929,944,948,949,953,966,972]"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi
if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
  echo "error: LIBERO dataset not found at ${dataset_root}" >&2
  exit 66
fi

if [[ "${policy_path_seen}" == true ]]; then
  if [[ ! -d "${policy_path}" ]]; then
    echo "error: policy checkpoint directory does not exist: ${policy_path}" >&2
    exit 66
  fi
  policy_path="$(cd -- "${policy_path}" && pwd -P)"
  "${conda_bin}" run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.act_initialization \
    validate-policy "--variant=${variant}" "--policy-path=${policy_path}"
  policy_source_arg="--policy.path=${policy_path}"
else
  policy_source_arg="--policy.type=act"
fi

if [[ "${print_command}" == true ]]; then
  if (( steps_number == 0 )); then
    printf '%q ' \
      "${conda_bin}" run --no-capture-output -n robot-practice \
      python -m robot_practice.experiments.act_initialization create \
      "--variant=${variant}" \
      "--output-dir=${output_dir}" \
      "--dataset-root=${dataset_root}"
  else
    printf '%q ' \
      "${conda_bin}" run --no-capture-output -n robot-practice \
      lerobot-train \
      "--dataset.repo_id=lerobot/libero" \
      "--dataset.root=${dataset_root}" \
      "--dataset.episodes=${episodes}" \
      "${policy_source_arg}" \
      "--policy.chunk_size=${chunk_size}" \
      "--policy.n_action_steps=${n_action_steps}" \
      "--policy.use_vae=false" \
      "--policy.push_to_hub=false" \
      "--batch_size=8" \
      "--steps=${steps_number}" \
      "--seed=1000" \
      "--eval_freq=0" \
      "--log_freq=$((steps_number < 100 ? steps_number : 100))" \
      "--save_checkpoint=true" \
      "--save_freq=${steps_number}" \
      "--wandb.enable=false" \
      "--output_dir=${output_dir}"
  fi
  printf '\n'
  exit 0
fi

if ! flock_bin="$(command -v flock)"; then
  echo "error: flock is not available on PATH" >&2
  exit 69
fi
mkdir -p -- "${output_base}"
if [[ ! -w "${output_base}" ]]; then
  echo "error: output base is not writable: ${output_base}" >&2
  exit 73
fi
if [[ -e "${output_dir}" ]]; then
  echo "error: output directory already exists: ${output_dir}" >&2
  exit 73
fi

lock_file="${output_base}/.launch.lock"
exec 9>"${lock_file}"
if ! "${flock_bin}" --nonblock 9; then
  echo "error: another ${variant_label} launch holds ${lock_file}" >&2
  exit 75
fi

echo "Starting fixed ${variant_label} experiment"
echo "Dataset episodes: ${episodes}"
echo "ACT config: chunk_size=${chunk_size}, n_action_steps=${n_action_steps}, use_vae=false"
echo "Training supervision horizon: chunk_size=${chunk_size} GT actions per sample"
echo "Checkpoint inference default: n_action_steps=${n_action_steps} (not used by training loss)"
if [[ "${policy_path_seen}" == true ]]; then
  echo "Initialization checkpoint: ${policy_path}"
fi
echo "Output directory: ${output_dir}"

if (( steps_number == 0 )); then
  exec "${conda_bin}" run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.act_initialization create \
    "--variant=${variant}" \
    "--output-dir=${output_dir}" \
    "--dataset-root=${dataset_root}"
fi

log_freq=$((steps_number < 100 ? steps_number : 100))
exec "${conda_bin}" run --no-capture-output -n robot-practice \
  lerobot-train \
  "--dataset.repo_id=lerobot/libero" \
  "--dataset.root=${dataset_root}" \
  "--dataset.episodes=${episodes}" \
  "${policy_source_arg}" \
  "--policy.chunk_size=${chunk_size}" \
  "--policy.n_action_steps=${n_action_steps}" \
  "--policy.use_vae=false" \
  "--policy.push_to_hub=false" \
  "--batch_size=8" \
  "--steps=${steps_number}" \
  "--seed=1000" \
  "--eval_freq=0" \
  "--log_freq=${log_freq}" \
  "--save_checkpoint=true" \
  "--save_freq=${steps_number}" \
  "--wandb.enable=false" \
  "--output_dir=${output_dir}"
