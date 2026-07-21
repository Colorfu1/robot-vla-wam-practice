#!/usr/bin/env bash
# Trigger examples (run from the repository root):
# Create the zero-update B0 checkpoint:
#   scripts/experiments/act/a_sanity.sh --steps=0
# Replace <timestamp> with the directory printed by the B0 command:
#   b0="outputs/act/a_sanity/<timestamp>/checkpoints/000000/pretrained_model"
# Train for 20,000 updates from that exact B0 checkpoint:
#   scripts/experiments/act/a_sanity.sh --steps=20000 --policy-path="${b0}"
# Print that same training command without launching it:
#   scripts/experiments/act/a_sanity.sh --steps=20000 --policy-path="${b0}" --print-command
#
# Accepted arguments:
#   --steps=N           Number of optimizer updates (default: 20000).
#                       N must be canonical decimal in [0, 2147483647].
#                       N=0 creates the B0 checkpoint without training.
#   --policy-path=PATH  Use a complete compatible ACT checkpoint for steps > 0.
#                       Relative paths resolve from the caller's working directory.
#   --print-command     Validate and print the command without creating output.
#   -h, --help          Print usage and exit.
# Operational options are order-independent and may each appear only once.
# Unknown options are rejected.
#
# Fixed A-Sanity dataset selection (not CLI arguments):
#   repo_id=lerobot/libero
#   task_index=29
#   episodes=[823,826,834,840,841]  (5 episodes, 859 frames total)

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: a_sanity.sh [--steps=N] [--policy-path=PATH] [--print-command]

Run one independently triggered ACT A-Sanity operation.
  --steps=0          Save the fixed step-0 initialization without training.
  --steps=N          Run N optimizer updates (default: 20000).
  --policy-path=PATH Initialize a positive-step run from a complete ACT checkpoint.
  --print-command    Print the resolved command without creating output.
EOF
}

steps="20000"
steps_seen=false
policy_path=""
policy_path_seen=false
print_command=false
print_command_seen=false

for argument in "$@"; do
  case "${argument}" in
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
output_base="${practice_root}/outputs/act/a_sanity"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${output_base}/${timestamp}"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

if ! flock_bin="$(command -v flock)"; then
  echo "error: flock is not available on PATH" >&2
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
    python -m robot_practice.experiments.act_a_sanity \
    validate-policy "--policy-path=${policy_path}"
fi

if (( steps_number == 0 )); then
  command_args=(
    "${conda_bin}"
    run
    --no-capture-output
    -n
    robot-practice
    python
    -m
    robot_practice.experiments.act_a_sanity
    create
    "--output-dir=${output_dir}"
    "--dataset-root=${dataset_root}"
  )
else
  if (( steps_number < 100 )); then
    log_freq="${steps_number}"
  else
    log_freq="100"
  fi

  if [[ "${policy_path_seen}" == true ]]; then
    policy_args=(
      "--policy.path=${policy_path}"
      "--policy.chunk_size=20"
      "--policy.n_action_steps=5"
      "--policy.use_vae=false"
      "--policy.push_to_hub=false"
    )
  else
    policy_args=(
      "--policy.type=act"
      "--policy.chunk_size=20"
      "--policy.n_action_steps=5"
      "--policy.use_vae=false"
      "--policy.push_to_hub=false"
    )
  fi

  train_args=(
    "--dataset.repo_id=lerobot/libero"
    "--dataset.root=${dataset_root}"
    "--dataset.episodes=[823,826,834,840,841]"
    "${policy_args[@]}"
    "--batch_size=8"
    "--steps=${steps_number}"
    "--seed=1000"
    "--eval_freq=0"
    "--log_freq=${log_freq}"
    "--save_checkpoint=true"
    "--save_freq=${steps_number}"
    "--wandb.enable=false"
    "--output_dir=${output_dir}"
  )

  command_args=(
    "${conda_bin}"
    run
    --no-capture-output
    -n
    robot-practice
    lerobot-train
    "${train_args[@]}"
  )
fi

if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command_args[@]}"
  printf '\n'
  exit 0
fi

if ! mkdir -p -- "${output_base}"; then
  echo "error: unable to create output base ${output_base}" >&2
  exit 73
fi

if [[ ! -d "${output_base}" || ! -w "${output_base}" ]]; then
  echo "error: output base is not writable: ${output_base}" >&2
  exit 73
fi

if [[ -e "${output_dir}" ]]; then
  echo "error: output directory already exists: ${output_dir}" >&2
  exit 73
fi

lock_file="${output_base}/.launch.lock"

if (( steps_number == 0 )); then
  echo "Creating ACT A-Sanity step-0 checkpoint"
else
  echo "Starting ACT A-Sanity training for ${steps_number} updates"
fi
echo "Dataset episodes: [823,826,834,840,841]"
if [[ "${policy_path_seen}" == true ]]; then
  echo "Initialization checkpoint: ${policy_path}"
fi
echo "Output directory: ${output_dir}"

exec "${flock_bin}" \
  --nonblock \
  --no-fork \
  --verbose \
  --conflict-exit-code=75 \
  "${lock_file}" \
  "${command_args[@]}"
