#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: a_sanity.sh [--print-command]

Launch the five-episode ACT A-Sanity overfitting experiment.
Use --print-command to print the resolved command without starting training.
EOF
}

print_command=false
if (( $# > 1 )); then
  usage >&2
  exit 64
elif (( $# == 1 )); then
  if [[ "$1" != "--print-command" ]]; then
    usage >&2
    exit 64
  fi
  print_command=true
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

train_args=(
  "--dataset.repo_id=lerobot/libero"
  "--dataset.root=${dataset_root}"
  "--dataset.episodes=[823,826,834,840,841]"
  "--policy.type=act"
  "--policy.chunk_size=20"
  "--policy.n_action_steps=5"
  "--policy.use_vae=false"
  "--policy.push_to_hub=false"
  "--batch_size=8"
  "--steps=20000"
  "--seed=1000"
  "--eval_freq=0"
  "--log_freq=100"
  "--save_checkpoint=true"
  "--save_freq=20000"
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

echo "Starting ACT A-Sanity training"
echo "Dataset episodes: [823,826,834,840,841]"
echo "Output directory: ${output_dir}"

exec "${flock_bin}" \
  --nonblock \
  --no-fork \
  --verbose \
  --conflict-exit-code=75 \
  "${lock_file}" \
  "${command_args[@]}"
