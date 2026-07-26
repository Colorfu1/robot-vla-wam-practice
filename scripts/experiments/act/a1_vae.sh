#!/usr/bin/env bash
# Purpose:
#   Train the A1-VAE controlled variant. It keeps the original LeRobot ACT loss,
#   chunk_size=20, and n_action_steps=5, and changes only the ACT VAE settings:
#   use_vae=true, latent_dim=32, kl_weight=10.0.
#
# Accepted arguments:
#   --steps=N          Positive optimizer update count; default: 20000.
#   --output-dir=DIR   Exact new output directory; default:
#                      outputs/act/a1_vae/<timestamp>.
#   --print-command    Validate and print the complete lerobot-train command only.
#   -h, --help         Show usage.
#
# Examples (run from the repository root):
#   scripts/experiments/act/a1_vae.sh --steps=20000
#   scripts/experiments/act/a1_vae.sh \
#     --steps=20000 \
#     --output-dir=outputs/act/a1_vae/run_20000_01
#   scripts/experiments/act/a1_vae.sh --steps=20000 --print-command

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: a1_vae.sh [--steps=N] [--output-dir=DIR] [--print-command]

Experiment:
  Standard LeRobot ACT chunk_size=20, n_action_steps=5.
  VAE is enabled explicitly with latent_dim=32 and kl_weight=10.0.

Options:
  --steps=N          Positive optimizer updates (default: 20000).
  --output-dir=DIR   Exact new result directory.
  --print-command    Print the resolved command without running training.
  -h, --help         Show this help.
EOF
}

steps="20000"
steps_seen=false
output_dir=""
output_dir_seen=false
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
    --output-dir=*)
      if [[ "${output_dir_seen}" == true ]]; then
        echo "error: --output-dir may be specified only once" >&2
        usage >&2
        exit 64
      fi
      output_dir="${argument#--output-dir=}"
      output_dir_seen=true
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

if [[ ! "${steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --steps must be a canonical positive integer: ${steps}" >&2
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
if [[ "${output_dir_seen}" == true && -z "${output_dir}" ]]; then
  echo "error: --output-dir must not be empty" >&2
  exit 64
fi

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi
if ! flock_bin="$(command -v flock)"; then
  echo "error: flock is not available on PATH" >&2
  exit 69
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
dataset_root="${practice_root}/datasets/lerobot/lerobot/libero"
output_base="${practice_root}/outputs/act/a1_vae"
timestamp="$(date +%Y%m%d_%H%M%S)"
episodes="[823,826,834,840,841,845,847,869,875,878,910,918,920,929,944,948,949,953,966,972]"

if [[ "${output_dir_seen}" != true ]]; then
  output_dir="${output_base}/${timestamp}"
elif [[ "${output_dir}" == "~/"* ]]; then
  output_dir="${HOME}/${output_dir#~/}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${practice_root}/${output_dir}"
fi

if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
  echo "error: LIBERO dataset metadata not found at ${dataset_root}/meta/info.json" >&2
  exit 66
fi

log_freq=$((steps_number < 100 ? steps_number : 100))
training_command=(
  "${conda_bin}" run --no-capture-output -n robot-practice
  lerobot-train
  "--dataset.repo_id=lerobot/libero"
  "--dataset.root=${dataset_root}"
  "--dataset.episodes=${episodes}"
  "--policy.type=act"
  "--policy.chunk_size=20"
  "--policy.n_action_steps=5"
  "--policy.use_vae=true"
  "--policy.latent_dim=32"
  "--policy.kl_weight=10.0"
  "--policy.push_to_hub=false"
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

if [[ "${print_command}" == true ]]; then
  printf '%q ' "${training_command[@]}"
  printf '\n'
  exit 0
fi

output_parent="$(dirname -- "${output_dir}")"
output_name="$(basename -- "${output_dir}")"
mkdir -p -- "${output_parent}"
if [[ ! -w "${output_parent}" ]]; then
  echo "error: output directory parent is not writable: ${output_parent}" >&2
  exit 73
fi
if [[ -e "${output_dir}" ]]; then
  echo "error: output directory already exists: ${output_dir}" >&2
  exit 73
fi

echo "Starting A1-VAE experiment"
echo "Dataset episodes: ${episodes}"
echo "Output directory: ${output_dir}"
echo "Policy: standard act"
echo "ACT config: chunk_size=20, n_action_steps=5"
echo "VAE config: use_vae=true, latent_dim=32, kl_weight=10.0"
echo "Loss: original ACT masked L1 + 10.0 * KLD"
echo "Training supervision horizon: chunk_size=20"
echo "Checkpoint inference default: n_action_steps=5 (not used by training loss)"

lock_file="${output_parent}/.${output_name}.launch.lock"
exec "${flock_bin}" \
  --nonblock \
  --no-fork \
  --verbose \
  --conflict-exit-code=75 \
  "${lock_file}" \
  "${training_command[@]}"
