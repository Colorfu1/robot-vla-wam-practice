#!/usr/bin/env bash
#
# M2 StackCube native LeRobot Diffusion Policy launcher.
#
# The first phase materializes a separate train/development-only LeRobot image
# dataset. The second phase invokes the installed LeRobot trainer directly;
# no LeRobot or Diffusion Policy source file is patched by this wrapper.
#
# Formal 8-GPU run:
#   bash scripts/experiments/maniskill/train_m2_diffusion.sh --num-gpus=8
#
# One-step local smoke:
#   bash scripts/experiments/maniskill/train_m2_diffusion.sh \
#     --num-gpus=1 --batch-size=1 --steps=1 --save-freq=1 --log-freq=1 \
#     --num-workers=0 --mixed-precision=no \
#     --output-dir=/tmp/m2_stackcube_diffusion_smoke

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_m2_diffusion.sh --num-gpus=N [options]

Required:
  --num-gpus=N, --num-gpu=N
                          Number of visible GPUs/processes. Must be supplied
                          exactly once.

Adapter and data:
  --config=PATH           Default: configs/experiments/maniskill/m2_diffusion_policy.json.
  --m2-config=PATH        Default: configs/experiments/maniskill/m2_w_action_only_flow.json.
  --dataset-root=DIR      Adapted RGB dataset output; default comes from --config.
  --repo-id=ID            Adapted local LeRobot repo ID; default comes from --config.

Training:
  --output-dir=DIR        Default: outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000.
  --job-name=NAME         Default: m2_stackcube_diffusion_seed1000.
  --batch-size=N          Per-process batch size. Default: 32.
  --steps=N               Default: 50000.
  --save-freq=N           Default: 10000.
  --log-freq=N            Default: 200.
  --seed=N                Default: 1000.
  --num-workers=N         DataLoader workers per process. Default: 4.
  --mixed-precision=MODE  bf16 or no. Default: bf16.

Other:
  --print-command         Print preparation and native LeRobot commands only.
  -h, --help              Show this help.

The generated native config fixes:
  Diffusion, n_obs_steps=2, horizon=16, n_action_steps=8,
  train-only MEAN_STD stats, explicit train episode indices, eval_freq=0,
  wandb=false, push_to_hub=false.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

num_gpus=""
num_gpus_seen=0
adapter_config="configs/experiments/maniskill/m2_diffusion_policy.json"
m2_config="configs/experiments/maniskill/m2_w_action_only_flow.json"
dataset_root=""
repo_id=""
output_dir="outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000"
job_name="m2_stackcube_diffusion_seed1000"
batch_size=32
steps=50000
save_freq=10000
log_freq=200
seed=1000
num_workers=4
mixed_precision="bf16"
print_command=false

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

set_num_gpus() {
  if ((num_gpus_seen > 0)); then
    echo "error: duplicate --num-gpus/--num-gpu" >&2
    exit 64
  fi
  num_gpus="$1"
  num_gpus_seen=$((num_gpus_seen + 1))
}

while (($# > 0)); do
  argument="$1"
  case "${argument}" in
    -h|--help)
      usage
      exit 0
      ;;
    --print-command)
      print_command=true
      shift
      ;;
    --num-gpus|--num-gpu)
      if (($# < 2)); then
        echo "error: ${argument} requires a value" >&2
        exit 64
      fi
      set_num_gpus "$2"
      shift 2
      ;;
    --num-gpus=*|--num-gpu=*)
      set_num_gpus "${argument#*=}"
      shift
      ;;
    --config=*)
      adapter_config="${argument#*=}"
      shift
      ;;
    --m2-config=*)
      m2_config="${argument#*=}"
      shift
      ;;
    --dataset-root=*)
      dataset_root="${argument#*=}"
      shift
      ;;
    --repo-id=*)
      repo_id="${argument#*=}"
      shift
      ;;
    --output-dir=*)
      output_dir="${argument#*=}"
      shift
      ;;
    --job-name=*)
      job_name="${argument#*=}"
      shift
      ;;
    --batch-size=*)
      batch_size="${argument#*=}"
      shift
      ;;
    --steps=*)
      steps="${argument#*=}"
      shift
      ;;
    --save-freq=*)
      save_freq="${argument#*=}"
      shift
      ;;
    --log-freq=*)
      log_freq="${argument#*=}"
      shift
      ;;
    --seed=*)
      seed="${argument#*=}"
      shift
      ;;
    --num-workers=*)
      num_workers="${argument#*=}"
      shift
      ;;
    --mixed-precision=*)
      mixed_precision="${argument#*=}"
      shift
      ;;
    *)
      echo "error: unknown argument: ${argument}" >&2
      echo "Run with --help for the supported options." >&2
      exit 64
      ;;
  esac
done

if ((num_gpus_seen != 1)); then
  echo "error: --num-gpus=N is required exactly once" >&2
  exit 64
fi
positive_integer "${num_gpus}" || {
  echo "error: --num-gpus must be a positive integer" >&2
  exit 64
}
positive_integer "${batch_size}" || {
  echo "error: --batch-size must be a positive integer" >&2
  exit 64
}
positive_integer "${steps}" || {
  echo "error: --steps must be a positive integer" >&2
  exit 64
}
positive_integer "${save_freq}" || {
  echo "error: --save-freq must be a positive integer" >&2
  exit 64
}
positive_integer "${log_freq}" || {
  echo "error: --log-freq must be a positive integer" >&2
  exit 64
}
nonnegative_integer "${seed}" || {
  echo "error: --seed must be a non-negative integer" >&2
  exit 64
}
nonnegative_integer "${num_workers}" || {
  echo "error: --num-workers must be a non-negative integer" >&2
  exit 64
}
case "${mixed_precision}" in
  bf16|no) ;;
  *)
    echo "error: --mixed-precision must be bf16 or no" >&2
    exit 64
    ;;
esac

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"

train_config="outputs/maniskill/m2_diffusion/configs/${job_name}.train_config.json"

prepare_command=(
  env
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  "${conda_bin}" run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.maniskill.m2_diffusion prepare
  "--config=${adapter_config}"
  "--m2-config=${m2_config}"
  "--output-dir=${output_dir}"
  "--job-name=${job_name}"
  "--train-config=${train_config}"
  "--batch-size=${batch_size}"
  "--steps=${steps}"
  "--save-freq=${save_freq}"
  "--log-freq=${log_freq}"
  "--seed=${seed}"
  "--num-workers=${num_workers}"
  "--num-gpus=${num_gpus}"
)
if [[ -n "${dataset_root}" ]]; then
  prepare_command+=("--dataset-root=${dataset_root}")
fi
if [[ -n "${repo_id}" ]]; then
  prepare_command+=("--repo-id=${repo_id}")
fi

launch_command=(
  env
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  "${conda_bin}" run --no-capture-output -n robot-practice
  accelerate launch
  "--mixed_precision=${mixed_precision}"
  "--num_processes=${num_gpus}"
  --num_machines=1
  --machine_rank=0
  --gpu_ids=all
  --dynamo_backend=no
  --main_process_port=0
  --rdzv_backend=static
)
if ((num_gpus > 1)); then
  launch_command+=(--multi_gpu)
fi
launch_command+=(-m lerobot.scripts.lerobot_train "--config_path=${train_config}")

print_array() {
  printf '%q ' "$@"
  printf '\n'
}

if [[ "${print_command}" == true ]]; then
  echo "# Prepare the separate RGB LeRobot dataset and native train config:"
  print_array "${prepare_command[@]}"
  echo "# Launch the untouched native LeRobot Diffusion Policy pipeline:"
  print_array "${launch_command[@]}"
  exit 0
fi

if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "error: refusing existing training output directory: ${output_dir}" >&2
  echo "       choose a new --output-dir; no overwrite/resume is implicit" >&2
  exit 73
fi

visible_gpus="$("${conda_bin}" run --no-capture-output -n robot-practice \
  python -c 'import torch; print(torch.cuda.device_count())')"
if ! nonnegative_integer "${visible_gpus}"; then
  echo "error: could not determine the visible CUDA device count: ${visible_gpus}" >&2
  exit 69
fi
if ((visible_gpus < num_gpus)); then
  echo "error: --num-gpus=${num_gpus} requested, but only ${visible_gpus} CUDA device(s) are visible" >&2
  exit 69
fi

echo "Preparing M2 train/development-only RGB LeRobot dataset..." >&2
"${prepare_command[@]}"
echo "Starting native LeRobot Diffusion Policy: ${num_gpus} GPU(s), per-GPU batch=${batch_size}" >&2
exec "${launch_command[@]}"
