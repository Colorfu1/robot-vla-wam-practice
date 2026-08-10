#!/usr/bin/env bash
#
# M0 PickCube formal Diffusion Policy training launcher.
#
# The --num-gpus/--num-gpu option belongs to this wrapper. It is translated
# into an explicit single-machine Accelerate launch; it must not be forwarded
# to lerobot-train, which does not define that option.
#
# The default per-GPU batch is 8. Therefore the formal 8-GPU command keeps
# the original smoke/global batch of 64 (8 GPUs x 8 samples per GPU).
#
# Example:
#   bash scripts/experiments/maniskill/train_formal.sh \
#     --num-gpus=8
#
# Print the exact command without starting training:
#   bash scripts/experiments/maniskill/train_formal.sh \
#     --num-gpus=8 --print-command

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_formal.sh --num-gpus=N [options]

Required:
  --num-gpus=N, --num-gpu=N
                          Number of visible GPUs/processes. The alias
                          --num-gpu is accepted for compatibility, but the
                          value must be supplied exactly once.

Dataset and output:
  --dataset-variant=full|trim
                          Default: full.
  --dataset-repo-id=ID   Default follows --dataset-variant.
  --dataset-root=DIR     Default follows --dataset-variant.
  --output-dir=DIR       Default: outputs/maniskill/m0_probe/train_50k.
  --job-name=NAME        Default: m0_pickcube_50k.

Training:
  --batch-size=N         Per-GPU batch size. Default: 8 (global batch 64 on
                          eight GPUs).
  --steps=N               Default: 50000.
  --save-freq=N           Default: 5000.
  --log-freq=N            Default: 200.
  --seed=N                Default: 1000.
  --num-workers=N         DataLoader workers per process. Default: 4.

Other:
  --print-command         Print the complete Accelerate command only.
  -h, --help              Show this help.

The launcher fixes the M0 policy/evaluation settings:
  Diffusion, CUDA, n_obs_steps=2, horizon=16, n_action_steps=8,
  push_to_hub=false, eval_freq=0, wandb=false.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

num_gpus=""
num_gpus_seen=0
dataset_variant="full"
dataset_repo_id=""
dataset_root=""
dataset_repo_id_explicit=false
dataset_root_explicit=false
output_dir="outputs/maniskill/m0_probe/train_50k"
job_name="m0_pickcube_50k"
batch_size=8
steps=50000
save_freq=5000
log_freq=200
seed=1000
num_workers=4
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
    --dataset-variant=*)
      dataset_variant="${argument#*=}"
      shift
      ;;
    --dataset-repo-id=*)
      dataset_repo_id="${argument#*=}"
      dataset_repo_id_explicit=true
      shift
      ;;
    --dataset-root=*)
      dataset_root="${argument#*=}"
      dataset_root_explicit=true
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
case "${dataset_variant}" in
  full|trim) ;;
  *)
    echo "error: --dataset-variant must be full or trim" >&2
    exit 64
    ;;
esac
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

if [[ "${dataset_repo_id_explicit}" == false ]]; then
  dataset_repo_id="local/maniskill_pickcube_rl_7d_${dataset_variant}"
fi
if [[ "${dataset_root_explicit}" == false ]]; then
  dataset_root="datasets/lerobot/maniskill/pickcube_rl_7d_${dataset_variant}"
fi

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

cd -- "${practice_root}"

training_arguments=(
  "--dataset.repo_id=${dataset_repo_id}"
  "--dataset.root=${dataset_root}"
  "--policy.type=diffusion"
  "--policy.device=cuda"
  "--policy.push_to_hub=false"
  "--policy.n_obs_steps=2"
  "--policy.horizon=16"
  "--policy.n_action_steps=8"
  "--output_dir=${output_dir}"
  "--job_name=${job_name}"
  "--seed=${seed}"
  "--batch_size=${batch_size}"
  "--num_workers=${num_workers}"
  "--steps=${steps}"
  "--save_freq=${save_freq}"
  "--log_freq=${log_freq}"
  "--eval_freq=0"
  "--wandb.enable=false"
)

launch_command=(
  env
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  "${conda_bin}" run --no-capture-output -n robot-practice
  accelerate launch
  "--mixed_precision=no"
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
launch_command+=(-m lerobot.scripts.lerobot_train "${training_arguments[@]}")

if [[ "${print_command}" == true ]]; then
  printf '%q ' "${launch_command[@]}"
  printf '\n'
  exit 0
fi

visible_gpus="$("${conda_bin}" run --no-capture-output -n robot-practice \
  python -c 'import torch; print(torch.cuda.device_count())')"
if ! nonnegative_integer "${visible_gpus}"; then
  echo "error: could not determine the visible CUDA device count: ${visible_gpus}" >&2
  exit 69
fi
if ((visible_gpus < num_gpus)); then
  echo "error: --num-gpus=${num_gpus} requested, but only ${visible_gpus} CUDA device(s) are visible" >&2
  echo "       check the container GPU allocation and CUDA_VISIBLE_DEVICES" >&2
  exit 69
fi

echo "M0 formal training: ${num_gpus} GPU(s), per-GPU batch=${batch_size}, global batch=$((batch_size * num_gpus))" >&2
echo "Dataset: ${dataset_root}" >&2
echo "Output: ${output_dir}" >&2
exec "${launch_command[@]}"
