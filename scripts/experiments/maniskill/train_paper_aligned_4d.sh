#!/usr/bin/env bash
# Train one paper-aligned arm with an explicit global-batch contract.
#
# A completed final checkpoint is reused.  An existing incomplete run is
# resumed from checkpoints/last instead of being deleted.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_paper_aligned_4d.sh --demo-count=1000|100 --num-gpus=N [options]

Options:
  --demo-count=N       1000 (all episodes) or 100 (episode ids 0..99)
  --num-gpus=N         Number of Accelerate processes; default is required explicitly
  --output-dir=DIR     Default: outputs/maniskill/paper_aligned_4d/train_<demo-count>
  --job-name=NAME      Default: paper4d_<demo-count>
  --batch-size=N       Per-GPU batch; default 32 (global 256 on 8 GPUs)
  --steps=N             Default 30000
  --save-freq=N         Default 2500
  --log-freq=N          Default 200
  --seed=N              Default 1000
  --num-workers=N       Default 4
  -h, --help            Show this help
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
demo_count=""
num_gpus=""
output_dir=""
job_name=""
batch_size=32
steps=30000
save_freq=2500
log_freq=200
seed=1000
num_workers=4

positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --demo-count=*) demo_count="${1#*=}"; shift ;;
    --num-gpus=*) num_gpus="${1#*=}"; shift ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --job-name=*) job_name="${1#*=}"; shift ;;
    --batch-size=*) batch_size="${1#*=}"; shift ;;
    --steps=*) steps="${1#*=}"; shift ;;
    --save-freq=*) save_freq="${1#*=}"; shift ;;
    --log-freq=*) log_freq="${1#*=}"; shift ;;
    --seed=*) seed="${1#*=}"; shift ;;
    --num-workers=*) num_workers="${1#*=}"; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 64 ;;
  esac
done

[[ "${demo_count}" == 100 || "${demo_count}" == 1000 ]] || {
  echo "error: --demo-count must be 100 or 1000" >&2
  exit 64
}
positive_integer "${num_gpus}" || { echo "error: --num-gpus is required" >&2; exit 64; }
for value in "${batch_size}" "${steps}" "${save_freq}" "${log_freq}" "${num_workers}"; do
  positive_integer "${value}" || { echo "error: numeric options must be positive" >&2; exit 64; }
done
nonnegative_integer "${seed}" || { echo "error: --seed must be non-negative" >&2; exit 64; }

if [[ -z "${output_dir}" ]]; then
  output_dir="outputs/maniskill/paper_aligned_4d/train_${demo_count}"
fi
if [[ -z "${job_name}" ]]; then
  job_name="paper4d_${demo_count}"
fi

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi
cd -- "${practice_root}"

final_policy="${output_dir}/checkpoints/$(printf '%06d' "${steps}")/pretrained_model"
if [[ -f "${final_policy}/config.json" && -f "${final_policy}/model.safetensors" ]]; then
  echo "paper4d training already complete: ${final_policy}"
  exit 0
fi

dataset_root="datasets/lerobot/maniskill/pickcube_mp_4d"
dataset_repo_id="local/maniskill_pickcube_mp_4d"
training_args=(
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

if [[ "${demo_count}" == 100 ]]; then
  episode_spec="["
  for ((episode = 0; episode < 100; episode++)); do
    if ((episode > 0)); then episode_spec+=","; fi
    episode_spec+="${episode}"
  done
  episode_spec+="]"
  training_args+=("--dataset.episodes=${episode_spec}")
fi

if [[ -d "${output_dir}" ]]; then
  resume_config="${output_dir}/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -f "${resume_config}" ]]; then
    echo "error: existing training directory has no resumable checkpoint: ${output_dir}" >&2
    echo "       refusing to delete it; inspect the directory and choose a recovery action" >&2
    exit 66
  fi
  training_args+=("--resume=true" "--config_path=${resume_config}")
  echo "resuming paper4d training from ${resume_config}"
fi

visible_gpus="$("${conda_bin}" run --no-capture-output -n robot-practice python -c 'import torch; print(torch.cuda.device_count())')"
if ! nonnegative_integer "${visible_gpus}" || ((visible_gpus < num_gpus)); then
  echo "error: requested ${num_gpus} GPUs, visible ${visible_gpus:-0}" >&2
  exit 69
fi

launch=(
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  "${conda_bin}" run --no-capture-output -n robot-practice
  accelerate launch --mixed_precision=no
  "--num_processes=${num_gpus}" --num_machines=1 --machine_rank=0
  --gpu_ids=all --dynamo_backend=no --main_process_port=0 --rdzv_backend=static
)
if ((num_gpus > 1)); then
  launch+=(--multi_gpu)
fi
launch+=(-m lerobot.scripts.lerobot_train "${training_args[@]}")

echo "paper4d training: demos=${demo_count}, GPUs=${num_gpus}, per_gpu_batch=${batch_size}, global_batch=$((batch_size * num_gpus))"
exec "${launch[@]}"
