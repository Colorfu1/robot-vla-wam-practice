#!/usr/bin/env bash
#
# Train the SmolVLA LIBERO B0 domain-adaptation policy with Accelerate, or
# resume the latest complete checkpoint of an existing B0 run.
#
# Fresh formal run:
#   bash scripts/experiments/smolvla_libero/train_b0.sh \
#     --mode=formal \
#     --base-policy=policies/smolvla_or_pi0/smolvla_base \
#     --dataset-root=datasets/lerobot/lerobot/libero \
#     --split-file=configs/experiments/smolvla_libero/background_split.json \
#     --num-processes=8 \
#     --batch-size=8 \
#     --steps=30000 \
#     --save-freq=5000 \
#     --learning-rate=0.0001 \
#     --train-seed=1000 \
#     --num-workers=4 \
#     --mixed-precision=bf16 \
#     --output-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation
#
# Resume:
#   bash scripts/experiments/smolvla_libero/train_b0.sh \
#     --resume-from=outputs/smolvla_libero/pilot/b0_domain_adaptation/checkpoints/005000
#
# Add --print-command to validate the arguments and print the complete
# Accelerate command without starting training.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_b0.sh --mode=smoke|formal --dataset-root=DIR --split-file=FILE \
    --output-dir=DIR [fresh options] [--print-command]

  train_b0.sh --resume-from=CHECKPOINT_DIR [--print-command]

Fresh options:
  --base-policy=REF        Local pretrained policy (default:
                           policies/smolvla_or_pi0/smolvla_base). Hub IDs are
                           rejected: B0 loading and training are offline-only.
  --num-processes=N        Accelerate process/GPU count (formal: 8; smoke: 1).
  --batch-size=N           Per-process batch size (formal: 8; smoke: 1).
  --steps=N                Optimizer updates (formal: 30000; smoke: 2).
  --save-freq=N            Checkpoint interval (formal: 5000; smoke: 2).
  --learning-rate=FLOAT    AdamW peak learning rate (default: 0.0001).
  --train-seed=N           Training seed (default: 1000).
  --num-workers=N          DataLoader workers per process (default: 4).
  --mixed-precision=bf16   Required mixed precision.

Resume accepts no fresh options. GPU count and BF16 are restored from the
original run_config.json. CHECKPOINT_DIR must be the latest complete checkpoint.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

print_only=false
print_count=0
resume_seen=false
forwarded=()
for argument in "$@"; do
  case "${argument}" in
    -h|--help)
      usage
      exit 0
      ;;
    --print-command)
      print_count=$((print_count + 1))
      print_only=true
      ;;
    --resume-from|--resume-from=*)
      resume_seen=true
      forwarded+=("${argument}")
      ;;
    *)
      forwarded+=("${argument}")
      ;;
  esac
done

if ((print_count > 1)); then
  echo "error: --print-command may be specified only once" >&2
  exit 64
fi

cd -- "${practice_root}"

offline_environment=(
  env
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
)

inspection="$(
  "${offline_environment[@]}" \
    "${conda_bin}" run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.smolvla_libero.train_b0 \
    "${forwarded[@]}" \
    --inspect-launch
)"
if [[ "${inspection}" == *$'\n'* ]]; then
  echo "error: invalid multi-line launch inspection: ${inspection}" >&2
  exit 70
fi

IFS=$'\t' read -r num_processes mixed_precision unexpected <<<"${inspection}"
if [[ ! "${num_processes}" =~ ^[1-9][0-9]*$ ]] \
  || [[ "${mixed_precision}" != "bf16" ]] \
  || [[ -n "${unexpected:-}" ]]; then
  echo "error: invalid launch inspection: ${inspection}" >&2
  exit 70
fi

module_arguments=("${forwarded[@]}")
if [[ "${resume_seen}" == "true" ]]; then
  module_arguments+=(
    "--num-processes=${num_processes}"
    "--mixed-precision=${mixed_precision}"
  )
fi

display_command=(
  "${offline_environment[@]}"
  conda run --no-capture-output -n robot-practice
  accelerate launch
  "--mixed_precision=${mixed_precision}"
  "--num_processes=${num_processes}"
  "--num_machines=1"
  "--machine_rank=0"
  "--gpu_ids=all"
  "--dynamo_backend=no"
  "--main_process_port=0"
  "--rdzv_backend=static"
)
if ((num_processes > 1)); then
  display_command+=(--multi_gpu)
fi
display_command+=(
  -m robot_practice.experiments.smolvla_libero.train_b0
  "${module_arguments[@]}"
)

if [[ "${print_only}" == "true" ]]; then
  printf '%q ' "${display_command[@]}"
  printf '\n'
  exit 0
fi

exec "${display_command[@]}"
