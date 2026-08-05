#!/usr/bin/env bash
#
# One-command SmolVLA LIBERO C3 LoRA + Paraphrase training launcher.
#
# Formal 8-GPU run with the frozen defaults:
#   bash scripts/experiments/smolvla_libero/train_c3.sh
#
# Validate and print the exact Accelerate command without training:
#   bash scripts/experiments/smolvla_libero/train_c3.sh --print-command
#
# One-GPU two-step smoke from a B0 checkpoint (selection provenance is only
# mandatory for formal mode):
#   bash scripts/experiments/smolvla_libero/train_c3.sh \
#     --mode=smoke \
#     --base-policy=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/030000/pretrained_model \
#     --steps=2 \
#     --save-freq=2 \
#     --num-workers=0 \
#     --output-dir=outputs/smolvla_libero/debug/c3_smoke
#
# Resume the latest complete C3 checkpoint:
#   bash scripts/experiments/smolvla_libero/train_c3.sh \
#     --resume-from=outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoints/005000

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_c3.sh [--print-command]
  train_c3.sh --mode=smoke|formal [fresh options] [--print-command]
  train_c3.sh --resume-from=CHECKPOINT_DIR [--print-command]

With no arguments, the launcher starts the frozen formal C3 run:
  selected B0 checkpoint, target Train split, 25% canonical / 75% reviewed
  train paraphrases, rank-32 LoRA, 8 GPUs, batch 1/GPU, BF16, 30000 steps,
  and checkpoints every 5000 steps.

Fresh options:
  --base-policy=DIR            B0 pretrained_model directory. Formal mode
                               requires selected_checkpoint.json provenance.
  --dataset-root=DIR           Default: datasets/lerobot/lerobot/libero.
  --split-file=FILE            Default: target_split.json.
  --paraphrases=FILE           Default: paraphrases.json bound by split SHA-256.
  --canonical-probability=0.25 Frozen canonical instruction probability.
  --num-processes=N            Formal defaults to 8; smoke to 1. Must not exceed
                               the visible GPUs; select cards with
                               CUDA_VISIBLE_DEVICES.
  --batch-size=N               Per-GPU batch. Formal requires N x num-processes = 8
                               (legal layouts: 8x1, 4x2, 2x4, 1x8); smoke defaults
                               to 1.
  --steps=N                    Formal: 30000; smoke: 2.
  --save-freq=N                Formal: 5000; smoke: 2.
  --learning-rate=0.0001       Frozen optimizer peak learning rate.
  --lora-rank=32               Frozen LoRA rank shared with C2.
  --train-seed=1000            Frozen data and instruction sampling seed.
  --num-workers=N              DataLoader workers per process (default: 4).
  --mixed-precision=bf16       Required mixed precision.
  --output-dir=DIR             Default formal output: c3_paraphrase_lora.

Resume accepts no fresh options. GPU count, BF16, paraphrase provenance, and
instruction-count snapshots are restored from the original run. CHECKPOINT_DIR
must be the latest complete checkpoint.
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
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
)

inspection="$(
  "${offline_environment[@]}" \
    "${conda_bin}" run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.smolvla_libero.train_c3 \
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
  -m robot_practice.experiments.smolvla_libero.train_c3
  "${module_arguments[@]}"
)

if [[ "${print_only}" == "true" ]]; then
  printf '%q ' "${display_command[@]}"
  printf '\n'
  exit 0
fi

exec "${display_command[@]}"
