#!/usr/bin/env bash
#
# One-command SmolVLA LIBERO C1 Default Non-PEFT training launcher.
#
# Formal run with the frozen global batch 8 (default layout 8x1; on fewer GPUs
# use e.g. --num-processes=4 --batch-size=2 so N*B stays 8):
#   bash scripts/experiments/smolvla_libero/train_c1.sh \
#     --mode=formal \
#     --dataset-root=datasets/lerobot/lerobot/libero \
#     --split-file=configs/experiments/smolvla_libero/target_split.json \
#     --output-dir=outputs/smolvla_libero/pilot/c1_default_non_peft
#
# Diagnostic C1 with 4/8 background replay and 4/8 target fine-tune data:
#   bash scripts/experiments/smolvla_libero/train_c1.sh \
#     --mode=diagnostic \
#     --experiment-tag=replay_k4 \
#     --background-replay=4 \
#     --dataset-root=datasets/lerobot/lerobot/libero \
#     --split-file=configs/experiments/smolvla_libero/target_split.json \
#     --num-processes=8 --batch-size=1 \
#     --steps=30000 --save-freq=5000 \
#     --learning-rate=0.0001 \
#     --output-dir=outputs/smolvla_libero/diagnostic/c1_replay_k4
#
# Validate and print the exact Accelerate command without training:
#   bash scripts/experiments/smolvla_libero/train_c1.sh --print-command
#
# One-GPU two-step smoke from the published selected B0 checkpoint:
#   bash scripts/experiments/smolvla_libero/train_c1.sh \
#     --mode=smoke \
#     --base-policy=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model \
#     --steps=2 \
#     --save-freq=2 \
#     --num-workers=0 \
#     --output-dir=outputs/smolvla_libero/debug/c1_smoke
#
# Resume the latest complete C1 checkpoint:
#   bash scripts/experiments/smolvla_libero/train_c1.sh \
#     --resume-from=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/005000

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_c1.sh --mode=smoke|formal|diagnostic --dataset-root=DIR --split-file=FILE \
    --output-dir=DIR [fresh options] [--print-command]
  train_c1.sh --resume-from=CHECKPOINT_DIR [--print-command]

Fresh options:
  --mode=smoke|formal|diagnostic
                          Required. diagnostic additionally requires
                          --experiment-tag and allows --learning-rate/--steps
                          overrides; its outputs are marked diagnostic and can
                          never be used as formal results.
  --experiment-tag=LABEL  Short diagnostic label; required only for
                          --mode=diagnostic.
  --background-replay=K   Diagnostic-only background samples per global batch
                          of 8 (0..7); target samples fill the remainder.
  --background-split-file=FILE
                          B0 background split; default:
                          configs/experiments/smolvla_libero/background_split.json.
  --dataset-root=DIR      Required.
  --split-file=FILE       Required.
  --output-dir=DIR        Required.
  --base-policy=DIR       B0 pretrained_model directory. Formal mode requires
                          selected_checkpoint.json provenance beside it. Default:
                          outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model.
  --num-processes=N       Formal default: 8; smoke: 1. Must not exceed the
                          visible GPUs; select cards with CUDA_VISIBLE_DEVICES.
  --batch-size=N          Per-GPU batch. Formal requires N x num-processes = 8
                          (legal layouts: 8x1, 4x2, 2x4, 1x8); smoke defaults to 1.
  --steps=N               Formal default: 30000; smoke: 2.
  --save-freq=N           Formal default: 5000; smoke: 2.
  --learning-rate=0.0001  Frozen optimizer peak learning rate.
  --train-seed=1000       Frozen training and sampler seed.
  --num-workers=N         DataLoader workers per process (default: 4).
  --mixed-precision=bf16  Required mixed precision.

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
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
)

inspection="$(
  "${offline_environment[@]}" \
    "${conda_bin}" run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.smolvla_libero.train_c1 \
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
  -m robot_practice.experiments.smolvla_libero.train_c1
  "${module_arguments[@]}"
)

if [[ "${print_only}" == "true" ]]; then
  printf '%q ' "${display_command[@]}"
  printf '\n'
  exit 0
fi

exec "${display_command[@]}"
