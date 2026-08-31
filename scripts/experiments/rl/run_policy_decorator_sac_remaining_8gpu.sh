#!/usr/bin/env bash
# Run the remaining official Policy Decorator SAC seeds in one 8-GPU wave.
# This is only a thin CLI wrapper; the upstream entrypoint launcher is unchanged.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"

gpus="${POLICY_DECORATOR_GPUS:-0,1,2,3,4,5,6,7}"
seeds="${POLICY_DECORATOR_SEEDS:-5,6,7,8,9,10}"
base_policy_ckpt="${practice_root}/outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt"
reference_root="${practice_root}/datasets/raw/policy_decorator/reference_repo"
# Keep this separate from any partial seed_5..seed_10 directories left by the
# interrupted run; the official launcher intentionally refuses to overwrite.
output_root="${practice_root}/outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/sac_seed5_10_8gpu"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/rl/run_policy_decorator_sac_remaining_8gpu.sh [options]

Options:
  --gpus=LIST                 GPU ids; default: 0,1,2,3,4,5,6,7
  --seeds=LIST                Seeds; default: 5,6,7,8,9,10
  --base-policy-ckpt=PATH     Frozen Diffusion Policy best.pt
  --reference-root=DIR        Pinned Policy Decorator checkout
  --output-root=DIR           Fresh output root for these seeds
  --help

The script delegates to run_policy_decorator_paper_multigpu.sh and does not
modify the upstream entrypoint or the existing launcher. Six seeds means six
active processes; with eight GPUs, two GPUs remain idle.
EOF
}

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus=*) gpus="${1#*=}"; shift ;;
    --seeds=*) seeds="${1#*=}"; shift ;;
    --base-policy-ckpt=*) base_policy_ckpt="${1#*=}"; shift ;;
    --reference-root=*) reference_root="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ "${base_policy_ckpt}" != /* ]]; then
  base_policy_ckpt="${practice_root}/${base_policy_ckpt}"
fi
if [[ "${reference_root}" != /* ]]; then
  reference_root="${practice_root}/${reference_root}"
fi
if [[ "${output_root}" != /* ]]; then
  output_root="${practice_root}/${output_root}"
fi

test -s "${base_policy_ckpt}" || {
  echo "error: missing/empty base checkpoint: ${base_policy_ckpt}" >&2
  exit 66
}
test -f "${reference_root}/online/pi_dec_diffusion_maniskill2.py" || {
  echo "error: missing pinned official script under ${reference_root}" >&2
  exit 66
}

exec bash "${script_dir}/run_policy_decorator_paper_multigpu.sh" \
  --gpus="${gpus}" \
  --seeds="${seeds}" \
  --base-policy-ckpt="${base_policy_ckpt}" \
  --reference-root="${reference_root}" \
  --output-root="${output_root}"
