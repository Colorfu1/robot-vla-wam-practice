#!/usr/bin/env bash
# Base-only evaluation followed by the official Policy Decorator SAC runs.
# Each seed is an independent single-GPU process; GPUs are scheduled in waves.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_policy_decorator_base_then_sac_multigpu.sh [options]

Options:
  --gpus=LIST                 GPU ids; default: 0,1,2,3 (or $POLICY_DECORATOR_GPUS)
  --seeds=LIST                Seeds; default: 1,2,3,4,5,6,7,8,9,10
  --base-policy-ckpt=PATH     Frozen official Diffusion Policy best.pt
  --reference-root=DIR        Pinned Policy Decorator checkout
  --output-root=DIR           Root containing base/ and sac/ results
  --help

The base phase reuses the pinned upstream five-environment evaluate() path
through the local zero-residual adapter (50 episodes/seed). Only after all base
seeds pass and the checkpoint hash is unchanged does the script call
run_policy_decorator_paper_multigpu.sh for official SAC + eval.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

gpus="${POLICY_DECORATOR_GPUS:-0,1,2,3}"
seeds_csv="1,2,3,4,5,6,7,8,9,10"
base_policy_ckpt="${practice_root}/outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt"
reference_root="${practice_root}/datasets/raw/policy_decorator/reference_repo"
output_root="${practice_root}/outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac"

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus=*) gpus="${1#*=}"; shift ;;
    --seeds=*) seeds_csv="${1#*=}"; shift ;;
    --base-policy-ckpt=*) base_policy_ckpt="${1#*=}"; shift ;;
    --reference-root=*) reference_root="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

resolve_existing() {
  local candidate="$1"
  [[ "${candidate}" == /* ]] || candidate="${practice_root}/${candidate}"
  printf '%s/%s\n' "$(cd -- "$(dirname -- "${candidate}")" && pwd)" "$(basename -- "${candidate}")"
}

reference_root="$(resolve_existing "${reference_root}")"
base_policy_ckpt="$(resolve_existing "${base_policy_ckpt}")"
[[ "${output_root}" == /* ]] || output_root="${practice_root}/${output_root}"
output_parent="$(dirname -- "${output_root}")"
mkdir -p "${output_parent}"
output_root="$(cd -- "${output_parent}" && pwd)/$(basename -- "${output_root}")"

official_script="${reference_root}/online/pi_dec_diffusion_maniskill2.py"
base_evaluator="${practice_root}/src/robot_practice/experiments/rl/policy_decorator_official_base_eval.py"
[[ -f "${official_script}" ]] || { echo "error: missing ${official_script}" >&2; exit 66; }
[[ -f "${base_evaluator}" ]] || { echo "error: missing ${base_evaluator}" >&2; exit 66; }
[[ -s "${base_policy_ckpt}" ]] || { echo "error: missing/empty ${base_policy_ckpt}" >&2; exit 66; }
command -v conda >/dev/null 2>&1 || { echo "error: conda is not available" >&2; exit 69; }

IFS=',' read -r -a gpu_ids <<< "${gpus}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"
(( ${#gpu_ids[@]} > 0 && ${#seeds[@]} > 0 )) || { echo "error: gpus/seeds cannot be empty" >&2; exit 64; }
declare -A seen_gpus=() seen_seeds=()
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "error: invalid GPU id: ${gpu}" >&2; exit 64; }
  [[ -z "${seen_gpus[${gpu}]+x}" ]] || { echo "error: duplicate GPU id: ${gpu}" >&2; exit 64; }
  seen_gpus[${gpu}]=1
done
for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "error: invalid seed: ${seed}" >&2; exit 64; }
  [[ -z "${seen_seeds[${seed}]+x}" ]] || { echo "error: duplicate seed: ${seed}" >&2; exit 64; }
  seen_seeds[${seed}]=1
done

base_root="${output_root}/base"
sac_root="${output_root}/sac"
if [[ -e "${base_root}" || -e "${sac_root}" ]]; then
  echo "error: refusing to reuse output; choose a new --output-root" >&2
  exit 73
fi
mkdir -p "${base_root}"
base_log="${base_root}/launcher.log"
log() { printf '%s\n' "$*" | tee -a "${base_log}"; }
base_sha256="$(sha256sum "${base_policy_ckpt}" | awk '{print $1}')"
log "[base-then-sac] START base seeds=${seeds_csv} gpus=${gpus}"

run_base_seed() {
  local seed="$1"
  local gpu="$2"
  local output="${base_root}/seed_${seed}"
  local seed_log="${base_root}/seed_${seed}.launch.log"
  mkdir "${output}"
  log "[base] START seed=${seed} gpu=${gpu}"
  if CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${practice_root}/src:${reference_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    conda run --no-capture-output -n maniskill2-pidec \
    python -m robot_practice.experiments.rl.policy_decorator_official_base_eval \
      --reference-root="${reference_root}" \
      --env-id=PegInsertionSide-v2 \
      --base-policy-ckpt="${base_policy_ckpt}" \
      --seed="${seed}" --episodes=50 --num-eval-envs=5 \
      --ddim-steps=4 --act-horizon=4 --output-dir="${output}" \
    2>&1 | tee "${seed_log}"; then
    log "[base] DONE seed=${seed} gpu=${gpu}"
  else
    status=$?
    log "[base] FAILED seed=${seed} status=${status}"
    return "${status}"
  fi
}

gpu_count="${#gpu_ids[@]}"
seed_count="${#seeds[@]}"
for ((offset=0; offset<seed_count; offset+=gpu_count)); do
  pids=()
  batch_end=$((offset + gpu_count))
  (( batch_end > seed_count )) && batch_end="${seed_count}"
  for ((index=offset; index<batch_end; index++)); do
    run_base_seed "${seeds[index]}" "${gpu_ids[index-offset]}" &
    pids+=("$!")
  done
  (
    while sleep 30; do
      log "[base] STATUS completed=${offset} total=${seed_count} active=${#pids[@]}"
    done
  ) &
  heartbeat_pid="$!"
  batch_status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      batch_status=1
    fi
  done
  kill "${heartbeat_pid}" 2>/dev/null || true
  wait "${heartbeat_pid}" 2>/dev/null || true
  (( batch_status == 0 )) || { log "[base] FAILED batch offset=${offset}"; exit 1; }
  log "[base] STATUS completed=${batch_end} total=${seed_count}"
done

for seed in "${seeds[@]}"; do
  test -s "${base_root}/seed_${seed}/metrics.json"
done
test "$(sha256sum "${base_policy_ckpt}" | awk '{print $1}')" = "${base_sha256}"
log "[base] DONE all seeds; checkpoint hash unchanged"

gpu_csv="$(IFS=,; printf '%s' "${gpu_ids[*]}")"
bash "${script_dir}/run_policy_decorator_paper_multigpu.sh" \
  --gpus="${gpu_csv}" --seeds="${seeds_csv}" \
  --base-policy-ckpt="${base_policy_ckpt}" \
  --reference-root="${reference_root}" --output-root="${sac_root}"

log "[base-then-sac] DONE base=${base_root} sac=${sac_root}"
