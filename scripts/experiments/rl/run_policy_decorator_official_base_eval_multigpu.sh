#!/usr/bin/env bash
# Run the pinned Policy Decorator five-environment evaluation for a frozen base.
# Each seed is an independent single-GPU process; GPUs are scheduled in waves.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_policy_decorator_official_base_eval_multigpu.sh \
    --gpus=0,1,2,3 \
    --base-policy-ckpt=PATH \
    [--reference-root=DIR] [--output-root=DIR] [--seeds=1,2,...,10]

Options:
  --gpus=LIST                 GPU ids, comma-separated.
  --base-policy-ckpt=PATH     Frozen official Diffusion Policy best.pt.
  --reference-root=DIR        Pinned Policy Decorator checkout.
  --output-root=DIR           Root for one output directory per seed.
  --seeds=LIST                Independent seeds; default: 1,2,3,4,5,6,7,8,9,10.
  --help                      Show this help.

This invokes the pinned upstream five-environment evaluate() path through a
local zero-residual adapter. It performs no SAC training and does not modify
the upstream checkout. The official eval contract is fixed to 50 episodes,
5 AsyncVectorEnv evaluation environments, state observations, and act_horizon=4.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

gpus=""
base_policy_ckpt=""
reference_root="${practice_root}/datasets/raw/policy_decorator/reference_repo"
output_root="${practice_root}/outputs/rl_three_stage_20260828/stage1/official_base_eval_paper10"
seeds_csv="1,2,3,4,5,6,7,8,9,10"

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus=*) gpus="${1#*=}"; shift ;;
    --base-policy-ckpt=*) base_policy_ckpt="${1#*=}"; shift ;;
    --reference-root=*) reference_root="${1#*=}"; shift ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --seeds=*) seeds_csv="${1#*=}"; shift ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

[[ -n "${gpus}" ]] || { echo "error: --gpus is required" >&2; exit 64; }
[[ -n "${base_policy_ckpt}" ]] || {
  echo "error: --base-policy-ckpt is required" >&2
  exit 64
}

resolve_path() {
  local candidate="$1"
  if [[ "${candidate}" != /* ]]; then
    candidate="${practice_root}/${candidate}"
  fi
  printf '%s/%s\n' "$(cd -- "$(dirname -- "${candidate}")" && pwd)" "$(basename -- "${candidate}")"
}

reference_root="$(resolve_path "${reference_root}")"
base_policy_ckpt="$(resolve_path "${base_policy_ckpt}")"
if [[ "${output_root}" != /* ]]; then
  output_root="${practice_root}/${output_root}"
fi
output_parent="$(dirname -- "${output_root}")"
mkdir -p "${output_parent}"
output_root="$(cd -- "${output_parent}" && pwd)/$(basename -- "${output_root}")"

[[ -f "${reference_root}/online/pi_dec_diffusion_maniskill2.py" ]] || {
  echo "error: missing pinned official script under ${reference_root}" >&2
  exit 66
}
[[ -s "${base_policy_ckpt}" ]] || {
  echo "error: missing/empty base checkpoint: ${base_policy_ckpt}" >&2
  exit 66
}
local_base_evaluator="${practice_root}/src/robot_practice/experiments/rl/policy_decorator_official_base_eval.py"
[[ -f "${local_base_evaluator}" ]] || {
  echo "error: missing local official-base evaluator" >&2
  exit 66
}
command -v conda >/dev/null 2>&1 || {
  echo "error: conda is not available on PATH" >&2
  exit 69
}

IFS=',' read -r -a gpu_ids <<< "${gpus}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"
(( ${#gpu_ids[@]} > 0 && ${#seeds[@]} > 0 )) || {
  echo "error: gpus/seeds cannot be empty" >&2
  exit 64
}
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

mkdir -p "${output_root}"
manifest="${output_root}/paper_protocol.txt"
launcher_log="${output_root}/launcher.log"
[[ ! -e "${manifest}" && ! -e "${launcher_log}" ]] || {
  echo "error: refusing to overwrite existing output manifest/log" >&2
  exit 73
}
for seed in "${seeds[@]}"; do
  [[ ! -e "${output_root}/seed_${seed}" && ! -e "${output_root}/seed_${seed}.launch.log" ]] || {
    echo "error: refusing to reuse existing seed output/log: seed=${seed}" >&2
    exit 73
  }
done

reference_revision="$(git -C "${reference_root}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
base_policy_sha256="$(sha256sum "${base_policy_ckpt}" | awk '{print $1}')"
{
  printf 'protocol=policy_decorator_official_base_eval\n'
  printf 'reference_root=%s\n' "${reference_root}"
  printf 'official_script=%s\n' "${reference_root}/online/pi_dec_diffusion_maniskill2.py"
  printf 'reference_revision=%s\n' "${reference_revision}"
  printf 'base_policy_ckpt=%s\n' "${base_policy_ckpt}"
  printf 'base_policy_sha256=%s\n' "${base_policy_sha256}"
  printf 'gpus=%s\nseeds=%s\n' "${gpus}" "${seeds_csv}"
  printf 'num_eval_envs=5\nnum_eval_episodes=50\nobs_mode=state\nact_horizon=4\n'
  printf 'residual_scale=0\nupstream_evaluate=true\n'
} > "${manifest}"

log() {
  printf '%s\n' "$*" | tee -a "${launcher_log}"
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  local seed_output="${output_root}/seed_${seed}"
  local seed_log="${output_root}/seed_${seed}.launch.log"
  mkdir -p "${seed_output}"
  log "[official-base] START seed=${seed} gpu=${gpu} output=${seed_output}"
  if CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="${practice_root}/src:${reference_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    conda run --no-capture-output -n maniskill2-pidec \
    python -m robot_practice.experiments.rl.policy_decorator_official_base_eval \
      --reference-root="${reference_root}" \
      --env-id=PegInsertionSide-v2 \
      --base-policy-ckpt="${base_policy_ckpt}" \
      --seed="${seed}" \
      --episodes=50 \
      --num-eval-envs=5 \
      --ddim-steps=4 \
      --act-horizon=4 \
      --output-dir="${seed_output}" \
    2>&1 | tee "${seed_log}"; then
    log "[official-base] DONE seed=${seed} gpu=${gpu}"
  else
    status=$?
    log "[official-base] FAILED seed=${seed} status=${status}"
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
    run_seed "${seeds[index]}" "${gpu_ids[index-offset]}" &
    pids+=("$!")
  done
  (
    while sleep 30; do
      log "[official-base] STATUS completed=${offset} total=${seed_count} active=${#pids[@]}"
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
  (( batch_status == 0 )) || { log "[official-base] FAILED batch offset=${offset}"; exit 1; }
  log "[official-base] STATUS completed=${batch_end} total=${seed_count}"
done

log "[official-base] DONE completed=${seed_count} total=${seed_count} output=${output_root}"
