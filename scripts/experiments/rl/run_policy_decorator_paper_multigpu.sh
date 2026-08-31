#!/usr/bin/env bash
# Run the pinned Policy Decorator recipe for ten independent paper seeds.
#
# The upstream online script is intentionally kept single-GPU. This launcher
# accelerates the experiment by running independent seeds in waves, one
# process per GPU; it does not use torchrun/DDP or change the upstream code.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_policy_decorator_paper_multigpu.sh \
    --gpus=0,1,2,3 \
    --base-policy-ckpt=PATH \
    [--reference-root=DIR] [--output-root=DIR] [--seeds=1,2,...,10]

Options:
  --gpus=LIST                 Physical/local GPU ids, comma-separated.
  --base-policy-ckpt=PATH     Frozen official Diffusion Policy best.pt.
  --reference-root=DIR        Pinned Policy Decorator checkout.
  --output-root=DIR           Root for one output directory per seed.
  --seeds=LIST                Independent seeds; default: 1,2,3,4,5,6,7,8,9,10.
  --help                      Show this help.

This command is fixed to the paper's Diffusion Policy PegInsertionSide recipe.
The paper specifies ten seeds but does not publish a canonical seed-id list, so
the default explicit list 1..10 is recorded in the manifest:
  env=PegInsertionSide-v2, res_scale=0.1, prog_explore=30000,
  total_timesteps=4000000, num_envs=16, eval_freq=50000,
  num_eval_episodes=50, num_eval_envs=5, save_freq=1000000.
Other upstream defaults are left unchanged. Each seed is a full upstream
residual-SAC run, including its periodic evaluation and final checkpoint.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

gpus=""
base_policy_ckpt=""
reference_root="${practice_root}/datasets/raw/policy_decorator/reference_repo"
output_root="${practice_root}/outputs/rl_three_stage_20260828/stage1/official_paper10"
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
official_script="${reference_root}/online/pi_dec_diffusion_maniskill2.py"

[[ -f "${official_script}" ]] || {
  echo "error: missing pinned official script: ${official_script}" >&2
  exit 66
}
[[ -s "${base_policy_ckpt}" ]] || {
  echo "error: missing/empty base checkpoint: ${base_policy_ckpt}" >&2
  exit 66
}
reference_revision="$(git -C "${reference_root}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
base_policy_sha256="$(sha256sum "${base_policy_ckpt}" | awk '{print $1}')"

IFS=',' read -r -a gpu_ids <<< "${gpus}"
(( ${#gpu_ids[@]} > 0 )) || { echo "error: --gpus is empty" >&2; exit 64; }
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "error: GPU ids must be non-negative integers: ${gpu}" >&2
    exit 64
  }
  [[ -z "${seen_gpus[${gpu}]+x}" ]] || {
    echo "error: duplicate GPU id: ${gpu}" >&2
    exit 64
  }
  seen_gpus[${gpu}]=1
done

IFS=',' read -r -a seeds <<< "${seeds_csv}"
(( ${#seeds[@]} > 0 )) || { echo "error: --seeds is empty" >&2; exit 64; }
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || {
    echo "error: seeds must be non-negative integers: ${seed}" >&2
    exit 64
  }
  [[ -z "${seen_seeds[${seed}]+x}" ]] || {
    echo "error: duplicate seed: ${seed}" >&2
    exit 64
  }
  seen_seeds[${seed}]=1
done

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

mkdir -p "${output_root}"
manifest="${output_root}/paper_protocol.txt"
if [[ -e "${manifest}" ]]; then
  echo "error: refusing to overwrite existing manifest: ${manifest}" >&2
  exit 73
fi
launcher_log="${output_root}/launcher.log"
if [[ -e "${launcher_log}" ]]; then
  echo "error: refusing to overwrite existing launcher log: ${launcher_log}" >&2
  exit 73
fi
for seed in "${seeds[@]}"; do
  if [[ -e "${output_root}/seed_${seed}" || -e "${output_root}/seed_${seed}.launch.log" ]]; then
    echo "error: refusing to reuse existing seed output/log: seed=${seed}" >&2
    exit 73
  fi
done
{
  printf 'protocol=policy_decorator_paper_diffusion_peg_insertion\n'
  printf 'reference_root=%s\n' "${reference_root}"
  printf 'official_script=%s\n' "${official_script}"
  printf 'reference_revision=%s\n' "${reference_revision}"
  printf 'base_policy_ckpt=%s\n' "${base_policy_ckpt}"
  printf 'base_policy_sha256=%s\n' "${base_policy_sha256}"
  printf 'gpus=%s\n' "${gpus}"
  printf 'seeds=%s\n' "${seeds_csv}"
  printf 'seed_ids_note=paper reports ten seeds; canonical ids are not published\n'
  printf 'total_timesteps=4000000\nnum_envs=16\neval_freq=50000\n'
  printf 'num_eval_episodes=50\nnum_eval_envs=5\nsave_freq=1000000\n'
  printf 'res_scale=0.1\nprog_explore=30000\n'
} > "${manifest}"

log() {
  printf '%s\n' "$*" | tee -a "${launcher_log}"
}

log "[paper10] protocol=policy_decorator_paper_diffusion_peg_insertion seeds=${seeds_csv} gpus=${gpus}"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local seed_output="${output_root}/seed_${seed}"
  local seed_log="${output_root}/seed_${seed}.launch.log"

  if [[ -e "${seed_output}" || -e "${seed_log}" ]]; then
    echo "error: refusing to reuse existing seed output/log: seed=${seed}" >&2
    return 73
  fi
  mkdir -p "${seed_output}"
  log "[paper10] START seed=${seed} gpu=${gpu} output=${seed_output}"

  # One visible CUDA device per upstream process. The forkserver env workers
  # inherit this mask; no process in this launcher sees another GPU.
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="${reference_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  conda run --no-capture-output -n maniskill2-pidec \
    python "${official_script}" \
      --env-id PegInsertionSide-v2 \
      --base-policy-ckpt "${base_policy_ckpt}" \
      --res-scale 0.1 \
      --prog-explore 30000 \
      --total-timesteps 4000000 \
      --num-envs 16 \
      --training-freq 64 \
      --eval-freq 50000 \
      --num-eval-episodes 50 \
      --num-eval-envs 5 \
      --save-freq 1000000 \
      --seed "${seed}" \
      --output-dir "${seed_output}" \
      --track=false \
      --capture-video=false \
    2>&1 | tee "${seed_log}"
  log "[paper10] DONE seed=${seed} gpu=${gpu}"
}

gpu_count="${#gpu_ids[@]}"
seed_count="${#seeds[@]}"
for ((offset = 0; offset < seed_count; offset += gpu_count)); do
  pids=()
  batch_end=$((offset + gpu_count))
  (( batch_end > seed_count )) && batch_end="${seed_count}"
  for ((index = offset; index < batch_end; index++)); do
    gpu_slot=$((index - offset))
    run_seed "${seeds[index]}" "${gpu_ids[gpu_slot]}" &
    pids+=("$!")
  done

  # Keep a structured heartbeat in the persistent launcher log while a batch
  # is running. The upstream worker logs remain the source of per-step detail.
  (
    while sleep 30; do
      log "[paper10] STATUS completed=${offset} total=${seed_count} active=${#pids[@]}"
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
  if (( batch_status != 0 )); then
    log "[paper10] FAILED batch offset=${offset}; inspect seed_*.launch.log" >&2
    exit 1
  fi
  completed="${batch_end}"
  log "[paper10] STATUS completed=${completed} total=${seed_count}"
done

log "[paper10] DONE completed=${seed_count} total=${seed_count} output=${output_root}"
