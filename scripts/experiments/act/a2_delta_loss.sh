#!/usr/bin/env bash
# Purpose:
#   Launch the ACT-delta-loss experiment (A2) with fixed hyperparameters:
#   chunk_size=20, n_action_steps=5, use_vae=false, policy.type=act_delta,
#   delta_loss_weight=0.1.
#
# Accepted arguments:
#   --steps=N           Optimizer updates; default: 20000.
#   --source-policy=PATH Build or rebuild A2 step-0 checkpoint from this ACT checkpoint.
#   --policy-path=PATH   Use an existing act_delta step-0/step-N checkpoint as initialization.
#   --output-dir=DIR     Explicit training output directory.
#   --print-command       Print resolved commands without running them.
#   -h, --help           Show usage.
#
# Notes:
#   Without --source-policy/--policy-path, this launches a fresh act_delta init.
#   --policy-path and --source-policy are mutually exclusive for --steps>0.
#   chunk_size=20 controls the supervised training horizon. n_action_steps=5
#   is the checkpoint inference default and is not used to truncate training loss.
#   The final --output-dir must not already exist; the launcher lock is stored
#   beside it so LeRobot can create the final directory itself.
#   Every --log_freq updates, A2 loss components are printed and appended to
#   <output-dir>/train_loss_components.jsonl.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: a2_delta_loss.sh --steps=N [--source-policy=PATH | --policy-path=PATH] [--output-dir=DIR] [--print-command]

Required arguments for --steps=0:
  --source-policy=PATH   Optional source ACT checkpoint used to build act_delta step-0.
  If omitted, the experiment starts from random initialization (policy.type=act_delta).

Recommended arguments for --steps>0:
  --policy-path=PATH     Optional act_delta checkpoint to resume from, OR
  --source-policy=PATH   Build act_delta step-0 automatically, then resume from it.

Examples:
  scripts/experiments/act/a2_delta_loss.sh \
    --steps=0 \
    --source-policy=outputs/act/a1_chunking/20260724_120152/checkpoints/000000/pretrained_model

  scripts/experiments/act/a2_delta_loss.sh \
    --steps=20000 \
    --policy-path=outputs/act/a2_delta_loss/20260724_120152/checkpoints/000000/pretrained_model
EOF
}

steps="20000"
steps_seen=false
source_policy=""
source_policy_seen=false
policy_path=""
policy_path_seen=false
output_dir=""
output_dir_seen=false
print_command=false

for argument in "$@"; do
  case "${argument}" in
    --steps=*)
      if [[ "${steps_seen}" == true ]]; then
        echo "error: --steps may be specified only once" >&2
        usage >&2
        exit 64
      fi
      steps="${argument#--steps=}"
      steps_seen=true
      ;;
    --source-policy=*)
      if [[ "${source_policy_seen}" == true ]]; then
        echo "error: --source-policy may be specified only once" >&2
        usage >&2
        exit 64
      fi
      source_policy="${argument#--source-policy=}"
      source_policy_seen=true
      ;;
    --policy-path=*)
      if [[ "${policy_path_seen}" == true ]]; then
        echo "error: --policy-path may be specified only once" >&2
        usage >&2
        exit 64
      fi
      policy_path="${argument#--policy-path=}"
      policy_path_seen=true
      ;;
    --output-dir=*)
      if [[ "${output_dir_seen}" == true ]]; then
        echo "error: --output-dir may be specified only once" >&2
        usage >&2
        exit 64
      fi
      output_dir="${argument#--output-dir=}"
      output_dir_seen=true
      ;;
    --print-command)
      print_command=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: ${argument}" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if ! [[ "${steps}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "error: --steps must be canonical decimal in [0, 2147483647]: ${steps}" >&2
  exit 64
fi
max_steps="2147483647"
if (( ${#steps} > ${#max_steps} )); then
  echo "error: --steps exceeds 2147483647: ${steps}" >&2
  exit 64
fi
steps_number=$((10#${steps}))
if (( steps_number > 2147483647 )); then
  echo "error: --steps exceeds 2147483647: ${steps}" >&2
  exit 64
fi

if (( steps_number == 0 )) && [[ "${policy_path_seen}" == true ]]; then
  echo "error: --policy-path cannot be used with --steps=0" >&2
  exit 64
fi
if (( steps_number > 0 )) && [[ "${source_policy_seen}" == true && "${policy_path_seen}" == true ]]; then
  echo "error: --source-policy and --policy-path are mutually exclusive for --steps>0" >&2
  exit 64
fi

if [[ "${source_policy_seen}" == true ]] && [[ -z "${source_policy}" ]]; then
  echo "error: --source-policy must not be empty" >&2
  exit 64
fi
if [[ "${policy_path_seen}" == true ]] && [[ -z "${policy_path}" ]]; then
  echo "error: --policy-path must not be empty" >&2
  exit 64
fi
if [[ "${output_dir_seen}" == true ]] && [[ -z "${output_dir}" ]]; then
  echo "error: --output-dir must not be empty" >&2
  exit 64
fi

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

if ! flock_bin="$(command -v flock)"; then
  echo "error: flock is not available on PATH" >&2
  exit 69
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
dataset_root="${practice_root}/datasets/lerobot/lerobot/libero"
output_base="${practice_root}/outputs/act/a2_delta_loss"
timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ "${output_dir_seen}" == true ]]; then
  output_dir="${output_dir}"
else
  output_dir="${output_base}/${timestamp}"
fi
if [[ "${output_dir}" == ~/* ]]; then
  output_dir="${HOME}/${output_dir#~/}"
fi
if [[ "${output_dir}" != /* ]]; then
  output_dir="${practice_root}/${output_dir}"
fi
episodes="[823,826,834,840,841,845,847,869,875,878,910,918,920,929,944,948,949,953,966,972]"
policy_delta_weight="0.1"

if [[ ! -f "${dataset_root}/meta/info.json" ]]; then
  echo "error: LIBERO dataset not found at ${dataset_root}" >&2
  exit 66
fi

if [[ "${policy_path_seen}" == true ]]; then
  if [[ ! -d "${policy_path}" ]]; then
    echo "error: policy checkpoint directory does not exist: ${policy_path}" >&2
    exit 66
  fi
  policy_path="$(cd -- "${policy_path}" && pwd -P)"
fi

validate_policy_checkpoint() {
  local checkpoint_path="${1}"
  python - "${checkpoint_path}" <<'PY'
import json
import sys

checkpoint_path = sys.argv[1]
with open(f"{checkpoint_path}/config.json", "r", encoding="utf-8") as stream:
    config = json.load(stream)
if config.get("type") != "act_delta":
    raise SystemExit("ACT-delta checkpoint must have config type=act_delta")
if config.get("delta_loss_weight") != 0.1:
    raise SystemExit(
        f"Expected delta_loss_weight=0.1, got {config.get('delta_loss_weight')!r}"
    )
if config.get("chunk_size") != 20 or config.get("n_action_steps") != 5:
    raise SystemExit(
        "A2 checkpoint must have chunk_size=20 and n_action_steps=5 "
        f"(got chunk_size={config.get('chunk_size')!r}, n_action_steps={config.get('n_action_steps')!r})"
    )
PY
}

init_checkpoint="${output_dir}/checkpoints/000000/pretrained_model"
if [[ "${source_policy_seen}" == true ]]; then
  if [[ ! -d "${source_policy}" ]]; then
    echo "error: source policy directory does not exist: ${source_policy}" >&2
    exit 66
  fi
  source_policy="$(cd -- "${source_policy}" && pwd -P)"
fi

if [[ "${source_policy_seen}" == true ]]; then
  build_init_command=(
    "${conda_bin}" run --no-capture-output -n robot-practice
    python -m robot_practice.experiments.act_delta_checkpoint
    create
    "--source-policy=${source_policy}"
    "--output-dir=${output_dir}"
    "--delta-loss-weight=${policy_delta_weight}"
  )
fi

if (( steps_number == 0 )); then
  if [[ "${source_policy_seen}" == true ]]; then
    if [[ "${print_command}" == true ]]; then
      printf '%q ' "${build_init_command[@]}"
      printf '\n'
      exit 0
    fi
    exec "${build_init_command[@]}"
  fi
fi

if [[ "${source_policy_seen}" == true ]]; then
  if [[ "${print_command}" == true ]]; then
    printf '%q ' "${build_init_command[@]}"
    printf '\n'
  else
    "${build_init_command[@]}"
  fi
  policy_path="${init_checkpoint}"
fi

if [[ -n "${policy_path}" ]]; then
  policy_source_arg=( "--policy.path=${policy_path}" )
else
  policy_source_arg=( "--policy.type=act_delta" "--policy.delta_loss_weight=${policy_delta_weight}" )
fi

if [[ "${print_command}" == false ]] && [[ -n "${policy_path}" ]]; then
  validate_policy_checkpoint "${policy_path}"
fi

if [[ "${print_command}" == false ]]; then
  output_parent="$(dirname -- "${output_dir}")"
  if ! mkdir -p -- "${output_parent}"; then
    echo "error: unable to create output directory parent ${output_parent}" >&2
    exit 73
  fi
  if [[ ! -w "${output_parent}" ]]; then
    echo "error: output directory parent is not writable: ${output_parent}" >&2
    exit 73
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "error: output directory already exists: ${output_dir}" >&2
    exit 73
  fi
fi

if (( steps_number > 0 )); then
  log_freq=$((steps_number < 100 ? steps_number : 100))
  save_freq=${steps_number}
else
  log_freq=1
  save_freq=1
fi

training_command=(
  env
  "ROBOT_PRACTICE_A2_LOSS_LOG_PATH=${output_dir}/train_loss_components.jsonl"
  "ROBOT_PRACTICE_A2_LOSS_LOG_FREQ=${log_freq}"
  "${conda_bin}" run --no-capture-output -n robot-practice
  lerobot-train
  "--dataset.repo_id=lerobot/libero"
  "--dataset.root=${dataset_root}"
  "--dataset.episodes=${episodes}"
  "${policy_source_arg[@]}"
  "--policy.chunk_size=20"
  "--policy.n_action_steps=5"
  "--policy.use_vae=false"
  "--policy.push_to_hub=false"
  "--policy.discover_packages_path=robot_practice.policies.act_delta"
  "--batch_size=8"
  "--steps=${steps_number}"
  "--seed=1000"
  "--eval_freq=0"
  "--log_freq=${log_freq}"
  "--save_checkpoint=true"
  "--save_freq=${save_freq}"
  "--wandb.enable=false"
  "--output_dir=${output_dir}"
)

if [[ "${print_command}" == true ]]; then
  printf '%q ' "${training_command[@]}"
  printf '\n'
  exit 0
fi

output_parent="$(dirname -- "${output_dir}")"
output_name="$(basename -- "${output_dir}")"
lock_file="${output_parent}/.${output_name}.launch.lock"
if [[ "${print_command}" == false ]]; then
  echo "Starting A2 delta-loss experiment"
  echo "Dataset episodes: ${episodes}"
  echo "Output directory: ${output_dir}"
  if [[ "${source_policy_seen}" == true ]]; then
    echo "Source ACT checkpoint: ${source_policy}"
    echo "Created act_delta init at ${init_checkpoint}"
  elif [[ -n "${policy_path}" ]]; then
    echo "Initialization policy: ${policy_path}"
  else
    echo "Initialization: random act_delta weights (no policy-path/source-policy)"
  fi
  echo "ACT delta loss weight: ${policy_delta_weight}"
  echo "Training supervision horizon: chunk_size=20 GT actions per sample"
  echo "Checkpoint inference default: n_action_steps=5 (not used by training loss)"
  echo "A2 component loss log: ${output_dir}/train_loss_components.jsonl (every ${log_freq} steps)"
fi

if [[ "${print_command}" == false ]]; then
  exec "${flock_bin}" \
    --nonblock \
    --no-fork \
    --verbose \
    --conflict-exit-code=75 \
    "${lock_file}" \
    "${training_command[@]}"
fi
