#!/usr/bin/env bash
# Probe one LIBERO task through the LeRobot FastWAM evaluator.
#
# Exists so a single-task A/B does not have to be typed as a long shell one-liner.
# Hand-written `lerobot-eval` invocations kept omitting parts of the W0 runtime
# environment (notably HF_HUB_OFFLINE and W0_FASTWAM_MODEL_ROOT), which made the
# child reach for Hugging Face and retry over an unreachable network instead of
# loading the repository-local Wan components.  Every variable the launcher's
# `_model_runtime_env` sets is applied here too.
#
# Usage:
#   run_w0_lr_task_probe.sh --task-id=5 --num-steps-wait=5 --gpu=0
#   run_w0_lr_task_probe.sh --task-id=5 --num-steps-wait=10 --gpu=1 --dry-run
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"

suite="libero_spatial"
task_id=""
num_steps_wait=""
gpu="0"
episodes="50"
obs_size="224"
episode_length=""
seed="42"
policy_path=""
output_dir=""
vendor_root=""
dtype="float32"
dry_run="0"

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for argument in "$@"; do
  case "${argument}" in
    --suite=*) suite="${argument#*=}" ;;
    --task-id=*) task_id="${argument#*=}" ;;
    --num-steps-wait=*) num_steps_wait="${argument#*=}" ;;
    --gpu=*) gpu="${argument#*=}" ;;
    --episodes=*) episodes="${argument#*=}" ;;
    --obs-size=*) obs_size="${argument#*=}" ;;
    --episode-length=*) episode_length="${argument#*=}" ;;
    --seed=*) seed="${argument#*=}" ;;
    --policy-path=*) policy_path="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --vendor-root=*) vendor_root="${argument#*=}" ;;
    --dtype=*) dtype="${argument#*=}" ;;
    --dry-run) dry_run="1" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${task_id}" ]]; then
  echo "error: --task-id is required" >&2
  exit 2
fi

# libero_10/libero_90 run twice as long per episode; keep the frozen protocol.
if [[ -z "${episode_length}" ]]; then
  case "${suite}" in
    libero_10|libero_90) episode_length="600" ;;
    *) episode_length="300" ;;
  esac
fi

# Default to the VISUAL=IDENTITY overlay produced by the sharded LR-REL run; fall
# back to the released checkpoint so the script still works standalone.
if [[ -z "${policy_path}" ]]; then
  overlay="${repo_root}/outputs/literature/w0_fastwam/lerobot_released_visualfix/policy_overlay"
  if [[ -d "${overlay}" ]]; then
    policy_path="${overlay}"
  else
    policy_path="${repo_root}/pretrained/w0_fastwam/lerobot"
  fi
fi

if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/outputs/literature/w0_fastwam/lr_task_probe/${suite}_task${task_id}_wait${num_steps_wait:-default}_obs${obs_size}_${dtype}"
fi

# Derive the runtime environment from the launcher itself rather than restating
# it here.  `_model_runtime_env` adds `<vendor>/FastWAM/src` to PYTHONPATH and
# sets W0_FASTWAM_SOURCE_ROOT only when that directory exists, so a hand-copied
# list silently loses those two on any host where the vendor tree lives
# elsewhere -- and the LeRobot `components.py` local-loader then fails with
# "cannot import the pinned FastWAM raw component loader".
if [[ -z "${vendor_root}" ]]; then
  vendor_root="${W0_VENDOR_ROOT:-/opt/robot-practice/w0-vendor}"
fi
if [[ ! -d "${vendor_root}" ]]; then
  vendor_root="${repo_root}/third_party/w0_fastwam"
fi
if [[ ! -d "${vendor_root}/FastWAM/src" ]]; then
  echo "error: ${vendor_root}/FastWAM/src not found; pass --vendor-root=..." >&2
  exit 2
fi

runtime_exports="$(
  cd "${repo_root}" && PYTHONPATH="${repo_root}/src" python3 - "${vendor_root}" <<'PY'
import os
import shlex
import sys

from robot_practice.experiments.literature.w0_fastwam_audit import (
    DEFAULT_CONFIG,
    _artifact_paths,
    _model_runtime_env,
    load_config,
)

# Drop the bootstrap entry so it does not leak into the child's PYTHONPATH.
os.environ.pop("PYTHONPATH", None)
config = load_config(DEFAULT_CONFIG)
for key, value in _model_runtime_env(_artifact_paths(config), vendor_root=sys.argv[1]).items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
eval "${runtime_exports}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export MUJOCO_EGL_DEVICE_ID="${gpu}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
if [[ -n "${num_steps_wait}" ]]; then
  export W0_LIBERO_NUM_STEPS_WAIT="${num_steps_wait}"
fi

cat <<EOF
[LR-PROBE] suite=${suite} task_id=${task_id} episodes=${episodes}
[LR-PROBE] num_steps_wait=${num_steps_wait:-lerobot-default} obs_size=${obs_size} dtype=${dtype}
[LR-PROBE] episode_length=${episode_length} seed=${seed} gpu=${gpu}
[LR-PROBE] vendor_root=${vendor_root}
[LR-PROBE] policy=${policy_path}
[LR-PROBE] output=${output_dir}
EOF
echo "${runtime_exports}" | sed 's/^/[LR-PROBE] runtime /'

command=(
  conda run --no-capture-output -n fastwam-lerobot lerobot-eval
  "--policy.path=${policy_path}"
  "--policy.device=cuda"
  "--policy.torch_dtype=${dtype}"
  "--policy.n_action_steps=10"
  "--policy.toggle_action_dimensions=[-1]"
  "--env.type=libero"
  "--env.task=${suite}"
  "--env.task_ids=[${task_id}]"
  "--env.observation_height=${obs_size}"
  "--env.observation_width=${obs_size}"
  "--eval.batch_size=1"
  "--eval.n_episodes=${episodes}"
  "--seed=${seed}"
  "--env.episode_length=${episode_length}"
  "--output_dir=${output_dir}"
)

if [[ "${dry_run}" == "1" ]]; then
  printf '[LR-PROBE] would run:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${output_dir}"
exec "${command[@]}"
