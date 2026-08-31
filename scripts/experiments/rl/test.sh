#!/usr/bin/env bash

# Read-only remote probe for the S1 and Light-WAM Peg ManiSkill2 environments.
#
# This script intentionally does not install packages, pull images, download assets,
# or change git configuration. It can be run from any directory inside the repository:
#
#   bash scripts/experiments/rl/test.sh

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
reference_root="${repo_root}/datasets/raw/policy_decorator/reference_repo"
expected_policy_decorator_commit="92ba9ba442587ae5989c286355ace2200a0537fb"
expected_image="${RL_THREE_STAGE_IMAGE:-test-lab-instance-cn-beijing.cr.volces.com/lidar-wwd/practice:latest}"

failures=0
warn() {
  echo "WARN: $*"
}

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

env_exists() {
  conda env list | awk -v env_name="$1" '$1 == env_name {found=1} END {exit !found}'
}

cd -- "${repo_root}"

echo "== identity =="
echo "repo_root=${repo_root}"
echo "hostname=$(hostname)"
echo "user=$(id -un) uid=$(id -u)"
if git_root_output="$(git -c "safe.directory=${repo_root}" -C "${repo_root}" rev-parse --show-toplevel 2>&1)"; then
  echo "git_root=${git_root_output}"
else
  fail "git cannot inspect the repository even with a one-shot safe.directory override: ${git_root_output}"
fi

echo "== docker identity, if the worker exposes the host socket =="
if command -v docker >/dev/null 2>&1 && [[ -S /var/run/docker.sock ]]; then
  docker_inspect_output=""
  if docker_inspect_output="$(docker inspect "$(hostname)" \
      --format 'container_id={{.Id}} image={{.Config.Image}} image_id={{.Image}}' 2>&1)"; then
    echo "${docker_inspect_output}"
  else
    warn "docker inspect by hostname failed: ${docker_inspect_output}"
    docker ps --no-trunc --format 'id={{.ID}} image={{.Image}} name={{.Names}}' || \
      warn "docker ps is unavailable"
  fi
  if docker image inspect "${expected_image}" \
      --format 'expected_image_id={{.Id}} repo_digests={{json .RepoDigests}}' 2>&1; then
    true
  else
    warn "expected RL image is not visible in the connected Docker daemon"
  fi
else
  echo "docker socket unavailable"
fi

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
      --format=csv,noheader; then
    fail "nvidia-smi cannot query the GPU"
  fi
else
  fail "nvidia-smi is unavailable"
fi

echo "== conda environments =="
conda env list

echo "== robot-practice runtime probe =="
if env_exists robot-practice; then
  if conda run --no-capture-output -n robot-practice python -c '
import importlib.metadata as metadata
import importlib.util
import sys

print("python:", sys.executable)
for name in ("torch", "mani-skill", "gymnasium", "stable-baselines3", "diffusers"):
    try:
        print(name + ":", metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name + ": MISSING")
print("mani_skill2 module:", importlib.util.find_spec("mani_skill2") is not None)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_error:", type(exc).__name__, str(exc))
'; then
    true
  else
    fail "robot-practice runtime probe failed"
  fi
else
  fail "robot-practice conda environment is missing"
fi

echo "== local Policy Decorator checkout =="
if [[ ! -d "${reference_root}/.git" ]]; then
  fail "Policy Decorator checkout is missing: ${reference_root}"
else
  if reference_commit="$(git -c "safe.directory=${reference_root}" \
      -C "${reference_root}" rev-parse HEAD 2>&1)"; then
    echo "reference_commit=${reference_commit}"
    if [[ "${reference_commit}" != "${expected_policy_decorator_commit}" ]]; then
      fail "Policy Decorator commit mismatch: expected ${expected_policy_decorator_commit}, got ${reference_commit}"
    fi
  else
    fail "cannot read the Policy Decorator commit: ${reference_commit}"
  fi
fi
if [[ -f "${reference_root}/envs/maniskill_fixed.py" ]]; then
  echo "envs/maniskill_fixed.py: present"
else
  fail "envs/maniskill_fixed.py is missing"
fi

echo "== S1 conda package probe =="
s1_env_ready=0
if env_exists maniskill2-pidec; then
  if conda run --no-capture-output -n maniskill2-pidec python -c '
import importlib.metadata as metadata
import sys

print("python:", sys.executable)
for name in ("torch", "mani-skill2", "gymnasium", "gymnasium-robotics", "stable-baselines3", "diffusers"):
    try:
        print(name + ":", metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name + ": MISSING")
import torch
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
'; then
    s1_env_ready=1
  else
    fail "maniskill2-pidec package/runtime probe failed"
  fi
else
  fail "maniskill2-pidec conda environment is missing"
fi

echo "== S1 PegInsertionSide-v2 registration probe =="
registration_ready=0
if [[ "${s1_env_ready}" == 1 && -f "${reference_root}/envs/maniskill_fixed.py" ]]; then
  if env PYTHONNOUSERSITE=1 \
      PYTHONPATH="${repo_root}/src:${reference_root}" \
      conda run --no-capture-output -n maniskill2-pidec python -c '
import gymnasium as gym
import mani_skill2.envs
import envs.maniskill_fixed

spec = gym.spec("PegInsertionSide-v2")
print("registered:", spec.id)
print("max_episode_steps:", spec.max_episode_steps)
print("entry_point:", spec.entry_point)
'; then
    registration_ready=1
  else
    fail "PegInsertionSide-v2 registration probe failed"
  fi
else
  warn "skipping registration probe because the S1 environment or local registration file is unavailable"
fi

echo "== one-episode S1 dummy smoke =="
if [[ "${registration_ready}" == 1 ]]; then
  smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/s1-probe.XXXXXX")"
  echo "smoke_root=${smoke_root}"
  if env PYTHONNOUSERSITE=1 \
      PYTHONPATH="${repo_root}/src" \
      conda run --no-capture-output -n maniskill2-pidec \
      python -m robot_practice.experiments.rl.policy_decorator_eval \
      --dummy \
      --reference-root="${reference_root}" \
      --env-id=PegInsertionSide-v2 \
      --arm=base \
      --seeds=1 \
      --episodes-per-seed=1 \
      --ddim-steps=4 \
      --act-horizon=4 \
      --dummy-steps=1 \
      --output-dir="${smoke_root}/s1_base"; then
    sed -n '1,240p' "${smoke_root}/s1_base/metrics.json"
    echo "S1_DUMMY_SMOKE=PASS"
  else
    fail "one-episode S1 dummy smoke failed"
  fi
else
  warn "skipping dummy smoke because PegInsertionSide-v2 registration did not pass"
fi

echo "== Light-WAM Peg environment probe =="
s2_peg_env_ready=0
if env_exists lightwam-peg; then
  if conda run --no-capture-output -n lightwam-peg python \
      docker/training/verify_rl_three_stage_env.py --stage=s2-peg; then
    s2_peg_env_ready=1
  else
    fail "lightwam-peg package/runtime probe failed"
  fi
else
  fail "lightwam-peg conda environment is missing"
fi

echo "== Light-WAM Peg environment registration probe =="
peg_registration_ready=0
if [[ "${s2_peg_env_ready}" == 1 && -f "${reference_root}/envs/maniskill_fixed.py" ]]; then
  if env PYTHONNOUSERSITE=1 \
      PYTHONPATH="${repo_root}/src:${reference_root}" \
      conda run --no-capture-output -n lightwam-peg python -c '
import gymnasium as gym
import mani_skill2.envs
import envs.maniskill_fixed

spec = gym.spec("PegInsertionSide-v2")
print("registered:", spec.id)
print("max_episode_steps:", spec.max_episode_steps)
'; then
    peg_registration_ready=1
  else
    fail "Light-WAM Peg registration probe failed"
  fi
else
  warn "skipping Light-WAM Peg registration probe because lightwam-peg is unavailable"
fi

echo "== one-episode Light-WAM Peg dummy smoke =="
if [[ "${s2_peg_env_ready}" == 1 ]]; then
  peg_smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/lightwam-peg-probe.XXXXXX")"
  echo "peg_smoke_root=${peg_smoke_root}"
  if PYTHON_BIN=/opt/conda/envs/lightwam-peg/bin/python \
      bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
      --phase=evaluate --dummy --arm=base --device=cpu \
      --eval-seeds=1 --episodes-per-seed=1 --max-steps=4 \
      --output-dir="${peg_smoke_root}/base"; then
    test -s "${peg_smoke_root}/base/metrics.json"
    echo "LIGHTWAM_PEG_DUMMY_SMOKE=PASS"
  else
    fail "one-episode Light-WAM Peg dummy smoke failed"
  fi
else
  warn "skipping Light-WAM Peg dummy smoke because lightwam-peg is unavailable"
fi

if (( failures > 0 )); then
  echo "RESULT=FAIL failures=${failures}" >&2
  exit 1
fi

echo "RESULT=PASS"
