#!/usr/bin/env bash
# Launch one group of the W0 FW-V/FW-NV causal pair in the background.
#
# Why this script exists: the equivalent direct invocation is a ~450-character single
# line, and pasting it into a remote shell has repeatedly been split by soft-wrapping.
# When that happens the tail of the line becomes a second command, so `&` and the output
# redirect detach from `nohup`, and --num-gpus/--output-root/--overwrite are silently
# dropped -- the run then goes to the foreground and writes to the default output root.
# Keeping the launch here means the remote command stays short enough to survive a paste.
#
# Usage:
#   scripts/experiments/literature/launch_w0_causal.sh <with_video|no_video> [epochs] [workers]
#
# epochs defaults to 10, which reproduces the upstream recipe
# (configs/task/libero_uncond_2cam224_1e-4.yaml: num_epochs=10, max_steps=null). The
# epoch budget also sets the cosine schedule length and the 5 percent warmup, so it is not
# interchangeable with a pinned step count.
#
# workers overrides DataLoader workers per rank; omit it to use the protocol value of 8.
# It is numerically inert -- the sampler order comes from the seed, so batches stay
# bit-identical -- but it is the main lever on host RAM. Four runs died to a cgroup OOM
# (memory.failcnt=25, peak on the 1788.9 GiB limit): a SIGKILLed DataLoader worker left one
# rank blocked in queue.get, it never enqueued its gradient allreduce, and the other seven
# spun in DeepSpeed's allreduce until the 600 s NCCL watchdog killed the job. 8 workers per
# rank means 64 pyav decoder processes, because torchcodec needs FFmpeg 4-7 and the image
# ships FFmpeg 8 with no older libavutil anywhere on it.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${repo}"

variant="${1:-}"
epochs="${2:-10}"
workers="${3:-}"

case "${variant}" in
  with_video | no_video) ;;
  *)
    echo "usage: $0 <with_video|no_video> [epochs] [workers]" >&2
    exit 2
    ;;
esac

if ! [[ "${epochs}" =~ ^[0-9]+$ ]] || [[ "${epochs}" -le 0 ]]; then
  echo "error: epochs must be a positive integer, got '${epochs}'" >&2
  exit 2
fi

worker_args=()
if [[ -n "${workers}" ]]; then
  if ! [[ "${workers}" =~ ^[0-9]+$ ]]; then
    echo "error: workers must be a non-negative integer, got '${workers}'" >&2
    exit 2
  fi
  worker_args=(--train-num-workers="${workers}")
fi

# Both groups share this parent so pair_contract.json is shared and verified, and so the
# sibling step-budget check can see the other group's manifest.
#
# W0_FREEZE_VIDEO_EXPERT selects the FW-FZ attribution arm, which is also a no_video run.
# Its parent is derived from the flag rather than passed separately, because passing it
# separately makes it possible to forget: the run would then overwrite FW-NV's outputs and
# the sibling budget check would compare it against the wrong group. FW-FZ is not part of
# the gated pair, so it must not share their parent.
pair_suffix=""
if [[ -n "${W0_FREEZE_VIDEO_EXPERT:-}" && "${W0_FREEZE_VIDEO_EXPERT}" != "0" ]]; then
  pair_suffix="_frozen"
fi
pair_root="${repo}/outputs/literature/w0_fastwam/causal_${epochs}ep${pair_suffix}"
output_root="${pair_root}/${variant}"
log="${pair_root}/launch_${variant}.log"
mkdir -p -- "${pair_root}"

# A stale render-device pin would override the launcher's autocalibrated EGL map.
unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
export PYTHONPATH="${repo}/src${PYTHONPATH:+:${PYTHONPATH}}"

if pgrep -f "scripts/train.py" >/dev/null 2>&1; then
  echo "error: a FastWAM training process is already running; stop it first:" >&2
  pgrep -af "scripts/train.py" >&2
  exit 3
fi

nohup bash "${repo}/scripts/experiments/literature/run_w0_fastwam.sh" \
  --phase=train-causal-pair \
  --config="${repo}/configs/experiments/literature/w0_fastwam_reproduction.json" \
  --vendor-root="${W0_VENDOR_ROOT:-/opt/robot-practice/w0-vendor}" \
  --variant="${variant}" \
  --train-epochs="${epochs}" \
  "${worker_args[@]}" \
  --steps=20000 \
  --num-gpus=8 \
  --output-root="${output_root}" \
  --overwrite \
  >"${log}" 2>&1 &

pid=$!
printf 'variant   %s\n' "${variant}"
printf 'epochs    %s\n' "${epochs}"
printf 'workers   %s\n' "${workers:-8 (protocol default)}"
printf 'freeze    %s\n' "${W0_FREEZE_VIDEO_EXPERT:-0 (video expert trainable)}"
printf 'output    %s\n' "${output_root}"
printf 'launchlog %s\n' "${log}"
printf 'trainlog  %s\n' "${output_root}/train.log"
printf 'pid       %s\n' "${pid}"
