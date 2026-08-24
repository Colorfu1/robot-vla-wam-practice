#!/usr/bin/env bash
# Dump every rank's Python stack while a training step stall is still in progress.
#
# Why: two 8-GPU runs died to a 600 s NCCL ALLREDUCE timeout in which exactly one rank was
# absent from the watchdog reports (step 265/rank 3, then step 598/rank 1). That rank never
# enqueued the collective, so it was stuck in host-side code, and NCCL FlightRecorder cannot
# see it -- FlightRecorder records collectives, and this rank had none. sitecustomize
# registers faulthandler on SIGUSR1, so dumping mid-stall shows where the stuck rank is.
#
# Two hard lessons are encoded below, both learned by destroying a run:
#
# 1. Signal ONLY the rank processes. DataLoader workers are forked from the ranks, so their
#    /proc cmdline is identical and a naive `pgrep -f scripts/train.py` matched 65 processes
#    instead of 8. A worker is identified by its parent also being in the candidate set.
#
# 2. SIGUSR1's default action is Term (signal(7)). faulthandler must therefore register it
#    with chain=False, which sitecustomize now does. Killing the workers made DataLoader's
#    SIGCHLD handler fire in every rank and tore the run down 15 s after the dump, so the
#    stall evidence was lost. This script additionally refuses to run if sitecustomize is
#    not the no-chain version, because that combination is destructive rather than merely
#    unhelpful.
#
# Dumps are serialized: eight simultaneous writers to one stderr produced unreadable
# interleaved output, so ranks are signalled one at a time.
#
# Usage:
#   scripts/experiments/literature/watch_w0_stall.sh <train.log> [stall_seconds] [poll_seconds]
#
# Defaults: stall_seconds=150, poll_seconds=15. At the measured ~4.0 s/step with
# log_every=10 a healthy run advances the counter about every 40 s, so 150 s of no progress
# is well clear of normal jitter and still leaves most of the 600 s NCCL window.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${script_dir}/../../.." && pwd)"

log="${1:-}"
stall_seconds="${2:-150}"
poll_seconds="${3:-15}"

if [[ -z "${log}" ]]; then
  echo "usage: $0 <train.log> [stall_seconds] [poll_seconds]" >&2
  exit 2
fi
for value in "${stall_seconds}" "${poll_seconds}"; do
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -le 0 ]]; then
    echo "error: seconds must be positive integers, got '${value}'" >&2
    exit 2
  fi
done

# Refuse to run against a chaining SIGUSR1 handler: that turns an observation into a kill.
sitecustomize="${repo}/src/robot_practice/experiments/literature/w0_runtime/sitecustomize.py"
if [[ -f "${sitecustomize}" ]] && ! grep -q '("SIGUSR1", False)' "${sitecustomize}"; then
  echo "error: sitecustomize registers SIGUSR1 with chain=True; SIGUSR1's default action is" >&2
  echo "       Term, so dumping would kill the ranks. Sync the no-chain fix first." >&2
  exit 4
fi

# Two separate questions, previously conflated into one and it broke the watcher:
#
#   candidate_pids -> "is training still alive at all?"  Matches during model loading, which
#                     takes about ten minutes before any CUDA context exists.
#   gpu_pids       -> "which of those are the ranks?"     Empty during that loading window.
#
# Requiring a CUDA context to answer the first question made the watcher exit immediately
# when started right after launch.
candidate_pids() {
  pgrep -f "scripts/train.py" 2>/dev/null || true
}

# Keep the agent and the ranks; drop the forked DataLoader workers.
#
# The process tree is three deep and all three levels share a cmdline containing
# scripts/train.py, so cmdline matching alone selects 65 processes instead of 8:
#
#   elastic agent      depth 0    no candidate ancestor
#     |- rank 0..7     depth 1    one candidate ancestor (the agent)
#          |- worker   depth 2    two candidate ancestors
#
# Depth within the candidate set is therefore the discriminator, and it needs nothing but
# /proc. Two earlier attempts failed here:
#
#   * "parent is also a candidate" dropped the ranks together with the workers and kept only
#     the agent, whose stack is a monitoring loop -- one crash's evidence lost.
#   * nvidia-smi --query-compute-apps returns nothing (or host-namespace pids) inside this
#     container, so requiring a CUDA context made the watcher find zero ranks and exit.
#
# Depth <= 1 keeps agent + ranks. Signalling the agent costs one short extra dump and is
# harmless with chain=False; signalling 57 workers is what must never happen again.
gpu_pids() {
  local -a candidates=()
  mapfile -t candidates < <(candidate_pids)
  [[ "${#candidates[@]}" -eq 0 ]] && return 0

  local pid cur depth guard
  for pid in "${candidates[@]}"; do
    depth=0
    cur="${pid}"
    guard=0
    while (( guard < 24 )); do
      guard=$(( guard + 1 ))
      cur="$(ps -o ppid= -p "${cur}" 2>/dev/null | tr -d ' ')"
      [[ -z "${cur}" || "${cur}" -le 1 ]] && break
      if printf '%s\n' "${candidates[@]}" | grep -qx "${cur}"; then
        depth=$(( depth + 1 ))
        (( depth > 1 )) && break
      fi
    done
    if (( depth <= 1 )); then
      printf '%s\n' "${pid}"
    fi
  done
}

# Host RAM telemetry, because a cgroup OOM is what actually kills these runs.
#
# A DataLoader worker was SIGKILLed at step 560 with memory.failcnt=25 and the cgroup peak
# sitting exactly on its 1788.9 GiB limit. The rank then blocked in queue.get waiting for
# that worker's batch, never enqueued its gradient allreduce, and the other seven spun in
# DeepSpeed's allreduce until the 600 s NCCL watchdog killed everything. GPU memory was
# never the problem -- it measured 64.2 GiB of 95.6 per rank -- and host RAM was never
# measured at all, which is why four crashes went unexplained.
#
# The growth shape decides the fix: a plateau means the prefetch buffers are simply too big
# for 64 workers (lower num_workers), while a straight line means the pyav fallback decoder
# leaks (lowering num_workers would only postpone the OOM).
cgroup_usage_bytes() {
  cat /sys/fs/cgroup/memory.current 2>/dev/null ||
    cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null ||
    echo ""
}

cgroup_limit_bytes() {
  cat /sys/fs/cgroup/memory.max 2>/dev/null ||
    cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null ||
    echo ""
}

cgroup_oom_kills() {
  awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null ||
    cat /sys/fs/cgroup/memory/memory.failcnt 2>/dev/null ||
    echo ""
}

# Attribute anonymous-memory growth to ranks vs DataLoader workers.
#
# ps RSS cannot answer this: workers are forked from their rank and share its pages
# copy-on-write, so ps counts the same memory once per process. That is why 64 workers each
# showed 34 GiB while the cgroup only held 1368 GiB in total -- the per-process figure was
# mostly double-counted. Pss_Anon divides each shared page by its number of sharers, so
# summing it across processes gives the real total and the split is meaningful.
#
# Growth is attributable regardless: fork sharing is fixed at fork time, so anything
# allocated afterwards is private to whoever allocated it.
anon_kb() {
  local pid="$1"
  local value
  value="$(awk '/^Pss_Anon:/ {print $2; found=1} END {if (!found) exit 1}' \
    "/proc/${pid}/smaps_rollup" 2>/dev/null)" && { printf '%s' "${value}"; return 0; }
  # Older kernels have no smaps_rollup; RssAnon double-counts forked pages but still shows
  # the growth trend.
  awk '/^RssAnon:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || printf '0'
}

report_attribution() {
  local -a ranks=() all=()
  mapfile -t ranks < <(gpu_pids)
  mapfile -t all < <(candidate_pids)
  [[ "${#all[@]}" -eq 0 ]] && return 0
  local pid kb rank_kb=0 worker_kb=0
  for pid in "${all[@]}"; do
    kb="$(anon_kb "${pid}")"
    [[ "${kb}" =~ ^[0-9]+$ ]] || continue
    if printf '%s\n' "${ranks[@]}" | grep -qx "${pid}"; then
      rank_kb=$(( rank_kb + kb ))
    else
      worker_kb=$(( worker_kb + kb ))
    fi
  done
  printf '[attr] agent+ranks=%s GiB workers=%s GiB (anon, share-adjusted)\n' \
    "$(( rank_kb / 1048576 ))" "$(( worker_kb / 1048576 ))"
}

report_memory() {
  local step="$1"
  local used limit oom workers pct=""
  used="$(cgroup_usage_bytes)"
  limit="$(cgroup_limit_bytes)"
  oom="$(cgroup_oom_kills)"
  workers="$(pgrep -fc "scripts/train.py" 2>/dev/null || echo '?')"
  if [[ "${used}" =~ ^[0-9]+$ && "${limit}" =~ ^[0-9]+$ && "${limit}" -gt 0 ]]; then
    pct=" ($(( used * 100 / limit ))% of limit)"
  fi
  printf '[mem] %s %s used=%s GiB%s procs=%s oom/failcnt=%s\n' \
    "$(date -u +%FT%TZ)" "${step:-<no step>}" \
    "$( [[ "${used}" =~ ^[0-9]+$ ]] && echo $(( used / 1073741824 )) || echo '?' )" \
    "${pct}" "${workers}" "${oom:-?}"
}

# Kernel-side view of a rank, which discriminates causes that look identical in Python.
#
#   D (uninterruptible sleep) -> blocked in a syscall, almost always storage or network IO
#   S (interruptible sleep)   -> waiting on a lock, condition variable or futex
#   R (running)               -> spinning, e.g. NCCL busy-wait
#
# wchan names the kernel function being waited on, which usually says which subsystem.
report_kernel_state() {
  local pid="$1"
  local state wchan
  state="$(awk '/^State:/ {print $2, $3}' "/proc/${pid}/status" 2>/dev/null || true)"
  wchan="$(cat "/proc/${pid}/wchan" 2>/dev/null || true)"
  echo "[watch]      main: state=${state:-?} wchan=${wchan:-?}"
  # Any thread in D state is the strongest single clue available; list them.
  local t tstate twchan
  for t in "/proc/${pid}/task"/*; do
    [[ -d "${t}" ]] || continue
    tstate="$(awk '{print $3}' "${t}/stat" 2>/dev/null || true)"
    if [[ "${tstate}" == "D" ]]; then
      twchan="$(cat "${t}/wchan" 2>/dev/null || true)"
      echo "[watch]      thread $(basename "${t}") in D state wchan=${twchan:-?}"
    fi
  done
}

current_step() {
  # The rich-formatted progress line wraps, so match only the step token itself.
  grep -aoE 'step=[0-9]+/[0-9]+' "${log}" 2>/dev/null | tail -1 || true
}

echo "[watch] log=${log} stall_seconds=${stall_seconds} poll_seconds=${poll_seconds}"
last_step=""
last_change=$(date +%s)
dumped_for=""

announced_ranks=0

while true; do
  mapfile -t alive < <(candidate_pids)
  if [[ "${#alive[@]}" -eq 0 ]]; then
    echo "[watch] no training process; exiting at $(date -u +%FT%TZ)"
    exit 0
  fi

  mapfile -t pids < <(gpu_pids)
  if [[ "${#pids[@]}" -eq 0 ]]; then
    # Startup: processes exist but no CUDA context yet. Wait; do not treat as finished.
    sleep "${poll_seconds}"
    continue
  fi
  if [[ "${announced_ranks}" -eq 0 ]]; then
    echo "[watch] tracking ${#pids[@]} processes (agent+ranks): ${pids[*]}"
    announced_ranks=1
  fi

  step="$(current_step)"
  now=$(date +%s)

  if [[ -z "${step}" ]]; then
    # No progress line yet. Without this the stall timer would run during startup; the
    # previous version only avoided a spurious dump because dumped_for and step were both
    # empty strings, which is an accident rather than a decision.
    last_change="${now}"
    sleep "${poll_seconds}"
    continue
  fi

  if [[ "${step}" != "${last_step}" ]]; then
    last_step="${step}"
    last_change="${now}"
    dumped_for=""
    # Sample RAM on every step change so the growth curve is recoverable afterwards.
    report_memory "${step}"
    report_attribution
    continue
  fi

  idle=$(( now - last_change ))
  if [[ "${idle}" -ge "${stall_seconds}" && "${dumped_for}" != "${step}" ]]; then
    report_memory "${step}"
    echo "[watch] stalled ${idle}s at ${step:-<no step yet>}; dumping ${#pids[@]} processes (agent+ranks; workers excluded) at $(date -u +%FT%TZ)"
    # Two passes ~60 s apart. One pass cannot tell "wedged at a single frame" from
    # "advancing far too slowly to log a step", and those have different fixes.
    for pass in 1 2; do
      echo "[watch] === pass ${pass} at $(date -u +%FT%TZ) ==="
      for pid in "${pids[@]}"; do
        echo "[watch]   -> SIGUSR1 pid=${pid}"
        report_kernel_state "${pid}"
        # chain=False in sitecustomize, so this dumps without terminating the rank.
        kill -USR1 "${pid}" 2>/dev/null || echo "[watch]   could not signal ${pid}"
        # Serialize: concurrent dumps to one stderr are unreadable.
        sleep 2
      done
      if [[ "${pass}" -eq 1 ]]; then
        echo "[watch] waiting 60s before the second pass"
        sleep 60
      fi
    done
    echo "[watch] dump complete at $(date -u +%FT%TZ)"
    # Dump once per stalled step so a long stall does not flood the log.
    dumped_for="${step}"
  fi

  sleep "${poll_seconds}"
done
