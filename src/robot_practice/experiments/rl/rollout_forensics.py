"""Opt-in, stdlib-only rollout evidence. Does not alter actions or worker scheduling."""

import json
import os
import signal
import time
import traceback
from collections import deque
from pathlib import Path


def retain_shared_buffers(module, constructor, **kwargs):
    """Keep synchronized buffers alive until the diagnostic vector itself is released.

    DPPO's constructor retains numpy views but drops its local synchronized Array
    containers, which can unlink their semaphores before spawn unpickles them.
    Scope the temporary hook to construction; never edit the installed module.
    """
    original = module.create_shared_memory
    buffers = []

    def capture(*args, **options):
        buffer = original(*args, **options)
        buffers.append(buffer)
        return buffer

    module.create_shared_memory = capture
    try:
        vector = constructor(**kwargs)
        vector._diagnostic_shared_buffers = buffers
        return vector
    finally:
        module.create_shared_memory = original


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def memory_snapshot():
    membership = read_text("/proc/self/cgroup")
    roots = {Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")}
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        relative = parts[2].lstrip("/")
        if ".." in Path(relative).parts:
            continue
        if parts[0] == "0":
            roots.add(Path("/sys/fs/cgroup") / relative)
        elif "memory" in parts[1].split(","):
            roots.add(Path("/sys/fs/cgroup/memory") / relative)
    values = {}
    for root in sorted(roots):
        for name in (
            "memory.current",
            "memory.max",
            "memory.peak",
            "memory.events",
            "memory.events.local",
            "memory.usage_in_bytes",
            "memory.limit_in_bytes",
            "memory.failcnt",
            "memory.oom_control",
        ):
            path = root / name
            if path.exists():
                values[str(path)] = read_text(path)
    return {
        "cgroup_membership": membership,
        "cgroup": values,
        "self_status": read_text("/proc/self/status"),
        "memory_pressure": read_text("/proc/pressure/memory"),
    }


def record(directory, event, **fields):
    """Append and flush evidence independently in each writer process."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    row = {"time_ns": time.time_ns(), "pid": os.getpid(), "event": event, **fields}
    with (directory / f"process_{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()


def process_snapshot(processes):
    rows = []
    for index, process in enumerate(processes):
        code = process.exitcode
        name = None
        if code is not None and code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = "unknown"
        rows.append(
            {
                "worker_index": index,
                "pid": process.pid,
                "exitcode": code,
                "signal": name,
                "status": read_text(f"/proc/{process.pid}/status"),
            }
        )
    return rows


class ForensicVector:
    """Capture child exit codes BEFORE the existing worker cleanup runs."""

    def __init__(self, vector, directory):
        self.vector = vector
        self.directory = directory
        self.completed = 0
        self.capture("vector_created")

    def __getattr__(self, name):
        return getattr(self.vector, name)

    def capture(self, event, **fields):
        record(
            self.directory,
            event,
            completed_chunks=self.completed,
            workers=process_snapshot(self.vector.processes),
            memory=memory_snapshot(),
            **fields,
        )

    def step(self, action):
        self.capture("step_begin")
        try:
            if os.environ.get("ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP") == "1":
                result = serial_vector_step(self.vector, action)
            else:
                result = self.vector.step(action)
        except BaseException:
            error = traceback.format_exc()
            self.capture("step_failed_before_cleanup", traceback=error)
            # EOF may precede exitcode availability. Do not terminate any child here.
            for process in self.vector.processes:
                process.join(timeout=0.05)
            self.capture("step_failed_recheck_before_cleanup", traceback=error)
            raise
        self.completed += 1
        self.capture("step_complete")
        return result

    def reset_arg(self, **kwargs):
        try:
            result = self.vector.reset_arg(**kwargs)
        except BaseException:
            self.capture("reset_failed_before_cleanup", traceback=traceback.format_exc())
            raise
        self.capture("reset_complete")
        return result

    def close(self):
        self.capture("cleanup_begin")
        try:
            return self.vector.close()
        finally:
            self.capture("cleanup_end")


class _SerialStepPipe:
    """Wait for each step reply before dispatching to the next existing worker.

    The upstream step_wait still assembles observations/rewards and handles errors.
    No workers or EGL contexts are removed; only step dispatch scheduling changes.
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.replies = deque()

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def send(self, message):
        self.pipe.send(message)
        if message[0] == "step":
            self.replies.append(self.pipe.recv())

    def recv(self):
        return self.replies.popleft() if self.replies else self.pipe.recv()

    def poll(self, timeout=0):
        return bool(self.replies) or self.pipe.poll(timeout)


def serial_vector_step(vector, action):
    original = vector.parent_pipes
    vector.parent_pipes = [_SerialStepPipe(pipe) for pipe in original]
    try:
        return vector.step(action)
    finally:
        vector.parent_pipes = original
