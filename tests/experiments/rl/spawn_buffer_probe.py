"""Subprocess fixture: delay spawn unpickling until the constructor has returned."""

import gc
import multiprocessing as mp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

gate = Path(sys.argv[1]) / "release"
if __name__ == "__mp_main__":
    deadline = time.monotonic() + 10
    while not gate.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("parent did not release child")
        time.sleep(0.01)


def consume(buffer):
    assert buffer[0] == 42


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    module = SimpleNamespace(create_shared_memory=lambda: ctx.Array("i", [42]))

    def construct():
        buffer = module.create_shared_memory()
        process = ctx.Process(target=consume, args=(buffer,))
        process.start()
        return SimpleNamespace(processes=[process])

    if sys.argv[2] == "keep":
        from robot_practice.experiments.rl.rollout_forensics import retain_shared_buffers

        vector = retain_shared_buffers(module, construct)
    else:
        vector = construct()
    gc.collect()
    gate.touch()
    vector.processes[0].join(15)
    if vector.processes[0].is_alive():
        vector.processes[0].terminate()
        vector.processes[0].join()
        raise TimeoutError("child hung")
    print(f"child_exitcode={vector.processes[0].exitcode}")
