import os
import resource
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def test_native_abort_hook_preserves_signal_and_records_stack(tmp_path):
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("gcc unavailable; do not install")
    source = Path(__file__).resolve().parents[3] / "scripts/experiments/rl/native_abort_trace.c"
    library = tmp_path / "probe.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(library),
            "-ldl",
        ],
        check=True,
    )
    env = dict(os.environ, LD_PRELOAD=str(library), ROBOT_PRACTICE_NATIVE_TRACE_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", "import os; os.abort()"],
        env=env,
        capture_output=True,
        text=True,
        preexec_fn=no_core,
        timeout=15,
        check=False,
    )
    assert result.returncode == -signal.SIGABRT
    paths = list(tmp_path.glob("native_*.log"))
    assert len(paths) == 1
    trace = paths[0].read_text()
    assert "reason=abort" in trace
    assert "object=" in trace and "symbol=abort" in trace
