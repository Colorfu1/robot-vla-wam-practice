import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_practice.experiments.rl.rollout_forensics import ForensicVector, memory_snapshot


@pytest.mark.parametrize("mode, expected", [("drop", 1), ("keep", 0)])
def test_real_spawn_buffer_lifetime(tmp_path, mode, expected):
    fixture = Path(__file__).with_name("spawn_buffer_probe.py")
    result = subprocess.run(
        [sys.executable, str(fixture), str(tmp_path), mode],
        capture_output=True,
        text=True,
        timeout=25,
        check=True,
    )
    assert f"child_exitcode={expected}" in result.stdout
    if mode == "drop":
        assert "SemLock._rebuild" in result.stderr
        assert "FileNotFoundError" in result.stderr
    else:
        assert "FileNotFoundError" not in result.stderr


def child_exit():
    os._exit(17)


def test_failure_records_actual_child_exit_before_cleanup(tmp_path):
    process = multiprocessing.get_context("spawn").Process(target=child_exit)
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 17

    class Vector:
        processes = (process,)

        def step(self, action):
            raise EOFError("test worker death")

        def close(self):
            pass

    wrapped = ForensicVector(Vector(), tmp_path)
    with pytest.raises(EOFError):
        wrapped.step(None)
    rows = [
        json.loads(line) for p in tmp_path.glob("*.jsonl") for line in p.read_text().splitlines()
    ]
    failure = next(row for row in rows if row["event"] == "step_failed_before_cleanup")
    assert failure["workers"][0]["exitcode"] == 17
    assert failure["completed_chunks"] == 0
    assert all(row["event"] != "cleanup_begin" for row in rows)
    wrapped.close()


def test_success_preserves_result_and_counts(tmp_path):
    result = object()

    class Vector:
        processes = ()

        def step(self, action):
            return result

        def reset_arg(self, **kwargs):
            return kwargs

    wrapped = ForensicVector(Vector(), tmp_path)
    assert wrapped.step(None) is result
    assert wrapped.completed == 1
    assert wrapped.reset_arg(options_list=[{}]) == {"options_list": [{}]}
    assert "cgroup" in memory_snapshot()


@pytest.mark.parametrize("worker", [None, 2])
def test_replay_selects_recorded_actions_and_seeds(tmp_path, monkeypatch, worker):
    from robot_practice.experiments.rl import lightwam_rollout_replay as replay

    source = tmp_path / "source"
    source.mkdir()
    action = np.arange(8 * 4 * 7).reshape(8, 4, 7)
    np.savez(
        source / "chunk_0000_input.npz",
        environment_seeds=np.arange(10001, 10009),
        state=np.zeros((8, 1, 9)),
        action_executed=action,
    )
    selection = slice(None) if worker is None else slice(worker, worker + 1)
    count = 8 if worker is None else 1

    class Env:
        def __init__(self, **kwargs):
            assert kwargs["num_envs"] == count
            assert kwargs["policy_seed"] == (10001 if worker is None else 10003)
            self.environment_seeds = list(
                range(kwargs["policy_seed"], kwargs["policy_seed"] + count)
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def reset(self):
            return {"state": np.zeros((count, 1, 9))}

        def step(self, value):
            np.testing.assert_array_equal(value, action[selection])
            return self.reset(), None

    monkeypatch.setattr(replay, "_DPPOSquareEnvClient", Env)
    monkeypatch.setattr(replay, "_resolve_dppo_runtime", lambda **kwargs: (None, None))
    for key in (
        "ROBOT_PRACTICE_ROLLOUT_FORENSICS",
        "ROBOT_PRACTICE_ROLLOUT_START_METHOD",
        "PYTHONFAULTHANDLER",
        "PYTHONUNBUFFERED",
        "ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP",
    ):
        monkeypatch.setenv(key, "")
    argv = [
        "replay",
        "--source",
        str(source),
        "--output-root",
        str(tmp_path / "out"),
        "--start-method",
        "spawn",
    ]
    if worker is not None:
        argv += ["--worker-index", str(worker)]
    monkeypatch.setattr(sys, "argv", argv)
    replay.main()
    assert "DONE replay completed=1 total=1" in (tmp_path / "out/run.log").read_text()


@pytest.mark.parametrize("fail", [False, True])
def test_serial_step_preserves_replies_and_restores_pipes(fail):
    from robot_practice.experiments.rl.rollout_forensics import serial_vector_step

    events = []

    class Pipe:
        def __init__(self, index):
            self.index = index

        def send(self, message):
            events.append(("send", self.index, message[1]))

        def recv(self):
            events.append(("recv", self.index))
            if fail and self.index == 1:
                raise EOFError("worker failed")
            return self.index

    class Vector:
        def __init__(self):
            self.parent_pipes = [Pipe(i) for i in range(3)]

        def step(self, action):
            for pipe, value in zip(self.parent_pipes, action):
                pipe.send(("step", value))
            assert all(pipe.poll() for pipe in self.parent_pipes)
            return [pipe.recv() for pipe in self.parent_pipes]

    vector = Vector()
    original = vector.parent_pipes
    if fail:
        with pytest.raises(EOFError):
            serial_vector_step(vector, [10, 20, 30])
        assert len(events) == 4
    else:
        assert serial_vector_step(vector, [10, 20, 30]) == [0, 1, 2]
        assert events == [
            ("send", 0, 10),
            ("recv", 0),
            ("send", 1, 20),
            ("recv", 1),
            ("send", 2, 30),
            ("recv", 2),
        ]
    assert vector.parent_pipes is original
