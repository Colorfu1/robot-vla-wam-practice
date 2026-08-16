from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from robot_practice.experiments.maniskill import m1_protocol_audit as audit


def _source_fixture(
    path: Path, *, reward_mode: str = "sparse", episode_count: int = 1
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "StackCube-v1",
                    "env_kwargs": {
                        "control_mode": "pd_joint_pos",
                        "obs_mode": "none",
                        "reward_mode": reward_mode,
                        "sim_backend": "physx_cpu",
                    },
                },
                "episodes": [
                    {"episode_id": episode_id, "episode_seed": episode_id}
                    for episode_id in range(episode_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(path, "w") as handle:
        for episode_id in range(episode_count):
            group = handle.create_group(f"traj_{episode_id}")
            group.create_dataset("actions", data=np.zeros((2, 8), dtype=np.float32))


def test_replay_runner_isolates_smoke_and_full_outputs(monkeypatch, tmp_path: Path) -> None:
    config = audit.load_config()
    source = tmp_path / "source" / "trajectory.h5"
    _source_fixture(source, episode_count=10)
    monkeypatch.setattr(audit, "source_h5", lambda _config, _task: source)

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        input_h5 = Path(command[command.index("--traj-path") + 1])
        output = input_h5.parent / "trajectory.none.pd_ee_delta_pos.physx_cpu.h5"
        with h5py.File(output, "w") as handle:
            for episode_id in range(10):
                group = handle.create_group(f"traj_{episode_id}")
                group.create_dataset("actions", data=np.zeros((2, 4), dtype=np.float32))
                group.create_dataset("success", data=np.asarray([False, True]))
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "episodes": [
                        {"episode_id": episode_id, "success": True} for episode_id in range(10)
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    smoke = audit.replay_task(
        config, "StackCube-v1", full=False, num_workers=10, override_output=tmp_path / "out"
    )
    full = audit.replay_task(
        config, "StackCube-v1", full=True, num_workers=10, override_output=tmp_path / "out"
    )
    assert smoke["phase"] == "replay_smoke"
    assert full["phase"] == "replay_full"
    assert smoke["replay_h5"] != full["replay_h5"]
    assert len(calls) == 2
    assert (tmp_path / "out" / "StackCube-v1" / "replay_smoke" / "phase_manifest.json").is_file()
    assert (tmp_path / "out" / "StackCube-v1" / "replay_full" / "phase_manifest.json").is_file()
    # A completed phase is resumable and does not launch a second replay.
    audit.replay_task(
        config, "StackCube-v1", full=False, num_workers=10, override_output=tmp_path / "out"
    )
    assert len(calls) == 2


def test_replay_runner_overrides_source_reward_mode_in_isolated_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    config = audit.load_config()
    source = tmp_path / "source" / "trajectory.h5"
    _source_fixture(source, reward_mode="dense", episode_count=10)
    monkeypatch.setattr(audit, "source_h5", lambda _config, _task: source)

    observed: dict[str, str] = {}

    def fake_run(command, **kwargs):
        input_h5 = Path(command[command.index("--traj-path") + 1])
        metadata = json.loads(input_h5.with_suffix(".json").read_text(encoding="utf-8"))
        observed["reward_mode"] = metadata["env_info"]["env_kwargs"]["reward_mode"]
        output = input_h5.parent / "trajectory.none.pd_ee_delta_pos.physx_cpu.h5"
        with h5py.File(output, "w") as handle:
            for episode_id in range(10):
                group = handle.create_group(f"traj_{episode_id}")
                group.create_dataset("actions", data=np.zeros((2, 4), dtype=np.float32))
                group.create_dataset("success", data=np.asarray([False, True]))
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "episodes": [
                        {"episode_id": episode_id, "success": True}
                        for episode_id in range(10)
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    audit.replay_task(
        config, "StackCube-v1", full=False, num_workers=10, override_output=tmp_path / "out"
    )

    assert observed["reward_mode"] == "sparse"
    source_metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    assert source_metadata["env_info"]["env_kwargs"]["reward_mode"] == "dense"


def test_full_replay_gate_fails_and_preserves_failure_manifest(monkeypatch, tmp_path: Path) -> None:
    config = audit.load_config()
    source = tmp_path / "source" / "trajectory.h5"
    _source_fixture(source, episode_count=10)
    monkeypatch.setattr(audit, "source_h5", lambda _config, _task: source)

    def fake_run(command, **kwargs):
        input_h5 = Path(command[command.index("--traj-path") + 1])
        output = input_h5.parent / "trajectory.none.pd_ee_delta_pos.physx_cpu.h5"
        with h5py.File(output, "w") as handle:
            for episode_id in range(10):
                group = handle.create_group(f"traj_{episode_id}")
                group.create_dataset("actions", data=np.zeros((2, 4), dtype=np.float32))
                group.create_dataset(
                    "success", data=np.asarray([False, episode_id != 9])
                )
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "episodes": [
                        {"episode_id": episode_id, "success": episode_id != 9}
                        for episode_id in range(10)
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="replay_full gate failed.*9/10"):
        audit.replay_task(
            config, "StackCube-v1", full=True, num_workers=10, override_output=tmp_path / "out"
        )

    failure_path = (
        tmp_path / "out" / "StackCube-v1" / "replay_full" / "phase_failure.json"
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["successful_replays"] == 9
    assert failure["expected_replays"] == 10
    assert not (
        tmp_path / "out" / "StackCube-v1" / "replay_full" / "phase_manifest.json"
    ).exists()


def test_old_full_manifest_cannot_be_reused(monkeypatch, tmp_path: Path) -> None:
    config = audit.load_config()
    phase_root = tmp_path / "out" / "StackCube-v1" / "replay_full"
    phase_root.mkdir(parents=True)
    (phase_root / "phase_manifest.json").write_text(
        json.dumps({"full": True, "successful_replays": None}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="not a hard-gate success"):
        audit.replay_task(
            config, "StackCube-v1", full=True, num_workers=10, override_output=tmp_path / "out"
        )
