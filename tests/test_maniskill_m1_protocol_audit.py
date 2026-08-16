from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from robot_practice.experiments.maniskill.m1_protocol_audit import (
    _observation_schema,
    _stratified_episode_order,
    audit_trajectory,
    chunk_slicing_audit,
    freeze_plug_horizon,
    load_config,
    plug_horizon,
    validate_config,
)


def test_m1_config_is_frozen_before_results() -> None:
    config = load_config()
    result = validate_config(config)
    assert result["tasks"] == ["StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1"]
    assert config["tasks"]["PlugCharger-v1"]["formal_horizon"] is None
    assert config["protocol"]["backend"] == "physx_cpu"


def test_plug_horizon_rounding_is_deterministic() -> None:
    assert plug_horizon(124.9) == 200
    assert plug_horizon(150.0) == 300
    assert plug_horizon(249.9) == 500


def test_chunk_slicing_never_crosses_episode_boundary() -> None:
    result = chunk_slicing_audit([3, 20], n_obs_steps=2, horizon=16, n_action_steps=8)
    assert result["cross_episode_windows"] == 0
    assert result["valid_windows"] == 23
    assert result["episode_end_padded_actions"] > 0
    assert result["mask_policy"].startswith("pad_only")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required to reproduce ManiSkill tensor output"
)
def test_observation_schema_accepts_cuda_tensors() -> None:
    observation = {"sensor_data": {"base_camera": {"rgb": torch.zeros((2, 3), device="cuda")}}}
    schema = _observation_schema(observation)
    assert schema["sensor_data.base_camera.rgb"] == {"shape": [2, 3], "dtype": "float32"}


def test_stratified_order_is_reproducible_and_spreads_deciles() -> None:
    rows = [{"episode_id": index, "first_success_step": 10 + (index % 10)} for index in range(100)]
    first = _stratified_episode_order(rows, "StackCube-v1")
    second = _stratified_episode_order(rows, "StackCube-v1")
    assert [row["episode_id"] for row in first] == [row["episode_id"] for row in second]
    assert {row["first_success_decile"] for row in first[:10]} == set(range(10))


def test_trajectory_audit_keeps_replay_and_chunk_outputs_separate(tmp_path: Path) -> None:
    config = load_config()
    replay_dir = tmp_path / "StackCube-v1" / "replay_full"
    replay_dir.mkdir(parents=True)
    replay = replay_dir / "replay.h5"
    metadata = {
        "env_info": {
            "env_id": "StackCube-v1",
            "env_kwargs": {
                "control_mode": "pd_ee_delta_pos",
                "obs_mode": "none",
                "sim_backend": "physx_cpu",
            },
        },
        "episodes": [{"episode_id": 0, "success": True}],
    }
    replay.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with h5py.File(replay, "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("actions", data=np.zeros((4, 4), dtype=np.float32))
        group.create_dataset("success", data=np.asarray([False, False, True, True]))
        group.create_dataset("rewards", data=np.asarray([0.0, 0.0, 1.0, 1.0]))

    result = audit_trajectory(config, "StackCube-v1", override_output=tmp_path)
    assert result["chunks"]["2/16/8"]["cross_episode_windows"] == 0
    assert (tmp_path / "StackCube-v1" / "trajectory_audit.json").is_file()
    assert (tmp_path / "StackCube-v1" / "replay_full" / "replay.h5").is_file()


def test_trajectory_audit_counts_successful_replays_for_survival(tmp_path: Path) -> None:
    config = load_config()
    replay_dir = tmp_path / "StackCube-v1" / "replay_full"
    replay_dir.mkdir(parents=True)
    replay = replay_dir / "replay.h5"
    metadata = {
        "env_info": {
            "env_id": "StackCube-v1",
            "env_kwargs": {
                "control_mode": "pd_ee_delta_pos",
                "obs_mode": "none",
                "sim_backend": "physx_cpu",
            },
        },
        "episodes": [
            {"episode_id": 0, "success": True},
            {"episode_id": 1, "success": False},
        ],
    }
    replay.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with h5py.File(replay, "w") as handle:
        for episode_id, success in ((0, True), (1, False)):
            group = handle.create_group(f"traj_{episode_id}")
            group.create_dataset("actions", data=np.zeros((4, 4), dtype=np.float32))
            group.create_dataset(
                "success", data=np.asarray([False, False, success, success])
            )
            group.create_dataset("rewards", data=np.asarray([0.0, 0.0, float(success), float(success)]))

    result = audit_trajectory(config, "StackCube-v1", override_output=tmp_path)
    survival = result["gates"]["replay_survival"]
    assert survival["replay_episode_count"] == 2
    assert survival["successful_replay_count"] == 1
    assert survival["observed"] == pytest.approx(1 / 1000)
    assert survival["passed"] is False


def test_plug_horizon_uses_successful_replays_and_requires_survival(tmp_path: Path) -> None:
    config = load_config()
    task_root = tmp_path / "PlugCharger-v1"
    task_root.mkdir(parents=True)
    rows = [
        {
            "episode_id": index,
            "num_action_steps": 180,
            "success_once": index < 96,
            "first_success_step": 170 if index < 96 else None,
        }
        for index in range(100)
    ]
    (task_root / "trajectory_audit.json").write_text(
        json.dumps(
            {
                "episodes": rows,
                "gates": {
                    "replay_survival": {
                        "observed": 0.96,
                        "threshold": 0.95,
                        "passed": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = freeze_plug_horizon(config, override_output=tmp_path)
    assert result["formal_horizon"] == 400
    assert result["replay_episode_count"] == 100
    assert result["successful_replay_count"] == 96
    assert result["success_by_formal_horizon_count"] == 96
    assert result["success_by_formal_horizon_rate"] == pytest.approx(1.0)

    (task_root / "trajectory_audit.json").write_text(
        json.dumps(
            {
                "episodes": rows,
                "gates": {
                    "replay_survival": {
                        "observed": 0.58,
                        "threshold": 0.95,
                        "passed": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="replay survival gate"):
        freeze_plug_horizon(config, override_output=tmp_path)
