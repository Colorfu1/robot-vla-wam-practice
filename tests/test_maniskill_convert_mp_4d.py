from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from robot_practice.experiments.maniskill.convert_mp_4d import audit_replay


def _write_synthetic_replay(tmp_path: Path) -> tuple[Path, Path]:
    replay = tmp_path / "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
    source = tmp_path / "trajectory.h5"
    metadata = {
        "env_info": {
            "env_id": "PickCube-v1",
            "env_kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_ee_delta_pos",
                "sim_backend": "physx_cpu",
            },
        },
        "episodes": [
            {"episode_id": 0, "episode_seed": 17},
            {"episode_id": 1, "episode_seed": 19},
        ],
    }
    replay.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    replay_metadata = dict(metadata)
    replay_metadata["episodes"] = [
        {"episode_id": 0, "episode_seed": 19},
        {"episode_id": 1, "episode_seed": 17},
    ]
    replay.with_suffix(".json").write_text(json.dumps(replay_metadata), encoding="utf-8")
    source.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with h5py.File(replay, "w") as handle:
        for episode_id in (0, 1):
            episode = handle.create_group(f"traj_{episode_id}")
            episode.create_dataset(
                "actions",
                data=np.asarray(
                    [[0.1, 0.0, 0.0, -1.0]] * 9 + [[0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            )
            episode.create_dataset("success", data=np.asarray([False] * 9 + [True]))
            obs = episode.create_group("obs")
            camera = obs.create_group("sensor_data").create_group("base_camera")
            camera.create_dataset("rgb", data=np.zeros((11, 128, 128, 3), dtype=np.uint8))
            agent = obs.create_group("agent")
            agent.create_dataset("qpos", data=np.zeros((11, 9), dtype=np.float32))
            agent.create_dataset("qvel", data=np.zeros((11, 9), dtype=np.float32))
            obs.create_group("extra").create_dataset(
                "goal_pos", data=np.zeros((11, 3), dtype=np.float32)
            )
    return replay, source


def test_motionplanning_4d_audit_checks_the_fixed_contract(tmp_path: Path) -> None:
    replay, source = _write_synthetic_replay(tmp_path)
    result = audit_replay(replay, source_h5=source)
    assert result["gate_pass"] is True
    assert result["summary"]["replay_survival_rate"] == 1.0
    assert result["summary"]["total_frames"] == 20
    assert result["actions"]["action_dim"] == 4
    assert result["actions"]["out_of_bounds_fraction"] == 0.0
    assert result["observations"]["rgb_shapes"] == [[128, 128, 3]]
    assert result["summary"]["replayed_source_episode_ids"] == [0, 1]
    assert [row["episode_id"] for row in result["episodes"]] == [0, 1]
    assert [row["replay_episode_id"] for row in result["episodes"]] == [1, 0]
