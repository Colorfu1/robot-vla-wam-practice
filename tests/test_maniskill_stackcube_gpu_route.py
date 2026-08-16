from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from robot_practice.experiments.maniskill import convert_stackcube as converter
from robot_practice.experiments.maniskill import eval_stackcube_gpu as rollout
from robot_practice.experiments.maniskill.state_identity import state_fingerprint


def _state(offset: float = 0.0) -> dict[str, dict[str, list[float]]]:
    return {
        "actors": {"cubeA": [offset] * 13, "cubeB": [offset + 1.0] * 13},
        "articulations": {"panda_wristcam": [offset + 2.0] * 31},
    }


def _write_state_bank(path: Path) -> None:
    rows = []
    for index in range(2):
        state = _state(float(index))
        rows.append(
            {
                "index": index,
                "state": state,
                "start_state_fingerprint": state_fingerprint(state),
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_dcore(path: Path) -> None:
    metadata = {
        "env_info": {
            "env_id": "StackCube-v1",
            "env_kwargs": {
                "sim_backend": "physx_cuda",
                "control_mode": "pd_ee_delta_pose",
            },
        },
        "episodes": [
            {"episode_id": 0, "source_episode_id": 7, "success": True},
        ],
    }
    with h5py.File(path, "w") as handle:
        episode = handle.create_group("traj_0")
        episode.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))
        obs = episode.create_group("obs")
        agent = obs.create_group("agent")
        agent.create_dataset("qpos", data=np.zeros((3, 9), dtype=np.float32))
        agent.create_dataset("qvel", data=np.ones((3, 9), dtype=np.float32))
        sensor = obs.create_group("sensor_data")
        for camera in ("base_camera", "hand_camera"):
            sensor.create_group(camera).create_dataset(
                "rgb", data=np.zeros((3, 128, 128, 3), dtype=np.uint8)
            )
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_stackcube_features_and_frame_contract(tmp_path: Path) -> None:
    replay = tmp_path / "trajectory.h5"
    _write_dcore(replay)

    with h5py.File(replay, "r") as handle:
        frames = list(converter.iter_episode_frames(handle["traj_0"]))

    assert len(frames) == 2
    assert set(frames[0]) == {
        converter.BASE_CAMERA_KEY,
        converter.HAND_CAMERA_KEY,
        converter.STATE_KEY,
        converter.ACTION_KEY,
        "task",
    }
    assert frames[0][converter.STATE_KEY].shape == (18,)
    assert frames[0][converter.ACTION_KEY].shape == (7,)
    assert converter.dataset_features()[converter.HAND_CAMERA_KEY]["dtype"] == "video"


def test_stackcube_frame_contract_rejects_obs_without_terminal_frame(tmp_path: Path) -> None:
    replay = tmp_path / "trajectory.h5"
    _write_dcore(replay)
    with h5py.File(replay, "a") as handle:
        del handle["traj_0"]["obs"]["agent"]["qvel"]
        handle["traj_0"]["obs"]["agent"].create_dataset(
            "qvel", data=np.ones((2, 9), dtype=np.float32)
        )
    with h5py.File(replay, "r") as handle:
        with pytest.raises(ValueError, match=r"obs=T\+1"):
            list(converter.iter_episode_frames(handle["traj_0"]))


def test_state_bank_uses_fingerprint_as_identity(tmp_path: Path) -> None:
    bank = tmp_path / "development.jsonl"
    _write_state_bank(bank)
    rows = rollout.load_state_bank(bank)
    assert [row["index"] for row in rows] == [0, 1]
    assert all(
        state_fingerprint(row["state"]) == row["start_state_fingerprint"] for row in rows
    )

    broken = tmp_path / "broken.jsonl"
    payload = json.loads(bank.read_text(encoding="utf-8").splitlines()[0])
    payload["start_state_fingerprint"] = "0" * 64
    broken.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        rollout.load_state_bank(broken)


def test_stackcube_policy_observation_is_two_camera_18d() -> None:
    observation = {
        "sensor_data": {
            "base_camera": {"rgb": torch.zeros((2, 128, 128, 3), dtype=torch.uint8)},
            "hand_camera": {"rgb": torch.ones((2, 128, 128, 3), dtype=torch.uint8)},
        },
        "agent": {
            "qpos": torch.zeros((2, 9), dtype=torch.float32),
            "qvel": torch.ones((2, 9), dtype=torch.float32),
        },
    }
    result = rollout.build_stackcube_policy_observation(observation)
    assert tuple(result[rollout.BASE_IMAGE_KEY].shape) == (2, 3, 128, 128)
    assert tuple(result[rollout.HAND_IMAGE_KEY].shape) == (2, 3, 128, 128)
    assert tuple(result[rollout.STATE_KEY].shape) == (2, 18)


def test_worker_command_is_state_bank_based_and_preserves_gpu_contract(tmp_path: Path) -> None:
    arguments = rollout.build_parser().parse_args(
        [
            "--policy-path",
            "outputs/selected/pretrained_model",
            "--dataset-root",
            "datasets/lerobot/maniskill/stackcube_dcore_7d",
            "--state-bank-root",
            "outputs/state_banks",
            "--seed-set",
            "development",
            "--max-episodes",
            "200",
            "--max-steps",
            "500",
            "--num-envs",
            "64",
            "--num-gpus",
            "8",
            "--output-dir",
            str(tmp_path / "worker"),
        ]
    )
    command = rollout._worker_command(arguments, worker_index=3, output_dir=tmp_path / "worker")
    assert "eval_stackcube_gpu" in " ".join(command)
    assert "--state-bank-root" in command
    assert "--max-episodes" in command
    assert "500" in command
    assert "--worker-count" in command and "8" in command
    assert "--split-file" not in command
