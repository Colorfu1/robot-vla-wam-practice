from __future__ import annotations

import errno
import json
from pathlib import Path

import numpy as np

from robot_practice.experiments.maniskill.eval_official_protocol import (
    _publish_directory,
    _rank_candidates,
    build_policy_observation_from_official,
)


def test_official_protocol_split_contract_is_cpu_single_env_holdout() -> None:
    payload = json.loads(
        Path("configs/experiments/maniskill/split_cpu.json").read_text(encoding="utf-8")
    )
    assert payload["sim_backend"] == "physx_cpu"
    assert payload["num_envs"] == 1
    assert payload["max_episode_steps"] == 100
    assert payload["ignore_terminations"] is True
    assert payload["sets"]["holdout"]["count"] == 400


def test_official_flattened_observation_keeps_only_the_21d_student_state() -> None:
    observation = {
        "rgb": np.zeros((2, 2, 128, 128, 3), dtype=np.uint8),
        "state": np.concatenate(
            (
                np.ones((2, 2, 18), dtype=np.float32),
                np.full((2, 2, 1), 7.0, dtype=np.float32),
                np.full((2, 2, 7), 8.0, dtype=np.float32),
                np.full((2, 2, 3), 9.0, dtype=np.float32),
            ),
            axis=-1,
        ),
    }
    result = build_policy_observation_from_official(observation)
    assert tuple(result["observation.images.base_camera"].shape) == (2, 3, 128, 128)
    assert tuple(result["observation.state"].shape) == (2, 21)
    assert result["observation.state"][0, :18].tolist() == [1.0] * 18
    assert result["observation.state"][0, -3:].tolist() == [9.0] * 3


def test_official_flattened_single_env_observation_keeps_the_two_frames() -> None:
    observation = {
        "rgb": np.zeros((2, 128, 128, 3), dtype=np.uint8),
        "state": np.zeros((2, 29), dtype=np.float32),
    }
    result = build_policy_observation_from_official(observation)
    assert tuple(result["observation.images.base_camera"].shape) == (1, 3, 128, 128)
    assert tuple(result["observation.state"].shape) == (1, 21)


def test_official_checkpoint_ranking_uses_success_once_then_terminal_then_earliest() -> None:
    ranked = _rank_candidates(
        [
            {
                "checkpoint_step": 5000,
                "success_once_rate": 0.8,
                "success_at_end_rate": 0.6,
            },
            {
                "checkpoint_step": 2500,
                "success_once_rate": 0.8,
                "success_at_end_rate": 0.6,
            },
            {
                "checkpoint_step": 7500,
                "success_once_rate": 0.9,
                "success_at_end_rate": 0.1,
            },
        ]
    )
    assert [item["checkpoint_step"] for item in ranked] == [7500, 2500, 5000]
    assert [item["rank"] for item in ranked] == [1, 2, 3]


def test_official_publish_directory_copies_on_cross_device(tmp_path: Path, monkeypatch) -> None:
    temporary = tmp_path / "temporary"
    destination = tmp_path / "published"
    (temporary / "nested").mkdir(parents=True)
    (temporary / "nested" / "payload.bin").write_bytes(b"payload")

    original_rename = Path.rename

    def raise_cross_device(self: Path, target: str | Path) -> Path:
        if self == temporary and Path(target) == destination:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", raise_cross_device)

    _publish_directory(temporary, destination)

    assert (destination / "nested" / "payload.bin").read_bytes() == b"payload"
    assert not temporary.exists()
