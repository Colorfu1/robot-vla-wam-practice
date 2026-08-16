from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from . import closed_loop_same_evaluator as same_evaluator
from . import compare_offline_mse as mse_probe
from . import rollout_train_samples as probe


def _make_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        for episode_id in range(6):
            group = handle.create_group(f"traj_{episode_id}")
            group.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
            env_states = group.create_group("env_states")
            actors = env_states.create_group("actors")
            actors.create_dataset("cubeA", data=np.full((4, 13), episode_id, dtype=np.float32))


def test_sample_rows_are_reproducible_and_use_env_state_zero(tmp_path: Path) -> None:
    h5_path = tmp_path / "trajectory.h5"
    split_path = tmp_path / "split_manifest.json"
    _make_h5(h5_path)
    split_path.write_text(json.dumps({"train": [0, 1, 2, 3, 4, 5]}), encoding="utf-8")

    first = probe._load_sample_rows(
        d_core_h5=h5_path,
        split_manifest=split_path,
        sample_count=3,
        seed=17,
    )
    second = probe._load_sample_rows(
        d_core_h5=h5_path,
        split_manifest=split_path,
        sample_count=3,
        seed=17,
    )
    assert [row["episode_id"] for row in first] == [row["episode_id"] for row in second]
    assert first[0]["state"]["actors"]["cubeA"] == [first[0]["episode_id"]] * 13
    assert first[0]["start_state_fingerprint"] == probe.state_fingerprint(first[0]["state"])
    assert first[0]["start_state_fingerprint"] != f"train-episode-{first[0]['episode_id']}"


def test_sample_rows_reject_non_train_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="restricted to the offline train split"):
        probe._split_episode_ids(tmp_path / "missing.json", "development")


def test_bare_cuda_is_normalized_to_logical_device_zero() -> None:
    assert probe._resolve_device("cuda") == probe.torch.device("cuda:0")
    assert probe._resolve_device("cuda:3") == probe.torch.device("cuda:3")
    assert probe._resolve_device("cpu") == probe.torch.device("cpu")


def test_train_development_gap_summary_flags_a_large_mse_gap() -> None:
    result = mse_probe.summarize_train_dev_gap(
        {"normalized_action_mse": 0.10},
        {"normalized_action_mse": 0.20},
    )
    assert result["development_minus_train_mse"] == 0.10
    assert result["development_to_train_mse_ratio"] == 2.0
    assert result["interpretation"] == "large_train_dev_gap_candidate"


def test_same_evaluator_state_bank_preserves_canonical_identity(tmp_path: Path) -> None:
    rows = [
        {
            "episode_id": 11,
            "source_group": "/traj_11",
            "source_action_count": 3,
            "state": {"actors": {"cubeA": [1.0, 2.0]}},
            "start_state_fingerprint": probe.state_fingerprint(
                {"actors": {"cubeA": [1.0, 2.0]}}
            ),
        }
    ]
    path = tmp_path / "development.jsonl"
    same_evaluator._write_train_state_bank(path, rows)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source_episode_id"] == 11
    assert payload["start_state_fingerprint"] == rows[0]["start_state_fingerprint"]
