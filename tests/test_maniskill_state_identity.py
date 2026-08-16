from __future__ import annotations

import h5py
import numpy as np
import pytest

from robot_practice.experiments.maniskill.state_identity import (
    invert_unique_fingerprints,
    start_state_fingerprint,
)


def test_start_state_is_stable_and_unique(tmp_path) -> None:
    path = tmp_path / "states.h5"
    with h5py.File(path, "w") as handle:
        for episode_id, offset in ((0, 0.0), (1, 1.0)):
            group = handle.create_group(f"traj_{episode_id}")
            states = group.create_group("env_states")
            states.create_dataset("qpos", data=np.asarray([[offset, 0.0], [offset + 1, 0.0]]))
            states.create_dataset("qvel", data=np.zeros((2, 2)))

        first = start_state_fingerprint(handle["traj_0"])
        second = start_state_fingerprint(handle["traj_1"])
        assert first == start_state_fingerprint(handle["traj_0"])
        assert first != second
        assert invert_unique_fingerprints({0: first, 1: second}, label="test") == {
            first: 0,
            second: 1,
        }


def test_duplicate_start_state_is_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.h5"
    with h5py.File(path, "w") as handle:
        fingerprints = {}
        for episode_id in (0, 1):
            group = handle.create_group(f"traj_{episode_id}")
            states = group.create_group("env_states")
            states.create_dataset("qpos", data=np.zeros((1, 2)))
            fingerprints[episode_id] = start_state_fingerprint(group)

    with pytest.raises(ValueError, match="duplicate start-state fingerprint"):
        invert_unique_fingerprints(fingerprints, label="duplicate-test")
