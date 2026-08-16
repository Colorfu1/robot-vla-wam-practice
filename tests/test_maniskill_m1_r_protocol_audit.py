from __future__ import annotations

from pathlib import Path

import robot_practice.experiments.maniskill.m1_r_protocol_audit as audit

CONFIG = Path("configs/experiments/maniskill/m1_r_policy_decorator_audit.json")


def test_m1_r_config_freezes_independent_ms2_v2_contract() -> None:
    config = audit._config(CONFIG)

    assert config["protocol_id"] == "m1_r_policy_decorator_ms2_v2"
    assert config["identity"]["dataset_env_id"] == "PegInsertionSide-v0"
    assert config["environment"]["runtime_env_id"] == "PegInsertionSide-v2"
    assert config["environment"]["action_dim"] == 7
    assert config["environment"]["observation_dim"] == 50
    assert config["environment"]["max_episode_steps"] == 200
    assert config["environment"]["reward_mode"] == "sparse"
    assert config["environment"]["reference_default_reward_mode"] == "normalized_dense"
    assert config["temporal"] == {
        "n_obs_steps": 2,
        "horizon": 16,
        "n_action_steps": 4,
        "execution_action_steps": 4,
        "cross_episode_windows": 0,
        "tail_policy": "pad_only_within_episode_and_mask",
    }


def test_offline_split_is_deterministic_and_fingerprint_based() -> None:
    fingerprints = {episode_id: f"fp-{episode_id:02d}" for episode_id in range(10)}

    splits = audit._split_episode_ids(fingerprints, development_count=2, test_count=3)

    assert splits == {
        "development": [0, 1],
        "test": [2, 3, 4],
        "train": [5, 6, 7, 8, 9],
    }


def test_fingerprint_changes_when_quantized_state_changes() -> None:
    import numpy as np

    state = np.zeros(4, dtype=np.float64)
    same = audit._fingerprint(state, 1e-6)
    different = audit._fingerprint(state + np.asarray([0.0, 0.0, 2e-6, 0.0]), 1e-6)

    assert same != different
