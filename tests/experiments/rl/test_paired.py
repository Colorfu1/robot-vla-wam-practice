from __future__ import annotations

import pytest

from robot_practice.experiments.rl.paired import summarize_paired, validate_paired_records


def _row(key: str, state: str, success: bool) -> dict[str, object]:
    return {"episode_key": key, "reset_state_id": state, "success": success}


def test_paired_summary_requires_same_reset_state() -> None:
    base = [_row("a", "state-a", False)]
    treatment = [_row("a", "state-b", True)]

    with pytest.raises(ValueError, match="reset_state_id mismatch"):
        validate_paired_records(base, treatment)


def test_paired_summary_is_deterministic() -> None:
    base = [
        {**_row("a", "state-a", False), "policy_seed": 1},
        {**_row("b", "state-b", True), "policy_seed": 2},
    ]
    treatment = [
        {**_row("a", "state-a", True), "policy_seed": 1},
        {**_row("b", "state-b", True), "policy_seed": 2},
    ]

    first = summarize_paired(base, treatment, bootstrap_repeats=100)
    second = summarize_paired(base, treatment, bootstrap_repeats=100)

    assert first == second
    assert first["observed_gap"] == 0.5
    assert set(first["per_policy_seed"]) == {"1", "2"}


def test_paired_summary_requires_matching_action_contract() -> None:
    base = [
        {
            **_row("a", "state-a", False),
            "reset_seed": 1001,
            "requested_env_id": "PegInsertionSide-v2",
            "action_dim": 7,
            "n_obs_steps": 2,
            "action_horizon": 4,
            "execution_action_steps": 4,
        }
    ]
    treatment = [
        {
            **_row("a", "state-a", True),
            "reset_seed": 1001,
            "requested_env_id": "PegInsertionSide-v2",
            "action_dim": 7,
            "n_obs_steps": 2,
            "action_horizon": 8,
            "execution_action_steps": 4,
        }
    ]

    with pytest.raises(ValueError, match="action_horizon mismatch"):
        validate_paired_records(base, treatment)
