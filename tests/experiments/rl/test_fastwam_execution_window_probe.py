import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_execution_window_probe import (
    KEYS,
    action_slices,
    compare_interval,
    splice,
)


def trace(n):
    return {**{k: np.arange(n + 1)[:, None].astype(float) for k in KEYS},
            **{k: np.arange(n)[:, None].astype(float)
               for k in ("actions", "rewards", "successes")}}


def test_splice_has_one_branch_state_and_no_missing_action():
    original = trace(12)
    intervention = {k: v[4:].copy() for k, v in original.items()}
    restored = splice(original, intervention, 4)
    assert all(np.array_equal(original[k], restored[k]) for k in original)
    assert all(v == 0 for v in compare_interval(original, restored, 0, 12).values())


def test_interval_checks_actions_as_well_as_states():
    original = trace(12)
    partial = {k: v[4:9 if k in KEYS else 8].copy() for k, v in original.items()}
    assert all(v == 0 for v in compare_interval(original, partial, 4, 8).values())
    partial["actions"][0] += .1
    with pytest.raises(ValueError, match="gate failed"):
        compare_interval(original, partial, 4, 8)


@pytest.mark.parametrize("execution", [4, 8, 16, 32])
def test_execution_uses_requested_prefix_and_truncates_only_at_budget(execution):
    raw = np.ones((32, 7)) * 2
    raw[:15, 6] = -1
    full = action_slices(raw, execution, 476)
    assert full.shape == (execution, 7)
    assert (full[:, :6] == 1).all()
    assert bool((full[:, 6] > 0).any()) == (execution >= 16)
    assert action_slices(raw, execution, 4).shape == (4, 7)


def test_nonfinite_prediction_fails():
    with pytest.raises(ValueError, match="Invalid"):
        action_slices(np.full((32, 7), np.nan), 4, 600)
