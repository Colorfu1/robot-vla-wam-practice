from __future__ import annotations

import numpy as np
import pytest

from robot_practice.evaluation.metrics import (
    action_jerk,
    action_tv,
    chunk_mae_by_horizon,
    compute_closed_loop_metrics,
    compute_offline_metrics,
    delta_action_mae,
    gripper_toggles,
    percentile,
)


def _chunk(values: list[float]) -> np.ndarray:
    return np.repeat(np.asarray(values, dtype=np.float64)[None, :, None], 7, axis=2)


def test_delta_action_mae_requires_both_adjacent_horizons_valid() -> None:
    target = _chunk([0.0, 1.0, 3.0])
    predicted = _chunk([0.0, 2.0, 4.0])
    valid = np.array([[True, True, False]])

    assert delta_action_mae(predicted, target, valid) == pytest.approx(1.0)
    assert chunk_mae_by_horizon(predicted, target, valid) == [0.0, 1.0, None]


def test_delta_action_mae_is_undefined_for_single_step_chunk() -> None:
    chunk = np.zeros((2, 1, 7), dtype=np.float64)
    assert delta_action_mae(chunk, chunk, np.ones((2, 1), dtype=bool)) is None


def test_tv_and_jerk_use_only_six_arm_dimensions() -> None:
    actions = np.array(
        [
            [0, 0, 0, 0, 0, 0, -1],
            [1, 1, 1, 1, 1, 1, 100],
            [3, 3, 3, 3, 3, 3, -100],
        ],
        dtype=np.float64,
    )

    assert action_tv(actions) == pytest.approx(1.5)
    assert action_jerk(actions) == pytest.approx(1.0)


def test_short_smoothness_sequences_are_null() -> None:
    assert action_tv(np.zeros((1, 7))) is None
    assert action_jerk(np.zeros((2, 7))) is None


def test_gripper_toggles_count_sign_transitions() -> None:
    actions = np.zeros((5, 7), dtype=np.float64)
    actions[:, -1] = [-1.0, -0.1, 0.0, 1.0, -1.0]
    assert gripper_toggles(actions) == 2


def test_percentile_uses_each_call_once() -> None:
    assert percentile([1.0, 2.0, 100.0], 50) == 2.0
    assert percentile([], 95) is None


def test_offline_aggregate_pools_errors_but_macro_averages_continuity() -> None:
    predicted = np.zeros((4, 2, 7), dtype=np.float64)
    target = np.zeros_like(predicted)
    predicted[3] = 2.0
    valid = np.ones((4, 2), dtype=bool)
    result = compute_offline_metrics(
        predicted_chunks=predicted,
        target_chunks=target,
        valid_horizon=valid,
        episode_indices=[1, 1, 1, 2],
        batch_size=2,
        model_query_latencies_ms=[10.0, 20.0],
        status="complete",
    )

    assert result["aggregate"]["first_action_mae"] == pytest.approx(0.5)
    assert result["aggregate"]["num_model_queries"] == 2
    assert result["aggregate"]["teacher_forced_action_tv"] == 0.0
    assert result["per_episode"][1]["teacher_forced_action_tv"] is None


def test_closed_loop_smoke_never_reports_formal_success_rate() -> None:
    episode = {
        "episode_index": 0,
        "seed": 1000,
        "init_state_id": 0,
        "actions": np.zeros((3, 7)),
        "rewards": np.array([0.0, 0.0, 1.0]),
        "success": True,
        "termination_reason": "success",
        "policy_step_latencies_ms": [1.0, 2.0, 3.0],
        "model_query_latencies_ms": [2.0],
    }

    smoke = compute_closed_loop_metrics([episode], formal=False, status="smoke")
    formal = compute_closed_loop_metrics([episode], formal=True, status="complete")

    assert smoke["aggregate"]["success_rate"] is None
    assert smoke["aggregate"]["num_successes"] == 1
    assert formal["aggregate"]["success_rate"] == 1.0
    assert formal["per_episode"][0]["gripper_toggles"] == 0


def test_metrics_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN or Infinity"):
        action_tv(np.full((2, 7), np.nan))
