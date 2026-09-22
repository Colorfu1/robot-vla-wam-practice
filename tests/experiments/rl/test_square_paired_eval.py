import numpy as np
import pytest

from robot_practice.experiments.rl.square_paired_eval import batch_rows, summarize_batches


def test_budgets_and_chunk_metric():
    rewards = np.zeros((150, 3))
    rewards[99, 0] = 4
    rewards[100, 1] = 4
    rewards[120, 2] = 3
    result = {
        "per_episode": [
            {"env_index": i, "environment_seed": 10001 + i, "success": i < 2} for i in range(3)
        ]
    }
    rows = batch_rows(result, rewards)
    metrics = summarize_batches(rows, 80)
    assert metrics["success_rate_400"] == 1 / 3
    assert metrics["success_rate_600"] == 2 / 3
    assert rows[1]["first_success_chunk_end_control_step"] == 404
    assert metrics["evaluated"] == 3 and metrics["requested"] == 80


def test_reject_wrong_budget():
    with pytest.raises(ValueError):
        batch_rows({}, np.zeros((100, 8)))


def test_empty_results_not_zero_success():
    assert summarize_batches([], 80)["success_rate_600"] is None
