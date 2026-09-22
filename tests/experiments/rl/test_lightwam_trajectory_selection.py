from types import SimpleNamespace

import numpy as np
import pytest

from robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic import (
    _compute_metrics,
    trajectory_windows,
)


def test_selection_preserves_trajectory_boundaries():
    ds = SimpleNamespace(
        traj_lengths=[6, 7, 5], windows=[(0, 6), (1, 6), (6, 7), (7, 7), (8, 7), (13, 5)]
    )
    indices, records = trajectory_windows(ds, [0, 2])
    assert indices == [0, 1, 5]
    assert [r["frame"] for r in records] == [0, 1, 0]
    assert [r["trajectory_id"] for r in records] == [0, 0, 2]
    for ids in ([], [0, 0], [-1], [3]):
        with pytest.raises(ValueError):
            trajectory_windows(ds, ids)


def test_gripper_error_is_separate_from_arm():
    target = np.ones((2, 4, 7), dtype=np.float32)
    pred = target.copy()
    pred[0, :, -1] = -1
    result = _compute_metrics(target, pred, action_dim=7, action_horizon=4)
    assert result["arm_mse_clipped"] == 0
    assert result["gripper_mse_clipped"] == 2
    assert result["gripper_sign_accuracy"] == 0.5
    assert result["gripper_close_recall"] == 0.5
    assert result["gripper_target_close_count"] == 8
