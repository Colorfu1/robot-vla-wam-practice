import json

import numpy as np
import pytest

from robot_practice.experiments.rl import fastwam_dual_visual_checks as visual
from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
from robot_practice.experiments.rl.fastwam_dual_train_fit_multigpu import merge


def test_selection_is_preclose_and_real_four_step_boundary():
    actions = np.zeros((20, 7))
    actions[7:, 6] = 1
    assert visual.preclose_frame(actions) == 6
    states = np.ones((4, 9))
    states[2] = 0
    assert visual.select_branch(states, np.zeros(9)) == (8, 0.0)
    with pytest.raises(ValueError, match="transition"):
        visual.preclose_frame(np.ones((20, 7)))


def test_one_shot_rgb_never_changes_proprio_or_controls_from_forecast(tmp_path, monkeypatch):
    observations, executed = [], []

    class Sim:
        def call(self, command, value):
            if command == "dual_observation":
                return {"rgb": np.zeros((3, 224, 448), dtype=np.uint8),
                        "state": np.arange(9, dtype=float)}
            if command == "step":
                executed.append(value[0].copy())
                return {"rewards": [0] * 4, "successes": [False] * 4}
            return {}

    def predict(adapter, obs, seed):
        observations.append({k: v.copy() for k, v in obs.items()})
        return np.full((32, 7), 0.25)

    monkeypatch.setattr(visual, "predict", predict)
    monkeypatch.setattr(visual, "forecast", lambda *a: 99.0)
    visual.continuation(Sim(), None, None, tmp_path / "branch", 4, 12, 0,
                        replacement=np.ones((3, 224, 448), dtype=np.uint8))
    assert observations[0]["rgb"].min() == 1
    assert observations[1]["rgb"].max() == 0
    assert all(np.array_equal(o["state"], np.arange(9)) for o in observations)
    assert all(a.shape == (4, 7) and (a == .25).all() for a in executed)


def test_visual_merge_has_no_fake_success_rate(tmp_path):
    shard = tmp_path / "gpu_0"
    shard.mkdir()
    data = {"status": "complete", "trajectory_ids": [0], "checkpoint_sha256": "hash",
            "max_steps": 600, "protocol": PROTOCOL, "visual_checks_only": True,
            "visual_cases": {"0": {"rollout_step": 48}}}
    (shard / "metrics.json").write_text(json.dumps(data))
    result = merge(tmp_path, [("0", [0])], "hash", 600, visual_checks_only=True)
    assert result["evaluated"] == 1
    assert "task_success_rate" not in result
    assert (tmp_path / "index.html").exists()
