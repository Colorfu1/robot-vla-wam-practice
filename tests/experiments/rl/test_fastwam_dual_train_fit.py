import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_dual_train_fit import (
    action_errors,
    rollout,
    select_windows,
)


def test_training_windows_do_not_cross_trajectories_and_cover_grasp_transition():
    actions = np.zeros((150, 7))
    actions[:, 6] = -1
    actions[95:, 6] = 1
    selected = select_windows([70, 80], actions, 1, 3)
    assert min(selected) == 70 and max(selected) == 117
    assert set(range(92, 96)) <= set(selected)
    assert all(start + 32 < 150 for start in selected)


def test_action_error_distinguishes_executed_prefix_from_full_horizon():
    target = np.ones((2, 32, 7))
    prediction = target.copy()
    prediction[:, 4:] = -1
    result = action_errors(prediction, target)
    assert result["executed_first4"]["mse_clipped"] == 0
    assert result["executed_first4"]["gripper_close_recall"] == 1
    assert result["full32"]["mse_clipped"] == 3.5
    with pytest.raises(ValueError):
        action_errors(prediction[:, :4], target[:, :4])


def test_training_scene_policy_uses_live_dual_frames_and_executes_four(tmp_path):
    import torch

    class Sim:
        step = 0

        def call(self, command, value):
            if command == "reset":
                self.step = 0
            elif command == "dual_observation":
                return {"rgb": np.full((3, 224, 448), self.step, dtype=np.uint8),
                        "state": np.full(9, self.step)}
            elif command == "step":
                action, expected_frame = value
                assert action.shape == (4, 7)
                assert expected_frame is None
                assert np.all(action == 1)
                self.step += 4
                return {"rewards": [0, 0, 0, int(self.step == 8)],
                        "successes": [False, False, False, self.step == 8]}
            else:
                raise AssertionError(command)

    class Adapter:
        def __init__(self):
            self.seen = []

        def predict_action(self, cond, seed):
            self.seen.append(cond["state"][0, 0, 0])
            assert cond["rgb"].shape == (1, 1, 3, 224, 448)
            assert seed == 42
            return torch.full((1, 32, 7), 2.0)

    adapter = Adapter()
    row = rollout(Sim(), 0, tmp_path / "policy", np.zeros((50, 7)),
                  adapter=adapter, max_steps=8, seed=42)
    assert adapter.seen == [0, 4]
    assert row["control_steps"] == 8
    assert row["first_success_step"] == 8
    assert (tmp_path / "policy" / "rollout.gif").is_file()
