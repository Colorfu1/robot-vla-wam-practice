import numpy as np
import torch

from robot_practice.experiments.rl.fastwam_dual_image_pair import paired_predict, stage_frames


def test_stage_selection_covers_close_onset_and_no_crossing():
    actions = np.zeros((100, 7))
    actions[:40, 6] = -1
    actions[40:, 6] = 1
    frames = dict(stage_frames(actions))
    assert {0, 24, 32, 36, 39, 40, 41, 44, 48} == set(frames)
    assert all(f + 32 < len(actions) for f in frames)


def test_pair_changes_only_rgb_and_repeats_control():
    calls = []

    class Adapter:
        def predict_action(self, cond, seed):
            calls.append((cond, seed))
            return torch.ones(1, 32, 7) * float(cond["rgb"].mean())

    saved = np.zeros((3, 224, 448), dtype=np.uint8)
    live = np.ones_like(saved)
    state = np.arange(9, dtype=float)
    a, b, repeat = paired_predict(Adapter(), saved, live, state, 42)
    assert repeat == 0
    assert np.all(a == 0) and np.all(b == 1)
    assert len(calls) == 3
    for cond, seed in calls:
        np.testing.assert_array_equal(cond["state"][0, 0], state)
        assert seed == 42
