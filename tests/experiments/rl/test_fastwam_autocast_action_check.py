from contextlib import contextmanager

import numpy as np
import pytest
import torch

from robot_practice.experiments.rl.fastwam_autocast_action_check import sample_pair, summary


def test_full_sampler_pair_keeps_input_seed_and_steps(monkeypatch):
    current = {"enabled": None}
    calls = []

    @contextmanager
    def autocast(**kwargs):
        assert kwargs["dtype"] == torch.bfloat16
        assert kwargs["device_type"] == "cuda"
        current["enabled"] = kwargs["enabled"]
        yield

    monkeypatch.setattr(torch, "autocast", autocast)

    class Adapter:
        num_inference_steps = 20

        def predict_action(self, cond, seed):
            assert self.num_inference_steps == 20
            calls.append((current["enabled"], seed, cond["rgb"].copy(), cond["state"].copy()))
            return torch.full((1, 32, 7), float(current["enabled"]))

    observation = {"rgb": np.ones((3, 224, 448), dtype=np.uint8), "state": np.arange(9.)}
    off, on, controls = sample_pair(Adapter(), observation, 123)
    assert [c[0] for c in calls] == [False, True, False, True]
    for _, seed, rgb, state in calls:
        assert seed == 123
        np.testing.assert_array_equal(rgb[0, 0], observation["rgb"])
        np.testing.assert_array_equal(state[0, 0], observation["state"])
    assert np.all(off == 0) and np.all(on == 1)
    assert max(controls.values()) == 0
    adapter = Adapter()
    adapter.num_inference_steps = 10
    with pytest.raises(ValueError, match="20"):
        sample_pair(adapter, observation, 123)


def test_summary_uses_final_actions_targets_and_clipped_execution():
    off = np.zeros((2, 32, 7))
    on = np.ones_like(off) * 2
    target = np.ones_like(off)
    result = summary(off, on, target)
    assert result["windows"] == 2
    assert result["bf16_autocast"]["executed_first4"]["mse_clipped"] == 0
    assert result["bf16_autocast"]["executed_first4"]["mse_raw"] == 1
    assert result["first4_arm_mse_improved_windows"] == 2
    assert result["first4_gripper_sign_disagreements"] == 8
