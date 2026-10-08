from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_practice.experiments.rl.fastwam_protocol_parity import (
    compare_case,
    differences,
    selected_frames,
)


def test_difference_metrics_and_invalid_inputs():
    result = differences(torch.ones(2), torch.ones(2) + .25)
    assert result["mae"] == .25
    assert result["relative_l2"] == pytest.approx(.25)
    assert not result["exact_equal"]
    with pytest.raises(ValueError):
        differences(torch.ones(2), torch.tensor([float("nan"), 1]))


def test_selected_frames_stay_inside_complete_training_windows():
    actions = np.zeros((80, 7))
    actions[:30, 6] = -1
    actions[30:, 6] = 1
    assert selected_frames(actions) == [0, 29, 30, 34]
    assert all(f + 32 < len(actions) for f in selected_frames(actions))


def test_comparison_matrix_uses_same_noisy_actions(monkeypatch):
    import robot_practice.experiments.rl.fastwam_protocol_parity as module

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "prefill", lambda *args: {})
    seen = []

    class Model:
        device = "cpu"
        torch_dtype = torch.float32

        def __init__(self):
            self.train_video_scheduler = SimpleNamespace(add_noise=lambda x, n, t: n.clone())
            self.train_action_scheduler = SimpleNamespace(add_noise=lambda x, n, t: n.clone())

        def build_inputs(self, sample, tiled):
            return {"first_frame_latents": torch.zeros(1, 3, 1, 2, 2),
                    "input_latents": torch.zeros(1, 3, 3, 2, 2),
                    "action": sample["action"], "context": sample["context"],
                    "context_mask": sample["context_mask"]}

        def _encode_input_image_latents_tensor(self, x, tiled):
            return torch.zeros(1, 3, 1, 2, 2)

        def _append_proprio_to_context(self, context, mask, proprio):
            return context, mask

        def _predict_joint_noise(self, **kwargs):
            seen.append(kwargs["latents_action"].clone())
            return None, kwargs["latents_action"].clone()

        def _predict_action_noise(self, **kwargs):
            seen.append(kwargs["latents_action"].clone())
            return kwargs["latents_action"].clone()

        _predict_action_noise_with_cache = _predict_action_noise

    sample = {"video": torch.zeros(1, 3, 9, 2, 2), "action": torch.zeros(1, 32, 7),
              "context": torch.zeros(1, 2, 4), "context_mask": torch.ones(1, 2).bool(),
              "proprio": torch.zeros(1, 9, 9)}
    result = compare_case(Model(), sample, 42)
    assert len(seen) == 18
    for i in range(0, len(seen), 6):
        assert all(torch.equal(seen[i], x) for x in seen[i:i+6])
    assert result["vae_full_video_first_vs_single_image"]["exact_equal"]
    for step in result["timesteps"]:
        assert step["training_path_vs_inference_path"]["first4"]["max_abs"] == 0
