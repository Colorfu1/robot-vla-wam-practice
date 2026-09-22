import pytest
import torch
from torch import nn

from robot_practice.experiments.rl.fastwam_square_eval import load_weights_strict
from robot_practice.experiments.rl.lightwam_base import LightWAMBaseAdapter


def test_strict_load():
    model = nn.Module()
    model.mot = nn.Linear(2, 2)
    model.proprio_encoder = nn.Linear(9, 2)
    payload = {"mot": model.mot.state_dict(),
               "proprio_encoder": model.proprio_encoder.state_dict(), "step": 20262}
    load_weights_strict(model, payload)
    with pytest.raises(ValueError):
        load_weights_strict(model, {"mot": payload["mot"]})
    with pytest.raises(RuntimeError):
        load_weights_strict(model, {**payload, "mot": {}})


def test_shared_adapter_dispatches_fastwam_contract():
    class FakeFastWAM(nn.Module):
        def infer_action(self, **kwargs):
            assert kwargs["input_image"].shape == (3, 224, 224)
            assert kwargs["proprio"].shape == (9,)
            assert kwargs["num_inference_steps"] == 20
            assert kwargs["action_horizon"] == 4
            assert kwargs["prompt"] is None
            return {"action": torch.ones(4, 7) * kwargs["seed"]}

    adapter = LightWAMBaseAdapter(
        FakeFastWAM(), context=torch.zeros(2, 4096), context_mask=torch.ones(2),
        num_inference_steps=20, clip_actions=False,
    )
    result = adapter.predict_action(
        {"rgb": torch.zeros(2, 3, 96, 96, dtype=torch.uint8), "state": torch.zeros(2, 9)},
        seed=[2, 3],
    )
    assert result.shape == (2, 4, 7)
    assert result[0].eq(2).all() and result[1].eq(3).all()
