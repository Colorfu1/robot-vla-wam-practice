import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from robot_practice.experiments.rl.dppo_square_bc_control import evaluate


def test_eval_failure_restores_training_rng(monkeypatch, tmp_path):
    import hydra

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)

    def fail(*args, **kwargs):
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError("intentional model construction failure")

    monkeypatch.setattr(hydra.utils, "instantiate", fail)
    agent = SimpleNamespace(ema_model=torch.nn.Linear(1, 1))
    # Creating the fixture also consumes torch RNG; reset to the captured state.
    torch.set_rng_state(expected[2])
    cfg = OmegaConf.create({"model": {"network": {"augment": True}}})
    with pytest.raises(RuntimeError, match="intentional"):
        evaluate(agent, cfg, SimpleNamespace(), tmp_path / "eval")
    assert random.getstate() == expected[0]
    actual = np.random.get_state()
    np.testing.assert_array_equal(actual[1], expected[1][1])
    assert actual[2:] == expected[1][2:]
    assert torch.equal(torch.get_rng_state(), expected[2])
