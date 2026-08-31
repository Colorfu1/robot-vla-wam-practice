from pathlib import Path

import torch

from robot_practice.experiments.rl.robomimic_dppo_residual import (
    FrozenDPPOBase,
    compose_bounded_residual,
    main,
    pixel_state_features,
)


def test_zero_residual_and_bound():
    base = torch.tensor([[[-0.8, 0.1, 0.0]]])
    residual = torch.tensor([[[10.0, -10.0, 0.2]]])
    bound = torch.tensor([[[0.05, 0.05, 0.1]]])
    out = compose_bounded_residual(base, residual, bound)
    assert torch.allclose(out, torch.tensor([[[-0.75, 0.05, 0.1]]]))
    assert torch.equal(compose_bounded_residual(base, torch.zeros_like(base), bound), base)


def test_frozen_base_hash_does_not_change():
    base = FrozenDPPOBase(torch.nn.Linear(4, 3))
    before = base.parameter_hash
    x = {"state": torch.zeros(1, 1, 4)}
    _ = base(x)
    base.assert_unchanged()
    assert base.parameter_hash == before


def test_pixel_whitelist_rejects_object_observation():
    obs = {"rgb": torch.zeros(1, 3, 96, 96), "state": torch.zeros(1, 9), "object": torch.zeros(1, 1)}
    try:
        pixel_state_features(obs)
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("object observation was accepted")


def test_dummy_train_pipeline(tmp_path: Path):
    config = Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml")
    assert main(["--phase=train", "--dummy", "--smoke", f"--config={config}", f"--output-root={tmp_path}"]) == 0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "progress.jsonl").exists()
    assert (tmp_path / "checkpoint.pt").exists()
    text = (tmp_path / "run.log").read_text()
    assert "STATUS completed=1 total=1" in text
    assert "DONE completed=1 total=1" in text
