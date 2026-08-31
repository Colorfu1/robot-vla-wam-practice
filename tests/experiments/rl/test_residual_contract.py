from __future__ import annotations

import torch
from torch import nn

from robot_practice.experiments.rl.residual_contract import (
    assert_frozen,
    compose_residual_action,
    freeze_module,
    residual_saturation_fraction,
)


def _vectors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ones(12), torch.full((12,), -0.5), torch.full((12,), 0.5)


def test_zero_residual_preserves_base_and_gripper() -> None:
    base = torch.arange(2 * 4 * 14, dtype=torch.float32).reshape(2, 4, 14) / 100
    residual = torch.zeros(2, 4, 12)

    result = compose_residual_action(base, residual, *_vectors())

    torch.testing.assert_close(result[..., :12], torch.clamp(base[..., :12], -0.5, 0.5))
    torch.testing.assert_close(result[..., 12:], base[..., 12:])


def test_residual_is_scaled_and_clipped_only_on_arm() -> None:
    base = torch.zeros(1, 4, 14)
    base[..., 12:] = torch.tensor([0.25, -0.25])
    residual = torch.full((1, 4, 12), 100.0)

    result = compose_residual_action(
        base,
        residual,
        torch.full((12,), 2.0),
        torch.full((12,), -0.05),
        torch.full((12,), 0.05),
    )

    torch.testing.assert_close(result[..., :12], torch.full((1, 4, 12), 0.05))
    torch.testing.assert_close(result[..., 12:], base[..., 12:])
    assert residual_saturation_fraction(residual) == 1.0


def test_base_freeze_is_explicit() -> None:
    base = nn.Linear(4, 4)
    freeze_module(base)
    assert_frozen(base)
    assert not base.training

