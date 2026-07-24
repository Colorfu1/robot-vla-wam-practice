from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from robot_practice.policies.act_delta.configuration_act_delta import ACTDeltaConfig
from robot_practice.policies.act_delta.loss_logging import LossComponentRecorder
from robot_practice.policies.act_delta.modeling_act_delta import ACTDeltaPolicy


class _FixedModel:
    def __init__(
        self,
        actions_hat: torch.Tensor,
    ) -> None:
        zeros = torch.zeros_like(actions_hat)
        self.actions_hat = actions_hat
        self.mu_hat = zeros
        self.log_sigma_x2_hat = zeros

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.actions_hat, (self.mu_hat, self.log_sigma_x2_hat)


def _policy(*, delta_loss_weight: float, use_vae: bool) -> tuple[ACTDeltaPolicy, torch.Tensor]:
    policy = ACTDeltaPolicy.__new__(ACTDeltaPolicy)
    policy.config = ACTDeltaConfig(
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (8,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        device="cpu",
        chunk_size=3,
        n_action_steps=3,
        use_vae=use_vae,
        pretrained_backbone_weights=None,
        push_to_hub=False,
        delta_loss_weight=delta_loss_weight,
    )

    actions = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
            [[0.0, 1.0], [2.0, 2.0], [4.0, 4.0]],
        ]
    )
    policy.model = _FixedModel(actions)
    target = torch.tensor(
        [
            [[1.5, 1.5], [2.0, 3.5], [4.0, 5.0]],
            [[-1.0, 0.0], [1.0, 1.0], [1.5, 2.0]],
        ]
    )
    return policy, target


def _assert_close(left: object, right: object) -> None:
    if isinstance(left, float) and isinstance(right, float):
        assert pytest.approx(left, rel=1e-6, abs=1e-6) == right
    else:
        assert pytest.approx(float(left), rel=1e-6, abs=1e-6) == float(right)


def test_act_delta_forward_adds_delta_loss_and_scales_by_weight() -> None:
    policy, target = _policy(delta_loss_weight=0.2, use_vae=False)
    batch = {
        ACTION: target,
        "action_is_pad": torch.tensor([[False, False, False], [False, True, True]]),
    }

    loss, loss_dict = policy.forward(batch)
    predicted = policy.model(batch)[0]

    valid = ~batch["action_is_pad"]
    expected_l1 = (
        F.l1_loss(batch[ACTION], predicted, reduction="none") * valid.unsqueeze(-1)
    ).mean()
    valid_pairs = valid[:, 1:] & valid[:, :-1]
    delta_error = (
        (predicted[:, 1:] - predicted[:, :-1]) - (target[:, 1:] - target[:, :-1])
    ).abs()
    expected_delta = (delta_error * valid_pairs.unsqueeze(-1)).sum() / valid_pairs.sum().clamp_min(
        1
    )
    expected_loss = expected_l1 + 0.2 * expected_delta

    _assert_close(loss_dict["l1_loss"], expected_l1.item())
    _assert_close(loss_dict["delta_action_loss"], expected_delta.item())
    assert loss_dict["delta_loss_weight"] == 0.2
    _assert_close(loss.item(), expected_loss.item())
    assert "kld_loss" not in loss_dict


def test_act_delta_forward_zero_delta_loss_when_no_valid_pairs() -> None:
    policy, target = _policy(delta_loss_weight=0.5, use_vae=False)
    batch = {
        ACTION: target,
        "action_is_pad": torch.tensor([[False, True, True], [False, True, True]]),
    }

    loss, loss_dict = policy.forward(batch)
    predicted = policy.model(batch)[0]
    valid = ~batch["action_is_pad"]
    expected_l1 = (
        F.l1_loss(batch[ACTION], predicted, reduction="none") * valid.unsqueeze(-1)
    ).mean()

    assert loss_dict["delta_action_loss"] == 0.0
    _assert_close(loss_dict["l1_loss"], expected_l1.item())
    _assert_close(loss.item(), expected_l1.item())
    assert "kld_loss" not in loss_dict


def test_act_delta_logs_windowed_loss_components_to_terminal_and_jsonl(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy, target = _policy(delta_loss_weight=0.2, use_vae=False)
    policy.training = True
    output_path = tmp_path / "train_loss_components.jsonl"
    logger = logging.getLogger("test.act_delta.losses")
    policy._loss_component_recorder = LossComponentRecorder(
        output_path=output_path,
        log_freq=2,
        logger=logger,
    )
    batch = {
        ACTION: target,
        "action_is_pad": torch.tensor([[False, False, False], [False, True, True]]),
    }

    caplog.set_level(logging.INFO, logger=logger.name)
    first_loss, first_dict = policy.forward(batch)
    assert not output_path.exists()
    second_loss, second_dict = policy.forward(batch)

    assert first_dict == second_dict
    _assert_close(first_loss.item(), second_loss.item())
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["step"] == 2
    assert record["window_size"] == 2
    _assert_close(record["l1_loss"], first_dict["l1_loss"])
    _assert_close(record["delta_action_loss"], first_dict["delta_action_loss"])
    _assert_close(
        record["weighted_delta_action_loss"],
        0.2 * first_dict["delta_action_loss"],
    )
    _assert_close(record["total_loss"], first_loss.item())
    assert "[A2 loss components] step=2 window=2" in caplog.text
