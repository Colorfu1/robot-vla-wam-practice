from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from robot_practice.policies.act_obs_equal.configuration_act_obs_equal import (
    ACTObsEqualConfig,
)
from robot_practice.policies.act_obs_equal.modeling_act_obs_equal import (
    ACTObsEqualPolicy,
    observation_equal_l1_loss,
)


class _FixedModel:
    def __init__(self, actions_hat: torch.Tensor) -> None:
        self.actions_hat = actions_hat

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[None, None]]:
        return self.actions_hat, (None, None)


def _config(*, use_vae: bool = False) -> ACTObsEqualConfig:
    return ACTObsEqualConfig(
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (8,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        device="cpu",
        chunk_size=3,
        n_action_steps=3,
        use_vae=use_vae,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )


def test_observation_equal_loss_averages_each_observation_before_batch() -> None:
    predicted = torch.zeros((2, 3, 2))
    target = torch.tensor(
        [
            [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
            [[8.0, 2.0], [100.0, 100.0], [200.0, 200.0]],
        ]
    )
    action_is_pad = torch.tensor(
        [
            [False, False, False],
            [False, True, True],
        ]
    )

    loss = observation_equal_l1_loss(predicted, target, action_is_pad)

    # Observation 0: mean([1, 2, 3]) = 2.
    # Observation 1: its sole valid action has mean error 5.
    assert loss.item() == pytest.approx((2.0 + 5.0) / 2.0)


def test_observation_equal_loss_ignores_padded_predictions_and_targets() -> None:
    action_is_pad = torch.tensor([[False, True, True]])
    baseline = observation_equal_l1_loss(
        torch.tensor([[[1.0, 3.0], [0.0, 0.0], [0.0, 0.0]]]),
        torch.zeros((1, 3, 2)),
        action_is_pad,
    )
    changed_padding = observation_equal_l1_loss(
        torch.tensor([[[1.0, 3.0], [-100.0, 500.0], [900.0, -700.0]]]),
        torch.tensor([[[0.0, 0.0], [800.0, -600.0], [-300.0, 400.0]]]),
        action_is_pad,
    )

    assert baseline.item() == pytest.approx(2.0)
    assert changed_padding.item() == pytest.approx(baseline.item())


def test_policy_forward_reports_observation_equal_l1() -> None:
    policy = ACTObsEqualPolicy.__new__(ACTObsEqualPolicy)
    policy.config = _config()
    policy.model = _FixedModel(torch.zeros((1, 3, 2)))
    batch = {
        ACTION: torch.tensor([[[2.0, 0.0], [4.0, 0.0], [100.0, 100.0]]]),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, loss_dict = policy.forward(batch)

    assert loss.item() == pytest.approx(1.5)
    assert loss_dict == pytest.approx({"l1_loss": 1.5, "loss": 1.5})


def test_observation_equal_config_and_policy_are_registered() -> None:
    assert PreTrainedConfig.get_choice_class("act_obs_equal") is ACTObsEqualConfig
    assert ACTObsEqualPolicy.config_class is ACTObsEqualConfig
    assert ACTObsEqualPolicy.name == "act_obs_equal"


def test_observation_equal_config_round_trips_as_local_checkpoint(tmp_path: Path) -> None:
    config = _config()
    config.save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(loaded, ACTObsEqualConfig)
    assert loaded.type == "act_obs_equal"
    assert loaded.use_vae is False


def test_observation_equal_config_rejects_vae() -> None:
    with pytest.raises(ValueError, match="requires use_vae=false"):
        _config(use_vae=True)


def test_observation_equal_loss_rejects_zero_valid_action_observation() -> None:
    with pytest.raises(ValueError, match="at least one valid"):
        observation_equal_l1_loss(
            torch.zeros((2, 3, 2)),
            torch.zeros((2, 3, 2)),
            torch.tensor(
                [
                    [False, False, False],
                    [True, True, True],
                ]
            ),
        )
