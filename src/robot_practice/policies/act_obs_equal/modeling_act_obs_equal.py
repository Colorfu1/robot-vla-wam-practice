"""ACT policy with equal total L1 weight for every sampled observation."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

from .configuration_act_obs_equal import ACTObsEqualConfig


def observation_equal_l1_loss(
    actions_hat: torch.Tensor,
    actions_target: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> torch.Tensor:
    """Average action dimensions, then valid horizon steps, then observations."""

    if actions_hat.shape != actions_target.shape:
        raise ValueError(
            "Predicted and target actions must have identical shapes: "
            f"{tuple(actions_hat.shape)} != {tuple(actions_target.shape)}"
        )
    if actions_hat.ndim != 3:
        raise ValueError(
            "ACT actions must have shape [batch, chunk, action_dim], "
            f"got {tuple(actions_hat.shape)}"
        )
    if action_is_pad.shape != actions_hat.shape[:2]:
        raise ValueError(
            "action_is_pad must have shape [batch, chunk], "
            f"got {tuple(action_is_pad.shape)} for actions {tuple(actions_hat.shape)}"
        )

    valid = ~action_is_pad.bool()
    valid_count = valid.sum(dim=1)
    if bool((valid_count == 0).any()):
        raise ValueError("Every observation must have at least one valid target action")

    per_action_l1 = F.l1_loss(actions_hat, actions_target, reduction="none").mean(dim=-1)
    per_observation_l1 = (per_action_l1 * valid).sum(dim=1) / valid_count.clamp_min(1)
    return per_observation_l1.mean()


class ACTObsEqualPolicy(ACTPolicy):
    """ACT policy whose sampled observations contribute equally to L1 training."""

    config_class = ACTObsEqualConfig
    name = "act_obs_equal"

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Run ACT and compute observation-equal masked L1 loss."""

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, _ = self.model(batch)
        l1_loss = observation_equal_l1_loss(
            actions_hat,
            batch[ACTION],
            batch["action_is_pad"],
        )
        loss_value = l1_loss.item()
        return l1_loss, {"l1_loss": loss_value, "loss": loss_value}
