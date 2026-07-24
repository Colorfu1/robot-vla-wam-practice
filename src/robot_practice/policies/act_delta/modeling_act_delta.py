"""ACT policy with delta-action regularization."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

from .configuration_act_delta import ACTDeltaConfig
from .loss_logging import LossComponentRecorder

LOGGER = logging.getLogger(__name__)


class ACTDeltaPolicy(ACTPolicy):
    """Action Chunking policy that adds an L1 delta-action consistency term."""

    config_class = ACTDeltaConfig
    name = "act_delta"

    def __init__(
        self,
        config: ACTDeltaConfig,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self._loss_component_recorder = LossComponentRecorder.from_environment(LOGGER)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        # Reference:
        # lerobot/policies/act/modeling_act.py::ACTPolicy.forward
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        action_is_pad = batch["action_is_pad"]
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~action_is_pad.unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                -0.5
                * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())
            ).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            act_loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            act_loss = l1_loss

        valid = ~action_is_pad
        valid_pairs = valid[:, 1:] & valid[:, :-1]
        if valid_pairs.any():
            delta_pred = actions_hat[:, 1:] - actions_hat[:, :-1]
            delta_target = batch[ACTION][:, 1:] - batch[ACTION][:, :-1]
            delta_error = (delta_pred - delta_target).abs()
            delta_loss = (
                (delta_error * valid_pairs.unsqueeze(-1)).sum()
                / valid_pairs.sum().clamp_min(1)
            )
        else:
            delta_loss = torch.tensor(0.0, device=actions_hat.device, dtype=actions_hat.dtype)

        loss = act_loss + self.config.delta_loss_weight * delta_loss
        loss_dict.update(
            {
                "delta_action_loss": delta_loss.item(),
                "delta_loss_weight": self.config.delta_loss_weight,
                "loss": loss.item(),
            }
        )
        recorder = getattr(self, "_loss_component_recorder", None)
        if recorder is None and not hasattr(self, "_loss_component_recorder"):
            recorder = LossComponentRecorder.from_environment(LOGGER)
            self._loss_component_recorder = recorder
        if getattr(self, "training", False) and recorder is not None:
            recorder.update(
                l1_loss=l1_loss.item(),
                delta_action_loss=delta_loss.item(),
                delta_loss_weight=self.config.delta_loss_weight,
                total_loss=loss.item(),
            )

        return loss, loss_dict
