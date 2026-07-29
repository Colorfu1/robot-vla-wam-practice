"""Prototype-ID conditioned stochastic Diffusion Policy."""

from __future__ import annotations

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torch import Tensor

from robot_practice.policies.mode_conditioning import (
    PROTOTYPE_ID,
    ModeConditionedDiffusionModel,
    condition_metrics,
    validate_and_latch_prototype,
)

from .configuration_mode_diffusion import ModeDiffusionConfig


class ModeDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy controlled by one rollout-persistent prototype ID."""

    config_class = ModeDiffusionConfig
    name = "mode_diffusion"

    def __init__(self, config: ModeDiffusionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.diffusion = ModeConditionedDiffusionModel(config)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._active_prototype_id: Tensor | None = None

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> Tensor:
        self._active_prototype_id = validate_and_latch_prototype(
            self._active_prototype_id,
            batch,
            num_prototypes=self.config.num_prototypes,
        )
        return super().select_action(batch, noise=noise)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> Tensor:
        queued = {
            key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues
        }
        if PROTOTYPE_ID not in batch:
            raise KeyError(f"Missing required categorical condition {PROTOTYPE_ID!r}")
        queued[PROTOTYPE_ID] = batch[PROTOTYPE_ID]
        return self.diffusion.generate_actions(queued, noise=noise)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        loss, _ = super().forward(batch)
        return loss, condition_metrics(self.diffusion.last_condition_stats)
