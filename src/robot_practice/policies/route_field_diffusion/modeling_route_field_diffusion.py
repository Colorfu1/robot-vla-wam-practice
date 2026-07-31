"""Route-field conditioned stochastic Diffusion Policy."""

from __future__ import annotations

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torch import Tensor

from robot_practice.policies.route_field_conditioning import (
    ROUTE_FIELD,
    RouteFieldConditionedDiffusionModel,
    route_condition_metrics,
    validate_and_latch_route_field,
)

from .configuration_route_field_diffusion import RouteFieldDiffusionConfig


class RouteFieldDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy controlled by one rollout-persistent spatial route field."""

    config_class = RouteFieldDiffusionConfig
    name = "route_field_diffusion"

    def __init__(self, config: RouteFieldDiffusionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.diffusion = RouteFieldConditionedDiffusionModel(config)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._active_route_field: Tensor | None = None

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> Tensor:
        self._active_route_field = validate_and_latch_route_field(
            self._active_route_field,
            batch,
            channels=self.config.route_field_channels,
            grid_size=self.config.route_grid_size,
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
        if ROUTE_FIELD not in batch:
            raise KeyError(f"Missing required spatial condition {ROUTE_FIELD!r}")
        queued[ROUTE_FIELD] = batch[ROUTE_FIELD]
        return self.diffusion.generate_actions(queued, noise=noise)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        loss, _ = super().forward(batch)
        return loss, route_condition_metrics(self.diffusion.last_condition_stats)
