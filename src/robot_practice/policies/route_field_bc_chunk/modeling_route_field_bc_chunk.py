"""Route-field conditioned deterministic BCChunk policy."""

from __future__ import annotations

import torch
from torch import Tensor

from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy
from robot_practice.policies.route_field_conditioning import (
    ROUTE_FIELD,
    RouteFieldConditionedDiffusionModel,
    route_condition_metrics,
    validate_and_latch_route_field,
)

from .configuration_route_field_bc_chunk import RouteFieldBCChunkConfig


class RouteFieldBCChunkPolicy(BCChunkPolicy):
    """BCChunk controlled by one rollout-persistent spatial route field."""

    config_class = RouteFieldBCChunkConfig
    name = "route_field_bc_chunk"

    def __init__(self, config: RouteFieldBCChunkConfig, **kwargs):
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
        if noise is not None:
            raise ValueError("RouteField-BCChunk is deterministic and does not accept noise")
        queued = {
            key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues
        }
        if ROUTE_FIELD not in batch:
            raise KeyError(f"Missing required spatial condition {ROUTE_FIELD!r}")
        queued[ROUTE_FIELD] = batch[ROUTE_FIELD]
        full_actions = self.predict_full_action_horizon(queued)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return full_actions[:, start:end]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        loss, metrics = super().forward(batch)
        return loss, {
            **metrics,
            **route_condition_metrics(self.diffusion.last_condition_stats),
        }
