"""Prototype-ID conditioned deterministic BCChunk policy."""

from __future__ import annotations

import torch
from torch import Tensor

from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy
from robot_practice.policies.mode_conditioning import (
    PROTOTYPE_ID,
    ModeConditionedDiffusionModel,
    condition_metrics,
    validate_and_latch_prototype,
)

from .configuration_mode_bc_chunk import ModeBCChunkConfig


class ModeBCChunkPolicy(BCChunkPolicy):
    """BCChunk controlled by one rollout-persistent prototype ID."""

    config_class = ModeBCChunkConfig
    name = "mode_bc_chunk"

    def __init__(self, config: ModeBCChunkConfig, **kwargs):
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
        if noise is not None:
            raise ValueError("Mode-BCChunk is deterministic and does not accept noise")
        queued = {
            key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues
        }
        if PROTOTYPE_ID not in batch:
            raise KeyError(f"Missing required categorical condition {PROTOTYPE_ID!r}")
        queued[PROTOTYPE_ID] = batch[PROTOTYPE_ID]
        full_actions = self.predict_full_action_horizon(queued)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return full_actions[:, start:end]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        loss, metrics = super().forward(batch)
        return loss, {
            **metrics,
            **condition_metrics(self.diffusion.last_condition_stats),
        }
