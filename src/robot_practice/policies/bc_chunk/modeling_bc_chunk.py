"""One-pass clean-action regression with Diffusion Policy model capacity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor

from .configuration_bc_chunk import BCChunkConfig


class BCChunkPolicy(DiffusionPolicy):
    """Predict a complete clean action horizon from a zero trajectory query."""

    config_class = BCChunkConfig
    name = "bc_chunk"

    def __init__(self, config: BCChunkConfig, **kwargs):
        super().__init__(config, **kwargs)

    @property
    def bc_config(self) -> BCChunkConfig:
        return self.config

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.image_features:
            return batch
        stacked = dict(batch)
        for key in self.config.image_features:
            if self.config.n_obs_steps == 1 and stacked[key].ndim == 4:
                stacked[key] = stacked[key].unsqueeze(1)
        stacked[OBS_IMAGES] = torch.stack(
            [stacked[key] for key in self.config.image_features],
            dim=-4,
        )
        return stacked

    def predict_full_action_horizon(self, batch: dict[str, Tensor]) -> Tensor:
        if OBS_STATE not in batch:
            raise KeyError(f"Missing required key {OBS_STATE!r}")
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(
                f"Expected {self.config.n_obs_steps} observation steps, got {n_obs_steps}"
            )
        global_cond = self.diffusion._prepare_global_conditioning(batch)
        query = torch.zeros(
            (
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
            ),
            dtype=global_cond.dtype,
            device=global_cond.device,
        )
        timesteps = torch.full(
            (batch_size,),
            self.config.fixed_timestep,
            dtype=torch.long,
            device=global_cond.device,
        )
        return self.diffusion.unet(query, timesteps, global_cond=global_cond)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
    ) -> Tensor:
        if noise is not None:
            raise ValueError("BCChunk is deterministic and does not accept a noise tensor")
        queued_batch = {
            key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues
        }
        full_actions = self.predict_full_action_horizon(queued_batch)
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return full_actions[:, start:end]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        prepared = self._stack_images(batch)
        if ACTION not in prepared or "action_is_pad" not in prepared:
            raise KeyError("BCChunk training requires action and action_is_pad")
        if prepared[ACTION].ndim != 3:
            raise ValueError("action must have shape [batch,horizon,action_dim]")
        if prepared[ACTION].shape[1] != self.config.horizon:
            raise ValueError(
                f"Expected action horizon {self.config.horizon}, got {prepared[ACTION].shape[1]}"
            )

        prediction = self.predict_full_action_horizon(prepared)
        loss = F.mse_loss(prediction, prepared[ACTION])
        return loss, {"clean_action_mse": float(loss.detach())}
