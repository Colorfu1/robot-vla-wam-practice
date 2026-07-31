"""Shared spatial route-field conditioning for local PushT policy variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
)
from torch import Tensor, nn

ROUTE_FIELD = "route_field"


def validate_route_field_config(config: Any) -> None:
    if int(config.route_grid_size) != 7:
        raise ValueError("The RouteField pilot requires route_grid_size=7")
    if int(config.route_field_channels) != 4:
        raise ValueError("The RouteField pilot requires route_field_channels=4")
    if int(config.route_embedding_dim) <= 0:
        raise ValueError("route_embedding_dim must be positive")


def _base_global_condition_dim(model: DiffusionModel) -> int:
    config = model.config
    dimension = int(config.robot_state_feature.shape[0])
    if config.image_features:
        if config.use_separate_rgb_encoder_per_camera:
            dimension += int(model.rgb_encoder[0].feature_dim) * len(config.image_features)
        else:
            dimension += int(model.rgb_encoder.feature_dim) * len(config.image_features)
    if config.env_state_feature:
        dimension += int(config.env_state_feature.shape[0])
    return dimension * int(config.n_obs_steps)


def normalize_route_fields(
    value: Tensor,
    *,
    batch_size: int,
    channels: int,
    grid_size: int,
) -> Tensor:
    """Validate and return one spatial field per batch item."""

    if value.ndim == 3 and batch_size == 1:
        value = value.unsqueeze(0)
    expected = (batch_size, channels, grid_size, grid_size)
    if value.ndim != 4 or tuple(value.shape) != expected:
        raise ValueError(f"{ROUTE_FIELD} must have shape {expected}, got {tuple(value.shape)}")
    if not value.is_floating_point():
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{ROUTE_FIELD} must contain only finite values")
    return value


class RouteFieldEncoder(nn.Module):
    """Small spatial encoder for the fixed 7x7 route map."""

    def __init__(self, *, channels: int, grid_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        self.network = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(32 * self.grid_size * self.grid_size, int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
            nn.SiLU(),
        )

    def forward(self, field: Tensor) -> Tensor:
        return self.network(field)


class RouteFieldConditionedDiffusionModel(DiffusionModel):
    """Diffusion model whose global FiLM condition includes a route-map embedding."""

    def __init__(self, config: Any):
        validate_route_field_config(config)
        super().__init__(config)
        self.route_field_encoder = RouteFieldEncoder(
            channels=int(config.route_field_channels),
            grid_size=int(config.route_grid_size),
            embedding_dim=int(config.route_embedding_dim),
        )
        global_cond_dim = _base_global_condition_dim(self) + int(config.route_embedding_dim)
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)
        self.last_condition_stats: dict[str, float] = {
            "embedding_l2_mean": 0.0,
            "occupancy_mean": 0.0,
        }

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        observation_condition = super()._prepare_global_conditioning(batch)
        if ROUTE_FIELD not in batch:
            raise KeyError(f"Missing required spatial condition {ROUTE_FIELD!r}")
        fields = normalize_route_fields(
            batch[ROUTE_FIELD],
            batch_size=observation_condition.shape[0],
            channels=int(self.config.route_field_channels),
            grid_size=int(self.config.route_grid_size),
        ).to(
            device=observation_condition.device,
            dtype=observation_condition.dtype,
        )
        embedding = self.route_field_encoder(fields)
        self.last_condition_stats = {
            "embedding_l2_mean": float(embedding.detach().norm(dim=-1).mean().item()),
            "occupancy_mean": float(fields[:, 0].detach().mean().item()),
        }
        return torch.cat((observation_condition, embedding), dim=-1)


def validate_and_latch_route_field(
    active: Tensor | None,
    batch: Mapping[str, Tensor],
    *,
    channels: int,
    grid_size: int,
) -> Tensor:
    """Latch one caller-supplied field for a complete rollout episode."""

    if ROUTE_FIELD not in batch:
        raise KeyError(f"Missing required spatial condition {ROUTE_FIELD!r}")
    raw = batch[ROUTE_FIELD]
    batch_size = int(raw.shape[0]) if raw.ndim == 4 else 1
    fields = normalize_route_fields(
        raw,
        batch_size=batch_size,
        channels=channels,
        grid_size=grid_size,
    ).detach()
    if active is not None and not torch.equal(active.to(device=fields.device), fields):
        raise ValueError(
            "route_field changed during an active rollout; call policy.reset() "
            "before selecting a different route"
        )
    return fields.clone()


def route_condition_metrics(stats: Mapping[str, float]) -> dict[str, float]:
    return {f"condition/{name}": float(value) for name, value in stats.items()}


__all__ = [
    "ROUTE_FIELD",
    "RouteFieldConditionedDiffusionModel",
    "RouteFieldEncoder",
    "normalize_route_fields",
    "route_condition_metrics",
    "validate_and_latch_route_field",
    "validate_route_field_config",
]
