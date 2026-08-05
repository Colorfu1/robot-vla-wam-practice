"""Configuration for route-field conditioned Diffusion Policy."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

from robot_practice.policies.route_field_conditioning import validate_route_field_config


@PreTrainedConfig.register_subclass("route_field_diffusion")
@dataclass
class RouteFieldDiffusionConfig(DiffusionConfig):
    """Diffusion config extended by one fixed spatial route map."""

    route_grid_size: int = 7
    route_field_channels: int = 4
    route_embedding_dim: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_route_field_config(self)
