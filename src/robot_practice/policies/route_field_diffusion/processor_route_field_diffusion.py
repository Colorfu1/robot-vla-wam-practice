"""LeRobot processor factory for RouteField-Diffusion."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

from .configuration_route_field_diffusion import RouteFieldDiffusionConfig


def make_route_field_diffusion_pre_post_processors(
    config: RouteFieldDiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    return make_diffusion_pre_post_processors(config, dataset_stats=dataset_stats)
