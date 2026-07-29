"""LeRobot processor factory for Mode-BCChunk."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)

from .configuration_mode_bc_chunk import ModeBCChunkConfig


def make_mode_bc_chunk_pre_post_processors(
    config: ModeBCChunkConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    return make_diffusion_pre_post_processors(config, dataset_stats=dataset_stats)
