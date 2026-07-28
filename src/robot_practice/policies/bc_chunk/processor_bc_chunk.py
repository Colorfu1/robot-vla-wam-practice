"""LeRobot processor factory for BCChunk."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)

from .configuration_bc_chunk import BCChunkConfig


def make_bc_chunk_pre_post_processors(
    config: BCChunkConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    return make_diffusion_pre_post_processors(config, dataset_stats=dataset_stats)
