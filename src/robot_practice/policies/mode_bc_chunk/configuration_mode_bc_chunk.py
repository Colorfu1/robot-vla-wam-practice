"""Configuration for prototype-ID conditioned BCChunk."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from robot_practice.policies.bc_chunk.configuration_bc_chunk import BCChunkConfig
from robot_practice.policies.mode_conditioning import validate_mode_config


@PreTrainedConfig.register_subclass("mode_bc_chunk")
@dataclass
class ModeBCChunkConfig(BCChunkConfig):
    """BCChunk config extended by one categorical global condition."""

    num_prototypes: int = 2
    mode_embedding_dim: int = 32
    condition_dropout_prob: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_mode_config(self)
