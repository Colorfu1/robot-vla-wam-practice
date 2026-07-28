"""Configuration for the deterministic BCChunk baseline."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("bc_chunk")
@dataclass
class BCChunkConfig(DiffusionConfig):
    """Diffusion-capacity network trained as one-pass clean-action regression."""

    fixed_timestep: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fixed_timestep != 0:
            raise ValueError("BCChunk fixed_timestep must be integer zero")
        if self.do_mask_loss_for_padding:
            raise ValueError("The PushT Pilot requires do_mask_loss_for_padding=false")
