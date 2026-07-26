"""Configuration for the observation-equal ACT loss experiment."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_obs_equal")
@dataclass
class ACTObsEqualConfig(ACTConfig):
    """ACT configuration restricted to the deterministic observation-equal experiment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.use_vae:
            raise ValueError("act_obs_equal requires use_vae=false")
