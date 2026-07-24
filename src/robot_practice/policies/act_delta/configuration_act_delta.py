"""Configuration for ACT with optional delta action regularization."""

from __future__ import annotations

import math

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_delta")
@dataclass
class ACTDeltaConfig(ACTConfig):
    """ACT configuration with an extra action-delta regularization term."""

    delta_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.delta_loss_weight) or self.delta_loss_weight < 0:
            raise ValueError("delta_loss_weight must be finite and non-negative")
