"""ACT policy variant with an auxiliary delta action loss term."""

from .configuration_act_delta import ACTDeltaConfig
from .modeling_act_delta import ACTDeltaPolicy

__all__ = ["ACTDeltaConfig", "ACTDeltaPolicy"]
