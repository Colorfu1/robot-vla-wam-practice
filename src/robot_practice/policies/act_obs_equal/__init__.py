"""ACT variant whose valid action horizon is averaged per observation."""

from .configuration_act_obs_equal import ACTObsEqualConfig
from .modeling_act_obs_equal import ACTObsEqualPolicy

__all__ = ["ACTObsEqualConfig", "ACTObsEqualPolicy"]
