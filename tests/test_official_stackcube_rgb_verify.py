from __future__ import annotations

from robot_practice.experiments.maniskill.official_stackcube_rgb_verify import (
    _stackcube_18d_observation_wrapper,
)


def test_official_18d_evaluator_uses_the_training_observation_wrapper() -> None:
    wrapper = _stackcube_18d_observation_wrapper()

    assert wrapper.__name__ == "M1WFlattenRGBObservationWrapper"
