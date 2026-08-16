from robot_practice.experiments.maniskill.peg_episode2_diagnostic import (
    JOINT_TARGET_DIVERGENCE_THRESHOLD_RAD,
    POSITION_DIVERGENCE_THRESHOLD_M,
    ROTATION_DIVERGENCE_THRESHOLD_RAD,
    SOURCE_EPISODE_ID,
)


def test_episode2_diagnostic_contract_is_frozen() -> None:
    assert SOURCE_EPISODE_ID == 2
    assert POSITION_DIVERGENCE_THRESHOLD_M == 0.005
    assert ROTATION_DIVERGENCE_THRESHOLD_RAD == 0.05
    assert JOINT_TARGET_DIVERGENCE_THRESHOLD_RAD == 0.1
