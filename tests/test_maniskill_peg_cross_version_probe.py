from robot_practice.experiments.maniskill.peg_cross_version_probe import THRESHOLDS, _divergence


def test_p3_thresholds_are_explicit_and_positive() -> None:
    assert set(THRESHOLDS) == {
        "eef_position_m",
        "eef_rotation_rad",
        "joint_position_rad",
        "peg_head_position_m",
        "peg_rotation_rad",
    }
    assert all(value > 0 for value in THRESHOLDS.values())


def test_p3_divergence_is_zero_for_identical_snapshots() -> None:
    snapshot = {
            "eef": {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
            "robot_qpos": [0] * 9,
            "peg": {"position": [1, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
            "peg_head": {"position": [2, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
    }
    result = _divergence(snapshot, snapshot)
    assert all(value == 0 for value in result.values())
