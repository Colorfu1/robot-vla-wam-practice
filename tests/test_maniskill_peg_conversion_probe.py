from __future__ import annotations

import numpy as np

from robot_practice.experiments.maniskill.peg_conversion_probe import (
    compact_axis_angle,
    fixed_delta_action,
    pose_error,
    quaternion_inverse,
    quaternion_multiply,
)


def test_quaternion_relative_contract_uses_target_times_current_inverse() -> None:
    current = np.asarray([1.0, 0.0, 0.0, 0.0])
    target = np.asarray([0.0, 0.0, 0.0, 1.0])
    relative = quaternion_multiply(target, quaternion_inverse(current))
    np.testing.assert_allclose(compact_axis_angle(relative), [0.0, 0.0, np.pi], atol=1e-7)


def test_fixed_delta_rotation_sign_matches_negative_rot_lower_contract() -> None:
    current = np.asarray([1.0, 0.0, 0.0, 0.0])
    target = np.asarray([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    action, diagnostics = fixed_delta_action(
        [0.0, 0.0, 0.0],
        target,
        [0.0, 0.0, 0.0],
        current,
        position_bound=0.1,
        rotation_bound=0.1,
    )
    assert diagnostics["rotation_clipped"] is True
    assert action[5] < 0.0
    assert np.isclose(np.linalg.norm(action[3:]), 1.0)


def test_fixed_delta_action_clips_translation_componentwise() -> None:
    action, diagnostics = fixed_delta_action(
        [0.2, -0.05, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(action[:3], [1.0, -0.5, 0.0])
    assert diagnostics["position_clipped"] is True


def test_pose_error_is_zero_for_identical_pose() -> None:
    position, rotation, position_norm, rotation_norm = pose_error(
        [0.1, 0.2, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [-1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(position, 0.0)
    np.testing.assert_allclose(rotation, 0.0)
    assert position_norm == 0.0
    assert rotation_norm == 0.0
