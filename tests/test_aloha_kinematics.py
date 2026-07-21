from __future__ import annotations

import numpy as np
import pytest

from robot_practice.aloha_kinematics import (
    PUPPET_GRIPPER_HALF_OPENING_CLOSED_M,
    PUPPET_GRIPPER_HALF_OPENING_OPEN_M,
    TOOL_OFFSET_M,
    base_transform,
    forward_arm,
    forward_bimanual,
    normalized_gripper_to_half_opening,
)


def test_zero_pose_matches_published_home_end_effector_positions() -> None:
    left = forward_arm(np.zeros(7), "left")
    right = forward_arm(np.zeros(7), "right")

    np.testing.assert_allclose(left.end_effector_translation, [0.067494, 0.5, 0.42705])
    np.testing.assert_allclose(right.end_effector_translation, [-0.067494, 0.5, 0.42705])
    np.testing.assert_allclose(left.end_effector_rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(right.end_effector_rotation, base_transform("right")[:3, :3])
    assert TOOL_OFFSET_M == pytest.approx(0.1072)


def test_forward_bimanual_splits_left_then_right() -> None:
    state = np.array(
        [0.2, -0.5, 0.8, 0.3, -0.4, 0.7, 0.25] * 2,
        dtype=np.float64,
    )
    pose = forward_bimanual(state)

    np.testing.assert_allclose(pose.left.joint_points[0], [-0.469, 0.5, 0.0])
    np.testing.assert_allclose(pose.right.joint_points[0], [0.469, 0.5, 0.0])
    for arm in (pose.left, pose.right):
        np.testing.assert_allclose(
            arm.end_effector_rotation.T @ arm.end_effector_rotation,
            np.eye(3),
            atol=1e-12,
        )
        assert np.linalg.det(arm.end_effector_rotation) == pytest.approx(1.0)


def test_nonzero_pose_matches_independent_mujoco_golden_values() -> None:
    joints = [0.2, -0.5, 0.8, 0.3, -0.4, 0.7, 0.5]
    left = forward_arm(joints, "left")
    right = forward_arm(joints, "right")

    np.testing.assert_allclose(
        left.end_effector_translation,
        [-0.102149683881, 0.553587192417, 0.344943325753],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        left.end_effector_rotation,
        [
            [0.992995925485, -0.115571407652, 0.024542650710],
            [0.083868635927, 0.543197962129, -0.835405306332],
            [0.083217449362, 0.831612423953, 0.549086179437],
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        right.end_effector_translation,
        [0.102149683881, 0.446412807583, 0.344943325753],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        right.end_effector_rotation,
        [
            [-0.992995925485, 0.115571407652, -0.024542650710],
            [-0.083868635927, -0.543197962129, 0.835405306332],
            [0.083217449362, 0.831612423953, 0.549086179437],
        ],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.1, PUPPET_GRIPPER_HALF_OPENING_CLOSED_M),
        (0.0, PUPPET_GRIPPER_HALF_OPENING_CLOSED_M),
        (1.0, PUPPET_GRIPPER_HALF_OPENING_OPEN_M),
        (1.1, PUPPET_GRIPPER_HALF_OPENING_OPEN_M),
    ],
)
def test_gripper_geometry_clips_only_for_display(value: float, expected: float) -> None:
    assert normalized_gripper_to_half_opening(value) == pytest.approx(expected)


def test_finger_points_have_twice_the_half_opening_separation() -> None:
    pose = forward_arm([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], "left")
    separation = np.linalg.norm(pose.finger_points[0] - pose.finger_points[1])
    assert separation == pytest.approx(2.0 * pose.gripper_half_opening)


@pytest.mark.parametrize("shape", [(6,), (8,), (14,)])
def test_forward_arm_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Expected seven"):
        forward_arm(np.zeros(shape), "left")


def test_forward_bimanual_rejects_non_finite_state() -> None:
    state = np.zeros(14)
    state[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        forward_bimanual(state)
