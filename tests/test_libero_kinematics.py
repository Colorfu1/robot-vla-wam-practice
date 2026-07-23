from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from robosuite.utils import transform_utils

from robot_practice.evaluation.libero_kinematics import (
    IK_POSITION_TOLERANCE_M,
    IK_ROTATION_TOLERANCE_RAD,
    DisplayFrame,
    FixedCameraCalibration,
    PandaMeasuredPose,
    PandaSkeletonReconstructor,
    extract_fixed_camera_calibration,
    rotation_geodesic_error,
)
from robot_practice.evaluation.libero_tasks import resolve_libero_task


def test_libero_display_frame_maps_native_forward_to_display_y() -> None:
    frame = DisplayFrame.libero_y_forward([-0.6, 0.2, 0.1])
    np.testing.assert_allclose(frame.transform_points([-0.6, 0.2, 0.1]), [0, 0, 0])
    np.testing.assert_allclose(frame.transform_vectors([1, 0, 0]), [0, 1, 0])
    np.testing.assert_allclose(frame.transform_vectors([0, -1, 0]), [1, 0, 0])
    np.testing.assert_allclose(frame.transform_vectors([0, 0, 1]), [0, 0, 1])
    assert (frame.x_label, frame.y_label, frame.z_label) == (
        "X",
        "Y Forward",
        "Z Up",
    )


@pytest.mark.parametrize(
    ("rotation", "message"),
    [
        (np.ones((2, 2)), "shape"),
        (np.diag([1.0, 1.0, 2.0]), "orthonormal"),
        (np.diag([1.0, 1.0, -1.0]), "right-handed"),
        (np.full((3, 3), np.nan), "finite"),
    ],
)
def test_display_frame_rejects_invalid_rotation(
    rotation: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DisplayFrame(np.zeros(3), rotation)


def test_rotation_geodesic_error_uses_so3_angle() -> None:
    identity = np.eye(3)
    quarter_turn = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert rotation_geodesic_error(identity, identity) == pytest.approx(0.0)
    assert rotation_geodesic_error(quarter_turn, identity) == pytest.approx(np.pi / 2)


class FakeCameraModel:
    def __init__(self) -> None:
        self.camera_names = ("agentview",)
        self.cam_bodyid = np.asarray([0])
        self.cam_mode = np.asarray([0])
        self.cam_fovy = np.asarray([45.0])

    def camera_name2id(self, name: str) -> int:
        if name != "agentview":
            raise ValueError(name)
        return 0


class FakeCameraData:
    def __init__(self) -> None:
        self.cam_xpos = np.asarray([[0.896577372, 0.000000522, 0.65]])
        self.cam_xmat = np.asarray(
            [
                [
                    -0.000001723,
                    -0.528769744,
                    0.848765314,
                    1.0,
                    -0.000000782,
                    0.000001543,
                    -0.000000152,
                    0.848765314,
                    0.528769744,
                ]
            ]
        )


class FakeCameraSim:
    def __init__(self) -> None:
        self.model = FakeCameraModel()
        self.data = FakeCameraData()
        self.move_after_forward = False
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1
        if self.move_after_forward:
            self.data.cam_xpos[0, 0] += 0.01


def _fixed_camera_calibration(sim: FakeCameraSim | None = None) -> FixedCameraCalibration:
    return extract_fixed_camera_calibration(
        sim or FakeCameraSim(),
        DisplayFrame.libero_y_forward([-0.6, 0.0, 0.0]),
        camera_name="agentview",
        image_shape_hwc=(256, 256, 3),
    )


def test_fixed_camera_calibration_converts_degrees_and_projects_asymmetric_points() -> None:
    calibration = _fixed_camera_calibration()
    assert calibration.fov_y_rad == pytest.approx(np.pi / 4)
    assert calibration.resolution_wh == (256, 256)
    assert calibration.image_shape_hwc == (256, 256, 3)
    expected_focal = 128.0 / np.tan(np.pi / 8)
    assert calibration.image_from_camera[0, 0] == pytest.approx(expected_focal)
    assert calibration.image_from_camera[1, 1] == pytest.approx(expected_focal)

    native_points = np.asarray(
        [
            [-0.6, 0.2, 0.1],
            [-0.1, -0.25, 0.3],
            [-0.1, -0.25, 0.0],
            [-0.1, 0.25, 0.0],
        ]
    )
    display = DisplayFrame.libero_y_forward([-0.6, 0.0, 0.0])
    pixels = calibration.project_display_points(display.transform_points(native_points))
    np.testing.assert_allclose(
        pixels,
        [
            [167.591233, 63.759209],
            [53.063350, 59.090222],
            [63.056351, 134.426155],
            [192.944364, 134.426260],
        ],
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("image_shape", "message"),
    [
        ((3, 256), "HWC"),
        ((256, 256, 2), "channels"),
        ((256, True, 3), "positive integers"),
    ],
)
def test_fixed_camera_calibration_rejects_invalid_hwc_metadata(
    image_shape: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_fixed_camera_calibration(
            FakeCameraSim(),
            DisplayFrame.libero_y_forward([-0.6, 0.0, 0.0]),
            camera_name="agentview",
            image_shape_hwc=image_shape,
        )


def test_fixed_camera_calibration_rejects_non_world_non_fixed_and_moving_camera() -> None:
    body_attached = FakeCameraSim()
    body_attached.model.cam_bodyid[0] = 1
    with pytest.raises(RuntimeError, match="world body"):
        _fixed_camera_calibration(body_attached)

    tracking = FakeCameraSim()
    tracking.model.cam_mode[0] = 1
    with pytest.raises(RuntimeError, match="fixed camera mode"):
        _fixed_camera_calibration(tracking)

    moving = FakeCameraSim()
    moving.move_after_forward = True
    with pytest.raises(RuntimeError, match="world pose changed"):
        _fixed_camera_calibration(moving)


def test_fixed_camera_calibration_validates_logged_hwc_frame_shape() -> None:
    calibration = _fixed_camera_calibration()
    calibration.validate_hwc_image(np.zeros((256, 256, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="HWC shape"):
        calibration.validate_hwc_image(np.zeros((3, 256, 256), dtype=np.uint8))


class ClosingEnvironment:
    def __init__(self) -> None:
        self.robots: list[Any] = []
        self.close_calls = 0

    def reset(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        self.close_calls += 1


def test_reconstructor_closes_environment_after_partial_initialization() -> None:
    environment = ClosingEnvironment()
    task = resolve_libero_task("libero_object", 8)
    with pytest.raises(RuntimeError, match="exactly one"):
        PandaSkeletonReconstructor(task, environment_factory=lambda _: environment)
    assert environment.close_calls == 1


def test_reconstructor_close_is_idempotent_and_context_safe() -> None:
    environment = ClosingEnvironment()
    reconstructor = object.__new__(PandaSkeletonReconstructor)
    reconstructor._environment = environment
    reconstructor._closed = False
    with pytest.raises(RuntimeError, match="test failure"):
        with reconstructor:
            raise RuntimeError("test failure")
    reconstructor.close()
    assert environment.close_calls == 1


def test_forward_measured_uses_exact_joint_positions() -> None:
    class FakeData:
        def __init__(self) -> None:
            self.qpos = np.zeros(9, dtype=np.float64)

    class FakeSim:
        def __init__(self) -> None:
            self.data = FakeData()
            self.forward_calls = 0

        def forward(self) -> None:
            self.forward_calls += 1

    reconstructor = object.__new__(PandaSkeletonReconstructor)
    reconstructor._closed = False
    reconstructor._environment = object()
    reconstructor._sim = FakeSim()
    reconstructor._gripper_qpos_indices = np.asarray([7, 8])
    captured_arm: list[np.ndarray] = []
    reconstructor._set_arm_qpos = lambda value: captured_arm.append(value.copy())
    reconstructor._body_points = lambda names: np.zeros((len(names), 3))
    arm = np.linspace(-0.3, 0.3, 7)
    gripper = np.asarray([0.02, -0.02])

    pose = reconstructor.forward_measured(arm, gripper)

    assert isinstance(pose, PandaMeasuredPose)
    np.testing.assert_array_equal(captured_arm, [arm])
    np.testing.assert_array_equal(reconstructor._sim.data.qpos[7:9], gripper)
    np.testing.assert_array_equal(pose.arm_qpos, arm)
    np.testing.assert_array_equal(pose.gripper_qpos, gripper)
    assert reconstructor._sim.forward_calls == 1


@pytest.mark.parametrize(
    ("arm", "gripper", "message"),
    [
        (np.zeros(6), np.zeros(2), "seven finite"),
        (np.full(7, np.nan), np.zeros(2), "seven finite"),
        (np.zeros(7), np.zeros(1), "two finite"),
        (np.zeros(7), np.full(2, np.nan), "two finite"),
    ],
)
def test_forward_measured_rejects_invalid_joint_positions(
    arm: np.ndarray,
    gripper: np.ndarray,
    message: str,
) -> None:
    reconstructor = object.__new__(PandaSkeletonReconstructor)
    reconstructor._closed = False
    reconstructor._environment = object()

    with pytest.raises(ValueError, match=message):
        reconstructor.forward_measured(arm, gripper)


RUN_REAL_KINEMATICS = os.environ.get("RUN_LIBERO_KINEMATICS_INTEGRATION") == "1"


@pytest.mark.skipif(
    not RUN_REAL_KINEMATICS,
    reason="set RUN_LIBERO_KINEMATICS_INTEGRATION=1 for real Panda IK tests",
)
def test_real_panda_fk_ik_sequence_and_gripper_geometry() -> None:
    task = resolve_libero_task("libero_object", 8)
    with PandaSkeletonReconstructor(task) as reconstructor:
        base_qpos = reconstructor.neutral_qpos
        states: list[np.ndarray] = []
        for step in range(1, 6):
            delta = np.asarray([0.01, -0.01, 0.005, 0.0, 0.01, -0.005, 0.01])
            source_qpos = np.clip(
                base_qpos + step * delta,
                reconstructor.joint_limits[:, 0] + 1e-5,
                reconstructor.joint_limits[:, 1] - 1e-5,
            )
            reconstructor._set_arm_qpos(source_qpos)
            position, rotation = reconstructor._eef_pose()
            axis_angle = transform_utils.quat2axisangle(
                transform_utils.mat2quat(rotation)
            )
            states.append(
                np.concatenate((position, axis_angle, np.asarray([0.02, -0.02])))
            )

        reconstructor.reset_episode()
        solutions: list[np.ndarray] = []
        for state in states:
            result = reconstructor.reconstruct(state)
            assert result.converged
            assert result.pose is not None
            assert result.position_residual_m <= IK_POSITION_TOLERANCE_M
            assert result.rotation_residual_rad <= IK_ROTATION_TOLERANCE_RAD
            assert result.pose.skeleton_points_native.shape == (11, 3)
            solutions.append(result.pose.arm_qpos)
        joint_steps = np.abs(np.diff(np.stack(solutions), axis=0))
        assert float(joint_steps.max()) <= 0.10

        closed_state = states[-1].copy()
        closed_state[6:8] = [0.0, 0.0]
        open_state = states[-1].copy()
        open_state[6:8] = [0.04, -0.04]
        closed = reconstructor.reconstruct(closed_state)
        opened = reconstructor.reconstruct(open_state)
        assert closed.pose is not None and opened.pose is not None
        closed_width = np.linalg.norm(
            closed.pose.left_finger_points_native[-1]
            - closed.pose.right_finger_points_native[-1]
        )
        open_width = np.linalg.norm(
            opened.pose.left_finger_points_native[-1]
            - opened.pose.right_finger_points_native[-1]
        )
        assert open_width > closed_width
