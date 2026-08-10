"""Offline Panda kinematics used by ManiSkill visualization recordings.

This module deliberately contains no simulator or environment calls.  It uses
the Panda joint chain from ManiSkill's ``panda_v2.urdf`` and solves a small
numerical IK problem against the recorded ``extra.tcp_pose``.  The result is a
visualization skeleton, not a replacement for simulator state or a robot mesh.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PANDA_BASE_POSITION = np.asarray([-0.615, 0.0, 0.0], dtype=np.float64)
PANDA_DEFAULT_QPOS = np.asarray(
    [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4],
    dtype=np.float64,
)
PANDA_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)

PANDA_SKELETON_LABELS = (
    "base",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "hand",
    "tcp",
)

# These fixed joint origins are copied from ManiSkill's panda_v2.urdf.  The
# URDF uses a z-axis revolute joint after each fixed origin rotation.
_JOINT_ORIGINS = np.asarray(
    [
        [0.0, 0.0, 0.333],
        [0.0, 0.0, 0.0],
        [0.0, -0.316, 0.0],
        [0.0825, 0.0, 0.0],
        [-0.0825, 0.384, 0.0],
        [0.0, 0.0, 0.0],
        [0.088, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_JOINT_RPYS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [-np.pi / 2, 0.0, 0.0],
        [np.pi / 2, 0.0, 0.0],
        [np.pi / 2, 0.0, 0.0],
        [-np.pi / 2, 0.0, 0.0],
        [np.pi / 2, 0.0, 0.0],
        [np.pi / 2, 0.0, 0.0],
    ],
    dtype=np.float64,
)

_IK_MAX_ITERATIONS = 80
_IK_POSITION_TOLERANCE_M = 1e-4
_IK_ROTATION_TOLERANCE_RAD = 1e-3
_IK_DAMPING = 1e-3
_IK_GAIN = 0.8
_IK_MAX_STEP_NORM_RAD = 0.15
_JACOBIAN_EPSILON_RAD = 1e-6


@dataclass(frozen=True)
class PandaSkeletonPose:
    """One visualization pose reconstructed from a TCP target."""

    skeleton_points: np.ndarray
    left_finger_points: np.ndarray
    right_finger_points: np.ndarray
    arm_qpos: np.ndarray
    gripper_qpos: np.ndarray
    position_residual_m: float
    rotation_residual_rad: float
    iterations: int
    ik_converged: bool
    used_qpos_fallback: bool


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    # URDF fixed-axis RPY convention is equivalent to Rz(yaw) Ry(pitch) Rx(roll).
    return _rotation_z(float(yaw)) @ _rotation_y(float(pitch)) @ _rotation_x(float(roll))


def _axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= np.linalg.norm(unit_axis)
    x, y, z = unit_axis
    cosine, sine = np.cos(angle), np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=np.float64,
    )


def _homogeneous(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _translation(translation: Sequence[float]) -> np.ndarray:
    return _homogeneous(np.eye(3, dtype=np.float64), translation)


def _quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError(f"Panda TCP quaternion must be finite shape (4,), got {value}")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector for a proper rotation matrix."""

    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew
    sine = np.sin(angle)
    if abs(float(sine)) > 1e-6:
        return angle / (2.0 * sine) * skew

    # The exact-pi branch is only used for unusual synthetic targets.  Select
    # the largest diagonal to obtain a stable axis and preserve the angle.
    diagonal = np.maximum(np.diag(rotation) + 1.0, 0.0)
    axis_index = int(np.argmax(diagonal))
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = np.sqrt(diagonal[axis_index]) / np.sqrt(2.0)
    other = (axis_index + 1) % 3
    third = (axis_index + 2) % 3
    if axis[axis_index] > 1e-8:
        axis[other] = (rotation[other, axis_index] + rotation[axis_index, other]) / (
            4.0 * axis[axis_index]
        )
        axis[third] = (rotation[third, axis_index] + rotation[axis_index, third]) / (
            4.0 * axis[axis_index]
        )
    return angle * axis / max(float(np.linalg.norm(axis)), 1e-12)


class PandaKinematics:
    """Pure-NumPy Panda FK/IK for stored ManiSkill PickCube frames."""

    def __init__(self, *, base_position: Sequence[float] = PANDA_BASE_POSITION) -> None:
        base = np.asarray(base_position, dtype=np.float64)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError(f"Panda base position must be finite shape (3,), got {base}")
        self.base_position = base.copy()

    @staticmethod
    def _validate_arm_qpos(qpos: Sequence[float]) -> np.ndarray:
        value = np.asarray(qpos, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"Panda arm qpos must be finite shape (7,), got {value}")
        return np.clip(value, PANDA_JOINT_LIMITS[:, 0], PANDA_JOINT_LIMITS[:, 1])

    @staticmethod
    def _validate_gripper_qpos(qpos: Sequence[float] | None) -> np.ndarray:
        if qpos is None:
            return np.asarray([0.04, 0.04], dtype=np.float64)
        value = np.asarray(qpos, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError(f"Panda gripper qpos must be finite shape (2,), got {value}")
        return np.clip(value, 0.0, 0.04)

    def _forward_transforms(
        self,
        arm_qpos: np.ndarray,
        gripper_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transform = _translation(self.base_position)
        skeleton_points = [transform[:3, 3].copy()]
        for origin, rpy, angle in zip(_JOINT_ORIGINS, _JOINT_RPYS, arm_qpos, strict=True):
            transform = transform @ _homogeneous(_rpy_matrix(rpy), origin)
            transform = transform @ _homogeneous(
                _axis_angle_matrix((0.0, 0.0, 1.0), angle), (0, 0, 0)
            )
            skeleton_points.append(transform[:3, 3].copy())

        link8 = transform @ _translation((0.0, 0.0, 0.107))
        hand = link8 @ _homogeneous(_rotation_z(-np.pi / 4), (0.0, 0.0, 0.0))
        tcp = hand @ _translation((0.0, 0.0, 0.1034))
        skeleton_points.extend((hand[:3, 3].copy(), tcp[:3, 3].copy()))

        left_base = hand @ _translation((0.0, 0.0, 0.0584))
        left_base = left_base @ _translation((0.0, float(gripper_qpos[0]), 0.0))
        left_pad = left_base @ _translation((0.0, 0.0, 0.0455))
        right_base = hand @ _translation((0.0, 0.0, 0.0584))
        right_base = right_base @ _translation((0.0, -float(gripper_qpos[1]), 0.0))
        right_pad = right_base @ _translation((0.0, 0.0, 0.0455))
        return (
            np.asarray(skeleton_points, dtype=np.float64),
            np.asarray([hand[:3, 3], left_base[:3, 3], left_pad[:3, 3]], dtype=np.float64),
            np.asarray([hand[:3, 3], right_base[:3, 3], right_pad[:3, 3]], dtype=np.float64),
            tcp,
        )

    def _forward_pose(
        self,
        arm_qpos: np.ndarray,
        gripper_qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._forward_transforms(arm_qpos, gripper_qpos)

    def from_qpos(
        self,
        arm_qpos: Sequence[float],
        gripper_qpos: Sequence[float] | None = None,
    ) -> PandaSkeletonPose:
        """Build a skeleton directly from qpos when no TCP target is available."""

        arm = self._validate_arm_qpos(arm_qpos)
        gripper = self._validate_gripper_qpos(gripper_qpos)
        skeleton, left, right, _ = self._forward_pose(arm, gripper)
        return PandaSkeletonPose(
            skeleton_points=skeleton,
            left_finger_points=left,
            right_finger_points=right,
            arm_qpos=arm,
            gripper_qpos=gripper,
            position_residual_m=0.0,
            rotation_residual_rad=0.0,
            iterations=0,
            ik_converged=False,
            used_qpos_fallback=True,
        )

    def solve(
        self,
        tcp_pose: Sequence[float],
        *,
        seed_qpos: Sequence[float] | None = None,
        gripper_qpos: Sequence[float] | None = None,
    ) -> PandaSkeletonPose:
        """Solve Panda arm IK for a recorded world-frame TCP pose."""

        target = np.asarray(tcp_pose, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError(f"Panda TCP pose must be finite shape (7,), got {target}")
        target_position = target[:3]
        target_rotation = _quaternion_wxyz_to_matrix(target[3:])
        arm = PANDA_DEFAULT_QPOS.copy() if seed_qpos is None else self._validate_arm_qpos(seed_qpos)
        gripper = self._validate_gripper_qpos(gripper_qpos)
        iterations = 0
        converged = False

        for iteration in range(_IK_MAX_ITERATIONS):
            skeleton, _, _, tcp = self._forward_pose(arm, gripper)
            current_position = tcp[:3, 3]
            current_rotation = tcp[:3, :3]
            error = np.concatenate(
                (
                    target_position - current_position,
                    _rotation_log(target_rotation @ current_rotation.T),
                )
            )
            if (
                np.linalg.norm(error[:3]) <= _IK_POSITION_TOLERANCE_M
                and np.linalg.norm(error[3:]) <= _IK_ROTATION_TOLERANCE_RAD
            ):
                converged = True
                iterations = iteration
                break

            jacobian = np.empty((6, 7), dtype=np.float64)
            for joint_index in range(7):
                perturbed = arm.copy()
                perturbed[joint_index] += _JACOBIAN_EPSILON_RAD
                _, _, _, perturbed_tcp = self._forward_pose(perturbed, gripper)
                jacobian[:3, joint_index] = (
                    perturbed_tcp[:3, 3] - current_position
                ) / _JACOBIAN_EPSILON_RAD
                jacobian[3:, joint_index] = (
                    _rotation_log(perturbed_tcp[:3, :3] @ current_rotation.T)
                    / _JACOBIAN_EPSILON_RAD
                )

            damped = jacobian @ jacobian.T + (_IK_DAMPING**2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(damped, error)
            delta *= _IK_GAIN
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > _IK_MAX_STEP_NORM_RAD:
                delta *= _IK_MAX_STEP_NORM_RAD / delta_norm
            arm = np.clip(
                arm + delta,
                PANDA_JOINT_LIMITS[:, 0],
                PANDA_JOINT_LIMITS[:, 1],
            )
            iterations = iteration + 1

        skeleton, left, right, tcp = self._forward_pose(arm, gripper)
        position_residual = float(np.linalg.norm(target_position - tcp[:3, 3]))
        rotation_residual = float(np.linalg.norm(_rotation_log(target_rotation @ tcp[:3, :3].T)))
        converged = converged or (
            position_residual <= _IK_POSITION_TOLERANCE_M
            and rotation_residual <= _IK_ROTATION_TOLERANCE_RAD
        )
        return PandaSkeletonPose(
            skeleton_points=skeleton,
            left_finger_points=left,
            right_finger_points=right,
            arm_qpos=arm,
            gripper_qpos=gripper,
            position_residual_m=position_residual,
            rotation_residual_rad=rotation_residual,
            iterations=iterations,
            ik_converged=converged,
            used_qpos_fallback=False,
        )
