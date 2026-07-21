# SPDX-License-Identifier: Apache-2.0
"""Nominal forward kinematics for the bimanual ViperX 300S ALOHA setup.

The local FK implementation is project-owned. Nominal geometry and gripper-calibration values
are adapted from Hugging Face gym-aloha (Apache-2.0), which retains the original ALOHA MIT
notice. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Side = Literal["left", "right"]
FloatArray = NDArray[np.float64]

ARM_JOINT_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
)
CANONICAL_JOINT_NAMES = tuple(
    f"{side}_{joint}" for side in ("left", "right") for joint in ARM_JOINT_NAMES
)

# Nominal geometry adapted from huggingface/gym-aloha's ViperX 300S MJCF model.
# Body translations precede the rotation of the joint owned by that body.
JOINT_OFFSETS: tuple[FloatArray, ...] = (
    np.array([0.0, 0.0, 0.079]),
    np.array([0.0, 0.0, 0.04805]),
    np.array([0.05955, 0.0, 0.3]),
    np.array([0.2, 0.0, 0.0]),
    np.array([0.1, 0.0, 0.0]),
    np.array([0.069744, 0.0, 0.0]),
)
JOINT_AXES = ("z", "y", "y", "x", "y", "x")
TOOL_OFFSET_M = 0.1072

LEFT_BASE_TRANSLATION = np.array([-0.469, 0.5, 0.0])
RIGHT_BASE_TRANSLATION = np.array([0.469, 0.5, 0.0])

# Real puppet calibration values adapted from gym_aloha.constants. The value is the positive
# displacement of one finger from the gripper center, not the full separation.
PUPPET_GRIPPER_HALF_OPENING_CLOSED_M = 0.01844
PUPPET_GRIPPER_HALF_OPENING_OPEN_M = 0.058


@dataclass(frozen=True)
class ArmPose:
    """Nominal geometry for one arm at one instant."""

    joint_points: FloatArray
    end_effector_translation: FloatArray
    end_effector_rotation: FloatArray
    finger_points: FloatArray
    gripper_half_opening: float


@dataclass(frozen=True)
class BimanualPose:
    """Nominal geometry for the left and right arms at one instant."""

    left: ArmPose
    right: ArmPose


def _translation(offset: ArrayLike) -> FloatArray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(offset, dtype=np.float64)
    return transform


def _rotation(axis: str, angle: float) -> FloatArray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    if axis == "x":
        rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
        )
    elif axis == "y":
        rotation = np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
        )
    elif axis == "z":
        rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
    else:
        raise ValueError(f"Unsupported rotation axis: {axis!r}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def base_transform(side: Side) -> FloatArray:
    """Return the nominal world-from-arm-base transform."""

    if side == "left":
        return _translation(LEFT_BASE_TRANSLATION)
    if side == "right":
        return _translation(RIGHT_BASE_TRANSLATION) @ _rotation("z", np.pi)
    raise ValueError(f"Unknown ALOHA arm side: {side!r}")


def normalized_gripper_to_half_opening(value: float) -> float:
    """Map normalized puppet gripper state to one finger's physical displacement."""

    if not np.isfinite(value):
        raise ValueError(f"Gripper value must be finite, got {value!r}")
    normalized = float(np.clip(value, 0.0, 1.0))
    closed = PUPPET_GRIPPER_HALF_OPENING_CLOSED_M
    opened = PUPPET_GRIPPER_HALF_OPENING_OPEN_M
    return closed + normalized * (opened - closed)


def forward_arm(joints: ArrayLike, side: Side) -> ArmPose:
    """Compute nominal FK for one seven-value ALOHA arm state.

    The first six values are revolute joint angles in radians. The final value
    is the normalized gripper state (0 closed, 1 open).
    """

    values = np.asarray(joints, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError(f"Expected seven {side} arm values, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{side.capitalize()} arm state contains non-finite values")

    transform = base_transform(side)
    points = [transform[:3, 3].copy()]
    for offset, axis, angle in zip(JOINT_OFFSETS, JOINT_AXES, values[:6], strict=True):
        transform = transform @ _translation(offset)
        points.append(transform[:3, 3].copy())
        transform = transform @ _rotation(axis, float(angle))

    end_effector = transform @ _translation([TOOL_OFFSET_M, 0.0, 0.0])
    points.append(end_effector[:3, 3].copy())

    half_opening = normalized_gripper_to_half_opening(float(values[6]))
    finger_offset = end_effector[:3, :3] @ np.array([0.0, half_opening, 0.0])
    finger_points = np.stack(
        [
            end_effector[:3, 3] + finger_offset,
            end_effector[:3, 3] - finger_offset,
        ]
    )

    return ArmPose(
        joint_points=np.stack(points),
        end_effector_translation=end_effector[:3, 3].copy(),
        end_effector_rotation=end_effector[:3, :3].copy(),
        finger_points=finger_points,
        gripper_half_opening=half_opening,
    )


def forward_bimanual(state: ArrayLike) -> BimanualPose:
    """Compute nominal FK for a canonical left-then-right 14-value state."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape != (14,):
        raise ValueError(f"Expected 14 ALOHA state values, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("ALOHA state contains non-finite values")
    return BimanualPose(
        left=forward_arm(values[:7], "left"),
        right=forward_arm(values[7:], "right"),
    )
