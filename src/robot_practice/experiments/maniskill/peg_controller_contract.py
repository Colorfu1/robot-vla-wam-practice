"""Pure ManiSkill2 -> ManiSkill3 EEF action-contract helpers.

The two PegInsertionSide generations expose similarly named 7D actions but
their arm controllers use different physical conventions:

* ManiSkill2 uses a body/EEF-frame translation and right-multiplied rotation.
* ManiSkill3 uses a root-frame translation and left-multiplied rotation.  Its
  normalized rotation action is multiplied by a negative ``rot_lower``.

This module keeps the semantic bridge independent from either simulator.  It
is intentionally small enough to unit-test without constructing an environment.
Quaternions use the SAPIEN convention ``[w, x, y, z]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

MS2_POSITION_BOUND_M = 0.1
MS2_ROTATION_BOUND_RAD = 0.1
MS3_POSITION_BOUND_M = 0.1
MS3_ROTATION_LOWER_RAD = -0.1


def _array(value: Sequence[float], size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"Expected {size} values, got shape {result.shape}")
    return result


def _normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    value = _array(quaternion, 4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion must have a finite, non-zero norm")
    value = value / norm
    if value[0] < 0:
        value = -value
    return value


def _rotation_from_wxyz(quaternion: Sequence[float]) -> Rotation:
    value = _normalize_quaternion(quaternion)
    return Rotation.from_quat(value[[1, 2, 3, 0]])


def _wxyz_from_rotation(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return np.asarray(xyzw[[3, 0, 1, 2]], dtype=np.float64)


def _rotation_vector_from_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    return np.asarray(_rotation_from_wxyz(quaternion).as_rotvec(), dtype=np.float64)


def _pose_dict(position: Sequence[float], quaternion: Sequence[float]) -> dict[str, list[float]]:
    return {
        "position": _array(position, 3).tolist(),
        "quaternion_wxyz": _normalize_quaternion(quaternion).tolist(),
    }


def _clip_rotvec_action(action: np.ndarray) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(action))
    if norm > 1.0:
        return action / norm, True
    return action, False


def decode_ms2_ee_action(
    action: Sequence[float],
    current_position: Sequence[float],
    current_quaternion: Sequence[float],
    *,
    position_bound_m: float = MS2_POSITION_BOUND_M,
    rotation_bound_rad: float = MS2_ROTATION_BOUND_RAD,
) -> dict[str, Any]:
    """Decode a normalized ManiSkill2 native action into a physical transform.

    The returned translation/rotation are expressed in the current EEF frame.
    ``target_*`` is the corresponding target pose in the robot-base frame.
    """

    values = _array(action, 7)
    if position_bound_m <= 0 or rotation_bound_rad <= 0:
        raise ValueError("MS2 action bounds must be positive")
    normalized_position = np.clip(values[:3], -1.0, 1.0)
    normalized_rotation, rotation_clipped = _clip_rotvec_action(values[3:6])
    delta_position_ee = normalized_position * float(position_bound_m)
    delta_rotation_ee = normalized_rotation * float(rotation_bound_rad)

    current_rotation = _rotation_from_wxyz(current_quaternion)
    target_rotation = current_rotation * Rotation.from_rotvec(delta_rotation_ee)
    target_position = _array(current_position, 3) + current_rotation.apply(delta_position_ee)
    return {
        "input_action": values.tolist(),
        "normalized_position": normalized_position.tolist(),
        "normalized_rotation": normalized_rotation.tolist(),
        "delta_position_ee_m": delta_position_ee.tolist(),
        "delta_rotation_ee_rad": delta_rotation_ee.tolist(),
        "delta_position_root_m": current_rotation.apply(delta_position_ee).tolist(),
        "delta_rotation_root_rad": current_rotation.apply(delta_rotation_ee).tolist(),
        "rotation_clipped": rotation_clipped,
        "target_pose": _pose_dict(target_position, _wxyz_from_rotation(target_rotation)),
        "contract": "MS2 body translation + right-multiplied rotation",
    }


def encode_ms3_ee_action(
    delta_position_root: Sequence[float],
    delta_rotation_root: Sequence[float],
    gripper_action: float = 0.0,
    *,
    position_bound_m: float = MS3_POSITION_BOUND_M,
    rotation_lower_rad: float = MS3_ROTATION_LOWER_RAD,
) -> dict[str, Any]:
    """Encode a root-frame physical transform for MS3 ``pd_ee_delta_pose``."""

    if position_bound_m <= 0 or rotation_lower_rad >= 0:
        raise ValueError("MS3 bounds must be positive position and negative rotation")
    physical_position = _array(delta_position_root, 3)
    physical_rotation = _array(delta_rotation_root, 3)
    normalized_position = physical_position / float(position_bound_m)
    normalized_rotation = physical_rotation / float(rotation_lower_rad)
    action = np.concatenate(
        [normalized_position, normalized_rotation, np.asarray([gripper_action], dtype=np.float64)]
    )
    return {
        "action": action.astype(np.float32).tolist(),
        "physical_delta_position_root_m": physical_position.tolist(),
        "physical_delta_rotation_root_rad": physical_rotation.tolist(),
        "normalized_position": normalized_position.tolist(),
        "normalized_rotation": normalized_rotation.tolist(),
        "position_clipped": bool(np.any(np.abs(normalized_position) > 1.0)),
        "rotation_norm_before_ms3_clip": float(np.linalg.norm(normalized_rotation)),
        "contract": "MS3 root translation + left/root-aligned rotation; rot_lower=-0.1",
    }


def convert_ms2_action_to_ms3(
    action: Sequence[float],
    current_position: Sequence[float],
    current_quaternion: Sequence[float],
    *,
    position_bound_m: float = MS2_POSITION_BOUND_M,
    rotation_bound_rad: float = MS2_ROTATION_BOUND_RAD,
    ms3_position_bound_m: float = MS3_POSITION_BOUND_M,
    ms3_rotation_lower_rad: float = MS3_ROTATION_LOWER_RAD,
) -> dict[str, Any]:
    """Decode an MS2 action and re-encode the same physical target for MS3."""

    decoded = decode_ms2_ee_action(
        action,
        current_position,
        current_quaternion,
        position_bound_m=position_bound_m,
        rotation_bound_rad=rotation_bound_rad,
    )
    encoded = encode_ms3_ee_action(
        decoded["delta_position_root_m"],
        decoded["delta_rotation_root_rad"],
        float(_array(action, 7)[6]),
        position_bound_m=ms3_position_bound_m,
        rotation_lower_rad=ms3_rotation_lower_rad,
    )
    return {"decoded_ms2": decoded, "encoded_ms3": encoded}


def pose_delta(
    before_position: Sequence[float],
    before_quaternion: Sequence[float],
    after_position: Sequence[float],
    after_quaternion: Sequence[float],
) -> dict[str, Any]:
    """Compute realized root-frame translation and SO(3) rotation delta."""

    before_rotation = _rotation_from_wxyz(before_quaternion)
    after_rotation = _rotation_from_wxyz(after_quaternion)
    # ``after * inverse(before)`` is the root-frame (left) relative rotation.
    # This matches the MS3 controller's root-aligned rotation convention and
    # lets the axis-direction check use one frame on both backends.
    delta_rotation = after_rotation * before_rotation.inv()
    return {
        "position_root_m": (
            _array(after_position, 3) - _array(before_position, 3)
        ).tolist(),
        "rotation_root_rad": np.asarray(delta_rotation.as_rotvec(), dtype=np.float64).tolist(),
        "position_norm_m": float(
            np.linalg.norm(_array(after_position, 3) - _array(before_position, 3))
        ),
        "rotation_norm_rad": float(np.linalg.norm(delta_rotation.as_rotvec())),
    }


def rotation_geodesic(
    quaternion_left: Sequence[float], quaternion_right: Sequence[float]
) -> float:
    """Return the SO(3) geodesic distance between two wxyz quaternions."""

    left = _rotation_from_wxyz(quaternion_left)
    right = _rotation_from_wxyz(quaternion_right)
    return float(np.linalg.norm((left.inv() * right).as_rotvec()))
