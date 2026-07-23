"""LIBERO Panda inverse kinematics and normalized display-frame utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from robosuite.utils import transform_utils
from robosuite.utils.control_utils import orientation_error

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from robot_practice.evaluation.libero_tasks import LiberoTaskIdentity

LOGGER = logging.getLogger(__name__)

IK_MAX_ITERATIONS = 80
IK_DAMPING = 1e-3
IK_GAIN = 0.8
IK_MAX_STEP_NORM_RAD = 0.15
IK_JOINT_LIMIT_MARGIN_RAD = 1e-5
IK_POSITION_TOLERANCE_M = 1e-3
IK_ROTATION_TOLERANCE_RAD = np.pi / 180.0
MUJOCO_WORLD_BODY_ID = 0
MUJOCO_FIXED_CAMERA_MODE = 0

PANDA_SKELETON_BODY_NAMES = (
    "robot0_base",
    "robot0_link0",
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
    "robot0_right_hand",
    "gripper0_eef",
)
PANDA_SKELETON_LABELS = tuple(
    name.removeprefix("robot0_").removeprefix("gripper0_")
    for name in PANDA_SKELETON_BODY_NAMES
)
PANDA_LEFT_FINGER_BODY_NAMES = (
    "robot0_right_hand",
    "gripper0_leftfinger",
    "gripper0_finger_joint1_tip",
)
PANDA_RIGHT_FINGER_BODY_NAMES = (
    "robot0_right_hand",
    "gripper0_rightfinger",
    "gripper0_finger_joint2_tip",
)


def _readonly(array: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DisplayFrame:
    """Rigid mapping from a robot-native frame into a normalized display frame."""

    origin_native: NDArray[np.float64]
    rotation_display_from_native: NDArray[np.float64]
    x_label: str = "X"
    y_label: str = "Y Forward"
    z_label: str = "Z Up"

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_native, dtype=np.float64)
        rotation = np.asarray(self.rotation_display_from_native, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError(f"Display-frame origin must be finite shape (3,), got {origin}")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError(
                "Display-frame rotation must be finite shape (3, 3), "
                f"got {rotation}"
            )
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9):
            raise ValueError("Display-frame rotation must be orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1e-9):
            raise ValueError(
                "Display-frame rotation must be right-handed with determinant +1, "
                f"got {determinant}"
            )
        if not all((self.x_label, self.y_label, self.z_label)):
            raise ValueError("Display-frame axis labels must be non-empty")
        object.__setattr__(self, "origin_native", _readonly(origin))
        object.__setattr__(self, "rotation_display_from_native", _readonly(rotation))

    @classmethod
    def libero_y_forward(cls, base_position_native: Sequence[float]) -> DisplayFrame:
        """Map LIBERO native +X forward to display +Y forward."""

        return cls(
            origin_native=np.asarray(base_position_native, dtype=np.float64),
            rotation_display_from_native=np.asarray(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )

    def transform_points(
        self,
        points_native: Sequence[float] | NDArray[Any],
    ) -> NDArray[np.float64]:
        points = np.asarray(points_native, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError(f"Points must end in three finite coordinates, got {points}")
        return (points - self.origin_native) @ self.rotation_display_from_native.T

    def transform_vectors(
        self,
        vectors_native: Sequence[float] | NDArray[Any],
    ) -> NDArray[np.float64]:
        vectors = np.asarray(vectors_native, dtype=np.float64)
        if vectors.shape[-1:] != (3,) or not np.isfinite(vectors).all():
            raise ValueError(f"Vectors must end in three finite coordinates, got {vectors}")
        return vectors @ self.rotation_display_from_native.T

    def transform_rotation(
        self,
        rotation_native: Sequence[Sequence[float]] | NDArray[Any],
    ) -> NDArray[np.float64]:
        rotation = np.asarray(rotation_native, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError(f"Rotation must be finite shape (3, 3), got {rotation}")
        return self.rotation_display_from_native @ rotation


@dataclass(frozen=True)
class FixedCameraCalibration:
    """One fixed MuJoCo camera expressed in the normalized display frame."""

    camera_name: str
    image_width: int
    image_height: int
    image_channels: int
    position_display: NDArray[np.float64]
    rotation_display_from_camera: NDArray[np.float64]
    fov_y_rad: float
    image_from_camera: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.camera_name:
            raise ValueError("camera_name must be non-empty")
        dimensions = (self.image_width, self.image_height, self.image_channels)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in dimensions
        ):
            raise ValueError(f"Camera image dimensions must be positive integers, got {dimensions}")
        if self.image_channels not in (1, 3, 4):
            raise ValueError(
                f"Camera image channels must be one of 1, 3, or 4, got {self.image_channels}"
            )
        position = np.asarray(self.position_display, dtype=np.float64)
        rotation = np.asarray(self.rotation_display_from_camera, dtype=np.float64)
        intrinsics = np.asarray(self.image_from_camera, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Camera position must be finite shape (3,), got {position}")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError(f"Camera rotation must be finite shape (3, 3), got {rotation}")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9):
            raise ValueError("Camera rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError("Camera rotation must be right-handed")
        if (
            not np.isfinite(self.fov_y_rad)
            or not 0.0 < self.fov_y_rad < np.pi
        ):
            raise ValueError(f"Camera vertical FOV must be within (0, pi), got {self.fov_y_rad}")
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(f"Camera intrinsics must be finite shape (3, 3), got {intrinsics}")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("Camera focal lengths must be positive")
        object.__setattr__(self, "image_width", int(self.image_width))
        object.__setattr__(self, "image_height", int(self.image_height))
        object.__setattr__(self, "image_channels", int(self.image_channels))
        object.__setattr__(self, "position_display", _readonly(position))
        object.__setattr__(self, "rotation_display_from_camera", _readonly(rotation))
        object.__setattr__(self, "fov_y_rad", float(self.fov_y_rad))
        object.__setattr__(self, "image_from_camera", _readonly(intrinsics))

    @property
    def image_shape_hwc(self) -> tuple[int, int, int]:
        return (self.image_height, self.image_width, self.image_channels)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)

    def validate_hwc_image(self, image: NDArray[Any]) -> None:
        shape = tuple(np.asarray(image).shape)
        if shape != self.image_shape_hwc:
            raise ValueError(
                f"Recorded {self.camera_name} image must have HWC shape "
                f"{self.image_shape_hwc}, got {shape}"
            )

    def project_display_points(
        self,
        points_display: Sequence[float] | NDArray[Any],
    ) -> NDArray[np.float64]:
        """Project display-frame points using MuJoCo's right/up/back camera axes."""

        points = np.asarray(points_display, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError(f"Projection points must end in three finite values, got {points}")
        flattened = points.reshape(-1, 3)
        points_camera = (
            flattened - self.position_display
        ) @ self.rotation_display_from_camera
        depth = -points_camera[:, 2]
        if np.any(depth <= 0.0):
            raise ValueError("Projection points must be in front of the camera")
        pixels = np.column_stack(
            (
                self.image_from_camera[0, 2]
                + self.image_from_camera[0, 0] * points_camera[:, 0] / depth,
                self.image_from_camera[1, 2]
                - self.image_from_camera[1, 1] * points_camera[:, 1] / depth,
            )
        )
        return pixels.reshape(points.shape[:-1] + (2,))


def extract_fixed_camera_calibration(
    sim: Any,
    display_frame: DisplayFrame,
    *,
    camera_name: str,
    image_shape_hwc: Sequence[int],
) -> FixedCameraCalibration:
    """Extract and validate one world-attached fixed MuJoCo camera."""

    shape = tuple(image_shape_hwc)
    if len(shape) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in shape
    ):
        raise ValueError(
            "Camera metadata shape must be HWC positive integers, "
            f"got {image_shape_hwc}"
        )
    image_height, image_width, image_channels = (int(value) for value in shape)
    if image_height <= 0 or image_width <= 0 or image_channels not in (1, 3, 4):
        raise ValueError(
            "Camera metadata shape must be HWC with channels in {1, 3, 4}, "
            f"got {shape}"
        )

    names = tuple(getattr(sim.model, "camera_names", ()))
    if names and names.count(camera_name) != 1:
        raise RuntimeError(
            f"MuJoCo model must contain exactly one {camera_name!r} camera, got {names}"
        )
    try:
        camera_id = int(sim.model.camera_name2id(camera_name))
    except Exception as exc:
        raise RuntimeError(f"MuJoCo model is missing camera {camera_name!r}") from exc
    if int(sim.model.cam_bodyid[camera_id]) != MUJOCO_WORLD_BODY_ID:
        raise RuntimeError(f"Camera {camera_name!r} must be attached to the world body")
    if int(sim.model.cam_mode[camera_id]) != MUJOCO_FIXED_CAMERA_MODE:
        raise RuntimeError(f"Camera {camera_name!r} must use fixed camera mode")

    position_native = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64).copy()
    rotation_native_from_camera = np.asarray(
        sim.data.cam_xmat[camera_id],
        dtype=np.float64,
    ).reshape(3, 3).copy()
    sim.forward()
    forwarded_position = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64)
    forwarded_rotation = np.asarray(
        sim.data.cam_xmat[camera_id],
        dtype=np.float64,
    ).reshape(3, 3)
    if not np.allclose(position_native, forwarded_position, atol=1e-9) or not np.allclose(
        rotation_native_from_camera,
        forwarded_rotation,
        atol=1e-9,
    ):
        raise RuntimeError(f"Camera {camera_name!r} world pose changed after sim.forward()")

    fov_y_degrees = float(sim.model.cam_fovy[camera_id])
    fov_y_rad = float(np.deg2rad(fov_y_degrees))
    focal_length = 0.5 * image_height / np.tan(0.5 * fov_y_rad)
    image_from_camera = np.asarray(
        [
            [focal_length, 0.0, 0.5 * image_width],
            [0.0, focal_length, 0.5 * image_height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return FixedCameraCalibration(
        camera_name=camera_name,
        image_width=image_width,
        image_height=image_height,
        image_channels=image_channels,
        position_display=display_frame.transform_points(position_native),
        rotation_display_from_camera=display_frame.transform_rotation(
            rotation_native_from_camera
        ),
        fov_y_rad=fov_y_rad,
        image_from_camera=image_from_camera,
    )


@dataclass(frozen=True)
class PandaSkeletonPose:
    """One converged Panda skeleton pose in the LIBERO native frame."""

    skeleton_points_native: NDArray[np.float64]
    left_finger_points_native: NDArray[np.float64]
    right_finger_points_native: NDArray[np.float64]
    arm_qpos: NDArray[np.float64]
    gripper_qpos: NDArray[np.float64]
    position_residual_m: float
    rotation_residual_rad: float
    iterations: int

    def __post_init__(self) -> None:
        expected = {
            "skeleton_points_native": (
                np.asarray(self.skeleton_points_native),
                (len(PANDA_SKELETON_BODY_NAMES), 3),
            ),
            "left_finger_points_native": (
                np.asarray(self.left_finger_points_native),
                (len(PANDA_LEFT_FINGER_BODY_NAMES), 3),
            ),
            "right_finger_points_native": (
                np.asarray(self.right_finger_points_native),
                (len(PANDA_RIGHT_FINGER_BODY_NAMES), 3),
            ),
            "arm_qpos": (np.asarray(self.arm_qpos), (7,)),
            "gripper_qpos": (np.asarray(self.gripper_qpos), (2,)),
        }
        for name, (array, shape) in expected.items():
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite shape {shape}, got {array}")
            object.__setattr__(self, name, _readonly(array))
        if not np.isfinite(self.position_residual_m) or self.position_residual_m < 0:
            raise ValueError("position_residual_m must be finite and non-negative")
        if not np.isfinite(self.rotation_residual_rad) or self.rotation_residual_rad < 0:
            raise ValueError("rotation_residual_rad must be finite and non-negative")
        if not 1 <= self.iterations <= IK_MAX_ITERATIONS:
            raise ValueError(f"iterations must be within [1, {IK_MAX_ITERATIONS}]")


@dataclass(frozen=True)
class PandaIKResult:
    """Converged pose or explicit non-convergence diagnostics."""

    pose: PandaSkeletonPose | None
    position_residual_m: float
    rotation_residual_rad: float
    iterations: int
    reason: str | None = None

    @property
    def converged(self) -> bool:
        return self.pose is not None


@dataclass(frozen=True)
class PandaMeasuredPose:
    """Panda geometry produced from measured simulator joint positions."""

    skeleton_points_native: NDArray[np.float64]
    left_finger_points_native: NDArray[np.float64]
    right_finger_points_native: NDArray[np.float64]
    arm_qpos: NDArray[np.float64]
    gripper_qpos: NDArray[np.float64]

    def __post_init__(self) -> None:
        expected = {
            "skeleton_points_native": (
                np.asarray(self.skeleton_points_native),
                (len(PANDA_SKELETON_BODY_NAMES), 3),
            ),
            "left_finger_points_native": (
                np.asarray(self.left_finger_points_native),
                (len(PANDA_LEFT_FINGER_BODY_NAMES), 3),
            ),
            "right_finger_points_native": (
                np.asarray(self.right_finger_points_native),
                (len(PANDA_RIGHT_FINGER_BODY_NAMES), 3),
            ),
            "arm_qpos": (np.asarray(self.arm_qpos), (7,)),
            "gripper_qpos": (np.asarray(self.gripper_qpos), (2,)),
        }
        for name, (array, shape) in expected.items():
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite shape {shape}, got {array}")
            object.__setattr__(self, name, _readonly(array))


EnvironmentFactory = Callable[[LiberoTaskIdentity], Any]


def _make_kinematics_environment(task: LiberoTaskIdentity) -> OffScreenRenderEnv:
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_path.is_file():
        raise FileNotFoundError(f"LIBERO BDDL file is missing: {bddl_path}")
    return OffScreenRenderEnv(bddl_file_name=str(bddl_path))


def rotation_geodesic_error(
    target: NDArray[np.float64],
    current: NDArray[np.float64],
) -> float:
    relative = target @ current.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


class PandaSkeletonReconstructor(AbstractContextManager["PandaSkeletonReconstructor"]):
    """Continuously reconstruct Panda link geometry from recorded EEF poses."""

    def __init__(
        self,
        task: LiberoTaskIdentity,
        *,
        environment_factory: EnvironmentFactory = _make_kinematics_environment,
    ) -> None:
        self._environment: Any | None = None
        self._closed = False
        try:
            self._environment = environment_factory(task)
            observation = self._environment.reset()
            self._initialize_model_contract(observation)
        except BaseException:
            self.close()
            raise

    def _initialize_model_contract(self, observation: dict[str, Any]) -> None:
        if self._environment is None:
            raise RuntimeError("Panda environment was not created")
        if len(self._environment.robots) != 1:
            raise RuntimeError("Expected exactly one LIBERO Panda robot")
        self._robot = self._environment.robots[0]
        self._sim = self._environment.sim
        self._arm_qpos_indices = np.asarray(
            self._robot._ref_joint_pos_indexes,
            dtype=np.int64,
        )
        self._arm_qvel_indices = np.asarray(
            self._robot._ref_joint_vel_indexes,
            dtype=np.int64,
        )
        self._arm_joint_ids = np.asarray(
            self._robot._ref_joint_indexes,
            dtype=np.int64,
        )
        if (
            self._arm_qpos_indices.shape != (7,)
            or self._arm_qvel_indices.shape != (7,)
            or self._arm_joint_ids.shape != (7,)
        ):
            raise RuntimeError("LIBERO Panda must expose seven arm joints")
        self._joint_limits = np.asarray(
            self._sim.model.jnt_range[self._arm_joint_ids],
            dtype=np.float64,
        )
        if self._joint_limits.shape != (7, 2):
            raise RuntimeError("LIBERO Panda joint limits must have shape (7, 2)")
        self._joint_low = self._joint_limits[:, 0] + IK_JOINT_LIMIT_MARGIN_RAD
        self._joint_high = self._joint_limits[:, 1] - IK_JOINT_LIMIT_MARGIN_RAD
        if np.any(self._joint_low >= self._joint_high):
            raise RuntimeError("LIBERO Panda contains invalid arm joint limits")

        self._neutral_qpos = np.asarray(
            self._sim.data.qpos[self._arm_qpos_indices],
            dtype=np.float64,
        ).copy()
        self._seed_qpos = self._neutral_qpos.copy()
        self._eef_site_id = int(self._robot.eef_site_id)
        self._eef_site_name = self._sim.model.site_id2name(self._eef_site_id)
        if not self._eef_site_name:
            raise RuntimeError("LIBERO Panda EEF site has no name")

        gripper_joint_names = tuple(self._robot.gripper.joints)
        if len(gripper_joint_names) != 2:
            raise RuntimeError("LIBERO Panda must expose two gripper joints")
        gripper_joint_ids = np.asarray(
            [self._sim.model.joint_name2id(name) for name in gripper_joint_names],
            dtype=np.int64,
        )
        qpos_addresses = [
            self._sim.model.get_joint_qpos_addr(name) for name in gripper_joint_names
        ]
        if any(not isinstance(address, (int, np.integer)) for address in qpos_addresses):
            raise RuntimeError("LIBERO Panda gripper joints must use scalar qpos addresses")
        self._gripper_qpos_indices = np.asarray(qpos_addresses, dtype=np.int64)
        self._gripper_limits = np.asarray(
            self._sim.model.jnt_range[gripper_joint_ids],
            dtype=np.float64,
        )
        if self._gripper_limits.shape != (2, 2):
            raise RuntimeError("LIBERO Panda gripper limits must have shape (2, 2)")

        required_bodies = set(PANDA_SKELETON_BODY_NAMES)
        required_bodies.update(PANDA_LEFT_FINGER_BODY_NAMES)
        required_bodies.update(PANDA_RIGHT_FINGER_BODY_NAMES)
        for name in sorted(required_bodies):
            try:
                self._sim.model.body_name2id(name)
            except Exception as exc:
                raise RuntimeError(f"LIBERO Panda model is missing body {name}") from exc

        base_native = np.asarray(
            self._sim.data.get_body_xpos("robot0_base"),
            dtype=np.float64,
        )
        self.display_frame = DisplayFrame.libero_y_forward(base_native)
        raw_joint_pos = observation.get("robot0_joint_pos")
        if raw_joint_pos is not None:
            observed = np.asarray(raw_joint_pos, dtype=np.float64)
            if observed.shape != (7,) or not np.isfinite(observed).all():
                raise RuntimeError("LIBERO Panda reset returned invalid robot0_joint_pos")

    @property
    def neutral_qpos(self) -> NDArray[np.float64]:
        return self._neutral_qpos.copy()

    @property
    def joint_limits(self) -> NDArray[np.float64]:
        return self._joint_limits.copy()

    def fixed_camera_calibration(
        self,
        camera_name: str,
        image_shape_hwc: Sequence[int],
    ) -> FixedCameraCalibration:
        self._ensure_open()
        return extract_fixed_camera_calibration(
            self._sim,
            self.display_frame,
            camera_name=camera_name,
            image_shape_hwc=image_shape_hwc,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def reset_episode(self) -> None:
        self._ensure_open()
        self._seed_qpos = self._neutral_qpos.copy()

    def reconstruct(self, observation_state: Sequence[float] | NDArray[Any]) -> PandaIKResult:
        self._ensure_open()
        state = np.asarray(observation_state, dtype=np.float64)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(
                "observation_state must contain eight finite values, "
                f"got {state}"
            )
        target_position = state[:3]
        target_rotation = transform_utils.quat2mat(
            transform_utils.axisangle2quat(state[3:6])
        )
        qpos = np.clip(self._seed_qpos.copy(), self._joint_low, self._joint_high)

        position_residual = float("inf")
        rotation_residual = float("inf")
        failure_reason: str | None = None
        iterations = IK_MAX_ITERATIONS
        converged = False
        for iteration in range(1, IK_MAX_ITERATIONS + 1):
            iterations = iteration
            self._set_arm_qpos(qpos)
            current_position, current_rotation = self._eef_pose()
            position_error = target_position - current_position
            rotation_error = orientation_error(target_rotation, current_rotation)
            position_residual = float(np.linalg.norm(position_error))
            rotation_residual = rotation_geodesic_error(
                target_rotation,
                current_rotation,
            )
            if (
                position_residual <= IK_POSITION_TOLERANCE_M
                and rotation_residual <= IK_ROTATION_TOLERANCE_RAD
            ):
                iterations = iteration
                converged = True
                break

            jacobian_position = self._sim.data.get_site_jacp(
                self._eef_site_name
            ).reshape(3, -1)[:, self._arm_qvel_indices]
            jacobian_rotation = self._sim.data.get_site_jacr(
                self._eef_site_name
            ).reshape(3, -1)[:, self._arm_qvel_indices]
            jacobian = np.vstack((jacobian_position, jacobian_rotation))
            error = np.concatenate((position_error, rotation_error))
            if not np.isfinite(jacobian).all() or not np.isfinite(error).all():
                failure_reason = "non_finite_jacobian_or_error"
                break
            regularized = (
                jacobian @ jacobian.T
                + IK_DAMPING**2 * np.eye(6, dtype=np.float64)
            )
            try:
                delta = jacobian.T @ np.linalg.solve(regularized, error)
            except np.linalg.LinAlgError:
                failure_reason = "linear_solve_failed"
                break
            delta *= IK_GAIN
            delta_norm = float(np.linalg.norm(delta))
            if not np.isfinite(delta_norm):
                failure_reason = "non_finite_joint_update"
                break
            if delta_norm > IK_MAX_STEP_NORM_RAD:
                delta *= IK_MAX_STEP_NORM_RAD / delta_norm
            qpos = np.clip(qpos + delta, self._joint_low, self._joint_high)

        if not converged:
            self._set_arm_qpos(self._seed_qpos)
            return PandaIKResult(
                pose=None,
                position_residual_m=position_residual,
                rotation_residual_rad=rotation_residual,
                iterations=iterations,
                reason=failure_reason or "maximum_iterations_reached",
            )

        gripper_qpos = np.clip(
            state[6:8],
            self._gripper_limits[:, 0],
            self._gripper_limits[:, 1],
        )
        self._sim.data.qpos[self._gripper_qpos_indices] = gripper_qpos
        self._sim.forward()
        pose = PandaSkeletonPose(
            skeleton_points_native=self._body_points(PANDA_SKELETON_BODY_NAMES),
            left_finger_points_native=self._body_points(
                PANDA_LEFT_FINGER_BODY_NAMES
            ),
            right_finger_points_native=self._body_points(
                PANDA_RIGHT_FINGER_BODY_NAMES
            ),
            arm_qpos=qpos,
            gripper_qpos=gripper_qpos,
            position_residual_m=position_residual,
            rotation_residual_rad=rotation_residual,
            iterations=iterations,
        )
        self._seed_qpos = qpos.copy()
        return PandaIKResult(
            pose=pose,
            position_residual_m=position_residual,
            rotation_residual_rad=rotation_residual,
            iterations=iterations,
        )

    def forward_measured(
        self,
        arm_qpos: Sequence[float] | NDArray[Any],
        gripper_qpos: Sequence[float] | NDArray[Any],
    ) -> PandaMeasuredPose:
        """Produce exact model geometry from measured LIBERO joint positions."""

        self._ensure_open()
        arm = np.asarray(arm_qpos, dtype=np.float64)
        gripper = np.asarray(gripper_qpos, dtype=np.float64)
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError(
                "measured arm_qpos must contain seven finite values, "
                f"got {arm}"
            )
        if gripper.shape != (2,) or not np.isfinite(gripper).all():
            raise ValueError(
                "measured gripper_qpos must contain two finite values, "
                f"got {gripper}"
            )
        self._set_arm_qpos(arm)
        self._sim.data.qpos[self._gripper_qpos_indices] = gripper
        self._sim.forward()
        return PandaMeasuredPose(
            skeleton_points_native=self._body_points(PANDA_SKELETON_BODY_NAMES),
            left_finger_points_native=self._body_points(PANDA_LEFT_FINGER_BODY_NAMES),
            right_finger_points_native=self._body_points(
                PANDA_RIGHT_FINGER_BODY_NAMES
            ),
            arm_qpos=arm,
            gripper_qpos=gripper,
        )

    def _set_arm_qpos(self, qpos: NDArray[np.float64]) -> None:
        self._sim.data.qpos[self._arm_qpos_indices] = qpos
        self._sim.forward()

    def _eef_pose(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        position = np.asarray(
            self._sim.data.site_xpos[self._eef_site_id],
            dtype=np.float64,
        ).copy()
        rotation = np.asarray(
            self._sim.data.site_xmat[self._eef_site_id],
            dtype=np.float64,
        ).reshape(3, 3).copy()
        return position, rotation

    def _body_points(self, names: Sequence[str]) -> NDArray[np.float64]:
        return np.stack(
            [
                np.asarray(self._sim.data.get_body_xpos(name), dtype=np.float64).copy()
                for name in names
            ]
        )

    def _ensure_open(self) -> None:
        if self._closed or self._environment is None:
            raise RuntimeError("PandaSkeletonReconstructor is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        environment = self._environment
        self._environment = None
        if environment is not None:
            environment.close()

    def __enter__(self) -> PandaSkeletonReconstructor:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
        return None
