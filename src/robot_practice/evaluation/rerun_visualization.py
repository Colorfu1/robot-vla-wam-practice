"""Rerun recording helpers with one global sequence timeline per evaluation mode."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from numpy.typing import NDArray

from robot_practice.evaluation.libero_kinematics import (
    PANDA_SKELETON_LABELS,
    DisplayFrame,
    FixedCameraCalibration,
    PandaSkeletonPose,
)

ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
OFFLINE_METRIC_ROWS = (
    ("Episodes", "num_episodes", "integer"),
    ("Frames", "num_frames", "integer"),
    ("Model queries", "num_model_queries", "integer"),
    ("Batch size", "batch_size", "integer"),
    ("First-action MAE", "first_action_mae", "float4"),
    ("Delta-action MAE", "delta_action_mae", "float5"),
    ("Action TV", "teacher_forced_action_tv", "float5"),
    ("Action Jerk", "teacher_forced_action_jerk", "float5"),
    ("Inference latency p50", "inference_latency_p50_ms", "latency"),
    ("Inference latency p95", "inference_latency_p95_ms", "latency"),
)
LIBERO_OSC_TRANSLATION_SCALE_M = 0.05
DEMONSTRATION_COLOR = (0, 114, 178)
PREDICTION_COLOR = (230, 159, 0)
GT_COMMAND_COLOR = (86, 180, 233, 190)
PREDICTION_PREVIEW_COLOR = (255, 190, 80, 96)
PREDICTION_PREVIEW_ENDPOINT_COLOR = (255, 190, 80, 128)
AXIS_COLORS = ((230, 50, 50), (50, 190, 80), (60, 110, 230))
ACTION_COLOR_HEX = {
    "x": ("#4E79A7", "#A0CBE8"),
    "y": ("#F28E2B", "#FFBE7D"),
    "z": ("#59A14F", "#8CD17D"),
    "roll": ("#E15759", "#FF9D9A"),
    "pitch": ("#B07AA1", "#D4A6C8"),
    "yaw": ("#9C755F", "#D7B5A6"),
    "gripper": ("#2F9EAA", "#76C7C9"),
}
OFFLINE_SEMANTICS_MARKDOWN = """\
# Offline visualization semantics

- **Deep blue:** recorded end-effector position history.
- **Light-blue segments:** expert first-action OSC translation commands, each anchored at the
  recorded end-effector that produced the action.
- **Deep-orange segments:** ACT first-action OSC translation commands, teacher-forced and not
  executed, each anchored at the same recorded end-effector as its expert command.
- **Translucent orange:** the current frame's complete future command preview, not executed.

Command-segment endpoints are controller position goals, not measured next-frame positions.
They do not simulate contacts, dynamics, controller convergence, workspace limits, or future
observations. Segments from different frames are deliberately not connected into a Pred
trajectory.

The blue Panda arm is reconstructed from the recorded end-effector pose with inverse
kinematics. Its joint configuration is not a recorded joint measurement.

The **Physically calibrated agentview** tab uses the selected task's fixed MuJoCo agentview
pose and intrinsics to project this geometry onto a visualization-only physical-camera
image. The original recorded image and the policy input are not changed.
The **Free 3D** tab is still an independent orbit view; its cyan camera frustum marks the
actual agentview camera. The eye-in-hand image remains image-only because the dataset does
not contain the robot joint state needed to recover that moving camera exactly.

To keep the 3D viewport clear, only the X/Y/Z axis labels are always visible. Select a
joint point, end-effector point, or trajectory in Rerun to inspect its retained label in
the Selection panel.
"""


class Recorder(Protocol):
    def set_time(self, timeline: str, *, sequence: int) -> None: ...

    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None: ...


@dataclass(frozen=True)
class ActionSeriesSpec:
    path: str
    name: str
    color: tuple[int, int, int]
    action_index: int
    source: str


def _hex_rgb(value: str) -> tuple[int, int, int]:
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError(f"Expected #RRGGBB color, got {value!r}")
    return tuple(int(value[offset : offset + 2], 16) for offset in (1, 3, 5))


def _action_series_specs() -> tuple[ActionSeriesSpec, ...]:
    specs: list[ActionSeriesSpec] = []
    for action_index, action_name in enumerate(ACTION_NAMES):
        gt_hex, pred_hex = ACTION_COLOR_HEX[action_name]
        dimension_root = f"comparison/actions/{action_index:02d}_{action_name}"
        specs.extend(
            (
                ActionSeriesSpec(
                    path=f"{dimension_root}/00_gt",
                    name=f"{action_name} · GT",
                    color=_hex_rgb(gt_hex),
                    action_index=action_index,
                    source="gt",
                ),
                ActionSeriesSpec(
                    path=f"{dimension_root}/01_pred",
                    name=f"{action_name} · Pred",
                    color=_hex_rgb(pred_hex),
                    action_index=action_index,
                    source="pred",
                ),
            )
        )
    return tuple(specs)


ACTION_SERIES_SPECS = _action_series_specs()
ACTION_SERIES_PATHS = tuple(spec.path for spec in ACTION_SERIES_SPECS)
AGENTVIEW_CAMERA_KEY = "observation.images.image"
RECORDED_AGENTVIEW_ENTITY_PATH = "demonstration/cameras/image"
CALIBRATED_AGENTVIEW_ENTITY_PATH = "calibration/agentview_corrected"
# Compatibility alias for callers that mean the original recorded camera image.
AGENTVIEW_ENTITY_PATH = RECORDED_AGENTVIEW_ENTITY_PATH
AGENTVIEW_FRUSTUM_COLOR = (0, 220, 255, 210)
AGENTVIEW_IMAGE_PLANE_DISTANCE_M = 0.25
AGENTVIEW_FRUSTUM_LINE_WIDTH = 0.003
OFFLINE_3D_CONTENTS = (
    "display_frame/**",
    "demonstration/robot/**",
    "demonstration/trajectory/**",
    "comparison/commands/**",
    "prediction/not_executed/chunk_preview",
    "prediction/not_executed/chunk_endpoint",
)
OFFLINE_CALIBRATED_3D_CONTENTS = OFFLINE_3D_CONTENTS + (
    CALIBRATED_AGENTVIEW_ENTITY_PATH,
)
OFFLINE_AGENTVIEW_OVERLAY_CONTENTS = (
    CALIBRATED_AGENTVIEW_ENTITY_PATH,
    "demonstration/robot/**",
    "demonstration/trajectory/**",
    "comparison/commands/**",
    "prediction/not_executed/chunk_preview",
    "prediction/not_executed/chunk_endpoint",
)


@dataclass
class OfflineTrajectoryHistory:
    """Accumulate the recorded EEF path and reset it at episode boundaries."""

    episode_index: int | None = None
    last_frame_index: int | None = None
    _positions: list[NDArray[np.float64]] = field(default_factory=list)

    def append(
        self,
        *,
        episode_index: int,
        frame_index: int,
        observation_state: Sequence[float] | torch.Tensor | NDArray[Any],
    ) -> NDArray[np.float64]:
        state = _validate_observation_state(observation_state)
        if self.episode_index != episode_index:
            self.episode_index = episode_index
            self.last_frame_index = None
            self._positions.clear()
        if self.last_frame_index is not None and frame_index <= self.last_frame_index:
            raise ValueError(
                "Offline visualization frame_index must increase within an episode; "
                f"got {frame_index} after {self.last_frame_index}"
            )
        self.last_frame_index = frame_index
        self._positions.append(state[:3].copy())
        return np.stack(self._positions)


@dataclass(frozen=True)
class OfflineCommandSegments:
    """Retained one-step GT and Pred command strips for the current episode."""

    target: NDArray[np.float64]
    predicted: NDArray[np.float64]


@dataclass
class OfflineCommandHistory:
    """Accumulate independent teacher-forced command segments per episode."""

    episode_index: int | None = None
    last_frame_index: int | None = None
    _target_segments: list[NDArray[np.float64]] = field(default_factory=list)
    _predicted_segments: list[NDArray[np.float64]] = field(default_factory=list)

    def append(
        self,
        *,
        episode_index: int,
        frame_index: int,
        observation_state: Sequence[float] | torch.Tensor | NDArray[Any],
        target_action: Sequence[float],
        predicted_action: Sequence[float],
    ) -> OfflineCommandSegments:
        state = _validate_observation_state(observation_state)
        target = _validate_action(target_action, "target_action")
        predicted = _validate_action(predicted_action, "predicted_action")
        if self.episode_index != episode_index:
            self.episode_index = episode_index
            self.last_frame_index = None
            self._target_segments.clear()
            self._predicted_segments.clear()
        if self.last_frame_index is not None and frame_index <= self.last_frame_index:
            raise ValueError(
                "Offline command frame_index must increase within an episode; "
                f"got {frame_index} after {self.last_frame_index}"
            )
        self.last_frame_index = frame_index
        origin = state[:3].copy()
        self._target_segments.append(
            np.stack((origin, origin + _libero_action_translation(target)))
        )
        self._predicted_segments.append(
            np.stack((origin, origin + _libero_action_translation(predicted)))
        )
        return OfflineCommandSegments(
            target=np.stack(self._target_segments),
            predicted=np.stack(self._predicted_segments),
        )


def _camera_path(camera_key: str, *, root: str = "cameras") -> str:
    return f"{root}/{camera_key.rsplit('.', maxsplit=1)[-1]}"


def image_to_hwc_uint8(image: torch.Tensor | NDArray[Any]) -> NDArray[np.uint8]:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {array.shape}")
    if array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-first or channel-last image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Image contains NaN or Infinity")
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.asarray(array, dtype=np.uint8)


def physical_agentview_image(
    image_hwc: NDArray[np.uint8],
    convention: str,
) -> NDArray[np.uint8]:
    """Return an independent contiguous display copy matching physical projection."""

    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Agentview image must be HWC, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Agentview image must be uint8, got {image.dtype}")
    if convention == "lerobot_180":
        corrected = image[:, ::-1, :]
    elif convention == "physical_opencv":
        corrected = image
    else:
        raise ValueError(f"Unsupported resolved agentview image convention: {convention!r}")
    return np.array(corrected, dtype=np.uint8, order="C", copy=True)


def offline_semantics_markdown(agentview_image_convention: str | None) -> str:
    if agentview_image_convention == "lerobot_180":
        detail = (
            "The calibrated image is a horizontal reflection of the recorded display copy "
            "because the resolved convention is `lerobot_180`."
        )
    elif agentview_image_convention == "physical_opencv":
        detail = (
            "The calibrated image preserves the recorded orientation because the resolved "
            "convention is `physical_opencv`."
        )
    elif agentview_image_convention is None:
        detail = "No calibrated agentview image is present in this recording."
    else:
        raise ValueError(
            "Unsupported resolved agentview image convention: "
            f"{agentview_image_convention!r}"
        )
    return f"{OFFLINE_SEMANTICS_MARKDOWN}\n\n{detail}\n"


def _camera_views(
    camera_keys: Sequence[str],
    *,
    root: str = "cameras",
    group_name: str = "Cameras",
    view_name_prefix: str = "",
) -> rrb.Container | rrb.View:
    views = [
        rrb.Spatial2DView(
            origin=_camera_path(key, root=root),
            contents=[_camera_path(key, root=root)],
            name=f"{view_name_prefix}{key.rsplit('.', maxsplit=1)[-1]}",
        )
        for key in camera_keys
    ]
    if not views:
        return rrb.TextDocumentView(origin="summary", contents=["summary/metrics"])
    if len(views) == 1:
        return views[0]
    return rrb.Horizontal(*views, name=group_name)


def _build_offline_action_comparison_view() -> rrb.TimeSeriesView:
    """Build the explicitly ordered, independently toggleable GT/prediction chart."""

    return rrb.TimeSeriesView(
        origin="/",
        contents=list(ACTION_SERIES_PATHS),
        name="14-series action comparison (click legend to toggle)",
        plot_legend=rrb.PlotLegend(visible=True),
    )


def _build_offline_spatial_comparison(
    *,
    has_agentview_calibration: bool,
) -> rrb.Container | rrb.View:
    trajectory = rrb.Spatial3DView(
        origin="/",
        contents=list(
            OFFLINE_CALIBRATED_3D_CONTENTS
            if has_agentview_calibration
            else OFFLINE_3D_CONTENTS
        ),
        name="Free 3D - recorded geometry and commands",
        line_grid=True,
    )
    if not has_agentview_calibration:
        return trajectory
    calibrated_overlay = rrb.Spatial2DView(
        origin=CALIBRATED_AGENTVIEW_ENTITY_PATH,
        contents=list(OFFLINE_AGENTVIEW_OVERLAY_CONTENTS),
        name="Physically calibrated agentview",
    )
    return rrb.Tabs(
        calibrated_overlay,
        trajectory,
        active_tab=0,
        name="Calibrated agentview and free 3D",
    )


def build_offline_blueprint(
    camera_keys: Sequence[str],
    fps: float,
    *,
    has_agentview_calibration: bool = False,
) -> rrb.Blueprint:
    recorded_images = _camera_views(
        camera_keys,
        root="demonstration/cameras",
        group_name="Demonstration - recorded images",
        view_name_prefix="Demonstration (recorded) - ",
    )
    spatial_comparison = _build_offline_spatial_comparison(
        has_agentview_calibration=has_agentview_calibration,
    )
    action_comparison = _build_offline_action_comparison_view()
    charts = rrb.Tabs(
        action_comparison,
        rrb.TimeSeriesView(
            origin="comparison/action_error",
            contents=["comparison/action_error/**"],
            name="Absolute error",
        ),
        rrb.TimeSeriesView(
            origin="runtime/latency",
            contents=["runtime/latency/**"],
            name="Model-call latency",
        ),
        rrb.TextDocumentView(
            origin="documentation",
            contents=["documentation/offline_semantics"],
            name="Recorded vs predicted",
        ),
        rrb.TextDocumentView(
            origin="summary",
            contents=["summary/metrics"],
            name="Metrics",
        ),
        active_tab=0,
        name="Offline signals",
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                recorded_images,
                spatial_comparison,
                name="Offline comparison",
            ),
            charts,
            row_shares=[0.62, 0.38],
        ),
        rrb.TimePanel(timeline="global_frame", fps=fps, expanded=True),
        collapse_panels=False,
    )


def build_libero_blueprint(camera_keys: Sequence[str], fps: float) -> rrb.Blueprint:
    charts = rrb.Tabs(
        rrb.TimeSeriesView(
            origin="actions/executed",
            contents=["actions/executed/**"],
            name="Executed actions",
        ),
        rrb.TimeSeriesView(
            origin="rollout",
            contents=["rollout/reward", "rollout/success_so_far", "rollout/done"],
            name="Reward and terminal state",
        ),
        rrb.TimeSeriesView(
            origin="latency",
            contents=["latency/**"],
            name="Policy latency",
        ),
        rrb.TextDocumentView(
            origin="summary",
            contents=["summary/metrics"],
            name="Metrics",
        ),
        name="Closed-loop signals",
    )
    return rrb.Blueprint(
        rrb.Vertical(_camera_views(camera_keys), charts, row_shares=[0.62, 0.38]),
        rrb.TimePanel(timeline="global_step", fps=fps, expanded=True),
        collapse_panels=False,
    )


def initialize_recording(
    *,
    output_path: Path,
    mode: str,
    camera_keys: Sequence[str],
    fps: float,
    offline_camera_calibration: FixedCameraCalibration | None = None,
    agentview_image_convention: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "offline":
        if (offline_camera_calibration is None) != (agentview_image_convention is None):
            raise ValueError(
                "Offline camera calibration and resolved agentview image convention "
                "must be supplied together"
            )
        application_id = "robot_practice_act_offline"
        blueprint = build_offline_blueprint(
            camera_keys,
            fps,
            has_agentview_calibration=offline_camera_calibration is not None,
        )
    elif mode == "libero":
        if offline_camera_calibration is not None or agentview_image_convention is not None:
            raise ValueError(
                "Offline camera calibration and image convention are only supported "
                "for offline recordings"
            )
        application_id = "robot_practice_act_libero"
        blueprint = build_libero_blueprint(camera_keys, fps)
    else:
        raise ValueError(f"Unknown recording mode: {mode}")
    rr.init(application_id, spawn=False)
    rr.save(output_path)
    if mode == "offline":
        log_offline_static_context(
            rr,
            agentview_calibration=offline_camera_calibration,
            agentview_image_convention=agentview_image_convention,
        )
    rr.send_blueprint(blueprint)


def finish_recording() -> None:
    rr.disconnect()


def _validate_action(action: Sequence[float], name: str) -> NDArray[np.float64]:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain seven finite values, got {array}")
    return array


def _libero_action_translation(action: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(action[:3], -1.0, 1.0) * LIBERO_OSC_TRANSLATION_SCALE_M


def _validate_observation_state(
    state: Sequence[float] | torch.Tensor | NDArray[Any],
) -> NDArray[np.float64]:
    if isinstance(state, torch.Tensor):
        array = state.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        array = np.asarray(state, dtype=np.float64)
    if array.shape != (8,) or not np.isfinite(array).all():
        raise ValueError(f"observation_state must contain eight finite values, got {array}")
    return array


def _validate_action_chunk(
    chunk: Sequence[Sequence[float]] | NDArray[Any],
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(chunk, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (7,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape (horizon, 7) with finite values, got {array}")
    return array


def integrate_libero_action_chunk(
    observation_state: Sequence[float] | torch.Tensor | NDArray[Any],
    predicted_chunk: Sequence[Sequence[float]] | NDArray[Any],
    valid_horizon: Sequence[bool] | NDArray[Any],
) -> NDArray[np.float64]:
    """Build a clipped OSC translation preview anchored at recorded EEF XYZ."""

    state = _validate_observation_state(observation_state)
    chunk = _validate_action_chunk(predicted_chunk, "predicted_chunk")
    valid = np.asarray(valid_horizon, dtype=bool)
    if valid.shape != (len(chunk),):
        raise ValueError(
            "valid_horizon must match predicted_chunk length; "
            f"got {valid.shape} for {chunk.shape}"
        )
    invalid_indices = np.flatnonzero(~valid)
    valid_length = int(invalid_indices[0]) if len(invalid_indices) else len(valid)
    if valid[valid_length:].any():
        raise ValueError("valid_horizon must be a contiguous valid prefix")
    translations = (
        np.clip(chunk[:valid_length, :3], -1.0, 1.0)
        * LIBERO_OSC_TRANSLATION_SCALE_M
    )
    offsets = np.concatenate(
        [np.zeros((1, 3), dtype=np.float64), np.cumsum(translations, axis=0)],
        axis=0,
    )
    return state[:3] + offsets


def _log_agentview_calibration(
    recorder: Recorder,
    calibration: FixedCameraCalibration,
) -> None:
    recorder.log(
        CALIBRATED_AGENTVIEW_ENTITY_PATH,
        rr.Transform3D(
            translation=calibration.position_display,
            mat3x3=calibration.rotation_display_from_camera,
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    recorder.log(
        CALIBRATED_AGENTVIEW_ENTITY_PATH,
        rr.Pinhole(
            image_from_camera=calibration.image_from_camera,
            resolution=calibration.resolution_wh,
            camera_xyz=rr.ViewCoordinates.RUB,
            image_plane_distance=AGENTVIEW_IMAGE_PLANE_DISTANCE_M,
            color=AGENTVIEW_FRUSTUM_COLOR,
            line_width=AGENTVIEW_FRUSTUM_LINE_WIDTH,
        ),
        static=True,
    )


def log_offline_static_context(
    recorder: Recorder,
    *,
    agentview_calibration: FixedCameraCalibration | None = None,
    agentview_image_convention: str | None = None,
) -> None:
    if (agentview_calibration is None) != (agentview_image_convention is None):
        raise ValueError(
            "Agentview calibration and resolved image convention must be supplied together"
        )
    recorder.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    recorder.log(
        "display_frame/base",
        rr.Points3D(
            [[0.0, 0.0, 0.0]],
            colors=[(210, 210, 210)],
            radii=0.025,
            labels=["Robot base / display origin"],
            show_labels=False,
        ),
        static=True,
    )
    recorder.log(
        "display_frame/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3), dtype=np.float64),
            vectors=np.eye(3, dtype=np.float64) * 0.25,
            colors=AXIS_COLORS,
            radii=0.006,
            labels=["X", "Y Forward", "Z Up"],
            show_labels=True,
        ),
        static=True,
    )
    for spec in ACTION_SERIES_SPECS:
        recorder.log(
            spec.path,
            rr.SeriesLines(
                colors=[spec.color],
                widths=[2.0],
                names=[spec.name],
                visible_series=[True],
            ),
            static=True,
        )
    recorder.log(
        "documentation/offline_semantics",
        rr.TextDocument(
            offline_semantics_markdown(agentview_image_convention),
            media_type="text/markdown",
        ),
        static=True,
    )
    if agentview_calibration is not None:
        _log_agentview_calibration(recorder, agentview_calibration)


def _log_panda_skeleton(
    recorder: Recorder,
    *,
    pose: PandaSkeletonPose,
    display_frame: DisplayFrame,
    stale: bool,
) -> None:
    skeleton_points = display_frame.transform_points(pose.skeleton_points_native)
    left_finger = display_frame.transform_points(pose.left_finger_points_native)
    right_finger = display_frame.transform_points(pose.right_finger_points_native)
    status = " (stale)" if stale else ""
    recorder.log(
        "demonstration/robot/skeleton",
        rr.LineStrips3D(
            [skeleton_points],
            colors=[DEMONSTRATION_COLOR],
            radii=0.01,
            labels=[f"Demonstration arm - IK reconstructed{status}"],
            show_labels=False,
        ),
    )
    recorder.log(
        "demonstration/robot/joints",
        rr.Points3D(
            skeleton_points,
            colors=[DEMONSTRATION_COLOR],
            radii=0.014,
            labels=PANDA_SKELETON_LABELS,
            show_labels=False,
        ),
    )
    recorder.log(
        "demonstration/robot/gripper",
        rr.LineStrips3D(
            [left_finger, right_finger],
            colors=[DEMONSTRATION_COLOR, DEMONSTRATION_COLOR],
            radii=0.009,
        ),
    )
    recorder.log("demonstration/robot/ik_stale", rr.Scalars(int(stale)))


def log_offline_frame(
    recorder: Recorder,
    *,
    global_frame: int,
    episode_index: int,
    frame_index: int,
    batch_id: int,
    observation_state: Sequence[float] | torch.Tensor | NDArray[Any],
    target_action: Sequence[float],
    predicted_action: Sequence[float],
    target_chunk: Sequence[Sequence[float]] | NDArray[Any],
    predicted_chunk: Sequence[Sequence[float]] | NDArray[Any],
    valid_horizon: Sequence[bool] | NDArray[Any],
    trajectory_history: OfflineTrajectoryHistory,
    command_history: OfflineCommandHistory,
    display_frame: DisplayFrame,
    skeleton_pose: PandaSkeletonPose | None,
    skeleton_stale: bool,
    clear_skeleton: bool,
    model_query_latency_ms: float | None,
    images: Mapping[str, torch.Tensor | NDArray[Any]],
    agentview_calibration: FixedCameraCalibration | None = None,
    agentview_image_convention: str | None = None,
) -> None:
    """Log one observation after setting the global frame timeline exactly once."""

    target = _validate_action(target_action, "target_action")
    predicted = _validate_action(predicted_action, "predicted_action")
    target_chunk_array = _validate_action_chunk(target_chunk, "target_chunk")
    predicted_chunk_array = _validate_action_chunk(predicted_chunk, "predicted_chunk")
    if target_chunk_array.shape != predicted_chunk_array.shape:
        raise ValueError(
            "target_chunk and predicted_chunk must have the same shape; "
            f"got {target_chunk_array.shape} and {predicted_chunk_array.shape}"
        )
    valid = np.asarray(valid_horizon, dtype=bool)
    if valid.shape != (len(predicted_chunk_array),):
        raise ValueError(
            "valid_horizon must match predicted_chunk length; "
            f"got {valid.shape} for {predicted_chunk_array.shape}"
        )
    if not len(valid) or not valid[0]:
        raise ValueError("The current first action must be valid for offline visualization")
    if (agentview_calibration is None) != (agentview_image_convention is None):
        raise ValueError(
            "Agentview calibration and resolved image convention must be supplied together"
        )
    prepared_images = {
        key: image_to_hwc_uint8(image)
        for key, image in images.items()
    }
    corrected_agentview: NDArray[np.uint8] | None = None
    if agentview_calibration is not None:
        if AGENTVIEW_CAMERA_KEY not in prepared_images:
            raise ValueError(
                f"Calibrated visualization requires image {AGENTVIEW_CAMERA_KEY!r}"
            )
        recorded_agentview = prepared_images[AGENTVIEW_CAMERA_KEY]
        agentview_calibration.validate_hwc_image(recorded_agentview)
        corrected_agentview = physical_agentview_image(
            recorded_agentview,
            agentview_image_convention,
        )
        agentview_calibration.validate_hwc_image(corrected_agentview)
    recorded_path = trajectory_history.append(
        episode_index=episode_index,
        frame_index=frame_index,
        observation_state=observation_state,
    )
    command_segments = command_history.append(
        episode_index=episode_index,
        frame_index=frame_index,
        observation_state=observation_state,
        target_action=target,
        predicted_action=predicted,
    )
    prediction_preview = integrate_libero_action_chunk(
        observation_state,
        predicted_chunk_array,
        valid,
    )
    recorded_path_display = display_frame.transform_points(recorded_path)
    target_commands_display = display_frame.transform_points(command_segments.target)
    predicted_commands_display = display_frame.transform_points(
        command_segments.predicted
    )
    prediction_preview_display = display_frame.transform_points(prediction_preview)
    if skeleton_stale and skeleton_pose is None:
        raise ValueError("skeleton_stale requires a skeleton_pose")
    recorder.set_time("global_frame", sequence=global_frame)

    if clear_skeleton:
        recorder.log("demonstration/robot", rr.Clear(recursive=True))
    if skeleton_pose is not None:
        _log_panda_skeleton(
            recorder,
            pose=skeleton_pose,
            display_frame=display_frame,
            stale=skeleton_stale,
        )
    for key, image_hwc in prepared_images.items():
        recorder.log(
            _camera_path(key, root="demonstration/cameras"),
            rr.Image(image_hwc).compress(),
        )
    if corrected_agentview is not None:
        recorder.log(
            CALIBRATED_AGENTVIEW_ENTITY_PATH,
            rr.Image(corrected_agentview).compress(),
        )
    recorder.log("demonstration/context/episode_index", rr.Scalars(episode_index))
    recorder.log("demonstration/context/frame_index", rr.Scalars(frame_index))
    recorder.log("runtime/batch_id", rr.Scalars(batch_id))
    recorder.log(
        "demonstration/trajectory/eef_recorded",
        rr.LineStrips3D(
            [recorded_path_display],
            colors=[DEMONSTRATION_COLOR],
            radii=0.003,
            labels=["Demonstration (recorded)"],
            show_labels=False,
        ),
    )
    recorder.log(
        "demonstration/trajectory/eef_current",
        rr.Points3D(
            [recorded_path_display[-1]],
            colors=[DEMONSTRATION_COLOR],
            radii=0.008,
            labels=["Demonstration EEF (recorded)"],
            show_labels=False,
        ),
    )
    recorder.log(
        "comparison/commands/gt_first_action_segments",
        rr.LineStrips3D(
            target_commands_display,
            colors=[GT_COMMAND_COLOR],
            radii=0.004,
            labels=["GT first-action command segments"],
            show_labels=False,
        ),
    )
    recorder.log(
        "comparison/commands/pred_first_action_segments",
        rr.LineStrips3D(
            predicted_commands_display,
            colors=[PREDICTION_COLOR],
            radii=0.005,
            labels=[
                "Pred first-action command segments (teacher-forced, not executed)"
            ],
            show_labels=False,
        ),
    )
    recorder.log(
        "prediction/not_executed/chunk_preview",
        rr.LineStrips3D(
            [prediction_preview_display],
            colors=[PREDICTION_PREVIEW_COLOR],
            radii=0.003,
            labels=["Pred future command preview (teacher-forced, not executed)"],
            show_labels=False,
        ),
    )
    recorder.log(
        "prediction/not_executed/chunk_endpoint",
        rr.Points3D(
            [prediction_preview_display[-1]],
            colors=[PREDICTION_PREVIEW_ENDPOINT_COLOR],
            radii=0.008,
            labels=["Pred future command endpoint (not executed)"],
            show_labels=False,
        ),
    )
    for spec in ACTION_SERIES_SPECS:
        values = target if spec.source == "gt" else predicted
        recorder.log(spec.path, rr.Scalars(float(values[spec.action_index])))
    for index, action_name in enumerate(ACTION_NAMES):
        recorder.log(
            f"comparison/action_error/{action_name}",
            rr.Scalars(float(abs(predicted[index] - target[index]))),
        )
    if model_query_latency_ms is not None:
        recorder.log(
            "runtime/latency/model_query_ms", rr.Scalars(model_query_latency_ms)
        )


def log_libero_step(
    recorder: Recorder,
    *,
    global_step: int,
    episode_index: int,
    step_index: int,
    action: Sequence[float],
    reward: float,
    success_so_far: bool,
    done_after_step: bool,
    policy_step_latency_ms: float,
    model_query_latency_ms: float | None,
    images: Mapping[str, torch.Tensor | NDArray[Any]],
) -> None:
    """Log one control transition on the single global-step timeline."""

    executed = _validate_action(action, "executed_action")
    recorder.set_time("global_step", sequence=global_step)
    for key, image in images.items():
        recorder.log(_camera_path(key), rr.Image(image_to_hwc_uint8(image)).compress())
    recorder.log("context/episode_index", rr.Scalars(episode_index))
    recorder.log("context/step_index", rr.Scalars(step_index))
    for index, action_name in enumerate(ACTION_NAMES):
        recorder.log(
            f"actions/executed/{action_name}", rr.Scalars(float(executed[index]))
        )
    recorder.log("rollout/reward", rr.Scalars(float(reward)))
    recorder.log("rollout/success_so_far", rr.Scalars(int(success_so_far)))
    recorder.log("rollout/done", rr.Scalars(int(done_after_step)))
    recorder.log("latency/policy_step_ms", rr.Scalars(policy_step_latency_ms))
    if model_query_latency_ms is not None:
        recorder.log("latency/model_query_ms", rr.Scalars(model_query_latency_ms))


def _finite_metric_number(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _format_offline_metric(value: Any, kind: str) -> str:
    number = _finite_metric_number(value)
    if number is None:
        return "N/A"
    if kind == "integer":
        return str(int(number)) if number >= 0 and number.is_integer() else "N/A"
    if kind == "float4":
        return f"{number:.4f}"
    if kind == "float5":
        return f"{number:.5f}"
    if kind == "latency":
        return f"{number:.2f} ms"
    raise ValueError(f"Unknown offline metric format: {kind}")


def _offline_metrics_markdown(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    definitions = metrics.get("metric_definitions", {})
    if not isinstance(definitions, Mapping):
        definitions = {}
    lines = [
        "# Offline evaluation metrics",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
    ]
    for label, key, kind in OFFLINE_METRIC_ROWS:
        value = _format_offline_metric(aggregate.get(key), kind)
        lines.append(f"| {label} | {value} | {definitions.get(key, '')} |")

    lines.extend(("", "## Per-dimension first-action MAE", ""))
    per_dimension = aggregate.get("per_dimension_mae")
    valid_values: list[float] | None = None
    if (
        isinstance(per_dimension, Sequence)
        and not isinstance(per_dimension, (str, bytes))
        and len(per_dimension) == len(ACTION_NAMES)
    ):
        converted = [_finite_metric_number(value) for value in per_dimension]
        if all(value is not None for value in converted):
            valid_values = [float(value) for value in converted if value is not None]
    if valid_values is None:
        lines.append("Per-dimension MAE unavailable")
    else:
        lines.extend(("| Dimension | MAE |", "|---|---:|"))
        lines.extend(
            f"| {name} | {value:.4f} |"
            for name, value in zip(ACTION_NAMES, valid_values, strict=True)
        )
    return "\n".join(lines)


def metrics_markdown(metrics: Mapping[str, Any]) -> str:
    if metrics.get("mode") == "offline":
        return _offline_metrics_markdown(metrics)
    aggregate = metrics.get("aggregate", {})
    definitions = metrics.get("metric_definitions", {})
    lines = [f"# {metrics.get('mode', 'evaluation')} metrics", ""]
    for name, value in aggregate.items():
        rendered = json.dumps(value, ensure_ascii=False)
        description = definitions.get(name, "")
        suffix = f" — {description}" if description else ""
        lines.append(f"- **{name}**: `{rendered}`{suffix}")
    return "\n".join(lines)


def log_metric_summary(recorder: Recorder, metrics: Mapping[str, Any]) -> None:
    recorder.log(
        "summary/metrics",
        rr.TextDocument(metrics_markdown(metrics), media_type="text/markdown"),
        static=True,
    )
