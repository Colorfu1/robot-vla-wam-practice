from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import robot_practice.evaluation.rerun_visualization as viz


class FakeRecorder:
    def __init__(self) -> None:
        self.times: list[tuple[str, int]] = []
        self.logs: list[tuple[str, Any, bool]] = []

    def set_time(self, timeline: str, *, sequence: int) -> None:
        self.times.append((timeline, sequence))

    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None:
        self.logs.append((entity_path, entity, static))


def _patch_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viz.rr, "Scalars", lambda value: value)

    class FakeImage:
        def __init__(self, image: np.ndarray) -> None:
            self.image = image

        def compress(self) -> "FakeImage":
            return self

    monkeypatch.setattr(viz.rr, "Image", FakeImage)

    class FakeGeometry:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.positions = args[0] if args else kwargs.get("positions")
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(viz.rr, "LineStrips3D", FakeGeometry)
    monkeypatch.setattr(viz.rr, "Points3D", FakeGeometry)
    monkeypatch.setattr(viz.rr, "Arrows3D", FakeGeometry)
    monkeypatch.setattr(viz.rr, "SeriesLines", FakeGeometry)
    monkeypatch.setattr(viz.rr, "Clear", FakeGeometry)
    monkeypatch.setattr(viz.rr, "Transform3D", FakeGeometry)
    monkeypatch.setattr(viz.rr, "Pinhole", FakeGeometry)


def _skeleton_pose() -> viz.PandaSkeletonPose:
    skeleton = np.zeros((11, 3), dtype=np.float64)
    skeleton[:, 0] = np.linspace(0.0, 0.5, len(skeleton))
    return viz.PandaSkeletonPose(
        skeleton_points_native=skeleton,
        left_finger_points_native=np.asarray(
            [[0.5, 0.0, 0.0], [0.55, -0.02, 0.0], [0.6, -0.03, 0.0]]
        ),
        right_finger_points_native=np.asarray(
            [[0.5, 0.0, 0.0], [0.55, 0.02, 0.0], [0.6, 0.03, 0.0]]
        ),
        arm_qpos=np.zeros(7),
        gripper_qpos=np.zeros(2),
        position_residual_m=0.0,
        rotation_residual_rad=0.0,
        iterations=1,
    )


def _measured_pose() -> viz.PandaMeasuredPose:
    skeleton = np.zeros((11, 3), dtype=np.float64)
    skeleton[:, 0] = np.linspace(0.0, 0.5, len(skeleton))
    return viz.PandaMeasuredPose(
        skeleton_points_native=skeleton,
        left_finger_points_native=np.asarray(
            [[0.5, 0.0, 0.0], [0.55, -0.02, 0.0], [0.6, -0.03, 0.0]]
        ),
        right_finger_points_native=np.asarray(
            [[0.5, 0.0, 0.0], [0.55, 0.02, 0.0], [0.6, 0.03, 0.0]]
        ),
        arm_qpos=np.zeros(7),
        gripper_qpos=np.zeros(2),
    )


def _offline_frame_kwargs() -> dict[str, object]:
    return {
        "global_frame": 12,
        "episode_index": 823,
        "frame_index": 4,
        "batch_id": 2,
        "observation_state": np.array([0.1, 0.2, 0.3, 0, 0, 0, -1, 1]),
        "target_action": np.zeros(7),
        "predicted_action": np.ones(7),
        "target_chunk": np.zeros((2, 7)),
        "predicted_chunk": np.ones((2, 7)),
        "valid_horizon": np.array([True, True]),
        "trajectory_history": viz.OfflineTrajectoryHistory(),
        "command_history": viz.OfflineCommandHistory(),
        "display_frame": viz.DisplayFrame.libero_y_forward([0.1, 0.2, 0.3]),
        "skeleton_pose": _skeleton_pose(),
        "skeleton_stale": False,
        "clear_skeleton": True,
        "model_query_latency_ms": 3.5,
        "images": {"observation.images.image": torch.zeros(3, 4, 5)},
    }


def _camera_calibration() -> viz.FixedCameraCalibration:
    return viz.FixedCameraCalibration(
        camera_name="agentview",
        image_width=5,
        image_height=4,
        image_channels=3,
        position_display=np.asarray([0.1, 0.2, 0.3]),
        rotation_display_from_camera=np.eye(3),
        fov_y_rad=np.deg2rad(45.0),
        image_from_camera=np.asarray(
            [
                [4.0, 0.0, 2.5],
                [0.0, 4.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def _libero_step_kwargs() -> dict[str, object]:
    return {
        "global_step": 8,
        "episode_index": 0,
        "step_index": 8,
        "observation_state": np.zeros(8),
        "next_observation_state": np.asarray([0.01, 0.02, 0.03, 0, 0, 0, 0, 0]),
        "measured_pose": _measured_pose(),
        "display_frame": viz.DisplayFrame.libero_y_forward([0, 0, 0]),
        "trajectory_history": viz.LiberoTrajectoryHistory(),
        "action": np.zeros(7),
        "queued_actions": np.empty((0, 7)),
        "discarded_actions": np.empty((0, 7)),
        "new_prediction_chunk": None,
        "model_queried": False,
        "plan_step_index": 1,
        "reward": 0.0,
        "success_so_far": False,
        "done_after_step": False,
        "policy_step_latency_ms": 1.0,
        "model_query_latency_ms": None,
        "images": {"observation.images.image": torch.zeros(3, 4, 5)},
        "post_action_agentview": torch.zeros(3, 4, 5),
        "agentview_calibration": _camera_calibration(),
        "agentview_image_convention": "lerobot_180",
    }


def test_image_conversion_supports_normalized_chw() -> None:
    image = torch.zeros(3, 4, 5)
    image[0] = 1.0
    converted = viz.image_to_hwc_uint8(image)
    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert converted[0, 0].tolist() == [255, 0, 0]


def test_offline_frame_sets_global_timeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    viz.log_offline_frame(recorder, **_offline_frame_kwargs())
    assert recorder.times == [("global_frame", 12)]
    paths = {path for path, _, _ in recorder.logs}
    expected = {
        "demonstration/cameras/image",
        "demonstration/context/episode_index",
        "demonstration/context/frame_index",
        "runtime/batch_id",
        "demonstration/robot",
        "demonstration/robot/skeleton",
        "demonstration/robot/joints",
        "demonstration/robot/gripper",
        "demonstration/robot/ik_stale",
        "demonstration/trajectory/eef_recorded",
        "demonstration/trajectory/eef_current",
        "comparison/commands/gt_first_action_segments",
        "comparison/commands/pred_first_action_segments",
        "prediction/not_executed/chunk_preview",
        "prediction/not_executed/chunk_endpoint",
        "runtime/latency/model_query_ms",
    }
    expected.update(viz.ACTION_SERIES_PATHS)
    expected.update(f"comparison/action_error/{name}" for name in viz.ACTION_NAMES)
    assert paths == expected

    entities = {path: entity for path, entity, _ in recorder.logs}
    hidden_label_contract = {
        "demonstration/robot/skeleton": [
            "Demonstration arm - IK reconstructed"
        ],
        "demonstration/robot/joints": viz.PANDA_SKELETON_LABELS,
        "demonstration/trajectory/eef_recorded": ["Demonstration (recorded)"],
        "demonstration/trajectory/eef_current": [
            "Demonstration EEF (recorded)"
        ],
        "comparison/commands/gt_first_action_segments": [
            "GT first-action command segments"
        ],
        "comparison/commands/pred_first_action_segments": [
            "Pred first-action command segments (teacher-forced, not executed)"
        ],
        "prediction/not_executed/chunk_preview": [
            "Pred future command preview (teacher-forced, not executed)"
        ],
        "prediction/not_executed/chunk_endpoint": [
            "Pred future command endpoint (not executed)"
        ],
    }
    for path, labels in hidden_label_contract.items():
        assert entities[path].kwargs["labels"] == labels
        assert entities[path].kwargs["show_labels"] is False
    assert "labels" not in entities["demonstration/robot/gripper"].kwargs
    assert "show_labels" not in entities["demonstration/robot/gripper"].kwargs
    assert entities["comparison/commands/gt_first_action_segments"].kwargs[
        "colors"
    ] == [viz.GT_COMMAND_COLOR]
    assert entities["comparison/commands/pred_first_action_segments"].kwargs[
        "colors"
    ] == [viz.PREDICTION_COLOR]
    assert entities["prediction/not_executed/chunk_preview"].kwargs["colors"] == [
        viz.PREDICTION_PREVIEW_COLOR
    ]
    assert entities["prediction/not_executed/chunk_endpoint"].kwargs["colors"] == [
        viz.PREDICTION_PREVIEW_ENDPOINT_COLOR
    ]


def test_action_chunk_preview_clips_and_integrates_controller_commands() -> None:
    state = np.array([1.0, 2.0, 3.0, 0, 0, 0, -1, 1])
    chunk = np.zeros((3, 7))
    chunk[0, :3] = [2.0, -2.0, 0.5]
    chunk[1, :3] = [1.0, 0.0, -1.0]
    points = viz.integrate_libero_action_chunk(
        state,
        chunk,
        np.array([True, True, False]),
    )
    np.testing.assert_allclose(
        points,
        [
            [1.0, 2.0, 3.0],
            [1.05, 1.95, 3.025],
            [1.10, 1.95, 2.975],
        ],
    )


def test_action_chunk_preview_all_padding_contains_only_anchor() -> None:
    state = np.zeros(8)
    points = viz.integrate_libero_action_chunk(
        state,
        np.ones((2, 7)),
        np.array([False, False]),
    )
    assert points.shape == (1, 3)
    np.testing.assert_array_equal(points[0], state[:3])


@pytest.mark.parametrize(
    ("state", "chunk", "valid", "message"),
    [
        (np.zeros(7), np.zeros((2, 7)), [True, True], "eight finite"),
        (np.full(8, np.nan), np.zeros((2, 7)), [True, True], "eight finite"),
        (np.zeros(8), np.zeros((2, 6)), [True, True], "shape"),
        (np.zeros(8), np.full((2, 7), np.inf), [True, True], "finite"),
        (np.zeros(8), np.zeros((2, 7)), [True], "must match"),
        (np.zeros(8), np.zeros((3, 7)), [True, False, True], "contiguous"),
    ],
)
def test_action_chunk_preview_rejects_invalid_inputs(
    state: np.ndarray,
    chunk: np.ndarray,
    valid: list[bool],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        viz.integrate_libero_action_chunk(state, chunk, valid)


def test_offline_trajectory_history_resets_between_episodes() -> None:
    history = viz.OfflineTrajectoryHistory()
    first = history.append(
        episode_index=823,
        frame_index=0,
        observation_state=np.array([0.1, 0.2, 0.3, 0, 0, 0, -1, 1]),
    )
    second = history.append(
        episode_index=823,
        frame_index=1,
        observation_state=np.array([0.2, 0.3, 0.4, 0, 0, 0, -1, 1]),
    )
    reset = history.append(
        episode_index=826,
        frame_index=0,
        observation_state=np.array([0.5, 0.6, 0.7, 0, 0, 0, -1, 1]),
    )
    assert first.shape == (1, 3)
    assert second.shape == (2, 3)
    assert reset.shape == (1, 3)
    np.testing.assert_allclose(reset[0], [0.5, 0.6, 0.7])


def test_offline_command_history_keeps_independent_segments_and_resets() -> None:
    history = viz.OfflineCommandHistory()
    first = history.append(
        episode_index=823,
        frame_index=0,
        observation_state=np.array([1.0, 2.0, 3.0, 0, 0, 0, -1, 1]),
        target_action=np.array([2.0, -2.0, 0.5, 0, 0, 0, 0]),
        predicted_action=np.array([-1.0, 0.5, 0.0, 0, 0, 0, 0]),
    )
    np.testing.assert_allclose(
        first.target,
        [[[1.0, 2.0, 3.0], [1.05, 1.95, 3.025]]],
    )
    np.testing.assert_allclose(
        first.predicted,
        [[[1.0, 2.0, 3.0], [0.95, 2.025, 3.0]]],
    )

    second = history.append(
        episode_index=823,
        frame_index=1,
        observation_state=np.array([4.0, 5.0, 6.0, 0, 0, 0, -1, 1]),
        target_action=np.zeros(7),
        predicted_action=np.zeros(7),
    )
    assert second.target.shape == (2, 2, 3)
    assert second.predicted.shape == (2, 2, 3)
    np.testing.assert_allclose(second.target[1, 0], [4.0, 5.0, 6.0])
    assert not np.array_equal(second.target[0, 1], second.target[1, 0])

    reset = history.append(
        episode_index=826,
        frame_index=0,
        observation_state=np.array([7.0, 8.0, 9.0, 0, 0, 0, -1, 1]),
        target_action=np.zeros(7),
        predicted_action=np.zeros(7),
    )
    assert reset.target.shape == (1, 2, 3)
    assert reset.predicted.shape == (1, 2, 3)
    np.testing.assert_allclose(reset.target[0, 0], [7.0, 8.0, 9.0])


def test_offline_static_paths_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FakeRecorder()
    _patch_entities(monkeypatch)
    monkeypatch.setattr(
        viz.rr,
        "ViewCoordinates",
        SimpleNamespace(RIGHT_HAND_Z_UP=object()),
    )
    monkeypatch.setattr(viz.rr, "TextDocument", lambda *args, **kwargs: (args, kwargs))
    viz.log_offline_static_context(recorder)
    expected = [
        ("/", True),
        ("display_frame/base", True),
        ("display_frame/axes", True),
        *((path, True) for path in viz.ACTION_SERIES_PATHS),
        ("documentation/offline_semantics", True),
    ]
    assert [(path, static) for path, _, static in recorder.logs] == expected

    base = next(entity for path, entity, _ in recorder.logs if path == "display_frame/base")
    assert base.kwargs["labels"] == ["Robot base / display origin"]
    assert base.kwargs["show_labels"] is False

    axes = next(entity for path, entity, _ in recorder.logs if path == "display_frame/axes")
    assert axes.kwargs["labels"] == ["X", "Y Forward", "Z Up"]
    assert axes.kwargs["show_labels"] is True
    assert tuple(axes.kwargs["colors"]) == viz.AXIS_COLORS

    styles = [
        entity
        for path, entity, _ in recorder.logs
        if path in viz.ACTION_SERIES_PATHS
    ]
    assert len(styles) == 14
    assert [style.kwargs["names"][0] for style in styles] == [
        spec.name for spec in viz.ACTION_SERIES_SPECS
    ]
    assert len({tuple(style.kwargs["colors"][0]) for style in styles}) == 14
    assert all(style.kwargs["visible_series"] == [True] for style in styles)


def test_offline_static_context_logs_calibrated_agentview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = FakeRecorder()
    _patch_entities(monkeypatch)
    world_coordinates = object()
    camera_coordinates = object()
    monkeypatch.setattr(
        viz.rr,
        "ViewCoordinates",
        SimpleNamespace(
            RIGHT_HAND_Z_UP=world_coordinates,
            RUB=camera_coordinates,
        ),
    )
    relation = object()
    monkeypatch.setattr(
        viz.rr,
        "TransformRelation",
        SimpleNamespace(ParentFromChild=relation),
    )
    monkeypatch.setattr(viz.rr, "TextDocument", lambda *args, **kwargs: (args, kwargs))
    calibration = _camera_calibration()

    viz.log_offline_static_context(
        recorder,
        agentview_calibration=calibration,
        agentview_image_convention="lerobot_180",
    )

    camera_entities = [
        entity
        for path, entity, static in recorder.logs
        if path == viz.CALIBRATED_AGENTVIEW_ENTITY_PATH and static
    ]
    assert len(camera_entities) == 2
    transform, pinhole = camera_entities
    np.testing.assert_allclose(
        transform.kwargs["translation"], calibration.position_display
    )
    np.testing.assert_allclose(
        transform.kwargs["mat3x3"], calibration.rotation_display_from_camera
    )
    assert transform.kwargs["relation"] is relation
    np.testing.assert_allclose(
        pinhole.kwargs["image_from_camera"], calibration.image_from_camera
    )
    assert pinhole.kwargs["resolution"] == (5, 4)
    assert pinhole.kwargs["camera_xyz"] is camera_coordinates
    assert pinhole.kwargs["image_plane_distance"] == 0.25
    assert pinhole.kwargs["color"] == (0, 220, 255, 210)
    assert pinhole.kwargs["line_width"] == 0.003


def test_offline_action_chart_has_exact_adjacent_gt_pred_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legend = object()
    monkeypatch.setattr(
        viz.rrb,
        "PlotLegend",
        lambda *, visible: legend if visible else None,
    )
    monkeypatch.setattr(viz.rrb, "TimeSeriesView", lambda **kwargs: kwargs)

    chart = viz._build_offline_action_comparison_view()

    assert chart["contents"] == [
        "comparison/actions/00_x/00_gt",
        "comparison/actions/00_x/01_pred",
        "comparison/actions/01_y/00_gt",
        "comparison/actions/01_y/01_pred",
        "comparison/actions/02_z/00_gt",
        "comparison/actions/02_z/01_pred",
        "comparison/actions/03_roll/00_gt",
        "comparison/actions/03_roll/01_pred",
        "comparison/actions/04_pitch/00_gt",
        "comparison/actions/04_pitch/01_pred",
        "comparison/actions/05_yaw/00_gt",
        "comparison/actions/05_yaw/01_pred",
        "comparison/actions/06_gripper/00_gt",
        "comparison/actions/06_gripper/01_pred",
    ]
    assert chart["plot_legend"] is legend


def test_offline_3d_view_contains_recorded_and_command_geometry() -> None:
    assert viz.OFFLINE_3D_CONTENTS == (
        "display_frame/**",
        "demonstration/robot/**",
        "demonstration/trajectory/**",
        "comparison/commands/**",
        "prediction/not_executed/chunk_preview",
        "prediction/not_executed/chunk_endpoint",
    )
    assert viz.OFFLINE_CALIBRATED_3D_CONTENTS == (
        *viz.OFFLINE_3D_CONTENTS,
        "calibration/agentview_corrected",
    )


def test_calibrated_agentview_is_first_tab_and_free_3d_contains_frustum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(viz.rrb, "Spatial2DView", lambda **kwargs: ("2d", kwargs))
    monkeypatch.setattr(viz.rrb, "Spatial3DView", lambda **kwargs: ("3d", kwargs))
    monkeypatch.setattr(
        viz.rrb,
        "Tabs",
        lambda *args, **kwargs: ("tabs", args, kwargs),
    )

    tabs = viz._build_offline_spatial_comparison(has_agentview_calibration=True)

    assert tabs[0] == "tabs"
    overlay, free_3d = tabs[1]
    assert tabs[2]["active_tab"] == 0
    assert overlay[0] == "2d"
    assert overlay[1]["origin"] == viz.CALIBRATED_AGENTVIEW_ENTITY_PATH
    assert overlay[1]["name"] == "Physically calibrated agentview"
    assert overlay[1]["contents"] == list(viz.OFFLINE_AGENTVIEW_OVERLAY_CONTENTS)
    assert free_3d[0] == "3d"
    assert free_3d[1]["contents"] == list(viz.OFFLINE_CALIBRATED_3D_CONTENTS)


def test_libero_spatial_view_contains_measured_rollout_and_calibrated_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(viz.rrb, "Spatial2DView", lambda **kwargs: ("2d", kwargs))
    monkeypatch.setattr(viz.rrb, "Spatial3DView", lambda **kwargs: ("3d", kwargs))
    monkeypatch.setattr(
        viz.rrb,
        "Tabs",
        lambda *args, **kwargs: ("tabs", args, kwargs),
    )

    tabs = viz._build_libero_spatial_view()

    overlay, free_3d = tabs[1]
    assert tabs[2]["active_tab"] == 0
    assert overlay[1]["origin"] == viz.LIBERO_CALIBRATED_AGENTVIEW_PATH
    assert overlay[1]["contents"] == list(viz.LIBERO_AGENTVIEW_OVERLAY_CONTENTS)
    assert free_3d[1]["contents"] == list(viz.LIBERO_CALIBRATED_3D_CONTENTS)
    assert "rollout/robot/**" in free_3d[1]["contents"]
    assert viz.LIBERO_3D_CONTENTS == (
        "display_frame/**",
        "rollout/robot/**",
        "rollout/trajectory/**",
        "rollout/command/**",
        "rollout/response/**",
        "rollout/plan/queued/**",
        "rollout/plan/discarded/**",
        "rollout/plan/new_prediction_chunk/**",
    )
    assert viz.LIBERO_CALIBRATED_3D_CONTENTS == (
        *viz.LIBERO_3D_CONTENTS,
        "rollout/calibration/**",
    )


def test_libero_blueprint_expands_native_blueprint_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = object()
    panel_calls: list[dict[str, object]] = []

    def blueprint_panel(**kwargs: object) -> object:
        panel_calls.append(kwargs)
        return panel

    monkeypatch.setattr(viz.rrb, "BlueprintPanel", blueprint_panel)
    monkeypatch.setattr(
        viz.rrb,
        "Blueprint",
        lambda *parts, **kwargs: (parts, kwargs),
    )

    blueprint = viz.build_libero_blueprint(
        ("observation.images.image", "observation.images.image2"),
        fps=10.0,
    )

    assert panel_calls == [{"expanded": True}]
    assert panel in blueprint[0]


def test_libero_action_and_state_series_are_independently_addressable() -> None:
    assert viz.LIBERO_ACTION_SERIES_PATHS == tuple(
        f"actions/executed/{name}" for name in viz.ACTION_NAMES
    )
    assert viz.LIBERO_STATE_SERIES_PATHS == tuple(
        f"state/observation/{name}" for name in viz.LIBERO_STATE_NAMES
    )
    assert len(set(viz.LIBERO_ACTION_SERIES_PATHS)) == 7
    assert len(set(viz.LIBERO_STATE_SERIES_PATHS)) == 8


def test_offline_frame_applies_display_transform_to_all_3d_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    viz.log_offline_frame(recorder, **_offline_frame_kwargs())

    current = next(
        entity
        for path, entity, _ in recorder.logs
        if path == "demonstration/trajectory/eef_current"
    )
    np.testing.assert_allclose(np.asarray(current.positions), [[0.0, 0.0, 0.0]])
    skeleton = next(
        entity
        for path, entity, _ in recorder.logs
        if path == "demonstration/robot/skeleton"
    )
    transformed = np.asarray(skeleton.positions[0])
    np.testing.assert_allclose(transformed[0], [0.2, -0.1, -0.3])
    np.testing.assert_allclose(transformed[-1], [0.2, 0.4, -0.3])

    target_commands = next(
        entity
        for path, entity, _ in recorder.logs
        if path == "comparison/commands/gt_first_action_segments"
    )
    np.testing.assert_allclose(
        np.asarray(target_commands.positions),
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    )
    predicted_commands = next(
        entity
        for path, entity, _ in recorder.logs
        if path == "comparison/commands/pred_first_action_segments"
    )
    np.testing.assert_allclose(
        np.asarray(predicted_commands.positions),
        [[[0.0, 0.0, 0.0], [-0.05, 0.05, 0.05]]],
    )
    preview = next(
        entity
        for path, entity, _ in recorder.logs
        if path == "prediction/not_executed/chunk_preview"
    )
    np.testing.assert_allclose(
        np.asarray(preview.positions[0])[:2],
        np.asarray(predicted_commands.positions)[0],
    )


def test_offline_frame_preserves_recorded_image_and_corrects_calibrated_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    image_chw = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 100.0
    kwargs = _offline_frame_kwargs()
    kwargs["images"] = {viz.AGENTVIEW_CAMERA_KEY: image_chw}
    kwargs["agentview_calibration"] = _camera_calibration()
    kwargs["agentview_image_convention"] = "lerobot_180"

    viz.log_offline_frame(recorder, **kwargs)

    recorded = next(
        entity.image
        for path, entity, _ in recorder.logs
        if path == viz.RECORDED_AGENTVIEW_ENTITY_PATH
    )
    corrected = next(
        entity.image
        for path, entity, _ in recorder.logs
        if path == viz.CALIBRATED_AGENTVIEW_ENTITY_PATH
    )
    expected = np.asarray(np.moveaxis(image_chw.numpy(), 0, -1) * 255.0, dtype=np.uint8)
    np.testing.assert_array_equal(recorded, expected)
    np.testing.assert_array_equal(corrected, expected[:, ::-1])
    assert corrected.flags.c_contiguous
    assert not np.shares_memory(recorded, corrected)


def test_physical_agentview_image_handles_both_conventions_and_boundaries() -> None:
    image = np.zeros((2, 5, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    image[1, 4] = [40, 50, 60]

    reflected = viz.physical_agentview_image(image, "lerobot_180")
    unchanged = viz.physical_agentview_image(image, "physical_opencv")

    np.testing.assert_array_equal(reflected[0, 4], image[0, 0])
    np.testing.assert_array_equal(reflected[1, 0], image[1, 4])
    np.testing.assert_array_equal(unchanged, image)
    assert reflected.flags.c_contiguous and unchanged.flags.c_contiguous
    assert not np.shares_memory(reflected, image)
    assert not np.shares_memory(unchanged, image)


def test_offline_frame_rejects_agentview_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entities(monkeypatch)
    kwargs = _offline_frame_kwargs()
    kwargs["images"] = {viz.AGENTVIEW_CAMERA_KEY: torch.zeros(3, 5, 4)}
    kwargs["agentview_calibration"] = _camera_calibration()
    kwargs["agentview_image_convention"] = "lerobot_180"
    with pytest.raises(ValueError, match="HWC shape"):
        viz.log_offline_frame(FakeRecorder(), **kwargs)


def test_offline_frame_rejects_padded_first_action() -> None:
    kwargs = _offline_frame_kwargs()
    kwargs["valid_horizon"] = np.array([False, False])
    with pytest.raises(ValueError, match="first action must be valid"):
        viz.log_offline_frame(FakeRecorder(), **kwargs)


def test_episode_clear_can_omit_first_failed_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    kwargs = _offline_frame_kwargs()
    kwargs["skeleton_pose"] = None
    viz.log_offline_frame(recorder, **kwargs)
    paths = {path for path, _, _ in recorder.logs}
    assert "demonstration/robot" in paths
    assert not any(path.startswith("demonstration/robot/") for path in paths)


def test_stale_skeleton_retains_hidden_inspectable_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    kwargs = _offline_frame_kwargs()
    kwargs["skeleton_stale"] = True
    viz.log_offline_frame(recorder, **kwargs)

    skeleton = next(
        entity
        for path, entity, _ in recorder.logs
        if path == "demonstration/robot/skeleton"
    )
    assert skeleton.kwargs["labels"] == [
        "Demonstration arm - IK reconstructed (stale)"
    ]
    assert skeleton.kwargs["show_labels"] is False


def test_libero_queue_hit_omits_model_query_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    viz.log_libero_step(recorder, **_libero_step_kwargs())
    assert recorder.times == [("global_step", 8)]
    paths = {path for path, _, _ in recorder.logs}
    expected = {
        "rollout/context/episode_index",
        "rollout/context/step_index",
        "rollout/context/model_queried",
        "rollout/context/plan_step_index",
        "rollout/cameras/input/image",
        "rollout/cameras/post_action/agentview",
        "rollout/calibration/agentview_corrected",
        "rollout/robot/skeleton",
        "rollout/trajectory/eef_measured",
        "rollout/command/executed",
        "rollout/response/eef_measured",
        "rollout/response/eef_endpoint",
        "rollout/plan",
        "rollout/reward",
        "rollout/success_so_far",
        "rollout/done",
        "latency/policy_step_ms",
    }
    expected.update(f"actions/executed/{name}" for name in viz.ACTION_NAMES)
    expected.update(
        f"state/observation/{name}" for name in viz.LIBERO_STATE_NAMES
    )
    assert expected <= paths
    assert "latency/model_query_ms" not in paths


def test_libero_geometry_anchors_command_response_queue_and_discarded_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    kwargs = _libero_step_kwargs()
    kwargs.update(
        {
            "observation_state": np.asarray([0.1, 0.2, 0.3, 0, 0, 0, 0, 0]),
            "next_observation_state": np.asarray(
                [0.11, 0.22, 0.33, 0, 0, 0, 0, 0]
            ),
            "display_frame": viz.DisplayFrame(np.zeros(3), np.eye(3)),
            "action": np.asarray([1, 0, 0, 0, 0, 0, 0]),
            "queued_actions": np.asarray(
                [
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                ]
            ),
            "discarded_actions": np.asarray([[0, 0, 1, 0, 0, 0, 0]]),
            "model_queried": True,
            "plan_step_index": 0,
            "model_query_latency_ms": 2.0,
        }
    )

    viz.log_libero_step(recorder, **kwargs)
    entities = {path: entity for path, entity, _ in recorder.logs}

    np.testing.assert_allclose(
        entities["rollout/command/executed"].positions[0],
        [[0.1, 0.2, 0.3], [0.15, 0.2, 0.3]],
    )
    assert entities["rollout/command/executed"].kwargs["radii"] == (
        viz.ROLLOUT_COMMAND_RADIUS
    )
    response = entities["rollout/response/eef_measured"]
    np.testing.assert_allclose(
        response.kwargs["origins"],
        [[0.1, 0.2, 0.3]],
    )
    np.testing.assert_allclose(
        response.kwargs["vectors"],
        [[0.01, 0.02, 0.03]],
    )
    assert response.kwargs["radii"] == viz.ROLLOUT_RESPONSE_RADIUS
    assert response.kwargs["colors"] == [viz.ROLLOUT_RESPONSE_COLOR]
    endpoint = entities["rollout/response/eef_endpoint"]
    np.testing.assert_allclose(endpoint.positions, [[0.11, 0.22, 0.33]])
    assert endpoint.kwargs["radii"] == viz.ROLLOUT_RESPONSE_ENDPOINT_RADIUS
    assert endpoint.kwargs["colors"] == [viz.ROLLOUT_RESPONSE_COLOR]
    assert viz.ROLLOUT_COMMAND_RADIUS < viz.ROLLOUT_RESPONSE_RADIUS
    queue = np.asarray(entities["rollout/plan/queued/preview"].positions[0])
    np.testing.assert_allclose(queue[0], [0.11, 0.22, 0.33])
    np.testing.assert_allclose(queue[-1], [0.11, 0.32, 0.33])
    discarded_dashes = entities["rollout/plan/discarded/preview"].positions
    np.testing.assert_allclose(discarded_dashes[0][0], queue[-1])
    assert "latency/model_query_ms" in entities


def test_libero_temporal_chunk_is_violet_non_queue_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    kwargs = _libero_step_kwargs()
    kwargs.update(
        {
            "new_prediction_chunk": np.ones((3, 7)),
            "model_queried": True,
            "plan_step_index": None,
            "model_query_latency_ms": 2.0,
        }
    )

    viz.log_libero_step(recorder, **kwargs)
    entities = {path: entity for path, entity, _ in recorder.logs}

    temporal = entities["rollout/plan/new_prediction_chunk/preview"]
    assert temporal.kwargs["colors"] == [viz.ROLLOUT_TEMPORAL_CHUNK_COLOR]
    assert "rollout/plan/queued/preview" not in entities
    assert "rollout/plan/discarded/preview" not in entities
    assert entities["rollout/context/plan_step_index"].kwargs["recursive"] is False


def test_libero_trajectory_history_resets_between_episodes() -> None:
    history = viz.LiberoTrajectoryHistory()
    first = history.append(
        episode_index=0,
        step_index=0,
        observation_state=np.asarray([1, 2, 3, 0, 0, 0, 0, 0]),
    )
    second = history.append(
        episode_index=0,
        step_index=1,
        observation_state=np.asarray([4, 5, 6, 0, 0, 0, 0, 0]),
    )
    reset = history.append(
        episode_index=1,
        step_index=0,
        observation_state=np.asarray([7, 8, 9, 0, 0, 0, 0, 0]),
    )

    assert first.shape == (1, 3)
    assert second.shape == (2, 3)
    np.testing.assert_array_equal(reset, [[7, 8, 9]])


def _offline_metrics_payload() -> dict[str, object]:
    return {
        "mode": "offline",
        "aggregate": {
            "num_episodes": 5,
            "num_frames": 859,
            "num_model_queries": 430,
            "batch_size": 2,
            "first_action_mae": 0.0207619,
            "per_dimension_mae": [0.01, 0.02, 0.03, 0.004, 0.005, 0.006, 0.07],
            "chunk_mae_by_horizon": [0.1, 0.2],
            "delta_action_mae": 0.009725,
            "teacher_forced_action_tv": 0.020114,
            "teacher_forced_action_jerk": 0.011465,
            "inference_latency_p50_ms": 4.126,
            "inference_latency_p95_ms": 7.894,
        },
        "metric_definitions": {
            "first_action_mae": "current-action error",
        },
    }


def test_offline_metrics_markdown_uses_ordered_compact_tables() -> None:
    markdown = viz.metrics_markdown(_offline_metrics_payload())

    assert markdown == "\n".join(
        (
            "# Offline evaluation metrics",
            "",
            "| Metric | Value | Meaning |",
            "|---|---:|---|",
            "| Episodes | 5 |  |",
            "| Frames | 859 |  |",
            "| Model queries | 430 |  |",
            "| Batch size | 2 |  |",
            "| First-action MAE | 0.0208 | current-action error |",
            "| Delta-action MAE | 0.00972 |  |",
            "| Action TV | 0.02011 |  |",
            "| Action Jerk | 0.01146 |  |",
            "| Inference latency p50 | 4.13 ms |  |",
            "| Inference latency p95 | 7.89 ms |  |",
            "",
            "## Per-dimension first-action MAE",
            "",
            "| Dimension | MAE |",
            "|---|---:|",
            "| x | 0.0100 |",
            "| y | 0.0200 |",
            "| z | 0.0300 |",
            "| roll | 0.0040 |",
            "| pitch | 0.0050 |",
            "| yaw | 0.0060 |",
            "| gripper | 0.0700 |",
        )
    )
    assert "chunk_mae_by_horizon" not in markdown


@pytest.mark.parametrize(
    ("key", "value", "row"),
    (
        ("num_frames", True, "| Frames | N/A |"),
        ("num_frames", "859", "| Frames | N/A |"),
        ("num_frames", 1.5, "| Frames | N/A |"),
        ("num_frames", -1, "| Frames | N/A |"),
        ("first_action_mae", np.nan, "| First-action MAE | N/A |"),
        ("delta_action_mae", np.inf, "| Delta-action MAE | N/A |"),
        ("teacher_forced_action_tv", -np.inf, "| Action TV | N/A |"),
    ),
)
def test_offline_metrics_markdown_rejects_invalid_scalars(
    key: str,
    value: object,
    row: str,
) -> None:
    metrics = _offline_metrics_payload()
    metrics["aggregate"][key] = value  # type: ignore[index]
    assert row in viz.metrics_markdown(metrics)


@pytest.mark.parametrize(
    "value",
    (
        None,
        [0.0] * 6,
        [0.0] * 6 + [True],
        [0.0] * 6 + ["0.1"],
        [0.0] * 6 + [np.nan],
        "0,0,0,0,0,0,0",
    ),
)
def test_offline_metrics_markdown_rejects_malformed_per_dimension(
    value: object,
) -> None:
    metrics = _offline_metrics_payload()
    metrics["aggregate"]["per_dimension_mae"] = value  # type: ignore[index]
    markdown = viz.metrics_markdown(metrics)
    assert "Per-dimension MAE unavailable" in markdown
    assert "| Dimension | MAE |" not in markdown


def test_non_offline_metrics_markdown_keeps_generic_presentation() -> None:
    markdown = viz.metrics_markdown(
        {
            "mode": "libero",
            "aggregate": {"success_rate": 0.5},
            "metric_definitions": {"success_rate": "successful rollout ratio"},
        }
    )
    assert markdown == (
        "# libero metrics\n\n"
        "- **success_rate**: `0.5` — successful rollout ratio"
    )


def test_recording_configures_file_sink_before_blueprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    blueprint = object()
    monkeypatch.setattr(
        viz,
        "build_offline_blueprint",
        lambda *args, **kwargs: blueprint,
    )
    monkeypatch.setattr(viz.rr, "init", lambda *args, **kwargs: events.append(("init", kwargs)))
    monkeypatch.setattr(viz.rr, "save", lambda path: events.append(("save", path)))
    monkeypatch.setattr(
        viz.rr, "send_blueprint", lambda value: events.append(("blueprint", value))
    )
    monkeypatch.setattr(
        viz,
        "log_offline_static_context",
        lambda recorder, **kwargs: events.append(("static", (recorder, kwargs))),
    )
    output = tmp_path / "prediction.rrd"
    viz.initialize_recording(
        output_path=output,
        mode="offline",
        camera_keys=(),
        fps=10.0,
    )
    assert events == [
        ("init", {"spawn": False}),
        ("save", output),
        (
            "static",
            (
                viz.rr,
                {
                    "agentview_calibration": None,
                    "agentview_image_convention": None,
                },
            ),
        ),
        ("blueprint", blueprint),
    ]


def test_small_rrd_is_valid(tmp_path: Path) -> None:
    output = tmp_path / "small.rrd"
    viz.initialize_recording(
        output_path=output,
        mode="offline",
        camera_keys=(),
        fps=10.0,
        agentview_calibration=_camera_calibration(),
        agentview_image_convention="lerobot_180",
    )
    viz.log_offline_frame(
        viz.rr,
        global_frame=0,
        episode_index=823,
        frame_index=0,
        batch_id=0,
        observation_state=np.zeros(8),
        target_action=np.zeros(7),
        predicted_action=np.zeros(7),
        target_chunk=np.zeros((2, 7)),
        predicted_chunk=np.zeros((2, 7)),
        valid_horizon=np.array([True, True]),
        trajectory_history=viz.OfflineTrajectoryHistory(),
        command_history=viz.OfflineCommandHistory(),
        display_frame=viz.DisplayFrame.libero_y_forward([0.0, 0.0, 0.0]),
        skeleton_pose=_skeleton_pose(),
        skeleton_stale=False,
        clear_skeleton=True,
        model_query_latency_ms=1.0,
        images={viz.AGENTVIEW_CAMERA_KEY: torch.zeros(3, 4, 5)},
        agentview_calibration=_camera_calibration(),
        agentview_image_convention="lerobot_180",
    )
    viz.finish_recording()
    assert output.stat().st_size > 0
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        pytest.skip("Rerun CLI is unavailable")
    result = subprocess.run(
        [rerun_cli, "rrd", "verify", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    printed = subprocess.run(
        [rerun_cli, "rrd", "print", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert printed.returncode == 0, printed.stderr
    assert "/demonstration/cameras/image" in printed.stdout
    assert "/calibration/agentview_corrected" in printed.stdout
    assert "Pinhole:" in printed.stdout
    assert "Transform3D:" in printed.stdout


def test_small_libero_rrd_contains_transition_and_plan_views(tmp_path: Path) -> None:
    output = tmp_path / "libero.rrd"
    calibration = _camera_calibration()
    viz.initialize_recording(
        output_path=output,
        mode="libero",
        camera_keys=(viz.AGENTVIEW_CAMERA_KEY,),
        fps=10.0,
        agentview_calibration=calibration,
        agentview_image_convention="lerobot_180",
    )
    kwargs = _libero_step_kwargs()
    kwargs.update(
        {
            "global_step": 0,
            "step_index": 0,
            "queued_actions": np.ones((2, 7)),
            "discarded_actions": np.ones((2, 7)),
            "model_queried": True,
            "plan_step_index": 0,
            "model_query_latency_ms": 2.0,
        }
    )
    viz.log_libero_step(viz.rr, **kwargs)
    viz.finish_recording()

    assert output.stat().st_size > 0
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        pytest.skip("Rerun CLI is unavailable")
    verified = subprocess.run(
        [rerun_cli, "rrd", "verify", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    printed = subprocess.run(
        [rerun_cli, "rrd", "print", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert printed.returncode == 0, printed.stderr
    assert "/rollout/cameras/input/image" in printed.stdout
    assert "/rollout/cameras/post_action/agentview" in printed.stdout
    assert "/rollout/robot/skeleton" in printed.stdout
    assert "/rollout/response/eef_measured" in printed.stdout
    assert "/rollout/plan/queued/preview" in printed.stdout
    assert "/rollout/plan/discarded/preview" in printed.stdout
