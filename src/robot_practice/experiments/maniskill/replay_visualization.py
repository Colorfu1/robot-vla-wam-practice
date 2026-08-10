"""Visualize stored ManiSkill replay episodes without stepping the simulator.

The replay HDF5 already contains RGB observations, policy state, actions, native
termination signals, and (for the M0 PickCube replay) optional privileged scene
state.  This module turns selected episodes into one Rerun recording each.  It
does not call ManiSkill, replay actions, or load a policy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from robot_practice.evaluation.common import launch_rerun_viewer, write_json_atomic
from robot_practice.rerun_launcher import OFFLINE_APPLICATION_ID

from .convert_demos import (
    ACTION_CHANNEL_NAMES_BY_DIM,
    IMAGE_SHAPE,
    STATE_SHAPE,
    episode_success,
)
from .panda_kinematics import PANDA_SKELETON_LABELS, PandaKinematics, PandaSkeletonPose

TASK_ID = "PickCube-v1"
DEFAULT_FPS = 20.0
TIMELINE = "episode_step"
CAMERA_PATH = "episode/camera/base_camera"
SCENE_ROOT = "episode/scene"
SIGNALS_ROOT = "episode/signals"
EVENTS_ROOT = "episode/events"
PANDA_ROOT = f"{SCENE_ROOT}/panda"
PANDA_METRICS_ROOT = f"{PANDA_ROOT}/ik"

ACTION_COLORS = (
    (78, 121, 167),
    (242, 142, 43),
    (89, 161, 79),
    (225, 87, 89),
    (176, 122, 161),
    (156, 117, 95),
    (47, 158, 170),
    (118, 183, 178),
)
STATE_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
)
TCP_COLOR = (0, 190, 210)
CUBE_COLOR = (230, 80, 80)
GOAL_COLOR = (255, 190, 0)
PANDA_COLOR = (70, 130, 255)
PANDA_FINGER_COLOR = (255, 145, 60)
# Stored replay is an offline/open-loop artifact.  Keep this ID aligned with the
# app-aware launcher so `robot-rerun episode_....rrd` selects Robot Rerun Offline.
VIEWER_KIND = "offline"


@dataclass(frozen=True)
class ReplayEpisode:
    """One selected replay episode with its observation/action timelines."""

    episode_id: int
    seed: int | None
    metadata_record: Mapping[str, Any]
    images: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    success: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    is_grasped: np.ndarray | None
    tcp_pose: np.ndarray | None
    cube_position: np.ndarray | None
    goal_position: np.ndarray | None

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_CHANNEL_NAMES_BY_DIM[self.action_dim]

    @property
    def action_steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def first_success_step(self) -> int | None:
        indices = np.flatnonzero(self.success)
        return None if indices.size == 0 else int(indices[0]) + 1

    @property
    def success_once(self) -> bool:
        return bool(np.any(self.success))

    @property
    def success_at_end(self) -> bool:
        return bool(self.success[-1]) if self.success.size else False

    @property
    def result_label(self) -> str:
        """Stable artifact label based on the terminal success definition."""

        return "success" if self.success_at_end else "failure"


def episode_artifact_stem(episode: ReplayEpisode) -> str:
    """Return the status-bearing stem shared by RRD, MP4, and manifests."""

    return f"episode_{episode.episode_id:04d}_{episode.result_label}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return value


def _metadata_path(replay_h5: Path) -> Path:
    path = replay_h5.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"Replay metadata sidecar is missing: {path}")
    return path


def _metadata_records(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Replay metadata must contain an episodes list")
    records: dict[int, dict[str, Any]] = {}
    for value in episodes:
        if not isinstance(value, Mapping):
            raise TypeError("Each replay metadata episode must be an object")
        episode_id = int(value["episode_id"])
        if episode_id in records:
            raise ValueError(f"Duplicate replay metadata episode_id={episode_id}")
        records[episode_id] = dict(value)
    return records


def _episode_seed(record: Mapping[str, Any] | None) -> int | None:
    if record is None:
        return None
    value: Any = record.get("episode_seed")
    if value is None:
        reset_kwargs = record.get("reset_kwargs")
        if isinstance(reset_kwargs, Mapping):
            value = reset_kwargs.get("seed")
    if isinstance(value, list):
        if len(value) != 1:
            return None
        value = value[0]
    return None if value is None else int(value)


def _episode_id_from_name(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected HDF5 group {name!r}; expected traj_<id>")
    try:
        return int(name.removeprefix("traj_"))
    except ValueError as exc:
        raise ValueError(f"Unexpected HDF5 group {name!r}") from exc


def available_episode_ids(replay_h5: str | Path) -> tuple[int, ...]:
    """Return numeric trajectory IDs in a replay HDF5, in numeric order."""

    path = Path(replay_h5).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Replay HDF5 does not exist: {path}")
    with h5py.File(path, "r") as handle:
        ids = [
            _episode_id_from_name(name)
            for name, node in handle.items()
            if isinstance(node, h5py.Group) and name.startswith("traj_")
        ]
    if not ids:
        raise ValueError(f"Replay HDF5 contains no traj_<id> groups: {path}")
    return tuple(sorted(ids))


def select_episode_ids(
    available: Sequence[int],
    requested: Sequence[int] | None,
    *,
    max_episodes: int,
) -> tuple[int, ...]:
    """Resolve explicit ``--episode`` values or take a leading episode prefix."""

    available_set = {int(value) for value in available}
    if not available_set:
        raise ValueError("No replay episodes are available")
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if requested:
        selected = tuple(int(value) for value in requested)
        if len(set(selected)) != len(selected):
            raise ValueError("Requested replay episode IDs must be unique")
        missing = [value for value in selected if value not in available_set]
        if missing:
            raise ValueError(f"Requested replay episodes are missing: {missing}")
        return selected
    return tuple(sorted(available_set))[:max_episodes]


def _dataset(node: h5py.Group, path: Sequence[str]) -> h5py.Dataset | None:
    current: Any = node
    for component in path:
        if not isinstance(current, h5py.Group) or component not in current:
            return None
        current = current[component]
    return current if isinstance(current, h5py.Dataset) else None


def _required_dataset(node: h5py.Group, path: Sequence[str]) -> h5py.Dataset:
    value = _dataset(node, path)
    if value is None:
        raise KeyError(f"Replay episode is missing {'/'.join(path)}")
    return value


def _timeline_array(
    node: h5py.Group,
    path: Sequence[str],
    *,
    frame_count: int,
    shape_tail: tuple[int, ...],
    dtype: Any = np.float32,
) -> np.ndarray:
    dataset = _required_dataset(node, path)
    values = np.asarray(dataset[: frame_count + 1], dtype=dtype)
    expected = (frame_count + 1, *shape_tail)
    if values.shape != expected:
        raise ValueError(f"{'/'.join(path)} must have shape {expected}, got {values.shape}")
    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        raise ValueError(f"{'/'.join(path)} contains non-finite values")
    return values


def _optional_timeline_array(
    node: h5py.Group,
    path: Sequence[str],
    *,
    frame_count: int,
    shape_tail: tuple[int, ...],
) -> np.ndarray | None:
    dataset = _dataset(node, path)
    if dataset is None:
        return None
    values = np.asarray(dataset[: frame_count + 1], dtype=np.float32)
    expected = (frame_count + 1, *shape_tail)
    if values.shape != expected:
        raise ValueError(f"{'/'.join(path)} must have shape {expected}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{'/'.join(path)} contains non-finite values")
    return values


def _optional_position_series(
    node: h5py.Group,
    path: Sequence[str],
    *,
    frame_count: int,
) -> np.ndarray | None:
    dataset = _dataset(node, path)
    if dataset is None:
        return None
    values = np.asarray(dataset[: frame_count + 1], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != frame_count + 1 or values.shape[1] < 3:
        raise ValueError(
            f"{'/'.join(path)} must have shape [{frame_count + 1},>=3], got {values.shape}"
        )
    positions = values[:, :3]
    if not np.isfinite(positions).all():
        raise ValueError(f"{'/'.join(path)} contains non-finite values")
    return positions


def _optional_bool_series(
    node: h5py.Group,
    path: Sequence[str],
    *,
    frame_count: int,
) -> np.ndarray | None:
    dataset = _dataset(node, path)
    if dataset is None:
        return None
    values = np.asarray(dataset[: frame_count + 1], dtype=bool).reshape(-1)
    expected = frame_count + 1
    if values.shape != (expected,):
        raise ValueError(f"{'/'.join(path)} must have shape [{expected}], got {values.shape}")
    return values


def _step_bool_series(
    node: h5py.Group,
    path: Sequence[str],
    *,
    frame_count: int,
    fallback: bool = False,
) -> np.ndarray:
    dataset = _dataset(node, path)
    if dataset is None:
        return np.full(frame_count, fallback, dtype=bool)
    values = np.asarray(dataset[:frame_count], dtype=bool).reshape(-1)
    if values.shape != (frame_count,):
        raise ValueError(f"{'/'.join(path)} must have shape [{frame_count}], got {values.shape}")
    return values


def load_replay_episode(
    episode: h5py.Group,
    *,
    episode_id: int,
    metadata_record: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
) -> ReplayEpisode:
    """Load one replay episode from an open HDF5 group without simulation."""

    actions_dataset = _required_dataset(episode, ("actions",))
    raw_actions = np.asarray(actions_dataset[:], dtype=np.float32)
    if raw_actions.ndim != 2 or raw_actions.shape[1] not in ACTION_CHANNEL_NAMES_BY_DIM:
        raise ValueError(
            f"Replay actions must have shape [T,4], [T,7] or [T,8], got {raw_actions.shape}"
        )
    if not np.isfinite(raw_actions).all():
        raise ValueError(f"Replay actions contain non-finite values in traj_{episode_id}")
    action_steps = int(raw_actions.shape[0])
    if action_steps <= 0:
        raise ValueError(f"Replay episode traj_{episode_id} has no actions")
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        action_steps = min(action_steps, max_steps)
    record = {} if metadata_record is None else dict(metadata_record)

    images = np.asarray(
        _required_dataset(episode, ("obs", "sensor_data", "base_camera", "rgb"))[
            : action_steps + 1
        ],
        dtype=np.uint8,
    )
    expected_images = (action_steps + 1, *IMAGE_SHAPE)
    if images.shape != expected_images:
        raise ValueError(f"Replay RGB must have shape {expected_images}, got {images.shape}")

    qpos = _timeline_array(
        episode,
        ("obs", "agent", "qpos"),
        frame_count=action_steps,
        shape_tail=(9,),
    )
    qvel = _timeline_array(
        episode,
        ("obs", "agent", "qvel"),
        frame_count=action_steps,
        shape_tail=(9,),
    )
    goal = _timeline_array(
        episode,
        ("obs", "extra", "goal_pos"),
        frame_count=action_steps,
        shape_tail=(3,),
    )
    state = np.concatenate((qpos, qvel, goal), axis=1).astype(np.float32, copy=False)
    if state.shape != (action_steps + 1, STATE_SHAPE[0]) or not np.isfinite(state).all():
        raise ValueError(f"Replay state must have shape {(action_steps + 1, STATE_SHAPE[0])}")

    success_fallback = bool(record.get("success", False))
    success_dataset = _dataset(episode, ("success",))
    if success_dataset is None:
        success = np.full(action_steps, success_fallback, dtype=bool)
    else:
        full_success = np.asarray(success_dataset[:], dtype=bool).reshape(-1)
        if full_success.shape[0] < action_steps:
            raise ValueError(
                f"Replay success must have at least {action_steps} values, got {full_success.shape}"
            )
        success = full_success[:action_steps]
    # Reuse the converter's native-label rule for metadata-only fallback validation.
    if success_dataset is not None:
        episode_success(record, episode)
    actions = raw_actions[:action_steps]
    return ReplayEpisode(
        episode_id=int(episode_id),
        seed=_episode_seed(record),
        metadata_record=record,
        images=images,
        state=state,
        actions=actions,
        success=success,
        terminated=_step_bool_series(episode, ("terminated",), frame_count=action_steps),
        truncated=_step_bool_series(episode, ("truncated",), frame_count=action_steps),
        is_grasped=_optional_bool_series(
            episode,
            ("obs", "extra", "is_grasped"),
            frame_count=action_steps,
        ),
        tcp_pose=_optional_timeline_array(
            episode,
            ("obs", "extra", "tcp_pose"),
            frame_count=action_steps,
            shape_tail=(7,),
        ),
        cube_position=_optional_position_series(
            episode,
            ("env_states", "actors", "cube"),
            frame_count=action_steps,
        ),
        goal_position=_optional_position_series(
            episode,
            ("env_states", "actors", "goal_site"),
            frame_count=action_steps,
        ),
    )


def _metadata_and_records(replay_h5: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    metadata = _read_json(_metadata_path(replay_h5))
    env_info = metadata.get("env_info")
    if not isinstance(env_info, Mapping) or env_info.get("env_id") != TASK_ID:
        raise ValueError(f"Replay metadata must describe {TASK_ID}")
    return metadata, _metadata_records(metadata)


def _action_names(action_dim: int) -> tuple[str, ...]:
    try:
        return ACTION_CHANNEL_NAMES_BY_DIM[action_dim]
    except KeyError as exc:
        raise ValueError(f"Unsupported ManiSkill action dimension: {action_dim}") from exc


def build_blueprint(*, fps: float, include_privileged_overlay: bool) -> rrb.Blueprint:
    """Build the ManiSkill replay layout used by every episode recording."""

    scene_contents = [
        f"{SCENE_ROOT}/tcp_path",
        f"{SCENE_ROOT}/tcp_position",
        f"{PANDA_ROOT}/skeleton",
        f"{PANDA_ROOT}/joints",
        f"{PANDA_ROOT}/gripper",
    ]
    if include_privileged_overlay:
        scene_contents.extend(
            (
                f"{SCENE_ROOT}/cube_path",
                f"{SCENE_ROOT}/cube_position",
                f"{SCENE_ROOT}/goal_path",
                f"{SCENE_ROOT}/goal_position",
            )
        )
    signal_tabs = rrb.Tabs(
        rrb.TimeSeriesView(
            origin=f"{SIGNALS_ROOT}/action",
            contents=[f"{SIGNALS_ROOT}/action/**"],
            name="7D action",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TimeSeriesView(
            origin=f"{SIGNALS_ROOT}/state",
            contents=[f"{SIGNALS_ROOT}/state/**"],
            name="21D policy state",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TimeSeriesView(
            origin=EVENTS_ROOT,
            contents=[f"{EVENTS_ROOT}/**"],
            name="Success and terminal signals",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TimeSeriesView(
            origin=PANDA_METRICS_ROOT,
            contents=[f"{PANDA_METRICS_ROOT}/**"],
            name="Panda IK diagnostics",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TextDocumentView(
            origin="episode/context",
            contents=["episode/context"],
            name="Replay metadata and semantics",
        ),
        name="Replay signals",
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=CAMERA_PATH,
                    contents=[CAMERA_PATH],
                    name="Recorded base camera",
                ),
                rrb.Spatial3DView(
                    origin=SCENE_ROOT,
                    contents=scene_contents,
                    name=(
                        "Privileged scene diagnostics"
                        if include_privileged_overlay
                        else "TCP diagnostic path"
                    ),
                    line_grid=True,
                ),
                column_shares=[0.6, 0.4],
                name="Replay frame",
            ),
            signal_tabs,
            row_shares=[0.64, 0.36],
        ),
        rrb.TimePanel(timeline=TIMELINE, fps=fps, expanded=True),
        collapse_panels=False,
    )


def _log_series_definitions(
    recording: Any,
    *,
    root: str,
    names: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
) -> None:
    for index, name in enumerate(names):
        recording.log(
            f"{root}/{name}",
            rr.SeriesLines(
                colors=[colors[index % len(colors)]],
                widths=[1.5],
                names=[name],
            ),
            static=True,
        )


def _log_static_context(
    recording: Any,
    episode: ReplayEpisode,
    *,
    control_mode: str | None,
    include_privileged_overlay: bool,
    recording_label: str,
) -> None:
    action_names = _action_names(episode.action_dim)
    state_names = tuple(
        [f"qpos_{index:02d}" for index in range(9)]
        + [f"qvel_{index:02d}" for index in range(9)]
        + ["goal_x", "goal_y", "goal_z"]
    )
    _log_series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/action",
        names=action_names,
        colors=ACTION_COLORS,
    )
    _log_series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/state",
        names=state_names,
        colors=STATE_COLORS,
    )
    for name, color in (
        ("success", (0, 170, 80)),
        ("is_grasped", (170, 80, 220)),
        ("terminated", (220, 80, 80)),
        ("truncated", (220, 150, 40)),
    ):
        recording.log(
            f"{EVENTS_ROOT}/{name}",
            rr.SeriesLines(colors=[color], widths=[2.0], names=[name]),
            static=True,
        )
    for name, color in (
        ("ik_converged", (50, 180, 255)),
        ("position_residual_m", (255, 120, 40)),
        ("rotation_residual_rad", (180, 90, 220)),
        ("qpos_fallback", (150, 150, 150)),
    ):
        recording.log(
            f"{PANDA_METRICS_ROOT}/{name}",
            rr.SeriesLines(colors=[color], widths=[1.5], names=[name]),
            static=True,
        )
    recording.log(f"{SCENE_ROOT}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    notes = (
        f"# ManiSkill {recording_label} episode {episode.episode_id}\n\n"
        f"- **Task:** {TASK_ID}\n"
        f"- **Seed label:** {episode.seed}\n"
        f"- **Control mode:** {control_mode or 'unknown'}\n"
        f"- **Action:** {episode.action_dim}D ({', '.join(action_names)})\n"
        f"- **Policy state:** 21D qpos + qvel + goal_pos\n"
        f"- **Action steps:** {episode.action_steps}\n"
        f"- **Success once:** {episode.success_once}\n"
        f"- **Success at end:** {episode.success_at_end}\n"
        f"- **Result label:** {episode.result_label}\n\n"
        "The camera panel is the recorded RGB observation. The goal is not drawn into that image.\n\n"
        "The blue Panda skeleton is visualization-only: each frame solves numerical IK from "
        "the recorded world-frame `extra.tcp_pose`, seeded by the recorded arm qpos. It is "
        "not an additional policy input or a simulator-rendered robot mesh.\n\n"
    )
    if include_privileged_overlay:
        notes += (
            "The 3D cube/goal overlay is privileged replay state for diagnosis only; it is not "
            "the policy input.\n"
        )
    else:
        notes += (
            "Use `--include-privileged-overlay` to show the replayed cube and goal in the 3D "
            "diagnostic view.\n"
        )
    recording.log(
        "episode/context",
        rr.TextDocument(notes, media_type="text/markdown"),
        static=True,
    )
    if episode.tcp_pose is not None:
        tcp_path = episode.tcp_pose[:, :3]
        recording.log(
            f"{SCENE_ROOT}/tcp_path",
            rr.LineStrips3D(strips=[tcp_path], colors=[TCP_COLOR], radii=0.004),
            static=True,
        )
    if include_privileged_overlay and episode.cube_position is not None:
        recording.log(
            f"{SCENE_ROOT}/cube_path",
            rr.LineStrips3D(
                strips=[episode.cube_position],
                colors=[CUBE_COLOR],
                radii=0.004,
            ),
            static=True,
        )
    if include_privileged_overlay and episode.goal_position is not None:
        recording.log(
            f"{SCENE_ROOT}/goal_path",
            rr.LineStrips3D(
                strips=[episode.goal_position],
                colors=[GOAL_COLOR],
                radii=0.003,
            ),
            static=True,
        )


def _panda_pose_for_frame(
    episode: ReplayEpisode,
    step: int,
    kinematics: PandaKinematics,
) -> PandaSkeletonPose:
    arm_qpos = episode.state[step, :7]
    gripper_qpos = episode.state[step, 7:9]
    if episode.tcp_pose is None:
        return kinematics.from_qpos(arm_qpos, gripper_qpos)
    return kinematics.solve(
        episode.tcp_pose[step],
        seed_qpos=arm_qpos,
        gripper_qpos=gripper_qpos,
    )


def _log_panda_frame(
    recording: Any,
    pose: PandaSkeletonPose,
) -> None:
    recording.log(
        f"{PANDA_ROOT}/skeleton",
        rr.LineStrips3D(
            [pose.skeleton_points],
            colors=[PANDA_COLOR],
            radii=0.009,
            labels=["Panda arm skeleton (IK)"],
            show_labels=False,
        ),
    )
    recording.log(
        f"{PANDA_ROOT}/joints",
        rr.Points3D(
            pose.skeleton_points,
            colors=[PANDA_COLOR],
            radii=0.014,
            labels=list(PANDA_SKELETON_LABELS),
            show_labels=False,
        ),
    )
    recording.log(
        f"{PANDA_ROOT}/gripper",
        rr.LineStrips3D(
            [pose.left_finger_points, pose.right_finger_points],
            colors=[PANDA_FINGER_COLOR, PANDA_FINGER_COLOR],
            radii=0.008,
            labels=["left finger", "right finger"],
            show_labels=False,
        ),
    )
    recording.log(
        f"{PANDA_METRICS_ROOT}/ik_converged",
        rr.Scalars(int(pose.ik_converged)),
    )
    recording.log(
        f"{PANDA_METRICS_ROOT}/position_residual_m",
        rr.Scalars(pose.position_residual_m),
    )
    recording.log(
        f"{PANDA_METRICS_ROOT}/rotation_residual_rad",
        rr.Scalars(pose.rotation_residual_rad),
    )
    recording.log(
        f"{PANDA_METRICS_ROOT}/qpos_fallback",
        rr.Scalars(int(pose.used_qpos_fallback)),
    )


def _log_frame(
    recording: Any,
    episode: ReplayEpisode,
    step: int,
    kinematics: PandaKinematics,
) -> None:
    recording.set_time(TIMELINE, sequence=step)
    recording.log(CAMERA_PATH, rr.Image(episode.images[step]).compress())
    for index, value in enumerate(episode.state[step]):
        if index < 9:
            name = f"qpos_{index:02d}"
        elif index < 18:
            name = f"qvel_{index - 9:02d}"
        else:
            name = ("goal_x", "goal_y", "goal_z")[index - 18]
        recording.log(f"{SIGNALS_ROOT}/state/{name}", rr.Scalars(float(value)))
    if step < episode.action_steps:
        for index, value in enumerate(episode.actions[step]):
            recording.log(
                f"{SIGNALS_ROOT}/action/{episode.action_names[index]}",
                rr.Scalars(float(value)),
            )
    success = bool(episode.success[step - 1]) if step > 0 else False
    recording.log(f"{EVENTS_ROOT}/success", rr.Scalars(int(success)))
    if step > 0:
        action_index = step - 1
        recording.log(
            f"{EVENTS_ROOT}/terminated",
            rr.Scalars(int(episode.terminated[action_index])),
        )
        recording.log(
            f"{EVENTS_ROOT}/truncated",
            rr.Scalars(int(episode.truncated[action_index])),
        )
    if episode.is_grasped is not None:
        recording.log(
            f"{EVENTS_ROOT}/is_grasped",
            rr.Scalars(int(episode.is_grasped[step])),
        )
    if episode.tcp_pose is not None:
        recording.log(
            f"{SCENE_ROOT}/tcp_position",
            rr.Points3D(
                [episode.tcp_pose[step, :3]],
                colors=[TCP_COLOR],
                radii=0.012,
                labels=["tcp"],
                show_labels=False,
            ),
        )
    if episode.cube_position is not None:
        recording.log(
            f"{SCENE_ROOT}/cube_position",
            rr.Points3D(
                [episode.cube_position[step]],
                colors=[CUBE_COLOR],
                radii=0.014,
                labels=["cube"],
                show_labels=False,
            ),
        )
    if episode.goal_position is not None:
        recording.log(
            f"{SCENE_ROOT}/goal_position",
            rr.Points3D(
                [episode.goal_position[step]],
                colors=[GOAL_COLOR],
                radii=0.014,
                labels=["goal"],
                show_labels=False,
            ),
        )
    _log_panda_frame(recording, _panda_pose_for_frame(episode, step, kinematics))


def write_episode_recording(
    episode: ReplayEpisode,
    *,
    output_path: Path,
    fps: float,
    control_mode: str | None,
    include_privileged_overlay: bool,
    application_id: str = OFFLINE_APPLICATION_ID,
    viewer_kind: str = VIEWER_KIND,
    recording_label: str = "replay",
) -> Path:
    """Write one complete Rerun recording and return its final path."""

    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing recording: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    if not application_id:
        raise ValueError("Rerun application_id must not be empty")
    recording = rr.RecordingStream(
        application_id,
        recording_id=f"episode_{episode.episode_id:04d}_{episode.result_label}",
    )
    kinematics = PandaKinematics()
    try:
        recording.save(temporary)
        recording.send_recording_name(
            f"ManiSkill PickCube {recording_label} | episode "
            f"{episode.episode_id:04d} | {episode.result_label}"
        )
        rr.send_blueprint(
            build_blueprint(
                fps=fps,
                include_privileged_overlay=include_privileged_overlay,
            ),
            make_active=False,
            make_default=True,
            recording=recording,
        )
        _log_static_context(
            recording,
            episode,
            control_mode=control_mode,
            include_privileged_overlay=include_privileged_overlay,
            recording_label=recording_label,
        )
        for step in range(episode.action_steps + 1):
            _log_frame(recording, episode, step, kinematics)
        recording.flush()
    finally:
        recording.disconnect()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Rerun recording was not written: {temporary}")
    temporary.replace(output_path)
    return output_path


def write_episode_video(episode: ReplayEpisode, *, output_path: Path, fps: float) -> Path:
    """Write the recorded RGB camera stream as a simple MP4 using ffmpeg."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--write-video requires ffmpeg on PATH")
    height, width, channels = episode.images.shape[1:]
    if channels != 3:
        raise ValueError(f"Expected RGB frames, got {episode.images.shape}")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        input=np.ascontiguousarray(episode.images).tobytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {detail}")
    return output_path


def _episode_summary(
    episode: ReplayEpisode,
    *,
    recording_path: Path,
    video_path: Path | None,
    application_id: str = OFFLINE_APPLICATION_ID,
    viewer_kind: str = VIEWER_KIND,
) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "action_dim": episode.action_dim,
        "action_names": list(episode.action_names),
        "action_steps": episode.action_steps,
        "frame_count": int(episode.images.shape[0]),
        "success_once": episode.success_once,
        "success_at_end": episode.success_at_end,
        "result": episode.result_label,
        "first_success_step": episode.first_success_step,
        "has_is_grasped": episode.is_grasped is not None,
        "has_tcp_pose": episode.tcp_pose is not None,
        "has_privileged_scene_state": (
            episode.cube_position is not None and episode.goal_position is not None
        ),
        "viewer_kind": viewer_kind,
        "application_id": application_id,
        "recording": recording_path.as_posix(),
        "video": None if video_path is None else video_path.as_posix(),
    }


def write_replay_visualization(
    *,
    replay_h5: str | Path,
    output_dir: str | Path,
    episode_ids: Sequence[int] | None = None,
    max_episodes: int = 1,
    max_steps: int | None = None,
    fps: float = DEFAULT_FPS,
    include_privileged_overlay: bool = False,
    write_video: bool = False,
    open_viewer: bool = False,
) -> dict[str, Any]:
    """Create Rerun/MP4 visualizations for selected stored replay episodes."""

    replay_path = Path(replay_h5).expanduser().resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(f"Replay HDF5 does not exist: {replay_path}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata, records = _metadata_and_records(replay_path)
    available = available_episode_ids(replay_path)
    selected = select_episode_ids(available, episode_ids, max_episodes=max_episodes)
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {}) if isinstance(env_info, Mapping) else {}
    control_mode = env_kwargs.get("control_mode") if isinstance(env_kwargs, Mapping) else None

    summaries: list[dict[str, Any]] = []
    with h5py.File(replay_path, "r") as handle:
        for episode_id in selected:
            episode_name = f"traj_{episode_id}"
            loaded = load_replay_episode(
                handle[episode_name],
                episode_id=episode_id,
                metadata_record=records.get(episode_id),
                max_steps=max_steps,
            )
            artifact_stem = episode_artifact_stem(loaded)
            recording_path = output / f"{artifact_stem}.rrd"
            write_episode_recording(
                loaded,
                output_path=recording_path,
                fps=fps,
                control_mode=control_mode if isinstance(control_mode, str) else None,
                include_privileged_overlay=include_privileged_overlay,
            )
            video_path = None
            if write_video:
                video_path = write_episode_video(
                    loaded,
                    output_path=output / f"{artifact_stem}.mp4",
                    fps=fps,
                )
            summaries.append(
                _episode_summary(
                    loaded,
                    recording_path=recording_path,
                    video_path=video_path,
                    application_id=OFFLINE_APPLICATION_ID,
                    viewer_kind=VIEWER_KIND,
                )
            )

    manifest = {
        "schema_version": 1,
        "task": TASK_ID,
        "replay_h5": replay_path.as_posix(),
        "control_mode": control_mode,
        "fps": fps,
        "selected_episode_ids": list(selected),
        "include_privileged_overlay": include_privileged_overlay,
        "write_video": write_video,
        "viewer": {
            "kind": VIEWER_KIND,
            "application_id": OFFLINE_APPLICATION_ID,
            "launcher": "robot-rerun",
            "reason": (
                "This is a stored replay/open-loop artifact; the closed-loop app is reserved "
                "for simulator rollout recordings."
            ),
        },
        "visualization_semantics": {
            "camera": "Recorded RGB observation only",
            "policy_state": "agent.qpos[0:9] + agent.qvel[0:9] + extra.goal_pos[0:3]",
            "privileged_overlay": (
                "Optional cube/goal/tcp replay state for diagnosis; not an additional policy input"
            ),
        },
        "episodes": summaries,
    }
    manifest_path = output / "visualization_manifest.json"
    write_json_atomic(manifest_path, manifest)
    if open_viewer:
        launch_rerun_viewer(Path(summaries[0]["recording"]), enabled=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Rerun/MP4 visualizations from stored ManiSkill replay HDF5 episodes."
    )
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episode",
        dest="episode_ids",
        type=int,
        action="append",
        help="Episode ID to visualize; repeat the option for multiple IDs.",
    )
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional action-step cap per episode; the final observation is retained.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--include-privileged-overlay",
        action="store_true",
        help="Show replayed cube/goal positions in the diagnostic 3D view.",
    )
    parser.add_argument(
        "--write-video",
        action="store_true",
        help="Also write the recorded camera RGB stream as episode_<id>.mp4.",
    )
    parser.add_argument(
        "--open-viewer",
        action="store_true",
        help=(
            "Open the first generated .rrd through robot-rerun, which routes it to "
            "the local Robot Rerun Offline app."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = write_replay_visualization(
        replay_h5=arguments.replay_h5,
        output_dir=arguments.output_dir,
        episode_ids=arguments.episode_ids,
        max_episodes=arguments.max_episodes,
        max_steps=arguments.max_steps,
        fps=arguments.fps,
        include_privileged_overlay=arguments.include_privileged_overlay,
        write_video=arguments.write_video,
        open_viewer=arguments.open_viewer,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
