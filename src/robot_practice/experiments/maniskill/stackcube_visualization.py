"""Rerun visualization for frozen StackCube training demonstrations.

The viewer follows the existing LIBERO/PushT convention: one Rerun recording
per case, a synchronized timeline, camera panels, selectable action/state
plots, and a diagnostic geometry panel.  The geometry overlay is read from the
stored D_core scene state and is never used as a policy observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image, ImageDraw, ImageFont

from robot_practice.evaluation.common import practice_root, write_json_atomic
from robot_practice.experiments.maniskill.convert_demos import (
    ACTION_CHANNEL_NAMES_BY_DIM,
)
from robot_practice.experiments.maniskill.panda_kinematics import (
    PANDA_SKELETON_LABELS,
    PandaKinematics,
)
from robot_practice.rerun_launcher import OFFLINE_APPLICATION_ID

TASK_ID = "StackCube-v1"
DEFAULT_D_CORE_H5 = Path(
    "outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5"
)
DEFAULT_D_CORE_JSON = Path(
    "outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.json"
)
DEFAULT_SPLIT_MANIFEST = Path(
    "outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/split_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10"
)
DEFAULT_SELECTION_SEED = 20260813
DEFAULT_FPS = 20.0
TIMELINE = "episode_step"
BASE_CAMERA_PATH = "episode/camera/base_camera"
HAND_CAMERA_PATH = "episode/camera/hand_camera"
SCENE_ROOT = "episode/scene"
SIGNALS_ROOT = "episode/signals"
EVENTS_ROOT = "episode/events"
PANDA_ROOT = f"{SCENE_ROOT}/panda"
CONTEXT_PATH = "episode/context"

ACTION_COLORS = (
    (78, 121, 167),
    (242, 142, 43),
    (89, 161, 79),
    (225, 87, 89),
    (176, 122, 161),
    (156, 117, 95),
    (47, 158, 170),
)
STATE_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
)
TCP_COLOR = (0, 190, 210)
CUBE_A_COLOR = (220, 70, 70)
CUBE_B_COLOR = (70, 180, 100)
PANDA_COLOR = (70, 130, 255)
PANDA_FINGER_COLOR = (255, 145, 60)
EVENT_COLORS = (
    (220, 70, 70),
    (70, 180, 100),
    (0, 190, 210),
    (230, 150, 40),
    (120, 120, 120),
)
VIEWER_KIND = "offline"


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _portable(path: str | Path) -> str:
    value = Path(path).expanduser().resolve()
    root = practice_root().resolve()
    try:
        return value.relative_to(root).as_posix()
    except ValueError:
        return value.as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    target = _rooted(path)
    with target.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {target}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _rooted(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class StackCubeEpisode:
    """One frozen D_core episode aligned on observation frames and actions."""

    episode_id: int
    split: str
    route: str
    source_episode_id: int
    success: bool
    first_success_step: int | None
    images_base: np.ndarray
    images_hand: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray
    cube_a: np.ndarray
    cube_b: np.ndarray

    @property
    def action_steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.images_base.shape[0])

    @property
    def action_names(self) -> tuple[str, ...]:
        return ACTION_CHANNEL_NAMES_BY_DIM[7]

    @property
    def result_label(self) -> str:
        return "success" if self.success else "failure"


def _episode_id_from_group(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected D_core group {name!r}")
    return int(name.removeprefix("traj_"))


def _require_array(group: h5py.Group, path: Sequence[str], *, dtype: Any) -> np.ndarray:
    current: Any = group
    for component in path:
        if not isinstance(current, h5py.Group) or component not in current:
            raise KeyError(f"D_core episode is missing {'/'.join(path)}")
        current = current[component]
    if not isinstance(current, h5py.Dataset):
        raise TypeError(f"D_core path {'/'.join(path)} is not a dataset")
    value = np.asarray(current[:], dtype=dtype)
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"D_core path {'/'.join(path)} contains non-finite values")
    return value


def available_episode_ids(d_core_h5: str | Path) -> tuple[int, ...]:
    path = _rooted(d_core_h5)
    if not path.is_file():
        raise FileNotFoundError(f"D_core HDF5 does not exist: {_portable(path)}")
    with h5py.File(path, "r") as handle:
        ids = [
            _episode_id_from_group(name)
            for name, node in handle.items()
            if isinstance(node, h5py.Group) and name.startswith("traj_")
        ]
    if not ids:
        raise ValueError(f"D_core contains no traj_<id> groups: {_portable(path)}")
    return tuple(sorted(ids))


def load_split_manifest(path: str | Path) -> dict[str, tuple[int, ...]]:
    payload = _read_json(path)
    counts = payload.get("counts")
    if counts != {"train": 787, "development": 100, "test": 100}:
        raise ValueError("StackCube split manifest is not the frozen 787/100/100 split")
    result: dict[str, tuple[int, ...]] = {}
    for split in ("train", "development", "test"):
        values = payload.get(split)
        if not isinstance(values, list) or len(values) != int(counts[split]):
            raise ValueError(f"StackCube split {split!r} is malformed")
        result[split] = tuple(int(value) for value in values)
    if set(result["train"]) & set(result["development"]):
        raise ValueError("StackCube train/development split overlap detected")
    if set(result["train"]) & set(result["test"]):
        raise ValueError("StackCube train/test split overlap detected")
    if set(result["development"]) & set(result["test"]):
        raise ValueError("StackCube development/test split overlap detected")
    return result


def select_train_episode_ids(
    train_ids: Sequence[int],
    *,
    count: int = 10,
    seed: int = DEFAULT_SELECTION_SEED,
    requested: Sequence[int] | None = None,
) -> tuple[int, ...]:
    available = tuple(sorted({int(value) for value in train_ids}))
    if not available:
        raise ValueError("StackCube train split is empty")
    if requested:
        selected = tuple(int(value) for value in requested)
        if len(selected) != len(set(selected)):
            raise ValueError("Requested StackCube episode IDs must be unique")
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"Requested episode IDs are not in train split: {missing}")
        return selected
    if count <= 0 or count > len(available):
        raise ValueError(f"count must be in [1, {len(available)}], got {count}")
    generator = np.random.default_rng(int(seed))
    selected = generator.choice(np.asarray(available), size=count, replace=False)
    return tuple(sorted(int(value) for value in selected.tolist()))


def _metadata_by_episode(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("D_core trajectory.json must contain an episodes list")
    records: dict[int, dict[str, Any]] = {}
    for value in episodes:
        if not isinstance(value, Mapping):
            raise TypeError("D_core episode metadata must be objects")
        episode_id = int(value["episode_id"])
        if episode_id in records:
            raise ValueError(f"Duplicate D_core episode metadata: {episode_id}")
        records[episode_id] = dict(value)
    return records


def load_stackcube_episode(
    group: h5py.Group,
    *,
    episode_id: int,
    metadata: Mapping[str, Any],
    split: str,
) -> StackCubeEpisode:
    actions = _require_array(group, ("actions",), dtype=np.float32)
    qpos = _require_array(group, ("obs", "agent", "qpos"), dtype=np.float32)
    qvel = _require_array(group, ("obs", "agent", "qvel"), dtype=np.float32)
    images_base = _require_array(
        group,
        ("obs", "sensor_data", "base_camera", "rgb"),
        dtype=np.uint8,
    )
    images_hand = _require_array(
        group,
        ("obs", "sensor_data", "hand_camera", "rgb"),
        dtype=np.uint8,
    )
    cube_a_state = _require_array(
        group,
        ("env_states", "actors", "cubeA"),
        dtype=np.float32,
    )
    cube_b_state = _require_array(
        group,
        ("env_states", "actors", "cubeB"),
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"StackCube actions must have shape [T,7], got {actions.shape}")
    action_steps = int(actions.shape[0])
    for name, value, expected_tail in (
        ("qpos", qpos, (9,)),
        ("qvel", qvel, (9,)),
        ("base RGB", images_base, (128, 128, 3)),
        ("hand RGB", images_hand, (128, 128, 3)),
        ("cubeA state", cube_a_state, (13,)),
        ("cubeB state", cube_b_state, (13,)),
    ):
        if value.shape != (action_steps + 1, *expected_tail):
            raise ValueError(
                f"StackCube {name} must have shape {(action_steps + 1, *expected_tail)}, "
                f"got {value.shape}"
            )
    return StackCubeEpisode(
        episode_id=int(episode_id),
        split=str(split),
        route=str(metadata.get("route", "unknown")),
        source_episode_id=int(metadata.get("source_episode_id", episode_id)),
        success=bool(metadata.get("success", True)),
        first_success_step=(
            None
            if metadata.get("first_success_step") is None
            else int(metadata["first_success_step"])
        ),
        images_base=images_base,
        images_hand=images_hand,
        qpos=qpos,
        qvel=qvel,
        actions=actions,
        cube_a=cube_a_state[:, :3].copy(),
        cube_b=cube_b_state[:, :3].copy(),
    )


def _series_definitions(
    recording: rr.RecordingStream,
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


def build_blueprint(*, fps: float) -> rrb.Blueprint:
    """Build the shared camera/geometry/timeline Rerun layout."""

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
            name="18D qpos/qvel",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TimeSeriesView(
            origin=EVENTS_ROOT,
            contents=[f"{EVENTS_ROOT}/**"],
            name="StackCube diagnostics",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TextDocumentView(
            origin=CONTEXT_PATH,
            contents=[CONTEXT_PATH],
            name="Case metadata and semantics",
        ),
        name="Signals and metadata",
    )
    scene_contents = [
        f"{SCENE_ROOT}/tcp_path",
        f"{SCENE_ROOT}/tcp_position",
        f"{SCENE_ROOT}/cubeA_path",
        f"{SCENE_ROOT}/cubeA_position",
        f"{SCENE_ROOT}/cubeB_path",
        f"{SCENE_ROOT}/cubeB_position",
        f"{PANDA_ROOT}/skeleton",
        f"{PANDA_ROOT}/joints",
        f"{PANDA_ROOT}/gripper",
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin=BASE_CAMERA_PATH,
                    contents=[BASE_CAMERA_PATH],
                    name="Recorded base camera",
                ),
                rrb.Spatial2DView(
                    origin=HAND_CAMERA_PATH,
                    contents=[HAND_CAMERA_PATH],
                    name="Recorded hand camera",
                ),
                rrb.Spatial3DView(
                    origin=SCENE_ROOT,
                    contents=scene_contents,
                    name="Panda and StackCube diagnostic geometry",
                    line_grid=True,
                ),
                column_shares=[0.38, 0.28, 0.34],
                name="Cameras and geometry",
            ),
            signal_tabs,
            row_shares=[0.64, 0.36],
        ),
        rrb.TimePanel(timeline=TIMELINE, fps=fps, expanded=True),
        collapse_panels=False,
    )


def _log_static_context(
    recording: rr.RecordingStream,
    episode: StackCubeEpisode,
    *,
    selection_seed: int,
) -> None:
    state_names = tuple(
        [f"qpos_{index:02d}" for index in range(9)]
        + [f"qvel_{index:02d}" for index in range(9)]
    )
    _series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/action",
        names=episode.action_names,
        colors=ACTION_COLORS,
    )
    _series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/state",
        names=state_names,
        colors=STATE_COLORS,
    )
    _series_definitions(
        recording,
        root=EVENTS_ROOT,
        names=(
            "cubeA_height_m",
            "cubeA_to_cubeB_distance_m",
            "tcp_to_cubeA_distance_m",
            "action_l1",
            "terminal_success",
        ),
        colors=EVENT_COLORS,
    )
    recording.log(f"{SCENE_ROOT}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    recording.log(
        CONTEXT_PATH,
        rr.TextDocument(
            "\n".join(
                (
                    f"# ManiSkill {TASK_ID} training case {episode.episode_id}",
                    "",
                    f"- **Split:** {episode.split}",
                    f"- **Source episode:** {episode.source_episode_id}",
                    f"- **Route:** {episode.route}",
                    f"- **Action:** 7D ({', '.join(episode.action_names)})",
                    f"- **Frames/actions:** {episode.frame_count}/{episode.action_steps}",
                    f"- **Success label:** {episode.success}",
                    f"- **First success step:** {episode.first_success_step}",
                    f"- **Selection seed:** {selection_seed}",
                    "",
                    (
                        "The two camera panels are the recorded RGB observations used by the "
                        "StackCube policy pipeline."
                    ),
                    "",
                    (
                        "The blue Panda skeleton is reconstructed from recorded qpos with the "
                        "existing offline Panda kinematics helper."
                    ),
                    "",
                    (
                        "The red/green cube paths come from stored D_core scene state and are "
                        "diagnostic overlays only; they are not policy observation keys."
                    ),
                )
            ),
            media_type="text/markdown",
        ),
        static=True,
    )
    kinematics = PandaKinematics()
    tcp_path = np.stack(
        [kinematics.from_qpos(qpos[:7], qpos[7:9]).skeleton_points[-1] for qpos in episode.qpos],
        axis=0,
    )
    recording.log(
        f"{SCENE_ROOT}/tcp_path",
        rr.LineStrips3D([tcp_path], colors=[TCP_COLOR], radii=0.004),
        static=True,
    )
    recording.log(
        f"{SCENE_ROOT}/cubeA_path",
        rr.LineStrips3D([episode.cube_a], colors=[CUBE_A_COLOR], radii=0.004),
        static=True,
    )
    recording.log(
        f"{SCENE_ROOT}/cubeB_path",
        rr.LineStrips3D([episode.cube_b], colors=[CUBE_B_COLOR], radii=0.004),
        static=True,
    )


def _log_panda_frame(recording: rr.RecordingStream, qpos: np.ndarray) -> np.ndarray:
    kinematics = PandaKinematics()
    pose = kinematics.from_qpos(qpos[:7], qpos[7:9])
    recording.log(
        f"{PANDA_ROOT}/skeleton",
        rr.LineStrips3D(
            [pose.skeleton_points],
            colors=[PANDA_COLOR],
            radii=0.009,
            labels=["Panda arm skeleton (qpos)"],
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
    return pose.skeleton_points[-1]


def _log_frame(recording: rr.RecordingStream, episode: StackCubeEpisode, step: int) -> None:
    recording.set_time(TIMELINE, sequence=step)
    recording.log(BASE_CAMERA_PATH, rr.Image(episode.images_base[step]).compress())
    recording.log(HAND_CAMERA_PATH, rr.Image(episode.images_hand[step]).compress())
    for index, value in enumerate(episode.qpos[step]):
        recording.log(f"{SIGNALS_ROOT}/state/qpos_{index:02d}", rr.Scalars(float(value)))
    for index, value in enumerate(episode.qvel[step]):
        recording.log(f"{SIGNALS_ROOT}/state/qvel_{index:02d}", rr.Scalars(float(value)))
    tcp_position = _log_panda_frame(recording, episode.qpos[step])
    recording.log(
        f"{SCENE_ROOT}/tcp_position",
        rr.Points3D([tcp_position], colors=[TCP_COLOR], radii=0.012, labels=["tcp"]),
    )
    recording.log(
        f"{SCENE_ROOT}/cubeA_position",
        rr.Points3D([episode.cube_a[step]], colors=[CUBE_A_COLOR], radii=0.014, labels=["cubeA"]),
    )
    recording.log(
        f"{SCENE_ROOT}/cubeB_position",
        rr.Points3D([episode.cube_b[step]], colors=[CUBE_B_COLOR], radii=0.014, labels=["cubeB"]),
    )
    cube_a_to_b = float(np.linalg.norm(episode.cube_a[step] - episode.cube_b[step]))
    tcp_to_a = float(np.linalg.norm(tcp_position - episode.cube_a[step]))
    action_l1 = 0.0
    if step < episode.action_steps:
        for index, value in enumerate(episode.actions[step]):
            recording.log(
                f"{SIGNALS_ROOT}/action/{episode.action_names[index]}",
                rr.Scalars(float(value)),
            )
        action_l1 = float(np.abs(episode.actions[step]).sum())
    recording.log(f"{EVENTS_ROOT}/cubeA_height_m", rr.Scalars(float(episode.cube_a[step, 2])))
    recording.log(f"{EVENTS_ROOT}/cubeA_to_cubeB_distance_m", rr.Scalars(cube_a_to_b))
    recording.log(f"{EVENTS_ROOT}/tcp_to_cubeA_distance_m", rr.Scalars(tcp_to_a))
    recording.log(f"{EVENTS_ROOT}/action_l1", rr.Scalars(action_l1))
    recording.log(f"{EVENTS_ROOT}/terminal_success", rr.Scalars(int(episode.success)))


def write_episode_recording(
    episode: StackCubeEpisode,
    *,
    output_path: str | Path,
    fps: float,
    selection_seed: int,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing recording: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    recording = rr.RecordingStream(
        OFFLINE_APPLICATION_ID,
        recording_id=f"stackcube_train_episode_{episode.episode_id:04d}",
    )
    try:
        recording.save(temporary)
        recording.send_recording_name(
            f"ManiSkill {TASK_ID} train | episode {episode.episode_id:04d} | {episode.route}"
        )
        rr.send_blueprint(
            build_blueprint(fps=fps),
            make_active=True,
            make_default=True,
            recording=recording,
        )
        _log_static_context(recording, episode, selection_seed=selection_seed)
        for step in range(episode.frame_count):
            _log_frame(recording, episode, step)
        recording.flush()
    finally:
        recording.disconnect()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Rerun recording was not written: {temporary}")
    temporary.replace(destination)
    return destination


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _fit_preview(image: np.ndarray, *, size: int) -> Image.Image:
    frame = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return frame.resize((size, size), Image.Resampling.NEAREST)


def write_contact_sheet(
    episodes: Sequence[StackCubeEpisode],
    *,
    output_path: str | Path,
    tile_size: int = 176,
) -> Path:
    """Create a compact static preview of the same camera frames logged to Rerun."""

    if not episodes:
        raise ValueError("Cannot create a contact sheet without episodes")
    columns = ("base · start", "base · middle", "base · end", "hand · start", "hand · middle", "hand · end")
    label_height = 42
    row_height = tile_size + label_height
    margin = 18
    label_width = 210
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + label_width + tile_size * len(columns),
            margin * 2 + label_height + row_height * len(episodes),
        ),
        (245, 246, 248),
    )
    draw = ImageDraw.Draw(canvas)
    header_font = _font(18)
    label_font = _font(16)
    small_font = _font(13)
    x0 = margin + label_width
    for column_index, title in enumerate(columns):
        x = x0 + column_index * tile_size
        draw.text((x + 4, margin), title, fill=(20, 20, 20), font=small_font)
    for row_index, episode in enumerate(episodes):
        y = margin + label_height + row_index * row_height
        draw.rectangle((margin, y, margin + label_width - 8, y + tile_size), fill=(232, 235, 239))
        draw.text(
            (margin + 8, y + 8),
            f"episode {episode.episode_id:04d}",
            fill=(10, 10, 10),
            font=header_font,
        )
        draw.text(
            (margin + 8, y + 34),
            f"{episode.route} · {episode.action_steps} steps",
            fill=(55, 55, 55),
            font=label_font,
        )
        positions = (
            0,
            max(0, episode.frame_count // 2),
            episode.frame_count - 1,
        )
        frames = tuple(episode.images_base[index] for index in positions) + tuple(
            episode.images_hand[index] for index in positions
        )
        for column_index, frame in enumerate(frames):
            x = x0 + column_index * tile_size
            canvas.paste(_fit_preview(frame, size=tile_size), (x, y))
            draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline=(190, 195, 202))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output


def _episode_summary(episode: StackCubeEpisode, recording_path: Path) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "split": episode.split,
        "route": episode.route,
        "source_episode_id": episode.source_episode_id,
        "frame_count": episode.frame_count,
        "action_steps": episode.action_steps,
        "success": episode.success,
        "first_success_step": episode.first_success_step,
        "recording": _portable(recording_path),
    }


def write_train_visualizations(
    *,
    d_core_h5: str | Path = DEFAULT_D_CORE_H5,
    d_core_json: str | Path = DEFAULT_D_CORE_JSON,
    split_manifest: str | Path = DEFAULT_SPLIT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    count: int = 10,
    seed: int = DEFAULT_SELECTION_SEED,
    episode_ids: Sequence[int] | None = None,
    fps: float = DEFAULT_FPS,
) -> dict[str, Any]:
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive finite number")
    output = _rooted(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {_portable(output)}")
    output.mkdir(parents=True, exist_ok=True)
    split_payload = load_split_manifest(split_manifest)
    available = set(available_episode_ids(d_core_h5))
    selected = select_train_episode_ids(
        split_payload["train"],
        count=count,
        seed=seed,
        requested=episode_ids,
    )
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Selected train episodes are missing from D_core: {missing}")
    metadata = _metadata_by_episode(d_core_json)
    missing_metadata = sorted(set(selected) - set(metadata))
    if missing_metadata:
        raise ValueError(f"Selected train episodes are missing metadata: {missing_metadata}")
    loaded: list[StackCubeEpisode] = []
    recordings: list[dict[str, Any]] = []
    episodes_dir = output / "episodes"
    with h5py.File(_rooted(d_core_h5), "r") as handle:
        for episode_id in selected:
            episode = load_stackcube_episode(
                handle[f"traj_{episode_id}"],
                episode_id=episode_id,
                metadata=metadata[episode_id],
                split="train",
            )
            loaded.append(episode)
            recording_path = episodes_dir / f"episode_{episode_id:04d}_{episode.result_label}.rrd"
            write_episode_recording(
                episode,
                output_path=recording_path,
                fps=fps,
                selection_seed=seed,
            )
            recordings.append(_episode_summary(episode, recording_path))
    preview_path = write_contact_sheet(loaded, output_path=output / "preview" / "contact_sheet.png")
    manifest = {
        "schema_version": 1,
        "task": TASK_ID,
        "split": "train",
        "selection": {
            "method": "numpy.default_rng_choice_without_replacement",
            "seed": int(seed),
            "requested_count": len(selected),
            "selected_episode_ids": list(selected),
        },
        "inputs": {
            "d_core_h5": _portable(d_core_h5),
            "d_core_h5_sha256": sha256_file(d_core_h5),
            "d_core_json": _portable(d_core_json),
            "split_manifest": _portable(split_manifest),
            "split_manifest_sha256": sha256_file(split_manifest),
        },
        "viewer": {
            "kind": VIEWER_KIND,
            "application_id": OFFLINE_APPLICATION_ID,
            "component": "Rerun",
            "launcher": "robot-rerun",
        },
        "semantics": {
            "camera": "Recorded base_camera and hand_camera RGB frames",
            "robot": "Visualization-only Panda skeleton reconstructed from qpos",
            "scene_overlay": "Stored cubeA/cubeB D_core state; diagnostic only, not policy input",
            "policy_observation_keys": [
                "observation.images.base_camera",
                "observation.images.hand_camera",
                "observation.state",
            ],
        },
        "preview": _portable(preview_path),
        "episodes": recordings,
    }
    write_json_atomic(output / "visualization_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-core-h5", default=DEFAULT_D_CORE_H5)
    parser.add_argument("--d-core-json", default=DEFAULT_D_CORE_JSON)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--episode", type=int, action="append", dest="episode_ids")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = write_train_visualizations(
        d_core_h5=args.d_core_h5,
        d_core_json=args.d_core_json,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        episode_ids=args.episode_ids,
        fps=args.fps,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
