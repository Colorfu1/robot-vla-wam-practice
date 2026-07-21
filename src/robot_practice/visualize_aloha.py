"""Synchronized 2D, signal, and nominal 3D visualization for ALOHA datasets."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

from robot_practice.aloha_kinematics import (
    ARM_JOINT_NAMES,
    CANONICAL_JOINT_NAMES,
    ArmPose,
    forward_bimanual,
)

LOGGER = logging.getLogger(__name__)

STATE_KEY = "observation.state"
ACTION_KEY = "action"
EFFORT_KEY = "observation.effort"
REQUIRED_NUMERIC_KEYS = (STATE_KEY, ACTION_KEY, EFFORT_KEY)

LEFT_COLOR = [30, 170, 255]
RIGHT_COLOR = [255, 105, 55]
JOINT_COLORS = (
    [76, 114, 176],
    [221, 132, 82],
    [85, 168, 104],
    [196, 78, 82],
    [129, 114, 179],
    [147, 120, 96],
    [218, 139, 195],
)
SKELETON_LABELS = (
    "base",
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "end_effector",
)


class Recorder(Protocol):
    """The small portion of the Rerun API used by frame logging."""

    def set_time(
        self,
        timeline: str,
        *,
        sequence: int | None = None,
        duration: float | None = None,
    ) -> None: ...

    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None: ...


@dataclass(frozen=True)
class VisualizationConfig:
    repo_id: str
    episode_index: int
    root: Path | None = None
    max_frames: int | None = None
    no_images: bool = False
    save: Path | None = None
    batch_size: int = 32
    num_workers: int = 0


@dataclass(frozen=True)
class VisualizationSummary:
    frames: int
    fps: float
    task: str
    root: Path
    output_path: Path | None


def validate_config(config: VisualizationConfig) -> None:
    if not config.repo_id or "/" not in config.repo_id:
        raise ValueError("--repo-id must look like '<owner>/<dataset>'")
    if config.episode_index < 0:
        raise ValueError("--episode-index must be non-negative")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if config.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if config.save is not None and config.save.suffix.lower() != ".rrd":
        raise ValueError("--save must point to a .rrd file")


def find_cached_dataset_root(repo_id: str, explicit_root: Path | None) -> Path | None:
    """Prefer an existing local LeRobot root, then an existing Hub snapshot."""

    if explicit_root is not None:
        return explicit_root.expanduser().resolve()

    lerobot_root = HF_LEROBOT_HOME / repo_id
    if (lerobot_root / "meta" / "info.json").is_file() and any(
        (lerobot_root / "data").rglob("*.parquet")
    ):
        return lerobot_root

    try:
        snapshot = snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=CODEBASE_VERSION,
            cache_dir=HF_LEROBOT_HOME / "hub",
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None
    snapshot_root = Path(snapshot)
    return snapshot_root if (snapshot_root / "meta" / "info.json").is_file() else None


def validate_episode_index(metadata: LeRobotDatasetMetadata, episode_index: int) -> None:
    if episode_index >= metadata.total_episodes:
        raise ValueError(
            f"Episode {episode_index} is outside [0, {metadata.total_episodes - 1}] "
            f"for {metadata.repo_id}"
        )


def _feature_names(metadata: LeRobotDatasetMetadata, key: str) -> tuple[str, ...] | None:
    names = metadata.features[key].get("names")
    if names is None:
        return None
    if isinstance(names, dict):
        names = names.get("motors")
    if not isinstance(names, (list, tuple)):
        raise ValueError(f"{key} feature names must be a list or a motors mapping")
    return tuple(str(name) for name in names)


def validate_dataset_layout(metadata: LeRobotDatasetMetadata) -> None:
    if metadata.robot_type != "aloha":
        raise ValueError(f"Expected robot_type='aloha', got {metadata.robot_type!r}")

    for key in REQUIRED_NUMERIC_KEYS:
        if key not in metadata.features:
            raise ValueError(f"Dataset is missing required feature {key!r}")
        shape = tuple(metadata.features[key].get("shape", ()))
        if shape != (14,):
            raise ValueError(f"{key} must have shape [14], got {list(shape)}")
        names = _feature_names(metadata, key)
        if names is not None and names != CANONICAL_JOINT_NAMES:
            raise ValueError(
                f"{key} motor order does not match canonical ALOHA order: "
                f"expected {CANONICAL_JOINT_NAMES}, got {names}"
            )


def load_dataset(
    config: VisualizationConfig,
) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    root = find_cached_dataset_root(config.repo_id, config.root)
    metadata = LeRobotDatasetMetadata(config.repo_id, root=root)
    validate_episode_index(metadata, config.episode_index)
    validate_dataset_layout(metadata)
    dataset = LeRobotDataset(
        config.repo_id,
        root=metadata.root,
        episodes=[config.episode_index],
        download_videos=not config.no_images,
    )
    return dataset, metadata


def _task_for_episode(dataset: LeRobotDataset) -> str:
    task_index_value = dataset.hf_dataset[0]["task_index"]
    task_index = int(task_index_value.item())
    tasks = dataset.meta.tasks
    matches = tasks[tasks["task_index"] == task_index]
    if matches.empty:
        return f"Unknown task index {task_index}"
    return str(matches.index[0])


def make_dataloader(
    dataset: LeRobotDataset,
    *,
    frame_count: int,
    no_images: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    source = dataset.hf_dataset if no_images else dataset
    selected = Subset(source, range(frame_count))
    return DataLoader(
        selected,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )


def compute_velocity(
    state: NDArray[np.float64],
    timestamp: float,
    previous_state: NDArray[np.float64] | None,
    previous_timestamp: float | None,
) -> NDArray[np.float64]:
    if previous_state is None or previous_timestamp is None:
        return np.zeros_like(state)
    delta = timestamp - previous_timestamp
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(
            f"Episode timestamps must strictly increase; got delta={delta!r} at {timestamp!r}"
        )
    return (state - previous_state) / delta


def _to_numpy(value: Any) -> NDArray[np.float64]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _to_hwc_uint8(image: torch.Tensor) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected a CHW image tensor, got {tuple(image.shape)}")
    return (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )


def _camera_entity_path(camera_key: str) -> str:
    return f"cameras/{camera_key.rsplit('.', maxsplit=1)[-1]}"


def build_blueprint(camera_keys: Sequence[str], fps: float) -> rrb.Blueprint:
    camera_views = [
        rrb.Spatial2DView(
            origin=_camera_entity_path(camera_key),
            contents=[_camera_entity_path(camera_key)],
            name=camera_key.rsplit(".", maxsplit=1)[-1],
        )
        for camera_key in camera_keys
    ]
    cameras = rrb.Vertical(*camera_views, name="Cameras") if camera_views else None
    robot = rrb.Spatial3DView(
        origin="robot",
        contents=["robot/**"],
        name="Nominal ALOHA 3D",
        line_grid=True,
    )

    main_view: rrb.Container | rrb.View
    if cameras is None:
        main_view = robot
    else:
        main_view = rrb.Horizontal(cameras, robot, column_shares=[0.55, 0.45], name="Episode")

    signal_tabs = rrb.Tabs(
        rrb.TimeSeriesView(
            origin="signals",
            contents=["signals/position/**", "signals/action/**"],
            name="Position vs action",
        ),
        rrb.TimeSeriesView(
            origin="signals/effort",
            contents=["signals/effort/**"],
            name="Effort",
        ),
        rrb.TimeSeriesView(
            origin="signals/velocity",
            contents=["signals/velocity/**"],
            name="Velocity",
        ),
        rrb.TimeSeriesView(
            origin="signals/action_minus_state",
            contents=["signals/action_minus_state/**"],
            name="Action - state",
        ),
        rrb.TextDocumentView(
            origin="episode/task",
            contents=["episode/task"],
            name="Task and model notes",
        ),
        name="Signals",
    )
    return rrb.Blueprint(
        rrb.Vertical(main_view, signal_tabs, row_shares=[0.68, 0.32]),
        rrb.TimePanel(timeline="frame_index", fps=fps, expanded=True),
        collapse_panels=False,
    )


def _initialize_recording(
    config: VisualizationConfig,
    camera_keys: Sequence[str],
    fps: float,
) -> rrb.Blueprint:
    blueprint = build_blueprint(camera_keys, fps)
    rr.init(
        "robot_practice_aloha_dataset",
        spawn=False,
    )
    if config.save is not None:
        config.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(config.save)
    else:
        rr.spawn()
    rr.send_blueprint(blueprint)
    return blueprint


def _log_static_context(
    recorder: Recorder,
    *,
    task: str,
    config: VisualizationConfig,
) -> None:
    recorder.log("robot", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    notes = (
        f"# Episode {config.episode_index}\n\n"
        f"**Task:** {task}\n\n"
        "The 3D ViperX 300S skeleton uses nominal gym-aloha geometry. "
        "It is not calibrated or registered to the RGB cameras."
    )
    recorder.log("episode/task", rr.TextDocument(notes, media_type="text/markdown"), static=True)

    for quantity in (
        "position",
        "action",
        "effort",
        "velocity",
        "action_minus_state",
    ):
        for side in ("left", "right"):
            for joint_index, joint_name in enumerate(ARM_JOINT_NAMES):
                path = f"signals/{quantity}/{side}/{joint_name}"
                recorder.log(
                    path,
                    rr.SeriesLines(
                        colors=[JOINT_COLORS[joint_index]],
                        widths=[2.0 if quantity == "position" else 1.0],
                        names=[f"{quantity}/{side}_{joint_name}"],
                    ),
                    static=True,
                )


def _log_signals(
    recorder: Recorder,
    state: NDArray[np.float64],
    action: NDArray[np.float64],
    effort: NDArray[np.float64],
    velocity: NDArray[np.float64],
) -> None:
    quantities = {
        "position": state,
        "action": action,
        "effort": effort,
        "velocity": velocity,
        "action_minus_state": action - state,
    }
    for quantity, values in quantities.items():
        for value_index, value in enumerate(values):
            side = "left" if value_index < 7 else "right"
            joint_name = ARM_JOINT_NAMES[value_index % 7]
            recorder.log(
                f"signals/{quantity}/{side}/{joint_name}",
                rr.Scalars(float(value)),
            )


def _log_arm_pose(recorder: Recorder, side: str, pose: ArmPose) -> None:
    color = LEFT_COLOR if side == "left" else RIGHT_COLOR
    root = f"robot/{side}"
    recorder.log(
        f"{root}/skeleton",
        rr.LineStrips3D(strips=[pose.joint_points], colors=[color], radii=0.008),
    )
    recorder.log(
        f"{root}/joints",
        rr.Points3D(
            pose.joint_points,
            colors=[color],
            radii=0.014,
            labels=SKELETON_LABELS,
            show_labels=False,
        ),
    )
    recorder.log(
        f"{root}/gripper",
        rr.LineStrips3D(strips=[pose.finger_points], colors=[color], radii=0.01),
    )
    recorder.log(
        f"{root}/end_effector",
        rr.Transform3D(
            translation=pose.end_effector_translation,
            mat3x3=pose.end_effector_rotation,
            axis_length=0.09,
        ),
    )


def log_frame(
    recorder: Recorder,
    *,
    frame_index: int,
    elapsed_seconds: float,
    state: NDArray[np.float64],
    action: NDArray[np.float64],
    effort: NDArray[np.float64],
    velocity: NDArray[np.float64],
    images: dict[str, torch.Tensor],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    recorder.set_time("frame_index", sequence=frame_index)
    recorder.set_time("timestamp", duration=elapsed_seconds)

    for camera_key, image in images.items():
        recorder.log(_camera_entity_path(camera_key), rr.Image(_to_hwc_uint8(image)).compress())

    _log_signals(recorder, state, action, effort, velocity)
    pose = forward_bimanual(state)
    _log_arm_pose(recorder, "left", pose.left)
    _log_arm_pose(recorder, "right", pose.right)
    return pose.left.end_effector_translation, pose.right.end_effector_translation


def log_complete_paths(
    recorder: Recorder,
    left_path: Sequence[NDArray[np.float64]],
    right_path: Sequence[NDArray[np.float64]],
) -> None:
    for side, points, color in (
        ("left", left_path, LEFT_COLOR),
        ("right", right_path, RIGHT_COLOR),
    ):
        recorder.log(
            f"robot/{side}/end_effector_path",
            rr.LineStrips3D(
                strips=[np.stack(points)],
                colors=[[*color, 110]],
                radii=0.004,
            ),
            static=True,
        )


def _batch_size(batch: dict[str, Any]) -> int:
    return len(batch["frame_index"])


def _iter_selected_frames(
    batches: Iterable[dict[str, Any]],
    camera_keys: Sequence[str],
) -> Iterator[
    tuple[
        int,
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        dict[str, torch.Tensor],
    ]
]:
    for batch in batches:
        for index in range(_batch_size(batch)):
            images = {key: batch[key][index] for key in camera_keys if key in batch}
            yield (
                int(batch["frame_index"][index].item()),
                float(batch["timestamp"][index].item()),
                _to_numpy(batch[STATE_KEY][index]),
                _to_numpy(batch[ACTION_KEY][index]),
                _to_numpy(batch[EFFORT_KEY][index]),
                images,
            )


def visualize(config: VisualizationConfig, recorder: Recorder = rr) -> VisualizationSummary:
    validate_config(config)
    dataset, metadata = load_dataset(config)
    task = _task_for_episode(dataset)
    total_frames = min(len(dataset), config.max_frames or len(dataset))
    camera_keys = () if config.no_images else tuple(metadata.camera_keys)

    LOGGER.info("Dataset root: %s", dataset.root)
    LOGGER.info(
        "Episode %d: %d frames at %.3f Hz; task=%s",
        config.episode_index,
        total_frames,
        dataset.fps,
        task,
    )
    LOGGER.warning("The 3D model is nominal and is not registered to the RGB cameras.")

    if recorder is rr:
        _initialize_recording(config, camera_keys, float(dataset.fps))
    _log_static_context(recorder, task=task, config=config)

    dataloader = make_dataloader(
        dataset,
        frame_count=total_frames,
        no_images=config.no_images,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    batches: Iterable[dict[str, Any]] = dataloader
    if recorder is rr:
        batches = tqdm(
            dataloader,
            total=math.ceil(total_frames / config.batch_size),
            desc="Logging ALOHA episode",
            unit="batch",
        )

    previous_state: NDArray[np.float64] | None = None
    previous_timestamp: float | None = None
    initial_timestamp: float | None = None
    left_path: list[NDArray[np.float64]] = []
    right_path: list[NDArray[np.float64]] = []

    for frame_index, timestamp, state, action, effort, images in _iter_selected_frames(
        batches, camera_keys
    ):
        for key, values in ((STATE_KEY, state), (ACTION_KEY, action), (EFFORT_KEY, effort)):
            if values.shape != (14,):
                raise ValueError(f"{key} must have shape (14,), got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{key} contains non-finite values at frame {frame_index}")
        if not np.isfinite(timestamp):
            raise ValueError(f"timestamp is non-finite at frame {frame_index}")

        velocity = compute_velocity(state, timestamp, previous_state, previous_timestamp)
        if initial_timestamp is None:
            initial_timestamp = timestamp
        left_ee, right_ee = log_frame(
            recorder,
            frame_index=frame_index,
            elapsed_seconds=timestamp - initial_timestamp,
            state=state,
            action=action,
            effort=effort,
            velocity=velocity,
            images=images,
        )
        left_path.append(left_ee)
        right_path.append(right_ee)
        previous_state = state
        previous_timestamp = timestamp

    if not left_path:
        raise ValueError("The selected episode contains no frames")
    log_complete_paths(recorder, left_path, right_path)
    if recorder is rr:
        rr.disconnect()

    return VisualizationSummary(
        frames=len(left_path),
        fps=float(dataset.fps),
        task=task,
        root=Path(dataset.root),
        output_path=config.save,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face LeRobot dataset repo ID")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode to visualize")
    parser.add_argument("--root", type=Path, help="Exact local LeRobot dataset root")
    parser.add_argument("--max-frames", type=int, help="Only inspect the first N episode frames")
    parser.add_argument("--no-images", action="store_true", help="Skip video download and decoding")
    parser.add_argument("--save", type=Path, help="Save to .rrd instead of opening a viewer")
    parser.add_argument("--batch-size", type=int, default=32, help="Frame decoding batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argument_parser().parse_args(argv)
    summary = visualize(VisualizationConfig(**vars(args)))
    LOGGER.info(
        "Visualized %d frames from %s at %.3f Hz%s",
        summary.frames,
        summary.root,
        summary.fps,
        f" -> {summary.output_path}" if summary.output_path else "",
    )


if __name__ == "__main__":
    main()
