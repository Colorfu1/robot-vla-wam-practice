"""Teacher-forced ACT prediction on one explicitly selected LIBERO task."""

from __future__ import annotations

import argparse
import logging
import shutil
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION

from robot_practice.evaluation.common import (
    build_run_config,
    create_output_directory,
    launch_rerun_viewer,
    load_act_checkpoint,
    log_terminal_metrics,
    mark_run_complete,
    release_output_directory_lock,
    timed_call,
    validate_act_libero_contract,
    validate_complete_checkpoint,
    write_json_atomic,
    write_parquet_atomic,
)
from robot_practice.evaluation.libero_checkpoint import (
    assess_checkpoint_task,
    log_checkpoint_task_assessment,
)
from robot_practice.evaluation.libero_kinematics import (
    FixedCameraCalibration,
    PandaSkeletonPose,
    PandaSkeletonReconstructor,
)
from robot_practice.evaluation.libero_dataset_cache import (
    AGENTVIEW_IMAGE_CONVENTIONS,
    DatasetSelection,
    resolve_agentview_image_convention,
    resolve_offline_dataset,
)
from robot_practice.evaluation.libero_tasks import (
    SUPPORTED_SUITES,
    LiberoTaskIdentity,
    log_task_identity,
    resolve_libero_task,
)
from robot_practice.evaluation.metrics import compute_offline_metrics
from robot_practice.evaluation.metrics_report import write_metrics_report
from robot_practice.evaluation.rerun_visualization import (
    OfflineCommandHistory,
    OfflineTrajectoryHistory,
    initialize_offline_recording_stream,
    log_metric_summary,
    log_offline_frame,
    stage_merged_recordings,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfflineEvaluationConfig:
    policy_path: Path
    suite: str
    task_id: int
    episodes: tuple[int, ...] | None = None
    dataset_root: Path | None = None
    batch_size: int = 1
    num_visualized_episodes: int = 1
    output_dir: Path | None = None
    overwrite: bool = False
    max_frames: int | None = None
    rebuild_dataset_cache: bool = False
    open_viewer: bool = True
    agentview_image_convention: str = "auto"


@dataclass(frozen=True)
class OfflineEvaluationResult:
    output_dir: Path
    metrics: dict[str, Any]
    viewer_opened: bool


@dataclass
class OfflineEpisodeRecording:
    episode_index: int | None
    path: Path
    stream: rr.RecordingStream
    trajectory_history: OfflineTrajectoryHistory = field(
        default_factory=OfflineTrajectoryHistory
    )
    command_history: OfflineCommandHistory = field(default_factory=OfflineCommandHistory)
    previous_skeleton_pose: PandaSkeletonPose | None = None
    previous_frame_index: int | None = None
    flush_attempted: bool = False
    disconnect_attempted: bool = False

    def accept_frame_index(self, frame_index: int) -> None:
        if self.previous_frame_index is None:
            if frame_index != 0:
                raise ValueError(
                    f"Dataset episode {self.episode_index} must start at frame zero, "
                    f"got {frame_index}"
                )
        elif frame_index <= self.previous_frame_index:
            raise ValueError(
                f"Dataset episode {self.episode_index} frame indices must be "
                f"monotonically increasing, got {self.previous_frame_index} then "
                f"{frame_index}"
            )
        self.previous_frame_index = frame_index


def flush_and_disconnect_recordings(
    recordings: Sequence[OfflineEpisodeRecording],
) -> None:
    """Attempt every flush before attempting every disconnect."""

    errors: list[BaseException] = []
    for recording in recordings:
        if recording.flush_attempted:
            continue
        try:
            recording.stream.flush()
        except BaseException as exc:
            errors.append(exc)
        finally:
            recording.flush_attempted = True
    for recording in recordings:
        if recording.disconnect_attempted:
            continue
        try:
            recording.stream.disconnect()
        except BaseException as exc:
            errors.append(exc)
        finally:
            recording.disconnect_attempted = True
    if errors:
        raise errors[0]


def validate_config(
    config: OfflineEvaluationConfig,
    task: LiberoTaskIdentity | None = None,
) -> LiberoTaskIdentity:
    task = task or resolve_libero_task(config.suite, config.task_id)
    if config.episodes is not None:
        if not config.episodes:
            raise ValueError("--episodes must contain at least one episode index")
        if len(set(config.episodes)) != len(config.episodes):
            raise ValueError("--episodes must not contain duplicate indices")
        if any(index < 0 for index in config.episodes):
            raise ValueError("--episodes values must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if config.num_visualized_episodes < 0:
        raise ValueError("--num-visualized-episodes must be non-negative")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if config.overwrite and config.output_dir is None:
        raise ValueError("--overwrite requires an explicit --output-dir")
    if config.dataset_root is not None and config.rebuild_dataset_cache:
        raise ValueError(
            "--dataset-root cannot be combined with --rebuild-dataset-cache"
        )
    if config.agentview_image_convention not in AGENTVIEW_IMAGE_CONVENTIONS:
        raise ValueError(
            "--agentview-image-convention must be one of "
            f"{list(AGENTVIEW_IMAGE_CONVENTIONS)}"
        )
    return task


def _load_policy_config(policy_path: Path) -> ACTConfig:
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise TypeError(f"Expected ACTConfig, got {type(config).__name__}")
    validate_act_libero_contract(config)
    return config


def _make_dataset(
    selection: DatasetSelection,
    policy_config: ACTConfig,
) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    metadata = LeRobotDatasetMetadata(selection.repo_id, root=selection.root)
    for episode in selection.episodes:
        if episode >= metadata.total_episodes:
            raise ValueError(
                f"Episode {episode} is outside [0, {metadata.total_episodes - 1}]"
            )

    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    if delta_timestamps is None or ACTION not in delta_timestamps:
        raise ValueError("ACT config did not resolve action delta timestamps")
    expected = np.arange(policy_config.chunk_size, dtype=np.float64) / float(metadata.fps)
    actual = np.asarray(delta_timestamps[ACTION], dtype=np.float64)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-9):
        raise ValueError(
            "ACT action delta timestamps must be frame offsets converted to seconds; "
            f"expected {expected.tolist()}, got {actual.tolist()}"
        )

    dataset = LeRobotDataset(
        selection.repo_id,
        root=selection.root,
        episodes=list(selection.episodes),
        delta_timestamps=delta_timestamps,
        download_videos=True,
    )
    task_indices = {int(value) for value in dataset.hf_dataset.unique("task_index")}
    if task_indices != {selection.task_index}:
        raise ValueError(
            "All offline episodes must belong to mapped dataset task index "
            f"{selection.task_index}; "
            f"got {sorted(task_indices)}"
        )
    return dataset, metadata


def _prediction_schema(chunk_size: int) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("suite", pa.string(), nullable=False),
            pa.field("libero_task_id", pa.int32(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("dataset_index", pa.int64(), nullable=False),
            pa.field("timestamp_s", pa.float64(), nullable=False),
            pa.field("task_index", pa.int64(), nullable=False),
            pa.field("batch_id", pa.int64(), nullable=False),
            pa.field("batch_row", pa.int32(), nullable=False),
            pa.field("batch_size", pa.int32(), nullable=False),
            pa.field("target_action", pa.list_(pa.float32(), 7), nullable=False),
            pa.field("predicted_action", pa.list_(pa.float32(), 7), nullable=False),
            pa.field(
                "target_chunk", pa.list_(pa.float32(), chunk_size * 7), nullable=False
            ),
            pa.field(
                "predicted_chunk", pa.list_(pa.float32(), chunk_size * 7), nullable=False
            ),
            pa.field("valid_horizon", pa.list_(pa.bool_(), chunk_size), nullable=False),
            pa.field("absolute_error", pa.list_(pa.float32(), 7), nullable=False),
            pa.field("model_query_latency_ms", pa.float64(), nullable=True),
        ],
        metadata={
            b"schema_version": b"2",
            b"chunk_shape": f"[{chunk_size},7]".encode(),
            b"action_dimension": b"7",
        },
    )
    return schema


def _batch_value(batch: dict[str, Any], key: str, index: int) -> Any:
    value = batch[key][index]
    return value.item() if isinstance(value, torch.Tensor) and value.ndim == 0 else value


def _observation_input_keys(config: ACTConfig) -> tuple[str, ...]:
    if not config.input_features:
        raise ValueError("ACT checkpoint has no input features")
    return tuple(config.input_features)


def _autocast_context(config: ACTConfig) -> Any:
    device = torch.device(config.device)
    if config.use_amp:
        return torch.autocast(device_type=device.type)
    return nullcontext()


def _make_skeleton_reconstructor(
    task: LiberoTaskIdentity,
    visualized_episode_ids: Sequence[int],
) -> PandaSkeletonReconstructor | None:
    """Avoid creating a MuJoCo environment for metrics-only offline evaluation."""

    if not visualized_episode_ids:
        return None
    return PandaSkeletonReconstructor(task)


def _camera_hwc_shape(
    metadata: LeRobotDatasetMetadata,
    camera_key: str,
) -> tuple[int, int, int]:
    """Read and validate the LeRobot HWC contract for one video feature."""

    feature = metadata.features.get(camera_key)
    if not isinstance(feature, Mapping):
        raise ValueError(f"Camera feature metadata is missing for {camera_key!r}")
    names = tuple(feature.get("names", ()))
    if names != ("height", "width", "channel"):
        raise ValueError(
            f"Camera feature {camera_key!r} must use HWC names "
            f"['height', 'width', 'channel'], got {list(names)}"
        )
    shape = tuple(feature.get("shape", ()))
    if (
        len(shape) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in shape
        )
        or shape[2] not in (1, 3, 4)
    ):
        raise ValueError(
            f"Camera feature {camera_key!r} must have a positive HWC shape with "
            f"1, 3, or 4 channels, got {shape}"
        )
    return tuple(int(value) for value in shape)


def evaluate_offline(
    config: OfflineEvaluationConfig,
    *,
    max_frames: int | None = None,
) -> OfflineEvaluationResult:
    """Run teacher-forced predictions and persist metrics, Parquet, and Rerun output."""

    if max_frames is not None:
        if config.max_frames is not None and config.max_frames != max_frames:
            raise ValueError("max_frames was supplied both in config and function arguments")
        config = OfflineEvaluationConfig(**{**asdict(config), "max_frames": max_frames})
    task = resolve_libero_task(config.suite, config.task_id)
    log_task_identity(task, LOGGER)
    validate_config(config, task)
    manifest = validate_complete_checkpoint(config.policy_path)
    LOGGER.info("[Policy] checkpoint: %s", manifest.path)
    policy_config = _load_policy_config(manifest.path)
    checkpoint_assessment = assess_checkpoint_task(manifest, task)
    log_checkpoint_task_assessment(checkpoint_assessment, LOGGER)
    selection = resolve_offline_dataset(
        task,
        episodes=config.episodes,
        dataset_root=config.dataset_root,
        rebuild_dataset_cache=config.rebuild_dataset_cache,
    )
    if config.num_visualized_episodes > len(selection.episodes):
        raise ValueError(
            "--num-visualized-episodes must not exceed the number of selected "
            f"episodes ({len(selection.episodes)})"
        )
    resolved_agentview_image_convention = resolve_agentview_image_convention(
        selection,
        config.agentview_image_convention,
        visualization_enabled=config.num_visualized_episodes > 0,
    )
    dataset, metadata = _make_dataset(selection, policy_config)
    loaded = load_act_checkpoint(manifest, dataset_metadata=metadata)

    total_frames = min(len(dataset), config.max_frames or len(dataset))
    if total_frames <= 0:
        raise ValueError("The selected episodes contain no frames")
    status = "smoke" if config.max_frames is not None else "complete"
    output_dir = create_output_directory(
        policy_path=manifest.path,
        mode="offline",
        output_dir=config.output_dir,
        suite=task.suite,
        task_id=task.task_id,
        overwrite=config.overwrite,
    )
    camera_keys = tuple(metadata.camera_keys)
    visualized_episode_ids = tuple(
        selection.episodes[: config.num_visualized_episodes]
    )
    visualized_episode_set = set(visualized_episode_ids)
    LOGGER.info("[Evaluation] mode: offline teacher-forced prediction")
    LOGGER.info("[Evaluation] selected episodes: %d", len(selection.episodes))
    LOGGER.info("[Evaluation] selected frames: %d", total_frames)
    LOGGER.info(
        "[Evaluation] visualized episodes: %d %s",
        len(visualized_episode_ids),
        list(visualized_episode_ids),
    )
    LOGGER.info("[Evaluation] chunk_size: %d", loaded.config.chunk_size)
    LOGGER.info("[Evaluation] n_action_steps: %d", loaded.config.n_action_steps)
    LOGGER.info("[Evaluation] perturbation: none (teacher-forced observations)")
    LOGGER.info(
        "[Visualization] agentview image convention: %s",
        resolved_agentview_image_convention or "not used",
    )
    LOGGER.info("[Evaluation] status: %s", status)
    LOGGER.info("[Output] directory: %s", output_dir)
    input_config = {
        "task": task.to_dict(),
        "checkpoint_task_assessment": checkpoint_assessment.to_dict(),
        "dataset_repo_id": selection.repo_id,
        "dataset_root": dataset.root.resolve(),
        "dataset_source": selection.source,
        "episode_namespace": selection.episode_namespace,
        "source_manifest": selection.source_manifest,
        "agentview_image_convention": resolved_agentview_image_convention,
        "fps": float(dataset.fps),
        "task_index": selection.task_index,
        "episodes": list(selection.episodes),
        "visualized_episodes": list(visualized_episode_ids),
    }
    arguments = asdict(config)
    run_config = build_run_config(
        status="incomplete",
        mode="offline",
        loaded=loaded,
        input_config=input_config,
        arguments=arguments,
    )
    try:
        write_json_atomic(output_dir / "run_config.json", run_config)
    except BaseException:
        release_output_directory_lock(output_dir)
        raise

    recording_path = output_dir / "prediction.rrd"
    temporary_recording_dir = output_dir / ".rerun_recordings"
    episode_recordings: list[OfflineEpisodeRecording] = []
    episode_recording_by_id: dict[int, OfflineEpisodeRecording] = {}
    rows: list[dict[str, Any]] = []
    predicted_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    episode_indices: list[int] = []
    query_latencies: list[float] = []
    skeleton_episode_index: int | None = None
    completed_skeleton_episodes: set[int] = set()
    skeleton_reconstructor: PandaSkeletonReconstructor | None = None
    agentview_calibration: FixedCameraCalibration | None = None

    try:
        skeleton_reconstructor = _make_skeleton_reconstructor(
            task,
            visualized_episode_ids,
        )
        if skeleton_reconstructor is not None:
            agentview_calibration = skeleton_reconstructor.fixed_camera_calibration(
                "agentview",
                _camera_hwc_shape(metadata, "observation.images.image"),
            )
            LOGGER.info(
                "[Visualization] calibrated fixed camera %s at %dx%d",
                agentview_calibration.camera_name,
                agentview_calibration.image_width,
                agentview_calibration.image_height,
            )
        temporary_recording_dir.mkdir(parents=True, exist_ok=False)
        recording_episode_ids: tuple[int | None, ...] = (
            visualized_episode_ids if visualized_episode_ids else (None,)
        )
        for episode_index in recording_episode_ids:
            filename = (
                "summary.rrd"
                if episode_index is None
                else f"episode_{episode_index}.rrd"
            )
            recording_name = (
                "Offline evaluation summary"
                if episode_index is None
                else f"Dataset episode {episode_index}"
            )
            temporary_path = temporary_recording_dir / filename
            stream = initialize_offline_recording_stream(
                output_path=temporary_path,
                recording_id=str(uuid.uuid4()),
                recording_name=recording_name,
                camera_keys=camera_keys,
                fps=float(dataset.fps),
                agentview_calibration=agentview_calibration,
                agentview_image_convention=resolved_agentview_image_convention,
            )
            episode_recording = OfflineEpisodeRecording(
                episode_index=episode_index,
                path=temporary_path,
                stream=stream,
            )
            episode_recordings.append(episode_recording)
            if episode_index is not None:
                episode_recording_by_id[episode_index] = episode_recording
        selected = Subset(dataset, range(total_frames))
        dataloader = DataLoader(
            selected,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        input_keys = _observation_input_keys(loaded.config)
        for batch_id, batch in enumerate(
            tqdm(dataloader, desc="Offline ACT prediction", unit="batch")
        ):
            missing_inputs = [key for key in input_keys if key not in batch]
            if missing_inputs:
                raise KeyError(f"Dataset batch is missing ACT observations: {missing_inputs}")
            if "action_is_pad" not in batch:
                raise KeyError("Dataset batch is missing action_is_pad")

            raw_target = batch[ACTION].detach().clone()
            valid_horizon = ~batch["action_is_pad"].detach().bool()
            policy_input = {key: batch[key] for key in input_keys}
            processed_input = loaded.preprocessor(policy_input)
            with torch.inference_mode(), _autocast_context(loaded.config):
                predicted_normalized, latency_ms = timed_call(
                    loaded.policy.predict_action_chunk,
                    processed_input,
                    device=loaded.config.device,
                )
            predicted_raw = loaded.postprocessor(predicted_normalized)
            if tuple(predicted_raw.shape) != tuple(raw_target.shape):
                raise ValueError(
                    "Prediction and target chunks must have identical shapes; "
                    f"got {tuple(predicted_raw.shape)} and {tuple(raw_target.shape)}"
                )

            predicted_numpy = predicted_raw.detach().cpu().float().numpy()
            target_numpy = raw_target.detach().cpu().float().numpy()
            valid_numpy = valid_horizon.detach().cpu().numpy()
            actual_batch_size = len(predicted_numpy)
            query_latencies.append(latency_ms)
            predicted_chunks.extend(predicted_numpy)
            target_chunks.extend(target_numpy)
            valid_masks.extend(valid_numpy)

            for batch_row in range(actual_batch_size):
                episode_index = int(_batch_value(batch, "episode_index", batch_row))
                frame_index = int(_batch_value(batch, "frame_index", batch_row))
                model_latency = latency_ms if batch_row == 0 else None
                target_action = target_numpy[batch_row, 0]
                predicted_action = predicted_numpy[batch_row, 0]
                if episode_index in visualized_episode_set:
                    episode_recording = episode_recording_by_id[episode_index]
                    if skeleton_reconstructor is None:
                        raise RuntimeError(
                            "Offline skeleton reconstructor was not initialized"
                        )
                    episode_changed = skeleton_episode_index != episode_index
                    if episode_changed:
                        if episode_index in completed_skeleton_episodes:
                            raise RuntimeError(
                                "Visualized dataset episodes are interleaved; "
                                f"episode {episode_index} appeared more than once"
                            )
                        if skeleton_episode_index is not None:
                            completed_skeleton_episodes.add(skeleton_episode_index)
                        skeleton_episode_index = episode_index
                        skeleton_reconstructor.reset_episode()
                    episode_recording.accept_frame_index(frame_index)
                    raw_observation_state = batch["observation.state"][batch_row]
                    ik_result = skeleton_reconstructor.reconstruct(
                        raw_observation_state.detach().cpu().numpy()
                    )
                    if ik_result.pose is not None:
                        skeleton_pose = ik_result.pose
                        episode_recording.previous_skeleton_pose = skeleton_pose
                        skeleton_stale = False
                    else:
                        skeleton_pose = episode_recording.previous_skeleton_pose
                        skeleton_stale = skeleton_pose is not None
                        LOGGER.warning(
                            "[Visualization] Panda IK failed at episode=%d frame=%d: "
                            "reason=%s position_residual_m=%.6f "
                            "rotation_residual_deg=%.3f; skeleton=%s",
                            episode_index,
                            frame_index,
                            ik_result.reason,
                            ik_result.position_residual_m,
                            np.degrees(ik_result.rotation_residual_rad),
                            "previous pose (stale)" if skeleton_stale else "omitted",
                        )
                    images = {key: batch[key][batch_row] for key in camera_keys}
                    log_offline_frame(
                        episode_recording.stream,
                        global_frame=frame_index,
                        episode_index=episode_index,
                        frame_index=frame_index,
                        batch_id=batch_id,
                        observation_state=raw_observation_state,
                        target_action=target_action,
                        predicted_action=predicted_action,
                        target_chunk=target_numpy[batch_row],
                        predicted_chunk=predicted_numpy[batch_row],
                        valid_horizon=valid_numpy[batch_row],
                        trajectory_history=episode_recording.trajectory_history,
                        command_history=episode_recording.command_history,
                        display_frame=skeleton_reconstructor.display_frame,
                        skeleton_pose=skeleton_pose,
                        skeleton_stale=skeleton_stale,
                        clear_skeleton=episode_changed,
                        model_query_latency_ms=model_latency,
                        images=images,
                        agentview_calibration=agentview_calibration,
                        agentview_image_convention=resolved_agentview_image_convention,
                    )
                rows.append(
                    {
                        "suite": task.suite,
                        "libero_task_id": task.task_id,
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "dataset_index": int(_batch_value(batch, "index", batch_row)),
                        "timestamp_s": float(_batch_value(batch, "timestamp", batch_row)),
                        "task_index": int(_batch_value(batch, "task_index", batch_row)),
                        "batch_id": batch_id,
                        "batch_row": batch_row,
                        "batch_size": actual_batch_size,
                        "target_action": target_action.tolist(),
                        "predicted_action": predicted_action.tolist(),
                        "target_chunk": target_numpy[batch_row].reshape(-1).tolist(),
                        "predicted_chunk": predicted_numpy[batch_row].reshape(-1).tolist(),
                        "valid_horizon": valid_numpy[batch_row].tolist(),
                        "absolute_error": np.abs(predicted_action - target_action).tolist(),
                        "model_query_latency_ms": model_latency,
                    }
                )
                episode_indices.append(episode_index)

        metrics = compute_offline_metrics(
            predicted_chunks=np.stack(predicted_chunks),
            target_chunks=np.stack(target_chunks),
            valid_horizon=np.stack(valid_masks),
            episode_indices=episode_indices,
            batch_size=config.batch_size,
            model_query_latencies_ms=query_latencies,
            status=status,
        )
        metrics["schema_version"] = 2
        metrics["task"] = task.to_dict()
        metrics["checkpoint_task_assessment"] = checkpoint_assessment.to_dict()
        metrics["dataset"] = {
            "repo_id": selection.repo_id,
            "root": selection.root,
            "source": selection.source,
            "task_index": selection.task_index,
            "episode_namespace": selection.episode_namespace,
            "episodes": list(selection.episodes),
            "source_manifest": selection.source_manifest,
        }
        metrics["visualized_episodes"] = list(visualized_episode_ids)
        for episode_recording in episode_recordings:
            log_metric_summary(episode_recording.stream, metrics)
        write_metrics_report(output_dir, metrics)
        write_json_atomic(output_dir / "metrics.json", metrics)
        table = pa.Table.from_pylist(rows, schema=_prediction_schema(loaded.config.chunk_size))
        write_parquet_atomic(output_dir / "predictions.parquet", table)

        flush_and_disconnect_recordings(episode_recordings)
        staging_path = stage_merged_recordings(
            [recording.path for recording in episode_recordings],
            recording_path,
        )
        run_config["status"] = status
        write_json_atomic(output_dir / "run_config.json", run_config)
        staging_path.replace(recording_path)
        shutil.rmtree(temporary_recording_dir)
        mark_run_complete(output_dir)
    finally:
        try:
            flush_and_disconnect_recordings(episode_recordings)
        except BaseException:
            LOGGER.exception("[Output] Failed to close every temporary Rerun recording")
        try:
            if skeleton_reconstructor is not None:
                skeleton_reconstructor.close()
        finally:
            release_output_directory_lock(output_dir)

    viewer_opened = launch_rerun_viewer(recording_path, enabled=config.open_viewer)
    log_terminal_metrics(metrics, LOGGER)
    LOGGER.info("[Output] metrics: %s", output_dir / "metrics.json")
    LOGGER.info("[Output] metrics summary: %s", output_dir / "metrics.md")
    LOGGER.info("[Output] predictions: %s", output_dir / "predictions.parquet")
    LOGGER.info("[Output] Rerun recording: %s", recording_path)
    LOGGER.info("[Output] run config: %s", output_dir / "run_config.json")
    LOGGER.info("[Output] viewer opened: %s", viewer_opened)
    return OfflineEvaluationResult(
        output_dir=output_dir,
        metrics=metrics,
        viewer_opened=viewer_opened,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path",
        required=True,
        type=Path,
        help="Complete local LeRobot pretrained_model directory",
    )
    parser.add_argument(
        "--suite",
        required=True,
        help=f"LIBERO suite ({', '.join(SUPPORTED_SUITES)})",
    )
    parser.add_argument(
        "--task-id",
        required=True,
        type=int,
        help="Zero-based task ID within --suite",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        help="Dataset-native episode indices (default: all task demonstrations)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Exact existing LeRobot dataset override for the selected task",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-visualized-episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a recognized existing --output-dir (requires --output-dir)",
    )
    parser.add_argument("--max-frames", type=int, help="Debug-only global frame bound")
    parser.add_argument(
        "--rebuild-dataset-cache",
        action="store_true",
        help="Rebuild a generated task-local cache (never mutates an explicit root)",
    )
    parser.add_argument(
        "--agentview-image-convention",
        choices=AGENTVIEW_IMAGE_CONVENTIONS,
        default="auto",
        help=(
            "Stored agentview orientation used only for calibrated visualization; "
            "auto requires explicit cache provenance"
        ),
    )
    parser.add_argument(
        "--no-open-viewer",
        action="store_false",
        dest="open_viewer",
        help="Save prediction.rrd without opening the viewer",
    )
    parser.set_defaults(open_viewer=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argument_parser().parse_args(argv)
    if args.episodes is not None:
        args.episodes = tuple(args.episodes)
    evaluate_offline(OfflineEvaluationConfig(**vars(args)))


if __name__ == "__main__":
    main()
