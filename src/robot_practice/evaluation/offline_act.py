"""Teacher-forced ACT prediction on the local five-episode LIBERO A-Sanity set."""

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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
    practice_root,
    timed_call,
    validate_complete_checkpoint,
    write_json_atomic,
    write_parquet_atomic,
)
from robot_practice.evaluation.metrics import compute_offline_metrics
from robot_practice.evaluation.rerun_visualization import (
    finish_recording,
    initialize_recording,
    log_metric_summary,
    log_offline_frame,
)

LOGGER = logging.getLogger(__name__)

DATASET_REPO_ID = "lerobot/libero"
TASK_INDEX = 29
DEFAULT_EPISODES = (823, 826, 834, 840, 841)


@dataclass(frozen=True)
class OfflineEvaluationConfig:
    policy_path: Path
    episodes: tuple[int, ...] = DEFAULT_EPISODES
    dataset_root: Path = practice_root() / "datasets" / "lerobot" / "lerobot" / "libero"
    batch_size: int = 1
    output_dir: Path | None = None
    max_frames: int | None = None
    open_viewer: bool = True


@dataclass(frozen=True)
class OfflineEvaluationResult:
    output_dir: Path
    metrics: dict[str, Any]
    viewer_opened: bool


def validate_config(config: OfflineEvaluationConfig) -> None:
    if not config.episodes:
        raise ValueError("--episodes must contain at least one episode index")
    if len(set(config.episodes)) != len(config.episodes):
        raise ValueError("--episodes must not contain duplicate indices")
    if any(index < 0 for index in config.episodes):
        raise ValueError("--episodes values must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    dataset_root = config.dataset_root.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot LIBERO dataset is missing at {dataset_root}")


def _load_policy_config(policy_path: Path) -> ACTConfig:
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise TypeError(f"Expected ACTConfig, got {type(config).__name__}")
    return config


def _make_dataset(
    config: OfflineEvaluationConfig,
    policy_config: ACTConfig,
) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    root = config.dataset_root.expanduser().resolve()
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=root)
    for episode in config.episodes:
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
        DATASET_REPO_ID,
        root=root,
        episodes=list(config.episodes),
        delta_timestamps=delta_timestamps,
        download_videos=True,
    )
    task_indices = {int(value) for value in dataset.hf_dataset.unique("task_index")}
    if task_indices != {TASK_INDEX}:
        raise ValueError(
            f"All offline episodes must belong to dataset task index {TASK_INDEX}; "
            f"got {sorted(task_indices)}"
        )
    return dataset, metadata


def _prediction_schema(chunk_size: int) -> pa.Schema:
    schema = pa.schema(
        [
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
            b"schema_version": b"1",
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
    validate_config(config)
    manifest = validate_complete_checkpoint(config.policy_path)
    policy_config = _load_policy_config(manifest.path)
    dataset, metadata = _make_dataset(config, policy_config)
    loaded = load_act_checkpoint(manifest, dataset_metadata=metadata)

    total_frames = min(len(dataset), config.max_frames or len(dataset))
    if total_frames <= 0:
        raise ValueError("The selected episodes contain no frames")
    status = "smoke" if config.max_frames is not None else "complete"
    output_dir = create_output_directory(
        policy_path=manifest.path,
        mode="offline",
        output_dir=config.output_dir,
    )
    camera_keys = tuple(metadata.camera_keys)
    input_config = {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_root": dataset.root.resolve(),
        "fps": float(dataset.fps),
        "task_index": TASK_INDEX,
        "episodes": list(config.episodes),
    }
    arguments = asdict(config)
    run_config = build_run_config(
        status="incomplete",
        mode="offline",
        loaded=loaded,
        input_config=input_config,
        arguments=arguments,
    )
    write_json_atomic(output_dir / "run_config.json", run_config)

    recording_path = output_dir / "prediction.rrd"
    recording_initialized = False
    rows: list[dict[str, Any]] = []
    predicted_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    episode_indices: list[int] = []
    query_latencies: list[float] = []

    try:
        initialize_recording(
            output_path=recording_path,
            mode="offline",
            camera_keys=camera_keys,
            fps=float(dataset.fps),
        )
        recording_initialized = True
        selected = Subset(dataset, range(total_frames))
        dataloader = DataLoader(
            selected,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        input_keys = _observation_input_keys(loaded.config)
        global_frame = 0
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
                images = {key: batch[key][batch_row] for key in camera_keys}
                log_offline_frame(
                    rr,
                    global_frame=global_frame,
                    episode_index=episode_index,
                    frame_index=frame_index,
                    batch_id=batch_id,
                    target_action=target_action,
                    predicted_action=predicted_action,
                    model_query_latency_ms=model_latency,
                    images=images,
                )
                rows.append(
                    {
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
                global_frame += 1

        metrics = compute_offline_metrics(
            predicted_chunks=np.stack(predicted_chunks),
            target_chunks=np.stack(target_chunks),
            valid_horizon=np.stack(valid_masks),
            episode_indices=episode_indices,
            batch_size=config.batch_size,
            model_query_latencies_ms=query_latencies,
            status=status,
        )
        log_metric_summary(rr, metrics)
        write_json_atomic(output_dir / "metrics.json", metrics)
        table = pa.Table.from_pylist(rows, schema=_prediction_schema(loaded.config.chunk_size))
        write_parquet_atomic(output_dir / "predictions.parquet", table)

        finish_recording()
        recording_initialized = False
        if not recording_path.is_file() or recording_path.stat().st_size == 0:
            raise RuntimeError(f"Rerun recording was not written: {recording_path}")
        run_config["status"] = status
        write_json_atomic(output_dir / "run_config.json", run_config)
        mark_run_complete(output_dir)
    finally:
        if recording_initialized:
            finish_recording()

    viewer_opened = launch_rerun_viewer(recording_path, enabled=config.open_viewer)
    LOGGER.info("Offline evaluation output: %s", output_dir)
    log_terminal_metrics(metrics, LOGGER)
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
        "--episodes",
        nargs="+",
        type=int,
        default=list(DEFAULT_EPISODES),
        help="Same-task dataset episode indices",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=OfflineEvaluationConfig.__dataclass_fields__["dataset_root"].default,
        help="Exact local lerobot/libero dataset root",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-frames", type=int, help="Debug-only global frame bound")
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
    args.episodes = tuple(args.episodes)
    evaluate_offline(OfflineEvaluationConfig(**vars(args)))


if __name__ == "__main__":
    main()
