"""Prepare a clean LeRobot RGB dataset and native Diffusion Policy config for M2.

The original StackCube LeRobot dataset is intentionally left untouched.  This
adapter creates a second dataset containing only the frozen M2 train and
development splits.  Camera observations use native LeRobot ``video`` features
with H.264 backing so Arrow only stores frame metadata and images are decoded
lazily by the regular LeRobot dataset path.  The source frames are validated as
RGB uint8 HWC, and the decoder output used by the policy is validated as RGB
float32 CHW in [0, 1].  The generated training config is a normal LeRobot
``TrainPipelineConfig`` and is executed by the unmodified
``lerobot.scripts.lerobot_train`` entry point.

The adapter never opens an offline test episode.  It only reads the train and
development episode IDs from the frozen split manifest and records an empty
test selection in its own manifest.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import practice_root, write_json_atomic
from robot_practice.experiments.maniskill import convert_stackcube, m2_w_action_flow

LOGGER = logging.getLogger(__name__)

TASK_ID = "StackCube-v1"
SCHEMA_VERSION = 1
ADAPTER_VERSION = 2
DATASET_STORAGE = "video"
DEFAULT_CONFIG = Path("configs/experiments/maniskill/m2_diffusion_policy.json")
DEFAULT_M2_CONFIG = Path("configs/experiments/maniskill/m2_w_action_only_flow.json")
DEFAULT_DATASET_ROOT = Path("datasets/lerobot/maniskill/stackcube_dcore_7d_dp_rgb_video")
DEFAULT_REPO_ID = "local/maniskill_stackcube_dcore_7d_dp_rgb_video"
DEFAULT_OUTPUT_DIR = Path("outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000")
DEFAULT_JOB_NAME = "m2_stackcube_diffusion_seed1000"
DEFAULT_TRAIN_CONFIG = Path(
    "outputs/maniskill/m2_diffusion/configs/m2_stackcube_diffusion_seed1000.train_config.json"
)
DATASET_MANIFEST_NAME = "m2_diffusion_dataset_manifest.json"
FEATURE_KEYS = (
    "observation.images.base_camera",
    "observation.images.hand_camera",
    "observation.state",
    "action",
)
IMAGE_KEYS = FEATURE_KEYS[:2]
STATE_KEY = FEATURE_KEYS[2]
ACTION_KEY = FEATURE_KEYS[3]


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _portable(path: str | Path) -> str:
    value = Path(path).expanduser().resolve()
    try:
        return value.relative_to(practice_root()).as_posix()
    except ValueError:
        return value.as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    target = _rooted(path)
    with target.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {target}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(_rooted(path), value)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _rooted(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    target = _rooted(path)
    payload = _read_json(target)
    payload["_config_path"] = _portable(target)
    return payload


def _without_private_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def validate_adapter_config(
    config: Mapping[str, Any] | str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validate the adapter-only contract without touching the dataset."""

    payload = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported M2 Diffusion config schema: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("experiment_id") != "m2_stackcube_diffusion_policy":
        raise ValueError("Unexpected M2 Diffusion experiment_id")
    if payload.get("task") != TASK_ID:
        raise ValueError("M2 Diffusion adapter is fixed to StackCube-v1")

    dataset = payload.get("dataset")
    protocol = payload.get("protocol")
    policy = payload.get("policy")
    training = payload.get("training")
    if not all(isinstance(item, Mapping) for item in (dataset, protocol, policy, training)):
        raise TypeError("dataset, protocol, policy and training sections are required")
    if dataset.get("storage") != DATASET_STORAGE:
        raise ValueError("M2 Diffusion dataset storage must be LeRobot video")
    if dataset.get("image_dtype") != "uint8" or dataset.get("image_layout") != "HWC":
        raise ValueError("M2 Diffusion images must be uint8 HWC")
    if dataset.get("image_color") != "RGB":
        raise ValueError("M2 Diffusion images must be RGB")
    if dataset.get("image_color_stage") != "decoded_model_input":
        raise ValueError("M2 Diffusion image_color must describe decoded model input")
    if str(dataset.get("video_codec", "")) not in {"h264", "hevc", "libsvtav1", "auto"}:
        raise ValueError("M2 Diffusion video_codec must be a supported LeRobot codec")
    if protocol.get("observation_keys") != [*IMAGE_KEYS, STATE_KEY]:
        raise ValueError("M2 Diffusion observation key order changed")
    if protocol.get("state_dim") != 18 or protocol.get("action_dim") != 7:
        raise ValueError("M2 Diffusion state/action dimensions must be 18/7")
    if protocol.get("image_shape") != [128, 128, 3]:
        raise ValueError("M2 Diffusion image shape must be [128,128,3]")
    if (protocol.get("n_obs_steps"), protocol.get("horizon"), protocol.get("n_action_steps")) != (
        2,
        16,
        8,
    ):
        raise ValueError("M2 Diffusion temporal contract must be 2/16/8")
    if policy.get("type") != "diffusion":
        raise ValueError("M2 Diffusion adapter must use the native diffusion policy")
    if policy.get("normalization_mapping") != {
        "VISUAL": "MEAN_STD",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }:
        raise ValueError("M2 Diffusion requires train-only MEAN_STD normalization")
    for key in ("seed", "global_batch_size", "optimizer_steps", "checkpoint_every", "log_every"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training.get("per_process_batch_size", 0)) <= 0:
        raise ValueError("training.per_process_batch_size must be positive")
    if int(training["world_size"]) * int(training["per_process_batch_size"]) != int(
        training["global_batch_size"]
    ):
        raise ValueError("training world_size * per_process_batch_size must equal global batch")
    return payload


def _load_m2_contract(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    m2_config = m2_w_action_flow.load_config(path)
    validation = m2_w_action_flow.validate_config(m2_config)
    return m2_config, validation


def _split_episode_ids(m2_config: Mapping[str, Any], split_name: str) -> tuple[int, ...]:
    """Read only the allowed train/development source IDs."""

    if split_name == "test":
        raise ValueError("The M2 Diffusion adapter never reads the offline test split")
    manifest = _read_json(m2_config["inputs"]["offline_split_manifest"])
    values = manifest.get(split_name)
    if not isinstance(values, list):
        raise TypeError(f"Offline split {split_name!r} is missing")
    expected = int(m2_config["splits"]["offline"][split_name])
    if len(values) != expected:
        raise ValueError(f"Offline split {split_name} has {len(values)} IDs, expected {expected}")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"Offline split {split_name} contains duplicate source IDs")
    return result


def build_split_selection(m2_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a local train/development episode index without selecting test data."""

    train_ids = _split_episode_ids(m2_config, "train")
    development_ids = _split_episode_ids(m2_config, "development")
    if set(train_ids) & set(development_ids):
        raise ValueError("M2 train and development source episode IDs overlap")
    selected = [("train", episode_id) for episode_id in train_ids]
    selected.extend(("development", episode_id) for episode_id in development_ids)
    local_train = list(range(len(train_ids)))
    local_development = list(range(len(train_ids), len(selected)))
    return {
        "train_source_episode_ids": list(train_ids),
        "development_source_episode_ids": list(development_ids),
        "selected": selected,
        "train_dataset_episode_indices": local_train,
        "development_dataset_episode_indices": local_development,
        "test_dataset_episode_indices": [],
        "test_access_allowed": False,
    }


def _dataset_fingerprint(
    *,
    m2_config: Mapping[str, Any],
    selection: Mapping[str, Any],
    repo_id: str,
    storage: str,
    video_codec: str,
) -> tuple[str, dict[str, Any]]:
    h5_path = _rooted(m2_config["inputs"]["d_core_h5"])
    split_path = _rooted(m2_config["inputs"]["offline_split_manifest"])
    h5_stat = h5_path.stat()
    payload = {
        "adapter_version": ADAPTER_VERSION,
        "task": TASK_ID,
        "repo_id": repo_id,
        "storage": storage,
        "video_codec": video_codec,
        "d_core_h5": _portable(h5_path),
        "d_core_h5_size": h5_stat.st_size,
        "d_core_h5_mtime_ns": h5_stat.st_mtime_ns,
        "d_core_manifest_sha256": str(m2_config["inputs"]["d_core_manifest_sha256"]),
        "offline_split_manifest": _portable(split_path),
        "offline_split_sha256": str(m2_config["inputs"]["offline_split_sha256"]),
        "train_source_episode_ids": selection["train_source_episode_ids"],
        "development_source_episode_ids": selection["development_source_episode_ids"],
    }
    return _json_sha256(payload), payload


def _validate_source_metadata(replay_h5: Path) -> dict[int, dict[str, Any]]:
    metadata = convert_stackcube._read_metadata(replay_h5)
    if metadata.get("env_info", {}).get("env_id") != TASK_ID:
        raise ValueError(f"D_core task must be {TASK_ID!r}")
    env_kwargs = metadata.get("env_info", {}).get("env_kwargs", {})
    if env_kwargs.get("sim_backend") != "physx_cuda":
        raise ValueError("StackCube formal dataset must come from sim_backend=physx_cuda")
    if env_kwargs.get("control_mode") != "pd_ee_delta_pose":
        raise ValueError("StackCube formal dataset must use pd_ee_delta_pose")
    return convert_stackcube._metadata_records(metadata)


def _update_minmax(
    minimum: dict[str, np.ndarray],
    maximum: dict[str, np.ndarray],
    frame: Mapping[str, Any],
) -> None:
    for key in IMAGE_KEYS:
        value = np.asarray(frame[key], dtype=np.float32) / 255.0
        channel_min = value.reshape(-1, 3).min(axis=0)
        channel_max = value.reshape(-1, 3).max(axis=0)
        minimum[key] = np.minimum(minimum[key], channel_min)
        maximum[key] = np.maximum(maximum[key], channel_max)
    for key in (STATE_KEY, ACTION_KEY):
        value = np.asarray(frame[key], dtype=np.float32).reshape(-1)
        minimum[key] = np.minimum(minimum[key], value)
        maximum[key] = np.maximum(maximum[key], value)


def _image_stat_shape(values: Sequence[float]) -> list[list[list[float]]]:
    return [[[float(value)]] for value in values]


def _train_only_stats(
    *,
    m2_config: Mapping[str, Any],
    generated_stats: Mapping[str, Any],
    train_minimum: Mapping[str, np.ndarray],
    train_maximum: Mapping[str, np.ndarray],
    train_frame_count: int,
) -> dict[str, Any]:
    """Replace all policy-relevant stats with M2 train-only values."""

    stats = {str(key): dict(value) for key, value in generated_stats.items()}
    cache_path = m2_w_action_flow._normalization_stats_cache_path(m2_config)
    train_stats, _, cache_path = m2_w_action_flow._load_or_compute_train_only_stats(
        m2_config, cache_path
    )
    means = {
        IMAGE_KEYS[0]: train_stats.image_mean.tolist(),
        IMAGE_KEYS[1]: train_stats.image_mean.tolist(),
        STATE_KEY: train_stats.state_mean.tolist(),
        ACTION_KEY: train_stats.action_mean.tolist(),
    }
    stds = {
        IMAGE_KEYS[0]: train_stats.image_std.tolist(),
        IMAGE_KEYS[1]: train_stats.image_std.tolist(),
        STATE_KEY: train_stats.state_std.tolist(),
        ACTION_KEY: train_stats.action_std.tolist(),
    }
    for key in FEATURE_KEYS:
        if key in IMAGE_KEYS:
            stats[key] = {
                "min": _image_stat_shape(train_minimum[key].tolist()),
                "max": _image_stat_shape(train_maximum[key].tolist()),
                "mean": _image_stat_shape(means[key]),
                "std": _image_stat_shape(stds[key]),
                "count": [int(train_frame_count)],
            }
        else:
            stats[key] = {
                "min": [float(value) for value in train_minimum[key]],
                "max": [float(value) for value in train_maximum[key]],
                "mean": [float(value) for value in means[key]],
                "std": [float(value) for value in stds[key]],
                "count": [int(train_frame_count)],
            }
    stats["_adapter_train_stats"] = {
        "source_split": "train",
        "source_frame_count": int(train_frame_count),
        "source_cache": _portable(cache_path),
    }
    return stats


def _video_features() -> dict[str, dict[str, Any]]:
    features = convert_stackcube.dataset_features()
    for key in IMAGE_KEYS:
        if features[key].get("dtype") != "video":
            raise ValueError(f"StackCube feature {key!r} must be a LeRobot video feature")
    return features


def _publish_directory(temporary: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite adapted dataset directory: "
            f"{_portable(output_dir)}"
        )
    try:
        temporary.rename(output_dir)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    shutil.copytree(temporary, output_dir, symlinks=False)
    shutil.rmtree(temporary)


def _verify_rgb_dataset(
    root: Path,
    *,
    repo_id: str,
    train_indices: Sequence[int],
    development_indices: Sequence[int],
    expected_episode_count: int,
) -> None:
    info = _read_json(root / "meta" / "info.json")
    features = info.get("features", {})
    for key in IMAGE_KEYS:
        feature = features.get(key, {})
        if feature.get("dtype") != "video" or feature.get("shape") != [128, 128, 3]:
            raise ValueError(f"Adapted feature {key!r} is not a LeRobot video feature")
        if feature.get("names") != ["height", "width", "channel"]:
            raise ValueError(f"Adapted feature {key!r} is not HWC")
    if not info.get("video_path"):
        raise ValueError("Adapted RGB dataset must use lazy video-backed images")
    if info.get("total_episodes") != expected_episode_count:
        raise ValueError("Adapted dataset episode count changed during conversion")
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise ValueError("Adapted RGB dataset contains no LeRobot parquet data")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sample_indices = [int(train_indices[0])]
    if development_indices:
        sample_indices.append(int(development_indices[0]))
    delta_timestamps = {
        key: [-0.05, 0.0] for key in (*IMAGE_KEYS, STATE_KEY)
    }
    delta_timestamps[ACTION_KEY] = [0.0] * 16
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=sample_indices,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
    )
    for episode_index in sample_indices:
        for key in IMAGE_KEYS:
            video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
            if not video_path.is_file():
                raise ValueError(f"Adapted video file is missing: {video_path}")
    for index in range(len(dataset)):
        item = dataset[index]
        for key in IMAGE_KEYS:
            value = item[key]
            if tuple(value.shape) != (2, 3, 128, 128):
                raise ValueError(f"Decoded RGB sample {key!r} has shape {tuple(value.shape)}")
            if value.dtype != np.float32 and str(value.dtype) != "torch.float32":
                raise ValueError(f"Decoded RGB sample {key!r} is not float32 after LeRobot loading")
            array = value.detach().cpu().numpy()
            if not bool(np.isfinite(array).all()):
                raise ValueError(f"Decoded RGB sample {key!r} contains non-finite values")
            if float(array.min()) < 0.0 or float(array.max()) > 1.0:
                raise ValueError(f"Decoded RGB sample {key!r} is outside [0,1]")
        if tuple(item[STATE_KEY].shape) != (2, 18):
            raise ValueError("Adapted state sample shape changed")
        if tuple(item[ACTION_KEY].shape) != (16, 7):
            raise ValueError("Adapted action sample shape changed")


def build_adapted_dataset(
    *,
    m2_config: Mapping[str, Any],
    output_dir: Path,
    repo_id: str,
    fingerprint: str,
    fingerprint_payload: Mapping[str, Any],
    selection: Mapping[str, Any],
    video_codec: str,
) -> dict[str, Any]:
    """Materialize train/development episodes as lazy RGB-decoded LeRobot videos."""

    replay_h5 = _rooted(m2_config["inputs"]["d_core_h5"])
    records = _validate_source_metadata(replay_h5)
    selected = list(selection["selected"])
    if not selected:
        raise ValueError("M2 Diffusion selected no episodes")
    for _, episode_id in selected:
        if int(episode_id) not in records:
            raise ValueError(f"D_core metadata is missing source episode {episode_id}")

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite adapted dataset: {_portable(output_dir)}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.incomplete-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing existing incomplete conversion directory: {temporary}")

    minimum = {
        IMAGE_KEYS[0]: np.full(3, np.inf, dtype=np.float64),
        IMAGE_KEYS[1]: np.full(3, np.inf, dtype=np.float64),
        STATE_KEY: np.full(18, np.inf, dtype=np.float64),
        ACTION_KEY: np.full(7, np.inf, dtype=np.float64),
    }
    maximum = {key: np.full_like(value, -np.inf) for key, value in minimum.items()}
    train_frame_count = 0
    dataset = None
    finalized = False
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=temporary,
            robot_type="panda",
            fps=20,
            features=_video_features(),
            use_videos=True,
            image_writer_threads=4,
            vcodec=video_codec,
            encoder_threads=4,
        )
        with h5py.File(replay_h5, "r") as handle:
            for local_index, (split_name, episode_id) in enumerate(selected):
                episode = handle[f"traj_{int(episode_id)}"]
                frame_count = 0
                for frame in convert_stackcube.iter_episode_frames(episode):
                    if split_name == "train":
                        _update_minmax(minimum, maximum, frame)
                        train_frame_count += 1
                    dataset.add_frame(frame)
                    frame_count += 1
                dataset.save_episode(parallel_encoding=True)
                if (local_index + 1) % 25 == 0 or local_index + 1 == len(selected):
                    LOGGER.info(
                        "converted RGB StackCube episodes %d/%d",
                        local_index + 1,
                        len(selected),
                    )
                if frame_count <= 0:
                    raise ValueError(f"D_core episode {episode_id} contains no actions")
        dataset.finalize()
        finalized = True

        generated_stats = _read_json(temporary / "meta" / "stats.json")
        if not all(np.isfinite(value).all() for value in minimum.values()):
            raise ValueError("Train-only RGB/state/action min statistics are not finite")
        train_stats = _train_only_stats(
            m2_config=m2_config,
            generated_stats=generated_stats,
            train_minimum=minimum,
            train_maximum=maximum,
            train_frame_count=train_frame_count,
        )
        train_stats.pop("_adapter_train_stats", None)
        _write_json(temporary / "meta" / "stats.json", train_stats)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "adapter_version": ADAPTER_VERSION,
            "status": "complete",
            "task": TASK_ID,
            "repo_id": repo_id,
            "storage": DATASET_STORAGE,
            "image_contract": {
                "dtype": "uint8",
                "layout": "HWC",
                "color": "RGB",
                "shape": [128, 128, 3],
                "backing": "video",
                "codec": video_codec,
                "decoded_dtype": "float32",
                "decoded_layout": "CHW",
                "decoded_range": [0.0, 1.0],
                "decoded_color": "RGB",
            },
            "source": {
                "d_core_h5": _portable(replay_h5),
                "d_core_manifest_sha256": str(m2_config["inputs"]["d_core_manifest_sha256"]),
                "offline_split_sha256": str(m2_config["inputs"]["offline_split_sha256"]),
                "test_data_read": False,
            },
            "splits": {
                "train": {
                    "dataset_episode_indices": selection["train_dataset_episode_indices"],
                    "source_episode_count": len(selection["train_source_episode_ids"]),
                },
                "development": {
                    "dataset_episode_indices": selection["development_dataset_episode_indices"],
                    "source_episode_count": len(selection["development_source_episode_ids"]),
                },
                "test": {"dataset_episode_indices": [], "access_allowed": False},
            },
            "counts": {
                "episodes": len(selected),
                "train_episodes": len(selection["train_dataset_episode_indices"]),
                "development_episodes": len(selection["development_dataset_episode_indices"]),
                "train_frames": train_frame_count,
            },
            "normalization": {
                "source_split": "train",
                "mapping": {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                "stats_sha256": _sha256_file(temporary / "meta" / "stats.json"),
            },
            "fingerprint": fingerprint,
            "fingerprint_payload": dict(fingerprint_payload),
        }
        _write_json(temporary / DATASET_MANIFEST_NAME, manifest)
        _verify_rgb_dataset(
            temporary,
            repo_id=repo_id,
            train_indices=selection["train_dataset_episode_indices"],
            development_indices=selection["development_dataset_episode_indices"],
            expected_episode_count=len(selected),
        )
        _publish_directory(temporary, output_dir)
    except BaseException:
        if dataset is not None and not finalized:
            try:
                dataset.finalize()
            except BaseException:
                LOGGER.exception("Could not finalize incomplete M2 Diffusion dataset")
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return _read_json(output_dir / DATASET_MANIFEST_NAME)


def _load_or_build_dataset(
    *,
    m2_config: Mapping[str, Any],
    dataset_root: Path,
    repo_id: str,
    selection: Mapping[str, Any],
    video_codec: str,
) -> dict[str, Any]:
    fingerprint, fingerprint_payload = _dataset_fingerprint(
        m2_config=m2_config,
        selection=selection,
        repo_id=repo_id,
        storage=DATASET_STORAGE,
        video_codec=video_codec,
    )
    manifest_path = dataset_root / DATASET_MANIFEST_NAME
    if dataset_root.exists() or dataset_root.is_symlink():
        if not manifest_path.is_file():
            raise FileExistsError(
                "Refusing existing dataset without an M2 Diffusion manifest: "
                f"{_portable(dataset_root)}"
            )
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("fingerprint") != fingerprint:
            raise FileExistsError(
                "Existing adapted dataset does not match the frozen source: "
                f"{_portable(dataset_root)}"
            )
        _verify_rgb_dataset(
            dataset_root,
            repo_id=str(manifest.get("repo_id", DEFAULT_REPO_ID)),
            train_indices=selection["train_dataset_episode_indices"],
            development_indices=selection["development_dataset_episode_indices"],
            expected_episode_count=len(selection["selected"]),
        )
        return manifest
    return build_adapted_dataset(
        m2_config=m2_config,
        output_dir=dataset_root,
        repo_id=repo_id,
        fingerprint=fingerprint,
        fingerprint_payload=fingerprint_payload,
        selection=selection,
        video_codec=video_codec,
    )


def _build_native_train_config(
    *,
    adapter_config: Mapping[str, Any],
    dataset_root: Path,
    repo_id: str,
    train_episode_indices: Sequence[int],
    output_dir: Path,
    job_name: str,
    batch_size: int,
    steps: int,
    save_freq: int,
    log_freq: int,
    seed: int,
    num_workers: int,
) -> dict[str, Any]:
    """Serialize a regular LeRobot TrainPipelineConfig."""

    from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import NormalizationMode
    from lerobot.datasets.transforms import ImageTransformsConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    dataset_cfg = DatasetConfig(
        repo_id=repo_id,
        root=_portable(dataset_root),
        episodes=[int(index) for index in train_episode_indices],
        image_transforms=ImageTransformsConfig(enable=False),
        use_imagenet_stats=False,
        video_backend=str(adapter_config["dataset"].get("video_backend", "torchcodec")),
        streaming=False,
    )
    policy_values = dict(adapter_config["policy"])
    policy = DiffusionConfig(
        n_obs_steps=int(adapter_config["protocol"]["n_obs_steps"]),
        input_features={},
        output_features={},
        device="cuda",
        use_amp=False,
        push_to_hub=False,
        horizon=int(adapter_config["protocol"]["horizon"]),
        n_action_steps=int(adapter_config["protocol"]["n_action_steps"]),
        normalization_mapping={
            key: NormalizationMode(value)
            for key, value in policy_values["normalization_mapping"].items()
        },
        vision_backbone=str(policy_values["vision_backbone"]),
        pretrained_backbone_weights=policy_values.get("pretrained_backbone_weights"),
        use_group_norm=bool(policy_values["use_group_norm"]),
        spatial_softmax_num_keypoints=int(policy_values["spatial_softmax_num_keypoints"]),
        use_separate_rgb_encoder_per_camera=bool(
            policy_values["use_separate_rgb_encoder_per_camera"]
        ),
        down_dims=tuple(int(value) for value in policy_values["down_dims"]),
        kernel_size=int(policy_values["kernel_size"]),
        n_groups=int(policy_values["n_groups"]),
        diffusion_step_embed_dim=int(policy_values["diffusion_step_embed_dim"]),
        use_film_scale_modulation=bool(policy_values["use_film_scale_modulation"]),
        noise_scheduler_type=str(policy_values["noise_scheduler_type"]),
        num_train_timesteps=int(policy_values["num_train_timesteps"]),
        beta_schedule=str(policy_values["beta_schedule"]),
        prediction_type=str(policy_values["prediction_type"]),
        clip_sample=bool(policy_values["clip_sample"]),
        clip_sample_range=float(policy_values["clip_sample_range"]),
        do_mask_loss_for_padding=bool(policy_values["do_mask_loss_for_padding"]),
        optimizer_lr=float(adapter_config["training"]["optimizer_lr"]),
        optimizer_betas=tuple(
            float(value) for value in adapter_config["training"]["optimizer_betas"]
        ),
        optimizer_eps=float(adapter_config["training"]["optimizer_eps"]),
        optimizer_weight_decay=float(adapter_config["training"]["optimizer_weight_decay"]),
        scheduler_warmup_steps=int(adapter_config["training"]["scheduler_warmup_steps"]),
    )
    # DiffusionConfig auto-selects CPU when preparation runs on a CPU-only host.
    # Keep the serialized intent as CUDA; the actual launch script checks GPUs
    # before invoking the native trainer.
    policy.device = "cuda"
    train_config = TrainPipelineConfig(
        dataset=dataset_cfg,
        env=None,
        policy=policy,
        output_dir=Path(_portable(output_dir)),
        job_name=job_name,
        resume=False,
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        steps=steps,
        eval_freq=0,
        log_freq=log_freq,
        save_checkpoint=True,
        save_freq=save_freq,
        use_policy_training_preset=True,
        eval=EvalConfig(n_episodes=1, batch_size=1),
        wandb=WandBConfig(enable=False),
    )
    return train_config.to_dict()


def prepare(
    *,
    config: str | Path = DEFAULT_CONFIG,
    m2_config: str | Path = DEFAULT_M2_CONFIG,
    dataset_root: str | Path | None = None,
    repo_id: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    job_name: str = DEFAULT_JOB_NAME,
    train_config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    batch_size: int = 32,
    steps: int = 50000,
    save_freq: int = 10000,
    log_freq: int = 200,
    seed: int = 1000,
    num_workers: int = 4,
    num_gpus: int | None = None,
) -> dict[str, Any]:
    adapter_config = validate_adapter_config(config)
    if min(batch_size, steps, save_freq, log_freq) <= 0:
        raise ValueError("batch_size, steps, save_freq and log_freq must be positive")
    if seed < 0 or num_workers < 0:
        raise ValueError("seed and num_workers must be non-negative")
    if num_gpus is not None and num_gpus <= 0:
        raise ValueError("num_gpus must be positive")

    m2_payload, m2_validation = _load_m2_contract(m2_config)
    source_root = _rooted(adapter_config["dataset"]["source_root"])
    frozen_source_root = _rooted(m2_payload["inputs"]["dataset_root"])
    if source_root.resolve() != frozen_source_root.resolve():
        raise ValueError("Adapter source_root does not match the frozen M2 LeRobot source root")

    adapted_root = _rooted(dataset_root or adapter_config["dataset"]["root"])
    if adapted_root.resolve() == frozen_source_root.resolve():
        raise ValueError("Adapted dataset root must be different from the original dataset root")
    adapted_repo_id = repo_id or str(adapter_config["dataset"]["repo_id"])
    selection = build_split_selection(m2_payload)
    dataset_manifest = _load_or_build_dataset(
        m2_config=m2_payload,
        dataset_root=adapted_root,
        repo_id=adapted_repo_id,
        selection=selection,
        video_codec=str(adapter_config["dataset"]["video_codec"]),
    )

    native_config = _build_native_train_config(
        adapter_config=adapter_config,
        dataset_root=adapted_root,
        repo_id=adapted_repo_id,
        train_episode_indices=selection["train_dataset_episode_indices"],
        output_dir=_rooted(output_dir),
        job_name=job_name,
        batch_size=batch_size,
        steps=steps,
        save_freq=save_freq,
        log_freq=log_freq,
        seed=seed,
        num_workers=num_workers,
    )
    native_path = _rooted(train_config_path)
    _write_json(native_path, native_config)
    adapter_manifest = {
        "schema_version": SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "task": TASK_ID,
        "native_train_config": _portable(native_path),
        "native_entrypoint": "lerobot.scripts.lerobot_train",
        "dataset_manifest": _portable(adapted_root / DATASET_MANIFEST_NAME),
        "dataset_fingerprint": dataset_manifest["fingerprint"],
        "m2_contract_validation": {
            "test_access_allowed": m2_validation["test_access_allowed"],
            "global_batch_size": m2_validation["global_batch_size"],
        },
        "runtime": {
            "num_gpus": num_gpus,
            "batch_size_per_process": batch_size,
            "effective_batch_size": batch_size * num_gpus if num_gpus else None,
            "steps": steps,
            "mixed_precision": "accelerate_cli",
        },
    }
    _write_json(native_path.with_suffix(".adapter_manifest.json"), adapter_manifest)
    return {
        "dataset_root": _portable(adapted_root),
        "dataset_repo_id": adapted_repo_id,
        "dataset_manifest": _portable(adapted_root / DATASET_MANIFEST_NAME),
        "train_config": _portable(native_path),
        "train_episode_count": len(selection["train_dataset_episode_indices"]),
        "development_episode_count": len(selection["development_dataset_episode_indices"]),
        "test_episode_count": 0,
        "test_access_count": 0,
        "native_entrypoint": "lerobot.scripts.lerobot_train",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare", help="materialize the RGB LeRobot dataset and native train config"
    )
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--m2-config", type=Path, default=DEFAULT_M2_CONFIG)
    prepare_parser.add_argument("--dataset-root", type=Path, default=None)
    prepare_parser.add_argument("--repo-id", default=None)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    prepare_parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    prepare_parser.add_argument("--batch-size", type=int, default=32)
    prepare_parser.add_argument("--steps", type=int, default=50000)
    prepare_parser.add_argument("--save-freq", type=int, default=10000)
    prepare_parser.add_argument("--log-freq", type=int, default=200)
    prepare_parser.add_argument("--seed", type=int, default=1000)
    prepare_parser.add_argument("--num-workers", type=int, default=4)
    prepare_parser.add_argument("--num-gpus", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if arguments.command == "prepare":
        result = prepare(
            config=arguments.config,
            m2_config=arguments.m2_config,
            dataset_root=arguments.dataset_root,
            repo_id=arguments.repo_id,
            output_dir=arguments.output_dir,
            job_name=arguments.job_name,
            train_config_path=arguments.train_config,
            batch_size=arguments.batch_size,
            steps=arguments.steps,
            save_freq=arguments.save_freq,
            log_freq=arguments.log_freq,
            seed=arguments.seed,
            num_workers=arguments.num_workers,
            num_gpus=arguments.num_gpus,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
