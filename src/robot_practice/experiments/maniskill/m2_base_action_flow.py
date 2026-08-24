"""Generic M2 Action-only Flow training on ManiSkill RGB HDF5 demonstrations.

This entry point is intentionally separate from the frozen M2-W workflow.  It
accepts a direct RGB replay HDF5 with either the official 4D
``pd_ee_delta_pos`` action or the M1-W 7D ``pd_ee_delta_pose`` action, while
keeping the same 18D qpos+qvel context, 16-step Flow chunk and model trunk.

The module only trains from the explicitly configured ``train`` episode split.
It does not perform environment rollout, checkpoint selection, or test
evaluation.  Those are separate protocol decisions and should not be silently
added to an offline training command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from robot_practice.evaluation.common import practice_root, write_json_atomic
from robot_practice.policies.wam_tiny.action_flow import (
    ACTION_KEY,
    BASE_IMAGE_KEY,
    HAND_IMAGE_KEY,
    STATE_KEY,
    ActionFlowConfig,
    ActionOnlyFlowModel,
    NormalizationStats,
    save_action_flow_checkpoint,
)

TASK_ID = "StackCube-v1"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/experiments/maniskill/m2_base_official_rgb_4d.json")
IMAGE_KEYS = (BASE_IMAGE_KEY, HAND_IMAGE_KEY)
H5_RDCC_NBYTES = 8 * 1024 * 1024
H5_RDCC_NSLOTS = 100_003
H5_RDCC_W0 = 0.75
STATS_IMAGE_CHUNK_FRAMES = 256
SUPPORTED_ACTION_DIMS = (4, 7)
EXPECTED_RGB_SHAPE = (128, 128, 3)
EXPECTED_STATE_DIM = 18
EXPECTED_HORIZON = 16
EXPECTED_OBS_STEPS = 2


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
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {target}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = _rooted(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, value)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _rooted(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    target = _rooted(path)
    value = _read_json(target)
    value["_config_path"] = _portable(target)
    return value


def _open_h5_read_only(path: str | Path) -> h5py.File:
    return h5py.File(
        _rooted(path),
        "r",
        rdcc_nbytes=H5_RDCC_NBYTES,
        rdcc_nslots=H5_RDCC_NSLOTS,
        rdcc_w0=H5_RDCC_W0,
    )


def _trajectory_keys(handle: h5py.File) -> list[str]:
    keys = [key for key in handle if key.startswith("traj_")]
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def _episode_group(handle: h5py.File, episode_id: int) -> h5py.Group:
    key = f"traj_{int(episode_id)}"
    if key not in handle:
        raise KeyError(f"HDF5 source is missing {key}")
    return handle[key]


@lru_cache(maxsize=8)
def _available_episode_ids(source_h5: str) -> frozenset[int]:
    """Return source episode IDs once for routes with filtered source data."""

    with _open_h5_read_only(source_h5) as handle:
        return frozenset(
            int(key.split("_")[-1])
            for key in handle
            if key.startswith("traj_") and key.split("_")[-1].isdigit()
        )


def _split_ids(config: Mapping[str, Any], split: str) -> tuple[int, ...]:
    inputs = config.get("inputs", {})
    episode_index_md = inputs.get("episode_index_md")
    episode_index_split = str(inputs.get("episode_index_split", "train"))
    if episode_index_md and split == episode_index_split:
        ids = _read_markdown_episode_ids(episode_index_md, section=f"{split}_episode_ids")
        if not ids:
            raise ValueError(f"Markdown episode index has no {split!r} episode IDs")
        return ids
    manifest = _read_json(config["inputs"]["split_manifest"])
    raw_splits = manifest.get("splits", manifest)
    if not isinstance(raw_splits, Mapping) or split not in raw_splits:
        raise TypeError(f"Split manifest has no {split!r} split")
    value = raw_splits[split]
    if isinstance(value, list):
        ids = tuple(int(item) for item in value)
    elif isinstance(value, Mapping):
        start = int(value["start"])
        stop = int(value["stop"])
        ids = tuple(range(start, stop))
        expected_count = value.get("count")
        if expected_count is not None and len(ids) != int(expected_count):
            raise ValueError(f"Split {split} range/count disagree")
    else:
        raise TypeError(f"Split {split!r} must be a list or range object")
    if bool(config.get("inputs", {}).get("filter_split_to_available_source_episodes", False)):
        available = _available_episode_ids(str(_rooted(config["inputs"]["source_h5"])))
        ids = tuple(episode_id for episode_id in ids if episode_id in available)
    if not ids:
        raise ValueError(f"Split {split!r} is empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Split {split!r} contains duplicate episode ids")
    return ids


def _read_markdown_episode_ids(path: str | Path, *, section: str) -> tuple[int, ...]:
    """Read a strict numeric episode-ID block from a Markdown index.

    The index remains human-readable, while the fenced numeric block gives
    the training route a deterministic machine-readable boundary.  Only the
    requested ``##``/``### <section>`` block is consumed; IDs in other
    sections are never implicitly mixed into a split.
    """

    target = _rooted(path)
    if not target.is_file():
        raise FileNotFoundError(f"M2 base episode index does not exist: {_portable(target)}")
    headings = {f"## {section}", f"### {section}"}
    in_section = False
    in_fence = False
    values: list[int] = []
    for line_number, raw_line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith("#"):
            if line in headings:
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence or not line:
            continue
        if re.fullmatch(r"[0-9]+", line) is None:
            raise ValueError(
                f"Markdown episode index {_portable(target)} has non-numeric ID "
                f"at line {line_number}: {line!r}"
            )
        values.append(int(line))
    if not in_section:
        raise ValueError(
            f"Markdown episode index {_portable(target)} has no section for {section!r}"
        )
    if in_fence:
        raise ValueError(f"Markdown episode index {_portable(target)} has an unclosed code fence")
    if len(set(values)) != len(values):
        raise ValueError(f"Markdown episode index {_portable(target)} contains duplicate episode IDs")
    if not values:
        raise ValueError(f"Markdown episode index {_portable(target)} section {section!r} is empty")
    return tuple(values)


def _expected_action_dim(config: Mapping[str, Any]) -> int:
    action_dim = int(config["protocol"]["action_dim"])
    if action_dim not in SUPPORTED_ACTION_DIMS:
        raise ValueError(f"M2 base supports action_dim=4 or 7, got {action_dim}")
    mode = str(config["protocol"]["control_mode"])
    expected_mode = {4: "pd_ee_delta_pos", 7: "pd_ee_delta_pose"}[action_dim]
    if mode != expected_mode:
        raise ValueError(f"action_dim={action_dim} requires control_mode={expected_mode}")
    return action_dim


def _source_action_dim(config: Mapping[str, Any]) -> int:
    source_action_dim = int(config["protocol"].get("source_action_dim", config["protocol"]["action_dim"]))
    if source_action_dim not in SUPPORTED_ACTION_DIMS:
        raise ValueError(f"M2 base supports source_action_dim=4 or 7, got {source_action_dim}")
    return source_action_dim


def _action_projection_indices(config: Mapping[str, Any]) -> tuple[int, ...]:
    target_action_dim = _expected_action_dim(config)
    source_action_dim = _source_action_dim(config)
    configured = config["protocol"].get("action_projection_indices")
    if configured is None:
        if source_action_dim != target_action_dim:
            raise ValueError(
                "source_action_dim differs from action_dim; "
                "protocol.action_projection_indices is required"
            )
        return tuple(range(source_action_dim))
    indices = tuple(int(value) for value in configured)
    if len(indices) != target_action_dim:
        raise ValueError(
            "protocol.action_projection_indices length must equal target action_dim "
            f"({target_action_dim}), got {len(indices)}"
        )
    if len(set(indices)) != len(indices) or any(index < 0 or index >= source_action_dim for index in indices):
        raise ValueError(
            "protocol.action_projection_indices must contain unique indices within "
            f"[0, {source_action_dim})"
        )
    return indices


def _validate_source_structure(config: Mapping[str, Any]) -> dict[str, Any]:
    source = _rooted(config["inputs"]["source_h5"])
    metadata = _rooted(config["inputs"]["source_metadata"])
    split_manifest = _rooted(config["inputs"]["split_manifest"])
    episode_index_md = config["inputs"].get("episode_index_md")
    required_paths = [
        (source, "source HDF5"),
        (metadata, "source metadata"),
        (split_manifest, "split manifest"),
    ]
    if episode_index_md:
        required_paths.append((_rooted(str(episode_index_md)), "Markdown episode index"))
    for path, label in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"M2 base {label} does not exist: {_portable(path)}")

    action_dim = _expected_action_dim(config)
    source_action_dim = _source_action_dim(config)
    action_projection_indices = _action_projection_indices(config)
    split_ids = {split: _split_ids(config, split) for split in ("train", "development", "test")}
    all_split_ids = [episode_id for values in split_ids.values() for episode_id in values]
    if len(set(all_split_ids)) != len(all_split_ids):
        raise ValueError("M2 base train/development/test episode splits overlap")

    with _open_h5_read_only(source) as handle:
        keys = _trajectory_keys(handle)
        expected_episode_count = int(config["inputs"].get("expected_episode_count", len(keys)))
        if len(keys) != expected_episode_count:
            raise ValueError(
                f"M2 base source has {len(keys)} episodes, expected {expected_episode_count}"
            )
        missing = [episode_id for episode_id in all_split_ids if f"traj_{episode_id}" not in handle]
        if missing:
            raise KeyError(f"M2 base split references missing episodes: {missing[:8]}")
        action_counts: list[int] = []
        for key in keys:
            group = handle[key]
            actions = group["actions"]
            if actions.ndim != 2 or actions.shape[1] != source_action_dim:
                raise ValueError(
                    f"{key}/actions must be [T,{source_action_dim}], got {tuple(actions.shape)}"
                )
            action_count = int(actions.shape[0])
            if action_count <= 0:
                raise ValueError(f"{key} has no actions")
            action_counts.append(action_count)
            for camera in ("base_camera", "hand_camera"):
                rgb = group[f"obs/sensor_data/{camera}/rgb"]
                if rgb.ndim != 4 or tuple(rgb.shape[1:]) != EXPECTED_RGB_SHAPE:
                    raise ValueError(f"{key}/{camera}/rgb has invalid shape {tuple(rgb.shape)}")
                if rgb.shape[0] != action_count + 1 or rgb.dtype != np.uint8:
                    raise ValueError(
                        f"{key}/{camera}/rgb must be uint8 [T+1,128,128,3], got "
                        f"{rgb.dtype}/{tuple(rgb.shape)}"
                    )
            for state_name in ("qpos", "qvel"):
                state = group[f"obs/agent/{state_name}"]
                if state.ndim != 2 or state.shape[0] != action_count + 1:
                    raise ValueError(f"{key}/obs/agent/{state_name} has invalid shape {state.shape}")
            if group["obs/agent/qpos"].shape[1] + group["obs/agent/qvel"].shape[1] != EXPECTED_STATE_DIM:
                raise ValueError(f"{key} qpos+qvel is not 18D")

    metadata_value = _read_json(metadata)
    result = {
        "source_h5": _portable(source),
        "source_metadata": _portable(metadata),
        "split_manifest": _portable(split_manifest),
        "source_h5_sha256": _sha256_file(source),
        "source_metadata_sha256": _sha256_file(metadata),
        "split_manifest_sha256": _sha256_file(split_manifest),
        "episode_count": len(keys),
        "available_episode_count": len(keys),
        "action_dim": action_dim,
        "source_action_dim": source_action_dim,
        "action_projection_indices": list(action_projection_indices),
        "split_counts": {name: len(values) for name, values in split_ids.items()},
        "action_count_total": int(sum(action_counts)),
        "metadata_keys": sorted(str(key) for key in metadata_value),
    }
    if episode_index_md:
        episode_index_path = _rooted(str(episode_index_md))
        result["episode_index_md"] = _portable(episode_index_path)
        result["episode_index_md_sha256"] = _sha256_file(episode_index_path)
        result["episode_index_split"] = str(config["inputs"].get("episode_index_split", "train"))
    return result


class RGBH5WindowDataset(Dataset[dict[str, Tensor]]):
    """Lazy two-frame RGB/state/action windows over one HDF5 split."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        split: str,
        stats: NormalizationStats | None = None,
        max_samples: int | None = None,
    ) -> None:
        if split == "test":
            raise ValueError("M2 base training dataset refuses the test split")
        self.config = config
        self.split = split
        self.path = _rooted(config["inputs"]["source_h5"])
        self.action_dim = _expected_action_dim(config)
        self.source_action_dim = _source_action_dim(config)
        self.action_projection_indices = _action_projection_indices(config)
        episode_ids = _split_ids(config, split)
        self.entries: list[tuple[int, int]] = []
        with _open_h5_read_only(self.path) as handle:
            for episode_id in episode_ids:
                action_count = int(_episode_group(handle, episode_id)["actions"].shape[0])
                self.entries.extend((episode_id, index) for index in range(action_count))
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.entries = self.entries[:max_samples]
        if not self.entries:
            raise ValueError(f"M2 base split {split!r} has no action windows")
        self.stats = stats
        self._handle: h5py.File | None = None
        self.read_audit = {
            "backend": "hdf5_lazy",
            "source": _portable(self.path),
            "split": split,
            "episode_count": len(episode_ids),
            "sample_count": len(self.entries),
            "source_action_dim": self.source_action_dim,
            "action_dim": self.action_dim,
            "action_projection_indices": list(self.action_projection_indices),
            "full_dataset_in_memory": False,
            "worker_local_persistent_handle": True,
            "window_reads": {"observation_steps": EXPECTED_OBS_STEPS, "action_horizon": EXPECTED_HORIZON},
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, RuntimeError, TypeError, ValueError):
                pass

    def _get_handle(self) -> h5py.File:
        if self._handle is None or not self._handle.id.valid:
            self._handle = _open_h5_read_only(self.path)
        return self._handle

    @staticmethod
    def _image_window(dataset: h5py.Dataset, start: int, stop: int) -> Tensor:
        values = np.asarray(dataset[start:stop])
        if values.dtype != np.uint8 or values.ndim != 4 or tuple(values.shape[1:]) != EXPECTED_RGB_SHAPE:
            raise ValueError(f"RGB window has invalid dtype/shape: {values.dtype}/{values.shape}")
        if values.shape[0] == 1:
            values = np.repeat(values, EXPECTED_OBS_STEPS, axis=0)
        if values.shape[0] != EXPECTED_OBS_STEPS:
            raise ValueError(f"RGB window must contain 2 frames, got {values.shape[0]}")
        return torch.from_numpy(values).permute(0, 3, 1, 2).contiguous().float() / 255.0

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        episode_id, action_index = self.entries[index]
        group = _episode_group(self._get_handle(), episode_id)
        action_count = int(group["actions"].shape[0])
        context_start = max(0, action_index - 1)
        context_stop = action_index + 1
        base_images = self._image_window(
            group["obs/sensor_data/base_camera/rgb"], context_start, context_stop
        )
        hand_images = self._image_window(
            group["obs/sensor_data/hand_camera/rgb"], context_start, context_stop
        )
        qpos = np.asarray(group["obs/agent/qpos"][context_start:context_stop], dtype=np.float32)
        qvel = np.asarray(group["obs/agent/qvel"][context_start:context_stop], dtype=np.float32)
        if qpos.shape[0] == 1:
            qpos = np.repeat(qpos, EXPECTED_OBS_STEPS, axis=0)
            qvel = np.repeat(qvel, EXPECTED_OBS_STEPS, axis=0)
        states = torch.from_numpy(np.concatenate((qpos, qvel), axis=-1))

        action_stop = min(action_count, action_index + EXPECTED_HORIZON)
        selected = np.asarray(
            group["actions"][action_index:action_stop], dtype=np.float32
        )[:, self.action_projection_indices]
        padding = np.arange(EXPECTED_HORIZON) >= selected.shape[0]
        if selected.shape[0] < EXPECTED_HORIZON:
            selected = np.concatenate(
                (selected, np.repeat(selected[-1:], EXPECTED_HORIZON - selected.shape[0], axis=0)),
                axis=0,
            )
        result = {
            BASE_IMAGE_KEY: base_images,
            HAND_IMAGE_KEY: hand_images,
            STATE_KEY: states,
            ACTION_KEY: torch.from_numpy(selected),
            "action_is_pad": torch.from_numpy(padding.astype(bool)),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "episode_id": torch.tensor(episode_id, dtype=torch.long),
        }
        if self.stats is not None:
            result[BASE_IMAGE_KEY] = self.stats.normalize_images(result[BASE_IMAGE_KEY])
            result[HAND_IMAGE_KEY] = self.stats.normalize_images(result[HAND_IMAGE_KEY])
            result[STATE_KEY] = self.stats.normalize_state(result[STATE_KEY])
            result[ACTION_KEY] = self.stats.normalize_action(result[ACTION_KEY])
        return result


def _mean_std(total: np.ndarray, squared: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 0.0)
    return mean, np.maximum(np.sqrt(variance), 1e-6)


def compute_train_only_stats(config: Mapping[str, Any], output_path: str | Path) -> NormalizationStats:
    """Compute and cache stats from train episodes only, in bounded RGB chunks."""

    source = _rooted(config["inputs"]["source_h5"])
    action_dim = _expected_action_dim(config)
    action_projection_indices = _action_projection_indices(config)
    image_sum = np.zeros(3, dtype=np.float64)
    image_squared_sum = np.zeros(3, dtype=np.float64)
    state_sum = np.zeros(EXPECTED_STATE_DIM, dtype=np.float64)
    state_squared_sum = np.zeros(EXPECTED_STATE_DIM, dtype=np.float64)
    action_sum = np.zeros(action_dim, dtype=np.float64)
    action_squared_sum = np.zeros(action_dim, dtype=np.float64)
    image_count = 0
    frame_count = 0
    with _open_h5_read_only(source) as handle:
        for episode_id in _split_ids(config, "train"):
            group = _episode_group(handle, episode_id)
            actions = np.asarray(group["actions"], dtype=np.float64)[:, action_projection_indices]
            qpos = np.asarray(group["obs/agent/qpos"][: actions.shape[0]], dtype=np.float64)
            qvel = np.asarray(group["obs/agent/qvel"][: actions.shape[0]], dtype=np.float64)
            states = np.concatenate((qpos, qvel), axis=-1)
            state_sum += states.sum(axis=0)
            state_squared_sum += np.square(states).sum(axis=0)
            action_sum += actions.sum(axis=0)
            action_squared_sum += np.square(actions).sum(axis=0)
            frame_count += actions.shape[0]
            for camera in ("base_camera", "hand_camera"):
                image_dataset = group[f"obs/sensor_data/{camera}/rgb"]
                for start in range(0, actions.shape[0], STATS_IMAGE_CHUNK_FRAMES):
                    stop = min(actions.shape[0], start + STATS_IMAGE_CHUNK_FRAMES)
                    images = np.asarray(image_dataset[start:stop], dtype=np.uint8)
                    image_sum += images.sum(axis=(0, 1, 2), dtype=np.float64) / 255.0
                    squared = np.multiply(images, images, dtype=np.uint32)
                    image_squared_sum += squared.sum(axis=(0, 1, 2), dtype=np.float64) / (255.0**2)
                    image_count += images.shape[0] * images.shape[1] * images.shape[2]
    if frame_count <= 0 or image_count <= 0:
        raise ValueError("Cannot compute M2 base normalization from an empty train split")
    image_mean, image_std = _mean_std(image_sum, image_squared_sum, image_count)
    state_mean, state_std = _mean_std(state_sum, state_squared_sum, frame_count)
    action_mean, action_std = _mean_std(action_sum, action_squared_sum, frame_count)
    stats = NormalizationStats(image_mean, image_std, state_mean, state_std, action_mean, action_std)
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "source_split": "train",
            "source_episode_count": len(_split_ids(config, "train")),
            "source_frame_count": frame_count,
            "source_action_dim": _source_action_dim(config),
            "action_dim": action_dim,
            "action_projection_indices": list(action_projection_indices),
            "stats": stats.to_dict(),
        },
    )
    return stats


def _cosine_warmup_lambda(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return max(float(step) / max(warmup, 1), 1e-8)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_output_target(path: str | Path, *, overwrite: bool) -> Path:
    target = _rooted(path)
    if target.is_symlink():
        raise ValueError(f"M2 base output must not be a symlink: {_portable(target)}")
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"M2 base output is not a directory: {_portable(target)}")
        if not overwrite and any(target.iterdir()):
            raise FileExistsError(f"Refusing existing non-empty M2 base output: {_portable(target)}")
        if overwrite:
            output_root = (practice_root() / "outputs").resolve()
            if not target.resolve().is_relative_to(output_root):
                raise ValueError("--overwrite is restricted to an output directory under outputs/")
    return target


def _prepare_output_dir(path: str | Path, *, overwrite: bool) -> Path:
    target = _validate_output_target(path, overwrite=overwrite)
    if target.exists():
        if overwrite:
            shutil.rmtree(target)
        else:
            return target
    target.mkdir(parents=True, exist_ok=True)
    return target


def _distributed_context(num_gpus: int | None) -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if num_gpus is not None and world_size != num_gpus:
        raise ValueError(f"torchrun world size={world_size} does not match --num-gpus={num_gpus}")
    return rank, local_rank, world_size, world_size > 1


def _model_from_config(config: Mapping[str, Any], device: torch.device) -> ActionOnlyFlowModel:
    model_config = dict(config["model"])
    model_config.update(
        {
            "state_dim": int(config["protocol"]["state_dim"]),
            "action_dim": int(config["protocol"]["action_dim"]),
            "n_obs_steps": int(config["protocol"]["n_obs_steps"]),
            "horizon": int(config["protocol"]["action_horizon"]),
            "flow_solver": config["flow"]["solver"],
            "integration_steps": int(config["flow"]["integration_steps"]),
            "seed": int(config["training"]["seed"]),
            "n_layers": int(model_config.get("layers", 6)),
            "n_heads": int(model_config.get("heads", 8)),
        }
    )
    return ActionOnlyFlowModel(ActionFlowConfig.from_mapping(model_config)).to(device)


def validate_config(
    config: str | Path | Mapping[str, Any] = DEFAULT_CONFIG,
    *,
    num_gpus: int | None = None,
    check_inputs: bool = True,
) -> dict[str, Any]:
    payload = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported M2 base config schema: {payload.get('schema_version')!r}")
    if payload.get("experiment_id") != "m2_base_action_flow":
        raise ValueError("experiment_id must be m2_base_action_flow")
    if payload.get("task") != TASK_ID:
        raise ValueError("M2 base currently supports StackCube-v1 only")
    protocol = payload.get("protocol")
    model = payload.get("model")
    flow = payload.get("flow")
    training = payload.get("training")
    inputs = payload.get("inputs")
    splits = payload.get("splits")
    for name, value in (("protocol", protocol), ("model", model), ("flow", flow), ("training", training), ("inputs", inputs), ("splits", splits)):
        if not isinstance(value, Mapping):
            raise TypeError(f"M2 base config section {name} must be an object")
    if protocol.get("backend") not in ("physx_cpu", "physx_cuda"):
        raise ValueError("protocol.backend must be physx_cpu or physx_cuda")
    action_dim = _expected_action_dim(payload)
    if protocol.get("state_dim") != EXPECTED_STATE_DIM:
        raise ValueError("protocol.state_dim must be 18")
    if protocol.get("n_obs_steps") != EXPECTED_OBS_STEPS or protocol.get("action_horizon") != EXPECTED_HORIZON:
        raise ValueError("M2 base requires n_obs_steps=2 and action_horizon=16")
    if protocol.get("image_shape") != [128, 128, 3]:
        raise ValueError("M2 base RGB shape must be [128,128,3]")
    expected_model = {
        "visual_encoder": "resnet18",
        "pretrained": False,
        "normalization": "groupnorm",
        "spatial_softmax_keypoints": 32,
        "d_model": 256,
        "layers": 6,
        "heads": 8,
        "mlp_ratio": 4,
        "dropout": 0.1,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model.{key} must be {expected!r}")
    if flow.get("path") != "x0_normal_to_x1_action" or flow.get("solver") != "euler" or flow.get("integration_steps") != 10:
        raise ValueError("M2 base Flow contract must be x0->x1 Euler with 10 integration steps")
    expected_training = {
        "global_batch_size": 256,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "warmup_steps": 1000,
        "checkpoint_every": 10000,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    optimizer_steps = int(training.get("optimizer_steps", 0))
    if optimizer_steps not in (50000, 100000):
        raise ValueError("training.optimizer_steps must be 50000 or 100000")
    expected_checkpoints = list(range(10000, optimizer_steps + 1, 10000))
    if training.get("selection_checkpoint_steps") != expected_checkpoints:
        raise ValueError(
            "training.selection_checkpoint_steps must contain every 10000-step checkpoint "
            f"through {optimizer_steps}"
        )
    world_size = int(training.get("world_size", 0))
    per_process = int(training.get("per_process_batch_size", 0))
    if world_size <= 0 or per_process <= 0 or world_size * per_process != 256:
        raise ValueError("training.world_size * per_process_batch_size must equal 256")
    if num_gpus is not None:
        if num_gpus <= 0 or 256 % num_gpus:
            raise ValueError("num_gpus must be a positive divisor of global batch 256")
        if torch.cuda.is_available() and torch.cuda.device_count() < num_gpus:
            raise ValueError(f"requested {num_gpus} GPUs but only {torch.cuda.device_count()} are visible")
    split_counts = {name: len(_split_ids(payload, name)) for name in ("train", "development", "test")}
    expected_counts = splits.get("counts")
    if isinstance(expected_counts, Mapping) and split_counts != {key: int(value) for key, value in expected_counts.items()}:
        raise ValueError(f"split manifest counts {split_counts} disagree with config {expected_counts}")
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": payload["experiment_id"],
        "task": TASK_ID,
        "action_dim": action_dim,
        "source_action_dim": _source_action_dim(payload),
        "action_projection_indices": list(_action_projection_indices(payload)),
        "state_dim": EXPECTED_STATE_DIM,
        "backend": protocol["backend"],
        "global_batch_size": 256,
        "world_size": world_size,
        "per_process_batch_size": per_process,
        "split_counts": split_counts,
        "test_access_allowed": False,
    }
    if check_inputs:
        source_info = _validate_source_structure(payload)
        expected_hash = str(inputs.get("source_h5_sha256", ""))
        if expected_hash and expected_hash != source_info["source_h5_sha256"]:
            raise ValueError(
                f"source HDF5 hash changed: expected {expected_hash}, got {source_info['source_h5_sha256']}"
            )
        expected_split_hash = str(inputs.get("split_manifest_sha256", ""))
        if expected_split_hash and expected_split_hash != source_info["split_manifest_sha256"]:
            raise ValueError("M2 base split manifest hash changed")
        expected_metadata_hash = str(inputs.get("source_metadata_sha256", ""))
        if expected_metadata_hash and expected_metadata_hash != source_info["source_metadata_sha256"]:
            raise ValueError("M2 base source metadata hash changed")
        expected_episode_index_hash = str(inputs.get("episode_index_md_sha256", ""))
        if expected_episode_index_hash and expected_episode_index_hash != source_info.get("episode_index_md_sha256"):
            raise ValueError("M2 base Markdown episode index hash changed")
        result["source"] = source_info
    return result


def _train_impl(
    config: Mapping[str, Any],
    *,
    output_root: str | Path,
    max_steps: int,
    num_gpus: int,
    smoke: bool,
    overwrite: bool,
) -> dict[str, Any]:
    rank, local_rank, world_size, distributed = _distributed_context(num_gpus)
    # Check the output path before any rank joins DDP.  Otherwise rank 0 can
    # fail on a stale output directory after process-group initialization,
    # leaving peer ranks blocked in a collective until the NCCL timeout.
    target = _validate_output_target(output_root, overwrite=overwrite)
    if rank == 0:
        validation = validate_config(config, num_gpus=num_gpus, check_inputs=True)
    else:
        validation = validate_config(config, num_gpus=num_gpus, check_inputs=False)
    if world_size != num_gpus:
        raise ValueError(f"M2 base requires active world_size={num_gpus}, got {world_size}")
    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("M2 base multi-GPU training requires CUDA")
    if rank == 0:
        # Do filesystem mutation before process-group initialization so a
        # permission/race error is reported directly by the launcher.
        _prepare_output_dir(target, overwrite=overwrite)
    if distributed:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _set_seed(int(config["training"]["seed"]) + rank)
    loader_workers = 0 if smoke else int(config["training"].get("num_workers", 0))
    log_stream: Any | None = None
    loss_stream: Any | None = None
    stats: NormalizationStats | None = None
    losses: list[float] = []
    try:
        if rank == 0:
            log_path = target / "training.log"
            loss_path = target / "loss_history.jsonl"
            _write_json(
                target / "run_manifest.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "phase": "smoke" if smoke else "train",
                    "task": TASK_ID,
                    "action_dim": int(config["protocol"]["action_dim"]),
                    "source_action_dim": _source_action_dim(config),
                    "action_projection_indices": list(_action_projection_indices(config)),
                    "control_mode": config["protocol"]["control_mode"],
                    "global_batch_size": 256,
                    "world_size": world_size,
                    "per_process_batch_size": 256 // world_size,
                    "optimizer_steps": max_steps,
                    "config_sha256": _json_sha256({key: value for key, value in config.items() if not key.startswith("_")}),
                    "source": validation.get("source"),
                    "training_episode_count": len(_split_ids(config, "train")),
                    "test_access_count": 0,
                    "data_loader": {
                        "backend": "hdf5_lazy",
                        "source": _portable(config["inputs"]["source_h5"]),
                        "num_workers_per_rank": loader_workers,
                        "persistent_workers": loader_workers > 0,
                        "prefetch_factor": 2 if loader_workers > 0 else None,
                        "worker_local_persistent_handle": True,
                        "full_dataset_in_memory": False,
                    },
                    "logs": {
                        "training_log": _portable(log_path),
                        "loss_history": _portable(loss_path),
                    },
                },
            )
            log_stream = log_path.open("w", encoding="utf-8", buffering=1)
            loss_stream = loss_path.open("w", encoding="utf-8", buffering=1)
            print(
                f"[M2-base] run_start phase={'smoke' if smoke else 'train'} "
                f"action_dim={config['protocol']['action_dim']} steps={max_steps} "
                f"world_size={world_size} global_batch_size=256",
                flush=True,
            )
            log_stream.write(
                f"[M2-base] run_start phase={'smoke' if smoke else 'train'} "
                f"action_dim={config['protocol']['action_dim']} steps={max_steps} "
                f"world_size={world_size} global_batch_size=256\n"
            )
            log_stream.flush()
            stats = compute_train_only_stats(config, target / "normalization_stats.json")
            manifest = _read_json(target / "run_manifest.json")
            manifest["normalization_stats"] = _portable(target / "normalization_stats.json")
            _write_json(target / "run_manifest.json", manifest)
            loss_stream.write(json.dumps({"event": "run_start", "phase": "smoke" if smoke else "train", "max_steps": max_steps}) + "\n")
            loss_stream.flush()
        if distributed:
            torch.distributed.barrier()
        if stats is None:
            payload = _read_json(target / "normalization_stats.json")
            stats = NormalizationStats.from_mapping(payload["stats"])
        dataset = RGBH5WindowDataset(config, split="train", stats=stats)
        sampler = (
            torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=int(config["training"]["seed"]),
            )
            if distributed
            else None
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": 256 // world_size,
            "shuffle": sampler is None,
            "sampler": sampler,
            "num_workers": loader_workers,
            "pin_memory": device.type == "cuda",
            "drop_last": True,
        }
        if loader_workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        loader = DataLoader(dataset, **loader_kwargs)
        model = _model_from_config(config, device)
        train_model: torch.nn.Module = (
            torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
            if distributed
            else model
        )
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            betas=tuple(float(value) for value in config["training"]["betas"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _cosine_warmup_lambda(
                step,
                warmup=int(config["training"]["warmup_steps"]),
                total=int(config["training"]["optimizer_steps"]),
            ),
        )
        iterator = iter(loader)
        started = time.perf_counter()
        checkpoint_every = int(config["training"]["checkpoint_every"])
        checkpoint_steps = set(range(checkpoint_every, max_steps + 1, checkpoint_every))
        checkpoint_steps.update(int(value) for value in config["training"]["selection_checkpoint_steps"])
        checkpoint_steps = {value for value in checkpoint_steps if value <= max_steps}
        if smoke:
            checkpoint_steps = {max_steps}
        for step in range(1, max_steps + 1):
            step_started = time.perf_counter()
            if sampler is not None and (step - 1) % max(len(loader), 1) == 0:
                sampler.set_epoch((step - 1) // max(len(loader), 1))
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            observations = {key: batch[key].to(device, non_blocking=True) for key in IMAGE_KEYS + (STATE_KEY,)}
            clean_actions = batch[ACTION_KEY].to(device, non_blocking=True)
            padding = batch["action_is_pad"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                source_noise = torch.randn_like(clean_actions)
                flow_time = torch.rand(clean_actions.shape[0], device=device, dtype=clean_actions.dtype)
                noisy = (1.0 - flow_time[:, None, None]) * source_noise + flow_time[:, None, None] * clean_actions
                target_velocity = clean_actions - source_noise
                prediction = train_model(observations, noisy, flow_time)
                error = (prediction - target_velocity).square()
                mask = (~padding).unsqueeze(-1).to(dtype=error.dtype)
                loss = (error * mask).sum() / mask.sum().clamp_min(1.0) / clean_actions.shape[-1]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"M2 base produced non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            elapsed = time.perf_counter() - started
            step_seconds = time.perf_counter() - step_started
            eta = (max_steps - step) * elapsed / max(step, 1)
            current_lr = scheduler.get_last_lr()[0]
            if rank == 0:
                record = {
                    "event": "step",
                    "step": step,
                    "loss": loss_value,
                    "lr": current_lr,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "step_seconds": step_seconds,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "global_samples_per_second": step * 256 / max(elapsed, 1e-9),
                }
                loss_stream.write(json.dumps(record, sort_keys=True) + "\n")
                loss_stream.flush()
                if step % (1 if smoke else 100) == 0:
                    line = (
                        f"[M2-base] step={step}/{max_steps} loss={loss_value:.6f} "
                        f"lr={current_lr:.3g} grad_norm={float(grad_norm.detach().cpu()):.3f} "
                        f"step_time={step_seconds:.3f}s eta={eta:.1f}s "
                        f"samples_per_sec={step * 256 / max(elapsed, 1e-9):.1f}"
                    )
                    print(line, flush=True)
                    log_stream.write(line + "\n")
                    log_stream.flush()
            if rank == 0 and step in checkpoint_steps:
                checkpoint_path = target / f"checkpoint_step_{step:08d}.pt"
                save_action_flow_checkpoint(
                    checkpoint_path,
                    model=model,
                    stats=stats,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    metadata={
                        "task": TASK_ID,
                        "action_dim": int(config["protocol"]["action_dim"]),
                        "source_action_dim": _source_action_dim(config),
                        "action_projection_indices": list(_action_projection_indices(config)),
                        "control_mode": config["protocol"]["control_mode"],
                        "config_sha256": _json_sha256({key: value for key, value in config.items() if not key.startswith("_")}),
                        "train_split": "train",
                        "test_access_count": 0,
                    },
                )
                line = f"[M2-base] checkpoint_saved step={step} path={_portable(checkpoint_path)}"
                print(line, flush=True)
                log_stream.write(line + "\n")
                log_stream.flush()
        if rank == 0:
            elapsed = time.perf_counter() - started
            manifest = _read_json(target / "run_manifest.json")
            manifest.update(
                {
                    "status": "complete",
                    "completed_steps": max_steps,
                    "elapsed_seconds": elapsed,
                    "mean_step_seconds": elapsed / max(max_steps, 1),
                    "mean_tail_loss": float(np.mean(losses[-min(len(losses), 100) :])),
                    "last_loss": losses[-1],
                    "checkpoints": [path.name for path in sorted(target.glob("checkpoint_step_*.pt"))],
                    "test_access_count": 0,
                }
            )
            _write_json(target / "run_manifest.json", manifest)
            line = f"[M2-base] training_complete steps={max_steps} elapsed={elapsed:.1f}s last_loss={losses[-1]:.6f}"
            print(line, flush=True)
            log_stream.write(line + "\n")
            log_stream.flush()
            return manifest
        return {"status": "complete", "rank": rank, "completed_steps": max_steps}
    except BaseException as exc:
        if rank == 0 and target.is_dir():
            try:
                manifest = _read_json(target / "run_manifest.json")
                manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "completed_steps": len(losses)})
                _write_json(target / "run_manifest.json", manifest)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        raise
    finally:
        if distributed and torch.distributed.is_initialized():
            # A failed rank must not be followed by another unconditional
            # barrier while the remaining ranks unwind.
            try:
                torch.distributed.destroy_process_group()
            except RuntimeError:
                pass
        if loss_stream is not None:
            loss_stream.close()
        if log_stream is not None:
            log_stream.close()


def train(config_path: str | Path, output_root: str | Path, *, num_gpus: int, overwrite: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    if num_gpus != int(config["training"]["world_size"]):
        raise ValueError(
            f"M2 base config requires --num-gpus={config['training']['world_size']}, got {num_gpus}"
        )
    return _train_impl(
        config,
        output_root=output_root,
        max_steps=int(config["training"]["optimizer_steps"]),
        num_gpus=num_gpus,
        smoke=False,
        overwrite=overwrite,
    )


def smoke(config_path: str | Path, output_root: str | Path, *, num_gpus: int = 1, overwrite: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    return _train_impl(config, output_root=output_root, max_steps=2, num_gpus=num_gpus, smoke=True, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--num-gpus", type=int, default=None)
    validate.add_argument("--skip-input-check", action="store_true")
    for name in ("smoke", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--num-gpus", type=int, default=1 if name == "smoke" else None, required=name == "train")
        command.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        result = validate_config(args.config, num_gpus=args.num_gpus, check_inputs=not args.skip_input_check)
    elif args.command == "smoke":
        result = smoke(args.config, args.output_root, num_gpus=args.num_gpus, overwrite=args.overwrite)
    else:
        result = train(args.config, args.output_root, num_gpus=args.num_gpus, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
