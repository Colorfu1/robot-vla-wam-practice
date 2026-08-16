"""M2-W StackCube Action-only Flow: contract, training, selection and freeze.

The command is deliberately a hard-gated workflow.  Every phase validates the
M1-W frozen protocol and the immutable data hashes before doing work.  The
module never opens the offline test split or the state-bank test split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
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
    load_action_flow_checkpoint,
    save_action_flow_checkpoint,
)

TASK_ID = "StackCube-v1"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/experiments/maniskill/m2_w_action_only_flow.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/maniskill/m2_task_base/StackCube-v1")
FORBIDDEN_KEYS = {
    "goal_pose",
    "object_pose",
    "success",
    "is_grasped",
    "relative_vector",
    "stage_label",
    "privileged",
    "privileged_state",
}
IMAGE_KEYS = (BASE_IMAGE_KEY, HAND_IMAGE_KEY)
NORMALIZATION_CACHE_VERSION = 2
H5_RDCC_NBYTES = 8 * 1024 * 1024
H5_RDCC_NSLOTS = 100_003
H5_RDCC_W0 = 0.75
STATS_IMAGE_CHUNK_FRAMES = 256
WORKER_PROGRESS_TIMEOUT_SECONDS = 300.0


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
    target = _rooted(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, value)


def _write_training_log_line(stream: Any | None, message: str) -> None:
    """Write and flush one human-readable training log line."""

    if stream is None:
        return
    stream.write(message.rstrip("\n") + "\n")
    stream.flush()


def _write_jsonl_record(stream: Any | None, record: Mapping[str, Any]) -> None:
    """Write and flush one structured training record."""

    if stream is None:
        return
    stream.write(
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    stream.flush()


def _emit_training_log(stream: Any | None, message: str) -> None:
    """Mirror a training message to stdout and the persistent run log."""

    print(message, flush=True)
    _write_training_log_line(stream, message)


def _visible_gpu_ids(num_gpus: int) -> list[str]:
    """Return physical GPU identifiers to assign one per evaluator process."""

    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask and mask.strip():
        identifiers = [item.strip() for item in mask.split(",")]
        if any(not item for item in identifiers):
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains an empty GPU identifier")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("M2-W multi-GPU rollout requires CUDA")
        identifiers = [str(index) for index in range(torch.cuda.device_count())]
    if len(identifiers) < num_gpus:
        raise RuntimeError(
            f"requested {num_gpus} evaluator GPUs but only {len(identifiers)} are visible"
        )
    return identifiers[:num_gpus]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _rooted(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    target = _rooted(path)
    payload = _read_json(target)
    payload["_config_path"] = _portable(target)
    return payload


def _walk_mapping_keys(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from _walk_mapping_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping_keys(child, f"{prefix}[{index}]")


def _assert_no_forbidden_observation_keys(config: Mapping[str, Any]) -> None:
    observation = config.get("protocol", {}).get("observation_keys", [])
    forbidden = config.get("protocol", {}).get("forbidden_observation_keys", [])
    if not isinstance(observation, list) or not isinstance(forbidden, list):
        raise TypeError("protocol observation key lists are required")
    lowered = {str(item).lower() for item in observation}
    blocked = {str(item).lower() for item in forbidden} | FORBIDDEN_KEYS
    found = sorted(key for key in lowered if any(item in key for item in blocked))
    if found:
        raise ValueError(f"M2-W observation schema contains privileged key(s): {found}")
    for path, value in _walk_mapping_keys(config.get("model", {})):
        lowered_path = path.lower()
        if any(
            token in lowered_path for token in ("future_target", "future_decoder", "auxiliary_loss")
        ) and value not in (False, 0, None):
            raise ValueError(f"Action-only model cannot enable {path}")


def _verify_hash(root: Path, path_value: str, expected: str, label: str) -> str:
    target = _rooted(path_value)
    if not target.is_file():
        raise FileNotFoundError(f"M2-W {label} does not exist: {_portable(target)}")
    observed = sha256_file(target)
    if observed != expected:
        raise ValueError(f"M2-W {label} hash changed: expected {expected}, got {observed}")
    return observed


def _validate_m1_precondition(config: Mapping[str, Any]) -> dict[str, Any]:
    from robot_practice.experiments.maniskill import m1_protocol_audit

    inputs = config["inputs"]
    m1_config_path = _rooted(inputs["m1_config"])
    m1_config = m1_protocol_audit.load_config(m1_config_path)
    m1_protocol_audit.validate_config(m1_config)
    protocol_path = _rooted(inputs["m1_frozen_protocol"])
    validation = m1_protocol_audit.validate_frozen(protocol_path, m1_config)
    if not validation.get("gate_pass"):
        raise ValueError("M1-W validate-frozen did not return gate_pass=true")
    frozen = _read_json(protocol_path)
    frozen_protocol = frozen.get("frozen_protocol")
    if not isinstance(frozen_protocol, Mapping):
        raise TypeError("M1-W frozen protocol has no frozen_protocol object")
    if frozen_protocol.get("backend") != "physx_cuda":
        raise ValueError("M2-W requires the M1-W physx_cuda frozen protocol")
    if frozen_protocol.get("control_frequency_hz") != 20:
        raise ValueError("M2-W requires the M1-W 20 Hz frozen protocol")
    tasks = frozen_protocol.get("tasks")
    if not isinstance(tasks, Mapping) or tuple(tasks) != (TASK_ID,):
        raise ValueError(f"M2-W requires a StackCube-only M1-W protocol, got {tasks}")
    if tasks[TASK_ID].get("formal_horizon") != 500:
        raise ValueError("M2-W requires formal_horizon=500")
    protocol = m1_config.get("protocol", {})
    task = m1_config.get("tasks", {}).get(TASK_ID, {})
    if protocol.get("native_source_control_mode") != "pd_ee_delta_pose":
        raise ValueError("M1-W native control mode must be pd_ee_delta_pose")
    if task.get("target_control_mode") != "pd_ee_delta_pose" or task.get("action_dim") != 7:
        raise ValueError("M1-W task contract must be native 7D pd_ee_delta_pose")
    return {
        "config": _portable(m1_config_path),
        "protocol": _portable(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "validation": validation,
    }


def validate_config(
    config: Mapping[str, Any] | str | Path = DEFAULT_CONFIG,
    *,
    num_gpus: int | None = None,
    require_batch_divisor: bool = True,
    check_inputs: bool = True,
) -> dict[str, Any]:
    """Validate the complete pre-result M2-W contract.

    ``check_inputs=False`` is intended only for isolated unit tests.  The CLI
    always uses the default hard gate so a formal run cannot start from a
    missing or changed M1/data artifact.
    """

    payload = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported M2-W config schema: {payload.get('schema_version')!r}")
    if payload.get("experiment_id") != "m2_w_action_only_flow":
        raise ValueError("M2-W experiment_id is fixed to m2_w_action_only_flow")
    if payload.get("task") != TASK_ID:
        raise ValueError("M2-W is StackCube-v1 only")

    protocol = payload.get("protocol")
    model = payload.get("model")
    flow = payload.get("flow")
    training = payload.get("training")
    splits = payload.get("splits")
    evaluation = payload.get("evaluation")
    inputs = payload.get("inputs")
    for name, value in (
        ("protocol", protocol),
        ("model", model),
        ("flow", flow),
        ("training", training),
        ("splits", splits),
        ("evaluation", evaluation),
        ("inputs", inputs),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"M2-W config section {name} must be an object")

    expected_protocol = {
        "backend": "physx_cuda",
        "control_mode": "pd_ee_delta_pose",
        "control_frequency_hz": 20,
        "formal_horizon": 500,
        "action_dim": 7,
        "state_dim": 18,
        "n_obs_steps": 2,
        "action_horizon": 16,
        "image_shape": [128, 128, 3],
        "allowed_n_action_steps": [4, 8],
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol.{key} must be {expected!r}, got {protocol.get(key)!r}")
    if protocol.get("action_key") != ACTION_KEY:
        raise ValueError("M2-W action key must be 'action'")
    _assert_no_forbidden_observation_keys(payload)
    if tuple(protocol.get("observation_keys", ())) != (BASE_IMAGE_KEY, HAND_IMAGE_KEY, STATE_KEY):
        raise ValueError("M2-W observation keys/order drifted")

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
        "attention": "noncausal_self_attention",
        "future_target": False,
        "future_decoder": False,
        "auxiliary_loss": False,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model.{key} must be {expected!r}, got {model.get(key)!r}")
    expected_flow = {
        "path": "x0_normal_to_x1_action",
        "time_distribution": "uniform_0_1",
        "solver": "euler",
        "integration_steps": 10,
        "classifier_free_guidance": False,
        "candidate_reranking": False,
        "oracle": False,
    }
    for key, expected in expected_flow.items():
        if flow.get(key) != expected:
            raise ValueError(f"flow.{key} must be {expected!r}, got {flow.get(key)!r}")

    expected_training = {
        "seed": 1000,
        "global_batch_size": 256,
        "gradient_accumulation_steps": 1,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "optimizer_steps": 50000,
        "warmup_steps": 1000,
        "schedule": "cosine_after_linear_warmup",
        "amp_dtype": "bfloat16",
        "checkpoint_every": 10000,
        "selection_checkpoint_steps": [10000, 20000, 30000, 40000, 50000],
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}, got {training.get(key)!r}")
    world_size = int(training.get("world_size", 0))
    per_process = int(training.get("per_process_batch_size", 0))
    if world_size <= 0 or per_process <= 0 or world_size * per_process != 256:
        raise ValueError("training world_size * per_process_batch_size must equal global batch 256")
    if num_gpus is not None:
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        if require_batch_divisor and 256 % num_gpus:
            raise ValueError("num_gpus must be a positive divisor of global batch 256")
        if torch.cuda.is_available() and torch.cuda.device_count() < num_gpus:
            raise ValueError(
                f"requested {num_gpus} GPUs but only {torch.cuda.device_count()} are visible"
            )

    if splits.get("offline") != {"train": 787, "development": 100, "test": 100}:
        raise ValueError("M2-W offline split must be train/development/test=787/100/100")
    if splits.get("state_banks") != {"train": 2000, "development": 200, "test": 400}:
        raise ValueError("M2-W state-bank split must be train/development/test=2000/200/400")
    if splits.get("training_split") != "train" or splits.get("development_split") != "development":
        raise ValueError("M2-W training/development split names are frozen")
    if splits.get("test_split") != "test" or splits.get("test_access_allowed") is not False:
        raise ValueError("M2-W test split must remain inaccessible")
    if evaluation.get("development_state_count") != 200 or evaluation.get("max_steps") != 500:
        raise ValueError("M2-W development evaluation requires 200 exact states and 500 steps")
    if evaluation.get("n_action_steps") != [4, 8]:
        raise ValueError("M2-W n_action_steps candidates must be [4, 8]")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": payload["experiment_id"],
        "task": TASK_ID,
        "global_batch_size": 256,
        "num_gpus": num_gpus,
        "test_access_allowed": False,
    }
    if check_inputs:
        m1 = _validate_m1_precondition(payload)
        expected_m1_hash = str(inputs.get("m1_frozen_protocol_sha256", ""))
        observed_m1_hash = _verify_hash(
            practice_root(), inputs["m1_frozen_protocol"], expected_m1_hash, "M1-W frozen protocol"
        )
        if observed_m1_hash != m1["protocol_sha256"]:
            raise ValueError("M1-W frozen protocol hash was checked inconsistently")
        for path_key, hash_key, label in (
            ("dataset_manifest", "dataset_manifest_sha256", "LeRobot dataset manifest"),
            ("dataset_info", "dataset_info_sha256", "LeRobot dataset info"),
            ("d_core_manifest", "d_core_manifest_sha256", "D_core manifest"),
            ("offline_split_manifest", "offline_split_sha256", "offline split manifest"),
            ("state_bank_manifest", "state_bank_manifest_sha256", "state-bank manifest"),
        ):
            _verify_hash(practice_root(), inputs[path_key], str(inputs[hash_key]), label)

        dataset_root = _rooted(inputs["dataset_root"])
        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"LeRobot dataset root does not exist: {_portable(dataset_root)}"
            )
        dataset_manifest = _read_json(inputs["dataset_manifest"])
        if (
            dataset_manifest.get("task") != TASK_ID
            or dataset_manifest.get("output", {}).get("episodes") != 987
        ):
            raise ValueError(
                "LeRobot manifest does not describe the frozen 987-episode StackCube D_core"
            )
        info = _read_json(inputs["dataset_info"])
        features = info.get("features", {})
        required_features = {
            BASE_IMAGE_KEY: {"shape": [128, 128, 3]},
            HAND_IMAGE_KEY: {"shape": [128, 128, 3]},
            STATE_KEY: {"shape": [18]},
            ACTION_KEY: {"shape": [7]},
        }
        for key, expected in required_features.items():
            if features.get(key, {}).get("shape") != expected["shape"]:
                raise ValueError(f"LeRobot feature {key!r} does not match M2-W")
        if info.get("total_episodes") != 987 or info.get("total_frames") != 106528:
            raise ValueError("LeRobot dataset size changed from 987 episodes/106528 frames")

        dcore = _read_json(inputs["d_core_manifest"])
        if dcore.get("gate_pass") is not True or dcore.get("episode_count") != 987:
            raise ValueError("D_core manifest is not a passing 987-episode manifest")
        split = _read_json(inputs["offline_split_manifest"])
        if split.get("counts") != {"train": 787, "development": 100, "test": 100}:
            raise ValueError("D_core offline split manifest changed")
        bank = _read_json(inputs["state_bank_manifest"])
        if bank.get("gate_pass") is not True or bank.get("sets") != {
            "train": 2000,
            "development": 200,
            "test": 400,
        }:
            raise ValueError("M1-W state-bank manifest is not the frozen 2000/200/400 split")
        pairs = bank.get("cross_split_pairs", [])
        if any(int(pair.get("num_duplicates", -1)) != 0 for pair in pairs):
            raise ValueError("M1-W state-bank cross-split leakage gate failed")
        result.update(
            {
                "m1_frozen_protocol_sha256": observed_m1_hash,
                "dataset_manifest_sha256": sha256_file(inputs["dataset_manifest"]),
                "offline_split_sha256": sha256_file(inputs["offline_split_manifest"]),
                "state_bank_manifest_sha256": sha256_file(inputs["state_bank_manifest"]),
            }
        )
    return result


def _split_episode_ids(config: Mapping[str, Any], split_name: str) -> tuple[int, ...]:
    if split_name == "test":
        raise ValueError("M2-W never reads the offline test split")
    split = _read_json(config["inputs"]["offline_split_manifest"])
    values = split.get(split_name)
    if not isinstance(values, list):
        raise TypeError(f"Offline split {split_name!r} is missing")
    expected = int(config["splits"]["offline"][split_name])
    if len(values) != expected:
        raise ValueError(
            f"Offline split {split_name} has {len(values)} episodes, expected {expected}"
        )
    return tuple(int(value) for value in values)


def _episode_group_id(handle: h5py.File, episode_id: int) -> str:
    direct = f"traj_{episode_id}"
    if direct in handle:
        return direct
    raise KeyError(f"D_core is missing frozen source episode {episode_id}")


def _open_h5_read_only(path: str | Path) -> h5py.File:
    """Open one read-only HDF5 handle with a bounded per-process raw cache."""

    return h5py.File(
        _rooted(path),
        "r",
        rdcc_nbytes=H5_RDCC_NBYTES,
        rdcc_nslots=H5_RDCC_NSLOTS,
        rdcc_w0=H5_RDCC_W0,
    )


class StackCubeWindowDataset(Dataset[dict[str, Tensor]]):
    """Lazy HDF5 windows over one explicitly allowed offline split."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        split: str,
        stats: NormalizationStats | None = None,
        max_samples: int | None = None,
    ) -> None:
        if split == "test":
            raise ValueError("StackCubeWindowDataset refuses the offline test split")
        self.config = config
        self.split = split
        self.path = _rooted(config["inputs"]["d_core_h5"])
        if not self.path.is_file():
            raise FileNotFoundError(f"D_core HDF5 does not exist: {_portable(self.path)}")
        episode_ids = _split_episode_ids(config, split)
        self.entries: list[tuple[int, int]] = []
        with h5py.File(self.path, "r") as handle:
            for episode_id in episode_ids:
                group = handle[_episode_group_id(handle, episode_id)]
                action_count = int(group["actions"].shape[0])
                if action_count <= 0:
                    raise ValueError(f"D_core episode {episode_id} has no actions")
                self.entries.extend((episode_id, index) for index in range(action_count))
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.entries = self.entries[:max_samples]
        if not self.entries:
            raise ValueError(f"D_core split {split!r} has no action windows")
        self.stats = stats
        # HDF5 handles must be opened after DataLoader forks its workers.  The
        # handle is therefore deliberately not opened in __init__.
        self._handle: h5py.File | None = None
        self.read_audit = {
            "split": split,
            "episode_count": len(episode_ids),
            "sample_count": len(self.entries),
            "forbidden_splits": ["test"],
            "backend": "hdf5_lazy",
            "full_dataset_in_memory": False,
            "worker_local_persistent_handle": True,
            "window_reads": {"observation_steps": 2, "action_horizon": 16},
        }

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _image(value: np.ndarray) -> Tensor:
        if value.dtype != np.uint8 or value.shape != (128, 128, 3):
            raise ValueError(
                f"StackCube RGB must be uint8 (128,128,3), got {value.dtype}/{value.shape}"
            )
        return torch.from_numpy(np.asarray(value)).permute(2, 0, 1).contiguous().float() / 255.0

    @staticmethod
    def _image_window(dataset: h5py.Dataset, start: int, stop: int) -> Tensor:
        values = np.asarray(dataset[start:stop])
        if values.dtype != np.uint8 or values.ndim != 4 or values.shape[1:] != (128, 128, 3):
            raise ValueError(
                "StackCube RGB window must be uint8 [N,128,128,3], "
                f"got {values.dtype}/{values.shape}"
            )
        if values.shape[0] == 1:
            values = np.repeat(values, 2, axis=0)
        if values.shape[0] != 2:
            raise ValueError(f"StackCube RGB window must contain 2 frames, got {values.shape[0]}")
        return torch.from_numpy(values).permute(0, 3, 1, 2).contiguous().float() / 255.0

    def _get_handle(self) -> h5py.File:
        handle = self._handle
        if handle is None or not handle.id.valid:
            self._handle = _open_h5_read_only(self.path)
        return self._handle

    def __getstate__(self) -> dict[str, Any]:
        """Never pickle/fork an already-open HDF5 handle into a worker."""

        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, RuntimeError, TypeError, ValueError):
                return

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        episode_id, action_index = self.entries[index]
        handle = self._get_handle()
        group = handle[_episode_group_id(handle, episode_id)]
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
            qpos = np.repeat(qpos, 2, axis=0)
            qvel = np.repeat(qvel, 2, axis=0)
        states = torch.from_numpy(np.concatenate((qpos, qvel), axis=-1))

        action_stop = min(action_count, action_index + 16)
        selected = np.asarray(group["actions"][action_index:action_stop], dtype=np.float32)
        if selected.shape[0] <= 0:
            raise ValueError(f"D_core episode {episode_id} has no action at index {action_index}")
        padding = np.arange(16) >= selected.shape[0]
        if selected.shape[0] < 16:
            selected = np.concatenate(
                (selected, np.repeat(selected[-1:], 16 - selected.shape[0], axis=0)), axis=0
            )
        clean_actions = torch.from_numpy(selected)
        result = {
            BASE_IMAGE_KEY: base_images,
            HAND_IMAGE_KEY: hand_images,
            STATE_KEY: states,
            ACTION_KEY: clean_actions,
            "action_is_pad": torch.from_numpy(padding.astype(bool)),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }
        if self.stats is not None:
            result[BASE_IMAGE_KEY] = self.stats.normalize_images(result[BASE_IMAGE_KEY])
            result[HAND_IMAGE_KEY] = self.stats.normalize_images(result[HAND_IMAGE_KEY])
            result[STATE_KEY] = self.stats.normalize_state(result[STATE_KEY])
            result[ACTION_KEY] = self.stats.normalize_action(result[ACTION_KEY])
        return result


def compute_train_only_stats(
    config: Mapping[str, Any], output_path: str | Path | None = None
) -> NormalizationStats:
    """Compute statistics from the 787 train episodes only."""

    path = _rooted(config["inputs"]["d_core_h5"])
    episode_ids = _split_episode_ids(config, "train")
    image_sum = np.zeros(3, dtype=np.float64)
    image_squared_sum = np.zeros(3, dtype=np.float64)
    state_sum = np.zeros(18, dtype=np.float64)
    state_squared_sum = np.zeros(18, dtype=np.float64)
    action_sum = np.zeros(7, dtype=np.float64)
    action_squared_sum = np.zeros(7, dtype=np.float64)
    image_count = 0
    frame_count = 0
    with _open_h5_read_only(path) as handle:
        for episode_id in episode_ids:
            group = handle[_episode_group_id(handle, episode_id)]
            actions = np.asarray(group["actions"], dtype=np.float64)
            qpos = np.asarray(group["obs/agent/qpos"][: actions.shape[0]], dtype=np.float64)
            qvel = np.asarray(group["obs/agent/qvel"][: actions.shape[0]], dtype=np.float64)
            states = np.concatenate((qpos, qvel), axis=-1)
            for camera in ("base_camera", "hand_camera"):
                image_dataset = group[f"obs/sensor_data/{camera}/rgb"]
                for start in range(0, actions.shape[0], STATS_IMAGE_CHUNK_FRAMES):
                    stop = min(actions.shape[0], start + STATS_IMAGE_CHUNK_FRAMES)
                    # Keep RGB in uint8 while reading and reduce each bounded
                    # chunk directly. This preserves exact integer sums and
                    # avoids materializing a full train split as float64.
                    images = np.asarray(image_dataset[start:stop], dtype=np.uint8)
                    image_sum += images.sum(axis=(0, 1, 2), dtype=np.float64) / 255.0
                    squared = np.multiply(images, images, dtype=np.uint32)
                    image_squared_sum += (
                        squared.sum(axis=(0, 1, 2), dtype=np.float64) / (255.0**2)
                    )
                    image_count += images.shape[0] * images.shape[1] * images.shape[2]
            state_sum += states.sum(axis=0)
            state_squared_sum += np.square(states).sum(axis=0)
            action_sum += actions.sum(axis=0)
            action_squared_sum += np.square(actions).sum(axis=0)
            frame_count += actions.shape[0]
    if image_count <= 0 or frame_count <= 0:
        raise ValueError("Cannot compute M2-W normalization from an empty train split")

    def mean_std(
        total: np.ndarray, squared: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = total / count
        variance = np.maximum(squared / count - np.square(mean), 0.0)
        return mean, np.maximum(np.sqrt(variance), 1e-6)

    image_mean, image_std = mean_std(image_sum, image_squared_sum, image_count)
    state_mean, state_std = mean_std(state_sum, state_squared_sum, frame_count)
    action_mean, action_std = mean_std(action_sum, action_squared_sum, frame_count)
    stats = NormalizationStats(
        image_mean, image_std, state_mean, state_std, action_mean, action_std
    )
    if output_path is not None:
        _write_json(
            output_path,
            {
                "schema_version": NORMALIZATION_CACHE_VERSION,
                "source_split": "train",
                "source_episode_count": len(episode_ids),
                "source_frame_count": frame_count,
                "cache_key": _normalization_stats_cache_key(config),
                "stats": stats.to_dict(),
            },
        )
    return stats


def _normalization_stats_cache_key(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _rooted(config["inputs"]["d_core_h5"])
    metadata = path.stat()
    inputs = config["inputs"]
    return {
        "version": NORMALIZATION_CACHE_VERSION,
        "split": "train",
        "d_core_h5": _portable(path),
        "d_core_h5_size": metadata.st_size,
        "d_core_h5_mtime_ns": metadata.st_mtime_ns,
        "d_core_manifest_sha256": str(inputs.get("d_core_manifest_sha256", "")),
        "offline_split_sha256": str(inputs.get("offline_split_sha256", "")),
        "train_episode_count": int(config["splits"]["offline"]["train"]),
    }


def _normalization_stats_cache_path(config: Mapping[str, Any]) -> Path:
    key = _normalization_stats_cache_key(config)
    digest = json_sha256(key)[:16]
    h5_path = _rooted(config["inputs"]["d_core_h5"])
    filename = f"train_stats_v{NORMALIZATION_CACHE_VERSION}_{digest}.json"
    return h5_path.parent / ".m2_w_normalization_cache" / filename


def _load_or_compute_train_only_stats(
    config: Mapping[str, Any], output_path: str | Path
) -> tuple[NormalizationStats, bool, Path]:
    """Reuse a validated train-only stats cache and materialize it into the run."""

    cache_path = _normalization_stats_cache_path(config)
    expected_key = _normalization_stats_cache_key(config)
    if cache_path.is_file():
        try:
            payload = _read_json(cache_path)
            if (
                payload.get("schema_version") == NORMALIZATION_CACHE_VERSION
                and payload.get("cache_key") == expected_key
            ):
                stats = NormalizationStats.from_mapping(payload["stats"])
                _write_json(output_path, payload)
                return stats, True, cache_path
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    stats = compute_train_only_stats(config, cache_path)
    payload = _read_json(cache_path)
    _write_json(output_path, payload)
    return stats, False, cache_path


def _load_stats(path: str | Path) -> NormalizationStats:
    payload = _read_json(path)
    return NormalizationStats.from_mapping(payload.get("stats", payload))


def _model_from_config(
    config: Mapping[str, Any], device: str | torch.device
) -> ActionOnlyFlowModel:
    model_cfg = dict(config["model"])
    model_cfg.update(
        {
            "state_dim": config["protocol"]["state_dim"],
            "action_dim": config["protocol"]["action_dim"],
            "n_obs_steps": config["protocol"]["n_obs_steps"],
            "horizon": config["protocol"]["action_horizon"],
            "flow_solver": config["flow"]["solver"],
            "integration_steps": config["flow"]["integration_steps"],
            "seed": config["training"]["seed"],
            "n_layers": model_cfg.get("layers", 6),
            "n_heads": model_cfg.get("heads", 8),
        }
    )
    model = ActionOnlyFlowModel(ActionFlowConfig.from_mapping(model_cfg)).to(device)
    return model


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_output_dir(
    path: str | Path,
    *,
    allow_existing_complete: bool = False,
    overwrite: bool = False,
) -> Path:
    target = _rooted(path)
    if target.is_symlink():
        raise ValueError(f"M2-W output path must not be a symbolic link: {_portable(target)}")
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"M2-W output path is not a directory: {_portable(target)}")
        if overwrite:
            output_root = (practice_root() / "outputs").resolve()
            if not target.resolve().is_relative_to(output_root):
                raise ValueError(
                    "M2-W --overwrite is restricted to an output directory under outputs/"
                )
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=False)
            return target
        entries = list(target.iterdir())
        if entries:
            manifest = target / "run_manifest.json"
            complete = False
            if manifest.is_file():
                try:
                    complete = _read_json(manifest).get("status") == "complete"
                except (OSError, json.JSONDecodeError):
                    complete = False
            if not (allow_existing_complete and complete):
                raise FileExistsError(
                    f"Refusing existing or incomplete M2-W output directory: {_portable(target)}"
                )
    else:
        target.mkdir(parents=True, exist_ok=False)
    return target


def _cosine_warmup_lambda(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return max(float(step) / max(warmup, 1), 1e-8)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _distributed_context(num_gpus: int | None) -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if num_gpus is not None and world_size != num_gpus:
        raise ValueError(f"torchrun world size={world_size} does not match --num-gpus={num_gpus}")
    distributed = world_size > 1
    return rank, local_rank, world_size, distributed


def _init_distributed(world_size: int) -> None:
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )


def _finish_distributed(distributed: bool) -> None:
    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _train_impl(
    config: Mapping[str, Any],
    *,
    output_root: str | Path,
    max_steps: int,
    num_gpus: int | None,
    smoke: bool,
    overwrite: bool,
) -> dict[str, Any]:
    rank, local_rank, world_size, distributed = _distributed_context(num_gpus)
    if num_gpus is not None:
        validate_config(config, num_gpus=num_gpus)
    else:
        validate_config(config)
    if world_size <= 0 or 256 % world_size:
        raise ValueError("The active world size must divide global batch 256")
    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("M2-W formal distributed training requires CUDA")
    _init_distributed(world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _set_seed(int(config["training"]["seed"]))
    startup_started = time.perf_counter()
    target = _rooted(output_root)
    loader_workers = 0 if smoke else int(config["training"].get("num_workers", 0))
    persistent_workers = loader_workers > 0
    stats_cache_path = _normalization_stats_cache_path(config)
    stats_cache_hit = False
    training_log_stream: Any | None = None
    loss_history_stream: Any | None = None
    if rank == 0:
        _prepare_output_dir(target, overwrite=overwrite)
        training_log_path = target / "training.log"
        loss_history_path = target / "loss_history.jsonl"
        _write_json(
            target / "run_manifest.json",
            {
                "schema_version": 1,
                "status": "running",
                "phase": "smoke" if smoke else "train",
                "overwrite": overwrite,
                "task": TASK_ID,
                "global_batch_size": 256,
                "world_size": world_size,
                "per_process_batch_size": 256 // world_size,
                "optimizer_steps": max_steps,
                "config_sha256": json_sha256(
                    {key: value for key, value in config.items() if not key.startswith("_")}
                ),
                "m1_frozen_protocol_sha256": sha256_file(config["inputs"]["m1_frozen_protocol"]),
                "test_access_count": 0,
                "data_loader": {
                    "backend": "hdf5_lazy",
                    "source": _portable(config["inputs"]["d_core_h5"]),
                    "num_workers_per_rank": loader_workers,
                    "persistent_workers": persistent_workers,
                    "prefetch_factor": 2 if persistent_workers else None,
                    "worker_local_persistent_handle": True,
                    "full_dataset_in_memory": False,
                },
                "normalization_stats_cache": _portable(stats_cache_path),
                "logs": {
                    "training_log": _portable(training_log_path),
                    "loss_history": _portable(loss_history_path),
                    "loss_record_interval_steps": 1,
                },
            },
        )
        training_log_stream = training_log_path.open("w", encoding="utf-8", buffering=1)
        loss_history_stream = loss_history_path.open("w", encoding="utf-8", buffering=1)
        _emit_training_log(
            training_log_stream,
            f"[M2-W] run_start phase={'smoke' if smoke else 'train'} "
            f"max_steps={max_steps} world_size={world_size} global_batch_size=256",
        )
        _write_jsonl_record(
            loss_history_stream,
            {
                "event": "run_start",
                "phase": "smoke" if smoke else "train",
                "max_steps": max_steps,
                "world_size": world_size,
                "global_batch_size": 256,
            },
        )
        stats, stats_cache_hit, _ = _load_or_compute_train_only_stats(
            config, target / "normalization_stats.json"
        )
        _emit_training_log(
            training_log_stream,
            f"[M2-W] train normalization stats cache={'hit' if stats_cache_hit else 'miss'} "
            f"path={_portable(stats_cache_path)}",
        )
    if distributed:
        torch.distributed.barrier()
    stats = _load_stats(target / "normalization_stats.json")
    if rank == 0:
        manifest = _read_json(target / "run_manifest.json")
        manifest["normalization_stats_cache_hit"] = stats_cache_hit
        _write_json(target / "run_manifest.json", manifest)
    dataset = StackCubeWindowDataset(config, split="train", stats=stats, max_samples=None)
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
    if persistent_workers:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **loader_kwargs)
    model = _model_from_config(config, device)
    train_model = (
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
    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.bfloat16
    iterator = iter(loader)
    startup_seconds = time.perf_counter() - startup_started
    if rank == 0:
        manifest = _read_json(target / "run_manifest.json")
        manifest["startup_seconds"] = startup_seconds
        _write_json(target / "run_manifest.json", manifest)
        _emit_training_log(
            training_log_stream,
            f"[M2-W] data/model ready startup={startup_seconds:.1f}s "
            f"workers_per_rank={loader_workers} persistent_workers={persistent_workers}",
        )
    losses: list[float] = []
    started = time.perf_counter()
    checkpoint_every = int(config["training"]["checkpoint_every"])
    checkpoint_steps = set(range(checkpoint_every, max_steps + 1, checkpoint_every))
    checkpoint_steps.update(
        int(value) for value in config["training"]["selection_checkpoint_steps"]
    )
    checkpoint_steps = {value for value in checkpoint_steps if value <= max_steps}
    if smoke:
        checkpoint_steps = {max_steps}
    try:
        for step in range(1, max_steps + 1):
            step_started = time.perf_counter()
            if sampler is not None and (step - 1) % max(len(loader), 1) == 0:
                sampler.set_epoch((step - 1) // max(len(loader), 1))
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            observations = {
                key: batch[key].to(device, non_blocking=True) for key in IMAGE_KEYS + (STATE_KEY,)
            }
            clean_actions = batch[ACTION_KEY].to(device, non_blocking=True)
            padding = batch["action_is_pad"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled
            ):
                source_noise = torch.randn_like(clean_actions)
                flow_time = torch.rand(
                    clean_actions.shape[0], device=device, dtype=clean_actions.dtype
                )
                noisy = (1.0 - flow_time[:, None, None]) * source_noise + flow_time[
                    :, None, None
                ] * clean_actions
                target_velocity = clean_actions - source_noise
                prediction = train_model(observations, noisy, flow_time)
                error = (prediction - target_velocity).square()
                mask = (~padding).unsqueeze(-1).to(dtype=error.dtype)
                loss = (error * mask).sum() / mask.sum().clamp_min(1.0) / clean_actions.shape[-1]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"M2-W training produced non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            step_seconds = time.perf_counter() - step_started
            elapsed_seconds = time.perf_counter() - started
            mean_step_seconds = elapsed_seconds / step
            eta_seconds = mean_step_seconds * (max_steps - step)
            global_samples_per_second = (step * 256) / max(elapsed_seconds, 1e-9)
            current_lr = scheduler.get_last_lr()[0]
            if rank == 0:
                _write_jsonl_record(
                    loss_history_stream,
                    {
                        "event": "step",
                        "step": step,
                        "loss": loss_value,
                        "lr": current_lr,
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "step_seconds": step_seconds,
                        "elapsed_seconds": elapsed_seconds,
                        "eta_seconds": eta_seconds,
                        "global_samples_per_second": global_samples_per_second,
                        "global_batch_size": 256,
                        "world_size": world_size,
                    },
                )
                if step % (1 if smoke else 100) == 0:
                    _emit_training_log(
                        training_log_stream,
                        f"[M2-W] step={step}/{max_steps} loss={loss_value:.6f} "
                        f"lr={current_lr:.3g} "
                        f"grad_norm={float(grad_norm.detach().cpu()):.3f} "
                        f"step_time={step_seconds:.3f}s elapsed={elapsed_seconds:.1f}s "
                        f"eta={eta_seconds:.1f}s samples_per_sec={global_samples_per_second:.1f}",
                    )
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
                        "config_sha256": json_sha256(
                            {key: value for key, value in config.items() if not key.startswith("_")}
                        ),
                        "m1_frozen_protocol_sha256": sha256_file(
                            config["inputs"]["m1_frozen_protocol"]
                        ),
                        "split": "train",
                        "test_access_count": 0,
                    },
                )
                _emit_training_log(
                    training_log_stream,
                    f"[M2-W] checkpoint_saved step={step} path={_portable(checkpoint_path)}",
                )
        if rank == 0:
            manifest = _read_json(target / "run_manifest.json")
            elapsed_seconds = time.perf_counter() - started
            manifest.update(
                {
                    "status": "complete",
                    "completed_steps": max_steps,
                    "mean_tail_loss": float(np.mean(losses[-min(len(losses), 100) :])),
                    "startup_seconds": startup_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "mean_step_seconds": elapsed_seconds / max(max_steps, 1),
                    "global_samples_per_second": (max_steps * 256) / max(elapsed_seconds, 1e-9),
                    "loss_history": {
                        "path": _portable(target / "loss_history.jsonl"),
                        "records": len(losses),
                        "record_interval_steps": 1,
                        "last_loss": losses[-1] if losses else None,
                    },
                    "checkpoints": [
                        path.name for path in sorted(target.glob("checkpoint_step_*.pt"))
                    ],
                }
            )
            _write_json(target / "run_manifest.json", manifest)
            _emit_training_log(
                training_log_stream,
                f"[M2-W] training_complete steps={max_steps} "
                f"mean_tail_loss={manifest['mean_tail_loss']:.6f} "
                f"elapsed={manifest['elapsed_seconds']:.1f}s",
            )
            if smoke:
                _emit_training_log(training_log_stream, "[M2-W] smoke_postcheck_start")
                _run_smoke_postcheck(config, target)
                _emit_training_log(training_log_stream, "[M2-W] smoke_postcheck_complete")
            return manifest
        return {"status": "complete", "rank": rank, "completed_steps": max_steps}
    except BaseException as exc:
        if rank == 0:
            _emit_training_log(
                training_log_stream,
                f"[M2-W] training_failed error={type(exc).__name__}: {exc}",
            )
            _write_jsonl_record(
                loss_history_stream,
                {
                    "event": "run_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_steps": len(losses),
                },
            )
            try:
                manifest = _read_json(target / "run_manifest.json")
                manifest.update(
                    {
                        "status": "failed",
                        "completed_steps": len(losses),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                _write_json(target / "run_manifest.json", manifest)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        raise
    finally:
        try:
            _finish_distributed(distributed)
        finally:
            if loss_history_stream is not None:
                loss_history_stream.close()
            if training_log_stream is not None:
                training_log_stream.close()


def _run_smoke_postcheck(config: Mapping[str, Any], output_root: Path) -> None:
    """Exercise checkpoint reload, deterministic Flow noise and 10 exact states."""

    checkpoint = output_root / "checkpoint_step_00000020.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("Smoke training did not produce checkpoint_step_00000020.pt")
    policy, payload = load_action_flow_checkpoint(
        checkpoint, device="cuda" if torch.cuda.is_available() else "cpu", n_action_steps=4
    )
    if payload.get("step") != 20:
        raise ValueError("Smoke checkpoint reload returned the wrong optimizer step")
    device = next(policy.model.parameters()).device
    observations = {
        BASE_IMAGE_KEY: torch.zeros((2, 2, 3, 128, 128), device=device),
        HAND_IMAGE_KEY: torch.zeros((2, 2, 3, 128, 128), device=device),
        STATE_KEY: torch.zeros((2, 2, 18), device=device),
    }
    fingerprints = ["m2-w-smoke-a", "m2-w-smoke-b"]
    normalized_a = policy.predict_action_chunk(observations, fingerprints=fingerprints)
    normalized_b = policy.predict_action_chunk(observations, fingerprints=fingerprints)
    if not torch.equal(normalized_a, normalized_b):
        raise ValueError("Smoke Flow source-noise derivation is not deterministic")
    if tuple(normalized_a.shape) != (2, 16, 7):
        raise ValueError(f"Smoke Flow output must be [2,16,7], got {tuple(normalized_a.shape)}")

    from .eval_stackcube_action_flow import run_development_rollout

    bank = _rooted(config["inputs"]["state_bank_root"]) / "development.jsonl"
    for n_action_steps in (4, 8):
        run_development_rollout(
            state_bank=bank,
            output_dir=output_root / f"smoke_rollout_n{n_action_steps}",
            checkpoint=checkpoint,
            n_action_steps=n_action_steps,
            max_episodes=10,
            max_steps=10,
            num_envs=10,
            device=str(device),
            smoke=True,
        )
    manifest = _read_json(output_root / "run_manifest.json")
    manifest["smoke_checks"] = {
        "checkpoint_reload": True,
        "flow_output_shape": [2, 16, 7],
        "noise_deterministic": True,
        "exact_state_rollouts": {"development_states": 10, "n_action_steps": [4, 8]},
        "test_access_count": 0,
    }
    _write_json(output_root / "run_manifest.json", manifest)


def train(
    config_path: str | Path,
    output_root: str | Path,
    *,
    num_gpus: int | None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    expected_world_size = int(config["training"]["world_size"])
    if num_gpus != expected_world_size:
        raise ValueError(
            "Formal M2-W training requires --num-gpus equal to the frozen training world_size "
            f"({expected_world_size}); create a new config/plan before changing it"
        )
    return _train_impl(
        config,
        output_root=output_root,
        max_steps=50000,
        num_gpus=num_gpus,
        smoke=False,
        overwrite=overwrite,
    )


def smoke(
    config_path: str | Path,
    output_root: str | Path,
    *,
    num_gpus: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    return _train_impl(
        config,
        output_root=output_root,
        max_steps=20,
        num_gpus=num_gpus,
        smoke=True,
        overwrite=overwrite,
    )


def _summary_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty development evaluation")

    def mean(name: str) -> float:
        return float(np.mean([bool(row[name]) for row in rows]))

    clip_total = sum(int(row.get("action_clip_elements", 0)) for row in rows)
    action_total = sum(int(row.get("action_total_elements", 0)) for row in rows)
    return {
        "count": len(rows),
        "success_at_end": mean("success_at_end"),
        "success_once": mean("success_once"),
        "ever_grasped": mean("is_cubeA_grasped"),
        "ever_on_cube": mean("is_cubeA_on_cubeB"),
        "action_clip_fraction": clip_total / max(action_total, 1),
        "finite_rate": 1.0,
        "test_access_count": 0,
        "split": "development",
        "state_bank_indices": [int(row["index"]) for row in rows],
    }


def merge_rollout_workers(output_dir: str | Path) -> dict[str, Any]:
    root = _rooted(output_dir)
    worker_paths = sorted(root.glob("metrics_worker_*.json"))
    if not worker_paths:
        raise FileNotFoundError(f"No evaluator worker metrics in {_portable(root)}")
    payloads = [_read_json(path) for path in worker_paths]
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        if payload.get("task") != TASK_ID:
            raise ValueError("Evaluator worker task mismatch")
        if payload.get("summary", {}).get("test_access_count", 0) != 0:
            raise ValueError("Evaluator reported test access")
        rows.extend(payload.get("rows", []))
    rows.sort(key=lambda row: int(row["index"]))
    if len({int(row["index"]) for row in rows}) != len(rows):
        raise ValueError("Evaluator worker state-bank slices overlap")
    summary = _summary_from_rows(rows)
    worker_summary = [payload.get("summary", {}) for payload in payloads]
    worker_metadata = [payload.get("worker", {}) for payload in payloads]
    summary.update(
        {
            "policy_latency_p95_ms": max(
                (
                    float(item["policy_latency_p95_ms"])
                    for item in worker_summary
                    if item.get("policy_latency_p95_ms") is not None
                ),
                default=None,
            ),
            "environment_step_latency_p95_ms": max(
                (
                    float(item["environment_step_latency_p95_ms"])
                    for item in worker_summary
                    if item.get("environment_step_latency_p95_ms") is not None
                ),
                default=None,
            ),
            "amortized_policy_latency_p95_ms": max(
                (
                    float(item["amortized_policy_latency_p95_ms"])
                    for item in worker_summary
                    if item.get("amortized_policy_latency_p95_ms") is not None
                ),
                default=None,
            ),
            "chunk_forward_count": sum(
                int(item.get("chunk_forward_count", 0)) for item in worker_summary
            ),
            "flow_solver": "euler",
            "flow_integration_steps": 10,
            "worker_devices": sorted(
                {
                    str(item.get("worker_device"))
                    for item in worker_metadata
                    if item.get("worker_device") is not None
                }
            ),
            "policy_devices": sorted(
                {
                    str(item.get("policy_device"))
                    for item in worker_metadata
                    if item.get("policy_device") is not None
                }
            ),
            "environment_devices": sorted(
                {
                    str(item.get("environment_device"))
                    for item in worker_metadata
                    if item.get("environment_device") is not None
                }
            ),
            "physical_devices": sorted(
                {
                    str(item.get("physical_device"))
                    for item in worker_metadata
                    if item.get("physical_device") is not None
                }
            ),
        }
    )
    result = {
        "schema_version": 1,
        "task": TASK_ID,
        "summary": summary,
        "rows": rows,
        "workers": worker_metadata,
    }
    _write_json(root / "metrics.json", result)
    return result


def _run_rollout_subprocess(
    *,
    state_bank: Path,
    output_dir: Path,
    num_gpus: int,
    n_action_steps: int,
    checkpoint: Path | None = None,
    reference: str | None = None,
    max_episodes: int = 200,
    num_envs: int = 64,
) -> dict[str, Any]:
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    if checkpoint is None and reference is None or checkpoint is not None and reference is not None:
        raise ValueError("rollout requires exactly one candidate/reference")
    _prepare_output_dir(output_dir)
    output_root = _rooted(output_dir)
    gpu_ids = _visible_gpu_ids(num_gpus)
    for stale in output_root.glob("progress_worker_*.json"):
        stale.unlink()
    processes: list[tuple[subprocess.Popen[Any], int, Any]] = []
    for worker_index in range(num_gpus):
        command = [
            sys.executable,
            "-m",
            "robot_practice.experiments.maniskill.eval_stackcube_action_flow",
            "--state-bank",
            str(state_bank),
            "--output-dir",
            str(output_dir),
            "--n-action-steps",
            str(n_action_steps),
            "--max-episodes",
            str(max_episodes),
            "--num-envs",
            str(num_envs),
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(num_gpus),
        ]
        if checkpoint is not None:
            command.extend(("--checkpoint", str(checkpoint)))
        else:
            command.extend(("--reference", str(reference)))
        command.extend(("--device", "cuda:0"))
        log_path = output_root / f"worker_{worker_index:03d}.log"
        log_stream = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu_ids[worker_index],
                "M2_W_PHYSICAL_GPU": gpu_ids[worker_index],
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        process = subprocess.Popen(
            command,
            cwd=practice_root(),
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        print(
            f"[M2-W parent] launched worker={worker_index} pid={process.pid} "
            f"physical_gpu={gpu_ids[worker_index]} log={_portable(log_path)}",
            flush=True,
        )
        processes.append((process, worker_index, log_stream))

    label = f"reference={reference}" if reference is not None else Path(checkpoint).stem
    started = time.perf_counter()
    total_episodes = max_episodes
    progress_signatures: dict[int, tuple[Any, ...] | None] = {
        worker_index: None for worker_index in range(num_gpus)
    }
    progress_changed_at = {
        worker_index: started for worker_index in range(num_gpus)
    }

    def render_progress() -> list[tuple[int, str, str]]:
        progress_payloads: list[dict[str, Any]] = []
        for worker_index in range(num_gpus):
            path = output_root / f"progress_worker_{worker_index:03d}.json"
            try:
                progress_payloads.append(_read_json(path))
            except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
                progress_payloads.append(
                    {
                        "completed_episodes": 0,
                        "total_episodes": max(1, total_episodes // num_gpus),
                        "current_batch_episodes": 0,
                        "current_step": 0,
                        "max_steps": 0,
                        "status": "starting",
                    }
                )
        now = time.perf_counter()
        for worker_index, payload in enumerate(progress_payloads):
            signature = (
                payload.get("status"),
                payload.get("completed_episodes"),
                payload.get("total_episodes"),
                payload.get("current_batch_episodes"),
                payload.get("current_step"),
                payload.get("max_steps"),
                payload.get("error"),
            )
            if signature != progress_signatures[worker_index]:
                progress_signatures[worker_index] = signature
                progress_changed_at[worker_index] = now
        completed = min(
            total_episodes,
            sum(int(payload.get("completed_episodes", 0)) for payload in progress_payloads),
        )
        in_flight = sum(
            min(
                int(payload.get("current_batch_episodes", 0)),
                int(payload.get("current_batch_episodes", 0))
                * int(payload.get("current_step", 0))
                / max(int(payload.get("max_steps", 0)), 1),
            )
            for payload in progress_payloads
        )
        progress_episodes = min(float(total_episodes), completed + in_flight)
        total = max(1, total_episodes)
        fraction = progress_episodes / total
        width = 28
        filled = min(width, int(fraction * width))
        bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
        current_step = max(int(payload.get("current_step", 0)) for payload in progress_payloads)
        max_steps = max(int(payload.get("max_steps", 0)) for payload in progress_payloads)
        complete_workers = sum(payload.get("status") == "complete" for payload in progress_payloads)
        active_workers = [
            f"{worker_index}:{payload.get('status', 'unknown')}@{int(payload.get('current_step', 0))}"
            for worker_index, payload in enumerate(progress_payloads)
            if payload.get("status") != "complete"
        ]
        worker_text = ",".join(active_workers) if active_workers else "-"
        elapsed = time.perf_counter() - started
        if progress_episodes > 0:
            eta = (total - progress_episodes) / (progress_episodes / max(elapsed, 1e-9))
            eta_text = f"{eta:.0f}s"
        else:
            eta_text = "--"
        print(
            f"\r[M2-W] {label} [{bar}] progress={progress_episodes:.1f}/{total} "
            f"episodes done={completed}/{total} step={current_step}/{max_steps or '?'} "
            f"workers={complete_workers}/{num_gpus} active={worker_text} eta={eta_text}",
            end="",
            flush=True,
        )
        stalled: list[tuple[int, str, str]] = []
        for worker_index, payload in enumerate(progress_payloads):
            process = processes[worker_index][0]
            status = str(payload.get("status", "starting"))
            if process.poll() is not None or status == "complete":
                continue
            if now - progress_changed_at[worker_index] > WORKER_PROGRESS_TIMEOUT_SECONDS:
                stalled.append(
                    (
                        worker_index,
                        status,
                        _portable(output_root / f"worker_{worker_index:03d}.log"),
                    )
                )
        return stalled

    def stop_live_workers() -> None:
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline and any(
            process.poll() is None for process, _, _ in processes
        ):
            time.sleep(0.1)
        for process, _, _ in processes:
            if process.poll() is None:
                process.kill()

    render_progress()
    failures = []
    while True:
        stalled_workers = render_progress()
        if stalled_workers:
            print(
                f"\n[M2-W] worker progress stalled for more than "
                f"{WORKER_PROGRESS_TIMEOUT_SECONDS:.0f}s: {stalled_workers}",
                file=sys.stderr,
                flush=True,
            )
            stop_live_workers()
            raise RuntimeError(
                "M2-W evaluator worker heartbeat stalled; inspect the listed worker logs"
            )
        exited_workers: list[tuple[int, int | None, str, str]] = []
        for process, worker_index, _ in processes:
            return_code = process.poll()
            if return_code is None:
                continue
            progress_path = output_root / f"progress_worker_{worker_index:03d}.json"
            try:
                progress = _read_json(progress_path)
                status = str(progress.get("status", "missing-status"))
            except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
                status = "missing-progress"
            if return_code != 0 or status != "complete":
                print(
                    f"\n[M2-W parent] worker={worker_index} exited pid={process.pid} "
                    f"returncode={return_code} status={status} "
                    f"log={_portable(output_root / f'worker_{worker_index:03d}.log')}",
                    file=sys.stderr,
                    flush=True,
                )
                exited_workers.append(
                    (
                        worker_index,
                        return_code,
                        status,
                        _portable(output_root / f"worker_{worker_index:03d}.log"),
                    )
                )
        if exited_workers:
            print(
                f"\n[M2-W] worker exited before completion: {exited_workers}",
                file=sys.stderr,
                flush=True,
            )
            stop_live_workers()
            raise RuntimeError(
                "M2-W evaluator worker exited before writing complete metrics; "
                "inspect the listed worker logs"
            )
        if all(process.poll() is not None for process, _, _ in processes):
            break
        time.sleep(0.5)
    print(flush=True)
    for process, worker_index, log_stream in processes:
        return_code = process.wait()
        log_stream.close()
        print(
            f"[M2-W parent] worker={worker_index} pid={process.pid} "
            f"finished returncode={return_code}",
            flush=True,
        )
        if return_code != 0:
            failures.append((worker_index, return_code, _portable(output_root / f"worker_{worker_index:03d}.log")))
    if failures:
        raise RuntimeError(f"M2-W evaluator workers failed: {failures}")
    return merge_rollout_workers(output_dir)


def _offline_action_mse(
    config: Mapping[str, Any],
    checkpoint: Path,
    *,
    device: str | torch.device,
    batch_size: int = 16,
) -> tuple[float, float]:
    stats_path = _rooted(checkpoint).parent / "normalization_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Training normalization stats are missing: {_portable(stats_path)}"
        )
    stats = _load_stats(stats_path)
    dataset = StackCubeWindowDataset(config, split="development", stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    policy, _ = load_action_flow_checkpoint(checkpoint, device=device, n_action_steps=4)
    model = policy.model
    model.eval()
    total_error = 0.0
    total_count = 0
    baseline_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            observations = {
                key: batch[key].to(device=device, dtype=torch.float32)
                for key in IMAGE_KEYS + (STATE_KEY,)
            }
            target = batch[ACTION_KEY].to(device=device, dtype=torch.float32)
            padding = batch["action_is_pad"].to(device=device)
            sample_indices = [int(value) for value in batch["sample_index"].tolist()]
            fingerprints = [f"offline-development-sample-{value}" for value in sample_indices]
            prediction = model.sample_actions(
                observations, fingerprints=fingerprints, chunk_index=0
            )
            mask = (~padding).unsqueeze(-1)
            total_error += float(((prediction - target).square() * mask).sum().cpu())
            baseline_error += float((target.square() * mask).sum().cpu())
            total_count += int(mask.sum().item()) * target.shape[-1]
    if total_count <= 0:
        raise ValueError("Development split yielded no unpadded action targets")
    return total_error / total_count, baseline_error / total_count


def _bootstrap_ci(differences: Sequence[float], *, seed: int, samples: int) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one matched state")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_success_bootstrap_ci(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 1000,
    samples: int = 10000,
) -> tuple[float, float]:
    reference_by_fingerprint = {str(row["start_state_fingerprint"]): row for row in reference_rows}
    differences = []
    for candidate in candidate_rows:
        fingerprint = str(candidate["start_state_fingerprint"])
        reference = reference_by_fingerprint.get(fingerprint)
        if reference is None:
            raise ValueError(f"Candidate/reference state fingerprint mismatch: {fingerprint}")
        differences.append(
            float(bool(candidate["success_at_end"])) - float(bool(reference["success_at_end"]))
        )
    return _bootstrap_ci(differences, seed=seed, samples=samples)


def selection_sort_key(entry: Mapping[str, Any]) -> tuple[float, ...]:
    """Mechanical M2-W checkpoint/n_action_steps ranking key."""

    summary = entry.get("summary", entry)
    return (
        -float(summary.get("success_at_end", -math.inf)),
        -float(summary.get("success_once", -math.inf)),
        -float(summary.get("ever_on_cube", -math.inf)),
        float(entry.get("normalized_action_mse", math.inf)),
        float(summary.get("amortized_policy_latency_p95_ms", math.inf) or math.inf),
        float(entry.get("checkpoint_step", math.inf)),
        float(entry.get("n_action_steps", math.inf)),
    )


def evaluate_development_gates(
    selected: Mapping[str, Any],
    *,
    strongest_reference: Mapping[str, Any],
    reference_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    summary = selected["summary"]
    flow_baseline = float(selected["train_action_mean_mse"])
    flow_mse = float(selected["normalized_action_mse"])
    relative_flow_gain = (flow_baseline - flow_mse) / max(flow_baseline, 1e-12)
    ci_low, ci_high = paired_success_bootstrap_ci(
        selected["rows"],
        reference_rows,
        seed=int(config["evaluation"]["paired_bootstrap_seed"]),
        samples=int(config["evaluation"]["paired_bootstrap_samples"]),
    )
    reference_success = float(strongest_reference["summary"]["success_at_end"])
    candidate_success = float(summary["success_at_end"])
    gates = {
        "data_identity": {
            "passed": summary.get("split") == "development"
            and summary.get("test_access_count", 1) == 0,
            "test_access_count": summary.get("test_access_count"),
        },
        "flow_learning": {
            "passed": relative_flow_gain >= 0.10,
            "normalized_action_mse": flow_mse,
            "train_action_mean_mse": flow_baseline,
            "relative_gain": relative_flow_gain,
        },
        "task_mastery": {
            "passed": (
                candidate_success >= 0.30
                and float(summary["success_once"]) >= 0.40
                and float(summary["ever_grasped"]) >= 0.70
                and float(summary["ever_on_cube"]) >= 0.40
            ),
            "thresholds": {
                "success_at_end": 0.30,
                "success_once": 0.40,
                "ever_grasped": 0.70,
                "ever_on_cube": 0.40,
            },
        },
        "above_reference": {
            "passed": candidate_success - reference_success >= 0.20 and ci_low > 0.0,
            "strongest_reference_success_at_end": reference_success,
            "paired_difference": candidate_success - reference_success,
            "bootstrap_95_ci": [ci_low, ci_high],
        },
        "headroom": {
            "passed": candidate_success <= 0.90,
            "threshold": 0.90,
        },
        "action_legality": {
            "passed": float(summary.get("finite_rate", 0.0)) == 1.0
            and float(summary.get("action_clip_fraction", 1.0)) <= 0.01,
            "finite_rate": summary.get("finite_rate"),
            "action_clip_fraction": summary.get("action_clip_fraction"),
        },
        "realtime": {
            "passed": (
                summary.get("amortized_policy_latency_p95_ms") is not None
                and float(summary["amortized_policy_latency_p95_ms"]) <= 50.0
            ),
            "amortized_policy_latency_p95_ms": summary.get("amortized_policy_latency_p95_ms"),
            "threshold_ms": 50.0,
        },
    }
    return {"gate_pass": all(bool(item["passed"]) for item in gates.values()), "gates": gates}


def _load_metrics(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("task") != TASK_ID:
        raise ValueError(f"Metrics task mismatch at {_portable(path)}")
    if payload.get("summary", {}).get("test_access_count", 0) != 0:
        raise ValueError(f"Metrics report test access at {_portable(path)}")
    return payload


def evaluate_references(
    config_path: str | Path,
    output_root: str | Path,
    *,
    num_gpus: int,
    num_envs: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config, num_gpus=num_gpus, require_batch_divisor=False)
    development_root = _prepare_output_dir(output_root, overwrite=overwrite)
    bank = _rooted(config["inputs"]["state_bank_root"]) / "development.jsonl"
    runs = {}
    for reference in ("zero", "iid_gaussian_random"):
        run_dir = development_root / "references" / reference
        metrics = _run_rollout_subprocess(
            state_bank=bank,
            output_dir=run_dir,
            num_gpus=num_gpus,
            n_action_steps=4,
            reference=reference,
            num_envs=num_envs,
        )
        runs[reference] = metrics
    payload = {
        "schema_version": 1,
        "task": TASK_ID,
        "split": "development",
        "test_access_count": 0,
        "references": {
            name: {"summary": value["summary"], "rows": value["rows"]}
            for name, value in runs.items()
        },
        "m1_frozen_protocol_sha256": sha256_file(config["inputs"]["m1_frozen_protocol"]),
    }
    _write_json(development_root / "reference_metrics.json", payload)
    return payload


def _checkpoint_path(train_root: Path, step: int) -> Path:
    path = train_root / f"checkpoint_step_{step:08d}.pt"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"M2-W candidate checkpoint is missing or not regular: {_portable(path)}"
        )
    return path


def select_development(
    config_path: str | Path,
    *,
    train_root: str | Path,
    output_root: str | Path,
    num_gpus: int,
    device: str | None = None,
    num_envs: int = 64,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config, num_gpus=num_gpus, require_batch_divisor=False)
    train_dir = _rooted(train_root)
    train_manifest = _read_json(train_dir / "run_manifest.json")
    if train_manifest.get("status") != "complete" or train_manifest.get("completed_steps") != 50000:
        raise ValueError("M2-W development selection requires a complete 50000-step training run")
    root = _rooted(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Development output root does not exist: {_portable(root)}")
    selection_artifacts = (
        root / "candidate_metrics.json",
        root / "selected_checkpoint.json",
        root / "candidates",
    )
    if any(path.exists() for path in selection_artifacts):
        if overwrite:
            for path in selection_artifacts:
                if path.is_symlink():
                    raise ValueError(
                        f"M2-W selection artifact must not be a symbolic link: {_portable(path)}"
                    )
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
        else:
            raise FileExistsError(
                f"Refusing to overwrite existing development selection: {_portable(root)}"
            )
    references_payload = _read_json(root / "reference_metrics.json")
    if references_payload.get("test_access_count") != 0:
        raise ValueError("Reference metrics report test access")
    reference_entries = references_payload["references"]
    strongest_name = max(
        reference_entries,
        key=lambda name: float(reference_entries[name]["summary"]["success_at_end"]),
    )
    strongest_reference = reference_entries[strongest_name]
    bank = _rooted(config["inputs"]["state_bank_root"]) / "development.jsonl"
    candidates = []
    for step in config["training"]["selection_checkpoint_steps"]:
        checkpoint = _checkpoint_path(train_dir, int(step))
        offline_mse, baseline_mse = _offline_action_mse(
            config,
            checkpoint,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        for n_action_steps in config["evaluation"]["n_action_steps"]:
            candidate_dir = root / "candidates" / f"step_{int(step):08d}_n{int(n_action_steps)}"
            metrics = _run_rollout_subprocess(
                state_bank=bank,
                output_dir=candidate_dir,
                num_gpus=num_gpus,
                n_action_steps=int(n_action_steps),
                checkpoint=checkpoint,
                num_envs=num_envs,
            )
            candidates.append(
                {
                    "checkpoint": _portable(checkpoint),
                    "checkpoint_step": int(step),
                    "n_action_steps": int(n_action_steps),
                    "normalized_action_mse": float(offline_mse),
                    "train_action_mean_mse": float(baseline_mse),
                    "summary": metrics["summary"],
                    "rows": metrics["rows"],
                    "metrics_path": _portable(candidate_dir / "metrics.json"),
                }
            )
    ranking = sorted(candidates, key=selection_sort_key)
    selected = copy.deepcopy(ranking[0])
    gates = evaluate_development_gates(
        selected,
        strongest_reference=strongest_reference,
        reference_rows=strongest_reference["rows"],
        config=config,
    )
    payload = {
        "schema_version": 1,
        "task": TASK_ID,
        "split": "development",
        "test_access_count": 0,
        "reference_policy": strongest_name,
        "references": {
            name: {"summary": value["summary"]} for name, value in reference_entries.items()
        },
        "ranking": ranking,
        "selected": selected,
        "gates": gates,
        "config_sha256": json_sha256(
            {key: value for key, value in config.items() if not key.startswith("_")}
        ),
        "m1_frozen_protocol_sha256": sha256_file(config["inputs"]["m1_frozen_protocol"]),
    }
    _write_json(root / "candidate_metrics.json", {"candidates": candidates, "test_access_count": 0})
    _write_json(root / "selected_checkpoint.json", payload)
    return payload


def freeze_baseline(
    config_path: str | Path,
    *,
    train_root: str | Path,
    selection_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    selection = _read_json(selection_path)
    if selection.get("gates", {}).get("gate_pass") is not True:
        raise ValueError("M2-W selection gates did not pass; baseline cannot be frozen")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise TypeError("selected_checkpoint.json has no selected candidate")
    checkpoint = _rooted(str(selected["checkpoint"]))
    train_dir = _rooted(train_root)
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(
            f"Selected checkpoint is not a regular file: {_portable(checkpoint)}"
        )
    if not checkpoint.resolve().is_relative_to(train_dir.resolve()):
        raise ValueError("Selected checkpoint must come from the declared formal train root")
    output = _prepare_output_dir(output_root, overwrite=overwrite)
    checkpoint_dir = output / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    frozen_checkpoint = checkpoint_dir / "model.pt"
    shutil.copy2(checkpoint, frozen_checkpoint)
    normalization = train_dir / "normalization_stats.json"
    if not normalization.is_file():
        raise FileNotFoundError("Training normalization_stats.json is missing")
    frozen_normalization = checkpoint_dir / "normalization_stats.json"
    shutil.copy2(normalization, frozen_normalization)
    source_config = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = {
        "schema_version": 1,
        "experiment_id": "m2_w_action_only_flow",
        "task": TASK_ID,
        "gate_pass": True,
        "selected": {
            "checkpoint_step": int(selected["checkpoint_step"]),
            "n_action_steps": int(selected["n_action_steps"]),
            "checkpoint": _portable(frozen_checkpoint),
            "source_checkpoint": _portable(checkpoint),
        },
        "hashes": {
            "checkpoint_sha256": sha256_file(frozen_checkpoint),
            "config_sha256": json_sha256(source_config),
            "m1_frozen_protocol_sha256": sha256_file(config["inputs"]["m1_frozen_protocol"]),
            "dataset_manifest_sha256": sha256_file(config["inputs"]["dataset_manifest"]),
            "offline_split_sha256": sha256_file(config["inputs"]["offline_split_manifest"]),
            "state_bank_manifest_sha256": sha256_file(config["inputs"]["state_bank_manifest"]),
            "normalization_stats_sha256": sha256_file(frozen_normalization),
        },
        "contract": {
            "backend": "physx_cuda",
            "control_mode": "pd_ee_delta_pose",
            "formal_horizon": 500,
            "action_dim": 7,
            "action_horizon": 16,
            "n_obs_steps": 2,
            "flow_solver": "euler",
            "flow_integration_steps": 10,
            "training_seed": 1000,
            "global_batch_size": 256,
        },
        "development_ranking": selection.get("ranking", []),
        "development_gates": selection.get("gates", {}),
        "test_access_count": 0,
        "checkpoint_is_copy": True,
    }
    _write_json(output / "frozen_base.json", payload)
    return payload


def validate_frozen_baseline(
    config_path: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    validation = validate_config(config)
    payload = _read_json(protocol_path)
    if payload.get("gate_pass") is not True or payload.get("task") != TASK_ID:
        raise ValueError("M2-W frozen baseline is not a passing StackCube artifact")
    if payload.get("test_access_count") != 0:
        raise ValueError("M2-W frozen baseline reports test access")
    contract = payload.get("contract", {})
    expected_contract = {
        "backend": "physx_cuda",
        "control_mode": "pd_ee_delta_pose",
        "formal_horizon": 500,
        "action_dim": 7,
        "action_horizon": 16,
        "n_obs_steps": 2,
        "flow_solver": "euler",
        "flow_integration_steps": 10,
        "training_seed": 1000,
        "global_batch_size": 256,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"Frozen M2-W contract drift at {key}: {contract.get(key)!r}")
    selected = payload.get("selected", {})
    n_action_steps = selected.get("n_action_steps")
    if n_action_steps not in (4, 8):
        raise ValueError("Frozen M2-W n_action_steps must be 4 or 8")
    checkpoint = _rooted(selected.get("checkpoint", ""))
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError("Frozen M2-W checkpoint must be a regular copied file")
    observed = sha256_file(checkpoint)
    if observed != payload.get("hashes", {}).get("checkpoint_sha256"):
        raise ValueError("Frozen M2-W checkpoint hash mismatch")
    normalization = checkpoint.parent / "normalization_stats.json"
    if not normalization.is_file() or normalization.is_symlink():
        raise FileNotFoundError(
            "Frozen M2-W normalization_stats.json must be copied beside checkpoint"
        )
    if sha256_file(normalization) != payload.get("hashes", {}).get("normalization_stats_sha256"):
        raise ValueError("Frozen M2-W normalization stats hash mismatch")
    for key, input_key in (
        ("m1_frozen_protocol_sha256", "m1_frozen_protocol"),
        ("dataset_manifest_sha256", "dataset_manifest"),
        ("offline_split_sha256", "offline_split_manifest"),
        ("state_bank_manifest_sha256", "state_bank_manifest"),
    ):
        if payload.get("hashes", {}).get(key) != sha256_file(config["inputs"][input_key]):
            raise ValueError(f"Frozen M2-W {key} no longer matches current input")
    source_config = {key: value for key, value in config.items() if not key.startswith("_")}
    if payload.get("hashes", {}).get("config_sha256") != json_sha256(source_config):
        raise ValueError("Frozen M2-W config hash no longer matches the declared config")
    return {
        "gate_pass": True,
        "task": TASK_ID,
        "protocol": _portable(protocol_path),
        "checkpoint": _portable(checkpoint),
        "n_action_steps": n_action_steps,
        "test_access_count": 0,
        "m1_frozen_protocol_sha256": validation.get("m1_frozen_protocol_sha256"),
    }


validate_frozen = validate_frozen_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--num-gpus", type=int)
    validate.add_argument("--skip-input-check", action="store_true", help=argparse.SUPPRESS)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    smoke_parser.add_argument("--output-root", type=Path, required=True)
    smoke_parser.add_argument("--num-gpus", type=int, default=1)
    smoke_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output directory under outputs/.",
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_parser.add_argument("--output-root", type=Path, required=True)
    train_parser.add_argument("--num-gpus", type=int)
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output directory under outputs/.",
    )

    references = subparsers.add_parser("evaluate-references")
    references.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    references.add_argument("--output-root", type=Path, required=True)
    references.add_argument("--num-gpus", type=int, required=True)
    references.add_argument("--num-envs", type=int, default=64)
    references.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output directory under outputs/.",
    )

    select = subparsers.add_parser("select-development")
    select.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    select.add_argument("--train-root", type=Path, required=True)
    select.add_argument("--output-root", type=Path, required=True)
    select.add_argument("--num-gpus", type=int, required=True)
    select.add_argument("--device")
    select.add_argument("--num-envs", type=int, default=64)
    select.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing candidate/selection artifacts, preserving references.",
    )

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze.add_argument("--train-root", type=Path, required=True)
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output directory under outputs/.",
    )

    frozen = subparsers.add_parser("validate-frozen")
    frozen.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    frozen.add_argument("--protocol", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        result = validate_config(
            args.config,
            num_gpus=args.num_gpus,
            check_inputs=not args.skip_input_check,
        )
    elif args.command == "smoke":
        result = smoke(
            args.config,
            args.output_root,
            num_gpus=args.num_gpus,
            overwrite=args.overwrite,
        )
    elif args.command == "train":
        result = train(
            args.config,
            args.output_root,
            num_gpus=args.num_gpus,
            overwrite=args.overwrite,
        )
    elif args.command == "evaluate-references":
        result = evaluate_references(
            args.config,
            args.output_root,
            num_gpus=args.num_gpus,
            num_envs=args.num_envs,
            overwrite=args.overwrite,
        )
    elif args.command == "select-development":
        result = select_development(
            args.config,
            train_root=args.train_root,
            output_root=args.output_root,
            num_gpus=args.num_gpus,
            device=args.device,
            num_envs=args.num_envs,
            overwrite=args.overwrite,
        )
    elif args.command == "freeze":
        result = freeze_baseline(
            args.config,
            train_root=args.train_root,
            selection_path=args.selection,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    elif args.command == "validate-frozen":
        result = validate_frozen_baseline(args.config, args.protocol)
    else:  # pragma: no cover - argparse makes this unreachable
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
