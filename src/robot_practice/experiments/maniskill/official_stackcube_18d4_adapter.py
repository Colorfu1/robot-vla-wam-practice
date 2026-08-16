"""Matched 18D-state/4D-action adapters for the StackCube ablation runs.

The two supported inputs deliberately share one training contract:

* two RGB cameras, concatenated to six channels;
* ``agent.qpos(9) + agent.qvel(9)`` as an 18D state;
* ``pd_ee_delta_pos`` actions with four channels.

``official_rgb_18d_4d`` reads the native official motion-planning H5.
``m1w_projected_rgb_4d`` reads the frozen M1-W H5 and projects native 7D
``pd_ee_delta_pose`` labels with ``[dx, dy, dz, gripper]``.  The latter is an
action ablation, not a native 4D demonstration: its labels were originally
executed under the 7D pose controller.

Only observation/action windows are read in ``__getitem__``.  No converted H5
is written, so the original frozen datasets remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .official_m1w_adapter import (
    CAMERA_NAMES,
    EXPECTED_RGB_SHAPE,
    M1WFlattenRGBObservationWrapper,
    _read_episode_ids,
    _traj_name,
)

OFFICIAL_FORMAT = "official_rgb_18d_4d"
M1W_PROJECTED_FORMAT = "m1w_projected_rgb_4d"
SUPPORTED_FORMATS = (OFFICIAL_FORMAT, M1W_PROJECTED_FORMAT)
EXPECTED_TASK = "StackCube-v1"
EXPECTED_CONTROL_MODE = "pd_ee_delta_pos"
EXPECTED_STATE_DIM = 18
EXPECTED_ACTION_DIM = 4
M1W_SOURCE_ACTION_DIM = 7
M1W_ACTION_PROJECTION = (0, 1, 2, 6)


def _sorted_episode_ids(handle: h5py.File) -> list[int]:
    ids = []
    for name in handle:
        if name.startswith("traj_"):
            try:
                ids.append(int(name.split("_")[-1]))
            except ValueError:
                continue
    if not ids:
        raise ValueError("H5 contains no traj_<id> groups")
    return sorted(ids)


def _validate_episode(
    trajectory: h5py.Group,
    *,
    episode_id: int,
    source_action_dim: int,
) -> int:
    actions = trajectory.get("actions")
    if actions is None or actions.ndim != 2:
        raise ValueError(f"traj_{episode_id}/actions must be a 2D dataset")
    length, action_dim = map(int, actions.shape)
    if action_dim != source_action_dim:
        raise ValueError(
            f"traj_{episode_id}/actions has dim {action_dim}, expected {source_action_dim}"
        )
    required = (
        "obs/agent/qpos",
        "obs/agent/qvel",
        "obs/sensor_data/base_camera/rgb",
        "obs/sensor_data/hand_camera/rgb",
    )
    for key in required:
        if key not in trajectory:
            raise KeyError(f"traj_{episode_id}/{key} is missing")
    qpos = trajectory["obs/agent/qpos"]
    qvel = trajectory["obs/agent/qvel"]
    if qpos.shape != (length + 1, 9) or qvel.shape != (length + 1, 9):
        raise ValueError(
            f"traj_{episode_id} qpos/qvel must be ({length + 1}, 9), "
            f"got {qpos.shape} and {qvel.shape}"
        )
    for camera in CAMERA_NAMES:
        rgb = trajectory[f"obs/sensor_data/{camera}/rgb"]
        if rgb.shape != (length + 1, *EXPECTED_RGB_SHAPE):
            raise ValueError(
                f"traj_{episode_id}/{camera}/rgb must be "
                f"({length + 1}, 128, 128, 3), got {rgb.shape}"
            )
        if rgb.dtype != np.uint8:
            raise ValueError(
                f"traj_{episode_id}/{camera}/rgb must be uint8, got {rgb.dtype}"
            )
    return length


class StackCube18D4DDataset(Dataset):
    """Lazy RGB/state/action windows for either matched 4D ablation input."""

    def __init__(
        self,
        data_path: str | Path,
        episode_ids_file: str | Path,
        obs_horizon: int,
        pred_horizon: int,
        dataset_format: str,
    ) -> None:
        if dataset_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported dataset format {dataset_format!r}; "
                f"expected one of {SUPPORTED_FORMATS}"
            )
        self.data_path = Path(data_path)
        self.episode_ids_file = Path(episode_ids_file)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.dataset_format = dataset_format
        if self.obs_horizon < 1 or self.pred_horizon < self.obs_horizon:
            raise ValueError("require 1 <= obs_horizon <= pred_horizon")
        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        self.episode_ids = _read_episode_ids(self.episode_ids_file)
        # Do not keep an HDF5 handle across DataLoader worker forks. The
        # training process validates/indexes with a short-lived handle, and
        # each worker lazily opens its own read-only handle on first access.
        self._file: h5py.File | None = None
        self._file_pid: int | None = None
        self._episode_lengths: dict[int, int] = {}
        self.slices: list[tuple[int, int, int, int]] = []
        with h5py.File(self.data_path, "r") as handle:
            self._validate_and_index(handle)
        self.pad_action_arm = np.zeros((EXPECTED_ACTION_DIM - 1,), dtype=np.float32)

    @property
    def source_action_dim(self) -> int:
        return 4 if self.dataset_format == OFFICIAL_FORMAT else M1W_SOURCE_ACTION_DIM

    def _validate_and_index(self, handle: h5py.File) -> None:
        total_transitions = 0
        pad_before = self.obs_horizon - 1
        pad_after = self.pred_horizon - self.obs_horizon
        available = set(handle.keys())
        for episode_id in self.episode_ids:
            name = _traj_name(episode_id)
            if name not in available:
                raise KeyError(f"{name} is missing from {self.data_path}")
            length = _validate_episode(
                handle[name],
                episode_id=episode_id,
                source_action_dim=self.source_action_dim,
            )
            self._episode_lengths[episode_id] = length
            total_transitions += length
            self.slices.extend(
                (episode_id, start, start + self.pred_horizon, length)
                for start in range(-pad_before, length - self.pred_horizon + pad_after)
            )
        if not self.slices:
            raise ValueError("ablation dataset produced no training windows")
        print(
            "StackCube 18D/4D lazy dataset: "
            f"format={self.dataset_format} episodes={len(self.episode_ids)} "
            f"transitions={total_transitions} windows={len(self.slices)} "
            f"h5={self.data_path}",
            flush=True,
        )

    @staticmethod
    def _pad_observation(values: np.ndarray, start: int, horizon: int) -> np.ndarray:
        if start >= 0:
            return values[start : start + horizon]
        prefix = np.repeat(values[0:1], -start, axis=0)
        return np.concatenate((prefix, values[0 : start + horizon]), axis=0)

    def _project_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.dataset_format == OFFICIAL_FORMAT:
            return np.asarray(actions, dtype=np.float32)
        return np.asarray(actions[..., M1W_ACTION_PROJECTION], dtype=np.float32)

    def _get_file(self) -> h5py.File:
        """Return the read-only H5 handle owned by the current process."""

        pid = os.getpid()
        if self._file is None or self._file_pid != pid:
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.data_path, "r")
            self._file_pid = pid
        return self._file

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        episode_id, start, end, length = self.slices[index]
        trajectory = self._get_file()[_traj_name(episode_id)]
        obs_start = max(0, start)
        obs_end = start + self.obs_horizon

        qpos = self._pad_observation(
            trajectory["obs/agent/qpos"][...], start, self.obs_horizon
        )
        qvel = self._pad_observation(
            trajectory["obs/agent/qvel"][...], start, self.obs_horizon
        )
        base_rgb = trajectory["obs/sensor_data/base_camera/rgb"][obs_start:obs_end]
        hand_rgb = trajectory["obs/sensor_data/hand_camera/rgb"][obs_start:obs_end]
        if start < 0:
            base_rgb = np.concatenate(
                (np.repeat(base_rgb[0:1], -start, axis=0), base_rgb), axis=0
            )
            hand_rgb = np.concatenate(
                (np.repeat(hand_rgb[0:1], -start, axis=0), hand_rgb), axis=0
            )
        rgb = np.transpose(np.concatenate((base_rgb, hand_rgb), axis=-1), (0, 3, 1, 2))

        actions = trajectory["actions"][max(0, start) : min(length, end)]
        actions = self._project_actions(actions)
        if start < 0:
            actions = np.concatenate(
                (np.repeat(actions[0:1], -start, axis=0), actions), axis=0
            )
        if end > length:
            pad_action = np.concatenate(
                (self.pad_action_arm, actions[-1, -1:]), axis=0
            )
            actions = np.concatenate(
                (actions, np.repeat(pad_action[None], end - length, axis=0)), axis=0
            )

        state = np.concatenate((qpos, qvel), axis=-1).astype(np.float32, copy=False)
        if state.shape != (self.obs_horizon, EXPECTED_STATE_DIM):
            raise RuntimeError(f"unexpected state window shape: {state.shape}")
        if rgb.shape != (self.obs_horizon, 6, 128, 128):
            raise RuntimeError(f"unexpected RGB window shape: {rgb.shape}")
        if actions.shape != (self.pred_horizon, EXPECTED_ACTION_DIM):
            raise RuntimeError(f"unexpected action window shape: {actions.shape}")
        return {
            "observations": {
                "state": torch.from_numpy(np.ascontiguousarray(state)),
                "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
            },
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
        }

    def __len__(self) -> int:
        return len(self.slices)

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None
            self._file_pid = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown ordering
        try:
            self.close()
        except (OSError, ValueError):
            pass


def _read_split_ids(split_manifest: Path) -> list[int]:
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "train" not in payload:
        raise ValueError(f"split manifest must contain train: {split_manifest}")
    return [int(value) for value in payload["train"]]


def prepare_ablation(
    *,
    dataset_format: str,
    h5_path: str | Path,
    output_dir: str | Path,
    split_manifest: str | Path | None = None,
    num_demos: int | None = None,
    task: str = EXPECTED_TASK,
) -> dict[str, Any]:
    """Validate one source and write a small episode-ID sidecar/manifest."""

    if dataset_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported dataset format: {dataset_format}")
    if task != EXPECTED_TASK:
        raise ValueError(f"only {EXPECTED_TASK} is supported")
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if num_demos is not None and num_demos <= 0:
        raise ValueError("num_demos must be positive")
    source_action_dim = 4 if dataset_format == OFFICIAL_FORMAT else 7
    with h5py.File(h5_path, "r") as handle:
        available = set(_sorted_episode_ids(handle))
        if dataset_format == M1W_PROJECTED_FORMAT:
            if split_manifest is None:
                raise ValueError("m1w_projected_rgb_4d requires --split-manifest")
            episode_ids = _read_split_ids(Path(split_manifest))
        else:
            episode_ids = sorted(available)
        if num_demos is not None:
            episode_ids = episode_ids[:num_demos]
        if not episode_ids:
            raise ValueError("no episodes selected")
        missing = [episode_id for episode_id in episode_ids if episode_id not in available]
        if missing:
            raise KeyError(f"selected episodes are missing from H5: {missing[:5]}")
        lengths = {
            episode_id: _validate_episode(
                handle[_traj_name(episode_id)],
                episode_id=episode_id,
                source_action_dim=source_action_dim,
            )
            for episode_id in episode_ids
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "train_episode_ids.json"
    ids_path.write_text(json.dumps(episode_ids, indent=2) + "\n", encoding="utf-8")
    split_sha256 = None
    if split_manifest is not None:
        split_path = Path(split_manifest)
        split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "dataset_format": dataset_format,
        "task": task,
        "control_mode": EXPECTED_CONTROL_MODE,
        "obs_mode": "rgb",
        "state_dim": EXPECTED_STATE_DIM,
        "state_definition": "agent.qpos(9) + agent.qvel(9)",
        "action_dim": EXPECTED_ACTION_DIM,
        "source_action_dim": source_action_dim,
        "action_definition": (
            "native official pd_ee_delta_pos"
            if dataset_format == OFFICIAL_FORMAT
            else "M1-W pd_ee_delta_pose[:, [dx, dy, dz, gripper]] projection"
        ),
        "native_action_warning": dataset_format == M1W_PROJECTED_FORMAT,
        "rgb_cameras": list(CAMERA_NAMES),
        "rgb_shape_per_camera": list(EXPECTED_RGB_SHAPE),
        "h5": {
            "path": str(h5_path),
            "size_bytes": h5_path.stat().st_size,
            "mtime_ns": h5_path.stat().st_mtime_ns,
        },
        "split_manifest": (
            {
                "path": str(split_manifest),
                "sha256": split_sha256,
            }
            if split_manifest is not None
            else None
        ),
        "train_episode_ids": str(ids_path),
        "train_episode_count": len(episode_ids),
        "train_transition_count": int(sum(lengths.values())),
        "episode_ids_first": episode_ids[:5],
        "episode_ids_last": episode_ids[-5:],
        "lazy_loading": {
            "h5_handle_per_process": True,
            "dataloader_workers_supported": True,
            "eager_host_or_gpu_copy": False,
        },
    }
    (output_dir / "adapter_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-format", choices=SUPPORTED_FORMATS, required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--num-demos", type=int)
    parser.add_argument("--task", default=EXPECTED_TASK)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = prepare_ablation(
        dataset_format=args.dataset_format,
        h5_path=args.h5,
        output_dir=args.output_dir,
        split_manifest=args.split_manifest,
        num_demos=args.num_demos,
        task=args.task,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "EXPECTED_ACTION_DIM",
    "EXPECTED_CONTROL_MODE",
    "EXPECTED_STATE_DIM",
    "M1W_PROJECTED_FORMAT",
    "OFFICIAL_FORMAT",
    "SUPPORTED_FORMATS",
    "M1WFlattenRGBObservationWrapper",
    "StackCube18D4DDataset",
    "main",
    "prepare_ablation",
]


if __name__ == "__main__":
    raise SystemExit(main())
