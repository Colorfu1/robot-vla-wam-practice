"""M1-W live-RGB adapter for the vendored ManiSkill Diffusion Policy.

The official ManiSkill Diffusion Policy example expects an HDF5 file whose
observations match the environment wrapper used by the example.  M1-W has a
different, deliberate contract:

* native ``pd_ee_delta_pose`` actions (7 dimensions);
* two live RGB cameras (base and hand), stored as ``NHWC`` uint8 arrays;
* state is ``agent.qpos + agent.qvel`` (18 dimensions), without privileged
  ``extra`` observations;
* the frozen HDF5 is too large to eagerly copy to CPU/GPU memory.

This module is intentionally local to the practice repository.  It keeps the
official model, noise scheduler, optimizer, and evaluation implementation in
the vendored baseline, while supplying the M1-W dataset and observation
wrapper through an explicit ``dataset_format=m1w_live_rgb`` switch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import gymnasium as gym
import h5py
import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import Dataset

M1W_DATASET_FORMAT = "m1w_live_rgb"
EXPECTED_TASK = "StackCube-v1"
EXPECTED_CONTROL_MODE = "pd_ee_delta_pose"
EXPECTED_ACTION_DIM = 7
EXPECTED_STATE_DIM = 18
EXPECTED_RGB_SHAPE = (128, 128, 3)
CAMERA_NAMES = ("base_camera", "hand_camera")


def _concat_last(values: Sequence[Any]) -> Any:
    """Concatenate NumPy or torch values on their feature/channel axis."""

    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(list(values), dim=-1)
    return np.concatenate(list(values), axis=-1)


def _read_episode_ids(path: str | Path) -> list[int]:
    """Read a plain ID list, or a split-manifest object containing ``train``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "episode_ids" in payload:
            payload = payload["episode_ids"]
        elif "train" in payload:
            payload = payload["train"]
        else:
            raise ValueError(
                f"episode id file {path} must contain 'episode_ids' or 'train'"
            )
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"episode id file {path} must contain a non-empty list")
    ids = [int(value) for value in payload]
    if len(set(ids)) != len(ids):
        raise ValueError(f"episode id file {path} contains duplicate IDs")
    return ids


def _traj_name(episode_id: int) -> str:
    return f"traj_{int(episode_id)}"


class M1WLiveRGBDataset(Dataset):
    """Lazy Diffusion Policy windows over the frozen M1-W HDF5.

    The dataset opens the HDF5 once in the training process and reads only the
    requested observation/action window in ``__getitem__``.  The official
    DDP launcher deliberately uses ``num_dataload_workers=0``: every torchrun
    rank owns one independent HDF5 handle and no complete copy is made in
    either host RAM or GPU memory.
    """

    def __init__(
        self,
        data_path: str | Path,
        episode_ids_file: str | Path,
        obs_horizon: int,
        pred_horizon: int,
        control_mode: str = EXPECTED_CONTROL_MODE,
        expected_action_dim: int = EXPECTED_ACTION_DIM,
    ) -> None:
        self.data_path = Path(data_path)
        self.episode_ids_file = Path(episode_ids_file)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.control_mode = control_mode
        self.expected_action_dim = int(expected_action_dim)
        if self.obs_horizon < 1 or self.pred_horizon < 1:
            raise ValueError("obs_horizon and pred_horizon must be positive")
        if self.obs_horizon > self.pred_horizon:
            raise ValueError("M1-W requires obs_horizon <= pred_horizon")
        if self.control_mode != EXPECTED_CONTROL_MODE:
            raise ValueError(
                f"M1-W adapter only supports {EXPECTED_CONTROL_MODE}, got {control_mode}"
            )
        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        self.episode_ids = _read_episode_ids(self.episode_ids_file)
        self._file = h5py.File(self.data_path, "r")
        self._episode_lengths: dict[int, int] = {}
        self.slices: list[tuple[int, int, int, int]] = []
        self._validate_and_index()
        self.pad_action_arm = np.zeros((self.expected_action_dim - 1,), dtype=np.float32)

    def _validate_and_index(self) -> None:
        total_transitions = 0
        pad_before = self.obs_horizon - 1
        pad_after = self.pred_horizon - self.obs_horizon
        for episode_id in self.episode_ids:
            name = _traj_name(episode_id)
            if name not in self._file:
                raise KeyError(f"M1-W episode {name} is missing from {self.data_path}")
            trajectory = self._file[name]
            actions = trajectory.get("actions")
            if actions is None or actions.ndim != 2:
                raise ValueError(f"{name}/actions must be a 2D dataset")
            length, action_dim = map(int, actions.shape)
            if action_dim != self.expected_action_dim:
                raise ValueError(
                    f"{name}/actions has dim {action_dim}, expected {self.expected_action_dim}"
                )
            required = (
                "obs/agent/qpos",
                "obs/agent/qvel",
                "obs/sensor_data/base_camera/rgb",
                "obs/sensor_data/hand_camera/rgb",
            )
            for key in required:
                if key not in trajectory:
                    raise KeyError(f"{name}/{key} is missing from {self.data_path}")
            qpos = trajectory["obs/agent/qpos"]
            qvel = trajectory["obs/agent/qvel"]
            base_rgb = trajectory["obs/sensor_data/base_camera/rgb"]
            hand_rgb = trajectory["obs/sensor_data/hand_camera/rgb"]
            if qpos.shape != (length + 1, 9) or qvel.shape != (length + 1, 9):
                raise ValueError(
                    f"{name} qpos/qvel must be ({length + 1}, 9), "
                    f"got {qpos.shape} and {qvel.shape}"
                )
            for camera, value in (("base_camera", base_rgb), ("hand_camera", hand_rgb)):
                if value.shape != (length + 1, *EXPECTED_RGB_SHAPE):
                    raise ValueError(
                        f"{name}/{camera}/rgb must be ({length + 1}, 128, 128, 3), "
                        f"got {value.shape}"
                    )
                if value.dtype != np.uint8:
                    raise ValueError(
                        f"{name}/{camera}/rgb must be uint8, got {value.dtype}"
                    )
            self._episode_lengths[episode_id] = length
            total_transitions += length
            self.slices.extend(
                (
                    episode_id,
                    start,
                    start + self.pred_horizon,
                    length,
                )
                for start in range(-pad_before, length - self.pred_horizon + pad_after)
            )
        if not self.slices:
            raise ValueError("M1-W dataset produced no training windows")
        print(
            "M1-W lazy dataset: "
            f"episodes={len(self.episode_ids)} transitions={total_transitions} "
            f"windows={len(self.slices)} h5={self.data_path}",
            flush=True,
        )

    @staticmethod
    def _pad_observation(values: np.ndarray, start: int, obs_horizon: int) -> np.ndarray:
        if start >= 0:
            return values[start : start + obs_horizon]
        prefix = np.repeat(values[0:1], -start, axis=0)
        return np.concatenate((prefix, values[0 : start + obs_horizon]), axis=0)

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        episode_id, start, end, length = self.slices[index]
        trajectory = self._file[_traj_name(episode_id)]
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
        rgb = np.concatenate((base_rgb, hand_rgb), axis=-1)
        rgb = np.transpose(rgb, (0, 3, 1, 2))

        actions = trajectory["actions"][max(0, start) : min(length, end)]
        if start < 0:
            actions = np.concatenate(
                (np.repeat(actions[0:1], -start, axis=0), actions), axis=0
            )
        if end > length:
            pad_action = np.concatenate(
                (self.pad_action_arm, np.asarray(actions[-1, -1:], dtype=np.float32)),
                axis=0,
            )
            actions = np.concatenate(
                (actions, np.repeat(pad_action[None], end - length, axis=0)), axis=0
            )

        state = np.concatenate((qpos, qvel), axis=-1).astype(np.float32, copy=False)
        actions = np.asarray(actions, dtype=np.float32)
        if state.shape != (self.obs_horizon, EXPECTED_STATE_DIM):
            raise RuntimeError(f"M1-W state window has unexpected shape {state.shape}")
        if rgb.shape[0] != self.obs_horizon or actions.shape != (
            self.pred_horizon,
            self.expected_action_dim,
        ):
            raise RuntimeError(
                f"M1-W window shapes are state={state.shape}, rgb={rgb.shape}, "
                f"actions={actions.shape}"
            )
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

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown ordering
        try:
            self.close()
        except Exception:
            pass


class M1WFlattenRGBObservationWrapper(gym.ObservationWrapper):
    """Flatten M1-W raw ManiSkill observations to ``state`` and ``rgb``.

    This mirrors the official ``FlattenRGBDObservationWrapper`` for RGB, but
    intentionally excludes ``observation['extra']`` and retains only qpos and
    qvel.  It works for both NumPy CPU observations and torch GPU observations.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.base_env = env.unwrapped
        raw_space = env.observation_space
        try:
            qpos_space = raw_space["agent"]["qpos"]
            qvel_space = raw_space["agent"]["qvel"]
            sensor_space = raw_space["sensor_data"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "M1-W wrapper requires raw agent qpos/qvel and sensor_data observations"
            ) from exc
        sensor_keys = getattr(sensor_space, "spaces", sensor_space)
        camera_names = [name for name in CAMERA_NAMES if name in sensor_keys]
        if len(camera_names) != len(CAMERA_NAMES):
            raise ValueError(
                f"M1-W wrapper requires cameras {CAMERA_NAMES}, got {list(sensor_space)}"
            )
        rgb_spaces = [sensor_space[name]["rgb"] for name in camera_names]
        if any(space.shape[-3:] != EXPECTED_RGB_SHAPE for space in rgb_spaces):
            raise ValueError(
                "M1-W wrapper requires 128x128x3 camera RGB, got "
                f"{[space.shape for space in rgb_spaces]}"
            )
        state_shape = tuple(qpos_space.shape[:-1]) + (qpos_space.shape[-1] + qvel_space.shape[-1],)
        rgb_shape = tuple(rgb_spaces[0].shape[:-1]) + (len(rgb_spaces) * 3,)
        if state_shape[-1] != EXPECTED_STATE_DIM:
            raise ValueError(f"M1-W state must be 18D, got {state_shape}")
        self.camera_names = tuple(camera_names)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
                ),
                "rgb": spaces.Box(low=0, high=255, shape=rgb_shape, dtype=np.uint8),
            }
        )
        # ManiSkill's FrameStack reads ``base_env._init_raw_obs`` when it is
        # constructed.  Update the base environment just like the official
        # FlattenRGBDObservationWrapper does, otherwise FrameStack would see
        # the pre-wrapper sensor_data dictionary.
        self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        agent = observation["agent"]
        state = _concat_last((agent["qpos"], agent["qvel"]))
        sensor_data = observation["sensor_data"]
        rgb = _concat_last(tuple(sensor_data[name]["rgb"] for name in self.camera_names))
        return {"state": state, "rgb": rgb}


def _validate_split_manifest(split_manifest: Path, h5_path: Path) -> dict[str, Any]:
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    required_splits = ("train", "development", "test")
    if any(split not in payload for split in required_splits):
        raise ValueError(f"split manifest must contain {required_splits}: {split_manifest}")
    split_ids = {split: [int(value) for value in payload[split]] for split in required_splits}
    all_ids = [value for values in split_ids.values() for value in values]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("M1-W split manifest contains overlapping episode IDs")
    with h5py.File(h5_path, "r") as handle:
        available = set(handle.keys())
        for split, ids in split_ids.items():
            missing = [_traj_name(value) for value in ids if _traj_name(value) not in available]
            if missing:
                raise KeyError(f"{split_manifest}: missing H5 trajectories {missing[:5]}")
        sample_id = split_ids["train"][0]
        sample = handle[_traj_name(sample_id)]
        action_shape = tuple(sample["actions"].shape)
        qpos_dim = int(sample["obs/agent/qpos"].shape[-1])
        qvel_dim = int(sample["obs/agent/qvel"].shape[-1])
        state_shape = (qpos_dim + qvel_dim,)
        rgb_shape = tuple(sample["obs/sensor_data/base_camera/rgb"].shape[1:])
        if action_shape[-1] != EXPECTED_ACTION_DIM:
            raise ValueError(f"M1-W H5 action dim is {action_shape[-1]}, expected 7")
        if state_shape[-1] != EXPECTED_STATE_DIM:
            raise ValueError(f"M1-W H5 state dim is {state_shape[-1]}, expected 18")
        if rgb_shape != EXPECTED_RGB_SHAPE:
            raise ValueError(f"M1-W H5 RGB shape is {rgb_shape}, expected {EXPECTED_RGB_SHAPE}")
    return {"splits": split_ids, "sample_action_shape": action_shape, "sample_rgb_shape": rgb_shape}


def prepare_adapter(
    h5_path: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    task: str = EXPECTED_TASK,
) -> dict[str, Any]:
    """Validate the frozen input and write small, portable adapter sidecars."""

    if task != EXPECTED_TASK:
        raise ValueError(f"M1-W official adapter currently supports {EXPECTED_TASK}, got {task}")
    h5_path = Path(h5_path)
    split_manifest = Path(split_manifest)
    output_dir = Path(output_dir)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not split_manifest.is_file():
        raise FileNotFoundError(split_manifest)
    checked = _validate_split_manifest(split_manifest, h5_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids_path = output_dir / "train_episode_ids.json"
    train_ids_path.write_text(
        json.dumps(checked["splits"]["train"], indent=2) + "\n", encoding="utf-8"
    )
    split_sha256 = __import__("hashlib").sha256(split_manifest.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "dataset_format": M1W_DATASET_FORMAT,
        "task": task,
        "control_mode": EXPECTED_CONTROL_MODE,
        "obs_mode": "rgb",
        "action_dim": EXPECTED_ACTION_DIM,
        "state_dim": EXPECTED_STATE_DIM,
        "state_definition": "agent.qpos(9) + agent.qvel(9)",
        "rgb_cameras": list(CAMERA_NAMES),
        "rgb_shape_per_camera": list(EXPECTED_RGB_SHAPE),
        "h5": {
            "path": str(h5_path),
            "size_bytes": h5_path.stat().st_size,
            "mtime_ns": h5_path.stat().st_mtime_ns,
        },
        "split_manifest": {
            "path": str(split_manifest),
            "sha256": split_sha256,
            "counts": {key: len(value) for key, value in checked["splits"].items()},
        },
        "train_episode_ids": str(train_ids_path),
        "lazy_loading": {
            "h5_handle_per_torchrun_rank": True,
            "num_dataload_workers": 0,
            "eager_host_or_gpu_copy": False,
        },
    }
    (output_dir / "adapter_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default=EXPECTED_TASK)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = prepare_adapter(args.h5, args.split_manifest, args.output_dir, args.task)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
