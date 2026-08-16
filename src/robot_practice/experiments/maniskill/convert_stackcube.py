"""Convert the GPU-native StackCube D_core HDF5 into a LeRobot dataset.

This is deliberately separate from the M0 PickCube converter.  StackCube has
no ``extra.goal_pos`` input: its visual student observes two RGB cameras and
the 18D proprioceptive vector ``qpos + qvel``.  The input HDF5 is already
materialized on ``physx_cuda``; this module only reads it and never creates or
steps a ManiSkill environment.
"""

from __future__ import annotations

import argparse
import errno
import importlib.metadata
import json
import logging
import os
import shutil
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

LOGGER = logging.getLogger(__name__)

TASK_ID = "StackCube-v1"
DEFAULT_FPS = 20
IMAGE_SHAPE = (128, 128, 3)
STATE_SHAPE = (18,)
ACTION_SHAPE = (7,)
BASE_CAMERA_KEY = "observation.images.base_camera"
HAND_CAMERA_KEY = "observation.images.hand_camera"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
FEATURE_KEYS = (BASE_CAMERA_KEY, HAND_CAMERA_KEY, STATE_KEY, ACTION_KEY)
SCHEMA_VERSION = 1


def dataset_features() -> dict[str, dict[str, Any]]:
    """Return the frozen StackCube LeRobot feature contract."""

    return {
        BASE_CAMERA_KEY: {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        HAND_CAMERA_KEY: {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": STATE_SHAPE,
            "names": ["state"],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": ACTION_SHAPE,
            "names": ["actions"],
        },
    }


def _portable(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _publish_directory(temporary: Path, output_dir: Path) -> None:
    """Publish a finished dataset and handle the repository's cross-device FS."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {output_dir}")
    try:
        temporary.rename(output_dir)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        LOGGER.warning("Dataset rename crossed filesystems; copying to %s", output_dir)
    try:
        shutil.copytree(temporary, output_dir, symlinks=False)
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    shutil.rmtree(temporary)


def _episode_id(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected HDF5 group {name!r}; expected traj_<id>")
    return int(name.removeprefix("traj_"))


def _read_metadata(replay_h5: Path) -> dict[str, Any]:
    path = replay_h5.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"D_core metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _metadata_records(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("D_core metadata episodes must be a list")
    records: dict[int, dict[str, Any]] = {}
    for item in episodes:
        if not isinstance(item, Mapping):
            raise TypeError("Each D_core episode record must be an object")
        episode_id = int(item["episode_id"])
        if episode_id in records:
            raise ValueError(f"Duplicate D_core episode_id={episode_id}")
        records[episode_id] = dict(item)
    return records


def _array_at(group: h5py.Group, path: tuple[str, ...], index: int) -> np.ndarray:
    current: Any = group
    for part in path:
        if not isinstance(current, h5py.Group) or part not in current:
            raise KeyError(f"HDF5 episode {group.name} is missing {'/'.join(path)}")
        current = current[part]
    if not isinstance(current, h5py.Dataset):
        raise TypeError(f"HDF5 path {'/'.join(path)} is not a dataset")
    return np.asarray(current[index])


def _frame_from_episode(episode: h5py.Group, index: int) -> dict[str, Any]:
    base = _array_at(episode, ("obs", "sensor_data", "base_camera", "rgb"), index)
    hand = _array_at(episode, ("obs", "sensor_data", "hand_camera", "rgb"), index)
    qpos = _array_at(episode, ("obs", "agent", "qpos"), index).reshape(-1)
    qvel = _array_at(episode, ("obs", "agent", "qvel"), index).reshape(-1)
    action = np.asarray(episode["actions"][index], dtype=np.float32).reshape(-1)
    for name, value in (("base_camera", base), ("hand_camera", hand)):
        if value.shape != IMAGE_SHAPE or value.dtype != np.uint8:
            raise ValueError(
                f"{episode.name} {name} RGB must be uint8 {IMAGE_SHAPE}, "
                f"got dtype={value.dtype}, shape={value.shape}"
            )
    if qpos.shape != (9,) or qvel.shape != (9,):
        raise ValueError(f"{episode.name} qpos/qvel must be (9,), got {qpos.shape}/{qvel.shape}")
    if action.shape != ACTION_SHAPE or not np.isfinite(action).all():
        raise ValueError(f"{episode.name} action must be finite {ACTION_SHAPE}, got {action.shape}")
    state = np.concatenate((qpos, qvel)).astype(np.float32, copy=False)
    if not np.isfinite(state).all():
        raise ValueError(f"{episode.name} state contains non-finite values")
    return {
        BASE_CAMERA_KEY: base,
        HAND_CAMERA_KEY: hand,
        STATE_KEY: state,
        ACTION_KEY: action,
        "task": TASK_ID,
    }


def iter_episode_frames(episode: h5py.Group) -> Iterator[dict[str, Any]]:
    """Yield obs[t] with action[t], excluding the final post-action obs."""

    if "actions" not in episode or "obs" not in episode:
        raise KeyError(f"{episode.name} must contain actions and obs")
    count = int(episode["actions"].shape[0])
    base_count = int(episode["obs"]["sensor_data"]["base_camera"]["rgb"].shape[0])
    hand_count = int(episode["obs"]["sensor_data"]["hand_camera"]["rgb"].shape[0])
    qpos_count = int(episode["obs"]["agent"]["qpos"].shape[0])
    qvel_count = int(episode["obs"]["agent"]["qvel"].shape[0])
    expected = count + 1
    if (base_count, hand_count, qpos_count, qvel_count) != (expected,) * 4:
        raise ValueError(
            f"{episode.name} must have obs=T+1 for both cameras/qpos/qvel; "
            f"actions={count}, observed={(base_count, hand_count, qpos_count, qvel_count)}"
        )
    if count <= 0:
        raise ValueError(f"{episode.name} contains no actions")
    for index in range(count):
        yield _frame_from_episode(episode, index)


def _episode_order(metadata: Mapping[str, Any], available: set[int]) -> list[int]:
    records = _metadata_records(metadata)
    unknown = set(records) - available
    if unknown:
        raise ValueError(f"D_core metadata references missing HDF5 episodes: {sorted(unknown)[:5]}")
    missing = available - set(records)
    if missing:
        raise ValueError(f"D_core HDF5 contains episodes missing from metadata: {sorted(missing)[:5]}")
    # D_core metadata order is the frozen stratified order.  Preserve it in
    # the LeRobot episode indices, while retaining source_episode_id in the
    # conversion manifest for traceability.
    return [int(item["episode_id"]) for item in metadata["episodes"]]


def convert_stackcube_dcore(
    *,
    replay_h5: Path,
    output_dir: Path,
    repo_id: str,
    fps: int = DEFAULT_FPS,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Create the StackCube D_core LeRobot dataset without simulator calls."""

    replay_h5 = replay_h5.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not replay_h5.is_file():
        raise FileNotFoundError(f"D_core HDF5 does not exist: {replay_h5}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {output_dir}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    metadata = _read_metadata(replay_h5)
    if metadata.get("env_info", {}).get("env_id") != TASK_ID:
        raise ValueError(f"D_core task must be {TASK_ID!r}")
    env_kwargs = metadata.get("env_info", {}).get("env_kwargs", {})
    if env_kwargs.get("sim_backend") != "physx_cuda":
        raise ValueError("StackCube formal dataset must come from sim_backend=physx_cuda")
    if env_kwargs.get("control_mode") != "pd_ee_delta_pose":
        raise ValueError("StackCube formal dataset must use pd_ee_delta_pose")

    with h5py.File(replay_h5, "r") as handle:
        available = {_episode_id(name) for name in handle if name.startswith("traj_")}
        order = _episode_order(metadata, available)
        if max_episodes is not None:
            order = order[:max_episodes]
        records = _metadata_records(metadata)
        if not order:
            raise ValueError("D_core contains no episodes to convert")
        selected: list[dict[str, Any]] = []
        action_count = 0
        dataset = None
        finalized = False
        temporary = output_dir.parent / f".{output_dir.name}.incomplete-{os.getpid()}"
        if temporary.exists():
            raise FileExistsError(f"Refusing to reuse incomplete conversion directory: {temporary}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=temporary,
                robot_type="panda",
                fps=fps,
                features=dataset_features(),
                use_videos=True,
                image_writer_threads=4,
                vcodec="libsvtav1",
            )
            for converted_index, episode_id in enumerate(order):
                episode = handle[f"traj_{episode_id}"]
                frame_count = 0
                for frame in iter_episode_frames(episode):
                    dataset.add_frame(frame)
                    frame_count += 1
                dataset.save_episode(parallel_encoding=False)
                action_count += frame_count
                record = records[episode_id]
                selected.append(
                    {
                        "dataset_episode_index": converted_index,
                        "h5_episode_id": episode_id,
                        "source_episode_id": int(
                            record.get("source_episode_id", episode.attrs.get("source_episode_id", episode_id))
                        ),
                        "start_state_fingerprint": record.get(
                            "source_start_state_fingerprint", record.get("target_start_state_fingerprint")
                        ),
                        "frames": frame_count,
                        "success_at_end": bool(record.get("success", True)),
                    }
                )
                if (converted_index + 1) % 25 == 0 or converted_index + 1 == len(order):
                    LOGGER.info("converted StackCube episodes %d/%d", converted_index + 1, len(order))
            dataset.finalize()
            finalized = True
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "created_at": iso_timestamp(),
                "task": TASK_ID,
                "repo_id": repo_id,
                "source": {
                    "d_core_h5": _portable(replay_h5),
                    "d_core_json": _portable(replay_h5.with_suffix(".json")),
                    "route": "frozen_gpu_carrier_live_rgb_replay_7d",
                    "backend": "physx_cuda",
                    "control_mode": "pd_ee_delta_pose",
                },
                "observation": {
                    "camera_keys": [BASE_CAMERA_KEY, HAND_CAMERA_KEY],
                    "camera_shape": list(IMAGE_SHAPE),
                    "state_key": STATE_KEY,
                    "state_shape": list(STATE_SHAPE),
                    "state_fields": ["agent.qpos[0:9]", "agent.qvel[0:9]"],
                },
                "action": {"key": ACTION_KEY, "shape": list(ACTION_SHAPE)},
                "timeline": "obs[t] + action[t] for t in T transitions; obs[T] is not a training frame",
                "split_manifest": _portable(replay_h5.parent / "split_manifest.json"),
                "episodes": selected,
                "output": {
                    "episodes": len(selected),
                    "frames": action_count,
                    "all_source_success_labels_true": all(item["success_at_end"] for item in selected),
                },
                "runtime": {
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "mani_skill": _version("mani_skill"),
                    "lerobot": _version("lerobot"),
                },
            }
            write_json_atomic(temporary / "stackcube_conversion_manifest.json", manifest)
            _publish_directory(temporary, output_dir)
        except BaseException:
            if dataset is not None and not finalized:
                try:
                    dataset.finalize()
                except BaseException:
                    LOGGER.exception("Could not finalize incomplete StackCube dataset")
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    return {
        "output_dir": _portable(output_dir),
        "episodes": len(selected),
        "frames": action_count,
        "task": TASK_ID,
        "state_dim": STATE_SHAPE[0],
        "action_dim": ACTION_SHAPE[0],
        "backend": "physx_cuda",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = convert_stackcube_dcore(
        replay_h5=arguments.replay_h5,
        output_dir=arguments.output_dir,
        repo_id=arguments.repo_id,
        fps=arguments.fps,
        max_episodes=arguments.max_episodes,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
