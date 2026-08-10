"""Capture and export ManiSkill closed-loop rollout episodes for local viewing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_practice.evaluation.common import write_json_atomic
from robot_practice.rerun_launcher import LIBERO_APPLICATION_ID

from .replay_visualization import (
    ReplayEpisode,
    episode_artifact_stem,
    write_episode_recording,
    write_episode_video,
)

VIEWER_KIND = "libero"
RECORDING_LABEL = "closed-loop rollout"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _nested_value(mapping: Mapping[str, Any], path: Sequence[str]) -> Any | None:
    current: Any = mapping
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _batch_array(
    observation: Mapping[str, Any],
    path: Sequence[str],
    *,
    batch_size: int,
    shape_tail: tuple[int, ...],
    dtype: Any,
) -> np.ndarray:
    value = _nested_value(observation, path)
    if value is None:
        raise KeyError(f"Observation is missing {'/'.join(path)}")
    array = _to_numpy(value)
    if array.ndim == len(shape_tail):
        array = array[None, ...]
    expected = (batch_size, *shape_tail)
    if array.shape != expected:
        raise ValueError(f"{'/'.join(path)} must have shape {expected}, got {array.shape}")
    result = np.asarray(array, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{'/'.join(path)} contains non-finite values")
    return result


def _optional_batch_array(
    observation: Mapping[str, Any],
    path: Sequence[str],
    *,
    batch_size: int,
    shape_tail: tuple[int, ...],
) -> np.ndarray | None:
    if _nested_value(observation, path) is None:
        return None
    return _batch_array(
        observation,
        path,
        batch_size=batch_size,
        shape_tail=shape_tail,
        dtype=np.float32,
    )


def _observation_arrays(
    observation: Mapping[str, Any],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    images = _batch_array(
        observation,
        ("sensor_data", "base_camera", "rgb"),
        batch_size=batch_size,
        shape_tail=(128, 128, 3),
        dtype=np.uint8,
    )
    qpos = _batch_array(
        observation,
        ("agent", "qpos"),
        batch_size=batch_size,
        shape_tail=(9,),
        dtype=np.float32,
    )
    qvel = _batch_array(
        observation,
        ("agent", "qvel"),
        batch_size=batch_size,
        shape_tail=(9,),
        dtype=np.float32,
    )
    goal = _batch_array(
        observation,
        ("extra", "goal_pos"),
        batch_size=batch_size,
        shape_tail=(3,),
        dtype=np.float32,
    )
    state = np.concatenate((qpos, qvel, goal), axis=1)
    tcp_pose = _optional_batch_array(
        observation,
        ("extra", "tcp_pose"),
        batch_size=batch_size,
        shape_tail=(7,),
    )
    obj_pose = _optional_batch_array(
        observation,
        ("extra", "obj_pose"),
        batch_size=batch_size,
        shape_tail=(7,),
    )
    cube_position = None if obj_pose is None else obj_pose[:, :3]
    return images, state, tcp_pose, cube_position


@dataclass
class _EpisodeCapture:
    episode_id: int
    seed: int
    images: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    is_grasped: list[bool] = field(default_factory=list)
    tcp_pose: list[np.ndarray] = field(default_factory=list)
    cube_position: list[np.ndarray] = field(default_factory=list)

    def to_replay_episode(self) -> ReplayEpisode:
        if not self.actions:
            raise ValueError(f"Episode {self.episode_id} has no captured actions")
        return ReplayEpisode(
            episode_id=self.episode_id,
            seed=self.seed,
            metadata_record={"episode_seed": self.seed},
            images=np.stack(self.images, axis=0).astype(np.uint8, copy=False),
            state=np.stack(self.state, axis=0).astype(np.float32, copy=False),
            actions=np.stack(self.actions, axis=0).astype(np.float32, copy=False),
            success=np.asarray(self.success, dtype=bool),
            terminated=np.asarray(self.terminated, dtype=bool),
            truncated=np.asarray(self.truncated, dtype=bool),
            is_grasped=np.asarray(self.is_grasped, dtype=bool),
            tcp_pose=(
                None
                if not self.tcp_pose
                else np.stack(self.tcp_pose, axis=0).astype(np.float32, copy=False)
            ),
            cube_position=(
                None
                if not self.cube_position
                else np.stack(self.cube_position, axis=0).astype(np.float32, copy=False)
            ),
            goal_position=np.stack(self.state, axis=0)[:, 18:21].astype(np.float32, copy=False),
        )


class ClosedLoopVisualizationRecorder:
    """Keep selected live vector-env episodes and export them after evaluation."""

    def __init__(self, *, max_episodes: int, fps: float = 20.0) -> None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")
        self.max_episodes = int(max_episodes)
        self.fps = float(fps)
        self._captures: dict[int, _EpisodeCapture] = {}
        self._current: dict[int, _EpisodeCapture] = {}

    @property
    def captured_episode_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._captures))

    def start_batch(
        self,
        observation: Mapping[str, Any],
        *,
        seeds: Sequence[int],
        episode_index_offset: int,
        signals: Mapping[str, np.ndarray],
    ) -> None:
        if self._current:
            raise RuntimeError("A visualization batch is already active")
        batch_size = len(seeds)
        images, state, tcp_pose, cube_position = _observation_arrays(
            observation,
            batch_size=batch_size,
        )
        for local_index, seed in enumerate(seeds):
            episode_id = episode_index_offset + local_index
            if len(self._captures) >= self.max_episodes:
                break
            capture = _EpisodeCapture(episode_id=episode_id, seed=int(seed))
            capture.images.append(np.array(images[local_index], copy=True))
            capture.state.append(np.array(state[local_index], copy=True))
            capture.is_grasped.append(bool(signals["is_grasped"][local_index]))
            if tcp_pose is not None:
                capture.tcp_pose.append(np.array(tcp_pose[local_index], copy=True))
            if cube_position is not None:
                capture.cube_position.append(np.array(cube_position[local_index], copy=True))
            self._captures[episode_id] = capture
            self._current[local_index] = capture

    def record_step(
        self,
        observation: Mapping[str, Any],
        *,
        action: np.ndarray,
        signals: Mapping[str, np.ndarray],
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        if not self._current:
            return
        batch_size = int(action.shape[0])
        images, state, tcp_pose, cube_position = _observation_arrays(
            observation,
            batch_size=batch_size,
        )
        action_array = _to_numpy(action).astype(np.float32, copy=False)
        for local_index, capture in self._current.items():
            capture.actions.append(np.array(action_array[local_index], copy=True))
            capture.images.append(np.array(images[local_index], copy=True))
            capture.state.append(np.array(state[local_index], copy=True))
            capture.success.append(bool(signals["success"][local_index]))
            capture.terminated.append(bool(terminated[local_index]))
            capture.truncated.append(bool(truncated[local_index]))
            capture.is_grasped.append(bool(signals["is_grasped"][local_index]))
            if capture.tcp_pose:
                if tcp_pose is None:
                    raise ValueError("tcp_pose disappeared during a visualization batch")
                capture.tcp_pose.append(np.array(tcp_pose[local_index], copy=True))
            if capture.cube_position:
                if cube_position is None:
                    raise ValueError("obj_pose disappeared during a visualization batch")
                capture.cube_position.append(np.array(cube_position[local_index], copy=True))

    def end_batch(self) -> None:
        """Release the active vector-batch references before the next reset batch."""

        self._current.clear()

    def finalize(self, output_dir: str | Path, *, control_mode: str) -> dict[str, Any]:
        if self._current:
            raise RuntimeError("Cannot finalize while a visualization batch is active")
        output = Path(output_dir).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to write into non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, Any]] = []
        for episode_id in self.captured_episode_ids:
            episode = self._captures[episode_id].to_replay_episode()
            artifact_stem = episode_artifact_stem(episode)
            recording_path = output / f"{artifact_stem}.rrd"
            write_episode_recording(
                episode,
                output_path=recording_path,
                fps=self.fps,
                control_mode=control_mode,
                include_privileged_overlay=True,
                application_id=LIBERO_APPLICATION_ID,
                viewer_kind=VIEWER_KIND,
                recording_label=RECORDING_LABEL,
            )
            video_path = write_episode_video(
                episode,
                output_path=output / f"{artifact_stem}.mp4",
                fps=self.fps,
            )
            summaries.append(
                {
                    "episode_id": episode.episode_id,
                    "seed": episode.seed,
                    "action_dim": episode.action_dim,
                    "action_names": list(episode.action_names),
                    "action_steps": episode.action_steps,
                    "frame_count": int(episode.images.shape[0]),
                    "success_once": episode.success_once,
                    "success_at_end": episode.success_at_end,
                    "result": episode.result_label,
                    "first_success_step": episode.first_success_step,
                    "recording": recording_path.as_posix(),
                    "video": video_path.as_posix(),
                    "viewer_kind": VIEWER_KIND,
                    "application_id": LIBERO_APPLICATION_ID,
                }
            )
        manifest = {
            "schema_version": 1,
            "source": "maniskill_closed_loop_rollout",
            "task": "PickCube-v1",
            "control_mode": control_mode,
            "fps": self.fps,
            "viewer": {
                "kind": VIEWER_KIND,
                "application_id": LIBERO_APPLICATION_ID,
                "launcher": "robot-rerun",
                "semantics": "Closed-loop ManiSkill rollout; not demonstration replay.",
            },
            "selected_episode_ids": list(self.captured_episode_ids),
            "episodes": summaries,
        }
        write_json_atomic(output / "visualization_manifest.json", manifest)
        return manifest
