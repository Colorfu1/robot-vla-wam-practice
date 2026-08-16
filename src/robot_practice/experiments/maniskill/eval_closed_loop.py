"""Native ManiSkill closed-loop evaluation for the M0 PickCube probe.

The evaluator deliberately stays outside LeRobot's environment wrappers.  It
can run one of three systems in an independent remote process: a random
baseline, the privileged PPO teacher, or the RGB/proprioceptive diffusion
student.  All systems use the same frozen reset batch, and success is read
only from ManiSkill's native ``info["success"]`` signal.  Pass
``--visualize-episodes=N`` on a remote run to retain RGB/action traces and
export local-viewer MP4/RRD artifacts after the rollout.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

from .split import TASK_ID, load_seed_sets, reset_batches_for_set
from .rollout_contract import require_rollout_backend

IMAGE_KEY = "observation.images.base_camera"
STATE_KEY = "observation.state"
STATE_FIELDS = (
    "agent.qpos[0:9]",
    "agent.qvel[0:9]",
    "extra.goal_pos[0:3]",
)
PRIVILEGED_STATE_FIELDS = (
    "agent.qpos[0:9]",
    "agent.qvel[0:9]",
    "extra.is_grasped[0:1]",
    "extra.tcp_pose[0:7]",
    "extra.goal_pos[0:3]",
    "extra.obj_pose[0:7]",
    "extra.tcp_to_obj_pos[0:3]",
    "extra.obj_to_goal_pos[0:3]",
)
EXCLUDED_STATE_FIELDS = (
    "extra.is_grasped",
    "extra.obj_pose",
    "extra.tcp_to_obj_pos",
    "extra.obj_to_goal_pos",
    "extra.tcp_pose",
)
STAGE_SIGNAL_KEYS = ("is_grasped", "is_obj_placed", "is_robot_static")
CONTROL_ACTION_DIMS = {
    "pd_ee_delta_pos": 4,
    "pd_ee_delta_pose": 7,
    "pd_joint_delta_pos": 8,
}
ACTION_METRIC_NAMES = (
    "translation_norm",
    "rotation_norm",
    "gripper_abs",
)
DEFAULT_MAX_STEPS = 50
DEFAULT_POLICY_SEED = 2000
DEFAULT_N_ACTION_STEPS = 8
EVALUATOR_SCHEMA_VERSION = 2
TEACHER_STUDENT_NOTE = (
    "The demonstrations come from a privileged expert, while the diffusion policy is a visual "
    "student; their success rates are references under different observations and objectives and "
    "must not be compared as a pure perception-causal effect."
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_torch(value: Any, *, dtype: Any | None = None, device: str | None = None) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def _select_batch(value: Any, indices: np.ndarray, *, batch_size: int) -> Any:
    """Select a batch slice from nested ManiSkill observations without list indexing warnings."""

    import torch

    if isinstance(value, Mapping):
        return {
            key: _select_batch(item, indices, batch_size=batch_size) for key, item in value.items()
        }
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            return value
        tensor_indices = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, tensor_indices)
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] != batch_size:
            return value
        return value[indices]
    return value


def _batched_bool(
    info: Mapping[str, Any],
    key: str,
    *,
    batch_size: int,
    required: bool = True,
) -> np.ndarray:
    if key not in info:
        if required:
            raise KeyError(f"ManiSkill info is missing native signal {key!r}")
        return np.zeros(batch_size, dtype=bool)
    values = _to_numpy(info[key]).astype(bool, copy=False).reshape(-1)
    if values.size == 1 and batch_size != 1:
        values = np.repeat(values, batch_size)
    if values.size != batch_size:
        raise ValueError(
            f"Native ManiSkill info[{key!r}] must have {batch_size} values, got {values.shape}"
        )
    return values.copy()


def native_signals(info: Mapping[str, Any], *, batch_size: int) -> dict[str, np.ndarray]:
    """Extract native success and stage signals without reimplementing predicates."""

    signals = {"success": _batched_bool(info, "success", batch_size=batch_size)}
    signals.update(
        {
            key: _batched_bool(info, key, batch_size=batch_size, required=False)
            for key in STAGE_SIGNAL_KEYS
        }
    )
    signals["fail"] = _batched_bool(info, "fail", batch_size=batch_size, required=False)
    return signals


def build_policy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Map a batched native ``obs_mode=rgb`` observation to the 21D student input."""

    import torch

    try:
        rgb = observation["sensor_data"]["base_camera"]["rgb"]
        qpos = observation["agent"]["qpos"]
        qvel = observation["agent"]["qvel"]
        goal_pos = observation["extra"]["goal_pos"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "ManiSkill obs_mode=rgb must contain sensor_data.base_camera.rgb, "
            "agent.qpos/qvel and extra.goal_pos"
        ) from exc

    rgb_tensor = _to_torch(rgb)
    if rgb_tensor.ndim == 3:
        rgb_tensor = rgb_tensor.unsqueeze(0)
    if rgb_tensor.ndim != 4 or tuple(rgb_tensor.shape[-1:]) != (3,):
        raise ValueError(
            f"Expected batched channel-last RGB [B,H,W,3], got {tuple(rgb_tensor.shape)}"
        )
    if rgb_tensor.dtype != torch.uint8:
        raise ValueError(f"Expected ManiSkill RGB observations to be uint8, got {rgb_tensor.dtype}")
    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0

    qpos_tensor = _to_torch(qpos, dtype=torch.float32)
    qvel_tensor = _to_torch(qvel, dtype=torch.float32)
    goal_tensor = _to_torch(goal_pos, dtype=torch.float32)
    if qpos_tensor.ndim == 1:
        qpos_tensor = qpos_tensor.unsqueeze(0)
    if qvel_tensor.ndim == 1:
        qvel_tensor = qvel_tensor.unsqueeze(0)
    if goal_tensor.ndim == 1:
        goal_tensor = goal_tensor.unsqueeze(0)
    if qpos_tensor.shape != qvel_tensor.shape or qpos_tensor.shape[0] != rgb_tensor.shape[0]:
        raise ValueError(
            "qpos/qvel and RGB batch dimensions disagree: "
            f"rgb={tuple(rgb_tensor.shape)}, qpos={tuple(qpos_tensor.shape)}, "
            f"qvel={tuple(qvel_tensor.shape)}"
        )
    if qpos_tensor.shape[1:] != (9,) or qvel_tensor.shape[1:] != (9,):
        raise ValueError(
            "PickCube proprioception must be qpos/qvel=(9,), got "
            f"{tuple(qpos_tensor.shape[1:])}/{tuple(qvel_tensor.shape[1:])}"
        )
    if goal_tensor.shape != (rgb_tensor.shape[0], 3):
        raise ValueError(f"PickCube goal_pos must have shape [B,3], got {tuple(goal_tensor.shape)}")

    state = torch.cat((qpos_tensor, qvel_tensor, goal_tensor), dim=-1)
    if state.shape[1:] != (21,) or not bool(torch.isfinite(state).all()):
        raise ValueError(f"Policy state must be finite with shape [B,21], got {tuple(state.shape)}")
    return {IMAGE_KEY: rgb_tensor, STATE_KEY: state}


def build_privileged_observation(observation: Mapping[str, Any]) -> Any:
    """Build the PPO's 42D state in ManiSkill ``flatten_state_dict`` order."""

    import torch

    try:
        agent = observation["agent"]
        extra = observation["extra"]
        values = (
            agent["qpos"],
            agent["qvel"],
            extra["is_grasped"],
            extra["tcp_pose"],
            extra["goal_pos"],
            extra["obj_pose"],
            extra["tcp_to_obj_pos"],
            extra["obj_to_goal_pos"],
        )
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "ManiSkill obs_mode=state_dict is missing one of the PPO 42D fields"
        ) from exc
    qpos_tensor = _to_torch(values[0], dtype=torch.float32)
    if qpos_tensor.ndim == 1:
        qpos_tensor = qpos_tensor.unsqueeze(0)
    if qpos_tensor.ndim < 2:
        raise ValueError(f"Privileged qpos must have a batch dimension, got {qpos_tensor.shape}")
    batch_size = int(qpos_tensor.shape[0])
    tensors = []
    for value in values:
        tensor = _to_torch(value, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            # ManiSkill emits singleton/vector fields such as is_grasped as
            # [B], while qpos/qvel are emitted as [B,D].
            tensor = (
                tensor.reshape(batch_size, -1) if tensor.shape[0] == batch_size else tensor[None, :]
            )
        elif tensor.shape[0] != batch_size and tensor.shape[0] == 1:
            tensor = tensor.reshape(1, -1)
        else:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, -1)
        if tensor.shape[0] != batch_size:
            raise ValueError(
                "Privileged observation fields must share the qpos batch size: "
                f"expected {batch_size}, got {tensor.shape[0]}"
            )
        tensors.append(tensor)
    state = torch.cat(tensors, dim=-1)
    if state.shape[1:] != (42,) or not bool(torch.isfinite(state).all()):
        raise ValueError(
            f"Privileged PPO state must be finite with shape [B,42], got {state.shape}"
        )
    return state


def _policy_action_dim(config: Any) -> int:
    feature = getattr(config, "action_feature", None)
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, Mapping):
        shape = feature.get("shape")
    if shape is None or len(shape) != 1:
        raise ValueError(f"Diffusion policy action_feature must be one-dimensional, got {shape}")
    return int(shape[0])


def _sync_policy_device(config: Any) -> None:
    import torch

    device = str(getattr(config, "device", ""))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_policy(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DiffusionPolicyDriver:
    """Use LeRobot's action queue while keeping ManiSkill outside LeRobot envs."""

    def __init__(
        self,
        loaded: Mapping[str, Any],
        *,
        observation_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]] = build_policy_observation,
        state_dim: int = 21,
        observation_description: str = "RGB + 21D non-privileged state",
    ) -> None:
        from lerobot.policies.utils import populate_queues
        from lerobot.utils.constants import ACTION, OBS_IMAGES

        self.policy = loaded["policy"]
        self.config = loaded["config"]
        self.preprocessor = loaded["preprocessor"]
        self.postprocessor = loaded["postprocessor"]
        self._populate_queues = populate_queues
        self._action_key = ACTION
        self._images_key = OBS_IMAGES
        self._image_features = tuple(getattr(self.config, "image_features", ()) or ())
        self._observation_builder = observation_builder
        self._state_dim = int(state_dim)
        self.observation_description = observation_description
        self.latencies_ms: list[float] = []
        self.batch_size: int | None = None
        self.policy.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.latencies_ms.clear()
        self.batch_size = None

    def prepare_observation(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.preprocessor(self._observation_builder(observation))

    def select_action(self, processed_observation: Mapping[str, Any]) -> np.ndarray:
        import torch

        prepared = dict(processed_observation)
        prepared.pop(self._action_key, None)
        if self._image_features:
            missing = [key for key in self._image_features if key not in prepared]
            if missing:
                raise KeyError(f"Processed policy observation is missing image features: {missing}")
            # LeRobot's policy queue stores all cameras under OBS_IMAGES. The
            # custom evaluator bypasses policy.select_action to measure the
            # forward latency, so it must perform this stacking itself.
            prepared[self._images_key] = torch.stack(
                [prepared[key] for key in self._image_features], dim=-4
            )
        state = prepared.get(STATE_KEY)
        if state is None:
            raise KeyError(f"Processed policy observation is missing {STATE_KEY!r}")
        state_array = _to_numpy(state)
        if state_array.ndim != 2:
            raise ValueError(
                f"Processed policy state must be batched [B,{self._state_dim}], got {state_array.shape}"
            )
        batch_size = int(state_array.shape[0])
        if self.batch_size is None:
            self.batch_size = batch_size
        elif self.batch_size != batch_size:
            raise ValueError(
                f"Policy batch size changed within one vector batch: {self.batch_size} -> {batch_size}"
            )
        self.policy._queues = self._populate_queues(self.policy._queues, prepared)
        action_queue = self.policy._queues[self._action_key]
        if len(action_queue) == 0:
            _sync_policy_device(self.config)
            started = time.perf_counter()
            with torch.inference_mode():
                normalized_chunk = self.policy.predict_action_chunk(prepared)
            _sync_policy_device(self.config)
            self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
            if normalized_chunk.ndim != 3 or normalized_chunk.shape[0] != batch_size:
                raise ValueError(
                    "Diffusion policy must return [batch,horizon,action_dim], "
                    f"got {tuple(normalized_chunk.shape)}"
                )
            if not bool(torch.isfinite(normalized_chunk).all()):
                raise ValueError("Diffusion policy returned non-finite normalized actions")
            action_queue.extend(normalized_chunk.transpose(0, 1))
        normalized_action = action_queue.popleft()
        action = self.postprocessor(normalized_action)
        array = _to_numpy(action).astype(np.float32, copy=False)
        if array.ndim == 1 and batch_size == 1:
            array = array[None, :]
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"Postprocessed action must be finite [batch,dim], got {array.shape}")
        if array.shape[0] != batch_size:
            raise ValueError(
                f"Postprocessed action batch size must be {batch_size}, got {array.shape}"
            )
        return array


class PPOPrivilegedDriver:
    """Deterministic actor-mean inference for ManiSkill's downloaded PPO checkpoint."""

    def __init__(self, checkpoint: Path, *, device: str | None = None) -> None:
        import torch
        from torch import nn

        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint}")
        self.checkpoint = checkpoint
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        self.actor = nn.Sequential(
            nn.Linear(42, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 7),
        ).to(self.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError(f"PPO checkpoint must be a state dict, got {type(state).__name__}")
        actor_state = {
            key.removeprefix("actor_mean."): value
            for key, value in state.items()
            if str(key).startswith("actor_mean.")
        }
        if not actor_state:
            raise KeyError("PPO checkpoint has no actor_mean.* parameters")
        self.actor.load_state_dict(actor_state, strict=True)
        self.actor.eval()
        self.latencies_ms: list[float] = []

    def reset(self) -> None:
        self.latencies_ms.clear()

    def prepare_observation(self, observation: Mapping[str, Any]) -> Any:
        return build_privileged_observation(observation)

    def select_action(self, state: Any) -> np.ndarray:
        import torch

        started = time.perf_counter()
        with torch.inference_mode():
            action = self.actor(_to_torch(state, dtype=torch.float32, device=str(self.device)))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return np.clip(_to_numpy(action), -1.0, 1.0).astype(np.float32, copy=False)


def load_diffusion_policy(
    *,
    policy_path: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    n_action_steps: int,
    device: str | None,
) -> dict[str, Any]:
    """Load a local LeRobot diffusion checkpoint and its saved processors."""

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_path = policy_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Policy directory does not exist: {policy_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root does not exist: {dataset_root}")
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    if getattr(config, "type", None) != "diffusion":
        raise TypeError(
            f"M0 closed-loop probe requires policy.type='diffusion', got {config.type!r}"
        )
    action_dim = _policy_action_dim(config)
    horizon = int(getattr(config, "horizon", 0))
    if n_action_steps <= 0 or (horizon and n_action_steps > horizon):
        raise ValueError(f"n_action_steps={n_action_steps} must be in [1, horizon={horizon}]")
    checkpoint_n_action_steps = int(getattr(config, "n_action_steps", n_action_steps))
    config.n_action_steps = n_action_steps
    if device is not None:
        config.device = device
    config.pretrained_path = policy_path
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = make_policy(cfg=config, ds_meta=dataset_metadata)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(config.device)}},
    )
    return {
        "policy": policy,
        "config": config,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "action_dim": action_dim,
        "checkpoint_n_action_steps": checkpoint_n_action_steps,
    }


def _zero_finished(action: np.ndarray, finished: np.ndarray) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).copy()
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[0] != finished.size:
        raise ValueError(f"Action must have shape [batch,dim], got {result.shape}")
    result[finished] = 0.0
    if not np.isfinite(result).all():
        raise ValueError("Environment action contains non-finite values")
    return result


def _sample_random_action(env: Any, *, batch_size: int, action_dim: int) -> np.ndarray:
    action = _to_numpy(env.action_space.sample()).astype(np.float32, copy=False)
    if action.ndim == 1:
        action = action[None, :]
    if action.shape != (batch_size, action_dim):
        raise ValueError(
            "ManiSkill action_space.sample() must return "
            f"{(batch_size, action_dim)}, got {action.shape}"
        )
    return action


def _termination_reason(
    *,
    success: bool,
    truncated: bool,
    fail: bool,
    terminated: bool,
) -> str:
    if success:
        return "success"
    if truncated:
        return "truncated"
    if fail:
        return "native_fail"
    if terminated:
        return "terminated"
    return "episode_limit"


def _driver_action(driver: Any, observation: Mapping[str, Any]) -> np.ndarray:
    if hasattr(driver, "prepare_observation"):
        return _to_numpy(driver.select_action(driver.prepare_observation(observation)))
    if hasattr(driver, "preprocessor"):
        return _to_numpy(driver.select_action(driver.preprocessor(observation)))
    return _to_numpy(driver.select_action(observation))


def _action_measures(action: np.ndarray) -> dict[str, np.ndarray | None]:
    return {
        "translation_norm": np.linalg.norm(action[:, :3], axis=1),
        "rotation_norm": (None if action.shape[1] < 7 else np.linalg.norm(action[:, 3:6], axis=1)),
        "gripper_abs": np.abs(action[:, -1]),
    }


def run_batch(
    *,
    env: Any,
    seeds: Sequence[int],
    policy_driver: Any | None,
    max_steps: int,
    action_dim: int,
    episode_index_offset: int,
    task: str = TASK_ID,
    environment_latencies_ms: list[float] | None = None,
    visualization_recorder: Any | None = None,
    owned_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Run one frozen vector batch without automatic reset after success."""

    if not seeds:
        return []
    batch_size = len(seeds)
    if owned_indices is None:
        owned = np.arange(batch_size, dtype=np.int64)
    else:
        owned = np.asarray(tuple(int(index) for index in owned_indices), dtype=np.int64)
        if owned.size == 0 or np.any(owned < 0) or np.any(owned >= batch_size):
            raise ValueError(f"owned_indices must be non-empty indices within [0, {batch_size})")
        if len(np.unique(owned)) != owned.size:
            raise ValueError("owned_indices must not contain duplicates")
    owned_mask = np.zeros(batch_size, dtype=bool)
    owned_mask[owned] = True
    observation, reset_info = env.reset(seed=[int(seed) for seed in seeds])
    signals = native_signals(reset_info, batch_size=batch_size)
    ever = {key: signals[key].copy() for key in STAGE_SIGNAL_KEYS}
    success_once = signals["success"].copy()
    success_at_end = signals["success"].copy()
    first_success_step = np.full(batch_size, -1, dtype=np.int32)
    first_success_step[success_once] = 0
    finished = np.zeros(batch_size, dtype=bool)
    steps = np.zeros(batch_size, dtype=np.int32)
    reasons = ["episode_limit"] * batch_size
    action_count = np.zeros(batch_size, dtype=np.int32)
    metric_sums = {name: np.zeros(batch_size, dtype=np.float64) for name in ACTION_METRIC_NAMES}
    jitter_sums = np.zeros(batch_size, dtype=np.float64)
    gripper_flips = np.zeros(batch_size, dtype=np.int32)
    previous_action = np.zeros((batch_size, action_dim), dtype=np.float32)
    has_previous_action = np.zeros(batch_size, dtype=bool)
    if policy_driver is not None and hasattr(policy_driver, "reset"):
        policy_driver.reset()
    if visualization_recorder is not None:
        visualization_recorder.start_batch(
            observation,
            seeds=seeds,
            episode_index_offset=episode_index_offset,
            signals=signals,
        )

    for step_index in range(1, max_steps + 1):
        if policy_driver is None:
            action = _sample_random_action(env, batch_size=batch_size, action_dim=action_dim)
        else:
            owned_observation = _select_batch(observation, owned, batch_size=batch_size)
            owned_action = _driver_action(policy_driver, owned_observation)
            owned_action = _zero_finished(owned_action, finished[owned])
            action = np.zeros((batch_size, action_dim), dtype=np.float32)
            action[owned] = owned_action
        action = _zero_finished(action, finished)
        if action.shape != (batch_size, action_dim):
            raise ValueError(
                f"Policy/environment action dimension mismatch: expected {(batch_size, action_dim)}, "
                f"got {action.shape}"
            )
        active = ~finished
        measures = _action_measures(action)
        owned_active = active & owned_mask
        for name in ACTION_METRIC_NAMES:
            values = measures[name]
            if values is not None:
                metric_sums[name][owned_active] += values[owned_active]
        action_count[owned_active] += 1
        if np.any(owned_active & has_previous_action):
            comparable = owned_active & has_previous_action
            jitter_sums[comparable] += np.linalg.norm(
                action[comparable] - previous_action[comparable], axis=1
            )
            before = np.sign(previous_action[comparable, -1])
            after = np.sign(action[comparable, -1])
            gripper_flips[comparable] += (before != after) & (before != 0) & (after != 0)
        previous_action[owned_active] = action[owned_active]
        has_previous_action[owned_active] = True

        started = time.perf_counter()
        observation, _, terminated, truncated, info = env.step(action)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if environment_latencies_ms is not None:
            environment_latencies_ms.append(elapsed_ms)
        signals = native_signals(info, batch_size=batch_size)
        for key in STAGE_SIGNAL_KEYS:
            ever[key] |= signals[key]
        active_success = active & signals["success"]
        first_success_step[(first_success_step < 0) & active_success] = step_index
        success_once[active] |= signals["success"][active]
        success_at_end[active] = signals["success"][active]
        terminated_array = _to_numpy(terminated).astype(bool, copy=False).reshape(-1)
        truncated_array = _to_numpy(truncated).astype(bool, copy=False).reshape(-1)
        if terminated_array.size != batch_size or truncated_array.size != batch_size:
            raise ValueError("ManiSkill terminated/truncated signals have the wrong batch size")
        if visualization_recorder is not None:
            visualization_recorder.record_step(
                observation,
                action=action,
                signals=signals,
                terminated=terminated_array,
                truncated=truncated_array,
            )
        newly_finished = active & (terminated_array | truncated_array)
        for index in np.flatnonzero(newly_finished):
            steps[index] = step_index
            reasons[index] = _termination_reason(
                success=bool(signals["success"][index]),
                truncated=bool(truncated_array[index]),
                fail=bool(signals["fail"][index]),
                terminated=bool(terminated_array[index]),
            )
        finished |= terminated_array | truncated_array

    unresolved = ~finished
    steps[unresolved] = max_steps
    for index in np.flatnonzero(unresolved):
        reasons[index] = "episode_limit"
    rows = []
    for index in owned:
        seed = seeds[index]
        count = max(int(action_count[index]), 1)
        rows.append(
            {
                "task": task,
                "episode_index": episode_index_offset + int(index),
                "seed": int(seed),
                "success_once": bool(success_once[index]),
                "success_at_end": bool(success_at_end[index]),
                "success": bool(success_at_end[index]),
                "first_success_step": (
                    None if first_success_step[index] < 0 else int(first_success_step[index])
                ),
                "steps": int(steps[index]),
                "termination_reason": reasons[index],
                "is_grasped": bool(ever["is_grasped"][index]),
                "is_obj_placed": bool(ever["is_obj_placed"][index]),
                "is_robot_static": bool(ever["is_robot_static"][index]),
                "translation_norm_mean": float(metric_sums["translation_norm"][index] / count),
                "rotation_norm_mean": (
                    None if action_dim < 7 else float(metric_sums["rotation_norm"][index] / count)
                ),
                "gripper_abs_mean": float(metric_sums["gripper_abs"][index] / count),
                "action_jitter_mean": float(jitter_sums[index] / max(count - 1, 1)),
                "gripper_flip_count": int(gripper_flips[index]),
            }
        )
    return rows


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(tuple(float(value) for value in latencies_ms), dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    count = len(rows)
    successes = sum(bool(row[key]) for row in rows)
    return {
        "num_success": successes,
        "num_episodes": count,
        "rate": successes / count if count else None,
    }


def _first_success_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [row["first_success_step"] for row in rows if row["first_success_step"] is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty episode set")
    count = len(rows)
    return {
        "num_episodes": count,
        "success_once": _rate(rows, "success_once"),
        "success_at_end": _rate(rows, "success_at_end"),
        "first_success_step": _first_success_summary(rows),
        "stage_signal_rates": {
            key: sum(bool(row[key]) for row in rows) / count for key in STAGE_SIGNAL_KEYS
        },
        "mean_episode_steps": float(np.mean([int(row["steps"]) for row in rows])),
        "control": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in (
                "translation_norm_mean",
                "gripper_abs_mean",
                "action_jitter_mean",
                "gripper_flip_count",
            )
        },
        "rotation_norm_mean": (
            None
            if all(row.get("rotation_norm_mean") is None for row in rows)
            else float(np.mean([float(row["rotation_norm_mean"]) for row in rows]))
        ),
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({str(row["termination_reason"]) for row in rows})
        },
    }


def _make_env(
    *, task: str, control_mode: str, num_envs: int, sim_backend: str, obs_mode: str
) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    runtime_backend = {"physx_cuda": "gpu", "physx_cpu": "cpu"}.get(sim_backend, sim_backend)
    return gym.make(
        task,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        sim_backend=runtime_backend,
    )


def _portable(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _default_dataset_args(control_mode: str) -> tuple[str, Path]:
    if control_mode == "pd_ee_delta_pose":
        return (
            "local/maniskill_pickcube_rl_7d",
            Path("datasets/lerobot/maniskill/pickcube_rl_7d_full"),
        )
    if control_mode == "pd_ee_delta_pos":
        return (
            "local/maniskill_pickcube_rl_4d",
            Path("datasets/lerobot/maniskill/pickcube_rl_4d_full"),
        )
    return (
        "local/maniskill_pickcube_joint",
        Path("datasets/lerobot/maniskill/pickcube_joint_delta_pos_full"),
    )


def _write_episodes_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([dict(row) for row in rows])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_from_driver(driver: Any | None) -> dict[str, Any]:
    if driver is None:
        return {"kind": "random_baseline"}
    if isinstance(driver, PPOPrivilegedDriver):
        return {
            "kind": "privileged_ppo",
            "checkpoint": _portable(driver.checkpoint),
            "observation": "42D privileged state",
        }
    return {"kind": "visual_diffusion", "observation": "RGB + 21D non-privileged state"}


def evaluate_closed_loop(
    *,
    policy_path: Path | None = None,
    task: str,
    control_mode: str,
    split_file: Path,
    seed_set: str,
    output_dir: Path,
    system: str = "visual",
    ppo_checkpoint: Path | None = None,
    num_envs: int | None = None,
    max_episodes: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    policy_seed: int = DEFAULT_POLICY_SEED,
    n_action_steps: int = DEFAULT_N_ACTION_STEPS,
    sim_backend: str | None = None,
    data_provenance: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_root: Path | None = None,
    leakage_report: Path | None = None,
    device: str | None = None,
    visualize_episodes: int = 0,
    worker_index: int = 0,
    worker_count: int = 1,
) -> dict[str, Any]:
    """Run one system on complete frozen reset batches."""

    if task != TASK_ID:
        raise ValueError(f"M0 only supports {TASK_ID}, got {task}")
    if system not in {"random", "ppo_privileged", "visual"}:
        raise ValueError("system must be random, ppo_privileged or visual")
    if control_mode not in CONTROL_ACTION_DIMS:
        raise ValueError(f"Unsupported M0 control mode: {control_mode}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if visualize_episodes < 0:
        raise ValueError("visualize_episodes must be non-negative")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    if worker_count > 1 and visualize_episodes:
        raise ValueError("visualization is only supported by a single evaluation worker")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {output_dir}")

    split = load_seed_sets(split_file, task=task)
    provenance = data_provenance or split.get("data_provenance")
    if provenance is None:
        raise ValueError(
            "rollout data provenance is required: add data_provenance to the split file "
            "or pass --data-provenance=official|m1w"
        )
    requested_backend = sim_backend or str(split["sim_backend"])
    rollout_contract = require_rollout_backend(provenance, requested_backend)
    backend = rollout_contract["actual_rollout_backend"]
    reset_batches = reset_batches_for_set(split, seed_set, limit=max_episodes)
    if not reset_batches:
        raise ValueError(f"No reset batches selected for {seed_set}")
    for batch in reset_batches:
        if num_envs is not None and num_envs != len(batch):
            raise ValueError(
                "M0 state batches must be reset in one vector call; "
                f"requested num_envs={num_envs}, batch has {len(batch)}"
            )
        if worker_count > len(batch):
            raise ValueError(
                f"worker_count={worker_count} cannot exceed reset batch size {len(batch)}"
            )

    driver: Any | None
    loaded: dict[str, Any] | None = None
    resolved_repo_id: str | None = None
    resolved_dataset_root: Path | None = None
    if system == "visual":
        if policy_path is None:
            raise ValueError("--policy-path is required for the visual system")
        dataset_default_repo, dataset_default_root = _default_dataset_args(control_mode)
        resolved_repo_id = dataset_repo_id or dataset_default_repo
        resolved_dataset_root = (dataset_root or dataset_default_root).expanduser().resolve()
        loaded = load_diffusion_policy(
            policy_path=policy_path,
            dataset_repo_id=resolved_repo_id,
            dataset_root=resolved_dataset_root,
            n_action_steps=n_action_steps,
            device=device,
        )
        driver = DiffusionPolicyDriver(loaded)
        obs_mode = "rgb"
    elif system == "ppo_privileged":
        if ppo_checkpoint is None:
            raise ValueError("--ppo-checkpoint is required for the privileged PPO system")
        driver = PPOPrivilegedDriver(ppo_checkpoint, device=device)
        # ``state`` is flattened by ManiSkill; the PPO driver needs the named
        # agent/extra fields to construct the verified 42D actor input.
        obs_mode = "state_dict"
    else:
        driver = None
        obs_mode = "rgb"

    action_dim = CONTROL_ACTION_DIMS[control_mode]
    if loaded is not None and int(loaded["action_dim"]) != action_dim:
        raise ValueError(
            f"Policy action dimension {loaded['action_dim']} does not match {control_mode}={action_dim}"
        )
    _seed_policy(policy_seed)
    visualization_recorder = None
    if visualize_episodes:
        from .closed_loop_visualization import ClosedLoopVisualizationRecorder

        visualization_recorder = ClosedLoopVisualizationRecorder(
            max_episodes=visualize_episodes,
            fps=20.0,
        )
    rows: list[dict[str, Any]] = []
    environment_latencies: list[float] = []
    control_frequency_hz: float | None = None
    offset = 0
    for batch_index, batch in enumerate(reset_batches):
        owned_indices = tuple(range(worker_index, len(batch), worker_count))
        environment = _make_env(
            task=task,
            control_mode=control_mode,
            num_envs=len(batch),
            sim_backend=backend,
            obs_mode=obs_mode,
        )
        try:
            if control_frequency_hz is None:
                control_frequency_hz = float(environment.unwrapped.control_freq)
            if system == "random":
                action_space_seed = getattr(environment.action_space, "seed", None)
                if action_space_seed is not None:
                    action_space_seed(policy_seed + batch_index)
            try:
                rows.extend(
                    run_batch(
                        env=environment,
                        seeds=batch,
                        policy_driver=driver,
                        max_steps=max_steps,
                        action_dim=action_dim,
                        episode_index_offset=offset,
                        task=task,
                        environment_latencies_ms=environment_latencies,
                        visualization_recorder=visualization_recorder,
                        owned_indices=owned_indices,
                    )
                )
            finally:
                if visualization_recorder is not None:
                    visualization_recorder.end_batch()
        finally:
            environment.close()
        offset += len(batch)

    for row in rows:
        row["evaluation_kind"] = system
    metrics = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "label": "Closed-loop",
        "system": system,
        "task": task,
        "control_mode": control_mode,
        **rollout_contract,
        "seed_set": seed_set,
        "reset_batches": [list(batch) for batch in reset_batches],
        "num_episodes": len(rows),
        "num_envs": [len(batch) for batch in reset_batches],
        "worker": {
            "index": worker_index,
            "count": worker_count,
            "state_batch_strategy": (
                "single_worker_full_batch"
                if worker_count == 1
                else "replicated_full_reset_batch_policy_shard"
            ),
        },
        "max_episode_steps": max_steps,
        "policy_seed": policy_seed,
        "control_frequency_hz": control_frequency_hz,
        "evaluation": summarize_rows(rows),
        "latency": {
            "policy_forward": latency_summary(
                [] if driver is None else getattr(driver, "latencies_ms", [])
            ),
            "environment_step": latency_summary(environment_latencies),
        },
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    incomplete = output_dir / ".incomplete"
    incomplete.write_text("ManiSkill closed-loop evaluation is incomplete.\n", encoding="utf-8")
    visualization_summary = None
    if visualization_recorder is not None:
        visualization_manifest = visualization_recorder.finalize(
            output_dir / "visualizations",
            control_mode=control_mode,
        )
        visualization_summary = {
            "enabled": True,
            "num_episodes": len(visualization_manifest["episodes"]),
            "output_dir": _portable(output_dir / "visualizations"),
            "manifest": _portable(output_dir / "visualizations" / "visualization_manifest.json"),
            "application_id": visualization_manifest["viewer"]["application_id"],
        }
    else:
        visualization_summary = {"enabled": False}
    metrics["visualization"] = visualization_summary
    manifest = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "label": "Closed-loop",
        "system": system,
        "task": task,
        "control_mode": control_mode,
        **rollout_contract,
        "policy_path": None if policy_path is None else _portable(policy_path),
        "ppo_checkpoint": None if ppo_checkpoint is None else _portable(ppo_checkpoint),
        "dataset": (
            None
            if resolved_repo_id is None
            else {"repo_id": resolved_repo_id, "root": _portable(resolved_dataset_root)}
        ),
        "split_file": _portable(split_file),
        "seed_set": seed_set,
        "reset_batches": [list(batch) for batch in reset_batches],
        "sim_backend": backend,
        "state_match_tolerance": split["state_match_tolerance"],
        "state_isolation": {
            "method": "initial_state_fingerprint",
            "report": None if leakage_report is None else _portable(leakage_report),
            "warning": "Seed integers are labels for reset batches, not a leakage proof.",
        },
        "num_envs": [len(batch) for batch in reset_batches],
        "worker": {
            "index": worker_index,
            "count": worker_count,
            "state_batch_strategy": (
                "single_worker_full_batch"
                if worker_count == 1
                else "replicated_full_reset_batch_policy_shard"
            ),
        },
        "max_episode_steps": max_steps,
        "policy_seed": policy_seed,
        "observation": {
            "keys": [IMAGE_KEY, STATE_KEY] if system == "visual" else list(PRIVILEGED_STATE_FIELDS),
            "image_shape": [128, 128, 3] if system == "visual" else None,
            "state_shape": [21]
            if system == "visual"
            else [42]
            if system == "ppo_privileged"
            else None,
            "state_fields": list(STATE_FIELDS)
            if system == "visual"
            else list(PRIVILEGED_STATE_FIELDS)
            if system == "ppo_privileged"
            else [],
            "excluded_state_fields": list(EXCLUDED_STATE_FIELDS) if system == "visual" else [],
        },
        "action": {
            "control_mode": control_mode,
            "shape": [action_dim],
            "n_obs_steps": (
                int(getattr(loaded["config"], "n_obs_steps", 1)) if loaded is not None else None
            ),
            "horizon": (
                int(getattr(loaded["config"], "horizon", 0)) if loaded is not None else None
            ),
            "n_action_steps": n_action_steps if loaded is not None else None,
        },
        "role_note": TEACHER_STUDENT_NOTE,
        "success_source": 'native info["success"] only',
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "mani_skill": _version("mani_skill"),
            "lerobot": _version("lerobot"),
        },
        "driver": _runtime_from_driver(driver),
        "visualization": visualization_summary,
    }
    write_json_atomic(output_dir / "metrics.json", metrics)
    _write_episodes_parquet(output_dir / "episodes.parquet", rows)
    write_json_atomic(output_dir / "run_manifest.json", manifest)
    incomplete.unlink()
    return metrics


def compare_metrics_files(paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """Create an offline comparison table from completed system metrics."""

    systems = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        system = str(payload.get("system", path.parent.name))
        evaluation = payload.get("evaluation", {})
        systems[system] = {
            "metrics": _portable(path),
            "success_once_rate": evaluation.get("success_once", {}).get("rate"),
            "success_at_end_rate": evaluation.get("success_at_end", {}).get("rate"),
            "first_success_step": evaluation.get("first_success_step"),
            "stage_signal_rates": evaluation.get("stage_signal_rates"),
        }
    result = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "systems": systems,
        "role_note": TEACHER_STUDENT_NOTE,
        "success_source": 'native info["success"] only',
    }
    write_json_atomic(output, result)
    return result


def _visible_gpu_ids(num_gpus: int) -> list[str]:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None and mask.strip():
        identifiers = [item.strip() for item in mask.split(",")]
        if any(not item for item in identifiers):
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains an empty GPU identifier")
    else:
        import torch

        identifiers = [str(index) for index in range(torch.cuda.device_count())]
    if len(identifiers) < num_gpus:
        raise RuntimeError(
            f"requested {num_gpus} GPU(s), but only {len(identifiers)} CUDA device(s) are visible"
        )
    return identifiers[:num_gpus]


def _multi_gpu_worker_argv(
    arguments: argparse.Namespace,
    *,
    worker_index: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robot_practice.experiments.maniskill.eval_closed_loop",
        f"--system={arguments.system}",
        f"--task={arguments.task}",
        f"--control-mode={arguments.control_mode}",
        f"--split-file={arguments.split_file}",
        f"--seed-set={arguments.seed_set}",
        f"--max-steps={arguments.max_steps}",
        f"--policy-seed={arguments.policy_seed}",
        f"--n-action-steps={arguments.n_action_steps}",
        f"--worker-index={worker_index}",
        f"--worker-count={arguments.num_gpus}",
        f"--output-dir={output_dir}",
    ]
    if arguments.policy_path is not None:
        command.append(f"--policy-path={arguments.policy_path}")
    if arguments.ppo_checkpoint is not None:
        command.append(f"--ppo-checkpoint={arguments.ppo_checkpoint}")
    if arguments.num_envs is not None:
        command.append(f"--num-envs={arguments.num_envs}")
    if arguments.max_episodes is not None:
        command.append(f"--max-episodes={arguments.max_episodes}")
    if arguments.sim_backend is not None:
        command.append(f"--sim-backend={arguments.sim_backend}")
    if arguments.data_provenance is not None:
        command.append(f"--data-provenance={arguments.data_provenance}")
    if arguments.dataset_repo_id is not None:
        command.append(f"--dataset-repo-id={arguments.dataset_repo_id}")
    if arguments.dataset_root is not None:
        command.append(f"--dataset-root={arguments.dataset_root}")
    if arguments.leakage_report is not None:
        command.append(f"--leakage-report={arguments.leakage_report}")
    if arguments.device is not None:
        command.append(f"--device={arguments.device}")
    return command


def evaluate_closed_loop_multi_gpu(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run the normal evaluator in managed GPU workers and merge the shards."""

    if arguments.visualize_episodes:
        raise ValueError("--visualize-episodes cannot be combined with --num-gpus > 1")
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {output_dir}")
    gpu_ids = _visible_gpu_ids(arguments.num_gpus)
    workers_root = output_dir / "workers"
    workers_root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output_dir / "launcher.json",
        {
            "system": arguments.system,
            "policy_path": None
            if arguments.policy_path is None
            else _portable(arguments.policy_path),
            "num_gpus": arguments.num_gpus,
            "num_envs": arguments.num_envs,
            "max_episodes": arguments.max_episodes,
            "max_steps": arguments.max_steps,
            "seed_set": arguments.seed_set,
            "data_provenance": arguments.data_provenance,
        },
    )

    processes: list[subprocess.Popen[str]] = []
    log_files: list[TextIO] = []
    try:
        for worker_index, gpu_id in enumerate(gpu_ids):
            worker_output = workers_root / f"gpu_{worker_index}"
            log_file = (workers_root / f"gpu_{worker_index}.log").open("w", encoding="utf-8")
            log_files.append(log_file)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            processes.append(
                subprocess.Popen(
                    _multi_gpu_worker_argv(
                        arguments,
                        worker_index=worker_index,
                        output_dir=worker_output,
                    ),
                    cwd=practice_root(),
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        print(f"[M0 closed-loop] workers 0/{arguments.num_gpus}", flush=True)
        for worker_index, process in enumerate(processes, start=1):
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"worker {worker_index - 1} failed with exit code {return_code}; "
                    f"inspect {workers_root / f'gpu_{worker_index - 1}.log'}"
                )
            print(
                f"\r[M0 closed-loop] workers {worker_index}/{arguments.num_gpus}",
                end="",
                flush=True,
            )
        print()

        from .merge_closed_loop_workers import merge_workers

        split = load_seed_sets(arguments.split_file, task=arguments.task)
        expected_episodes = sum(
            len(batch)
            for batch in reset_batches_for_set(
                split, arguments.seed_set, limit=arguments.max_episodes
            )
        )
        return merge_workers(
            workers_root=workers_root,
            output_dir=output_dir,
            expected_episodes=expected_episodes,
            expected_workers=arguments.num_gpus,
        )
    except Exception:
        running = [process for process in processes if process.poll() is None]
        for process in running:
            process.terminate()
        for process in running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        for log_file in log_files:
            log_file.close()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--ppo-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--system", choices=("random", "ppo_privileged", "visual"), default="visual"
    )
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--control-mode", choices=tuple(CONTROL_ACTION_DIMS), required=False)
    parser.add_argument(
        "--split-file", type=Path, default=Path("configs/experiments/maniskill/split.json")
    )
    parser.add_argument(
        "--seed-set", choices=("train", "development", "test"), default="development"
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--policy-seed", type=int, default=DEFAULT_POLICY_SEED)
    parser.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument(
        "--data-provenance",
        default=None,
        help="Dataset provenance contract: official requires CPU; m1w requires GPU.",
    )
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--leakage-report", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--visualize-episodes",
        type=int,
        default=0,
        help="Capture the first N rollout episodes as MP4 and RRD artifacts; 0 disables capture.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--compare-metrics", type=Path, nargs="+", default=None)
    parser.add_argument("--comparison-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.compare_metrics is not None:
        if arguments.num_gpus != 1:
            raise SystemExit("--num-gpus is only valid for closed-loop evaluation")
        if arguments.comparison_output is None:
            raise SystemExit("--comparison-output is required with --compare-metrics")
        result = compare_metrics_files(arguments.compare_metrics, arguments.comparison_output)
    else:
        if arguments.control_mode is None or arguments.output_dir is None:
            raise SystemExit("--control-mode and --output-dir are required for evaluation")
        if arguments.num_gpus <= 0:
            raise SystemExit("--num-gpus must be positive")
        if arguments.num_gpus > 1 and arguments.worker_count != 1:
            raise SystemExit("--num-gpus cannot be combined with hidden worker arguments")
        if arguments.num_gpus > 1:
            result = evaluate_closed_loop_multi_gpu(arguments)
        else:
            result = evaluate_closed_loop(
                policy_path=arguments.policy_path,
                task=arguments.task,
                control_mode=arguments.control_mode,
                split_file=arguments.split_file,
                seed_set=arguments.seed_set,
                output_dir=arguments.output_dir,
                system=arguments.system,
                ppo_checkpoint=arguments.ppo_checkpoint,
                num_envs=arguments.num_envs,
                max_episodes=arguments.max_episodes,
                max_steps=arguments.max_steps,
                policy_seed=arguments.policy_seed,
                n_action_steps=arguments.n_action_steps,
                sim_backend=arguments.sim_backend,
                data_provenance=arguments.data_provenance,
                dataset_repo_id=arguments.dataset_repo_id,
                dataset_root=arguments.dataset_root,
                leakage_report=arguments.leakage_report,
                device=arguments.device,
                visualize_episodes=arguments.visualize_episodes,
                worker_index=arguments.worker_index,
                worker_count=arguments.worker_count,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
