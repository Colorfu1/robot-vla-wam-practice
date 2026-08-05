"""Deterministic single-environment PushT rollout with explicit Diffusion RNG."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.io_utils import write_video
from torch import Tensor

from robot_practice.policies.mode_conditioning import (
    PROTOTYPE_ID,
    validate_and_latch_prototype,
)
from robot_practice.policies.route_field_conditioning import ROUTE_FIELD

from .noise import (
    full_sampling_reverse_noise_seed,
    initial_noise_seed,
    make_initial_noise,
    reverse_noise_seed,
)
from .policy_loading import ControlledSample, controlled_sample


class Processor(Protocol):
    def __call__(self, value: Any) -> Any: ...


class EnvironmentFactory(Protocol):
    def __call__(
        self,
        *,
        episode_length: int,
        render: bool,
    ) -> gym.Env: ...


INITIAL_NOISE_SENSITIVITY = "initial-noise-sensitivity"
FULL_DDPM_SAMPLING = "full-ddpm-sampling"
DIFFUSION_SAMPLING_PROTOCOLS = frozenset(
    {
        INITIAL_NOISE_SENSITIVITY,
        FULL_DDPM_SAMPLING,
    }
)
DIFFUSION_POLICY_TYPES = frozenset(
    {"diffusion", "mode_diffusion", "route_field_diffusion"}
)


def is_diffusion_sampling_policy(policy: torch.nn.Module) -> bool:
    return str(policy.config.type) in DIFFUSION_POLICY_TYPES


@dataclass(frozen=True)
class ReplanRecord:
    replan_index: int
    environment_seed: int
    policy_seed: int
    initial_noise_seed: int | None
    initial_noise_sha256: str | None
    reverse_noise_seed: int | None
    reverse_generator_state_sha256: str | None
    scheduler: str | None
    num_inference_steps: int | None
    latency_ms: float
    normalized_chunk: np.ndarray
    sampling_protocol: str | None = None


@dataclass(frozen=True)
class RolloutResult:
    model: str
    environment_seed: int
    policy_seed: int
    success: bool
    sum_reward: float
    max_reward: float
    final_reward: float
    episode_length: int
    agent_path: np.ndarray
    block_path: np.ndarray
    action_targets: np.ndarray
    rewards: np.ndarray
    frames: tuple[np.ndarray, ...]
    replans: tuple[ReplanRecord, ...]
    condition_id: str = "reference"
    scheduler: str | None = None
    num_inference_steps: int | None = None
    sampling_protocol: str | None = None
    termination_reason: str = "unknown"


def make_pusht_env(
    *,
    episode_length: int = 300,
    render: bool = True,
) -> gym.Env:
    import gym_pusht  # noqa: F401

    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=episode_length,
        disable_env_checker=False,
    )


def _stack_images_for_policy(
    policy: torch.nn.Module,
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    image_features = getattr(policy.config, "image_features", {})
    if not image_features:
        return dict(batch)
    prepared = dict(batch)
    prepared[OBS_IMAGES] = torch.stack(
        [prepared[key] for key in image_features],
        dim=-4,
    )
    return prepared


class ExplicitChunkExecutor:
    def __init__(
        self,
        policy: torch.nn.Module,
        *,
        environment_seed: int,
        policy_seed: int,
        sampling_protocol: str = INITIAL_NOISE_SENSITIVITY,
    ) -> None:
        if sampling_protocol not in DIFFUSION_SAMPLING_PROTOCOLS:
            raise ValueError(
                f"sampling_protocol must be one of {sorted(DIFFUSION_SAMPLING_PROTOCOLS)}"
            )
        self.policy = policy
        self.environment_seed = environment_seed
        self.policy_seed = policy_seed
        self.sampling_protocol = sampling_protocol
        self.replan_index = 0
        self.replans: list[ReplanRecord] = []
        self._active_prototype_id: Tensor | None = None
        self.policy.reset()

    def _frozen_queue_batch(self, current_batch: dict[str, Tensor]) -> dict[str, Tensor]:
        frozen = {
            key: torch.stack(list(self.policy._queues[key]), dim=1)
            for key in current_batch
            if key in self.policy._queues
        }
        if PROTOTYPE_ID in current_batch:
            frozen[PROTOTYPE_ID] = current_batch[PROTOTYPE_ID]
        if ROUTE_FIELD in current_batch:
            frozen[ROUTE_FIELD] = current_batch[ROUTE_FIELD]
        return frozen

    def _query_chunk(
        self,
        current_batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Any]]:
        if not is_diffusion_sampling_policy(self.policy):
            chunk = self.policy.predict_action_chunk(current_batch)
            return chunk, {
                "initial_noise_seed": None,
                "initial_noise_sha256": None,
                "reverse_noise_seed": None,
                "reverse_generator_state_sha256": None,
                "scheduler": None,
                "num_inference_steps": None,
            }
        if not isinstance(self.policy, DiffusionPolicy):
            raise TypeError("policy.type=diffusion must load a DiffusionPolicy")

        first_seed = initial_noise_seed(
            environment_seed=self.environment_seed,
            policy_seed=self.policy_seed,
            replan_index=self.replan_index,
        )
        if self.sampling_protocol == INITIAL_NOISE_SENSITIVITY:
            second_seed = reverse_noise_seed(
                environment_seed=self.environment_seed,
                replan_index=self.replan_index,
            )
        else:
            second_seed = full_sampling_reverse_noise_seed(
                environment_seed=self.environment_seed,
                policy_seed=self.policy_seed,
                replan_index=self.replan_index,
            )
        explicit = make_initial_noise(
            seed=first_seed,
            batch_size=1,
            device=get_device_from_parameters(self.policy),
            dtype=get_dtype_from_parameters(self.policy),
            horizon=self.policy.config.horizon,
            action_dim=self.policy.config.action_feature.shape[0],
        )
        sampled: ControlledSample = controlled_sample(
            self.policy,
            self._frozen_queue_batch(current_batch),
            initial_noise=explicit.tensor,
            reverse_seed=second_seed,
        )
        return sampled.actions, {
            "initial_noise_seed": first_seed,
            "initial_noise_sha256": explicit.tensor_sha256,
            "reverse_noise_seed": second_seed,
            "reverse_generator_state_sha256": sampled.reverse_generator_state_sha256,
            "scheduler": sampled.scheduler,
            "num_inference_steps": sampled.num_inference_steps,
        }

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = dict(batch)
        prepared.pop(ACTION, None)
        prepared = _stack_images_for_policy(self.policy, prepared)
        if PROTOTYPE_ID in prepared:
            self._active_prototype_id = validate_and_latch_prototype(
                self._active_prototype_id,
                prepared,
                num_prototypes=int(self.policy.config.num_prototypes),
            )
        self.policy._queues = populate_queues(self.policy._queues, prepared)
        action_queue: deque[Tensor] = self.policy._queues[ACTION]
        if len(action_queue) == 0:
            device = get_device_from_parameters(self.policy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            chunk, metadata = self._query_chunk(prepared)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if chunk.ndim != 3 or chunk.shape[0] != 1:
                raise ValueError(f"Policy chunk must have shape [1,steps,dim], got {chunk.shape}")
            latency_ms = (time.perf_counter() - started) * 1000.0
            action_queue.extend(chunk.transpose(0, 1))
            self.replans.append(
                ReplanRecord(
                    replan_index=self.replan_index,
                    environment_seed=self.environment_seed,
                    policy_seed=self.policy_seed,
                    latency_ms=latency_ms,
                    normalized_chunk=chunk[0].detach().cpu().numpy(),
                    sampling_protocol=(
                        self.sampling_protocol
                        if is_diffusion_sampling_policy(self.policy)
                        else None
                    ),
                    **metadata,
                )
            )
            self.replan_index += 1
        return action_queue.popleft()


def _postprocess_action(postprocessor: Processor, action: Tensor) -> np.ndarray:
    result = postprocessor(action)
    if not isinstance(result, Tensor):
        raise TypeError(f"Policy postprocessor returned {type(result).__name__}, expected Tensor")
    array = result.detach().cpu().numpy()
    if array.shape != (1, 2) or not np.isfinite(array).all():
        raise ValueError(
            f"Postprocessed PushT action must have finite shape [1,2], got {array.shape}"
        )
    return array[0]


def run_rollout(
    *,
    policy: torch.nn.Module,
    preprocessor: Processor,
    postprocessor: Processor,
    environment_seed: int,
    policy_seed: int,
    record_frames: bool = True,
    episode_length: int = 300,
    condition_id: str = "reference",
    sampling_protocol: str = INITIAL_NOISE_SENSITIVITY,
    environment_factory: EnvironmentFactory | None = None,
) -> RolloutResult:
    factory = make_pusht_env if environment_factory is None else environment_factory
    environment = factory(episode_length=episode_length, render=record_frames)
    executor = ExplicitChunkExecutor(
        policy,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        sampling_protocol=sampling_protocol,
    )
    observation, info = environment.reset(seed=environment_seed)
    agent_path = [np.asarray(info["pos_agent"], dtype=np.float64)]
    block_path = [np.asarray(info["block_pose"], dtype=np.float64)]
    frames: list[np.ndarray] = [environment.render()] if record_frames else []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    success = False
    termination_reason = "episode_limit"
    try:
        for _ in range(episode_length):
            policy_observation = preprocess_observation(observation)
            processed = preprocessor(policy_observation)
            normalized_action = executor.select_action(processed)
            action = _postprocess_action(postprocessor, normalized_action)
            observation, reward, terminated, truncated, info = environment.step(action)
            actions.append(action.copy())
            rewards.append(float(reward))
            agent_path.append(np.asarray(info["pos_agent"], dtype=np.float64))
            block_path.append(np.asarray(info["block_pose"], dtype=np.float64))
            if record_frames:
                frames.append(environment.render())
            success = success or bool(info.get("is_success", False))
            if terminated or truncated:
                termination_reason = (
                    "success" if success else ("terminated" if terminated else "timeout")
                )
                break
    finally:
        environment.close()

    reward_array = np.asarray(rewards, dtype=np.float64)
    scheduler = executor.replans[0].scheduler if executor.replans else None
    num_inference_steps = executor.replans[0].num_inference_steps if executor.replans else None
    return RolloutResult(
        model=policy.config.type,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        success=success,
        sum_reward=float(reward_array.sum()) if reward_array.size else 0.0,
        max_reward=float(reward_array.max()) if reward_array.size else 0.0,
        final_reward=float(reward_array[-1]) if reward_array.size else 0.0,
        episode_length=len(actions),
        agent_path=np.stack(agent_path),
        block_path=np.stack(block_path),
        action_targets=np.stack(actions) if actions else np.empty((0, 2)),
        rewards=reward_array,
        frames=tuple(frames),
        replans=tuple(executor.replans),
        condition_id=condition_id,
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        sampling_protocol=(sampling_protocol if is_diffusion_sampling_policy(policy) else None),
        termination_reason=termination_reason,
    )


def save_rollout_video(
    result: RolloutResult,
    destination: str | Path,
    *,
    fps: int = 10,
) -> Path:
    if not result.frames:
        raise ValueError("Rollout does not contain rendered frames")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(path), np.stack(result.frames), fps)
    return path


def check_environment_replay(
    *,
    environment_seed: int,
    actions: np.ndarray,
    atol: float = 1e-8,
) -> dict[str, Any]:
    action_values = np.asarray(actions, dtype=np.float32)
    if action_values.ndim != 2 or action_values.shape[1] != 2:
        raise ValueError("actions must have shape [steps,2]")

    def replay() -> tuple[dict[str, np.ndarray], list[tuple[Any, ...]]]:
        environment = make_pusht_env(episode_length=max(300, len(action_values)))
        observation, info = environment.reset(seed=environment_seed)
        initial = {
            "pixels": observation["pixels"].copy(),
            "agent_pos": observation["agent_pos"].copy(),
            "block_pose": np.asarray(info["block_pose"]).copy(),
        }
        transitions: list[tuple[Any, ...]] = []
        try:
            for action in action_values:
                observation, reward, terminated, truncated, info = environment.step(action)
                transitions.append(
                    (
                        observation["pixels"].copy(),
                        observation["agent_pos"].copy(),
                        np.asarray(info["block_pose"]).copy(),
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                        bool(info["is_success"]),
                    )
                )
                if terminated or truncated:
                    break
        finally:
            environment.close()
        return initial, transitions

    first_initial, first_transitions = replay()
    second_initial, second_transitions = replay()
    if not np.array_equal(first_initial["pixels"], second_initial["pixels"]):
        raise RuntimeError("Fixed-seed PushT reset did not reproduce the initial pixels")
    for key in ("agent_pos", "block_pose"):
        if not np.allclose(first_initial[key], second_initial[key], rtol=0, atol=atol):
            raise RuntimeError(f"Fixed-seed PushT reset did not reproduce {key}")
    if len(first_transitions) != len(second_transitions):
        raise RuntimeError("Fixed action replay produced different episode lengths")
    for step, (first, second) in enumerate(zip(first_transitions, second_transitions, strict=True)):
        if not np.array_equal(first[0], second[0]):
            raise RuntimeError(f"Replay pixels differ at step {step}")
        for index, name in ((1, "agent_pos"), (2, "block_pose")):
            if not np.allclose(first[index], second[index], rtol=0, atol=atol):
                raise RuntimeError(f"Replay {name} differs at step {step}")
        if first[3:] != second[3:]:
            raise RuntimeError(f"Replay reward/done/success differs at step {step}")
    return {
        "environment_seed": environment_seed,
        "num_actions": len(action_values),
        "num_executed_steps": len(first_transitions),
        "float_atol": atol,
        "pixels_exact": True,
        "reproduced": True,
    }
