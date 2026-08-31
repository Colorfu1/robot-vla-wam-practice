"""Residual SAC on ManiSkill2 PegInsertionSide with a frozen Light-WAM base.

The Light-WAM model and the residual learner are intentionally separate.  The
only online coupling is:

``native ManiSkill observation -> LightWAMBasePolicy -> base action chunk``
``-> ResidualSAC actor -> clipped composed action chunk -> ManiSkill``

The base policy is never part of the SAC optimizer.  A transition is stored at
the replan boundary and contains the accumulated reward for the executed
low-level actions.  This keeps the residual learner independent of the
official Light-WAM implementation and makes the single-GPU route easy to
audit before adding vectorized environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch import nn

from .lightwam_maniskill import (
    PEG_ACTION_DIM,
    PEG_DEFAULT_ACTION_HORIZON,
    PEG_DEFAULT_REPLAN_STEPS,
    PEG_ENV_ID,
    PEG_PROPRIO_DIM,
    LightWAMBasePolicy,
    PegLightWAMContract,
    PegLightWAMObservation,
    build_peg_lightwam_observation,
)


class BasePolicy(Protocol):
    contract: PegLightWAMContract
    action_std: np.ndarray

    def predict(
        self,
        observation: Mapping[str, Any] | PegLightWAMObservation,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        ...

    def normalize_proprio(self, observation: PegLightWAMObservation) -> np.ndarray:
        ...

    @property
    def parameter_hash(self) -> str:
        ...


class _DummyBox:
    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class DummyPegInsertionEnv:
    """Data-free control loop with the same nested observation keys as ManiSkill2."""

    def __init__(self, *, episode_length: int = 32) -> None:
        self.episode_length = int(episode_length)
        self.action_space = _DummyBox(
            np.full(PEG_ACTION_DIM, -1.0, dtype=np.float32),
            np.full(PEG_ACTION_DIM, 1.0, dtype=np.float32),
        )
        self._qpos = np.zeros(9, dtype=np.float32)
        self._step = 0

    def _observation(self) -> dict[str, Any]:
        return {
            "agent": {
                "qpos": self._qpos.copy(),
                "qvel": np.zeros(9, dtype=np.float32),
                "base_pose": np.zeros(7, dtype=np.float32),
            },
            "extra": {"tcp_pose": np.zeros(7, dtype=np.float32)},
            "sensor_data": {
                "base_camera": {"rgb": np.zeros((128, 128, 3), dtype=np.uint8)},
                "hand_camera": {"rgb": np.full((128, 128, 3), 255, dtype=np.uint8)},
            },
        }

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed
        self._qpos.fill(0.0)
        self._step = 0
        return self._observation(), {"success": False}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (PEG_ACTION_DIM,):
            raise ValueError(
                f"dummy Peg action must have shape ({PEG_ACTION_DIM},), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("dummy Peg action contains non-finite values")
        self._qpos[:PEG_ACTION_DIM] += 0.25 * np.clip(values, -1.0, 1.0)
        self._step += 1
        distance = float(np.linalg.norm(self._qpos[:PEG_ACTION_DIM]))
        success = distance < 1e-6
        truncated = self._step >= self.episode_length
        return self._observation(), -distance, success, truncated, {"success": success}

    def close(self) -> None:
        return None


class DummyPegBasePolicy:
    """Frozen zero-action base used only for interface smoke tests."""

    def __init__(self, contract: PegLightWAMContract) -> None:
        self.contract = contract
        self.action_std = np.ones(contract.action_dim, dtype=np.float32)
        self._hash = hashlib.sha256(b"dummy-peg-lightwam-base-v1").hexdigest()

    @property
    def parameter_hash(self) -> str:
        return self._hash

    def normalize_proprio(self, observation: PegLightWAMObservation) -> np.ndarray:
        return np.asarray(observation.proprio, dtype=np.float32)

    def predict(
        self,
        observation: Mapping[str, Any] | PegLightWAMObservation,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        del observation, seed
        return np.zeros(
            (self.contract.action_horizon, self.contract.action_dim), dtype=np.float32
        )


def _layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    """Match the small SAC head initializer used by Policy Decorator."""

    if not isinstance(layer, nn.Linear):
        raise TypeError(f"official SAC layer_init expects nn.Linear, got {type(layer)}")
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualActor(nn.Module):
    """Policy Decorator-style SAC actor for a normalized residual chunk in [-1, 1]."""

    log_std_min = -20.0
    log_std_max = 2.0

    def __init__(self, observation_dim: int, residual_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # The official Policy Decorator Actor uses the same 3x256 MLP and
        # orthogonal 0.01 output heads; only the input/output dimensions are
        # changed for the Light-WAM Peg contract.
        self.mean = _layer_init(nn.Linear(hidden_dim, residual_dim), std=0.01)
        self.log_std = _layer_init(nn.Linear(hidden_dim, residual_dim), std=0.01)

    def _distribution_parameters(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(observation)
        log_std = self.log_std(hidden).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return self.mean(hidden), log_std

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution_parameters(observation)
        if deterministic:
            value = torch.tanh(mean)
            return value, torch.zeros((observation.shape[0], 1), device=observation.device)
        normal = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = normal.rsample()
        value = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - value.square() + 1e-6)
        return value, log_prob.sum(dim=-1, keepdim=True)


class ResidualCritic(nn.Module):
    def __init__(self, observation_dim: int, residual_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + residual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )

    def forward(self, observation: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((observation, residual), dim=-1))


@dataclass(frozen=True)
class ResidualContract:
    env_id: str = PEG_ENV_ID
    action_dim: int = PEG_ACTION_DIM
    proprio_dim: int = PEG_PROPRIO_DIM
    action_horizon: int = PEG_DEFAULT_ACTION_HORIZON
    replan_steps: int = PEG_DEFAULT_REPLAN_STEPS
    actor_input: str = "proprio_base_action"
    residual_channels: str = "all_executed_action_channels"

    @property
    def residual_dim(self) -> int:
        return self.replan_steps * self.action_dim

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "residual_horizon": self.replan_steps,
            "residual_dim": self.residual_dim,
        }


ResidualTransition = tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, float]


class ChunkReplayBuffer:
    """Fixed-capacity replay storage with O(1) overwrite at the capacity limit."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"replay capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._storage: list[ResidualTransition | None] = [None] * self.capacity
        self._next_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: ResidualTransition) -> None:
        self._storage[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[ResidualTransition]:
        count = min(int(batch_size), self._size)
        if count <= 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = random.sample(range(self._size), count)
        transitions = [self._storage[index] for index in indices]
        if any(transition is None for transition in transitions):
            raise RuntimeError("replay buffer has an uninitialized slot")
        return [transition for transition in transitions if transition is not None]


def compose_peg_residual_action(
    base_action: np.ndarray,
    residual_unit: np.ndarray,
    *,
    action_std: np.ndarray,
    res_scale: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Compose a normalized residual with a physical Light-WAM action chunk."""

    base = np.asarray(base_action, dtype=np.float32)
    residual = np.asarray(residual_unit, dtype=np.float32)
    std = np.asarray(action_std, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    if base.ndim != 2 or base.shape[-1] != PEG_ACTION_DIM:
        raise ValueError(f"base action must be [H,{PEG_ACTION_DIM}], got {base.shape}")
    if residual.ndim != 2 or residual.shape[-1] != PEG_ACTION_DIM:
        raise ValueError(
            f"residual must have shape [H,{PEG_ACTION_DIM}], got {residual.shape}"
        )
    if residual.shape[0] > base.shape[0]:
        raise ValueError(
            f"residual horizon {residual.shape[0]} exceeds base horizon {base.shape[0]}"
        )
    for name, value in (("action_std", std), ("action_low", low), ("action_high", high)):
        if value.shape != (PEG_ACTION_DIM,):
            raise ValueError(f"{name} must have shape ({PEG_ACTION_DIM},), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if np.any(std <= 0.0):
        raise ValueError("action_std must be strictly positive")
    if res_scale < 0.0:
        raise ValueError(f"res_scale must be non-negative, got {res_scale}")
    if np.any(low > high):
        raise ValueError("action_low must not exceed action_high")
    if not np.isfinite(base).all() or not np.isfinite(residual).all():
        raise ValueError("base and residual actions must be finite")
    result = base.copy()
    result[: residual.shape[0]] += float(res_scale) * std[None, :] * residual
    return np.clip(result, low[None, :], high[None, :]).astype(np.float32, copy=False)


class ResidualSAC:
    def __init__(
        self,
        *,
        observation_dim: int,
        residual_dim: int,
        hidden_dim: int,
        device: torch.device,
        actor_lr: float,
        critic_lr: float,
        alpha: float,
        gamma: float,
        tau: float,
    ) -> None:
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.actor = ResidualActor(observation_dim, residual_dim, hidden_dim).to(device)
        self.critic1 = ResidualCritic(observation_dim, residual_dim, hidden_dim).to(device)
        self.critic2 = ResidualCritic(observation_dim, residual_dim, hidden_dim).to(device)
        self.target1 = ResidualCritic(observation_dim, residual_dim, hidden_dim).to(device)
        self.target2 = ResidualCritic(observation_dim, residual_dim, hidden_dim).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(critic_lr),
        )

    def action(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, float]:
        values = _tensor_from_array(observation, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(values, deterministic=deterministic)
        action_np = np.asarray(action[0].reshape(-1).cpu().tolist(), dtype=np.float32)
        return action_np, float(log_prob[0].item())

    def update(
        self,
        replay: ChunkReplayBuffer,
        batch_size: int,
    ) -> dict[str, float]:
        count = min(int(batch_size), len(replay))
        if count <= 0:
            raise ValueError("cannot update SAC with an empty replay buffer")
        batch = replay.sample(count)
        observations = _tensor_from_array(np.stack([item[0] for item in batch]), device=self.device)
        residuals = _tensor_from_array(np.stack([item[1] for item in batch]), device=self.device)
        rewards = _tensor_from_array(
            np.asarray([item[2] for item in batch], dtype=np.float32), device=self.device
        ).unsqueeze(-1)
        next_observations = _tensor_from_array(
            np.stack([item[3] for item in batch]), device=self.device
        )
        done = _tensor_from_array(
            np.asarray([item[4] for item in batch], dtype=np.float32), device=self.device
        ).unsqueeze(-1)
        discounts = _tensor_from_array(
            np.asarray([item[5] for item in batch], dtype=np.float32), device=self.device
        ).unsqueeze(-1)

        with torch.no_grad():
            next_residual, next_log_prob = self.actor.sample(next_observations)
            target_q = torch.minimum(
                self.target1(next_observations, next_residual),
                self.target2(next_observations, next_residual),
            ) - self.alpha * next_log_prob
            target = rewards + (1.0 - done) * discounts * target_q

        q1 = self.critic1(observations, residuals)
        q2 = self.critic2(observations, residuals)
        critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(
            q2, target
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_residual, log_prob = self.actor.sample(observations)
        actor_loss = (
            self.alpha * log_prob
            - torch.minimum(
                self.critic1(observations, sampled_residual),
                self.critic2(observations, sampled_residual),
            )
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for target, source in zip(self.target1.parameters(), self.critic1.parameters()):
                target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
            for target, source in zip(self.target2.parameters(), self.critic2.parameters()):
                target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "mean_log_prob": float(log_prob.detach().mean().cpu()),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "tau": self.tau,
        }

    def load_actor(self, payload: Mapping[str, Any]) -> None:
        if "actor" not in payload:
            raise KeyError("residual checkpoint is missing actor")
        self.actor.load_state_dict(payload["actor"])
        self.actor.eval()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _status(output: Path, event: str, completed: int, total: int) -> None:
    line = f"{event} completed={int(completed)} total={int(total)}"
    print(line, flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    _append_jsonl(
        output / "progress.jsonl",
        {"event": event, "completed": int(completed), "total": int(total)},
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    return device


def _tensor_from_array(values: Any, *, device: torch.device) -> torch.Tensor:
    """Avoid a slow Python-scalar copy while tolerating Torch/NumPy ABI skew."""

    array = np.asarray(values, dtype=np.float32)
    try:
        return torch.as_tensor(array, dtype=torch.float32, device=device)
    except (RuntimeError, TypeError):
        return torch.tensor(array.tolist(), dtype=torch.float32, device=device)


def _residual_enable_probability(low_level_actions: int, prog_explore: int) -> float:
    """Match the upstream linear residual-enable schedule for one environment."""

    if low_level_actions < 0 or prog_explore < 0:
        raise ValueError("low_level_actions and prog_explore must be non-negative")
    if prog_explore == 0:
        return 1.0
    return min(float(low_level_actions) / float(prog_explore), 1.0)


def _make_env(arguments: argparse.Namespace) -> Any:
    if arguments.dummy:
        return DummyPegInsertionEnv(episode_length=max(1, int(arguments.max_steps)))
    import sys

    reference_root = Path(arguments.reference_root).expanduser().resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Policy Decorator reference checkout is missing: {reference_root}")
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401

    try:
        __import__("envs.maniskill_fixed")
    except Exception as exc:
        raise RuntimeError(
            "could not register PegInsertionSide-v2 from envs.maniskill_fixed; "
            f"reference_root={reference_root}: {type(exc).__name__}: {exc}"
        ) from exc
    environment = gym.make(
        arguments.env_id,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        reward_mode="sparse",
        render_mode="rgb_array",
    )
    if tuple(environment.action_space.shape or ()) != (PEG_ACTION_DIM,):
        environment.close()
        raise ValueError(
            f"{arguments.env_id} action space must be ({PEG_ACTION_DIM},), "
            f"got {environment.action_space.shape}"
        )
    return environment


def _build_base(arguments: argparse.Namespace, device: torch.device) -> BasePolicy:
    contract = PegLightWAMContract(
        action_horizon=int(arguments.action_horizon),
        replan_steps=int(arguments.replan_steps),
    )
    if arguments.dummy:
        return DummyPegBasePolicy(contract)
    if not arguments.base_checkpoint or not arguments.base_config:
        raise ValueError(
            "formal Light-WAM residual execution requires --base-checkpoint and --base-config; "
            "the S1 Policy Decorator best.pt is not a Light-WAM checkpoint"
        )
    base = LightWAMBasePolicy(
        checkpoint_path=arguments.base_checkpoint,
        config_path=arguments.base_config,
        stats_path=arguments.base_stats,
        vendor_root=arguments.vendor_root,
        device=device,
        model_dtype=arguments.model_dtype,
        action_horizon=contract.action_horizon,
        num_inference_steps=int(arguments.ddim_steps),
        prompt=arguments.prompt,
        context_path=arguments.context_path,
        text_embedding_cache_dir=arguments.text_embedding_cache_dir,
    )
    if base.contract.replan_steps != contract.replan_steps:
        base.contract = PegLightWAMContract(
            **{
                **base.contract.as_dict(),
                "camera_names": tuple(base.contract.camera_names),
                "replan_steps": contract.replan_steps,
            }
        )
    return base


def _feature(
    observation: PegLightWAMObservation,
    base_chunk: np.ndarray,
    *,
    base: BasePolicy,
    actor_input: str,
) -> np.ndarray:
    proprio = np.asarray(base.normalize_proprio(observation), dtype=np.float32)
    if proprio.shape != (PEG_PROPRIO_DIM,):
        raise ValueError(
            f"normalized Peg proprio must have shape ({PEG_PROPRIO_DIM},), got {proprio.shape}"
        )
    if actor_input == "proprio":
        values = proprio
    elif actor_input == "proprio_base_action":
        values = np.concatenate(
            (proprio, np.asarray(base_chunk, dtype=np.float32).reshape(-1)), axis=-1
        )
    else:
        raise ValueError(f"unsupported actor_input={actor_input!r}")
    if not np.isfinite(values).all():
        raise ValueError("residual actor feature contains non-finite values")
    return np.ascontiguousarray(values, dtype=np.float32)


def _state_id(observation: PegLightWAMObservation) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(observation.proprio).tobytes())
    digest.update(bytes(observation.image.detach().cpu().contiguous().untyped_storage()))
    return digest.hexdigest()


def _env_reset(environment: Any, seed: int) -> tuple[Any, Any]:
    result = environment.reset(seed=int(seed))
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("ManiSkill reset must return (observation, info)")
    return result


def _success(info: Mapping[str, Any]) -> bool:
    if "success" not in info:
        raise KeyError('ManiSkill info is missing native "success"')
    values = np.asarray(info["success"]).reshape(-1)
    return bool(values.size and values[0])


def _save_residual_checkpoint(
    output: Path,
    *,
    agent: ResidualSAC,
    contract: ResidualContract,
    arguments: argparse.Namespace,
    base: BasePolicy,
    low_level_actions: int,
    updates: int,
) -> Path:
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "peg_insertion_lightwam_residual_sac",
        "contract": contract.as_dict(),
        "observation_dim": int(agent.actor.net[0].in_features),
        "residual_dim": int(contract.residual_dim),
        "hidden_dim": int(agent.actor.net[0].out_features),
        "seed": int(arguments.seed),
        "low_level_actions": int(low_level_actions),
        "updates": int(updates),
        "res_scale": float(arguments.res_scale),
        "action_std": np.asarray(base.action_std, dtype=np.float32).tolist(),
        "action_low": np.asarray(arguments._action_low, dtype=np.float32).tolist(),
        "action_high": np.asarray(arguments._action_high, dtype=np.float32).tolist(),
        "base_checkpoint": (
            None if arguments.base_checkpoint is None else str(arguments.base_checkpoint)
        ),
        "base_parameter_hash": base.parameter_hash,
        "agent": agent.checkpoint(),
    }
    path = checkpoint_dir / f"step_{int(low_level_actions):012d}.pt"
    torch.save(payload, path)
    torch.save(payload, checkpoint_dir / "latest.pt")
    return path


def run_training(arguments: argparse.Namespace) -> dict[str, Any]:
    output = Path(arguments.output_dir).expanduser().resolve()
    _prepare_output(output)
    if arguments.num_envs != 1:
        raise ValueError(
            "the first formal ManiSkill2 adapter supports one environment; set --num-envs=1 "
            "until a batched rgbd Light-WAM adapter is added"
        )
    if arguments.total_low_level_actions <= 0:
        raise ValueError("--total-low-level-actions must be positive")
    if arguments.env_id != PEG_ENV_ID:
        raise ValueError(
            f"this adapter is currently pinned to {PEG_ENV_ID}, got {arguments.env_id}"
        )
    if arguments.replan_steps <= 0 or arguments.replan_steps > arguments.action_horizon:
        raise ValueError("--replan-steps must be in [1, --action-horizon]")
    if arguments.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative")
    if arguments.batch_size <= 0 or arguments.replay_size <= 0:
        raise ValueError("--batch-size and --replay-size must be positive")
    if arguments.utd <= 0.0:
        raise ValueError("--utd must be positive")
    if not 0.0 < arguments.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0.0 < arguments.tau <= 1.0:
        raise ValueError("--tau must be in (0, 1]")
    if arguments.alpha < 0.0:
        raise ValueError("--alpha must be non-negative")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = _device(arguments.device)
    contract = ResidualContract(
        env_id=arguments.env_id,
        action_horizon=arguments.action_horizon,
        replan_steps=arguments.replan_steps,
        actor_input=arguments.actor_input,
    )
    environment = _make_env(arguments)
    arguments._action_low = np.asarray(environment.action_space.low, dtype=np.float32).copy()
    arguments._action_high = np.asarray(environment.action_space.high, dtype=np.float32).copy()
    base = _build_base(arguments, device)
    base_hash_before = base.parameter_hash
    observation_raw, _reset_info = _env_reset(environment, arguments.seed)
    processed = build_peg_lightwam_observation(observation_raw, contract=base.contract)
    base_chunk = np.asarray(base.predict(processed), dtype=np.float32)
    if base_chunk.shape != (contract.action_horizon, contract.action_dim):
        raise ValueError(
            f"base action chunk has wrong shape {base_chunk.shape}, "
            f"expected {(contract.action_horizon, contract.action_dim)}"
        )
    feature = _feature(processed, base_chunk, base=base, actor_input=contract.actor_input)
    residual_agent = ResidualSAC(
        observation_dim=int(feature.size),
        residual_dim=contract.residual_dim,
        hidden_dim=int(arguments.hidden_dim),
        device=device,
        actor_lr=arguments.actor_lr,
        critic_lr=arguments.critic_lr,
        alpha=arguments.alpha,
        gamma=arguments.gamma,
        tau=arguments.tau,
    )
    replay = ChunkReplayBuffer(arguments.replay_size)
    losses: list[dict[str, float]] = []
    low_level_actions = 0
    decisions = 0
    updates = 0
    episodes = 0
    next_report = max(1, min(arguments.total_low_level_actions, arguments.log_every))
    next_save = max(1, int(arguments.save_every))
    episode_seed = int(arguments.seed)
    episode_successes = 0
    current_episode_success = False
    update_credit = 0.0
    started = time.perf_counter()
    try:
        while low_level_actions < arguments.total_low_level_actions:
            if decisions > 0 and (processed is None or done):
                episodes += 1
                if current_episode_success:
                    episode_successes += 1
                episode_seed = int(arguments.seed + episodes)
                observation_raw, _reset_info = _env_reset(environment, episode_seed)
                processed = build_peg_lightwam_observation(observation_raw, contract=base.contract)
                base_chunk = np.asarray(base.predict(processed), dtype=np.float32)
                feature = _feature(
                    processed, base_chunk, base=base, actor_input=contract.actor_input
                )
                current_episode_success = False
            remaining = arguments.total_low_level_actions - low_level_actions
            execute_steps = min(arguments.replan_steps, remaining)
            # Keep the official Policy Decorator schedule: ``learning_starts``
            # selects uniform residual exploration, while ``prog_explore``
            # controls the probability that a residual is enabled at all.
            # These are separate knobs in the upstream SAC loop.
            if low_level_actions < arguments.learning_starts:
                residual_unit = np.random.uniform(
                    -1.0, 1.0, size=(contract.replan_steps, contract.action_dim)
                ).astype(np.float32)
            else:
                residual_flat, _ = residual_agent.action(feature, deterministic=False)
                residual_unit = residual_flat.reshape(contract.replan_steps, contract.action_dim)
            residual_enable_probability = _residual_enable_probability(
                low_level_actions, arguments.prog_explore
            )
            if np.random.random() >= residual_enable_probability:
                residual_unit = np.zeros_like(residual_unit)
            final_chunk = compose_peg_residual_action(
                base_chunk,
                residual_unit,
                action_std=base.action_std,
                res_scale=arguments.res_scale,
                action_low=arguments._action_low,
                action_high=arguments._action_high,
            )
            previous_feature = feature.copy()
            reward_total = 0.0
            discount = 1.0
            done = False
            success_once = False
            for horizon_index in range(execute_steps):
                observation_raw, reward, terminated, truncated, info = environment.step(
                    final_chunk[horizon_index]
                )
                reward_total += discount * float(np.asarray(reward).reshape(-1)[0])
                discount *= arguments.gamma
                low_level_actions += 1
                success_once = success_once or _success(info)
                done = bool(terminated) or bool(truncated)
                if done:
                    break
            current_episode_success = current_episode_success or success_once
            if done:
                next_feature = np.zeros_like(previous_feature)
            else:
                next_processed = build_peg_lightwam_observation(
                    observation_raw, contract=base.contract
                )
                next_base_chunk = np.asarray(base.predict(next_processed), dtype=np.float32)
                next_feature = _feature(
                    next_processed,
                    next_base_chunk,
                    base=base,
                    actor_input=contract.actor_input,
                )
                processed = next_processed
                base_chunk = next_base_chunk
                feature = next_feature
            replay.add(
                (
                    previous_feature,
                    residual_unit.reshape(-1),
                    reward_total,
                    next_feature,
                    done,
                    float(discount if not done else 0.0),
                )
            )
            decisions += 1
            if low_level_actions >= arguments.learning_starts and len(replay) >= 1:
                update_credit += float(arguments.utd)
                while update_credit >= 1.0:
                    losses.append(residual_agent.update(replay, arguments.batch_size))
                    updates += 1
                    update_credit -= 1.0
            if (
                low_level_actions >= next_report
                or low_level_actions >= arguments.total_low_level_actions
            ):
                _status(output, "STATUS", low_level_actions, arguments.total_low_level_actions)
                next_report += max(1, arguments.log_every)
            if (
                low_level_actions >= next_save
                or low_level_actions >= arguments.total_low_level_actions
            ):
                _save_residual_checkpoint(
                    output,
                    agent=residual_agent,
                    contract=contract,
                    arguments=arguments,
                    base=base,
                    low_level_actions=low_level_actions,
                    updates=updates,
                )
                next_save += max(1, arguments.save_every)
        if done:
            episodes += 1
            if current_episode_success:
                episode_successes += 1
    finally:
        base_hash_after = base.parameter_hash
        environment.close()
    if base_hash_before != base_hash_after:
        raise RuntimeError("formal residual training changed the frozen Light-WAM base hash")
    last_loss = losses[-1] if losses else None
    metrics = {
        "schema_version": 1,
        "result_label": "Training",
        "formal_result": not arguments.dummy,
        "dummy": bool(arguments.dummy),
        "env_id": arguments.env_id,
        "contract": contract.as_dict(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device.index or 0)
        if device.type == "cuda"
        else None,
        "executed_low_level_actions": low_level_actions,
        "decision_steps": decisions,
        "episodes_finished": episodes,
        "episode_success_rate": (episode_successes / episodes if episodes else None),
        "replay_size": len(replay),
        "sac_updates": updates,
        "last_loss": last_loss,
        "base_parameter_hash_before": base_hash_before,
        "base_parameter_hash_after": base_hash_after,
        "base_parameters_frozen": True,
        "elapsed_s": time.perf_counter() - started,
        "data_or_model_downloaded": False,
    }
    _write_json(
        output / "args.json",
        {key: value for key, value in vars(arguments).items() if not key.startswith("_")},
    )
    _write_json(output / "metrics.json", metrics)
    _status(output, "DONE", low_level_actions, arguments.total_low_level_actions)
    return metrics


def _load_residual_agent(
    checkpoint_path: str | Path,
    *,
    feature_dim: int,
    contract: ResidualContract,
    hidden_dim: int,
    device: torch.device,
    arguments: argparse.Namespace,
) -> tuple[ResidualSAC, Mapping[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"residual checkpoint does not exist: {path}")
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"residual checkpoint must be a mapping: {path}")
    saved = payload.get("contract")
    if not isinstance(saved, Mapping):
        raise ValueError(f"residual checkpoint has no contract: {path}")
    for key in (
        "env_id",
        "action_dim",
        "proprio_dim",
        "action_horizon",
        "replan_steps",
        "residual_horizon",
        "residual_channels",
        "actor_input",
    ):
        if saved.get(key) != contract.as_dict().get(key):
            raise ValueError(
                f"residual checkpoint contract mismatch for {key}: "
                f"saved={saved.get(key)!r} current={contract.as_dict().get(key)!r}"
            )
    saved_observation_dim = payload.get("observation_dim")
    if saved_observation_dim is not None and int(saved_observation_dim) != int(feature_dim):
        raise ValueError(
            "residual checkpoint observation dimension mismatch: "
            f"saved={saved_observation_dim} current={feature_dim}"
        )
    saved_residual_dim = payload.get("residual_dim")
    if saved_residual_dim is not None and int(saved_residual_dim) != contract.residual_dim:
        raise ValueError(
            "residual checkpoint residual dimension mismatch: "
            f"saved={saved_residual_dim} current={contract.residual_dim}"
        )
    saved_hidden_dim = int(payload.get("hidden_dim", hidden_dim))
    if saved_hidden_dim <= 0:
        raise ValueError(f"residual checkpoint hidden_dim must be positive, got {saved_hidden_dim}")
    agent_payload = payload.get("agent")
    if not isinstance(agent_payload, Mapping):
        raise ValueError(f"residual checkpoint has no valid agent payload: {path}")
    agent = ResidualSAC(
        observation_dim=feature_dim,
        residual_dim=contract.residual_dim,
        hidden_dim=saved_hidden_dim,
        device=device,
        actor_lr=arguments.actor_lr,
        critic_lr=arguments.critic_lr,
        alpha=arguments.alpha,
        gamma=arguments.gamma,
        tau=arguments.tau,
    )
    agent.load_actor(agent_payload)
    return agent, payload


def _evaluation_reset_seed(
    seed: int, episode_index: int, episodes_per_seed: int
) -> int:
    """Allocate disjoint deterministic reset-seed ranges for each eval seed."""

    return int(seed * episodes_per_seed + episode_index)


def _evaluation_inference_seed(reset_seed: int, decision_step: int) -> int:
    """Give base diffusion sampling identical seeds to paired evaluations."""

    if decision_step < 0:
        raise ValueError(f"decision_step must be non-negative, got {decision_step}")
    # Keep the mapping stable across separate base/residual processes while
    # leaving room for every replan within a 200-step episode.
    return int(reset_seed * 1_000_000 + decision_step)


def _check_saved_vector(
    payload: Mapping[str, Any],
    key: str,
    current: np.ndarray,
    *,
    checkpoint_path: Path,
) -> None:
    saved = payload.get(key)
    if saved is None:
        return
    saved_array = np.asarray(saved, dtype=np.float32)
    if saved_array.shape != current.shape or not np.allclose(
        saved_array, current, rtol=1e-5, atol=1e-6
    ):
        raise ValueError(
            f"residual checkpoint {key} does not match the current run: "
            f"checkpoint={checkpoint_path} saved_shape={saved_array.shape} "
            f"current_shape={current.shape}"
        )


def run_evaluation(arguments: argparse.Namespace) -> dict[str, Any]:
    output = Path(arguments.output_dir).expanduser().resolve()
    _prepare_output(output)
    if arguments.replan_steps <= 0 or arguments.replan_steps > arguments.action_horizon:
        raise ValueError("--replan-steps must be in [1, --action-horizon]")
    if arguments.env_id != PEG_ENV_ID:
        raise ValueError(
            f"this adapter is currently pinned to {PEG_ENV_ID}, got {arguments.env_id}"
        )
    if arguments.episodes_per_seed <= 0:
        raise ValueError("--episodes-per-seed must be positive")
    if arguments.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    device = _device(arguments.device)
    contract = ResidualContract(
        env_id=arguments.env_id,
        action_horizon=arguments.action_horizon,
        replan_steps=arguments.replan_steps,
        actor_input=arguments.actor_input,
    )
    environment = _make_env(arguments)
    arguments._action_low = np.asarray(environment.action_space.low, dtype=np.float32).copy()
    arguments._action_high = np.asarray(environment.action_space.high, dtype=np.float32).copy()
    base = _build_base(arguments, device)
    base_hash_before = base.parameter_hash
    residual_agent: ResidualSAC | None = None
    residual_payload: Mapping[str, Any] | None = None
    seeds = [int(value.strip()) for value in arguments.eval_seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--eval-seeds must contain at least one integer")
    episodes_total = len(seeds) * arguments.episodes_per_seed
    manifest_path = output / "episode_manifest.jsonl"
    episodes_path = output / "episodes.jsonl"
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for seed_index, seed in enumerate(seeds):
            for episode_index in range(arguments.episodes_per_seed):
                reset_seed = _evaluation_reset_seed(
                    seed,
                    episode_index,
                    arguments.episodes_per_seed,
                )
                observation_raw, reset_info = _env_reset(environment, reset_seed)
                del reset_info
                processed = build_peg_lightwam_observation(observation_raw, contract=base.contract)
                base_chunk = np.asarray(
                    base.predict(
                        processed,
                        seed=_evaluation_inference_seed(reset_seed, decision_step=0),
                    ),
                    dtype=np.float32,
                )
                feature = _feature(
                    processed, base_chunk, base=base, actor_input=contract.actor_input
                )
                if residual_agent is None and arguments.arm == "residual":
                    residual_agent, residual_payload = _load_residual_agent(
                        arguments.residual_checkpoint,
                        feature_dim=int(feature.size),
                        contract=contract,
                        hidden_dim=arguments.hidden_dim,
                        device=device,
                        arguments=arguments,
                    )
                    saved_base_hash = residual_payload.get("base_parameter_hash")
                    if (
                        saved_base_hash is not None
                        and saved_base_hash != base_hash_before
                    ):
                        raise ValueError(
                            "residual checkpoint was trained with a different frozen Light-WAM "
                            f"base: saved_hash={saved_base_hash} current_hash={base_hash_before}"
                        )
                    saved_res_scale = residual_payload.get("res_scale")
                    if saved_res_scale is not None and not np.isclose(
                        float(saved_res_scale), float(arguments.res_scale)
                    ):
                        raise ValueError(
                            "residual checkpoint res_scale does not match the current run: "
                            f"saved={saved_res_scale} current={arguments.res_scale}"
                        )
                    _check_saved_vector(
                        residual_payload,
                        "action_std",
                        np.asarray(base.action_std, dtype=np.float32),
                        checkpoint_path=Path(arguments.residual_checkpoint),
                    )
                    _check_saved_vector(
                        residual_payload,
                        "action_low",
                        arguments._action_low,
                        checkpoint_path=Path(arguments.residual_checkpoint),
                    )
                    _check_saved_vector(
                        residual_payload,
                        "action_high",
                        arguments._action_high,
                        checkpoint_path=Path(arguments.residual_checkpoint),
                    )
                episode_key = f"lightwam-peg-{seed_index}-{episode_index}"
                manifest = {
                    "schema_version": 1,
                    "episode_key": episode_key,
                    "policy_variant": arguments.arm,
                    "eval_seed": int(seed),
                    "policy_seed": int(seed),
                    "reset_seed": int(reset_seed),
                    "reset_state_id": _state_id(processed),
                    "requested_env_id": arguments.env_id,
                    "runtime_env_id": arguments.env_id,
                    "action_dim": contract.action_dim,
                    "proprio_dim": contract.proprio_dim,
                    "action_horizon": contract.action_horizon,
                    "execution_action_steps": contract.replan_steps,
                    "base_inference_seed_scheme": "reset_seed*1000000+decision_step",
                    "training_seed": (
                        None
                        if residual_payload is None
                        else residual_payload.get("seed")
                    ),
                }
                _append_jsonl(manifest_path, manifest)
                success_once = False
                first_success_step: int | None = None
                low_level_actions = 0
                decision_steps = 0
                return_total = 0.0
                done = False
                episode_started = time.perf_counter()
                while low_level_actions < arguments.max_steps and not done:
                    if arguments.arm == "base":
                        residual_unit = np.zeros_like(base_chunk)
                    else:
                        if residual_agent is None:
                            raise RuntimeError("residual agent was not loaded")
                        residual_flat, _ = residual_agent.action(feature, deterministic=True)
                        residual_unit = residual_flat.reshape(
                            contract.replan_steps, contract.action_dim
                        )
                    final_chunk = compose_peg_residual_action(
                        base_chunk,
                        residual_unit,
                        action_std=base.action_std,
                        res_scale=arguments.res_scale if arguments.arm == "residual" else 0.0,
                        action_low=arguments._action_low,
                        action_high=arguments._action_high,
                    )
                    for horizon_index in range(
                        min(contract.replan_steps, arguments.max_steps - low_level_actions)
                    ):
                        observation_raw, reward, terminated, truncated, info = environment.step(
                            final_chunk[horizon_index]
                        )
                        return_total += float(np.asarray(reward).reshape(-1)[0])
                        low_level_actions += 1
                        if _success(info) and not success_once:
                            success_once = True
                            first_success_step = low_level_actions
                        done = bool(terminated) or bool(truncated)
                        if done:
                            break
                    decision_steps += 1
                    if not done and low_level_actions < arguments.max_steps:
                        processed = build_peg_lightwam_observation(
                            observation_raw, contract=base.contract
                        )
                        base_chunk = np.asarray(
                            base.predict(
                                processed,
                                seed=_evaluation_inference_seed(
                                    reset_seed, decision_step=decision_steps
                                ),
                            ),
                            dtype=np.float32,
                        )
                        feature = _feature(
                            processed, base_chunk, base=base, actor_input=contract.actor_input
                        )
                row = {
                    **manifest,
                    "success": bool(success_once),
                    "success_once": bool(success_once),
                    "first_success_step": first_success_step,
                    "steps": low_level_actions,
                    "decision_steps": decision_steps,
                    "return": return_total,
                    "latency_s": time.perf_counter() - episode_started,
                    "model_kind": (
                        "frozen_lightwam_base"
                        if arguments.arm == "base"
                        else "frozen_lightwam_plus_residual_sac"
                    ),
                }
                rows.append(row)
                _append_jsonl(episodes_path, row)
                _status(output, "STATUS", len(rows), episodes_total)
    finally:
        base_hash_after = base.parameter_hash
        environment.close()
    if base_hash_before != base_hash_after:
        raise RuntimeError("evaluation changed the frozen Light-WAM base hash")
    metrics = {
        "schema_version": 1,
        "result_label": "Smoke" if arguments.dummy else "Closed-loop",
        "formal_result": not arguments.dummy,
        "dummy": bool(arguments.dummy),
        "requested_env_id": arguments.env_id,
        "runtime_env_id": arguments.env_id,
        "policy_variant": arguments.arm,
        "contract": contract.as_dict(),
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else None,
        "mean_return": float(np.mean([row["return"] for row in rows])) if rows else None,
        "mean_steps": float(np.mean([row["steps"] for row in rows])) if rows else None,
        "base_parameter_hash_before": base_hash_before,
        "base_parameter_hash_after": base_hash_after,
        "base_parameters_frozen": True,
        "elapsed_s": time.perf_counter() - started,
        "data_or_model_downloaded": False,
    }
    _write_json(output / "metrics.json", metrics)
    _status(output, "DONE", len(rows), episodes_total)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--env-id", default=PEG_ENV_ID)
    parser.add_argument("--reference-root", default="datasets/raw/policy_decorator/reference_repo")
    parser.add_argument("--vendor-root", default="third_party/lightwam/Light-WAM")
    parser.add_argument("--base-checkpoint", "--base-policy-ckpt", dest="base_checkpoint")
    parser.add_argument("--base-config", "--base-policy-config", dest="base_config")
    parser.add_argument("--base-stats", dest="base_stats")
    parser.add_argument("--context-path")
    parser.add_argument("--text-embedding-cache-dir")
    parser.add_argument("--prompt", default="insert the peg into the hole")
    parser.add_argument("--model-dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=PEG_DEFAULT_ACTION_HORIZON)
    parser.add_argument("--replan-steps", type=int, default=PEG_DEFAULT_REPLAN_STEPS)
    parser.add_argument("--res-scale", type=float, default=0.1)
    parser.add_argument(
        "--actor-input",
        choices=("proprio", "proprio_base_action"),
        default="proprio_base_action",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="1,2,3")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", required=True)
    parser.add_argument("--total-low-level-actions", type=int, default=4_000_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--actor-lr", "--policy-lr", dest="actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", "--q-lr", dest="critic_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--utd", type=float, default=0.25)
    parser.add_argument("--prog-explore", type=int, default=30_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--residual-checkpoint", "--residual-ckpt", dest="residual_checkpoint")
    parser.add_argument("--arm", choices=("base", "residual"), default="residual")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.action_horizon <= 0 or arguments.hidden_dim <= 0:
        raise ValueError("action-horizon and hidden-dim must be positive")
    if arguments.res_scale < 0.0 or arguments.prog_explore < 0:
        raise ValueError("res-scale must be non-negative and prog-explore must be non-negative")
    if arguments.log_every <= 0 or arguments.save_every <= 0:
        raise ValueError("log-every and save-every must be positive")
    if arguments.phase == "train":
        if arguments.dummy and arguments.total_low_level_actions > 64:
            arguments.total_low_level_actions = 64
        run_training(arguments)
    else:
        if arguments.arm == "residual" and not arguments.residual_checkpoint:
            raise ValueError("--arm=residual requires --residual-checkpoint")
        run_evaluation(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
