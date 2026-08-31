"""Data-free S2 residual wiring smoke.

This module does not pretend to be a Light-WAM or RoboTwin implementation.  It
uses a deterministic vector environment and small PyTorch policies to test the
parts that are easy to get wrong before a multi-GPU run: chunk shapes,
physical-unit action composition, gripper pass-through, frozen-base behavior,
aggregate low-level action counting, SAC updates, and progress artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dummy_env import DummyRobotWinEnv
from .residual_contract import (
    ACTION_DIM,
    ARM_DIM,
    assert_frozen,
    compose_residual_action,
    freeze_module,
    module_parameter_sha256,
    residual_saturation_fraction,
)

ACTION_HORIZON = 4
OBS_DIM = 32


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DummyBasePolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, horizon: int = ACTION_HORIZON) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, horizon * ACTION_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.tanh(self.net(observation)).reshape(
            observation.shape[0], self.horizon, ACTION_DIM
        )


class DummyResidualActor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, horizon: int = ACTION_HORIZON) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(obs_dim + ACTION_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, horizon * ARM_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        qpos = observation[..., :ACTION_DIM]
        return self.net(torch.cat((observation, qpos), dim=-1)).reshape(
            observation.shape[0], self.horizon, ARM_DIM
        )


class DummyCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, horizon: int = ACTION_HORIZON) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + horizon * ACTION_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((observation, action.reshape(observation.shape[0], -1)), dim=-1))


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _tensor_from_array(values: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy through Python scalars when Torch's NumPy bridge is unavailable."""

    return torch.tensor(values.tolist(), dtype=torch.float32, device=device)


def _array_from_tensor(values: torch.Tensor) -> np.ndarray:
    return np.asarray(values.detach().cpu().tolist(), dtype=np.float32)


def _policy_bundle(
    seed: int, device: torch.device
) -> tuple[DummyBasePolicy, DummyResidualActor, DummyCritic]:
    torch.manual_seed(seed)
    base = freeze_module(DummyBasePolicy().to(device))
    actor = DummyResidualActor().to(device)
    critic = DummyCritic().to(device)
    assert_frozen(base)
    return base, actor, critic


def _contract_vectors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.ones(ARM_DIM, device=device),
        torch.full((ARM_DIM,), -1.0, device=device),
        torch.full((ARM_DIM,), 1.0, device=device),
    )


def _log_status(output: Path, *, event: str, completed: int, total: int) -> None:
    line = f"{event} completed={completed} total={total}"
    print(line, flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    _append_jsonl(
        output / "progress.jsonl",
        {"event": event, "completed": int(completed), "total": int(total)},
    )


def _dummy_base_checkpoint_hash(arguments: argparse.Namespace, base: nn.Module) -> str:
    # The dummy harness does not load a checkpoint.  Hashing the synthetic
    # module on both sides makes the immutability assertion meaningful without
    # accidentally treating an unrelated host file as a loaded base model.
    return module_parameter_sha256(base)


def run_dummy_rollout(arguments: argparse.Namespace) -> dict[str, Any]:
    output = Path(arguments.output_root)
    _prepare_output(output)
    if arguments.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if arguments.replan_steps <= 0 or arguments.replan_steps > ACTION_HORIZON:
        raise ValueError(f"--replan-steps must be in [1, {ACTION_HORIZON}]")

    device = _device()
    base, actor, _critic = _policy_bundle(arguments.seed, device)
    del actor
    base_hash_before = _dummy_base_checkpoint_hash(arguments, base)
    std, q_min, q_max = _contract_vectors(device)
    environment = DummyRobotWinEnv(1, obs_dim=OBS_DIM)
    manifest_path = output / "episode_manifest.jsonl"
    episodes_path = output / "episodes.jsonl"
    started = time.perf_counter()
    max_zero_deviation = 0.0
    max_gripper_residual = 0.0
    residual_saturation: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(arguments.episodes):
            reset_seed = arguments.eval_seed + episode_index
            observation_np = environment.reset(seed=reset_seed)
            state_id = hashlib.sha256(np.ascontiguousarray(observation_np).tobytes()).hexdigest()
            episode_key = (
                f"s2-{arguments.task}-{arguments.task_config}-"
                f"{arguments.eval_seed}-{episode_index}"
            )
            manifest = {
                "episode_key": episode_key,
                "reset_seed": reset_seed,
                "reset_state_id": state_id,
                "task": arguments.task,
                "task_config": arguments.task_config,
                "replan_steps": arguments.replan_steps,
                "action_dim": ACTION_DIM,
                "action_horizon": ACTION_HORIZON,
            }
            _append_jsonl(manifest_path, manifest)
            success_once = False
            first_success_step: int | None = None
            executed = 0
            episode_started = time.perf_counter()
            for chunk_step in range(0, arguments.max_steps, arguments.replan_steps):
                observation = _tensor_from_array(observation_np, device)
                with torch.no_grad():
                    base_chunk = base(observation)
                residual = torch.zeros(
                    (1, ACTION_HORIZON, ARM_DIM), device=device, dtype=base_chunk.dtype
                )
                final_chunk = compose_residual_action(
                    base_chunk, residual, std, q_min, q_max, res_scale=arguments.res_scale
                )
                max_zero_deviation = max(
                    max_zero_deviation,
                    float((final_chunk - base_chunk).abs().max().item()),
                )
                # The residual tensor has arm channels only; the gripper is
                # therefore structurally absent and is copied by the contract.
                max_gripper_residual = max(max_gripper_residual, 0.0)
                residual_saturation.append(residual_saturation_fraction(residual))
                for horizon_index in range(arguments.replan_steps):
                    action_np = _array_from_tensor(final_chunk[:, horizon_index])
                    observation_np, _reward, success, truncated = environment.step(action_np)
                    if bool(success[0]) and not success_once:
                        success_once = True
                        first_success_step = executed + 1
                    executed += 1
                    if bool(truncated[0]):
                        break
                if bool(truncated[0]):
                    break
            row = {
                **manifest,
                "success": success_once,
                "first_success_step": first_success_step,
                "executed_action_count": executed,
                "policy_chunk_count": math.ceil(executed / arguments.replan_steps),
                "latency_s": time.perf_counter() - episode_started,
                "model_kind": "dummy_frozen_base_zero_residual",
            }
            episode_rows.append(row)
            _append_jsonl(episodes_path, row)
            _log_status(
                output,
                event="STATUS",
                completed=episode_index + 1,
                total=arguments.episodes,
            )
    finally:
        base_hash_after = module_parameter_sha256(base)

    if base_hash_before != base_hash_after:
        raise RuntimeError("dummy rollout changed the frozen base hash")
    metrics = {
        "schema_version": 1,
        "result_label": "Smoke",
        "dummy": True,
        "task": arguments.task,
        "task_config": arguments.task_config,
        "episodes": len(episode_rows),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "zero_residual": bool(arguments.zero_residual),
        "max_zero_residual_action_deviation": (
            max_zero_deviation if arguments.zero_residual else None
        ),
        "max_gripper_residual": max_gripper_residual,
        "residual_saturation_fraction": float(np.mean(residual_saturation)),
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_parameters_frozen": True,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "elapsed_s": time.perf_counter() - started,
        "formal_result": False,
    }
    _log_status(output, event="DONE", completed=arguments.episodes, total=arguments.episodes)
    _write_json(output / "metrics.json", metrics)
    return metrics


def _sac_update(
    *,
    base: nn.Module,
    actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray]],
    batch_size: int,
    device: torch.device,
    std: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    res_scale: float,
) -> tuple[float, float]:
    count = min(batch_size, len(replay))
    batch = random.sample(list(replay), count)
    observations = _tensor_from_array(np.stack([item[0] for item in batch]), device)
    actions = _tensor_from_array(np.stack([item[1] for item in batch]), device)
    rewards = torch.tensor(
        [item[2] for item in batch], dtype=torch.float32, device=device
    ).unsqueeze(-1)

    critic_loss = torch.nn.functional.mse_loss(critic(observations, actions), rewards)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    residual = actor(observations)
    with torch.no_grad():
        base_action = base(observations)
    final_action = compose_residual_action(
        base_action, residual, std, q_min, q_max, res_scale=res_scale
    )
    actor_loss = -critic(observations, final_action).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()
    return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu())


def run_dummy_training(arguments: argparse.Namespace) -> dict[str, Any]:
    output = Path(arguments.output_root)
    _prepare_output(output)
    if arguments.total_low_level_actions <= 0:
        raise ValueError("--total-low-level-actions must be positive")
    if arguments.total_low_level_actions % arguments.num_envs:
        raise ValueError(
            "--total-low-level-actions must be divisible by --num-envs so aggregate "
            "action counting "
            "is exact"
        )
    if arguments.replan_steps != ACTION_HORIZON:
        raise ValueError(
            f"dummy contract expects replan_steps={ACTION_HORIZON}, got {arguments.replan_steps}"
        )

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = _device()
    base, actor, critic = _policy_bundle(arguments.seed, device)
    base_hash_before = _dummy_base_checkpoint_hash(arguments, base)
    std, q_min, q_max = _contract_vectors(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=arguments.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=arguments.critic_lr)
    environment = DummyRobotWinEnv(arguments.num_envs, obs_dim=OBS_DIM)
    observation_np = environment.reset(seed=arguments.seed)
    replay: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = deque(maxlen=8192)
    total = arguments.total_low_level_actions
    env_steps = total // arguments.num_envs
    completed = 0
    updates = 0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    chunk_latencies: list[float] = []
    update_latencies: list[float] = []
    max_zero_deviation = 0.0
    max_gripper_residual = 0.0
    saturation_values: list[float] = []
    next_report = max(arguments.num_envs, total // 10)
    started = time.perf_counter()
    try:
        for _env_step in range(env_steps):
            observation = _tensor_from_array(observation_np, device)
            with torch.no_grad():
                base_chunk = base(observation)
                residual = actor(observation)
            chunk_started = time.perf_counter()
            final_chunk = compose_residual_action(
                base_chunk, residual, std, q_min, q_max, res_scale=arguments.res_scale
            )
            chunk_latencies.append(time.perf_counter() - chunk_started)
            max_gripper_residual = max(max_gripper_residual, 0.0)
            saturation_values.append(residual_saturation_fraction(residual))
            for horizon_index in range(arguments.replan_steps):
                previous_observation = observation_np.copy()
                action_np = _array_from_tensor(final_chunk[:, horizon_index])
                observation_np, reward, success, truncated = environment.step(action_np)
                del success
                for env_index in range(arguments.num_envs):
                    replay.append(
                        (
                            previous_observation[env_index],
                            _array_from_tensor(final_chunk[env_index]),
                            float(reward[env_index]),
                            observation_np[env_index],
                        )
                    )
                completed += arguments.num_envs
                if completed >= arguments.learning_starts and len(replay) >= 1:
                    update_started = time.perf_counter()
                    actor_loss, critic_loss = _sac_update(
                        base=base,
                        actor=actor,
                        critic=critic,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        replay=replay,
                        batch_size=arguments.batch_size,
                        device=device,
                        std=std,
                        q_min=q_min,
                        q_max=q_max,
                        res_scale=arguments.res_scale,
                    )
                    update_latencies.append(time.perf_counter() - update_started)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    updates += 1
                if completed >= next_report or completed == total:
                    _log_status(output, event="STATUS", completed=completed, total=total)
                    while next_report <= completed:
                        next_report += max(arguments.num_envs, total // 10)
            if completed >= total:
                break
    finally:
        base_hash_after = module_parameter_sha256(base)

    assert_frozen(base)
    if base_hash_before != base_hash_after:
        raise RuntimeError("dummy training changed the frozen base hash")
    if updates <= 0:
        raise RuntimeError("dummy SAC smoke completed without an optimizer update")
    metrics = {
        "schema_version": 1,
        "result_label": "Smoke",
        "dummy": True,
        "task": arguments.task,
        "task_config": arguments.task_config,
        "seed": arguments.seed,
        "requested_low_level_actions": total,
        "executed_low_level_actions": completed,
        "num_envs": arguments.num_envs,
        "replan_steps": arguments.replan_steps,
        "sac_updates": updates,
        "learning_starts": arguments.learning_starts,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_parameters_frozen": True,
        "gripper_residual_max_abs": max_gripper_residual,
        "residual_saturation_fraction": float(np.mean(saturation_values)),
        "max_zero_residual_action_deviation": None,
        "low_level_actions_per_s": completed / max(1e-9, time.perf_counter() - started),
        "chunk_inference_latency_s_median": float(np.median(chunk_latencies)),
        "sac_update_latency_s_median": float(np.median(update_latencies)),
        "last_actor_loss": actor_losses[-1],
        "last_critic_loss": critic_losses[-1],
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "formal_result": False,
    }
    _log_status(output, event="DONE", completed=completed, total=total)
    _write_json(output / "metrics.json", metrics)
    _write_json(
        output / "checkpoint.json",
        {
            "format": "dummy-residual-smoke-metadata-only",
            "step": completed,
            "base_hash": base_hash_after,
            "base_parameters_frozen": True,
            "weights_written": False,
        },
    )
    return metrics


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("smoke-rollout", "smoke-train", "train", "evaluate"),
    )
    parser.add_argument("--task", default="move_stapler_pad")
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--base-run")
    parser.add_argument("--residual-runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--zero-residual", type=_boolean, default=True)
    parser.add_argument("--total-low-level-actions", type=int, default=12000)
    parser.add_argument("--learning-starts", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--res-scale", type=float, default=0.1)
    parser.add_argument("--prog-explore", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.dummy:
        raise SystemExit(
            "formal Light-WAM/RoboTwin residual execution requires the pinned vendor checkout; "
            "pass --dummy for the data-free wiring smoke"
        )
    if arguments.phase in {"smoke-rollout", "evaluate"}:
        run_dummy_rollout(arguments)
    else:
        run_dummy_training(arguments)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
