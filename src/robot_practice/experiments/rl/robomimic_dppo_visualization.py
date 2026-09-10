"""Record and visualize the official robomimic DPPO pixel evaluation.

The upstream DPPO evaluator writes only aggregate ``result.npz`` metrics.  This
module deliberately keeps the upstream model and environment construction, but
adds a small recording layer around the same closed-loop evaluation contract:

* one evaluation seed owns 50 environments;
* each environment receives the seed ``policy_seed + env_index``;
* 100 policy decisions of four low-level actions complete one 400-step episode;
  the recorded policy-level timeline therefore runs at 5 Hz (the environment
  control frequency is 20 Hz);
* the policy receives the same ``agentview_image`` and normalized 9D state;
* the normalized action chunk returned by DPPO is sent to the environment.

``record`` is intended for the DPPO runtime (it imports the vendored upstream
checkout lazily and writes dependency-light NPZ bundles).  ``visualize`` is
intended for the local ``robot-practice`` environment and converts those bundles
to one Rerun recording per episode, plus an optional merged selector archive.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
TASK_SUITE = "robomimic"
TASK_NAME = "square"
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_EPISODES_PER_SEED = 50
DEFAULT_EVAL_STEPS = 100
DEFAULT_ENV_COUNT = 50
DEFAULT_ACT_STEPS = 4
DEFAULT_DDIM_STEPS = 5
DEFAULT_FT_DENOISING_STEPS = 5
DEFAULT_SUCCESS_THRESHOLD = 1.0
DEFAULT_CONTROL_FPS = 20.0
DEFAULT_FPS = DEFAULT_CONTROL_FPS / DEFAULT_ACT_STEPS
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
STATE_NAMES = (
    "robot0_eef_pos_x",
    "robot0_eef_pos_y",
    "robot0_eef_pos_z",
    "robot0_eef_quat_x",
    "robot0_eef_quat_y",
    "robot0_eef_quat_z",
    "robot0_eef_quat_w",
    "robot0_gripper_qpos_left",
    "robot0_gripper_qpos_right",
)
APP_ID = "robot_practice_robomimic_dppo"
TIMELINE = "episode_step"
IMAGE_PATH = "episode/camera/agentview"
SCENE_ROOT = "episode/scene"
SIGNALS_ROOT = "episode/signals"
METADATA_PATH = "episode/metadata"
IMAGE_SIZE = (96, 96)
ACTION_DIM = 7
STATE_DIM = 9


def _parse_int_csv(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique integers")
    if any(item < 0 for item in values):
        raise ValueError(f"{name} must contain non-negative integers")
    return values


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}\n"
    with path.open("a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def _prepare_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"expected one RGB image with three dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"expected an RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmax(image)) <= 1.0 + 1e-6:
            image = image * 255.0
    return np.asarray(np.clip(image, 0, 255), dtype=np.uint8)


def _batch_image_to_hwc(batch: Any, index: int) -> np.ndarray:
    value = np.asarray(batch)[index]
    if value.ndim == 4:
        value = value[-1]
    return _image_to_hwc_uint8(value)


def _batch_state(batch: Any, index: int) -> np.ndarray:
    value = np.asarray(batch)[index]
    if value.ndim == 2:
        value = value[-1]
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (STATE_DIM,):
        raise ValueError(
            f"expected one normalized state with shape {(STATE_DIM,)}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("state contains non-finite values")
    return np.ascontiguousarray(value)


def _load_observation_normalization(path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path) as normalization:
        obs_min = np.asarray(normalization["obs_min"], dtype=np.float32)
        obs_max = np.asarray(normalization["obs_max"], dtype=np.float32)
    if obs_min.shape != (STATE_DIM,) or obs_max.shape != (STATE_DIM,):
        raise ValueError(
            f"expected 9D observation normalization, got {obs_min.shape} and {obs_max.shape}"
        )
    return obs_min, obs_max


def _unnormalize_state(
    normalized: np.ndarray,
    normalization: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray | None:
    if normalization is None:
        return None
    obs_min, obs_max = normalization
    return ((normalized + 1.0) * 0.5 * (obs_max - obs_min) + obs_min).astype(
        np.float32,
        copy=False,
    )


@dataclass(frozen=True)
class DPPOEpisode:
    """One recorded policy-evaluation episode."""

    episode_id: int
    policy_seed: int
    env_index: int
    environment_seed: int
    frames: np.ndarray
    state_normalized: np.ndarray
    state_raw: np.ndarray | None
    action_chunks: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    act_steps: int
    success_threshold: float
    return_total: float
    success: bool
    first_success_step: int | None

    @property
    def steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def episode_best_reward(self) -> float:
        if self.rewards.size == 0:
            return 0.0
        return float(np.max(self.rewards) / self.act_steps)

    @property
    def eef_path(self) -> np.ndarray:
        state = self.state_raw if self.state_raw is not None else self.state_normalized
        return np.asarray(state[:, :3], dtype=np.float32)


def _validate_episode(episode: DPPOEpisode) -> None:
    steps = episode.rewards.shape[0]
    if episode.frames.shape[0] != steps:
        raise ValueError(
            f"frames and rewards length mismatch: {episode.frames.shape[0]} != {steps}"
        )
    if episode.state_normalized.shape != (steps, STATE_DIM):
        raise ValueError(
            "state_normalized must have shape "
            f"{(steps, STATE_DIM)}, got {episode.state_normalized.shape}"
        )
    if episode.state_raw is not None and episode.state_raw.shape != (steps, STATE_DIM):
        raise ValueError(
            f"state_raw must have shape {(steps, STATE_DIM)}, got {episode.state_raw.shape}"
        )
    if episode.action_chunks.ndim != 3 or episode.action_chunks.shape[0] != steps:
        raise ValueError(
            "action_chunks must have shape [steps,horizon,7], "
            f"got {episode.action_chunks.shape}"
        )
    if episode.action_chunks.shape[2] != ACTION_DIM:
        raise ValueError(f"action_chunks action dimension must be {ACTION_DIM}")
    for name, array in (
        ("frames", episode.frames),
        ("state_normalized", episode.state_normalized),
        ("action_chunks", episode.action_chunks),
        ("rewards", episode.rewards),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if episode.frames.ndim != 4 or episode.frames.shape[-1] != 3:
        raise ValueError(
            "frames must have shape [steps,height,width,3], "
            f"got {episode.frames.shape}"
        )


def save_episode(path: Path, episode: DPPOEpisode) -> None:
    _validate_episode(episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    temporary.unlink(missing_ok=True)
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        episode_id=np.asarray(episode.episode_id, dtype=np.int64),
        policy_seed=np.asarray(episode.policy_seed, dtype=np.int64),
        env_index=np.asarray(episode.env_index, dtype=np.int64),
        environment_seed=np.asarray(episode.environment_seed, dtype=np.int64),
        frames=episode.frames,
        state_normalized=episode.state_normalized,
        state_raw=(
            episode.state_raw
            if episode.state_raw is not None
            else np.empty((0, STATE_DIM), dtype=np.float32)
        ),
        action_chunks=episode.action_chunks,
        rewards=episode.rewards,
        terminated=episode.terminated,
        truncated=episode.truncated,
        act_steps=np.asarray(episode.act_steps, dtype=np.int64),
        success_threshold=np.asarray(episode.success_threshold, dtype=np.float32),
        return_total=np.asarray(episode.return_total, dtype=np.float32),
        success=np.asarray(episode.success, dtype=np.bool_),
        first_success_step=np.asarray(
            -1 if episode.first_success_step is None else episode.first_success_step,
            dtype=np.int64,
        ),
    )
    temporary.replace(path)


def load_episode(path: Path) -> DPPOEpisode:
    if not path.is_file():
        raise FileNotFoundError(f"DPPO raw episode is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        schema_version = int(data["schema_version"])
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported DPPO visualization schema: {schema_version}")
        raw_state = np.asarray(data["state_raw"], dtype=np.float32)
        state_raw = None if raw_state.size == 0 else raw_state
        first_success_step = int(data["first_success_step"])
        episode = DPPOEpisode(
            episode_id=int(data["episode_id"]),
            policy_seed=int(data["policy_seed"]),
            env_index=int(data["env_index"]),
            environment_seed=int(data["environment_seed"]),
            frames=np.asarray(data["frames"], dtype=np.uint8),
            state_normalized=np.asarray(data["state_normalized"], dtype=np.float32),
            state_raw=state_raw,
            action_chunks=np.asarray(data["action_chunks"], dtype=np.float32),
            rewards=np.asarray(data["rewards"], dtype=np.float32),
            terminated=np.asarray(data["terminated"], dtype=np.bool_),
            truncated=np.asarray(data["truncated"], dtype=np.bool_),
            act_steps=int(data["act_steps"]),
            success_threshold=float(data["success_threshold"]),
            return_total=float(data["return_total"]),
            success=bool(data["success"]),
            first_success_step=None if first_success_step < 0 else first_success_step,
        )
    _validate_episode(episode)
    return episode


def _episode_summary(episode: DPPOEpisode, *, file: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "policy_seed": episode.policy_seed,
        "env_index": episode.env_index,
        "environment_seed": episode.environment_seed,
        "steps": episode.steps,
        "frames": int(episode.frames.shape[0]),
        "success": episode.success,
        "first_success_step": episode.first_success_step,
        "return_total": episode.return_total,
        "episode_best_reward": episode.episode_best_reward,
        "terminated_steps": int(np.count_nonzero(episode.terminated)),
        "truncated_steps": int(np.count_nonzero(episode.truncated)),
    }
    if file is not None:
        summary["file"] = file
    return summary


def _record_contract(
    *,
    seeds: Sequence[int],
    env_count: int,
    eval_steps: int,
    act_steps: int,
    ddim_steps: int,
    ft_denoising_steps: int,
    checkpoint_step: int | None,
    policy: str,
) -> dict[str, Any]:
    return {
        "task_suite": TASK_SUITE,
        "task": TASK_NAME,
        "policy": policy,
        "seeds": [int(seed) for seed in seeds],
        "episodes_per_seed": int(env_count),
        "expected_total_episodes": int(len(seeds) * env_count),
        "eval_envs": int(env_count),
        "eval_steps": int(eval_steps),
        "max_episode_steps": int(eval_steps * act_steps),
        "act_steps": int(act_steps),
        "horizon_steps": 4,
        "action_dim": ACTION_DIM,
        "observation_keys": [
            "agentview_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        "image_shape": [3, 96, 96],
        "state_dim": STATE_DIM,
        "ddpm_denoising_steps": 100,
        "ddim_steps": int(ddim_steps),
        "ft_denoising_steps": int(ft_denoising_steps),
        "checkpoint_step": None if checkpoint_step is None else int(checkpoint_step),
        "reset_seed_scheme": "policy_seed_plus_env_index",
        "control_frequency_hz": DEFAULT_CONTROL_FPS,
        "policy_frequency_hz": DEFAULT_CONTROL_FPS / act_steps,
        "recorded_frame_semantics": (
            "pre_action_observation_at_each_policy_decision; "
            "one frame represents act_steps low-level actions"
        ),
        "success_definition": "max(aggregated_reward) / act_steps >= 1",
    }


def _checkpoint_for_seed(
    checkpoint_root: Path | None,
    checkpoint: Path | None,
    seed: int,
    checkpoint_step: int | None,
) -> Path:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DPPO checkpoint is missing: {checkpoint}")
        return checkpoint
    if checkpoint_root is None:
        raise ValueError("one of checkpoint_root or checkpoint must be provided")
    if checkpoint_step is None:
        raise ValueError("checkpoint_step is required for a per-seed checkpoint root")
    path = checkpoint_root / f"seed_{seed}" / "checkpoint" / f"state_{checkpoint_step}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"DPPO checkpoint for seed {seed} is missing: {path}")
    return path


def _infer_checkpoint_step(checkpoint: Path) -> int | None:
    match = re.fullmatch(r"state_(\d+)\.pt", checkpoint.name)
    return None if match is None else int(match.group(1))


def _compose_eval_config(
    *,
    dppo_root: Path,
    checkpoint: Path,
    normalization: Path,
    output_dir: Path,
    seed: int,
    env_count: int,
    eval_steps: int,
    ddim_steps: int,
    ft_denoising_steps: int,
) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = dppo_root / "cfg" / "robomimic" / "eval" / "square"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="eval_diffusion_mlp_img",
            overrides=[
                f"base_policy_path={checkpoint}",
                f"normalization_path={normalization}",
                "robomimic_env_cfg_path="
                f"{dppo_root / 'cfg' / 'robomimic' / 'env_meta' / 'square-img.json'}",
                f"logdir={output_dir / 'upstream_eval'}",
                f"seed={seed}",
                "device=cuda:0",
                f"ddim_steps={ddim_steps}",
                f"ft_denoising_steps={ft_denoising_steps}",
                f"n_steps={eval_steps}",
                f"env.n_envs={env_count}",
                "render_num=0",
            ],
        )
    OmegaConf.resolve(config)
    return config


def record_seed(
    *,
    dppo_root: Path,
    checkpoint: Path,
    normalization_path: Path,
    output_dir: Path,
    seed: int,
    env_count: int,
    eval_steps: int,
    act_steps: int,
    ddim_steps: int,
    ft_denoising_steps: int,
    checkpoint_step: int | None,
    policy: str,
    env_indices: Sequence[int],
    success_threshold: float,
    gpu: str,
) -> dict[str, Any]:
    """Run one exact upstream evaluation seed while storing selected episodes."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if str(dppo_root) not in sys.path:
        sys.path.insert(0, str(dppo_root))

    from agent.eval.eval_diffusion_img_agent import EvalImgDiffusionAgent
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    normalization = _load_observation_normalization(normalization_path)
    config = _compose_eval_config(
        dppo_root=dppo_root,
        checkpoint=checkpoint,
        normalization=normalization_path,
        output_dir=output_dir,
        seed=seed,
        env_count=env_count,
        eval_steps=eval_steps,
        ddim_steps=ddim_steps,
        ft_denoising_steps=ft_denoising_steps,
    )
    agent = EvalImgDiffusionAgent(config)
    if int(agent.act_steps) != int(act_steps):
        raise ValueError(f"configured act_steps={agent.act_steps}, expected {act_steps}")
    if agent.n_envs != env_count or agent.n_steps != eval_steps:
        raise ValueError("recording config does not match requested evaluation contract")

    selected = tuple(int(index) for index in env_indices)
    if any(index < 0 or index >= env_count for index in selected):
        raise ValueError(f"selected env index must be in [0, {env_count}), got {selected}")
    buffers: dict[int, dict[str, list[Any]]] = {
        index: {
            "frames": [],
            "state_normalized": [],
            "action_chunks": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
        }
        for index in selected
    }
    _log(
        run_log,
        f"START seed={seed} gpu={gpu} envs={env_count} eval_steps={eval_steps} "
        f"selected={len(selected)} checkpoint={checkpoint}",
    )
    try:
        prev_obs = agent.reset_env_all()
        for step in range(eval_steps):
            with torch.no_grad():
                cond = {
                    key: torch.from_numpy(prev_obs[key]).float().to(agent.device)
                    for key in agent.obs_dims
                }
                samples = agent.model(cond=cond, deterministic=True)
                actions = samples.trajectories.cpu().numpy()
            action_venv = actions[:, : agent.act_steps]
            obs, rewards, terminated, truncated, _ = agent.venv.step(action_venv)
            for index, buffer in buffers.items():
                buffer["frames"].append(_batch_image_to_hwc(prev_obs["rgb"], index))
                buffer["state_normalized"].append(_batch_state(prev_obs["state"], index))
                buffer["action_chunks"].append(np.asarray(actions[index], dtype=np.float32))
                buffer["rewards"].append(float(rewards[index]))
                buffer["terminated"].append(bool(terminated[index]))
                buffer["truncated"].append(bool(truncated[index]))
            prev_obs = obs
            if step % 10 == 0 or step + 1 == eval_steps:
                _log(run_log, f"STATUS seed={seed} completed={step + 1} total={eval_steps}")

        episode_summaries: list[dict[str, Any]] = []
        for index, buffer in buffers.items():
            frames = np.stack(buffer["frames"]).astype(np.uint8, copy=False)
            state_normalized = np.stack(buffer["state_normalized"]).astype(np.float32, copy=False)
            state_raw_list = [
                _unnormalize_state(state, normalization) for state in state_normalized
            ]
            state_raw = (
                np.stack(state_raw_list).astype(np.float32, copy=False)
                if state_raw_list and state_raw_list[0] is not None
                else None
            )
            action_chunks = np.stack(buffer["action_chunks"]).astype(np.float32, copy=False)
            rewards_array = np.asarray(buffer["rewards"], dtype=np.float32)
            terminated_array = np.asarray(buffer["terminated"], dtype=np.bool_)
            truncated_array = np.asarray(buffer["truncated"], dtype=np.bool_)
            success_scores = rewards_array / float(act_steps)
            success_indices = np.flatnonzero(success_scores >= success_threshold)
            first_success_step = None if success_indices.size == 0 else int(success_indices[0]) + 1
            episode = DPPOEpisode(
                episode_id=index,
                policy_seed=seed,
                env_index=index,
                environment_seed=seed + index,
                frames=frames,
                state_normalized=state_normalized,
                state_raw=state_raw,
                action_chunks=action_chunks,
                rewards=rewards_array,
                terminated=terminated_array,
                truncated=truncated_array,
                act_steps=act_steps,
                success_threshold=success_threshold,
                return_total=float(rewards_array.sum()),
                success=bool(success_indices.size),
                first_success_step=first_success_step,
            )
            path = output_dir / f"episode_{index:04d}.npz"
            save_episode(path, episode)
            episode_summaries.append(_episode_summary(episode, file=path.name))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "robomimic_dppo_raw_rollouts",
            "contract": _record_contract(
                seeds=(seed,),
                env_count=env_count,
                eval_steps=eval_steps,
                act_steps=act_steps,
                ddim_steps=ddim_steps,
                ft_denoising_steps=ft_denoising_steps,
                checkpoint_step=checkpoint_step,
                policy=policy,
            ),
            "policy_seed": seed,
            "checkpoint": str(checkpoint),
            "normalization": str(normalization_path),
            "available_episode_count": env_count,
            "selected_episode_ids": list(selected),
            "recorded_episode_count": len(episode_summaries),
            "episodes": sorted(episode_summaries, key=lambda value: int(value["episode_id"])),
        }
        _write_json(output_dir / "manifest.json", manifest)
        _log(
            run_log,
            f"DONE seed={seed} completed={len(episode_summaries)} total={len(selected)} failures=0",
        )
        return manifest
    finally:
        try:
            agent.venv.close()
        finally:
            del agent
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def record(
    *,
    dppo_root: Path,
    checkpoint_root: Path | None = None,
    checkpoint: Path | None = None,
    normalization_path: Path,
    output_root: Path,
    seeds: Sequence[int],
    env_count: int = DEFAULT_ENV_COUNT,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    act_steps: int = DEFAULT_ACT_STEPS,
    ddim_steps: int = DEFAULT_DDIM_STEPS,
    ft_denoising_steps: int = DEFAULT_FT_DENOISING_STEPS,
    checkpoint_step: int | None = None,
    policy: str | None = None,
    max_record_episodes: int | None = None,
    env_indices: Sequence[int] | None = None,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    gpu: str = "0",
) -> dict[str, Any]:
    """Record selected episodes for all requested DPPO seeds.

    ``checkpoint_root`` selects one checkpoint below ``seed_<N>/`` per policy
    seed (the fine-tuned DPPO case).  ``checkpoint`` selects one shared
    checkpoint for every policy seed (the frozen base case).
    """

    if env_count <= 0 or eval_steps <= 0 or act_steps <= 0:
        raise ValueError("env_count, eval_steps, and act_steps must be positive")
    if (checkpoint_root is None) == (checkpoint is None):
        raise ValueError("provide exactly one of checkpoint_root or checkpoint")
    if checkpoint is not None and checkpoint_step is None:
        checkpoint_step = _infer_checkpoint_step(checkpoint)
    if checkpoint_root is not None and checkpoint_step is None:
        checkpoint_step = 300
    if policy is None:
        policy = (
            "shared_checkpoint"
            if checkpoint is not None
            else "official_dppo_pixel_finetuned"
        )
    if max_record_episodes is not None and not 0 < max_record_episodes <= env_count:
        raise ValueError(f"max_record_episodes must be in [1, {env_count}]")
    if env_indices is not None:
        selected = tuple(env_indices)
        if not selected or any(index < 0 or index >= env_count for index in selected):
            raise ValueError(f"env_indices must be non-empty and lie in [0, {env_count})")
    else:
        count = env_count if max_record_episodes is None else max_record_episodes
        selected = tuple(range(count))
    if not normalization_path.is_file():
        raise FileNotFoundError(f"normalization file is missing: {normalization_path}")
    checkpoint_by_seed = {
        seed: _checkpoint_for_seed(checkpoint_root, checkpoint, seed, checkpoint_step)
        for seed in seeds
    }
    _prepare_empty_directory(output_root)
    seed_manifests: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        _prepare_empty_directory(seed_root)
        seed_manifests.append(
            record_seed(
                dppo_root=dppo_root,
                checkpoint=checkpoint_by_seed[seed],
                normalization_path=normalization_path,
                output_dir=seed_root,
                seed=seed,
                env_count=env_count,
                eval_steps=eval_steps,
                act_steps=act_steps,
                ddim_steps=ddim_steps,
                ft_denoising_steps=ft_denoising_steps,
                checkpoint_step=checkpoint_step,
                policy=policy,
                env_indices=selected,
                success_threshold=success_threshold,
                gpu=gpu,
            )
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "robomimic_dppo_raw_rollouts",
        "policy": policy,
        "checkpoint_mode": "shared" if checkpoint is not None else "per_seed",
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "contract": _record_contract(
            seeds=seeds,
            env_count=env_count,
            eval_steps=eval_steps,
            act_steps=act_steps,
            ddim_steps=ddim_steps,
            ft_denoising_steps=ft_denoising_steps,
            checkpoint_step=checkpoint_step,
            policy=policy,
        ),
        "selected_episode_ids_per_seed": list(selected),
        "recorded_episode_count": int(len(seeds) * len(selected)),
        "seed_manifests": [
            {"seed": int(seed), "manifest": f"seed_{seed}/manifest.json"}
            for seed in seeds
        ],
        "episodes": [
            {
                **summary,
                "file": f"seed_{seed}/episode_{int(summary['episode_id']):04d}.npz",
            }
            for seed, seed_manifest in zip(seeds, seed_manifests, strict=True)
            for summary in seed_manifest["episodes"]
        ],
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _read_recorded_episodes(raw_root: Path) -> list[tuple[Path, DPPOEpisode]]:
    manifest_path = raw_root / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text())
        records = payload.get("episodes", [])
        if not isinstance(records, list):
            raise ValueError(f"episodes must be a list in {manifest_path}")
        result: list[tuple[Path, DPPOEpisode]] = []
        for record in records:
            if not isinstance(record, Mapping) or "file" not in record:
                raise ValueError(f"invalid episode record in {manifest_path}")
            path = raw_root / str(record["file"])
            result.append((path, load_episode(path)))
        return sorted(result, key=lambda item: (item[1].policy_seed, item[1].episode_id))
    paths = sorted(raw_root.glob("seed_*/episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no DPPO raw episode manifest or bundles found under {raw_root}")
    return [(path, load_episode(path)) for path in paths]


def _build_blueprint(rrb: Any, *, fps: float) -> Any:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    name="Agentview RGB",
                    origin=IMAGE_PATH,
                    contents=[IMAGE_PATH],
                ),
                rrb.Spatial3DView(
                    name="End-effector trajectory",
                    origin=SCENE_ROOT,
                    contents=[f"{SCENE_ROOT}/**"],
                ),
                column_shares=[2.0, 1.0],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Reward and success",
                    origin=f"{SIGNALS_ROOT}/reward",
                    contents=[
                        f"{SIGNALS_ROOT}/reward",
                        f"{SIGNALS_ROOT}/success_score",
                        f"{SIGNALS_ROOT}/cumulative_reward",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="First action of each chunk",
                    origin=f"{SIGNALS_ROOT}/action/first",
                    contents=[f"{SIGNALS_ROOT}/action/first/**"],
                ),
                rrb.TimeSeriesView(
                    name="Full action chunk",
                    origin=f"{SIGNALS_ROOT}/action/chunk",
                    contents=[f"{SIGNALS_ROOT}/action/chunk/**"],
                ),
                column_shares=[1.0, 1.0, 1.0],
            ),
            rrb.TextDocumentView(
                name="Episode metadata",
                origin=METADATA_PATH,
                contents=[METADATA_PATH],
            ),
            row_shares=[3.0, 1.5, 1.0],
        ),
        rrb.TimePanel(timeline=TIMELINE, fps=fps, expanded=True),
        auto_layout=False,
        auto_views=False,
    )


def _write_episode_recording(
    episode: DPPOEpisode,
    *,
    output_path: Path,
    fps: float,
    policy_label: str,
) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    _validate_episode(episode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.incomplete")
    temporary.unlink(missing_ok=True)
    recording = rr.RecordingStream(
        APP_ID,
        recording_id=f"seed_{episode.policy_seed}_episode_{episode.episode_id}",
    )
    recording.save(temporary)
    try:
        recording.send_recording_name(
            f"DPPO | seed={episode.policy_seed} | episode={episode.episode_id} "
            f"| env_seed={episode.environment_seed} | {'SUCCESS' if episode.success else 'FAILURE'}"
        )
        rr.send_blueprint(
            _build_blueprint(rrb, fps=fps),
            make_active=True,
            make_default=True,
            recording=recording,
        )
        metadata = "\n".join(
            (
                f"task_suite: {TASK_SUITE}",
                f"task: {TASK_NAME}",
                f"policy: {policy_label}",
                f"policy_seed: {episode.policy_seed}",
                f"env_index: {episode.env_index}",
                f"environment_seed: {episode.environment_seed}",
                f"steps: {episode.steps}",
                f"action_horizon: {episode.action_chunks.shape[1]}",
                f"act_steps: {episode.act_steps}",
                f"return_total: {episode.return_total:.6f}",
                f"episode_best_reward: {episode.episode_best_reward:.6f}",
                f"success: {episode.success}",
                f"first_success_step: {episode.first_success_step}",
                "camera: agentview_image (the sole DPPO image input)",
                "state: normalized robot0_eef_pos + robot0_eef_quat + robot0_gripper_qpos",
                "3D trajectory: end-effector position; object pose is not a policy observation",
                f"timeline_fps: {fps:.6f} (one frame per {episode.act_steps} low-level actions)",
                f"control_frequency_hz: {DEFAULT_CONTROL_FPS:.6f}",
            )
        )
        recording.log(
            METADATA_PATH,
            rr.TextDocument(metadata, media_type="text/markdown"),
            static=True,
        )
        recording.log(
            f"{SCENE_ROOT}/complete_eef_path",
            rr.LineStrips3D([episode.eef_path], colors=[(0, 190, 210)], radii=0.004),
            static=True,
        )
        recording.log(
            f"{SCENE_ROOT}/origin",
            rr.Points3D([[0.0, 0.0, 0.0]], colors=[(150, 150, 150)], radii=0.006),
            static=True,
        )
        cumulative = 0.0
        for step in range(episode.steps):
            recording.set_time(TIMELINE, sequence=step)
            recording.log(IMAGE_PATH, rr.Image(episode.frames[step]).compress())
            eef = episode.eef_path[step]
            recording.log(
                f"{SCENE_ROOT}/current_eef",
                rr.Points3D([eef], colors=[(0, 220, 255)], radii=0.010, labels=["eef"]),
            )
            for state_index, name in enumerate(STATE_NAMES):
                recording.log(
                    f"{SIGNALS_ROOT}/state/{name}",
                    rr.Scalars(float(episode.state_normalized[step, state_index])),
                )
            first_action = episode.action_chunks[step, 0]
            for action_index, name in enumerate(ACTION_NAMES):
                recording.log(
                    f"{SIGNALS_ROOT}/action/first/{name}",
                    rr.Scalars(float(first_action[action_index])),
                )
            for horizon_index in range(episode.action_chunks.shape[1]):
                for action_index, name in enumerate(ACTION_NAMES):
                    recording.log(
                        f"{SIGNALS_ROOT}/action/chunk/{horizon_index:02d}/{name}",
                        rr.Scalars(float(episode.action_chunks[step, horizon_index, action_index])),
                    )
            reward = float(episode.rewards[step])
            cumulative += reward
            recording.log(f"{SIGNALS_ROOT}/reward", rr.Scalars(reward))
            recording.log(
                f"{SIGNALS_ROOT}/success_score",
                rr.Scalars(reward / float(episode.act_steps)),
            )
            recording.log(f"{SIGNALS_ROOT}/cumulative_reward", rr.Scalars(cumulative))
            recording.log(
                f"{SIGNALS_ROOT}/terminated",
                rr.Scalars(float(episode.terminated[step])),
            )
            recording.log(
                f"{SIGNALS_ROOT}/truncated",
                rr.Scalars(float(episode.truncated[step])),
            )
    finally:
        recording.flush()
        recording.disconnect()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Rerun recording was not written: {temporary}")
    temporary.replace(output_path)


def _write_episode_video(episode: DPPOEpisode, output_path: Path, *, fps: float) -> None:
    from PIL import Image

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--write-video requires ffmpeg on PATH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robomimic-dppo-video-") as frame_dir:
        frame_root = Path(frame_dir)
        for index, frame in enumerate(episode.frames):
            Image.fromarray(frame).save(frame_root / f"frame_{index:06d}.png")
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_root / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"ffmpeg video encoding failed: {detail}")


def _merge_recordings(paths: Sequence[Path], output_path: Path) -> Path | None:
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        print("WARN: rerun CLI is unavailable; individual .rrd files were kept", file=sys.stderr)
        return None
    if not paths:
        raise ValueError("cannot merge an empty recording list")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_merge = output_path.with_name(f".{output_path.stem}.merged.rrd")
    temporary_compact = output_path.with_name(f".{output_path.stem}.compact.rrd")
    for path in (temporary_merge, temporary_compact, output_path):
        path.unlink(missing_ok=True)

    def run(arguments: Sequence[str], operation: str) -> None:
        completed = subprocess.run(
            [rerun_cli, "rrd", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Rerun {operation} failed: {detail}")

    try:
        run(["merge", "-o", str(temporary_merge), *(str(path) for path in paths)], "merge")
        run(["compact", "-o", str(temporary_compact), str(temporary_merge)], "compact")
        run(["verify", str(temporary_compact)], "verify")
        temporary_compact.replace(output_path)
    finally:
        temporary_merge.unlink(missing_ok=True)
        temporary_compact.unlink(missing_ok=True)
    return output_path


def visualize(
    *,
    raw_root: Path,
    output_dir: Path,
    max_episodes: int | None = None,
    fps: float = DEFAULT_FPS,
    write_video: bool = False,
    open_viewer: bool = False,
) -> dict[str, Any]:
    """Convert recorded DPPO episodes into Rerun/MP4 artifacts."""

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    episodes = _read_recorded_episodes(raw_root)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError("no episodes selected for visualization")
    policy_label = "unknown"
    raw_manifest_path = raw_root / "manifest.json"
    if raw_manifest_path.is_file():
        raw_manifest = json.loads(raw_manifest_path.read_text())
        policy_label = str(raw_manifest.get("policy", policy_label))
    _prepare_empty_directory(output_dir)
    recordings: list[dict[str, Any]] = []
    recording_paths: list[Path] = []
    for source_path, episode in episodes:
        result_label = "success" if episode.success else "failure"
        stem = f"seed_{episode.policy_seed}_episode_{episode.episode_id:04d}_{result_label}"
        recording_path = output_dir / f"{stem}.rrd"
        _write_episode_recording(
            episode,
            output_path=recording_path,
            fps=fps,
            policy_label=policy_label,
        )
        recording_paths.append(recording_path)
        video_path = None
        if write_video:
            video_path = output_dir / f"{stem}.mp4"
            _write_episode_video(episode, video_path, fps=fps)
        recordings.append(
            {
                **_episode_summary(episode),
                "source": _portable(source_path),
                "recording": _portable(recording_path),
                "video": None if video_path is None else _portable(video_path),
            }
        )
    merged_path = _merge_recordings(
        recording_paths,
        output_dir / "robomimic_dppo_visualization.rrd",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "robomimic_dppo_rerun_visualization",
        "raw_root": _portable(raw_root),
        "fps": fps,
        "control_frequency_hz": DEFAULT_CONTROL_FPS,
        "policy_frequency_hz": DEFAULT_CONTROL_FPS / episodes[0][1].act_steps,
        "frame_semantics": (
            "one pre-action agentview frame per policy decision; "
            "each decision applies four low-level actions"
        ),
        "selected_episode_count": len(recordings),
        "merged_recording": None if merged_path is None else _portable(merged_path),
        "viewer": {
            "launcher": "robot-rerun or rerun",
            "application_id": APP_ID,
            "policy": policy_label,
            "camera": "agentview_image",
            "state": "normalized 9D robot state",
            "geometry": "end-effector position trajectory only",
        },
        "episodes": recordings,
    }
    _write_json(output_dir / "visualization_manifest.json", manifest)
    if open_viewer:
        viewer = shutil.which("robot-rerun") or shutil.which("rerun")
        target = merged_path or recording_paths[0]
        if viewer is None:
            print(f"WARN: no Rerun viewer found; open manually: {target}", file=sys.stderr)
        else:
            subprocess.Popen(
                [viewer, str(target)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    return manifest


def audit(
    *,
    eval_root: Path,
    seeds: Sequence[int],
    episodes_per_seed: int = DEFAULT_EPISODES_PER_SEED,
) -> dict[str, Any]:
    """Audit expected and actually materialized official evaluation episodes."""

    expected_total = len(seeds) * episodes_per_seed
    rows: list[dict[str, Any]] = []
    missing_results: list[int] = []
    for seed in seeds:
        result_path = eval_root / f"seed_{seed}" / "result.npz"
        if not result_path.is_file():
            missing_results.append(seed)
            continue
        with np.load(result_path, allow_pickle=False) as result:
            rows.append(
                {
                    "seed": seed,
                    "result": _portable(result_path),
                    "episodes": int(result["num_episode"]),
                    "success_rate": float(result["eval_success_rate"]),
                }
            )
    actual_total = sum(int(row["episodes"]) for row in rows)
    payload = {
        "task_suite": TASK_SUITE,
        "task": TASK_NAME,
        "evaluation_is_static_dataset": False,
        "evaluation_definition": "one rollout per vector environment using policy_seed + env_index",
        "expected": {
            "seeds": list(seeds),
            "episodes_per_seed": episodes_per_seed,
            "total_episodes": expected_total,
        },
        "materialized_result_files": len(rows),
        "materialized_episode_total": actual_total,
        "missing_result_seeds": missing_results,
        "per_seed": rows,
        "contract_ok": not missing_results
        and all(int(row["episodes"]) == episodes_per_seed for row in rows),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="audit official result.npz episode counts")
    audit_parser.add_argument("--eval-root", type=Path, required=True)
    audit_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    audit_parser.add_argument("--episodes-per-seed", type=int, default=DEFAULT_EPISODES_PER_SEED)

    record_parser = subparsers.add_parser(
        "record",
        help="rerun DPPO evaluation and store raw selected episodes",
    )
    record_parser.add_argument("--dppo-root", type=Path, default=Path("/opt/robot-practice/dppo"))
    checkpoint_group = record_parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint-root",
        type=Path,
        help="per-seed root containing seed_<N>/checkpoint/state_<step>.pt",
    )
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        help="one shared checkpoint reused for every policy seed",
    )
    record_parser.add_argument("--normalization", type=Path, required=True)
    record_parser.add_argument("--output-root", type=Path, required=True)
    record_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    record_parser.add_argument("--gpu", default="0")
    record_parser.add_argument("--env-count", type=int, default=DEFAULT_ENV_COUNT)
    record_parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    record_parser.add_argument("--act-steps", type=int, default=DEFAULT_ACT_STEPS)
    record_parser.add_argument("--ddim-steps", type=int, default=DEFAULT_DDIM_STEPS)
    record_parser.add_argument("--ft-denoising-steps", type=int, default=DEFAULT_FT_DENOISING_STEPS)
    record_parser.add_argument("--checkpoint-step", type=int, default=None)
    record_parser.add_argument(
        "--policy-label",
        default=None,
        help="label stored in the raw manifest and Rerun metadata",
    )
    record_parser.add_argument("--max-record-episodes", type=int, default=None)
    record_parser.add_argument(
        "--env-indices",
        default=None,
        help="comma-separated vector env indices to record",
    )
    record_parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)

    visualize_parser = subparsers.add_parser("visualize", help="convert raw DPPO episodes to Rerun")
    visualize_parser.add_argument("--raw-root", type=Path, required=True)
    visualize_parser.add_argument("--output-dir", type=Path, required=True)
    visualize_parser.add_argument("--max-episodes", type=int, default=None)
    visualize_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    visualize_parser.add_argument("--write-video", action="store_true")
    visualize_parser.add_argument("--open-viewer", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "audit":
        audit(
            eval_root=arguments.eval_root.expanduser().resolve(),
            seeds=_parse_int_csv(arguments.seeds, name="--seeds"),
            episodes_per_seed=arguments.episodes_per_seed,
        )
        return 0
    if arguments.command == "record":
        seeds = _parse_int_csv(arguments.seeds, name="--seeds")
        env_indices = (
            None
            if arguments.env_indices is None
            else _parse_int_csv(arguments.env_indices, name="--env-indices")
        )
        checkpoint_root = (
            None
            if arguments.checkpoint_root is None
            else arguments.checkpoint_root.expanduser().resolve()
        )
        checkpoint = (
            None
            if arguments.checkpoint is None
            else arguments.checkpoint.expanduser().resolve()
        )
        manifest = record(
            dppo_root=arguments.dppo_root.expanduser().resolve(),
            checkpoint_root=checkpoint_root,
            checkpoint=checkpoint,
            normalization_path=arguments.normalization.expanduser().resolve(),
            output_root=arguments.output_root.expanduser().resolve(),
            seeds=seeds,
            env_count=arguments.env_count,
            eval_steps=arguments.eval_steps,
            act_steps=arguments.act_steps,
            ddim_steps=arguments.ddim_steps,
            ft_denoising_steps=arguments.ft_denoising_steps,
            checkpoint_step=arguments.checkpoint_step,
            policy=arguments.policy_label,
            max_record_episodes=arguments.max_record_episodes,
            env_indices=env_indices,
            success_threshold=arguments.success_threshold,
            gpu=arguments.gpu,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if arguments.command == "visualize":
        manifest = visualize(
            raw_root=arguments.raw_root.expanduser().resolve(),
            output_dir=arguments.output_dir.expanduser().resolve(),
            max_episodes=arguments.max_episodes,
            fps=arguments.fps,
            write_video=arguments.write_video,
            open_viewer=arguments.open_viewer,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
