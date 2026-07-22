"""Closed-loop ACT evaluation on one explicitly selected LIBERO task."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from tqdm import trange

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video

from robot_practice.evaluation.common import (
    ModelQueryTimer,
    build_run_config,
    create_output_directory,
    launch_rerun_viewer,
    load_act_checkpoint,
    log_terminal_metrics,
    mark_run_complete,
    release_output_directory_lock,
    timed_call,
    validate_complete_checkpoint,
    write_json_atomic,
    write_parquet_atomic,
)
from robot_practice.evaluation.libero_checkpoint import (
    assess_checkpoint_task,
    log_checkpoint_task_assessment,
)
from robot_practice.evaluation.libero_tasks import (
    SUPPORTED_SUITES,
    LiberoTaskIdentity,
    log_task_identity,
    resolve_libero_task,
)
from robot_practice.evaluation.metrics import compute_closed_loop_metrics
from robot_practice.evaluation.rerun_visualization import (
    finish_recording,
    image_to_hwc_uint8,
    initialize_recording,
    log_libero_step,
    log_metric_summary,
)

LOGGER = logging.getLogger(__name__)

CONTROL_FREQUENCY_HZ = 10.0
RAW_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_KEYS = ("observation.images.image", "observation.images.image2")


@dataclass(frozen=True)
class LiberoEvaluationConfig:
    policy_path: Path
    suite: str
    task_id: int
    n_episodes: int = 20
    num_visualized_episodes: int = 1
    perturbation: str = "none"
    seed: int = 1000
    output_dir: Path | None = None
    overwrite: bool = False
    max_steps: int | None = None
    open_viewer: bool = True


@dataclass(frozen=True)
class LiberoEvaluationResult:
    output_dir: Path
    metrics: dict[str, Any]
    viewer_opened: bool


def validate_config(
    config: LiberoEvaluationConfig,
    task: LiberoTaskIdentity | None = None,
) -> LiberoTaskIdentity:
    task = task or resolve_libero_task(config.suite, config.task_id)
    if config.n_episodes <= 0:
        raise ValueError("--n-episodes must be positive")
    if not 0 <= config.num_visualized_episodes <= config.n_episodes:
        raise ValueError(
            "--num-visualized-episodes must be between zero and --n-episodes"
        )
    if config.perturbation != "none":
        raise ValueError("Only --perturbation=none is implemented in the first version")
    if config.seed < 0:
        raise ValueError("--seed must be non-negative")
    if config.overwrite and config.output_dir is None:
        raise ValueError("--overwrite requires an explicit --output-dir")
    if config.max_steps is not None and not 1 <= config.max_steps <= task.episode_length:
        raise ValueError(
            f"--max-steps must be between 1 and {task.episode_length} for {task.suite}"
        )
    return task


def make_libero_config(task: LiberoTaskIdentity) -> LiberoEnv:
    return LiberoEnv(
        task=task.suite,
        task_ids=[task.task_id],
        fps=int(CONTROL_FREQUENCY_HZ),
        episode_length=task.episode_length,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        camera_name=",".join(RAW_CAMERAS),
        init_states=True,
        observation_height=256,
        observation_width=256,
        control_mode="relative",
    )


def extract_terminal_success(info: dict[str, Any], *, terminal: bool) -> bool:
    """Read only vectorized terminal ``final_info``; never infer success from reward."""

    if not terminal:
        return False
    mask = info.get("_final_info")
    if mask is None or np.asarray(mask, dtype=bool).shape != (1,):
        raise RuntimeError("Terminal LIBERO step is missing vectorized _final_info mask")
    if not bool(np.asarray(mask, dtype=bool)[0]):
        raise RuntimeError("Terminal LIBERO step has a false _final_info mask")
    final_info = info.get("final_info")
    if not isinstance(final_info, dict) or "is_success" not in final_info:
        raise RuntimeError("Terminal LIBERO step is missing final_info.is_success")
    success_values = np.asarray(final_info["is_success"], dtype=bool)
    if success_values.shape != (1,):
        raise RuntimeError(
            f"Expected one final_info.is_success value, got {success_values.shape}"
        )
    nested_mask = final_info.get("_is_success")
    if nested_mask is not None and not bool(np.asarray(nested_mask, dtype=bool)[0]):
        raise RuntimeError("Terminal LIBERO final_info has a false _is_success mask")
    return bool(success_values[0])


def _rollout_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("suite", pa.string(), nullable=False),
            pa.field("libero_task_id", pa.int32(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("init_state_id", pa.int64(), nullable=False),
            pa.field("step_index", pa.int32(), nullable=False),
            pa.field("global_step", pa.int64(), nullable=False),
            pa.field("timestamp_s", pa.float64(), nullable=False),
            pa.field("executed_action", pa.list_(pa.float32(), 7), nullable=False),
            pa.field("reward", pa.float32(), nullable=False),
            pa.field("terminated", pa.bool_(), nullable=False),
            pa.field("truncated", pa.bool_(), nullable=False),
            pa.field("done_after_step", pa.bool_(), nullable=False),
            pa.field("success_this_step", pa.bool_(), nullable=False),
            pa.field("success_so_far", pa.bool_(), nullable=False),
            pa.field("termination_reason", pa.string(), nullable=True),
            pa.field("policy_step_latency_ms", pa.float64(), nullable=False),
            pa.field("model_query_latency_ms", pa.float64(), nullable=True),
            pa.field("visualized", pa.bool_(), nullable=False),
        ],
        metadata={
            b"schema_version": b"2",
            b"action_dimension": b"7",
            b"control_frequency_hz": b"10.0",
        },
    )


def _autocast_context(device: str, use_amp: bool) -> Any:
    resolved = torch.device(device)
    if use_amp:
        return torch.autocast(device_type=resolved.type)
    return nullcontext()


def _extract_policy_images(observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    images: dict[str, torch.Tensor] = {}
    for key in CAMERA_KEYS:
        if key not in observation:
            raise KeyError(f"Processed LIBERO observation is missing camera {key}")
        image = observation[key]
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"LIBERO camera {key} must have shape [1,C,H,W], got {type(image)}")
        images[key] = image[0].detach().cpu().clone()
    return images


def _make_eval_info(
    metrics: dict[str, Any],
    *,
    task: LiberoTaskIdentity,
    formal: bool,
    elapsed_s: float,
) -> dict[str, Any]:
    per_episode = [
        {
            "episode_ix": item["episode_index"],
            "seed": item["seed"],
            "init_state_id": item["init_state_id"],
            "num_steps": item["num_steps"],
            "sum_reward": item["sum_reward"],
            "max_reward": item["max_reward"],
            "success": item["success"],
            "termination_reason": item["termination_reason"],
        }
        for item in metrics["per_episode"]
    ]
    aggregate = metrics["aggregate"]
    return {
        "schema_version": 2,
        "suite": task.suite,
        "task_id": task.task_id,
        "task_name": task.task_name,
        "language": task.language,
        "is_formal": formal,
        "per_episode": per_episode,
        "aggregated": {
            "n_episodes": aggregate["num_episodes"],
            "n_successes": aggregate["num_successes"],
            "avg_sum_reward": aggregate["avg_sum_reward"],
            "avg_max_reward": aggregate["avg_max_reward"],
            "pc_success": (
                aggregate["success_rate"] * 100
                if aggregate["success_rate"] is not None
                else None
            ),
            "eval_s": elapsed_s,
            "eval_ep_s": elapsed_s / aggregate["num_episodes"],
        },
    }


def _set_init_state_id(env: gym.vector.VectorEnv, init_state_id: int) -> None:
    if env.num_envs != 1 or not hasattr(env, "envs"):
        raise RuntimeError(
            "Controlled LIBERO evaluation requires one SyncVectorEnv sub-environment"
        )
    base_env = env.envs[0]
    if not hasattr(base_env, "init_state_id"):
        raise RuntimeError("LeRobot LIBERO environment does not expose init_state_id")
    base_env.init_state_id = init_state_id


def evaluate_libero(
    config: LiberoEvaluationConfig,
    *,
    max_steps: int | None = None,
) -> LiberoEvaluationResult:
    """Run serial single-task rollouts and persist simulator-grounded results."""

    if max_steps is not None:
        if config.max_steps is not None and config.max_steps != max_steps:
            raise ValueError("max_steps was supplied both in config and function arguments")
        config = LiberoEvaluationConfig(**{**asdict(config), "max_steps": max_steps})
    task = resolve_libero_task(config.suite, config.task_id)
    log_task_identity(task, LOGGER)
    validate_config(config, task)
    manifest = validate_complete_checkpoint(config.policy_path)
    LOGGER.info("[Policy] checkpoint: %s", manifest.path)
    checkpoint_assessment = assess_checkpoint_task(manifest, task)
    log_checkpoint_task_assessment(checkpoint_assessment, LOGGER)
    env_config = make_libero_config(task)
    loaded = load_act_checkpoint(manifest, env_config=env_config)

    step_limit = config.max_steps or task.episode_length
    formal = step_limit == task.episode_length
    status = "complete" if formal else "smoke"
    output_dir = create_output_directory(
        policy_path=manifest.path,
        mode="libero",
        output_dir=config.output_dir,
        suite=task.suite,
        task_id=task.task_id,
        overwrite=config.overwrite,
    )
    LOGGER.info("[Evaluation] mode: closed-loop LIBERO rollout")
    LOGGER.info("[Evaluation] episodes: %d", config.n_episodes)
    LOGGER.info(
        "[Evaluation] visualized episodes: %d", config.num_visualized_episodes
    )
    LOGGER.info("[Evaluation] episode step limit: %d", step_limit)
    LOGGER.info("[Evaluation] chunk_size: %d", loaded.config.chunk_size)
    LOGGER.info("[Evaluation] n_action_steps: %d", loaded.config.n_action_steps)
    LOGGER.info("[Evaluation] perturbation: %s", config.perturbation)
    LOGGER.info("[Evaluation] status: %s", status)
    LOGGER.info("[Output] directory: %s", output_dir)
    input_config = {
        "task": task.to_dict(),
        "checkpoint_task_assessment": checkpoint_assessment.to_dict(),
        "cameras": list(RAW_CAMERAS),
        "obs_type": "pixels_agent_pos",
        "control_mode": "relative",
        "observation_size": [256, 256],
        "episode_length": task.episode_length,
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "seeds": [config.seed + episode for episode in range(config.n_episodes)],
        "init_state_ids": list(range(config.n_episodes)),
    }
    run_config = build_run_config(
        status="incomplete",
        mode="libero",
        loaded=loaded,
        input_config=input_config,
        arguments=asdict(config),
    )
    try:
        write_json_atomic(output_dir / "run_config.json", run_config)
        envs = make_env(env_config, n_envs=1, use_async_envs=False)
    except BaseException:
        release_output_directory_lock(output_dir)
        raise
    try:
        env = envs[task.suite][task.task_id]
        if env.autoreset_mode is not gym.vector.AutoresetMode.NEXT_STEP:
            raise RuntimeError(
                f"Expected Gymnasium NEXT_STEP autoreset, got {env.autoreset_mode}"
            )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_config,
            policy_cfg=loaded.config,
        )
    except BaseException:
        close_envs(envs)
        release_output_directory_lock(output_dir)
        raise

    recording_path = output_dir / "rollout.rrd"
    recording_initialized = False
    rollout_rows: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    global_step = 0
    evaluation_started = time.perf_counter()

    try:
        initialize_recording(
            output_path=recording_path,
            mode="libero",
            camera_keys=CAMERA_KEYS,
            fps=CONTROL_FREQUENCY_HZ,
        )
        recording_initialized = True
        with ModelQueryTimer(loaded.policy, loaded.config.device) as model_timer:
            for episode_index in trange(
                config.n_episodes,
                desc=f"{task.suite} task-{task.task_id} rollout",
                unit="episode",
            ):
                episode_seed = config.seed + episode_index
                init_state_id = episode_index
                _set_init_state_id(env, init_state_id)
                observation, _ = env.reset(seed=[episode_seed])
                loaded.policy.reset()
                actions: list[np.ndarray] = []
                rewards: list[float] = []
                policy_latencies: list[float] = []
                model_latencies: list[float] = []
                success_seen = False
                termination_reason: str | None = None
                visualized = episode_index < config.num_visualized_episodes
                video_frames: list[np.ndarray] = []

                for step_index in range(step_limit):
                    policy_observation = preprocess_observation(observation)
                    policy_observation = add_envs_task(env, policy_observation)
                    policy_observation = env_preprocessor(policy_observation)
                    images = _extract_policy_images(policy_observation)
                    processed_observation = loaded.preprocessor(policy_observation)

                    model_timer.begin_step()
                    with torch.inference_mode(), _autocast_context(
                        str(loaded.config.device), loaded.config.use_amp
                    ):
                        action, policy_latency_ms = timed_call(
                            loaded.policy.select_action,
                            processed_observation,
                            device=loaded.config.device,
                        )
                    model_latency_ms = model_timer.current_latency_ms
                    action = loaded.postprocessor(action)
                    action_transition = env_postprocessor({ACTION: action})
                    action_numpy = action_transition[ACTION].detach().cpu().float().numpy()
                    if action_numpy.shape != (1, 7):
                        raise ValueError(
                            "Postprocessed LIBERO action must have shape (1, 7), "
                            f"got {action_numpy.shape}"
                        )

                    next_observation, reward, terminated, truncated, info = env.step(
                        action_numpy
                    )
                    raw_terminated = bool(np.asarray(terminated, dtype=bool)[0])
                    raw_truncated = bool(np.asarray(truncated, dtype=bool)[0])
                    terminal = raw_terminated or raw_truncated
                    success_this_step = extract_terminal_success(info, terminal=terminal)
                    success_seen = success_seen or success_this_step
                    synthetic_time_limit = not terminal and step_index + 1 == step_limit
                    done_after_step = terminal or synthetic_time_limit
                    recorded_truncated = raw_truncated or synthetic_time_limit
                    if done_after_step:
                        if success_seen:
                            termination_reason = "success"
                        elif synthetic_time_limit:
                            termination_reason = "time_limit"
                        else:
                            termination_reason = "environment_done"

                    executed_action = action_numpy[0].copy()
                    reward_value = float(np.asarray(reward, dtype=np.float64)[0])
                    actions.append(executed_action)
                    rewards.append(reward_value)
                    policy_latencies.append(policy_latency_ms)
                    if model_latency_ms is not None:
                        model_latencies.append(model_latency_ms)

                    if visualized:
                        log_libero_step(
                            rr,
                            global_step=global_step,
                            episode_index=episode_index,
                            step_index=step_index,
                            action=executed_action,
                            reward=reward_value,
                            success_so_far=success_seen,
                            done_after_step=done_after_step,
                            policy_step_latency_ms=policy_latency_ms,
                            model_query_latency_ms=model_latency_ms,
                            images=images,
                        )
                        video_frames.append(image_to_hwc_uint8(images[CAMERA_KEYS[0]]))

                    rollout_rows.append(
                        {
                            "suite": task.suite,
                            "libero_task_id": task.task_id,
                            "episode_index": episode_index,
                            "seed": episode_seed,
                            "init_state_id": init_state_id,
                            "step_index": step_index,
                            "global_step": global_step,
                            "timestamp_s": step_index / CONTROL_FREQUENCY_HZ,
                            "executed_action": executed_action.tolist(),
                            "reward": reward_value,
                            "terminated": raw_terminated,
                            "truncated": recorded_truncated,
                            "done_after_step": done_after_step,
                            "success_this_step": success_this_step,
                            "success_so_far": success_seen,
                            "termination_reason": termination_reason,
                            "policy_step_latency_ms": policy_latency_ms,
                            "model_query_latency_ms": model_latency_ms,
                            "visualized": visualized,
                        }
                    )
                    global_step += 1
                    if done_after_step:
                        break
                    observation = next_observation

                if termination_reason is None:
                    raise RuntimeError("LIBERO rollout exited without a termination reason")
                episode_results.append(
                    {
                        "episode_index": episode_index,
                        "seed": episode_seed,
                        "init_state_id": init_state_id,
                        "actions": np.stack(actions),
                        "rewards": np.asarray(rewards, dtype=np.float64),
                        "success": success_seen,
                        "termination_reason": termination_reason,
                        "policy_step_latencies_ms": policy_latencies,
                        "model_query_latencies_ms": model_latencies,
                    }
                )
                if visualized:
                    videos_dir = output_dir / "videos"
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    write_video(
                        str(videos_dir / f"episode_{episode_index:03d}.mp4"),
                        np.stack(video_frames),
                        CONTROL_FREQUENCY_HZ,
                    )

        elapsed_s = time.perf_counter() - evaluation_started
        metrics = compute_closed_loop_metrics(
            episode_results,
            formal=formal,
            status=status,
        )
        metrics["schema_version"] = 2
        metrics["task"] = task.to_dict()
        metrics["checkpoint_task_assessment"] = checkpoint_assessment.to_dict()
        metrics["perturbation"] = config.perturbation
        metrics["visualized_episodes"] = list(
            range(config.num_visualized_episodes)
        )
        eval_info = _make_eval_info(
            metrics,
            task=task,
            formal=formal,
            elapsed_s=elapsed_s,
        )
        log_metric_summary(rr, metrics)
        write_json_atomic(output_dir / "metrics.json", metrics)
        write_json_atomic(output_dir / "eval_info.json", eval_info)
        table = pa.Table.from_pylist(rollout_rows, schema=_rollout_schema())
        write_parquet_atomic(output_dir / "rollouts.parquet", table)

        finish_recording()
        recording_initialized = False
        if not recording_path.is_file() or recording_path.stat().st_size == 0:
            raise RuntimeError(f"Rerun recording was not written: {recording_path}")
        run_config["status"] = status
        write_json_atomic(output_dir / "run_config.json", run_config)
        mark_run_complete(output_dir)
    finally:
        if recording_initialized:
            finish_recording()
        close_envs(envs)
        release_output_directory_lock(output_dir)

    viewer_opened = launch_rerun_viewer(recording_path, enabled=config.open_viewer)
    log_terminal_metrics(metrics, LOGGER)
    LOGGER.info("[Output] metrics: %s", output_dir / "metrics.json")
    LOGGER.info("[Output] eval info: %s", output_dir / "eval_info.json")
    LOGGER.info("[Output] rollouts: %s", output_dir / "rollouts.parquet")
    LOGGER.info("[Output] Rerun recording: %s", recording_path)
    LOGGER.info("[Output] run config: %s", output_dir / "run_config.json")
    if config.num_visualized_episodes:
        LOGGER.info("[Output] videos: %s", output_dir / "videos")
    LOGGER.info("[Output] viewer opened: %s", viewer_opened)
    return LiberoEvaluationResult(
        output_dir=output_dir,
        metrics=metrics,
        viewer_opened=viewer_opened,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path",
        required=True,
        type=Path,
        help="Complete local LeRobot pretrained_model directory",
    )
    parser.add_argument(
        "--suite",
        required=True,
        help=f"LIBERO suite ({', '.join(SUPPORTED_SUITES)})",
    )
    parser.add_argument(
        "--task-id",
        required=True,
        type=int,
        help="Zero-based task ID within --suite",
    )
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--num-visualized-episodes", type=int, default=1)
    parser.add_argument("--perturbation", default="none")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a recognized existing --output-dir (requires --output-dir)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Debug-only per-episode bound (upper limit depends on --suite)",
    )
    parser.add_argument(
        "--no-open-viewer",
        action="store_false",
        dest="open_viewer",
        help="Save rollout.rrd without opening the viewer",
    )
    parser.set_defaults(open_viewer=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argument_parser().parse_args(argv)
    evaluate_libero(LiberoEvaluationConfig(**vars(args)))


if __name__ == "__main__":
    main()
