"""Closed-loop ACT evaluation on one explicitly selected LIBERO task."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.dataframe as rdf
import torch
from tqdm import trange

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video

from robot_practice.evaluation.common import (
    ActionPlanStep,
    ActionPlanTracker,
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
from robot_practice.evaluation.libero_kinematics import PandaSkeletonReconstructor
from robot_practice.evaluation.libero_tasks import (
    SUPPORTED_SUITES,
    LiberoTaskIdentity,
    log_task_identity,
    resolve_libero_task,
)
from robot_practice.evaluation.metrics import compute_closed_loop_metrics
from robot_practice.evaluation.metrics_report import write_metrics_report
from robot_practice.evaluation.rerun_visualization import (
    LiberoTrajectoryHistory,
    image_to_hwc_uint8,
    initialize_libero_recording_stream,
    log_libero_step,
    log_metric_summary,
    stage_merged_recordings,
)

LOGGER = logging.getLogger(__name__)

CONTROL_FREQUENCY_HZ = 10.0
RAW_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_KEYS = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
PERTURBATION_MODES = ("none", "proprio_gaussian")
PROPRIO_NOISE_SPACE = "checkpoint_normalized_observation.state"
_PROPRIO_NOISE_DOMAIN = 0xA17C0DE


@dataclass(frozen=True)
class LiberoEvaluationConfig:
    policy_path: Path
    suite: str
    task_id: int
    n_episodes: int = 20
    num_visualized_episodes: int = 1
    n_action_steps: int | None = None
    temporal_ensemble_coeff: float | None = None
    perturbation: str = "none"
    proprio_noise_std: float = 0.01
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


@dataclass
class LiberoEpisodeRecording:
    episode_index: int | None
    seed: int | None
    init_state_id: int | None
    path: Path
    stream: rr.RecordingStream
    trajectory_history: LiberoTrajectoryHistory = field(
        default_factory=LiberoTrajectoryHistory
    )
    previous_step_index: int | None = None
    final_name: str | None = None
    final_name_sent: bool = False
    flush_attempted: bool = False
    disconnect_attempted: bool = False

    def accept_step_index(self, step_index: int) -> None:
        if self.previous_step_index is None:
            if step_index != 0:
                raise ValueError(
                    f"LIBERO episode {self.episode_index} must start at step zero, "
                    f"got {step_index}"
                )
        elif step_index <= self.previous_step_index:
            raise ValueError(
                f"LIBERO episode {self.episode_index} step indices must be "
                f"monotonically increasing, got {self.previous_step_index} then "
                f"{step_index}"
            )
        self.previous_step_index = step_index

    def send_final_name(self, name: str) -> None:
        if self.final_name_sent:
            raise RuntimeError(
                f"LIBERO episode {self.episode_index} Recording name was already sent"
            )
        self.stream.send_recording_name(name)
        self.final_name = name
        self.final_name_sent = True


def flush_and_disconnect_recordings(
    recordings: Sequence[LiberoEpisodeRecording],
) -> None:
    """Attempt every unattempted flush before every unattempted disconnect."""

    errors: list[BaseException] = []
    for recording in recordings:
        if recording.flush_attempted:
            continue
        try:
            recording.stream.flush()
        except BaseException as exc:
            errors.append(exc)
        finally:
            recording.flush_attempted = True
    for recording in recordings:
        if recording.disconnect_attempted:
            continue
        try:
            recording.stream.disconnect()
        except BaseException as exc:
            errors.append(exc)
        finally:
            recording.disconnect_attempted = True
    if errors:
        raise errors[0]


def libero_episode_recording_name(
    *,
    episode_index: int,
    seed: int,
    init_state_id: int,
    success: bool,
    termination_reason: str,
) -> str:
    outcome = "success" if success else f"failure:{termination_reason}"
    return (
        f"Episode {episode_index:03d} | seed={seed} | init={init_state_id} | {outcome}"
    )


LIBERO_SUMMARY_RECORDING_NAME = "LIBERO closed-loop evaluation summary"


def verify_libero_recording_archive(
    archive_path: Path,
    recordings: Sequence[LiberoEpisodeRecording],
) -> None:
    """Verify selector-visible names and episode isolation in a merged RRD."""

    archive = rdf.load_archive(archive_path)
    if archive.num_recordings() != len(recordings):
        raise RuntimeError(
            "Merged LIBERO RRD contains an unexpected number of Recordings: "
            f"expected {len(recordings)}, got {archive.num_recordings()}"
        )
    recording_ids = [recording.recording_id() for recording in archive.all_recordings()]
    if len(set(recording_ids)) != len(recording_ids):
        raise RuntimeError("Merged LIBERO RRD contains duplicate Recording IDs")
    expected_by_name = {
        recording.final_name: recording
        for recording in recordings
        if recording.final_name is not None
    }
    if len(expected_by_name) != len(recordings):
        raise RuntimeError("Every LIBERO Recording must have one unique final name")

    observed_names: set[str] = set()
    for recording in archive.all_recordings():
        properties = (
            recording.view(index=None, contents="/__properties/**")
            .select_static()
            .read_all()
            .to_pydict()
        )
        name_values = properties.get("property:RecordingInfo:name")
        if name_values is None or len(name_values) != 1 or len(name_values[0]) != 1:
            raise RuntimeError(
                f"LIBERO Recording {recording.recording_id()} has no exact final name"
            )
        name = str(name_values[0][0])
        expected = expected_by_name.get(name)
        if expected is None:
            raise RuntimeError(f"Unexpected LIBERO Recording name: {name}")
        observed_names.add(name)
        if expected.episode_index is None:
            continue
        episode_table = (
            recording.view(
                index="global_step",
                contents="/rollout/context/episode_index",
            )
            .select()
            .read_all()
            .to_pydict()
        )
        timeline = episode_table.get("global_step", [])
        episode_values = episode_table.get(
            "/rollout/context/episode_index:Scalars:scalars",
            [],
        )
        if not timeline or timeline[0] != 0:
            raise RuntimeError(
                f"LIBERO Recording {name} timeline does not start at zero"
            )
        observed_episode_indices = {
            int(value[0]) for value in episode_values if value is not None and len(value)
        }
        if observed_episode_indices != {expected.episode_index}:
            raise RuntimeError(
                f"LIBERO Recording {name} contains episode indices "
                f"{sorted(observed_episode_indices)}"
            )
    if observed_names != set(expected_by_name):
        raise RuntimeError("Merged LIBERO RRD is missing expected Recording names")


@dataclass(frozen=True)
class PreparedLiberoObservation:
    policy_observation: dict[str, Any]
    images: dict[str, torch.Tensor]
    state: np.ndarray
    joint_positions: np.ndarray


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
    if config.n_action_steps is not None and config.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    if config.temporal_ensemble_coeff is not None:
        if not np.isfinite(config.temporal_ensemble_coeff):
            raise ValueError("--temporal-ensemble-coeff must be finite")
        if config.n_action_steps != 1:
            raise ValueError(
                "--temporal-ensemble-coeff requires explicit --n-action-steps=1"
            )
    if config.perturbation not in PERTURBATION_MODES:
        raise ValueError(
            f"--perturbation must be one of {', '.join(PERTURBATION_MODES)}"
        )
    if not np.isfinite(config.proprio_noise_std) or config.proprio_noise_std <= 0:
        raise ValueError("--proprio-noise-std must be finite and positive")
    if config.seed < 0:
        raise ValueError("--seed must be non-negative")
    if config.overwrite and config.output_dir is None:
        raise ValueError("--overwrite requires an explicit --output-dir")
    if config.max_steps is not None and not 1 <= config.max_steps <= task.episode_length:
        raise ValueError(
            f"--max-steps must be between 1 and {task.episode_length} for {task.suite}"
        )
    return task


def resolve_n_action_steps(
    *,
    checkpoint_n_action_steps: int,
    override: int | None,
    chunk_size: int,
    temporal_ensemble_coeff: float | None,
) -> int:
    """Resolve and validate an evaluation-only ACT execution horizon override."""

    effective = checkpoint_n_action_steps if override is None else override
    if effective <= 0:
        raise ValueError("Effective n_action_steps must be positive")
    if effective > chunk_size:
        raise ValueError(
            "--n-action-steps cannot exceed checkpoint chunk_size: "
            f"{effective} > {chunk_size}"
        )
    if temporal_ensemble_coeff is not None and effective != 1:
        raise ValueError(
            "--n-action-steps must equal 1 when checkpoint temporal ensembling is enabled"
        )
    return effective


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
            pa.field("observation_state", pa.list_(pa.float32(), 8), nullable=False),
            pa.field(
                "policy_state_normalized_clean",
                pa.list_(pa.float32(), 8),
                nullable=False,
            ),
            pa.field(
                "policy_state_normalized_input",
                pa.list_(pa.float32(), 8),
                nullable=False,
            ),
            pa.field(
                "next_observation_state",
                pa.list_(pa.float32(), 8),
                nullable=False,
            ),
            pa.field(
                "observation_joint_positions",
                pa.list_(pa.float32(), 7),
                nullable=False,
            ),
            pa.field(
                "next_observation_joint_positions",
                pa.list_(pa.float32(), 7),
                nullable=False,
            ),
            pa.field("executed_action", pa.list_(pa.float32(), 7), nullable=False),
            pa.field("model_queried", pa.bool_(), nullable=False),
            pa.field("plan_step_index", pa.int32(), nullable=True),
            pa.field(
                "predicted_chunk",
                pa.list_(pa.list_(pa.float32(), 7)),
                nullable=True,
            ),
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
            b"schema_version": b"4",
            b"action_dimension": b"7",
            b"state_dimension": b"8",
            b"arm_joint_dimension": b"7",
            b"control_frequency_hz": b"10.0",
        },
    )


def prepare_policy_input(
    processed_observation: dict[str, Any],
    *,
    perturbation: str,
    proprio_noise_std: float,
    seed: int,
    episode_index: int,
    step_index: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Apply deterministic noise only to the checkpoint-normalized policy state."""

    if perturbation not in PERTURBATION_MODES:
        raise ValueError(f"Unknown perturbation mode: {perturbation}")
    state = processed_observation.get(STATE_KEY)
    if not isinstance(state, torch.Tensor) or tuple(state.shape) != (1, 8):
        actual = type(state).__name__ if not isinstance(state, torch.Tensor) else state.shape
        raise ValueError(
            f"Processed {STATE_KEY} must be a tensor with shape (1, 8), got {actual}"
        )
    if not bool(torch.isfinite(state).all()):
        raise ValueError(f"Processed {STATE_KEY} contains non-finite values")

    clean = state.detach().cpu().float().numpy()[0].copy()
    if perturbation == "none":
        return processed_observation, clean, clean.copy()

    seed_sequence = np.random.SeedSequence(
        [seed, episode_index, step_index, _PROPRIO_NOISE_DOMAIN]
    )
    noise = np.random.default_rng(seed_sequence).normal(
        loc=0.0,
        scale=proprio_noise_std,
        size=tuple(state.shape),
    )
    noisy_state = state.clone()
    noisy_state.add_(torch.as_tensor(noise, dtype=state.dtype, device=state.device))
    policy_input = dict(processed_observation)
    policy_input[STATE_KEY] = noisy_state
    noisy = noisy_state.detach().cpu().float().numpy()[0].copy()
    return policy_input, clean, noisy


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


def _extract_policy_state(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get(STATE_KEY)
    if (
        not isinstance(state, torch.Tensor)
        or state.shape != (1, 8)
        or not bool(torch.isfinite(state).all())
    ):
        shape = getattr(state, "shape", None)
        raise ValueError(
            f"Processed LIBERO {STATE_KEY} must have finite shape [1,8], got {shape}"
        )
    return state[0].detach().cpu().float().numpy().copy()


def _extract_measured_joint_positions(observation: dict[str, Any]) -> np.ndarray:
    try:
        raw = observation["robot_state"]["joints"]["pos"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Raw LIBERO observation is missing robot_state.joints.pos"
        ) from exc
    joints = np.asarray(raw, dtype=np.float32)
    if joints.shape != (1, 7) or not np.isfinite(joints).all():
        raise ValueError(
            "Raw LIBERO robot_state.joints.pos must have finite shape [1,7], "
            f"got {joints.shape}"
        )
    return joints[0].copy()


def _prepare_libero_observation(
    observation: dict[str, Any],
    *,
    env: gym.vector.VectorEnv,
    env_preprocessor: Any,
) -> PreparedLiberoObservation:
    policy_observation = preprocess_observation(observation)
    policy_observation = add_envs_task(env, policy_observation)
    policy_observation = env_preprocessor(policy_observation)
    return PreparedLiberoObservation(
        policy_observation=policy_observation,
        images=_extract_policy_images(policy_observation),
        state=_extract_policy_state(policy_observation),
        joint_positions=_extract_measured_joint_positions(observation),
    )


def _postprocess_policy_action_tensor(
    actions_policy: torch.Tensor,
    *,
    checkpoint_postprocessor: Any,
    env_postprocessor: Any,
) -> torch.Tensor:
    if (
        not isinstance(actions_policy, torch.Tensor)
        or actions_policy.ndim not in (2, 3)
        or actions_policy.shape[-1] != 7
        or not bool(torch.isfinite(actions_policy).all())
    ):
        shape = getattr(actions_policy, "shape", None)
        raise ValueError(
            "Policy actions must have finite shape [batch,7] or "
            f"[batch,horizon,7], got {shape}"
        )
    original_shape = actions_policy.shape
    flattened = actions_policy.reshape(-1, 7)
    checkpoint_actions = checkpoint_postprocessor(flattened)
    transition = env_postprocessor({ACTION: checkpoint_actions})
    controller_actions = transition[ACTION]
    if (
        not isinstance(controller_actions, torch.Tensor)
        or controller_actions.shape != flattened.shape
        or not bool(torch.isfinite(controller_actions).all())
    ):
        shape = getattr(controller_actions, "shape", None)
        raise ValueError(
            "Postprocessed LIBERO actions must preserve finite flattened shape "
            f"{tuple(flattened.shape)}, got {shape}"
        )
    return controller_actions.reshape(original_shape)


def _controller_plan_partitions(
    plan_step: ActionPlanStep,
    *,
    queried_chunk_controller: np.ndarray | None,
    active_chunk_controller: np.ndarray | None,
    executed_action_controller: np.ndarray,
    n_action_steps: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
    if plan_step.model_queried:
        if queried_chunk_controller is None:
            raise RuntimeError("Model query is missing its controller-space chunk")
        active_chunk_controller = queried_chunk_controller
    if plan_step.plan_step_index is None:
        empty = np.empty((0, 7), dtype=np.float32)
        return None, empty, empty, queried_chunk_controller
    if active_chunk_controller is None:
        raise RuntimeError("Standard ACT plan has no controller-space active chunk")
    index = plan_step.plan_step_index
    if not np.allclose(
        active_chunk_controller[index],
        executed_action_controller,
        rtol=1e-5,
        atol=1e-6,
    ):
        max_error = float(
            np.max(
                np.abs(
                    active_chunk_controller[index] - executed_action_controller
                )
            )
        )
        raise RuntimeError(
            "Controller-space executed action does not match the tracked ACT plan "
            f"at index {index}; max_abs_error={max_error:.6g}"
        )
    queued = active_chunk_controller[index + 1 : n_action_steps].copy()
    discarded = active_chunk_controller[n_action_steps:].copy()
    if index + 1 == n_action_steps:
        active_chunk_controller = None
    return active_chunk_controller, queued, discarded, None


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
    loaded = load_act_checkpoint(
        manifest,
        env_config=env_config,
        n_action_steps_override=config.n_action_steps,
        temporal_ensemble_coeff_override=config.temporal_ensemble_coeff,
    )
    checkpoint_n_action_steps = loaded.checkpoint_n_action_steps
    checkpoint_temporal_ensemble_coeff = (
        loaded.checkpoint_temporal_ensemble_coeff
    )
    effective_n_action_steps = loaded.config.n_action_steps
    effective_temporal_ensemble_coeff = loaded.config.temporal_ensemble_coeff
    loaded.policy.reset()

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
    LOGGER.info(
        "[Evaluation] n_action_steps: checkpoint=%d effective=%d override=%s",
        checkpoint_n_action_steps,
        effective_n_action_steps,
        config.n_action_steps is not None,
    )
    LOGGER.info(
        "[Evaluation] temporal_ensemble_coeff: checkpoint=%s effective=%s override=%s",
        checkpoint_temporal_ensemble_coeff,
        effective_temporal_ensemble_coeff,
        config.temporal_ensemble_coeff is not None,
    )
    LOGGER.info("[Evaluation] perturbation: %s", config.perturbation)
    if config.perturbation == "proprio_gaussian":
        LOGGER.info(
            "[Evaluation] proprioception noise: std=%g in %s",
            config.proprio_noise_std,
            PROPRIO_NOISE_SPACE,
        )
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
        "policy_execution": {
            "checkpoint_n_action_steps": checkpoint_n_action_steps,
            "effective_n_action_steps": effective_n_action_steps,
            "n_action_steps_overridden": config.n_action_steps is not None,
            "checkpoint_temporal_ensemble_coeff": (
                checkpoint_temporal_ensemble_coeff
            ),
            "effective_temporal_ensemble_coeff": (
                effective_temporal_ensemble_coeff
            ),
            "temporal_ensemble_coeff_overridden": (
                config.temporal_ensemble_coeff is not None
            ),
        },
        "seeds": [config.seed + episode for episode in range(config.n_episodes)],
        "init_state_ids": list(range(config.n_episodes)),
        "perturbation": {
            "name": config.perturbation,
            "proprio_noise_std": (
                config.proprio_noise_std
                if config.perturbation == "proprio_gaussian"
                else None
            ),
            "noise_seed": config.seed,
            "space": (
                PROPRIO_NOISE_SPACE
                if config.perturbation == "proprio_gaussian"
                else None
            ),
        },
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

    try:
        kinematics = PandaSkeletonReconstructor(task)
        agentview_calibration = kinematics.fixed_camera_calibration(
            "agentview",
            (256, 256, 3),
        )
    except BaseException:
        close_envs(envs)
        release_output_directory_lock(output_dir)
        raise

    recording_path = output_dir / "rollout.rrd"
    temporary_recording_dir = output_dir / ".rerun_recordings"
    episode_recordings: list[LiberoEpisodeRecording] = []
    episode_recording_by_index: dict[int, LiberoEpisodeRecording] = {}
    rollout_rows: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    global_step = 0
    evaluation_started = time.perf_counter()
    plan_tracker = ActionPlanTracker(
        n_action_steps=loaded.config.n_action_steps,
        temporal_ensemble_coeff=loaded.config.temporal_ensemble_coeff,
    )
    publication_ready = False

    try:
        temporary_recording_dir.mkdir(parents=True, exist_ok=False)
        recording_episode_indices: tuple[int | None, ...] = (
            tuple(range(config.num_visualized_episodes))
            if config.num_visualized_episodes
            else (None,)
        )
        for episode_index in recording_episode_indices:
            filename = (
                "summary.rrd"
                if episode_index is None
                else f"episode_{episode_index:03d}.rrd"
            )
            temporary_path = temporary_recording_dir / filename
            stream = initialize_libero_recording_stream(
                output_path=temporary_path,
                recording_id=str(uuid.uuid4()),
                camera_keys=CAMERA_KEYS,
                fps=CONTROL_FREQUENCY_HZ,
                agentview_calibration=agentview_calibration,
                agentview_image_convention="lerobot_180",
            )
            episode_recording = LiberoEpisodeRecording(
                episode_index=episode_index,
                seed=(
                    None if episode_index is None else config.seed + episode_index
                ),
                init_state_id=episode_index,
                path=temporary_path,
                stream=stream,
            )
            episode_recordings.append(episode_recording)
            if episode_index is not None:
                episode_recording_by_index[episode_index] = episode_recording
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
                plan_tracker.reset_episode()
                active_chunk_controller: np.ndarray | None = None
                prepared = _prepare_libero_observation(
                    observation,
                    env=env,
                    env_preprocessor=env_preprocessor,
                )
                actions: list[np.ndarray] = []
                rewards: list[float] = []
                policy_latencies: list[float] = []
                model_latencies: list[float] = []
                success_seen = False
                termination_reason: str | None = None
                episode_recording = episode_recording_by_index.get(episode_index)
                visualized = episode_recording is not None
                video_frames: list[np.ndarray] = []

                for step_index in range(step_limit):
                    processed_observation = loaded.preprocessor(
                        prepared.policy_observation
                    )
                    (
                        policy_input,
                        policy_state_normalized_clean,
                        policy_state_normalized_input,
                    ) = prepare_policy_input(
                        processed_observation,
                        perturbation=config.perturbation,
                        proprio_noise_std=config.proprio_noise_std,
                        seed=config.seed,
                        episode_index=episode_index,
                        step_index=step_index,
                    )

                    model_timer.begin_step()
                    with torch.inference_mode(), _autocast_context(
                        str(loaded.config.device), loaded.config.use_amp
                    ):
                        action_policy, policy_latency_ms = timed_call(
                            loaded.policy.select_action,
                            policy_input,
                            device=loaded.config.device,
                        )
                    model_latency_ms = model_timer.current_latency_ms
                    plan_step = plan_tracker.record_step(
                        selected_action_policy=action_policy,
                        queried_chunk_policy=model_timer.current_chunk_policy,
                    )
                    action_controller = _postprocess_policy_action_tensor(
                        action_policy,
                        checkpoint_postprocessor=loaded.postprocessor,
                        env_postprocessor=env_postprocessor,
                    )
                    action_numpy = (
                        action_controller.detach().cpu().float().numpy()
                    )
                    if action_numpy.shape != (1, 7):
                        raise ValueError(
                            "Postprocessed LIBERO action must have shape (1, 7), "
                            f"got {action_numpy.shape}"
                        )
                    queried_chunk_controller: np.ndarray | None = None
                    if plan_step.queried_chunk_policy is not None:
                        queried_chunk_controller = (
                            _postprocess_policy_action_tensor(
                                plan_step.queried_chunk_policy,
                                checkpoint_postprocessor=loaded.postprocessor,
                                env_postprocessor=env_postprocessor,
                            )[0]
                            .detach()
                            .cpu()
                            .float()
                            .numpy()
                            .copy()
                        )
                    (
                        active_chunk_controller,
                        queued_actions,
                        discarded_actions,
                        new_prediction_chunk,
                    ) = _controller_plan_partitions(
                        plan_step,
                        queried_chunk_controller=queried_chunk_controller,
                        active_chunk_controller=active_chunk_controller,
                        executed_action_controller=action_numpy[0],
                        n_action_steps=loaded.config.n_action_steps,
                    )

                    next_observation, reward, terminated, truncated, info = env.step(
                        action_numpy
                    )
                    next_prepared = _prepare_libero_observation(
                        next_observation,
                        env=env,
                        env_preprocessor=env_preprocessor,
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

                    measured_pose = kinematics.forward_measured(
                        prepared.joint_positions,
                        prepared.state[6:8],
                    )
                    if visualized:
                        if episode_recording is None:
                            raise RuntimeError(
                                "Visualized LIBERO episode has no RecordingStream"
                            )
                        episode_recording.accept_step_index(step_index)
                        log_libero_step(
                            episode_recording.stream,
                            global_step=step_index,
                            episode_index=episode_index,
                            step_index=step_index,
                            observation_state=prepared.state,
                            next_observation_state=next_prepared.state,
                            policy_state_normalized_clean=(
                                policy_state_normalized_clean
                            ),
                            policy_state_normalized_input=(
                                policy_state_normalized_input
                            ),
                            measured_pose=measured_pose,
                            display_frame=kinematics.display_frame,
                            trajectory_history=episode_recording.trajectory_history,
                            action=executed_action,
                            queued_actions=queued_actions,
                            discarded_actions=discarded_actions,
                            new_prediction_chunk=new_prediction_chunk,
                            model_queried=plan_step.model_queried,
                            plan_step_index=plan_step.plan_step_index,
                            reward=reward_value,
                            success_so_far=success_seen,
                            done_after_step=done_after_step,
                            policy_step_latency_ms=policy_latency_ms,
                            model_query_latency_ms=model_latency_ms,
                            images=prepared.images,
                            post_action_agentview=next_prepared.images[
                                CAMERA_KEYS[0]
                            ],
                            agentview_calibration=agentview_calibration,
                            agentview_image_convention="lerobot_180",
                        )
                        if step_index == 0:
                            video_frames.append(
                                image_to_hwc_uint8(
                                    prepared.images[CAMERA_KEYS[0]]
                                )
                            )
                        video_frames.append(
                            image_to_hwc_uint8(
                                next_prepared.images[CAMERA_KEYS[0]]
                            )
                        )

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
                            "observation_state": prepared.state.tolist(),
                            "policy_state_normalized_clean": (
                                policy_state_normalized_clean.tolist()
                            ),
                            "policy_state_normalized_input": (
                                policy_state_normalized_input.tolist()
                            ),
                            "next_observation_state": next_prepared.state.tolist(),
                            "observation_joint_positions": (
                                prepared.joint_positions.tolist()
                            ),
                            "next_observation_joint_positions": (
                                next_prepared.joint_positions.tolist()
                            ),
                            "executed_action": executed_action.tolist(),
                            "model_queried": plan_step.model_queried,
                            "plan_step_index": plan_step.plan_step_index,
                            "predicted_chunk": (
                                None
                                if queried_chunk_controller is None
                                else queried_chunk_controller.tolist()
                            ),
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
                    prepared = next_prepared

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
                    if episode_recording is None:
                        raise RuntimeError(
                            "Visualized LIBERO episode has no RecordingStream"
                        )
                    episode_recording.send_final_name(
                        libero_episode_recording_name(
                            episode_index=episode_index,
                            seed=episode_seed,
                            init_state_id=init_state_id,
                            success=success_seen,
                            termination_reason=termination_reason,
                        )
                    )
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
        metrics["perturbation_config"] = input_config["perturbation"]
        metrics["visualized_episodes"] = list(
            range(config.num_visualized_episodes)
        )
        eval_info = _make_eval_info(
            metrics,
            task=task,
            formal=formal,
            elapsed_s=elapsed_s,
        )
        if config.num_visualized_episodes == 0:
            if len(episode_recordings) != 1:
                raise RuntimeError("Summary-only LIBERO evaluation requires one Recording")
            episode_recordings[0].send_final_name(LIBERO_SUMMARY_RECORDING_NAME)
        for episode_recording in episode_recordings:
            log_metric_summary(episode_recording.stream, metrics)
        write_metrics_report(output_dir, metrics)
        write_json_atomic(output_dir / "metrics.json", metrics)
        write_json_atomic(output_dir / "eval_info.json", eval_info)
        table = pa.Table.from_pylist(rollout_rows, schema=_rollout_schema())
        write_parquet_atomic(output_dir / "rollouts.parquet", table)

        flush_and_disconnect_recordings(episode_recordings)
        staging_path = stage_merged_recordings(
            [recording.path for recording in episode_recordings],
            recording_path,
        )
        verify_libero_recording_archive(staging_path, episode_recordings)
        staging_path.replace(recording_path)
        if not recording_path.is_file() or recording_path.stat().st_size == 0:
            raise RuntimeError(f"Rerun recording was not written: {recording_path}")
        run_config["status"] = status
        write_json_atomic(output_dir / "run_config.json", run_config)
        shutil.rmtree(temporary_recording_dir)
        publication_ready = True
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors: list[BaseException] = []
        try:
            flush_and_disconnect_recordings(episode_recordings)
        except BaseException as exc:
            cleanup_errors.append(exc)
            LOGGER.exception("[Output] Failed to close every temporary Rerun recording")
        try:
            kinematics.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
            LOGGER.exception("[Environment] Failed to close Panda kinematics")
        try:
            close_envs(envs)
        except BaseException as exc:
            cleanup_errors.append(exc)
            LOGGER.exception("[Environment] Failed to close LIBERO environments")
        if not cleanup_errors and publication_ready and not active_exception:
            try:
                mark_run_complete(output_dir)
            except BaseException as exc:
                cleanup_errors.append(exc)
                LOGGER.exception("[Output] Failed to mark LIBERO evaluation complete")
        else:
            try:
                release_output_directory_lock(output_dir)
            except BaseException as exc:
                cleanup_errors.append(exc)
                LOGGER.exception("[Output] Failed to release output directory lock")
        if cleanup_errors and not active_exception:
            raise cleanup_errors[0]

    viewer_opened = launch_rerun_viewer(recording_path, enabled=config.open_viewer)
    log_terminal_metrics(metrics, LOGGER)
    LOGGER.info("[Output] metrics: %s", output_dir / "metrics.json")
    LOGGER.info("[Output] metrics summary: %s", output_dir / "metrics.md")
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
    parser.add_argument(
        "--n-action-steps",
        type=int,
        help=(
            "Evaluation-only ACT execution horizon; defaults to the checkpoint value "
            "and must not exceed chunk_size"
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        help=(
            "Evaluation-only ACT temporal ensemble coefficient; requires explicit "
            "--n-action-steps=1 and defaults to the checkpoint value"
        ),
    )
    parser.add_argument(
        "--perturbation",
        choices=PERTURBATION_MODES,
        default="none",
        help="Policy-input perturbation; proprio_gaussian affects normalized state only",
    )
    parser.add_argument(
        "--proprio-noise-std",
        type=float,
        default=0.01,
        help="Gaussian std in checkpoint-normalized state space (default: 0.01)",
    )
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
