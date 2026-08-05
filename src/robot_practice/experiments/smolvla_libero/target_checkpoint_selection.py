"""C1/C2/C3 shared target checkpoint selection.

Every frozen checkpoint is evaluated on the three target LIBERO-Goal tasks with:

1. 360 closed-loop rollouts (6 checkpoints x 3 tasks x 20 validation states);
2. a fixed-frame, fixed-flow-noise offline open-loop action error over the 30
   offline validation demonstrations (2,969 frames total);
3. a three-level ranking: macro closed-loop success rate desc, macro normalized
   offline action MSE asc, then earlier checkpoint step.

The closed-loop job machinery, resume bookkeeping, and entity publication are
reused from ``b0_checkpoint_selection``; only the task set, the offline
criterion, and the ranking key are new.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION
from tqdm import tqdm

from robot_practice.evaluation.libero_act import _autocast_context, make_libero_config
from robot_practice.evaluation.libero_smolvla import _load_smolvla
from robot_practice.evaluation.libero_tasks import resolve_libero_task

from .b0_checkpoint_selection import (
    SelectionJob,
    _prepare_output,
    _read_json,
    _sha256_file,
    _strict_int_list,
    build_evaluation_command,
    publish_selected_checkpoint,
    run_evaluation_jobs,
)
from .b0_training import (
    DATASET_REPO_ID,
    portable_path,
    resolve_repository_path,
    write_json_atomic,
)

SUITE = "libero_goal"
TARGET_TASK_IDS = (1, 4, 8)
FROZEN_CHECKPOINT_STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
DEFAULT_POLICY_SEED = 1500
EXPECTED_VALIDATION_STATES = 20
EXPECTED_TEST_STATES = 30
EXPECTED_OFFICIAL_STATES = frozenset(range(50))
EXPECTED_OFFLINE_FRAMES = 2969
N_ACTION_STEPS = 10
CHUNK_SIZE = 50
OFFLINE_FLOW_NOISE_NAMESPACE = 3
OFFLINE_EPS = 1e-8
C_FAMILY_JOB_NAMES = (
    "smolvla_libero_c1_default_non_peft",
    "smolvla_libero_c2_canonical_lora",
    "smolvla_libero_c3_paraphrase_lora",
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetSelectionArguments:
    run_dir: Path
    checkpoint_steps: tuple[int, ...] | None
    split_file: Path
    dataset_root: Path
    num_gpus: int
    env_seed: int
    policy_seed: int
    libero_assets_dir: Path
    output_dir: Path
    selected_dir: Path
    resume: bool
    overwrite: bool
    print_command: bool
    max_validation_states: int | None
    max_offline_frames_per_task: int | None


def _parse_checkpoint_steps(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not re.fullmatch(r"[0-9]+", part) for part in parts):
        raise argparse.ArgumentTypeError(
            "--checkpoint-steps must be comma-separated non-negative integers"
        )
    return tuple(int(part) for part in parts)


def load_target_selection_split(path: Path) -> dict[str, Any]:
    """Validate the frozen target split used by every C-family selection."""

    payload = _read_json(path)
    if payload.get("suite") != SUITE:
        raise ValueError(f"Target split must use suite={SUITE}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise TypeError("Target split tasks must be an object")
    parsed_task_ids = tuple(sorted(int(task_id) for task_id in tasks))
    if parsed_task_ids != TARGET_TASK_IDS:
        raise ValueError(f"Target tasks must be exactly {TARGET_TASK_IDS}, got {parsed_task_ids}")
    for task_id in TARGET_TASK_IDS:
        task = tasks[str(task_id)]
        if not isinstance(task, dict):
            raise TypeError(f"Target task {task_id} must be an object")
        closed_loop = task.get("closed_loop_split")
        if not isinstance(closed_loop, dict):
            raise TypeError(f"Target task {task_id} lacks closed_loop_split")
        validation = _strict_int_list(
            closed_loop.get("validation_init_state_ids"),
            field=f"task {task_id} validation_init_state_ids",
            expected_count=EXPECTED_VALIDATION_STATES,
        )
        test = _strict_int_list(
            closed_loop.get("test_init_state_ids"),
            field=f"task {task_id} test_init_state_ids",
            expected_count=EXPECTED_TEST_STATES,
        )
        if set(validation) & set(test):
            raise ValueError(f"Target task {task_id} validation/test states overlap")
        if set(validation) | set(test) != EXPECTED_OFFICIAL_STATES:
            raise ValueError(f"Target task {task_id} validation/test states must cover 0..49")
        demonstrations = task.get("demonstration_split")
        if not isinstance(demonstrations, dict):
            raise TypeError(f"Target task {task_id} lacks demonstration_split")
        demos = demonstrations.get("offline_validation_demonstrations")
        if not isinstance(demos, list) or len(demos) != 10:
            raise ValueError(f"Target task {task_id} must have exactly 10 offline validation demos")
        demo_episodes = [int(record["episode_index"]) for record in demos]
        if len(set(demo_episodes)) != 10:
            raise ValueError(f"Target task {task_id} offline demo episodes must be unique")
        if not isinstance(task.get("canonical"), str) or not task["canonical"].strip():
            raise ValueError(f"Target task {task_id} must declare a canonical instruction")
    total_frames = sum(
        int(tasks[str(task_id)]["demonstration_split"]["offline_validation_frames"])
        for task_id in TARGET_TASK_IDS
    )
    if total_frames != EXPECTED_OFFLINE_FRAMES:
        raise ValueError(
            "Target offline validation frames must be "
            f"{EXPECTED_OFFLINE_FRAMES}, got {total_frames}"
        )
    return payload


def _validate_checkpoint_dir(policy_path: Path, step: int) -> Path:
    required = ["config.json", "train_config.json"]
    full_model = (policy_path / "model.safetensors").is_file()
    adapter_model = (policy_path / "adapter_config.json").is_file() and (
        policy_path / "adapter_model.safetensors"
    ).is_file()
    if not full_model and not adapter_model:
        required.append("model.safetensors or adapter_config.json+adapter_model.safetensors")
    missing = [filename for filename in required if not (policy_path / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {step:06d} is incomplete; missing {', '.join(missing)}"
        )
    train_config = _read_json(policy_path / "train_config.json")
    job_name = train_config.get("job_name")
    if job_name not in C_FAMILY_JOB_NAMES:
        raise ValueError(
            f"Checkpoint {step:06d} belongs to {job_name!r}, not a C1/C2/C3 run; "
            "refusing to select a cross-run checkpoint"
        )
    return policy_path


def _run_is_diagnostic(run_dir: Path) -> bool:
    run_config = run_dir / "run_config.json"
    if not run_config.is_file():
        return False
    payload = _read_json(run_config)
    training = payload.get("training")
    return isinstance(training, dict) and training.get("mode") == "diagnostic"


def target_selection_set_name(*, run_dir: Path) -> str:
    return "diagnostic_target_validation" if _run_is_diagnostic(run_dir) else "target_validation"


def target_result_set_name(*, run_dir: Path) -> str:
    return (
        "diagnostic_target_checkpoint_selection"
        if _run_is_diagnostic(run_dir)
        else "target_checkpoint_selection"
    )


def resolve_candidate_checkpoints(
    *,
    run_dir: Path,
    checkpoint_steps: tuple[int, ...] | None,
) -> dict[int, Path]:
    """Resolve frozen or explicitly enumerated C-family checkpoint directories."""

    steps = FROZEN_CHECKPOINT_STEPS if checkpoint_steps is None else checkpoint_steps
    if not steps:
        raise ValueError("checkpoint_steps must not be empty")
    if len(set(steps)) != len(steps):
        raise ValueError(f"checkpoint_steps contains duplicates: {steps}")
    if not set(steps) <= set(FROZEN_CHECKPOINT_STEPS):
        raise ValueError(
            f"checkpoint_steps must be a subset of {FROZEN_CHECKPOINT_STEPS}, got {steps}"
        )
    checkpoints_root = (run_dir / "checkpoints").resolve()
    discovered: dict[int, Path] = {}
    for step in steps:
        policy_path = checkpoints_root / f"{step:06d}" / "pretrained_model"
        discovered[step] = _validate_checkpoint_dir(policy_path, step)
    return dict(sorted(discovered.items()))


def build_target_jobs(
    *,
    checkpoints: dict[int, Path],
    split: dict[str, Any],
    output_dir: Path,
    max_validation_states: int | None,
) -> list[SelectionJob]:
    tasks = split["tasks"]
    jobs: list[SelectionJob] = []
    for checkpoint_step, checkpoint_path in sorted(checkpoints.items()):
        for task_id in TARGET_TASK_IDS:
            init_state_ids = tuple(
                tasks[str(task_id)]["closed_loop_split"]["validation_init_state_ids"][
                    :max_validation_states
                ]
            )
            task_output = (
                output_dir / "jobs" / f"checkpoint_{checkpoint_step:06d}" / f"task_{task_id:02d}"
            )
            jobs.append(
                SelectionJob(
                    checkpoint_step=checkpoint_step,
                    checkpoint_path=checkpoint_path,
                    task_id=task_id,
                    init_state_ids=init_state_ids,
                    output_dir=task_output,
                    log_path=output_dir
                    / "logs"
                    / f"checkpoint_{checkpoint_step:06d}_task_{task_id:02d}.log",
                )
            )
    return jobs


def _flow_noise_seed(
    *,
    split_seed: int,
    task_id: int,
    episode_index: int,
    frame_index: int,
) -> int:
    state = np.random.SeedSequence(
        [
            split_seed,
            OFFLINE_FLOW_NOISE_NAMESPACE,
            task_id,
            episode_index,
            frame_index,
        ]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _checkpoint_weights_sha256(policy_path: Path) -> str:
    full = policy_path / "model.safetensors"
    adapter = policy_path / "adapter_model.safetensors"
    if full.is_file():
        return _sha256_file(full)
    if adapter.is_file():
        return _sha256_file(adapter)
    raise FileNotFoundError(f"Checkpoint has no model or adapter weights: {policy_path}")


def evaluate_open_loop(
    *,
    checkpoint_path: Path,
    split: dict[str, Any],
    dataset_root: Path,
    max_frames_per_task: int | None = None,
) -> dict[str, Any]:
    """Teacher-forced action error over the frozen offline validation frames.

    Frames are enumerated deterministically from the split (all frames of the 30
    offline validation demonstrations), and the SmolVLA flow-matching noise is
    pinned by a per-frame seed derived from ``split_seed``, task, episode, and
    frame, so every checkpoint sees the same inputs and the same noise.
    """

    first_task = resolve_libero_task(SUITE, TARGET_TASK_IDS[0])
    env_config = make_libero_config(first_task)
    loaded = _load_smolvla(
        policy_reference=str(checkpoint_path),
        local_path=checkpoint_path,
        env_config=env_config,
        n_action_steps_override=N_ACTION_STEPS,
    )
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(loaded.config, metadata)
    action_stats = metadata.stats[ACTION]
    action_mean = torch.as_tensor(
        np.asarray(action_stats["mean"], dtype=np.float32),
        dtype=torch.float32,
    )
    action_std = torch.as_tensor(
        np.asarray(action_stats["std"], dtype=np.float32),
        dtype=torch.float32,
    )
    split_seed = int(split.get("split_seed", 1000))
    per_task: dict[str, dict[str, Any]] = {}
    total_frames = 0
    with torch.inference_mode():
        for task_id in TARGET_TASK_IDS:
            task = split["tasks"][str(task_id)]
            canonical = str(task["canonical"])
            demos = task["demonstration_split"]["offline_validation_demonstrations"]
            episodes = sorted(int(record["episode_index"]) for record in demos)
            dataset = LeRobotDataset(
                DATASET_REPO_ID,
                root=dataset_root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
            )
            # Dataset rows carry the pre-rename observation keys
            # (observation.images.image / image2); the checkpoint preprocessor
            # maps them to the policy's camera features.
            input_keys = tuple(sorted(key for key in dataset[0] if key.startswith("observation.")))
            selected_count = (
                len(dataset)
                if max_frames_per_task is None
                else min(len(dataset), max_frames_per_task)
            )
            normalized_sum = 0.0
            denormalized_sum = 0.0
            measured_frames = 0
            for local_index in tqdm(
                range(selected_count),
                desc=f"open-loop task {task_id}",
                unit="frame",
                dynamic_ncols=True,
            ):
                item = dataset[local_index]
                padding = item.get("action_is_pad")
                if padding is not None and bool(padding[0]):
                    continue
                episode_index = int(item["episode_index"])
                frame_index = int(item["frame_index"])
                batch = {key: item[key].unsqueeze(0) for key in input_keys}
                batch["task"] = [canonical]
                processed = loaded.preprocessor(batch)
                noise_seed = _flow_noise_seed(
                    split_seed=split_seed,
                    task_id=task_id,
                    episode_index=episode_index,
                    frame_index=frame_index,
                )
                torch.manual_seed(noise_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(noise_seed)
                with _autocast_context(
                    str(loaded.config.device),
                    loaded.config.use_amp,
                ):
                    predicted_normalized = loaded.policy.predict_action_chunk(processed)
                predicted_raw = loaded.postprocessor(predicted_normalized)
                target_raw = (
                    item[ACTION]
                    .unsqueeze(0)
                    .to(
                        device=predicted_raw.device,
                        dtype=predicted_raw.dtype,
                    )
                )
                prediction = predicted_raw[:, :N_ACTION_STEPS, :]
                target = target_raw[:, :N_ACTION_STEPS, :]
                denormalized_mse = float(torch.mean((prediction - target) ** 2).cpu())
                mean = action_mean.to(target.device)
                std = action_std.to(target.device) + OFFLINE_EPS
                target_normalized = (target - mean) / std
                predicted_normalized_cpu = (
                    predicted_normalized[:, :N_ACTION_STEPS, :].detach().cpu()
                )
                normalized_mse = float(
                    torch.mean((predicted_normalized_cpu - target_normalized) ** 2).cpu()
                )
                normalized_sum += normalized_mse
                denormalized_sum += denormalized_mse
                measured_frames += 1
            if measured_frames == 0:
                raise RuntimeError(f"Task {task_id} offline validation produced no measured frames")
            per_task[str(task_id)] = {
                "frame_count": measured_frames,
                "normalized_mse": normalized_sum / measured_frames,
                "denormalized_mse": denormalized_sum / measured_frames,
            }
            total_frames += measured_frames
    macro_normalized = float(
        np.mean([per_task[str(task_id)]["normalized_mse"] for task_id in TARGET_TASK_IDS])
    )
    macro_denormalized = float(
        np.mean([per_task[str(task_id)]["denormalized_mse"] for task_id in TARGET_TASK_IDS])
    )
    return {
        "frame_count": total_frames,
        "macro_normalized_mse": macro_normalized,
        "macro_denormalized_mse": macro_denormalized,
        "per_task": per_task,
    }


def summarize_target_checkpoint(
    checkpoint_step: int,
    task_metrics: dict[int, dict[str, Any]],
    offline_error: dict[str, Any],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    total_episodes = 0
    total_successes = 0
    for task_id in TARGET_TASK_IDS:
        metrics = task_metrics[task_id]
        aggregate = metrics["aggregate"]
        total_episodes += aggregate["num_episodes"]
        total_successes += aggregate["num_successes"]
        per_task[str(task_id)] = {
            "num_episodes": aggregate["num_episodes"],
            "num_successes": aggregate["num_successes"],
            "success_rate": aggregate["success_rate"],
        }
    success_rates = [float(per_task[str(task_id)]["success_rate"]) for task_id in TARGET_TASK_IDS]
    return {
        "checkpoint_step": checkpoint_step,
        "num_tasks": len(per_task),
        "num_episodes": total_episodes,
        "num_successes": total_successes,
        "macro_success_rate": float(np.mean(success_rates)),
        "offline_normalized_action_mse": float(offline_error["macro_normalized_mse"]),
        "offline_denormalized_action_mse": float(offline_error["macro_denormalized_mse"]),
        "offline_frame_count": int(offline_error["frame_count"]),
        "offline_per_task": offline_error["per_task"],
        "per_task": per_task,
    }


def checkpoint_rank_key(result: dict[str, Any]) -> tuple[float, float, int]:
    """Three-level key; the third level is the checkpoint step itself."""

    return (
        -float(result["macro_success_rate"]),
        float(result["offline_normalized_action_mse"] or float("inf")),
        int(result["checkpoint_step"]),
    )


def build_ranking_audit(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        entry: dict[str, Any] = {
            "rank": rank,
            "checkpoint_step": int(result["checkpoint_step"]),
            "macro_success_rate": float(result["macro_success_rate"]),
            "offline_normalized_action_mse": float(result["offline_normalized_action_mse"]),
        }
        if rank > 1:
            previous = results[rank - 2]
            if result["macro_success_rate"] != previous["macro_success_rate"]:
                level = 1
                note = "macro closed-loop success rate"
            elif (
                result["offline_normalized_action_mse"] != previous["offline_normalized_action_mse"]
            ):
                level = 2
                note = "macro normalized offline action MSE"
            else:
                level = 3
                note = "earlier checkpoint step"
            entry["tie_breaker_level"] = level
            entry["tie_breaker_note"] = note
            entry["tie_breaker_reference_step"] = int(previous["checkpoint_step"])
        audit.append(entry)
    return audit


def _write_target_metrics_markdown(
    path: Path,
    results: list[dict[str, Any]],
    selected_step: int,
    audit: list[dict[str, Any]],
) -> None:
    lines = [
        "# SmolVLA C1/C2/C3 target checkpoint selection",
        "",
        "结果类别：Formal / Closed-loop / Validation。",
        "",
        "| Rank | Step | Macro SR | Offline norm MSE | Successes | Tie-break |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, result in enumerate(results, start=1):
        audit_entry = next(entry for entry in audit if entry["rank"] == rank)
        tie_break = (
            f"level {audit_entry['tie_breaker_level']}"
            if "tie_breaker_level" in audit_entry
            else "winner"
        )
        lines.append(
            "| "
            f"{rank} | {result['checkpoint_step']:06d} | "
            f"{result['macro_success_rate']:.4f} | "
            f"{result['offline_normalized_action_mse']:.6g} | "
            f"{result['num_successes']}/{result['num_episodes']} | "
            f"{tie_break} |"
        )
    lines.extend(
        [
            "",
            f"Selected checkpoint: `{selected_step:06d}`.",
            "",
            (
                "Ranking criteria: macro closed-loop success rate desc, macro normalized "
                "offline action MSE asc, then earlier checkpoint step."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _open_loop_result_path(output_dir: Path, step: int) -> Path:
    return output_dir / "jobs" / "open_loop" / f"checkpoint_{step:06d}.json"


def _open_loop_log_path(output_dir: Path, step: int) -> Path:
    return output_dir / "logs" / f"open_loop_checkpoint_{step:06d}.log"


def _open_loop_result_reusable(
    result: dict[str, Any] | None,
    *,
    checkpoint_id: str,
    split_sha256: str,
    max_frames_per_task: int | None,
) -> bool:
    if result is None:
        return False
    return (
        result.get("status") == "complete"
        and result.get("checkpoint_id") == checkpoint_id
        and result.get("split_sha256") == split_sha256
        and result.get("max_frames_per_task") == max_frames_per_task
    )


def _open_loop_command(
    *,
    arguments: TargetSelectionArguments,
    step: int,
    output_path: Path,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robot_practice.experiments.smolvla_libero.target_checkpoint_selection",
        "--open-loop-only",
        f"--run-dir={arguments.run_dir}",
        f"--checkpoint-step={step}",
        f"--split-file={arguments.split_file}",
        f"--dataset-root={arguments.dataset_root}",
        f"--open-loop-output={output_path}",
    ]
    if arguments.max_offline_frames_per_task is not None:
        command.append(f"--max-offline-frames-per-task={arguments.max_offline_frames_per_task}")
    if overwrite:
        command.append("--overwrite")
    return command


def _planned_open_loop_frames(
    split: dict[str, Any],
    checkpoints: dict[int, Path],
    max_frames_per_task: int | None,
) -> int:
    per_task = 0
    for task_id in TARGET_TASK_IDS:
        frames = int(
            split["tasks"][str(task_id)]["demonstration_split"]["offline_validation_frames"]
        )
        per_task += frames if max_frames_per_task is None else min(frames, max_frames_per_task)
    return per_task * len(checkpoints)


def _run_open_loop_parallel(
    *,
    checkpoints: dict[int, Path],
    split: dict[str, Any],
    arguments: TargetSelectionArguments,
    split_sha256: str,
    progress_total: int,
) -> dict[int, dict[str, Any]]:
    """Dispatch one checkpoint per GPU through the same pool pattern as closed loop."""

    jobs_dir = arguments.output_dir / "jobs" / "open_loop"
    logs_dir = arguments.output_dir / "logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    for step in sorted(checkpoints):
        result_path = _open_loop_result_path(arguments.output_dir, step)
        existing = _read_json(result_path) if result_path.is_file() else None
        checkpoint_id = _checkpoint_weights_sha256(checkpoints[step])
        if _open_loop_result_reusable(
            existing,
            checkpoint_id=checkpoint_id,
            split_sha256=split_sha256,
            max_frames_per_task=arguments.max_offline_frames_per_task,
        ):
            results[step] = existing
        else:
            pending.append(step)

    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in range(arguments.num_gpus):
        gpu_pool.put(gpu)
    active_lock = threading.Lock()
    active_processes: dict[subprocess.Popen[str], int] = {}

    def run_one(step: int) -> tuple[int, dict[str, Any]]:
        gpu = gpu_pool.get()
        log_path = _open_loop_log_path(arguments.output_dir, step)
        result_path = _open_loop_result_path(arguments.output_dir, step)
        command = _open_loop_command(
            arguments=arguments,
            step=step,
            output_path=result_path,
            overwrite=result_path.exists(),
        )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(shlex.quote(part) for part in command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            with active_lock:
                active_processes[process] = gpu
            try:
                returncode = process.wait()
            finally:
                with active_lock:
                    active_processes.pop(process, None)
                gpu_pool.put(gpu)
        if returncode != 0:
            raise RuntimeError(
                f"Open-loop checkpoint {step:06d} failed on GPU {gpu}; see {log_path}"
            )
        return step, _read_json(result_path)

    def terminate_active() -> None:
        with active_lock:
            processes = list(active_processes)
        for process in processes:
            process.terminate()

    executor = ThreadPoolExecutor(max_workers=arguments.num_gpus)
    futures = {executor.submit(run_one, step): step for step in pending}
    try:
        with tqdm(
            total=progress_total,
            desc="Open-loop action error",
            unit="frame",
            dynamic_ncols=True,
        ) as progress:
            for step, result in results.items():
                progress.update(int(result["frame_count"]))
            for future in as_completed(futures):
                try:
                    step, result = future.result()
                except BaseException:
                    terminate_active()
                    raise
                results[step] = result
                progress.update(int(result["frame_count"]))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return {step: results[step] for step in sorted(checkpoints)}


def select_target_checkpoint(arguments: TargetSelectionArguments) -> dict[str, Any]:
    from robot_practice.evaluation.libero_assets import validate_libero_assets_dir

    assets_dir = validate_libero_assets_dir(arguments.libero_assets_dir)
    split = load_target_selection_split(arguments.split_file)
    checkpoints = resolve_candidate_checkpoints(
        run_dir=arguments.run_dir,
        checkpoint_steps=arguments.checkpoint_steps,
    )
    diagnostic = _run_is_diagnostic(arguments.run_dir)
    selection_set = target_selection_set_name(run_dir=arguments.run_dir)
    result_set = target_result_set_name(run_dir=arguments.run_dir)
    jobs = build_target_jobs(
        checkpoints=checkpoints,
        split=split,
        output_dir=arguments.output_dir,
        max_validation_states=arguments.max_validation_states,
    )
    total_rollouts = sum(job.rollout_count for job in jobs)
    offline_frames = sum(
        int(split["tasks"][str(task_id)]["demonstration_split"]["offline_validation_frames"])
        for task_id in TARGET_TASK_IDS
    )

    if arguments.print_command:
        open_loop_parallelism = min(arguments.num_gpus, len(checkpoints))
        payload = {
            "diagnostic": diagnostic,
            "checkpoint_steps": list(checkpoints),
            "tasks": list(TARGET_TASK_IDS),
            "jobs": len(jobs),
            "validation_rollouts": total_rollouts,
            "offline_validation_frames": offline_frames,
            "open_loop_frames_per_checkpoint": offline_frames,
            "open_loop_total_frames": _planned_open_loop_frames(
                split,
                checkpoints,
                arguments.max_offline_frames_per_task,
            ),
            "open_loop_parallelism": open_loop_parallelism,
            "num_gpus": arguments.num_gpus,
            "policy_seed": arguments.policy_seed,
            "first_command": build_evaluation_command(
                jobs[0],
                env_seed=arguments.env_seed,
                policy_seed=arguments.policy_seed,
                n_action_steps=N_ACTION_STEPS,
                libero_assets_dir=assets_dir,
                overwrite=arguments.overwrite,
            ),
        }
        print(json.dumps(payload, indent=2))
        return payload

    _prepare_output(arguments)
    write_json_atomic(
        arguments.output_dir / "selection_manifest.json",
        {
            "schema_version": 1,
            "status": "running",
            "result_set": result_set,
            "selection_set": selection_set,
            "arguments": {
                **{field: getattr(arguments, field) for field in vars(arguments)},
                "run_dir": portable_path(arguments.run_dir),
                "split_file": portable_path(arguments.split_file),
                "dataset_root": portable_path(arguments.dataset_root),
                "libero_assets_dir": portable_path(assets_dir),
                "output_dir": portable_path(arguments.output_dir),
                "selected_dir": portable_path(arguments.selected_dir),
            },
            "checkpoint_steps": list(checkpoints),
            "task_ids": list(TARGET_TASK_IDS),
            "libero_assets_dir": portable_path(assets_dir),
            "split_sha256": _sha256_file(arguments.split_file),
            "validation_rollouts": total_rollouts,
            "offline_validation_frames": offline_frames,
        },
    )

    collected = run_evaluation_jobs(
        jobs,
        num_gpus=arguments.num_gpus,
        env_seed=arguments.env_seed,
        policy_seed=arguments.policy_seed,
        n_action_steps=N_ACTION_STEPS,
        libero_assets_dir=assets_dir,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
        progress_description="C-family checkpoint selection",
    )

    split_sha256 = _sha256_file(arguments.split_file)
    open_loop_results = _run_open_loop_parallel(
        checkpoints=checkpoints,
        split=split,
        arguments=arguments,
        split_sha256=split_sha256,
        progress_total=_planned_open_loop_frames(
            split,
            checkpoints,
            arguments.max_offline_frames_per_task,
        ),
    )
    results = [
        summarize_target_checkpoint(
            step,
            collected[step],
            open_loop_results[step],
        )
        for step in sorted(checkpoints)
    ]
    ranked = sorted(results, key=checkpoint_rank_key)
    audit = build_ranking_audit(ranked)
    selected = ranked[0]
    selected_step = int(selected["checkpoint_step"])
    selection = {
        "schema_version": 1,
        "status": "complete",
        "selection_set": selection_set,
        "selected_checkpoint_step": selected_step,
        "source_checkpoint": portable_path(
            arguments.run_dir / "checkpoints" / f"{selected_step:06d}"
        ),
        "selected_checkpoint": portable_path(arguments.selected_dir),
        "primary_metric": "macro_success_rate",
        "tie_breakers": [
            "macro_normalized_action_mse_asc",
            "checkpoint_step_asc",
        ],
        "validation_rollouts": total_rollouts,
        "offline_validation_frames": offline_frames,
        "policy_seed": arguments.policy_seed,
        "libero_assets_dir": portable_path(assets_dir),
        "split_file": portable_path(arguments.split_file),
        "split_sha256": _sha256_file(arguments.split_file),
    }
    write_json_atomic(
        arguments.output_dir / "checkpoint_results.json",
        {
            "schema_version": 1,
            "status": "complete",
            "result_set": result_set,
            "ranking": ranked,
            "ranking_audit": audit,
        },
    )
    write_json_atomic(arguments.output_dir / "ranking_audit.json", {"audit": audit})
    write_json_atomic(arguments.output_dir / "selected_checkpoint.json", selection)
    _write_target_metrics_markdown(
        arguments.output_dir / "metrics.md",
        ranked,
        selected_step,
        audit,
    )
    publish_selected_checkpoint(
        selection_file=arguments.output_dir / "selected_checkpoint.json",
        selected_dir=arguments.selected_dir,
        overwrite=arguments.overwrite or arguments.resume,
        selection_set=selection_set,
    )
    manifest = _read_json(arguments.output_dir / "selection_manifest.json")
    manifest["status"] = "complete"
    manifest["selected_checkpoint_step"] = selected_step
    manifest["ranking_audit"] = audit
    write_json_atomic(arguments.output_dir / "selection_manifest.json", manifest)
    LOGGER.info("Selected target checkpoint %06d", selected_step)
    LOGGER.info("Selection report: %s", arguments.output_dir / "metrics.md")
    LOGGER.info("Selected checkpoint copy: %s", arguments.selected_dir)
    return selection


def run_open_loop_only(namespace: argparse.Namespace) -> None:
    """Evaluate one checkpoint and write only its open-loop result JSON."""

    split = load_target_selection_split(resolve_repository_path(namespace.split_file))
    checkpoints = resolve_candidate_checkpoints(
        run_dir=resolve_repository_path(namespace.run_dir),
        checkpoint_steps=(int(namespace.checkpoint_step),),
    )
    step = int(namespace.checkpoint_step)
    checkpoint_path = checkpoints[step]
    output = resolve_repository_path(namespace.open_loop_output)
    if output.exists() and not namespace.overwrite:
        raise FileExistsError(f"Open-loop output already exists: {output}; use --overwrite")
    result = evaluate_open_loop(
        checkpoint_path=checkpoint_path,
        split=split,
        dataset_root=resolve_repository_path(namespace.dataset_root),
        max_frames_per_task=namespace.max_offline_frames_per_task,
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_step": step,
        "checkpoint_id": _checkpoint_weights_sha256(checkpoint_path),
        "split_sha256": _sha256_file(resolve_repository_path(namespace.split_file)),
        "max_frames_per_task": namespace.max_offline_frames_per_task,
        **result,
    }
    write_json_atomic(output, payload)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/smolvla_libero/pilot/c1_default_non_peft"),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=_parse_checkpoint_steps,
        help=(
            "Comma-separated frozen checkpoint steps; defaults to "
            f"{','.join(str(step) for step in FROZEN_CHECKPOINT_STEPS)}"
        ),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/target_split.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/lerobot/lerobot/libero"),
    )
    parser.add_argument("--suite", default=SUITE)
    parser.add_argument("--task-ids", default="1,4,8")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=DEFAULT_POLICY_SEED)
    parser.add_argument(
        "--libero-assets-dir",
        type=Path,
        default=Path("datasets/raw/libero_assets"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selected-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Copy an existing completed selection to --selected-dir without rollouts",
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="Completed selected_checkpoint.json consumed by --publish-only",
    )
    parser.add_argument(
        "--max-validation-states-per-task",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-offline-frames-per-task",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--open-loop-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--open-loop-output",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_arguments(namespace: argparse.Namespace) -> TargetSelectionArguments:
    if namespace.suite != SUITE:
        raise ValueError(f"--suite must be {SUITE}, got {namespace.suite!r}")
    task_ids = tuple(int(value) for value in namespace.task_ids.split(","))
    if task_ids != TARGET_TASK_IDS:
        raise ValueError(f"--task-ids must be {TARGET_TASK_IDS}, got {task_ids}")
    if namespace.policy_seed != DEFAULT_POLICY_SEED:
        raise ValueError(
            f"--policy-seed must be {DEFAULT_POLICY_SEED}, got {namespace.policy_seed}"
        )
    num_gpus = namespace.num_gpus or 8
    if num_gpus <= 0:
        raise ValueError("--num-gpus must be positive")
    run_dir = resolve_repository_path(namespace.run_dir)
    split_file = resolve_repository_path(namespace.split_file)
    dataset_root = resolve_repository_path(namespace.dataset_root)
    libero_assets_dir = resolve_repository_path(namespace.libero_assets_dir)
    output_dir = resolve_repository_path(namespace.output_dir or run_dir / "checkpoint_selection")
    selected_dir = resolve_repository_path(
        namespace.selected_dir or run_dir / "checkpoints" / "selected"
    )
    if output_dir == run_dir or selected_dir == run_dir:
        raise ValueError("Selection output paths must not replace --run-dir")
    if namespace.resume and namespace.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    for name in ("max_validation_states_per_task", "max_offline_frames_per_task"):
        value = getattr(namespace, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return TargetSelectionArguments(
        run_dir=run_dir,
        checkpoint_steps=namespace.checkpoint_steps,
        split_file=split_file,
        dataset_root=dataset_root,
        num_gpus=num_gpus,
        env_seed=namespace.env_seed,
        policy_seed=namespace.policy_seed,
        libero_assets_dir=libero_assets_dir,
        output_dir=output_dir,
        selected_dir=selected_dir,
        resume=namespace.resume,
        overwrite=namespace.overwrite,
        print_command=namespace.print_command,
        max_validation_states=namespace.max_validation_states_per_task,
        max_offline_frames_per_task=namespace.max_offline_frames_per_task,
    )


def publish_target_checkpoint(namespace: argparse.Namespace) -> None:
    """Publish an existing completed target selection without rerunning rollouts."""
    if namespace.selection_file is None:
        raise ValueError("--publish-only requires --selection-file")
    run_dir = resolve_repository_path(namespace.run_dir)
    selection_file = resolve_repository_path(namespace.selection_file)
    output_dir = resolve_repository_path(namespace.output_dir or run_dir / "checkpoint_selection")
    selected_dir = resolve_repository_path(
        namespace.selected_dir or run_dir / "checkpoints" / "selected"
    )
    selection = publish_selected_checkpoint(
        selection_file=selection_file,
        selected_dir=selected_dir,
        overwrite=namespace.overwrite,
        selection_set=target_selection_set_name(run_dir=run_dir),
    )
    manifest_path = output_dir / "selection_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    manifest["status"] = "complete"
    manifest["selected_checkpoint_step"] = selection["selected_checkpoint_step"]
    audit_path = output_dir / "ranking_audit.json"
    if audit_path.is_file():
        audit = _read_json(audit_path)
        if isinstance(audit, dict) and "audit" in audit:
            manifest["ranking_audit"] = audit["audit"]
    write_json_atomic(manifest_path, manifest)
    LOGGER.info(
        "Published target checkpoint %06d to %s",
        selection["selected_checkpoint_step"],
        selected_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_argument_parser()
    namespace = parser.parse_args(argv)
    if namespace.publish_only:
        publish_target_checkpoint(namespace)
        return
    if namespace.open_loop_only:
        if namespace.checkpoint_step is None:
            parser.error("--open-loop-only requires --checkpoint-step")
        if namespace.open_loop_output is None:
            parser.error("--open-loop-only requires --open-loop-output")
        if namespace.num_gpus is not None:
            parser.error("--open-loop-only does not accept --num-gpus")
        run_open_loop_only(namespace)
        return
    select_target_checkpoint(_resolve_arguments(namespace))


if __name__ == "__main__":
    main()
