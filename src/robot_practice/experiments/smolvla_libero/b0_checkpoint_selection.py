"""Strict multi-GPU closed-loop checkpoint selection for SmolVLA LIBERO B0."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from robot_practice.evaluation.libero_assets import validate_libero_assets_dir
from robot_practice.experiments.smolvla_libero.b0_training import (
    portable_path,
    resolve_repository_path,
    write_json_atomic,
)

LOGGER = logging.getLogger(__name__)
EXPECTED_SUITE = "libero_goal"
EXPECTED_TASK_IDS = (0, 2, 3, 5, 6, 7, 9)
EXPECTED_VALIDATION_STATES = 20
EXPECTED_TEST_STATES = 30
EXPECTED_OFFICIAL_STATES = frozenset(range(50))
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
ROLLOUT_PROGRESS_PATTERN = re.compile(
    r"\[RolloutProgress\]\s+completed=(?P<completed>\d+)\s+total=(?P<total>\d+)"
)
HEARTBEAT_INTERVAL_S = 15.0


@dataclass(frozen=True)
class SelectionJob:
    checkpoint_step: int
    checkpoint_path: Path
    task_id: int
    init_state_ids: tuple[int, ...]
    output_dir: Path
    log_path: Path

    @property
    def rollout_count(self) -> int:
        return len(self.init_state_ids)


@dataclass(frozen=True)
class SelectionArguments:
    run_dir: Path
    checkpoint_paths: tuple[Path, ...] | None
    split_file: Path
    num_gpus: int
    env_seed: int
    policy_seed: int
    n_action_steps: int
    libero_assets_dir: Path
    output_dir: Path
    selected_dir: Path
    resume: bool
    overwrite: bool
    print_command: bool


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_int_list(value: Any, *, field: str, expected_count: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{field} must contain exactly {expected_count} values")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{field} must contain integers only")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate values")
    if not set(result) <= EXPECTED_OFFICIAL_STATES:
        raise ValueError(f"{field} must contain official init-state IDs in 0..49")
    return result


def load_selection_split(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("suite") != EXPECTED_SUITE:
        raise ValueError(f"Selection split must use suite={EXPECTED_SUITE}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise TypeError("Selection split tasks must be an object")
    parsed_task_ids = tuple(sorted(int(task_id) for task_id in tasks))
    if parsed_task_ids != EXPECTED_TASK_IDS:
        raise ValueError(
            f"Selection tasks must be exactly {EXPECTED_TASK_IDS}, got {parsed_task_ids}"
        )
    for task_id in EXPECTED_TASK_IDS:
        task = tasks[str(task_id)]
        if not isinstance(task, dict):
            raise TypeError(f"Selection task {task_id} must be an object")
        validation = _strict_int_list(
            task.get("validation_init_state_ids"),
            field=f"task {task_id} validation_init_state_ids",
            expected_count=EXPECTED_VALIDATION_STATES,
        )
        test = _strict_int_list(
            task.get("test_init_state_ids"),
            field=f"task {task_id} test_init_state_ids",
            expected_count=EXPECTED_TEST_STATES,
        )
        if set(validation) & set(test):
            raise ValueError(f"Selection task {task_id} validation/test states overlap")
        if set(validation) | set(test) != EXPECTED_OFFICIAL_STATES:
            raise ValueError(f"Selection task {task_id} validation/test states must cover 0..49")
    return payload


def _selection_protocol(split: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    protocol = split.get("selection_protocol")
    if not isinstance(protocol, dict):
        raise TypeError("Selection split has no selection_protocol object")
    steps = protocol.get("checkpoint_steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("selection_protocol.checkpoint_steps must be a non-empty list")
    checkpoint_steps = tuple(int(step) for step in steps)
    if checkpoint_steps != tuple(sorted(set(checkpoint_steps))):
        raise ValueError("selection_protocol.checkpoint_steps must be sorted and unique")
    policy_seed = protocol.get("policy_seed")
    if isinstance(policy_seed, bool) or not isinstance(policy_seed, int) or policy_seed < 0:
        raise ValueError("selection_protocol.policy_seed must be a non-negative integer")
    return checkpoint_steps, policy_seed


def discover_checkpoints(run_dir: Path, expected_steps: tuple[int, ...]) -> dict[int, Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoints_dir}")
    discovered: dict[int, Path] = {}
    for step in expected_steps:
        policy_path = checkpoints_dir / f"{step:06d}" / "pretrained_model"
        missing = [
            filename
            for filename in REQUIRED_CHECKPOINT_FILES
            if not (policy_path / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Checkpoint {step:06d} is incomplete; missing {', '.join(missing)}"
            )
        discovered[step] = policy_path
    return discovered


def discover_candidate_checkpoints(
    *,
    run_dir: Path,
    expected_steps: tuple[int, ...],
    checkpoint_paths: tuple[Path, ...] | None,
) -> dict[int, Path]:
    """Resolve the run layout or an explicitly enumerated frozen candidate set."""

    if checkpoint_paths is None:
        return discover_checkpoints(run_dir, expected_steps)
    checkpoints_root = (run_dir / "checkpoints").resolve()
    discovered: dict[int, Path] = {}
    for policy_path in checkpoint_paths:
        resolved = policy_path.resolve()
        if resolved.name != "pretrained_model":
            raise ValueError(
                "Each --checkpoint-path must point to a pretrained_model directory, got "
                f"{portable_path(resolved)}"
            )
        step_name = resolved.parent.name
        if re.fullmatch(r"[0-9]{6}", step_name) is None:
            raise ValueError(f"Cannot infer checkpoint step from {portable_path(resolved)}")
        step = int(step_name)
        if resolved.parent.parent != checkpoints_root:
            raise ValueError(
                "Explicit checkpoint paths must belong to --run-dir/checkpoints: "
                f"{portable_path(resolved)}"
            )
        if step in discovered:
            raise ValueError(f"Duplicate explicit checkpoint step: {step:06d}")
        missing = [
            filename for filename in REQUIRED_CHECKPOINT_FILES if not (resolved / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Checkpoint {step:06d} is incomplete; missing {', '.join(missing)}"
            )
        discovered[step] = resolved
    observed_steps = tuple(sorted(discovered))
    if observed_steps != expected_steps:
        raise ValueError(
            "Explicit --checkpoint-path values must cover the complete frozen candidate set; "
            f"expected {expected_steps}, got {observed_steps}"
        )
    return discovered


def build_jobs(
    *,
    checkpoints: dict[int, Path],
    split: dict[str, Any],
    output_dir: Path,
) -> list[SelectionJob]:
    tasks = split["tasks"]
    jobs: list[SelectionJob] = []
    for checkpoint_step, checkpoint_path in sorted(checkpoints.items()):
        for task_id in EXPECTED_TASK_IDS:
            init_state_ids = tuple(tasks[str(task_id)]["validation_init_state_ids"])
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


def build_evaluation_command(
    job: SelectionJob,
    *,
    env_seed: int,
    policy_seed: int,
    n_action_steps: int,
    libero_assets_dir: Path,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robot_practice.evaluation.libero_smolvla",
        f"--policy-path={portable_path(job.checkpoint_path)}",
        f"--suite={EXPECTED_SUITE}",
        f"--task-id={job.task_id}",
        f"--n-episodes={job.rollout_count}",
        "--num-visualized-episodes=0",
        f"--n-action-steps={n_action_steps}",
        f"--seed={env_seed}",
        f"--policy-seed={policy_seed}",
        "--init-state-ids=" + ",".join(str(value) for value in job.init_state_ids),
        f"--libero-assets-dir={portable_path(libero_assets_dir)}",
        f"--output-dir={portable_path(job.output_dir)}",
        "--no-open-viewer",
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _completed_job_metrics(
    job: SelectionJob,
    *,
    env_seed: int,
    policy_seed: int,
    n_action_steps: int,
) -> dict[str, Any] | None:
    metrics_path = job.output_dir / "metrics.json"
    run_config_path = job.output_dir / "run_config.json"
    if not metrics_path.is_file() or not run_config_path.is_file():
        return None
    metrics = _read_json(metrics_path)
    run_config = _read_json(run_config_path)
    aggregate = metrics.get("aggregate")
    if (
        metrics.get("status") != "complete"
        or not isinstance(aggregate, dict)
        or aggregate.get("num_episodes") != job.rollout_count
        or metrics.get("task", {}).get("task_id") != job.task_id
    ):
        return None
    run_input = run_config.get("input")
    run_policy = run_config.get("policy")
    if not isinstance(run_input, dict) or not isinstance(run_policy, dict):
        return None
    policy_execution = run_input.get("policy_execution")
    if not isinstance(policy_execution, dict):
        return None
    expected_seeds = [env_seed + index for index in range(job.rollout_count)]
    policy_path = run_policy.get("path")
    if not isinstance(policy_path, str):
        return None
    if (
        run_config.get("status") != "complete"
        or resolve_repository_path(policy_path) != job.checkpoint_path.resolve()
        or run_input.get("policy_seed") != policy_seed
        or run_input.get("seeds") != expected_seeds
        or policy_execution.get("effective_n_action_steps") != n_action_steps
    ):
        return None
    observed_states = tuple(
        int(episode["init_state_id"]) for episode in metrics.get("per_episode", [])
    )
    if observed_states != job.init_state_ids:
        return None
    return metrics


def _run_job(
    job: SelectionJob,
    *,
    gpu_pool: queue.Queue[int],
    progress_events: queue.Queue[tuple[SelectionJob, int, int]],
    active_processes: dict[tuple[int, int], subprocess.Popen[str]],
    active_processes_lock: threading.Lock,
    env_seed: int,
    policy_seed: int,
    n_action_steps: int,
    libero_assets_dir: Path,
    resume: bool,
    overwrite: bool,
) -> tuple[SelectionJob, dict[str, Any], int, bool]:
    if resume:
        existing = _completed_job_metrics(
            job,
            env_seed=env_seed,
            policy_seed=policy_seed,
            n_action_steps=n_action_steps,
        )
        if existing is not None:
            return job, existing, -1, True

    gpu = gpu_pool.get()
    try:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_evaluation_command(
            job,
            env_seed=env_seed,
            policy_seed=policy_seed,
            n_action_steps=n_action_steps,
            libero_assets_dir=libero_assets_dir,
            overwrite=overwrite or resume,
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        job_key = (job.checkpoint_step, job.task_id)
        with job.log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=resolve_repository_path("."),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with active_processes_lock:
                active_processes[job_key] = process
            previous_completed = 0
            try:
                if process.stdout is None:
                    raise RuntimeError("Selection subprocess did not expose stdout")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    match = ROLLOUT_PROGRESS_PATTERN.search(line)
                    if match is None:
                        continue
                    completed_rollouts = int(match.group("completed"))
                    expected_total = int(match.group("total"))
                    if expected_total != job.rollout_count:
                        raise RuntimeError(
                            f"checkpoint {job.checkpoint_step:06d} task {job.task_id} "
                            f"reported total={expected_total}, expected {job.rollout_count}"
                        )
                    if not previous_completed < completed_rollouts <= expected_total:
                        raise RuntimeError(
                            f"checkpoint {job.checkpoint_step:06d} task {job.task_id} "
                            "reported non-monotonic rollout progress"
                        )
                    progress_events.put((job, completed_rollouts - previous_completed, gpu))
                    previous_completed = completed_rollouts
                returncode = process.wait()
            finally:
                with active_processes_lock:
                    active_processes.pop(job_key, None)
        if returncode != 0:
            raise RuntimeError(
                f"checkpoint {job.checkpoint_step:06d} task {job.task_id} failed "
                f"on GPU {gpu}; inspect {portable_path(job.log_path)}"
            )
        metrics = _completed_job_metrics(
            job,
            env_seed=env_seed,
            policy_seed=policy_seed,
            n_action_steps=n_action_steps,
        )
        if metrics is None:
            raise RuntimeError(
                f"checkpoint {job.checkpoint_step:06d} task {job.task_id} did not "
                "produce a complete, protocol-matching metrics.json"
            )
        return job, metrics, gpu, False
    finally:
        gpu_pool.put(gpu)


def _terminate_active_processes(
    active_processes: dict[tuple[int, int], subprocess.Popen[str]],
    lock: threading.Lock,
) -> None:
    with lock:
        processes = list(active_processes.values())
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def _drain_progress_events(
    events: queue.Queue[tuple[SelectionJob, int, int]],
    progress: tqdm,
    observed: dict[tuple[int, int], int],
) -> None:
    while True:
        try:
            job, delta, gpu = events.get_nowait()
        except queue.Empty:
            return
        key = (job.checkpoint_step, job.task_id)
        observed[key] = observed.get(key, 0) + delta
        progress.update(delta)
        progress.set_postfix(
            checkpoint=f"{job.checkpoint_step:06d}",
            task=job.task_id,
            gpu=gpu,
        )


def summarize_checkpoint(
    checkpoint_step: int,
    task_metrics: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    all_episodes: list[dict[str, Any]] = []
    for task_id in EXPECTED_TASK_IDS:
        metrics = task_metrics[task_id]
        aggregate = metrics["aggregate"]
        episodes = metrics["per_episode"]
        all_episodes.extend(episodes)
        per_task[str(task_id)] = {
            "num_episodes": aggregate["num_episodes"],
            "num_successes": aggregate["num_successes"],
            "success_rate": aggregate["success_rate"],
            "avg_max_reward": aggregate["avg_max_reward"],
        }
    success_rates = [float(item["success_rate"]) for item in per_task.values()]
    max_rewards = [float(episode["max_reward"]) for episode in all_episodes]
    successful_steps = [int(episode["num_steps"]) for episode in all_episodes if episode["success"]]
    return {
        "checkpoint_step": checkpoint_step,
        "num_tasks": len(per_task),
        "num_episodes": len(all_episodes),
        "num_successes": sum(bool(episode["success"]) for episode in all_episodes),
        "macro_success_rate": float(np.mean(success_rates)),
        "pooled_success_rate": float(
            np.mean([bool(episode["success"]) for episode in all_episodes])
        ),
        "mean_max_reward": float(np.mean(max_rewards)),
        "mean_successful_episode_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "per_task": per_task,
    }


def checkpoint_rank_key(result: dict[str, Any]) -> tuple[float, float, float, int]:
    successful_steps = result["mean_successful_episode_steps"]
    return (
        -float(result["macro_success_rate"]),
        -float(result["mean_max_reward"]),
        float("inf") if successful_steps is None else float(successful_steps),
        int(result["checkpoint_step"]),
    )


def _write_metrics_markdown(
    path: Path,
    results: list[dict[str, Any]],
    selected_step: int,
) -> None:
    lines = [
        "# SmolVLA B0 checkpoint selection",
        "",
        "结果类别：Formal / Closed-loop / Validation。",
        "",
        "| Rank | Step | Macro SR | Successes | Mean max reward | Successful steps |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, result in enumerate(results, start=1):
        successful_steps = result["mean_successful_episode_steps"]
        lines.append(
            "| "
            f"{rank} | {result['checkpoint_step']:06d} | "
            f"{result['macro_success_rate']:.4f} | "
            f"{result['num_successes']}/{result['num_episodes']} | "
            f"{result['mean_max_reward']:.4f} | "
            f"{'N/A' if successful_steps is None else f'{successful_steps:.2f}'} |"
        )
    lines.extend(
        [
            "",
            f"Selected checkpoint: `{selected_step:06d}`.",
            "",
            (
                "Primary metric: macro success rate across seven tasks. Tie-breakers: "
                "mean max reward (higher), mean successful episode steps (lower), then "
                "earlier checkpoint step."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_selected_checkpoint(
    *,
    source_checkpoint_dir: Path,
    selected_dir: Path,
    selection: dict[str, Any],
) -> None:
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Directory rename/replace is not portable across the distributed and
        # overlay filesystems used by the training machines. Publish the selected
        # checkpoint as a direct, self-contained copy with no directory move.
        shutil.copytree(source_checkpoint_dir, selected_dir, symlinks=False)
        write_json_atomic(selected_dir / "selected_checkpoint.json", selection)
    except BaseException:
        if selected_dir.exists():
            shutil.rmtree(selected_dir)
        raise


def run_evaluation_jobs(
    jobs: list[SelectionJob],
    *,
    num_gpus: int,
    env_seed: int,
    policy_seed: int,
    n_action_steps: int,
    libero_assets_dir: Path,
    resume: bool,
    overwrite: bool,
    progress_description: str,
) -> dict[int, dict[int, dict[str, Any]]]:
    """Run task-level LIBERO jobs with bounded GPU concurrency and fail-fast cleanup."""

    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    total_rollouts = sum(job.rollout_count for job in jobs)
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in range(num_gpus):
        gpu_pool.put(gpu)
    checkpoint_steps = {job.checkpoint_step for job in jobs}
    collected: dict[int, dict[int, dict[str, Any]]] = {
        step: {} for step in checkpoint_steps
    }
    progress_events: queue.Queue[tuple[SelectionJob, int, int]] = queue.Queue()
    observed_progress: dict[tuple[int, int], int] = {}
    active_processes: dict[tuple[int, int], subprocess.Popen[str]] = {}
    active_processes_lock = threading.Lock()
    futures: list[Future[tuple[SelectionJob, dict[str, Any], int, bool]]] = []
    executor = ThreadPoolExecutor(max_workers=num_gpus)
    progress = tqdm(
        total=total_rollouts,
        desc=progress_description,
        unit="rollout",
        dynamic_ncols=True,
    )
    try:
        for job in jobs:
            futures.append(
                executor.submit(
                    _run_job,
                    job,
                    gpu_pool=gpu_pool,
                    progress_events=progress_events,
                    active_processes=active_processes,
                    active_processes_lock=active_processes_lock,
                    env_seed=env_seed,
                    policy_seed=policy_seed,
                    n_action_steps=n_action_steps,
                    libero_assets_dir=libero_assets_dir,
                    resume=resume,
                    overwrite=overwrite,
                )
            )
        pending = set(futures)
        last_heartbeat = time.monotonic()
        while pending:
            done, pending = wait(
                pending,
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            _drain_progress_events(progress_events, progress, observed_progress)
            for future in done:
                job, metrics, gpu, reused = future.result()
                collected[job.checkpoint_step][job.task_id] = metrics
                key = (job.checkpoint_step, job.task_id)
                remaining = job.rollout_count - observed_progress.get(key, 0)
                if remaining < 0:
                    raise RuntimeError(
                        f"Observed too many progress events for checkpoint "
                        f"{job.checkpoint_step:06d} task {job.task_id}"
                    )
                if remaining:
                    progress.update(remaining)
                    observed_progress[key] = job.rollout_count
                progress.set_postfix(
                    checkpoint=f"{job.checkpoint_step:06d}",
                    task=job.task_id,
                    gpu="cached" if reused else gpu,
                )
            if time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                with active_processes_lock:
                    active_jobs = len(active_processes)
                progress.set_postfix(
                    active_jobs=active_jobs,
                    completed=f"{progress.n}/{total_rollouts}",
                    phase="running",
                )
                progress.refresh()
                last_heartbeat = time.monotonic()
        _drain_progress_events(progress_events, progress, observed_progress)
    except BaseException:
        for future in futures:
            future.cancel()
        _terminate_active_processes(active_processes, active_processes_lock)
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        progress.close()
    return collected


def _prepare_output(arguments: SelectionArguments) -> None:
    if arguments.selected_dir.exists() and not arguments.resume:
        if not arguments.overwrite:
            raise FileExistsError(
                f"Selected checkpoint directory already exists: {arguments.selected_dir}; "
                "use --overwrite"
            )
        shutil.rmtree(arguments.selected_dir)
    if arguments.output_dir.exists():
        if arguments.resume:
            return
        if not arguments.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {arguments.output_dir}; "
                "use --resume or --overwrite"
            )
        shutil.rmtree(arguments.output_dir)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)


def select_checkpoints(arguments: SelectionArguments) -> dict[str, Any]:
    assets_dir = validate_libero_assets_dir(arguments.libero_assets_dir)
    split = load_selection_split(arguments.split_file)
    expected_steps, frozen_policy_seed = _selection_protocol(split)
    if arguments.policy_seed != frozen_policy_seed:
        raise ValueError(
            f"--policy-seed={arguments.policy_seed} violates the frozen split value "
            f"{frozen_policy_seed}"
        )
    checkpoints = discover_candidate_checkpoints(
        run_dir=arguments.run_dir,
        expected_steps=expected_steps,
        checkpoint_paths=arguments.checkpoint_paths,
    )
    jobs = build_jobs(
        checkpoints=checkpoints,
        split=split,
        output_dir=arguments.output_dir,
    )
    total_rollouts = sum(job.rollout_count for job in jobs)

    if arguments.print_command:
        payload = {
            "checkpoints": list(expected_steps),
            "checkpoint_paths": [portable_path(checkpoints[step]) for step in expected_steps],
            "tasks": list(EXPECTED_TASK_IDS),
            "jobs": len(jobs),
            "validation_rollouts": total_rollouts,
            "num_gpus": arguments.num_gpus,
            "first_command": build_evaluation_command(
                jobs[0],
                env_seed=arguments.env_seed,
                policy_seed=arguments.policy_seed,
                n_action_steps=arguments.n_action_steps,
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
            "arguments": {
                **asdict(arguments),
                "run_dir": portable_path(arguments.run_dir),
                "split_file": portable_path(arguments.split_file),
                "libero_assets_dir": portable_path(assets_dir),
                "output_dir": portable_path(arguments.output_dir),
                "selected_dir": portable_path(arguments.selected_dir),
            },
            "checkpoint_steps": list(expected_steps),
            "task_ids": list(EXPECTED_TASK_IDS),
            "libero_assets_dir": portable_path(assets_dir),
            "split_sha256": _sha256_file(arguments.split_file),
            "rollouts_per_task": EXPECTED_VALIDATION_STATES,
            "total_rollouts": total_rollouts,
        },
    )

    collected = run_evaluation_jobs(
        jobs,
        num_gpus=arguments.num_gpus,
        env_seed=arguments.env_seed,
        policy_seed=arguments.policy_seed,
        n_action_steps=arguments.n_action_steps,
        libero_assets_dir=assets_dir,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
        progress_description="B0 checkpoint selection",
    )

    ranked = sorted(
        (summarize_checkpoint(step, collected[step]) for step in expected_steps),
        key=checkpoint_rank_key,
    )
    selected = ranked[0]
    selected_step = int(selected["checkpoint_step"])
    selection = {
        "schema_version": 1,
        "status": "complete",
        "selection_set": "background_validation",
        "selected_checkpoint_step": selected_step,
        "source_checkpoint": portable_path(
            arguments.run_dir / "checkpoints" / f"{selected_step:06d}"
        ),
        "selected_checkpoint": portable_path(arguments.selected_dir),
        "primary_metric": "macro_success_rate",
        "tie_breakers": [
            "mean_max_reward_desc",
            "mean_successful_episode_steps_asc",
            "checkpoint_step_asc",
        ],
        "validation_rollouts": total_rollouts,
        "policy_seed": arguments.policy_seed,
        "libero_assets_dir": portable_path(assets_dir),
        "split_file": portable_path(arguments.split_file),
        "split_sha256": _sha256_file(arguments.split_file),
    }
    write_json_atomic(
        arguments.output_dir / "checkpoint_results.json",
        {"schema_version": 1, "status": "complete", "ranking": ranked},
    )
    write_json_atomic(arguments.output_dir / "selected_checkpoint.json", selection)
    _write_metrics_markdown(
        arguments.output_dir / "metrics.md",
        ranked,
        selected_step,
    )
    _copy_selected_checkpoint(
        source_checkpoint_dir=(arguments.run_dir / "checkpoints" / f"{selected_step:06d}"),
        selected_dir=arguments.selected_dir,
        selection=selection,
    )
    manifest = _read_json(arguments.output_dir / "selection_manifest.json")
    manifest["status"] = "complete"
    manifest["selected_checkpoint_step"] = selected_step
    write_json_atomic(arguments.output_dir / "selection_manifest.json", manifest)
    LOGGER.info("Selected B0 checkpoint %06d", selected_step)
    LOGGER.info("Selection report: %s", arguments.output_dir / "metrics.md")
    LOGGER.info("Selected checkpoint copy: %s", arguments.selected_dir)
    return selection


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/smolvla_libero/pilot/b0_domain_adaptation"),
    )
    parser.add_argument(
        "--checkpoint-path",
        action="append",
        type=Path,
        help=(
            "Repository-relative pretrained_model path for one frozen candidate; repeat once "
            "for every registered checkpoint"
        ),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/background_closed_loop_split.json"),
    )
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--policy-seed", type=int, default=2000)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument(
        "--libero-assets-dir",
        type=Path,
        default=Path("datasets/raw/libero_assets"),
        help="Offline LIBERO assets tree; validated before any GPU job starts",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selected-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete protocol-matching task jobs and continue missing jobs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selection output and any selected checkpoint copy",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Validate inputs and print the resolved selection workload without running",
    )
    return parser


def _resolve_arguments(namespace: argparse.Namespace) -> SelectionArguments:
    run_dir = resolve_repository_path(namespace.run_dir)
    raw_checkpoint_paths = namespace.checkpoint_path
    if raw_checkpoint_paths is not None and any(path.is_absolute() for path in raw_checkpoint_paths):
        raise ValueError("--checkpoint-path values must be repository-relative")
    checkpoint_paths = (
        None
        if raw_checkpoint_paths is None
        else tuple(resolve_repository_path(path) for path in raw_checkpoint_paths)
    )
    split_file = resolve_repository_path(namespace.split_file)
    libero_assets_dir = resolve_repository_path(namespace.libero_assets_dir)
    output_dir = resolve_repository_path(namespace.output_dir or run_dir / "checkpoint_selection")
    selected_dir = resolve_repository_path(
        namespace.selected_dir or run_dir / "checkpoints" / "selected"
    )
    if namespace.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive")
    if namespace.env_seed < 0 or namespace.policy_seed < 0:
        raise ValueError("--env-seed and --policy-seed must be non-negative")
    if namespace.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    if namespace.resume and namespace.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if output_dir == run_dir or selected_dir == run_dir:
        raise ValueError("Selection output paths must not replace --run-dir")
    return SelectionArguments(
        run_dir=run_dir,
        checkpoint_paths=checkpoint_paths,
        split_file=split_file,
        num_gpus=namespace.num_gpus,
        env_seed=namespace.env_seed,
        policy_seed=namespace.policy_seed,
        n_action_steps=namespace.n_action_steps,
        libero_assets_dir=libero_assets_dir,
        output_dir=output_dir,
        selected_dir=selected_dir,
        resume=namespace.resume,
        overwrite=namespace.overwrite,
        print_command=namespace.print_command,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    namespace = build_argument_parser().parse_args()
    select_checkpoints(_resolve_arguments(namespace))


if __name__ == "__main__":
    main()
