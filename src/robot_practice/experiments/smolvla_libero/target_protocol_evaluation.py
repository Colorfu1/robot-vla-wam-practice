"""Unified multi-GPU target-task evaluation for SmolVLA B0/C1/C2/C3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from robot_practice.evaluation.common import write_parquet_atomic
from robot_practice.evaluation.libero_assets import validate_libero_assets_dir
from robot_practice.evaluation.libero_goal_classifier import (
    FAILED_GOAL,
    GOAL_CLASSIFIER_NAME,
    GOAL_LABELS,
)
from robot_practice.experiments.smolvla_libero.b0_checkpoint_selection import (
    REQUIRED_CHECKPOINT_FILES,
    ROLLOUT_PROGRESS_PATTERN,
)
from robot_practice.experiments.smolvla_libero.b0_training import (
    load_json_object,
    portable_path,
    resolve_repository_path,
    write_json_atomic,
)
from robot_practice.experiments.smolvla_libero.data_preparation import (
    HELD_OUT_PARAPHRASE_COUNT,
    sha256_file,
    verify_paraphrase_binding,
)
from robot_practice.experiments.smolvla_libero.target_training import (
    TargetSplit,
    load_target_split,
)

LOGGER = logging.getLogger(__name__)
PROTOCOLS = ("canonical", "held_out_paraphrase", "prompt_swap")
EXPECTED_TASK_IDS = (1, 4, 8)
EXPECTED_TEST_STATES = 30
EXPECTED_CANONICAL_ROLLOUTS = 90
EXPECTED_PARAPHRASE_ROLLOUTS = 90
EXPECTED_PROMPT_SWAP_ROLLOUTS = 30
EXPECTED_TOTAL_ROLLOUTS = 210
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 1000
HEARTBEAT_INTERVAL_S = 15.0
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class TargetEvaluationArguments:
    policy_path: Path
    model_label: str
    split_file: Path
    paraphrases: Path
    protocol_config: Path
    protocols: tuple[str, ...]
    num_gpus: int
    env_seed: int
    policy_seed: int
    n_action_steps: int
    num_visualized_episodes_per_job: int
    max_steps: int | None
    libero_assets_dir: Path
    output_dir: Path
    resume: bool
    overwrite: bool
    print_command: bool


@dataclass(frozen=True)
class TargetEvaluationJob:
    job_id: str
    protocol: str
    task_id: int
    init_state_ids: tuple[int, ...]
    episode_seeds: tuple[int, ...]
    instruction: str
    output_dir: Path
    log_path: Path
    prompt_index: int | None = None
    expected_goal: str | None = None

    @property
    def rollout_count(self) -> int:
        return len(self.init_state_ids)


@dataclass(frozen=True)
class TargetEvaluationSpec:
    split: TargetSplit
    paraphrases_payload: dict[str, Any]
    protocol_payload: dict[str, Any]


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_policy(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Policy directory does not exist: {portable_path(path)}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        # LoRA checkpoints replace the full model file with a complete adapter pair.
        missing_without_weights = [name for name in missing if name != "model.safetensors"]
        adapter_complete = all(
            (path / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")
        )
        if missing_without_weights or not adapter_complete:
            raise FileNotFoundError("Incomplete SmolVLA policy; missing " + ", ".join(missing))
    return path.resolve()


def _test_states(split: TargetSplit, task_id: int) -> tuple[int, ...]:
    task = split.payload["tasks"][str(task_id)]
    closed_loop = task.get("closed_loop_split")
    if not isinstance(closed_loop, dict):
        raise TypeError(f"Target task {task_id} lacks closed_loop_split")
    values = closed_loop.get("test_init_state_ids")
    validation = closed_loop.get("validation_init_state_ids")
    if not isinstance(values, list) or not isinstance(validation, list):
        raise TypeError(f"Target task {task_id} closed-loop states must be lists")
    result = tuple(values)
    if len(result) != EXPECTED_TEST_STATES or len(set(result)) != len(result):
        raise ValueError(f"Target task {task_id} must have 30 unique test states")
    if set(result) & set(validation):
        raise ValueError(f"Target task {task_id} validation/test states overlap")
    if set(result) | set(validation) != set(range(50)):
        raise ValueError(f"Target task {task_id} closed-loop split must cover states 0..49")
    return result


def load_evaluation_spec(arguments: TargetEvaluationArguments) -> TargetEvaluationSpec:
    split = load_target_split(arguments.split_file)
    if split.task_ids != EXPECTED_TASK_IDS:
        raise ValueError(f"Target evaluation requires task IDs {EXPECTED_TASK_IDS}")
    audit = verify_paraphrase_binding(
        target_split=arguments.split_file,
        paraphrases=arguments.paraphrases,
    )
    paraphrases = load_json_object(arguments.paraphrases)
    protocol = load_json_object(arguments.protocol_config)
    if protocol.get("schema_version") != 1 or protocol.get("suite") != "libero_goal":
        raise ValueError("Unsupported target evaluation protocol config")
    if tuple(protocol.get("task_ids", [])) != EXPECTED_TASK_IDS:
        raise ValueError("Target evaluation protocol task IDs mismatch")
    if tuple(protocol.get("protocols", [])) != PROTOCOLS:
        raise ValueError("Target evaluation protocol list mismatch")
    if protocol.get("target_split_sha256") != split.sha256:
        raise ValueError("Evaluation protocol target split hash mismatch")
    if protocol.get("paraphrases_sha256") != audit["source_sha256"]:
        raise ValueError("Evaluation protocol paraphrases hash mismatch")
    if protocol.get("policy_seed") != arguments.policy_seed:
        raise ValueError(
            f"--policy-seed={arguments.policy_seed} violates the frozen protocol value"
        )
    if protocol.get("env_seed") != arguments.env_seed:
        raise ValueError(f"--env-seed={arguments.env_seed} violates the frozen protocol value")
    if protocol.get("n_action_steps") != arguments.n_action_steps:
        raise ValueError(
            f"--n-action-steps={arguments.n_action_steps} violates the frozen protocol value"
        )
    if protocol.get("episode_length") != 300:
        raise ValueError("Target evaluation protocol episode length must be 300")
    if protocol.get("total_rollouts") != EXPECTED_TOTAL_ROLLOUTS:
        raise ValueError("Target evaluation protocol total_rollouts mismatch")
    for task_id in EXPECTED_TASK_IDS:
        _test_states(split, task_id)
        task = paraphrases.get("tasks", {}).get(str(task_id))
        if not isinstance(task, dict):
            raise TypeError(f"Paraphrase task {task_id} is missing")
        held_out = task.get("held_out_paraphrases")
        if not isinstance(held_out, list) or len(held_out) != HELD_OUT_PARAPHRASE_COUNT:
            raise ValueError(f"Task {task_id} must have five held-out paraphrases")
    prompt_swap = protocol.get("prompt_swap")
    if not isinstance(prompt_swap, dict):
        raise TypeError("Evaluation protocol lacks prompt_swap")
    reference_task = prompt_swap.get("reference_task_id")
    reference_states = prompt_swap.get("reference_init_state_ids")
    if reference_task not in EXPECTED_TASK_IDS or not isinstance(reference_states, list):
        raise ValueError("Prompt Swap reference task/states are invalid")
    if len(reference_states) != 10 or len(set(reference_states)) != 10:
        raise ValueError("Prompt Swap requires ten unique reference states")
    if not set(reference_states).issubset(_test_states(split, int(reference_task))):
        raise ValueError("Prompt Swap states must be held-out states of the reference task")
    if prompt_swap.get("goal_classifier") != GOAL_CLASSIFIER_NAME:
        raise ValueError("Prompt Swap goal classifier mismatch")
    prompts = prompt_swap.get("prompts")
    if not isinstance(prompts, dict) or tuple(sorted(prompts)) != tuple(sorted(GOAL_LABELS)):
        raise ValueError("Prompt Swap prompts must cover stove/cabinet/plate")
    canonical_by_goal = {
        "stove": split.payload["tasks"]["1"]["canonical"],
        "cabinet": split.payload["tasks"]["4"]["canonical"],
        "plate": split.payload["tasks"]["8"]["canonical"],
    }
    if prompts != canonical_by_goal:
        raise ValueError("Prompt Swap prompts do not match frozen canonical instructions")
    return TargetEvaluationSpec(
        split=split,
        paraphrases_payload=paraphrases,
        protocol_payload=protocol,
    )


def build_jobs(
    arguments: TargetEvaluationArguments,
    spec: TargetEvaluationSpec,
) -> list[TargetEvaluationJob]:
    jobs: list[TargetEvaluationJob] = []
    selected_protocols = set(arguments.protocols)
    for task_id in EXPECTED_TASK_IDS:
        states = _test_states(spec.split, task_id)
        seeds = tuple(arguments.env_seed + index for index in range(len(states)))
        canonical = str(spec.split.payload["tasks"][str(task_id)]["canonical"])
        if "canonical" in selected_protocols:
            jobs.append(
                TargetEvaluationJob(
                    job_id=f"canonical/task_{task_id:02d}",
                    protocol="canonical",
                    task_id=task_id,
                    init_state_ids=states,
                    episode_seeds=seeds,
                    instruction=canonical,
                    output_dir=arguments.output_dir / "jobs" / "canonical" / f"task_{task_id:02d}",
                    log_path=arguments.output_dir
                    / "logs"
                    / "canonical"
                    / f"task_{task_id:02d}.log",
                )
            )
        if "held_out_paraphrase" in selected_protocols:
            task_paraphrases = spec.paraphrases_payload["tasks"][str(task_id)][
                "held_out_paraphrases"
            ]
            for prompt_index, instruction in enumerate(task_paraphrases):
                selected_indices = tuple(
                    index
                    for index in range(len(states))
                    if index % HELD_OUT_PARAPHRASE_COUNT == prompt_index
                )
                jobs.append(
                    TargetEvaluationJob(
                        job_id=(
                            f"held_out_paraphrase/task_{task_id:02d}/prompt_{prompt_index:02d}"
                        ),
                        protocol="held_out_paraphrase",
                        task_id=task_id,
                        init_state_ids=tuple(states[index] for index in selected_indices),
                        episode_seeds=tuple(seeds[index] for index in selected_indices),
                        instruction=str(instruction),
                        prompt_index=prompt_index,
                        output_dir=(
                            arguments.output_dir
                            / "jobs"
                            / "held_out_paraphrase"
                            / f"task_{task_id:02d}"
                            / f"prompt_{prompt_index:02d}"
                        ),
                        log_path=(
                            arguments.output_dir
                            / "logs"
                            / "held_out_paraphrase"
                            / f"task_{task_id:02d}_prompt_{prompt_index:02d}.log"
                        ),
                    )
                )
    if "prompt_swap" in selected_protocols:
        prompt_swap = spec.protocol_payload["prompt_swap"]
        reference_task = int(prompt_swap["reference_task_id"])
        reference_states = tuple(int(value) for value in prompt_swap["reference_init_state_ids"])
        state_order = _test_states(spec.split, reference_task)
        reference_seeds = tuple(
            arguments.env_seed + state_order.index(state) for state in reference_states
        )
        for goal in GOAL_LABELS:
            jobs.append(
                TargetEvaluationJob(
                    job_id=f"prompt_swap/{goal}",
                    protocol="prompt_swap",
                    task_id=reference_task,
                    init_state_ids=reference_states,
                    episode_seeds=reference_seeds,
                    instruction=str(prompt_swap["prompts"][goal]),
                    expected_goal=goal,
                    output_dir=arguments.output_dir / "jobs" / "prompt_swap" / goal,
                    log_path=arguments.output_dir / "logs" / "prompt_swap" / f"{goal}.log",
                )
            )
    expected = {
        "canonical": EXPECTED_CANONICAL_ROLLOUTS,
        "held_out_paraphrase": EXPECTED_PARAPHRASE_ROLLOUTS,
        "prompt_swap": EXPECTED_PROMPT_SWAP_ROLLOUTS,
    }
    observed = Counter()
    for job in jobs:
        observed[job.protocol] += job.rollout_count
    for protocol in arguments.protocols:
        if observed[protocol] != expected[protocol]:
            raise RuntimeError(
                f"Protocol {protocol} built {observed[protocol]} rollouts; "
                f"expected {expected[protocol]}"
            )
    return jobs


def build_evaluation_command(
    job: TargetEvaluationJob,
    arguments: TargetEvaluationArguments,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robot_practice.evaluation.libero_smolvla",
        f"--policy-path={portable_path(arguments.policy_path)}",
        "--suite=libero_goal",
        f"--task-id={job.task_id}",
        f"--n-episodes={job.rollout_count}",
        (
            "--num-visualized-episodes="
            f"{min(arguments.num_visualized_episodes_per_job, job.rollout_count)}"
        ),
        f"--n-action-steps={arguments.n_action_steps}",
        f"--seed={arguments.env_seed}",
        f"--policy-seed={arguments.policy_seed}",
        "--init-state-ids=" + ",".join(str(value) for value in job.init_state_ids),
        "--episode-seeds=" + ",".join(str(value) for value in job.episode_seeds),
        f"--instruction={job.instruction}",
        f"--libero-assets-dir={portable_path(arguments.libero_assets_dir)}",
        f"--output-dir={portable_path(job.output_dir)}",
        "--no-open-viewer",
    ]
    if job.expected_goal is not None:
        command.extend(
            [
                f"--goal-classifier={GOAL_CLASSIFIER_NAME}",
                f"--expected-goal={job.expected_goal}",
            ]
        )
    if arguments.max_steps is not None:
        command.append(f"--max-steps={arguments.max_steps}")
    if arguments.overwrite or arguments.resume:
        command.append("--overwrite")
    return command


def _completed_job_metrics(
    job: TargetEvaluationJob,
    arguments: TargetEvaluationArguments,
) -> dict[str, Any] | None:
    metrics_path = job.output_dir / "metrics.json"
    config_path = job.output_dir / "run_config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        return None
    metrics = load_json_object(metrics_path)
    run_config = load_json_object(config_path)
    aggregate = metrics.get("aggregate")
    run_input = run_config.get("input")
    run_policy = run_config.get("policy")
    if not all(isinstance(value, dict) for value in (aggregate, run_input, run_policy)):
        return None
    expected_status = "smoke" if arguments.max_steps is not None else "complete"
    policy_path = run_policy.get("path")
    if not isinstance(policy_path, str):
        return None
    if (
        metrics.get("status") != expected_status
        or run_config.get("status") != expected_status
        or aggregate.get("num_episodes") != job.rollout_count
        or metrics.get("task", {}).get("task_id") != job.task_id
        or resolve_repository_path(policy_path) != arguments.policy_path
        or run_input.get("init_state_ids") != list(job.init_state_ids)
        or run_input.get("seeds") != list(job.episode_seeds)
        or run_input.get("policy_seed") != arguments.policy_seed
        or run_input.get("effective_instruction") != job.instruction
        or run_input.get("goal_classifier")
        != (GOAL_CLASSIFIER_NAME if job.expected_goal is not None else None)
        or run_input.get("expected_goal") != job.expected_goal
    ):
        return None
    episodes = metrics.get("per_episode")
    if not isinstance(episodes, list):
        return None
    if tuple(int(item["init_state_id"]) for item in episodes) != job.init_state_ids:
        return None
    if tuple(int(item["seed"]) for item in episodes) != job.episode_seeds:
        return None
    if job.expected_goal is not None:
        if any(item.get("expected_goal") != job.expected_goal for item in episodes):
            return None
        required_goal_fields = {
            "first_goal_reached",
            "first_goal_step",
            "terminal_goal",
        }
        if any(not required_goal_fields.issubset(item) for item in episodes):
            return None
        if expected_status == "complete" and any(item.get("num_steps") != 300 for item in episodes):
            return None
    return metrics


def _run_job(
    job: TargetEvaluationJob,
    arguments: TargetEvaluationArguments,
    *,
    gpu_pool: queue.Queue[int],
    progress_events: queue.Queue[tuple[TargetEvaluationJob, int, int]],
    active_processes: dict[str, subprocess.Popen[str]],
    active_lock: threading.Lock,
) -> tuple[TargetEvaluationJob, dict[str, Any], int, bool]:
    if arguments.resume:
        existing = _completed_job_metrics(job, arguments)
        if existing is not None:
            return job, existing, -1, True
    gpu = gpu_pool.get()
    try:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_evaluation_command(job, arguments)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with job.log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + shlex.join(command) + "\n")
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
            with active_lock:
                active_processes[job.job_id] = process
            previous = 0
            try:
                if process.stdout is None:
                    raise RuntimeError("Evaluation subprocess did not expose stdout")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    match = ROLLOUT_PROGRESS_PATTERN.search(line)
                    if match is None:
                        continue
                    completed = int(match.group("completed"))
                    total = int(match.group("total"))
                    if total != job.rollout_count or not previous < completed <= total:
                        raise RuntimeError(f"Invalid rollout progress from {job.job_id}")
                    progress_events.put((job, completed - previous, gpu))
                    previous = completed
                returncode = process.wait()
            finally:
                with active_lock:
                    active_processes.pop(job.job_id, None)
        if returncode != 0:
            raise RuntimeError(
                f"Target evaluation job {job.job_id} failed on GPU {gpu}; "
                f"inspect {portable_path(job.log_path)}"
            )
        metrics = _completed_job_metrics(job, arguments)
        if metrics is None:
            raise RuntimeError(f"Target evaluation job {job.job_id} produced mismatched outputs")
        return job, metrics, gpu, False
    finally:
        gpu_pool.put(gpu)


def _terminate_processes(
    active: dict[str, subprocess.Popen[str]],
    lock: threading.Lock,
) -> None:
    with lock:
        processes = list(active.values())
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()


def run_jobs(
    jobs: list[TargetEvaluationJob],
    arguments: TargetEvaluationArguments,
) -> dict[str, dict[str, Any]]:
    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in range(arguments.num_gpus):
        gpu_pool.put(gpu)
    events: queue.Queue[tuple[TargetEvaluationJob, int, int]] = queue.Queue()
    active: dict[str, subprocess.Popen[str]] = {}
    active_lock = threading.Lock()
    observed: dict[str, int] = {}
    collected: dict[str, dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=arguments.num_gpus)
    futures: list[Future[tuple[TargetEvaluationJob, dict[str, Any], int, bool]]] = []
    total = sum(job.rollout_count for job in jobs)
    progress = tqdm(
        total=total,
        desc=f"{arguments.model_label} target protocols",
        unit="rollout",
        dynamic_ncols=True,
    )

    def drain() -> None:
        while True:
            try:
                job, delta, gpu = events.get_nowait()
            except queue.Empty:
                return
            observed[job.job_id] = observed.get(job.job_id, 0) + delta
            progress.update(delta)
            progress.set_postfix(protocol=job.protocol, task=job.task_id, gpu=gpu)

    try:
        for job in jobs:
            futures.append(
                executor.submit(
                    _run_job,
                    job,
                    arguments,
                    gpu_pool=gpu_pool,
                    progress_events=events,
                    active_processes=active,
                    active_lock=active_lock,
                )
            )
        pending = set(futures)
        heartbeat = time.monotonic()
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            drain()
            for future in done:
                job, metrics, gpu, reused = future.result()
                collected[job.job_id] = metrics
                remaining = job.rollout_count - observed.get(job.job_id, 0)
                if remaining < 0:
                    raise RuntimeError(f"Too many progress events for {job.job_id}")
                if remaining:
                    progress.update(remaining)
                    observed[job.job_id] = job.rollout_count
                progress.set_postfix(
                    protocol=job.protocol,
                    task=job.task_id,
                    gpu="cached" if reused else gpu,
                )
            if time.monotonic() - heartbeat >= HEARTBEAT_INTERVAL_S:
                with active_lock:
                    active_count = len(active)
                progress.set_postfix(
                    active_jobs=active_count,
                    completed=f"{progress.n}/{total}",
                    phase="running",
                )
                progress.refresh()
                heartbeat = time.monotonic()
        drain()
    except BaseException:
        for future in futures:
            future.cancel()
        _terminate_processes(active, active_lock)
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        progress.close()
    return collected


def _protocol_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task_id in EXPECTED_TASK_IDS:
        selected = [record for record in records if record["task_id"] == task_id]
        if not selected:
            continue
        successes = sum(bool(record["success"]) for record in selected)
        per_task[str(task_id)] = {
            "num_episodes": len(selected),
            "num_successes": successes,
            "success_rate": successes / len(selected),
        }
    successes = sum(bool(record["success"]) for record in records)
    rates = [value["success_rate"] for value in per_task.values()]
    return {
        "num_episodes": len(records),
        "num_successes": successes,
        "pooled_success_rate": successes / len(records) if records else None,
        "macro_success_rate": float(np.mean(rates)) if rates else None,
        "per_task": per_task,
    }


def _paired_gap(
    canonical: list[dict[str, Any]],
    paraphrase: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_by_case = {
        (record["task_id"], record["init_state_id"]): bool(record["success"])
        for record in canonical
    }
    paraphrase_by_case = {
        (record["task_id"], record["init_state_id"]): bool(record["success"])
        for record in paraphrase
    }
    if canonical_by_case.keys() != paraphrase_by_case.keys():
        raise RuntimeError("Canonical/paraphrase case sets are not paired")
    keys = sorted(canonical_by_case)
    differences = np.asarray(
        [float(canonical_by_case[key]) - float(paraphrase_by_case[key]) for key in keys],
        dtype=np.float64,
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_SAMPLES, len(differences)),
    )
    samples = differences[indices].mean(axis=1)
    return {
        "definition": "canonical_success_rate_minus_held_out_paraphrase_success_rate",
        "num_paired_cases": len(differences),
        "point_estimate": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def collect_episode_records(
    jobs: list[TargetEvaluationJob],
    collected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in jobs:
        metrics = collected[job.job_id]
        for episode in metrics["per_episode"]:
            terminal_goal = episode.get("terminal_goal", episode.get("predicted_goal"))
            records.append(
                {
                    "protocol": job.protocol,
                    "job_id": job.job_id,
                    "task_id": job.task_id,
                    "prompt_index": job.prompt_index,
                    "instruction": job.instruction,
                    "expected_goal": job.expected_goal,
                    # Keep the historical field as a terminal-state alias. The
                    # first reached goal is the primary Prompt Swap measure.
                    "predicted_goal": terminal_goal,
                    "first_goal_reached": episode.get("first_goal_reached"),
                    "first_goal_step": episode.get("first_goal_step"),
                    "terminal_goal": terminal_goal,
                    "episode_index": int(episode["episode_index"]),
                    "seed": int(episode["seed"]),
                    "init_state_id": int(episode["init_state_id"]),
                    "num_steps": int(episode["num_steps"]),
                    "success": bool(episode["success"]),
                    "termination_reason": str(episode["termination_reason"]),
                }
            )
    return records


def summarize_results(
    arguments: TargetEvaluationArguments,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "smoke" if arguments.max_steps is not None else "complete",
        "result_set": "target_protocol_test",
        "model_label": arguments.model_label,
        "policy_path": portable_path(arguments.policy_path),
        "protocols": list(arguments.protocols),
        "total_rollouts": len(records),
    }
    canonical = [record for record in records if record["protocol"] == "canonical"]
    paraphrase = [record for record in records if record["protocol"] == "held_out_paraphrase"]
    prompt_swap = [record for record in records if record["protocol"] == "prompt_swap"]
    if canonical:
        result["canonical"] = _protocol_summary(canonical)
    if paraphrase:
        result["held_out_paraphrase"] = _protocol_summary(paraphrase)
    if canonical and paraphrase:
        result["para_gap"] = _paired_gap(canonical, paraphrase)
    if prompt_swap:

        def confusion(field: str) -> dict[str, dict[str, int]]:
            matrix = {
                goal: {column: 0 for column in (*GOAL_LABELS, FAILED_GOAL)} for goal in GOAL_LABELS
            }
            for record in prompt_swap:
                expected = str(record["expected_goal"])
                if field in record:
                    value = record[field]
                else:
                    value = record.get("predicted_goal")
                predicted = str(value or FAILED_GOAL)
                if expected not in matrix:
                    raise RuntimeError(f"Unknown Prompt Swap expected goal: {expected}")
                if predicted not in matrix[expected]:
                    predicted = FAILED_GOAL
                matrix[expected][predicted] += 1
            return matrix

        first_matrix = confusion("first_goal_reached")
        terminal_matrix = confusion("terminal_goal")
        first_compliant = sum(first_matrix[goal][goal] for goal in GOAL_LABELS)
        terminal_compliant = sum(terminal_matrix[goal][goal] for goal in GOAL_LABELS)
        result["prompt_swap"] = {
            "num_episodes": len(prompt_swap),
            "num_compliant": first_compliant,
            "prompt_compliance": first_compliant / len(prompt_swap),
            "first_goal_num_compliant": first_compliant,
            "first_goal_prompt_compliance": first_compliant / len(prompt_swap),
            "terminal_goal_num_compliant": terminal_compliant,
            "terminal_goal_prompt_compliance": terminal_compliant / len(prompt_swap),
            "num_full_horizon_episodes": sum(
                int(record.get("num_steps", -1)) == 300 for record in prompt_swap
            ),
            "confusion_matrix": first_matrix,
            "terminal_goal_confusion_matrix": terminal_matrix,
        }
    return result


def _write_confusion_csv(path: Path, result: dict[str, Any]) -> None:
    prompt_swap = result.get("prompt_swap")
    if not isinstance(prompt_swap, dict):
        return
    matrix = prompt_swap["confusion_matrix"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input_prompt", *GOAL_LABELS, FAILED_GOAL])
        for goal in GOAL_LABELS:
            writer.writerow(
                [goal, *[matrix[goal][column] for column in (*GOAL_LABELS, FAILED_GOAL)]]
            )


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _write_metrics_markdown(path: Path, result: dict[str, Any]) -> None:
    category = (
        "Smoke / Closed-loop" if result["status"] == "smoke" else "Formal / Closed-loop / Test"
    )
    lines = [
        f"# SmolVLA target protocols: {result['model_label']}",
        "",
        f"结果类别：{category}。",
        "",
        f"Policy: `{result['policy_path']}`.",
        "",
        "| Protocol | Rollouts | Successes | Primary rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, title in (
        ("canonical", "Canonical"),
        ("held_out_paraphrase", "Held-out paraphrase"),
    ):
        value = result.get(key)
        if isinstance(value, dict):
            lines.append(
                f"| {title} | {value['num_episodes']} | {value['num_successes']} | "
                f"{_format_rate(value['macro_success_rate'])} |"
            )
    if isinstance(result.get("prompt_swap"), dict):
        swap = result["prompt_swap"]
        lines.append(
            f"| Prompt Swap | {swap['num_episodes']} | {swap['num_compliant']} | "
            f"{swap['first_goal_prompt_compliance']:.4f} |"
        )
        lines.append(
            f"| Prompt Swap (terminal hold) | {swap['num_episodes']} | "
            f"{swap['terminal_goal_num_compliant']} | "
            f"{swap['terminal_goal_prompt_compliance']:.4f} |"
        )
    if isinstance(result.get("para_gap"), dict):
        gap = result["para_gap"]
        lines.extend(
            [
                "",
                (
                    "Para Gap (canonical - paraphrase): "
                    f"`{gap['point_estimate']:.4f}`; paired bootstrap 95% CI "
                    f"`[{gap['ci95'][0]:.4f}, {gap['ci95'][1]:.4f}]`."
                ),
            ]
        )
    if isinstance(result.get("prompt_swap"), dict):
        matrix = result["prompt_swap"]["confusion_matrix"]
        lines.extend(
            [
                "",
                "## Prompt–Goal confusion matrix",
                "",
                "| Input prompt | Stove | Cabinet | Plate | Failed |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for goal in GOAL_LABELS:
            row = matrix[goal]
            lines.append(
                f"| {goal} | {row['stove']} | {row['cabinet']} | {row['plate']} | {row['failed']} |"
            )
        lines.extend(
            [
                "",
                (
                    "Primary Prompt Swap scoring uses the first reached goal; "
                    f"terminal-state compliance is "
                    f"`{result['prompt_swap']['terminal_goal_prompt_compliance']:.4f}` "
                    "and is a hold-at-end diagnostic."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Test results are descriptive only and must not be used for checkpoint selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _contract_payload(
    arguments: TargetEvaluationArguments,
    jobs: list[TargetEvaluationJob],
) -> dict[str, Any]:
    return {
        "model_label": arguments.model_label,
        "policy_path": portable_path(arguments.policy_path),
        "policy_config_sha256": sha256_file(arguments.policy_path / "config.json"),
        "split_file": portable_path(arguments.split_file),
        "split_sha256": sha256_file(arguments.split_file),
        "paraphrases": portable_path(arguments.paraphrases),
        "paraphrases_sha256": sha256_file(arguments.paraphrases),
        "protocol_config": portable_path(arguments.protocol_config),
        "protocol_config_sha256": sha256_file(arguments.protocol_config),
        "protocols": list(arguments.protocols),
        "num_gpus": arguments.num_gpus,
        "env_seed": arguments.env_seed,
        "policy_seed": arguments.policy_seed,
        "n_action_steps": arguments.n_action_steps,
        "num_visualized_episodes_per_job": arguments.num_visualized_episodes_per_job,
        "max_steps": arguments.max_steps,
        "jobs": [
            {
                **asdict(job),
                "output_dir": portable_path(job.output_dir),
                "log_path": portable_path(job.log_path),
            }
            for job in jobs
        ],
        "total_rollouts": sum(job.rollout_count for job in jobs),
    }


def _prepare_output(
    arguments: TargetEvaluationArguments,
    contract: dict[str, Any],
) -> Path:
    manifest_path = arguments.output_dir / "run_manifest.json"
    contract_hash = _canonical_hash(contract)
    if arguments.output_dir.exists():
        if arguments.overwrite:
            shutil.rmtree(arguments.output_dir)
        elif arguments.resume:
            if not manifest_path.is_file():
                raise FileNotFoundError("Resume output lacks run_manifest.json")
            existing = load_json_object(manifest_path)
            if existing.get("contract_sha256") != contract_hash:
                raise ValueError("Resume arguments do not match the frozen evaluation contract")
            return manifest_path
        else:
            raise FileExistsError(
                f"Output directory exists: {portable_path(arguments.output_dir)}; "
                "use --resume or --overwrite"
            )
    arguments.output_dir.mkdir(parents=True)
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "status": "running",
            "result_set": "target_protocol_test",
            "contract_sha256": contract_hash,
            "contract": contract,
        },
    )
    return manifest_path


def evaluate_target_protocols(arguments: TargetEvaluationArguments) -> dict[str, Any]:
    validate_libero_assets_dir(arguments.libero_assets_dir)
    spec = load_evaluation_spec(arguments)
    jobs = build_jobs(arguments, spec)
    contract = _contract_payload(arguments, jobs)
    if arguments.print_command:
        payload = {
            "model_label": arguments.model_label,
            "policy_path": portable_path(arguments.policy_path),
            "protocols": list(arguments.protocols),
            "num_jobs": len(jobs),
            "total_rollouts": sum(job.rollout_count for job in jobs),
            "num_gpus": arguments.num_gpus,
            "output_dir": portable_path(arguments.output_dir),
            "commands": [shlex.join(build_evaluation_command(job, arguments)) for job in jobs],
        }
        print(json.dumps(payload, indent=2))
        return payload
    manifest_path = _prepare_output(arguments, contract)
    collected = run_jobs(jobs, arguments)
    records = collect_episode_records(jobs, collected)
    result = summarize_results(arguments, records)
    write_json_atomic(arguments.output_dir / "metrics.json", result)
    _write_metrics_markdown(arguments.output_dir / "metrics.md", result)
    _write_confusion_csv(arguments.output_dir / "prompt_goal_confusion.csv", result)
    table = pa.Table.from_pylist(records)
    write_parquet_atomic(arguments.output_dir / "episodes.parquet", table)
    manifest = load_json_object(manifest_path)
    manifest["status"] = result["status"]
    manifest["metrics_sha256"] = sha256_file(arguments.output_dir / "metrics.json")
    write_json_atomic(manifest_path, manifest)
    LOGGER.info("Target protocol evaluation complete: %s", arguments.output_dir / "metrics.md")
    return result


def _protocol_list(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("--protocols must contain unique protocol names")
    unknown = [value for value in values if value not in PROTOCOLS]
    if unknown:
        raise argparse.ArgumentTypeError("Unknown protocols: " + ", ".join(unknown))
    return tuple(protocol for protocol in PROTOCOLS if protocol in values)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/target_split.json"),
    )
    parser.add_argument(
        "--paraphrases",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/paraphrases.json"),
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/target_eval_protocol.json"),
    )
    parser.add_argument("--protocols", type=_protocol_list, default=PROTOCOLS)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--policy-seed", type=int, default=2000)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--num-visualized-episodes-per-job", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--libero-assets-dir",
        type=Path,
        default=Path("datasets/raw/libero_assets"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser


def _resolve_arguments(namespace: argparse.Namespace) -> TargetEvaluationArguments:
    policy_path = _validate_policy(resolve_repository_path(namespace.policy_path))
    model_label = str(namespace.model_label)
    if _SAFE_LABEL.fullmatch(model_label) is None:
        raise ValueError(
            "--model-label must contain only letters, digits, dot, underscore, and hyphen"
        )
    if namespace.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive")
    if namespace.env_seed < 0 or namespace.policy_seed < 0:
        raise ValueError("--env-seed and --policy-seed must be non-negative")
    if namespace.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    if namespace.num_visualized_episodes_per_job < 0:
        raise ValueError("--num-visualized-episodes-per-job must be non-negative")
    if namespace.max_steps is not None and not 1 <= namespace.max_steps <= 300:
        raise ValueError("--max-steps must be between 1 and 300")
    if namespace.resume and namespace.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    output_dir = resolve_repository_path(namespace.output_dir)
    if output_dir == policy_path or policy_path in output_dir.parents:
        raise ValueError("--output-dir must not be inside --policy-path")
    return TargetEvaluationArguments(
        policy_path=policy_path,
        model_label=model_label,
        split_file=resolve_repository_path(namespace.split_file),
        paraphrases=resolve_repository_path(namespace.paraphrases),
        protocol_config=resolve_repository_path(namespace.protocol_config),
        protocols=namespace.protocols,
        num_gpus=namespace.num_gpus,
        env_seed=namespace.env_seed,
        policy_seed=namespace.policy_seed,
        n_action_steps=namespace.n_action_steps,
        num_visualized_episodes_per_job=namespace.num_visualized_episodes_per_job,
        max_steps=namespace.max_steps,
        libero_assets_dir=resolve_repository_path(namespace.libero_assets_dir),
        output_dir=output_dir,
        resume=namespace.resume,
        overwrite=namespace.overwrite,
        print_command=namespace.print_command,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    namespace = build_argument_parser().parse_args()
    evaluate_target_protocols(_resolve_arguments(namespace))


if __name__ == "__main__":
    main()
