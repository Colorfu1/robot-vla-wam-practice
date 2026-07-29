"""E3 paired DDIM/DDPM sampling-step, success, and latency evaluation."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any

import matplotlib
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_IMAGES

from .config import (
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_output_root,
    load_named_config,
    practice_root,
)
from .dataset_split import load_split
from .latency import benchmark_latency
from .noise import initial_noise_seed, make_initial_noise, reverse_noise_seed
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import (
    controlled_sample,
    load_policy_with_processors,
    replace_scheduler,
)
from .provenance import (
    collect_runtime_provenance,
    sha256_file,
    state_dict_hashes,
    write_json_atomic,
)
from .rerun_visualization import E3SceneRecording
from .rollout import RolloutResult, make_pusht_env, run_rollout, save_rollout_video
from .statistics import wilson_interval

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_E3_CONFIG = load_named_config("e3_sampling_steps.json")


def _parse_steps(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed) or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{name} must contain unique positive integers")
    return parsed


def _conditions(
    schedulers: str,
    ddim_steps: str,
    ddpm_steps: str,
) -> tuple[tuple[str, int], ...]:
    selected = tuple(item.strip().upper() for item in schedulers.split(",") if item.strip())
    if not selected or any(item not in {"DDIM", "DDPM"} for item in selected):
        raise ValueError("schedulers must be a comma-separated subset of ddim,ddpm")
    if len(selected) != len(set(selected)):
        raise ValueError("schedulers must not contain duplicates")
    resolved: list[tuple[str, int]] = []
    if "DDIM" in selected:
        resolved.extend(("DDIM", step) for step in _parse_steps(ddim_steps, name="ddim-steps"))
    if "DDPM" in selected:
        resolved.extend(("DDPM", step) for step in _parse_steps(ddpm_steps, name="ddpm-steps"))
    return tuple(resolved)


def _condition_id(scheduler: str, steps: int) -> str:
    return f"{scheduler.lower()}_{steps:03d}"


def partition_environment_seeds(
    environment_seeds: Sequence[int],
    *,
    num_gpus: int,
) -> tuple[tuple[int, ...], ...]:
    """Round-robin E3 environments across explicit GPU workers."""

    seeds = tuple(int(seed) for seed in environment_seeds)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    if num_gpus > len(seeds):
        raise ValueError("num_gpus cannot exceed the number of environment seeds")
    return tuple(seeds[gpu_index::num_gpus] for gpu_index in range(num_gpus))


def _validate_num_gpus(*, num_gpus: int, num_environment_seeds: int) -> None:
    partition_environment_seeds(
        tuple(range(num_environment_seeds)),
        num_gpus=num_gpus,
    )
    if num_gpus == 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU E3 evaluation requires CUDA")
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise RuntimeError(
            f"num_gpus={num_gpus} requested, but only {available} CUDA device(s) are visible"
        )


@dataclass(frozen=True)
class SamplingWorkerConfig:
    gpu_index: int
    diffusion_policy_path: str
    environment_seeds: tuple[int, ...]
    conditions: tuple[tuple[str, int], ...]
    policy_seed: int
    episode_length: int
    output_root: str
    video_environment_seeds: tuple[int, ...]


@dataclass(frozen=True)
class SamplingWorkerEvent:
    kind: str
    gpu_index: int
    result: RolloutResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class LatencyWorkerConfig:
    gpu_index: int
    diffusion_policy_path: str
    conditions: tuple[tuple[str, int], ...]
    environment_seed: int
    policy_seed: int
    checkpoint_id: str
    total_rollouts: int
    code_commit: str | None
    episode_length: int
    warmup_iterations: int
    measurement_iterations: int


@dataclass(frozen=True)
class LatencyWorkerEvent:
    kind: str
    gpu_index: int
    condition_id: str | None = None
    latency_kind: str | None = None
    rows: list[dict[str, Any]] | None = None
    summaries: dict[str, dict[str, Any]] | None = None
    parameter_hashes: dict[str, str] | None = None
    n_action_steps: int | None = None
    device_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ParallelLatencyMetadata:
    num_gpus: int
    measurement_iterations_per_gpu: tuple[int, ...]
    device_names: tuple[str, ...]
    parameter_hashes: dict[str, str]
    n_action_steps: int


def partition_latency_iterations(
    measurement_iterations: int,
    *,
    num_gpus: int,
) -> tuple[int, ...]:
    """Split the aggregate per-condition latency sample budget across GPUs."""

    if (
        isinstance(measurement_iterations, bool)
        or not isinstance(measurement_iterations, int)
        or measurement_iterations < 1
    ):
        raise ValueError("measurement_iterations must be a positive integer")
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    worker_count = min(measurement_iterations, num_gpus)
    quotient, remainder = divmod(measurement_iterations, worker_count)
    return tuple(
        quotient + (1 if gpu_index < remainder else 0) for gpu_index in range(worker_count)
    )


def _render_progress_bar(*, completed: int, total: int, width: int = 30) -> None:
    if total < 1 or not 0 <= completed <= total:
        raise ValueError("Invalid progress-bar state")
    filled = width * completed // total
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if completed == total else "\r"
    print(
        f"E3 latency [{bar}] {completed}/{total}",
        end=ending,
        flush=True,
    )


def _checkpoint_sha256(path: str | Path) -> str:
    weights = Path(path).expanduser().resolve() / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint weights are missing: {weights}")
    return sha256_file(weights)


def _load_e3_policy(
    diffusion_policy_path: str | Path,
    *,
    device: str | None,
    conditions: Sequence[tuple[str, int]],
) -> Any:
    loaded = load_policy_with_processors(
        diffusion_policy_path,
        local_files_only=True,
        device=device,
    )
    if not isinstance(loaded.policy, DiffusionPolicy):
        raise TypeError("diffusion-policy must contain policy.type=diffusion")
    if loaded.config.n_action_steps != 8:
        raise ValueError("E3 requires n_action_steps=8")
    if any(steps > loaded.config.num_train_timesteps for _, steps in conditions):
        raise ValueError(
            f"Inference steps cannot exceed num_train_timesteps={loaded.config.num_train_timesteps}"
        )
    return loaded


def _execute_environment_shard(
    *,
    loaded: Any,
    conditions: Sequence[tuple[str, int]],
    environment_seeds: Sequence[int],
    policy_seed: int,
    episode_length: int,
    output_root: Path,
    video_environment_seeds: set[int],
    result_callback: Callable[[RolloutResult], None] | None = None,
) -> list[RolloutResult]:
    parameter_hashes = state_dict_hashes(loaded.policy.state_dict())
    condition_ids = tuple(_condition_id(scheduler, steps) for scheduler, steps in conditions)
    results: list[RolloutResult] = []
    for environment_seed in environment_seeds:
        recorder = E3SceneRecording(
            output_root / "recordings" / f"env_{environment_seed:04d}.rrd",
            environment_seed=environment_seed,
            condition_ids=condition_ids,
        )
        complete = False
        try:
            for scheduler, steps in conditions:
                condition_id = _condition_id(scheduler, steps)
                replace_scheduler(
                    loaded.policy,
                    scheduler_name=scheduler,
                    num_inference_steps=steps,
                )
                if state_dict_hashes(loaded.policy.state_dict()) != parameter_hashes:
                    raise RuntimeError(f"{condition_id} changed model parameter hashes")
                result = run_rollout(
                    policy=loaded.policy,
                    preprocessor=loaded.preprocessor,
                    postprocessor=loaded.postprocessor,
                    environment_seed=environment_seed,
                    policy_seed=policy_seed,
                    record_frames=True,
                    episode_length=episode_length,
                    condition_id=condition_id,
                )
                recorder.add(result)
                if environment_seed in video_environment_seeds:
                    save_rollout_video(
                        result,
                        output_root / "videos" / f"{condition_id}_env_{environment_seed:04d}.mp4",
                    )
                lightweight = replace(result, frames=())
                results.append(lightweight)
                if result_callback is not None:
                    result_callback(lightweight)
            complete = True
        finally:
            recorder.close(validate_complete=complete)
    return results


def _rollout_worker(
    config: SamplingWorkerConfig,
    event_queue: Any,
) -> None:
    try:
        torch.cuda.set_device(config.gpu_index)
        loaded = _load_e3_policy(
            config.diffusion_policy_path,
            device=f"cuda:{config.gpu_index}",
            conditions=config.conditions,
        )

        def publish(result: RolloutResult) -> None:
            event_queue.put(
                SamplingWorkerEvent(
                    kind="result",
                    gpu_index=config.gpu_index,
                    result=result,
                )
            )

        _execute_environment_shard(
            loaded=loaded,
            conditions=config.conditions,
            environment_seeds=config.environment_seeds,
            policy_seed=config.policy_seed,
            episode_length=config.episode_length,
            output_root=Path(config.output_root),
            video_environment_seeds=set(config.video_environment_seeds),
            result_callback=publish,
        )
        event_queue.put(
            SamplingWorkerEvent(
                kind="done",
                gpu_index=config.gpu_index,
            )
        )
    except (Exception, KeyboardInterrupt):
        event_queue.put(
            SamplingWorkerEvent(
                kind="error",
                gpu_index=config.gpu_index,
                error=traceback.format_exc(),
            )
        )
        raise


def _execute_parallel_rollouts(
    *,
    diffusion_policy_path: str | Path,
    conditions: Sequence[tuple[str, int]],
    environment_seeds: Sequence[int],
    policy_seed: int,
    episode_length: int,
    output_root: Path,
    video_environment_seeds: set[int],
    num_gpus: int,
) -> tuple[list[RolloutResult], tuple[tuple[int, ...], ...]]:
    shards = partition_environment_seeds(environment_seeds, num_gpus=num_gpus)
    expected_results = len(conditions) * len(environment_seeds)
    context = mp.get_context("spawn")
    event_queue = context.Queue()
    workers = []
    results_by_key: dict[tuple[str, int], RolloutResult] = {}
    completed_workers: set[int] = set()
    condition_order = {
        _condition_id(scheduler, steps): ordinal
        for ordinal, (scheduler, steps) in enumerate(conditions)
    }
    try:
        for gpu_index, shard in enumerate(shards):
            config = SamplingWorkerConfig(
                gpu_index=gpu_index,
                diffusion_policy_path=str(diffusion_policy_path),
                environment_seeds=shard,
                conditions=tuple(conditions),
                policy_seed=policy_seed,
                episode_length=episode_length,
                output_root=str(output_root),
                video_environment_seeds=tuple(sorted(video_environment_seeds)),
            )
            worker = context.Process(
                target=_rollout_worker,
                args=(config, event_queue),
                name=f"pusht-e3-gpu-{gpu_index}",
            )
            worker.start()
            workers.append(worker)

        while len(results_by_key) < expected_results or len(completed_workers) < num_gpus:
            try:
                event: SamplingWorkerEvent = event_queue.get(timeout=1)
            except Empty:
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    details = ", ".join(
                        f"{worker.name} exitcode={worker.exitcode}" for worker in failed
                    )
                    raise RuntimeError(f"E3 rollout worker failed: {details}")
                exited_without_done = [
                    worker
                    for gpu_index, worker in enumerate(workers)
                    if worker.exitcode == 0 and gpu_index not in completed_workers
                ]
                if exited_without_done:
                    details = ", ".join(worker.name for worker in exited_without_done)
                    raise RuntimeError(
                        f"E3 rollout worker exited without a completion event: {details}"
                    )
                if (
                    len(results_by_key) < expected_results
                    and workers
                    and all(worker.exitcode is not None for worker in workers)
                ):
                    raise RuntimeError(
                        "E3 rollout workers completed with only "
                        f"{len(results_by_key)}/{expected_results} results"
                    )
                continue
            if event.kind == "error":
                raise RuntimeError(
                    f"E3 rollout worker on GPU {event.gpu_index} failed:\n{event.error}"
                )
            if event.kind == "done":
                if event.gpu_index in completed_workers:
                    raise RuntimeError(
                        f"Duplicate E3 worker completion event from GPU {event.gpu_index}"
                    )
                completed_workers.add(event.gpu_index)
                continue
            if event.kind != "result" or event.result is None:
                raise RuntimeError(f"Unexpected E3 worker event: {event.kind}")
            result = event.result
            key = (result.condition_id, result.environment_seed)
            if key in results_by_key:
                raise RuntimeError(f"Duplicate E3 rollout result from workers: {key}")
            results_by_key[key] = result
            print(
                f"E3 multi-GPU rollout progress: {len(results_by_key)}/{expected_results}",
                flush=True,
            )

        for worker in workers:
            worker.join(timeout=30)
        alive = [worker for worker in workers if worker.is_alive()]
        if alive:
            details = ", ".join(worker.name for worker in alive)
            raise RuntimeError(f"E3 rollout workers did not shut down: {details}")
        failed = [worker for worker in workers if worker.exitcode != 0]
        if failed:
            details = ", ".join(f"{worker.name} exitcode={worker.exitcode}" for worker in failed)
            raise RuntimeError(f"E3 rollout worker shutdown failed: {details}")
    except (Exception, KeyboardInterrupt):
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        raise
    finally:
        event_queue.close()
        event_queue.join_thread()

    ordered = sorted(
        results_by_key.values(),
        key=lambda result: (
            condition_order[result.condition_id],
            result.environment_seed,
        ),
    )
    return ordered, shards


def _freeze_history(
    policy: DiffusionPolicy,
    preprocessor: Any,
    observation: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    processed = preprocessor(preprocess_observation(observation))
    history: dict[str, torch.Tensor] = {}
    for key in policy.config.input_features:
        if key not in processed:
            raise KeyError(f"Preprocessor output is missing {key!r}")
        value = processed[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Preprocessor output {key!r} must be a tensor")
        history[key] = value.unsqueeze(1).repeat(
            1,
            policy.config.n_obs_steps,
            *(1 for _ in value.shape[1:]),
        )
    if policy.config.image_features:
        history[OBS_IMAGES] = torch.stack(
            [history[key] for key in policy.config.image_features],
            dim=-4,
        )
    return history


def _rollout_row(
    result: Any,
    *,
    checkpoint_id: str,
    train_seed: int,
    total_rollouts: int,
    code_commit: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": "E3",
        "condition_id": result.condition_id,
        "model": "diffusion",
        "checkpoint_id": checkpoint_id,
        "train_seed": train_seed,
        "environment_seed": result.environment_seed,
        "policy_seed": result.policy_seed,
        "replan_index": None,
        "initial_noise_seed": None,
        "initial_noise_sha256": None,
        "reverse_noise_seed": None,
        "reverse_generator_state_sha256": None,
        "scheduler": result.scheduler,
        "num_inference_steps": result.num_inference_steps,
        "num_action_samples": 1,
        "num_rollouts": total_rollouts,
        "dataset_revision": DATASET_REVISION,
        "code_commit": code_commit,
        "success": result.success,
        "sum_reward": result.sum_reward,
        "max_reward": result.max_reward,
        "final_reward": result.final_reward,
        "episode_length": result.episode_length,
        "agent_path": result.agent_path.tolist(),
        "block_path": result.block_path.tolist(),
        "action_targets": result.action_targets.tolist(),
        "rewards": result.rewards.tolist(),
    }


def _replan_rows(
    result: Any,
    *,
    checkpoint_id: str,
    train_seed: int,
    total_rollouts: int,
    code_commit: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "E3",
            "condition_id": result.condition_id,
            "model": "diffusion",
            "checkpoint_id": checkpoint_id,
            "train_seed": train_seed,
            "environment_seed": result.environment_seed,
            "policy_seed": result.policy_seed,
            "replan_index": replan.replan_index,
            "initial_noise_seed": replan.initial_noise_seed,
            "initial_noise_sha256": replan.initial_noise_sha256,
            "reverse_noise_seed": replan.reverse_noise_seed,
            "reverse_generator_state_sha256": replan.reverse_generator_state_sha256,
            "scheduler": replan.scheduler,
            "num_inference_steps": replan.num_inference_steps,
            "num_action_samples": 1,
            "num_rollouts": total_rollouts,
            "dataset_revision": DATASET_REVISION,
            "code_commit": code_commit,
            "chunk_latency_ms": replan.latency_ms,
            "normalized_chunk": replan.normalized_chunk.tolist(),
        }
        for replan in result.replans
    ]


def _plot_results(
    output_dir: Path,
    condition_rows: Sequence[dict[str, Any]],
    latency_rows: Sequence[dict[str, Any]],
) -> None:
    labels = [str(row["condition_id"]) for row in condition_rows]
    successes = [float(row["success_rate"]) for row in condition_rows]
    p95_values = [float(row["end_to_end_p95_ms"]) for row in condition_rows]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(labels, successes, marker="o")
    axis.set(ylabel="success rate", xlabel="scheduler / inference steps", ylim=(0, 1.05))
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_dir / "sampling_steps_vs_success.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(p95_values, successes, s=70)
    for label, latency, success in zip(labels, p95_values, successes, strict=True):
        axis.annotate(label, (latency, success), xytext=(5, 4), textcoords="offset points")
    axis.set(
        xlabel="p95 end-to-end chunk latency (ms)",
        ylabel="success rate",
        ylim=(0, 1.05),
    )
    figure.tight_layout()
    figure.savefig(output_dir / "success_latency_pareto.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for condition_id in labels:
        values = [
            float(row["latency_ms"])
            for row in latency_rows
            if row["condition_id"] == condition_id and row["latency_kind"] == "end_to_end"
        ]
        axis.hist(values, bins=40, alpha=0.35, label=condition_id)
    axis.set(xlabel="end-to-end chunk latency (ms)", ylabel="count")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "latency_distribution.png", dpi=160)
    plt.close(figure)


def _benchmark_sampling_latency(
    *,
    loaded: Any,
    conditions: Sequence[tuple[str, int]],
    environment_seed: int,
    policy_seed: int,
    checkpoint_id: str,
    total_rollouts: int,
    code_commit: str | None,
    episode_length: int,
    warmup_iterations: int,
    measurement_iterations: int,
    latency_gpu_index: int | None = None,
    latency_device_name: str | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    environment = make_pusht_env(episode_length=episode_length)
    raw_observation, _ = environment.reset(seed=environment_seed)
    environment.close()
    device = get_device_from_parameters(loaded.policy)
    dtype = get_dtype_from_parameters(loaded.policy)
    first_seed = initial_noise_seed(
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        replan_index=0,
    )
    second_seed = reverse_noise_seed(
        environment_seed=environment_seed,
        replan_index=0,
    )
    explicit = make_initial_noise(
        seed=first_seed,
        batch_size=1,
        device=device,
        dtype=dtype,
        horizon=loaded.config.horizon,
        action_dim=loaded.config.action_feature.shape[0],
    )
    parameter_hashes = state_dict_hashes(loaded.policy.state_dict())
    latency_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for scheduler, steps in conditions:
        condition_id = _condition_id(scheduler, steps)
        replace_scheduler(
            loaded.policy,
            scheduler_name=scheduler,
            num_inference_steps=steps,
        )
        if state_dict_hashes(loaded.policy.state_dict()) != parameter_hashes:
            raise RuntimeError(f"{condition_id} changed model parameter hashes")
        frozen_history = _freeze_history(
            loaded.policy,
            loaded.preprocessor,
            raw_observation,
        )

        def model_only(
            frozen_batch: dict[str, torch.Tensor] = frozen_history,
            initial_noise: torch.Tensor = explicit.tensor,
            reverse_seed_value: int = second_seed,
        ) -> object:
            return controlled_sample(
                loaded.policy,
                frozen_batch,
                initial_noise=initial_noise,
                reverse_seed=reverse_seed_value,
            )

        def end_to_end(
            raw: dict[str, np.ndarray] = raw_observation,
            initial_noise: torch.Tensor = explicit.tensor,
            reverse_seed_value: int = second_seed,
        ) -> object:
            history = _freeze_history(
                loaded.policy,
                loaded.preprocessor,
                raw,
            )
            sample = controlled_sample(
                loaded.policy,
                history,
                initial_noise=initial_noise,
                reverse_seed=reverse_seed_value,
            )
            return loaded.postprocessor(sample.actions)

        with torch.inference_mode():
            model_summary, model_samples = benchmark_latency(
                model_only,
                device=device,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            if progress_callback is not None:
                progress_callback(condition_id, "model_only")
            end_summary, end_samples = benchmark_latency(
                end_to_end,
                device=device,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            if progress_callback is not None:
                progress_callback(condition_id, "end_to_end")
        for kind, samples, summary in (
            ("model_only", model_samples, model_summary),
            ("end_to_end", end_samples, end_summary),
        ):
            latency_rows.extend(
                {
                    "experiment_id": "E3",
                    "condition_id": condition_id,
                    "scheduler": scheduler,
                    "num_inference_steps": steps,
                    "latency_kind": kind,
                    "iteration": index,
                    "latency_ms": float(value),
                    "latency_gpu_index": latency_gpu_index,
                    "latency_device_name": latency_device_name,
                    "peak_memory_bytes": summary.peak_memory_bytes,
                    "checkpoint_id": checkpoint_id,
                    "environment_seed": None,
                    "policy_seed": policy_seed,
                    "replan_index": None,
                    "initial_noise_seed": first_seed,
                    "reverse_noise_seed": second_seed,
                    "num_action_samples": 1,
                    "num_rollouts": total_rollouts,
                    "dataset_revision": DATASET_REVISION,
                    "code_commit": code_commit,
                }
                for index, value in enumerate(samples)
            )
        summaries[condition_id] = {
            **{f"model_only_{key}": value for key, value in model_summary.to_dict().items()},
            **{f"end_to_end_{key}": value for key, value in end_summary.to_dict().items()},
            "amortized_end_to_end_p95_ms": (end_summary.p95_ms / loaded.config.n_action_steps),
        }
    return latency_rows, summaries


def _latency_worker(
    config: LatencyWorkerConfig,
    event_queue: Any,
) -> None:
    try:
        torch.cuda.set_device(config.gpu_index)
        loaded = _load_e3_policy(
            config.diffusion_policy_path,
            device=f"cuda:{config.gpu_index}",
            conditions=config.conditions,
        )
        offset = config.gpu_index % len(config.conditions)
        ordered_conditions = config.conditions[offset:] + config.conditions[:offset]

        def publish_progress(condition_id: str, latency_kind: str) -> None:
            event_queue.put(
                LatencyWorkerEvent(
                    kind="progress",
                    gpu_index=config.gpu_index,
                    condition_id=condition_id,
                    latency_kind=latency_kind,
                )
            )

        rows, summaries = _benchmark_sampling_latency(
            loaded=loaded,
            conditions=ordered_conditions,
            environment_seed=config.environment_seed,
            policy_seed=config.policy_seed,
            checkpoint_id=config.checkpoint_id,
            total_rollouts=config.total_rollouts,
            code_commit=config.code_commit,
            episode_length=config.episode_length,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            latency_gpu_index=config.gpu_index,
            latency_device_name=torch.cuda.get_device_name(config.gpu_index),
            progress_callback=publish_progress,
        )
        event_queue.put(
            LatencyWorkerEvent(
                kind="result",
                gpu_index=config.gpu_index,
                rows=rows,
                summaries=summaries,
                parameter_hashes=state_dict_hashes(loaded.policy.state_dict()),
                n_action_steps=int(loaded.config.n_action_steps),
                device_name=torch.cuda.get_device_name(config.gpu_index),
            )
        )
    except (Exception, KeyboardInterrupt):
        event_queue.put(
            LatencyWorkerEvent(
                kind="error",
                gpu_index=config.gpu_index,
                error=traceback.format_exc(),
            )
        )
        raise


def _aggregate_parallel_latency_rows(
    worker_rows: Sequence[dict[str, Any]],
    *,
    conditions: Sequence[tuple[str, int]],
    measurement_iterations: int,
    n_action_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    expected_rows = len(conditions) * 2 * measurement_iterations
    if len(worker_rows) != expected_rows:
        raise RuntimeError(f"E3 latency workers produced {len(worker_rows)}/{expected_rows} rows")
    ordered_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for scheduler, steps in conditions:
        condition_id = _condition_id(scheduler, steps)
        condition_summary: dict[str, Any] = {}
        for latency_kind in ("model_only", "end_to_end"):
            selected = sorted(
                (
                    row
                    for row in worker_rows
                    if row["condition_id"] == condition_id and row["latency_kind"] == latency_kind
                ),
                key=lambda row: (
                    int(row["latency_gpu_index"]),
                    int(row["iteration"]),
                ),
            )
            if len(selected) != measurement_iterations:
                raise RuntimeError(
                    f"{condition_id}/{latency_kind} has "
                    f"{len(selected)}/{measurement_iterations} latency samples"
                )
            reindexed = [{**row, "iteration": iteration} for iteration, row in enumerate(selected)]
            ordered_rows.extend(reindexed)
            values = np.asarray(
                [float(row["latency_ms"]) for row in reindexed],
                dtype=np.float64,
            )
            peak_values = [
                int(row["peak_memory_bytes"])
                for row in reindexed
                if row["peak_memory_bytes"] is not None
            ]
            prefix = f"{latency_kind}_"
            condition_summary.update(
                {
                    f"{prefix}iterations": len(values),
                    f"{prefix}mean_ms": float(np.mean(values)),
                    f"{prefix}p50_ms": float(np.percentile(values, 50)),
                    f"{prefix}p90_ms": float(np.percentile(values, 90)),
                    f"{prefix}p95_ms": float(np.percentile(values, 95)),
                    f"{prefix}p99_ms": float(np.percentile(values, 99)),
                    f"{prefix}std_ms": float(np.std(values)),
                    f"{prefix}peak_memory_bytes": (max(peak_values) if peak_values else None),
                }
            )
        condition_summary["amortized_end_to_end_p95_ms"] = (
            float(condition_summary["end_to_end_p95_ms"]) / n_action_steps
        )
        summaries[condition_id] = condition_summary
    return ordered_rows, summaries


def _execute_parallel_latency(
    *,
    diffusion_policy_path: str | Path,
    conditions: Sequence[tuple[str, int]],
    environment_seed: int,
    policy_seed: int,
    checkpoint_id: str,
    total_rollouts: int,
    code_commit: str | None,
    episode_length: int,
    warmup_iterations: int,
    measurement_iterations: int,
    output_root: Path,
    num_gpus: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    ParallelLatencyMetadata,
]:
    iteration_counts = partition_latency_iterations(
        measurement_iterations,
        num_gpus=num_gpus,
    )
    worker_count = len(iteration_counts)
    context = mp.get_context("spawn")
    event_queue = context.Queue()
    workers = []
    results_by_gpu: dict[int, LatencyWorkerEvent] = {}
    completed_progress: set[tuple[int, str, str]] = set()
    expected_progress = worker_count * len(conditions) * 2
    progress_dir = output_root / "latency_progress"
    progress_dir.mkdir()
    _render_progress_bar(completed=0, total=expected_progress)
    try:
        for gpu_index, worker_iterations in enumerate(iteration_counts):
            config = LatencyWorkerConfig(
                gpu_index=gpu_index,
                diffusion_policy_path=str(diffusion_policy_path),
                conditions=tuple(conditions),
                environment_seed=environment_seed,
                policy_seed=policy_seed,
                checkpoint_id=checkpoint_id,
                total_rollouts=total_rollouts,
                code_commit=code_commit,
                episode_length=episode_length,
                warmup_iterations=warmup_iterations,
                measurement_iterations=worker_iterations,
            )
            worker = context.Process(
                target=_latency_worker,
                args=(config, event_queue),
                name=f"pusht-e3-latency-gpu-{gpu_index}",
            )
            worker.start()
            workers.append(worker)

        while len(results_by_gpu) < worker_count or len(completed_progress) < expected_progress:
            try:
                event: LatencyWorkerEvent = event_queue.get(timeout=1)
            except Empty:
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    details = ", ".join(
                        f"{worker.name} exitcode={worker.exitcode}" for worker in failed
                    )
                    raise RuntimeError(f"E3 latency worker failed: {details}")
                if (
                    len(results_by_gpu) < worker_count
                    and workers
                    and all(worker.exitcode is not None for worker in workers)
                ):
                    raise RuntimeError("E3 latency workers exited before returning every result")
                continue
            if event.kind == "error":
                raise RuntimeError(
                    f"E3 latency worker on GPU {event.gpu_index} failed:\n{event.error}"
                )
            if event.kind == "progress":
                if event.condition_id is None or event.latency_kind is None:
                    raise RuntimeError("E3 latency progress event is incomplete")
                key = (event.gpu_index, event.condition_id, event.latency_kind)
                if key in completed_progress:
                    raise RuntimeError(f"Duplicate E3 latency progress event: {key}")
                completed_progress.add(key)
                write_json_atomic(
                    progress_dir
                    / (f"gpu_{event.gpu_index:02d}_{event.condition_id}_{event.latency_kind}.json"),
                    {
                        "status": "complete",
                        "gpu_index": event.gpu_index,
                        "condition_id": event.condition_id,
                        "latency_kind": event.latency_kind,
                        "measurement_iterations": iteration_counts[event.gpu_index],
                        "warmup_iterations": warmup_iterations,
                    },
                )
                _render_progress_bar(
                    completed=len(completed_progress),
                    total=expected_progress,
                )
                continue
            if event.kind != "result" or event.rows is None:
                raise RuntimeError(f"Unexpected E3 latency worker event: {event.kind}")
            if event.gpu_index in results_by_gpu:
                raise RuntimeError(f"Duplicate E3 latency result from GPU {event.gpu_index}")
            results_by_gpu[event.gpu_index] = event

        for worker in workers:
            worker.join(timeout=30)
        alive = [worker for worker in workers if worker.is_alive()]
        if alive:
            details = ", ".join(worker.name for worker in alive)
            raise RuntimeError(f"E3 latency workers did not shut down: {details}")
        failed = [worker for worker in workers if worker.exitcode != 0]
        if failed:
            details = ", ".join(f"{worker.name} exitcode={worker.exitcode}" for worker in failed)
            raise RuntimeError(f"E3 latency worker shutdown failed: {details}")
    except (Exception, KeyboardInterrupt):
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        raise
    finally:
        event_queue.close()
        event_queue.join_thread()

    worker_events = [results_by_gpu[index] for index in range(worker_count)]
    parameter_hashes = worker_events[0].parameter_hashes
    action_steps = worker_events[0].n_action_steps
    if parameter_hashes is None or action_steps is None:
        raise RuntimeError("E3 latency worker metadata is incomplete")
    if any(event.parameter_hashes != parameter_hashes for event in worker_events):
        raise RuntimeError("E3 latency workers loaded different model parameters")
    if any(event.n_action_steps != action_steps for event in worker_events):
        raise RuntimeError("E3 latency workers disagree on n_action_steps")
    device_names = tuple(str(event.device_name) for event in worker_events)
    if len(set(device_names)) != 1:
        raise RuntimeError(
            "E3 distributed latency requires homogeneous GPUs, got " + ", ".join(device_names)
        )
    raw_rows = [
        row for event in worker_events for row in (event.rows if event.rows is not None else [])
    ]
    latency_rows, summaries = _aggregate_parallel_latency_rows(
        raw_rows,
        conditions=conditions,
        measurement_iterations=measurement_iterations,
        n_action_steps=action_steps,
    )
    return (
        latency_rows,
        summaries,
        ParallelLatencyMetadata(
            num_gpus=worker_count,
            measurement_iterations_per_gpu=iteration_counts,
            device_names=device_names,
            parameter_hashes=parameter_hashes,
            n_action_steps=action_steps,
        ),
    )


def evaluate_sampling_steps(
    *,
    diffusion_policy_path: str | Path,
    output_dir: Path,
    conditions: Sequence[tuple[str, int]],
    num_episodes: int,
    policy_seed: int,
    train_seed: int,
    latency_warmup_iterations: int,
    latency_iterations: int,
    videos_per_condition: int,
    episode_length: int,
    overwrite: bool,
    num_gpus: int = 1,
) -> Path:
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    resolved_conditions = tuple(conditions)
    condition_ids = tuple(
        _condition_id(scheduler, steps) for scheduler, steps in resolved_conditions
    )
    if not resolved_conditions or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("E3 conditions must be non-empty and unique")
    split = load_split()
    if not 1 <= num_episodes <= len(split.final_environment_seeds):
        raise ValueError(f"num_episodes must be in [1,{len(split.final_environment_seeds)}]")
    if videos_per_condition < 0 or videos_per_condition > num_episodes:
        raise ValueError("videos_per_condition must lie in [0,num_episodes]")
    environment_seeds = split.final_environment_seeds[:num_episodes]
    _validate_num_gpus(
        num_gpus=num_gpus,
        num_environment_seeds=len(environment_seeds),
    )
    resolved_policy_path = Path(diffusion_policy_path).expanduser().resolve()
    checkpoint_id = _checkpoint_sha256(resolved_policy_path)
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    total_rollouts = len(resolved_conditions) * num_episodes
    video_environment_seeds = set(environment_seeds[:videos_per_condition])

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        video_dir = temporary / "videos"
        video_dir.mkdir()
        recordings_dir = temporary / "recordings"
        recordings_dir.mkdir()
        loaded = None
        if num_gpus == 1:
            loaded = _load_e3_policy(
                resolved_policy_path,
                device=None,
                conditions=resolved_conditions,
            )
            rollout_results = _execute_environment_shard(
                loaded=loaded,
                conditions=resolved_conditions,
                environment_seeds=environment_seeds,
                policy_seed=policy_seed,
                episode_length=episode_length,
                output_root=temporary,
                video_environment_seeds=video_environment_seeds,
            )
            gpu_shards = (tuple(environment_seeds),)
        else:
            rollout_results, gpu_shards = _execute_parallel_rollouts(
                diffusion_policy_path=resolved_policy_path,
                conditions=resolved_conditions,
                environment_seeds=environment_seeds,
                policy_seed=policy_seed,
                episode_length=episode_length,
                output_root=temporary,
                video_environment_seeds=video_environment_seeds,
                num_gpus=num_gpus,
            )

        if len(rollout_results) != total_rollouts:
            raise RuntimeError(
                f"E3 produced {len(rollout_results)}/{total_rollouts} rollout results"
            )
        expected_keys = {
            (condition_id, environment_seed)
            for condition_id in condition_ids
            for environment_seed in environment_seeds
        }
        observed_keys = {
            (result.condition_id, result.environment_seed) for result in rollout_results
        }
        if observed_keys != expected_keys:
            raise RuntimeError("E3 rollout results do not cover the exact paired condition grid")
        expected_recordings = {
            recordings_dir / f"env_{environment_seed:04d}.rrd"
            for environment_seed in environment_seeds
        }
        observed_recordings = set(recordings_dir.glob("env_*.rrd"))
        if observed_recordings != expected_recordings or any(
            path.stat().st_size == 0 for path in expected_recordings
        ):
            raise RuntimeError("E3 scene Recordings do not cover every environment exactly once")

        if num_gpus == 1:
            if loaded is None:
                raise RuntimeError("Single-GPU E3 policy was not loaded")
            latency_device = get_device_from_parameters(loaded.policy)
            latency_device_name = (
                torch.cuda.get_device_name(latency_device)
                if latency_device.type == "cuda"
                else str(latency_device)
            )
            latency_progress_dir = temporary / "latency_progress"
            latency_progress_dir.mkdir()
            completed_latency_steps = 0
            total_latency_steps = len(resolved_conditions) * 2
            _render_progress_bar(completed=0, total=total_latency_steps)

            def publish_single_latency_progress(
                condition_id: str,
                latency_kind: str,
            ) -> None:
                nonlocal completed_latency_steps
                write_json_atomic(
                    latency_progress_dir / f"gpu_00_{condition_id}_{latency_kind}.json",
                    {
                        "status": "complete",
                        "gpu_index": 0,
                        "condition_id": condition_id,
                        "latency_kind": latency_kind,
                        "measurement_iterations": latency_iterations,
                        "warmup_iterations": latency_warmup_iterations,
                    },
                )
                completed_latency_steps += 1
                _render_progress_bar(
                    completed=completed_latency_steps,
                    total=total_latency_steps,
                )

            parameter_hashes = state_dict_hashes(loaded.policy.state_dict())
            latency_rows, latency_summaries = _benchmark_sampling_latency(
                loaded=loaded,
                conditions=resolved_conditions,
                environment_seed=environment_seeds[0],
                policy_seed=policy_seed,
                checkpoint_id=checkpoint_id,
                total_rollouts=total_rollouts,
                code_commit=code_commit,
                episode_length=episode_length,
                warmup_iterations=latency_warmup_iterations,
                measurement_iterations=latency_iterations,
                latency_gpu_index=0 if latency_device.type == "cuda" else None,
                latency_device_name=latency_device_name,
                progress_callback=publish_single_latency_progress,
            )
            latency_num_gpus = 1
            latency_iteration_counts = (latency_iterations,)
            latency_device_names = (latency_device_name,)
            latency_execution = "single-process-after-rollouts"
        else:
            latency_rows, latency_summaries, latency_metadata = _execute_parallel_latency(
                diffusion_policy_path=resolved_policy_path,
                conditions=resolved_conditions,
                environment_seed=environment_seeds[0],
                policy_seed=policy_seed,
                checkpoint_id=checkpoint_id,
                total_rollouts=total_rollouts,
                code_commit=code_commit,
                episode_length=episode_length,
                warmup_iterations=latency_warmup_iterations,
                measurement_iterations=latency_iterations,
                output_root=temporary,
                num_gpus=num_gpus,
            )
            parameter_hashes = latency_metadata.parameter_hashes
            latency_num_gpus = latency_metadata.num_gpus
            latency_iteration_counts = latency_metadata.measurement_iterations_per_gpu
            latency_device_names = latency_metadata.device_names
            latency_execution = "distributed-all-conditions-per-gpu"
        condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
        rollout_results = sorted(
            rollout_results,
            key=lambda result: (
                condition_order[result.condition_id],
                result.environment_seed,
            ),
        )
        rollout_rows = [
            _rollout_row(
                result,
                checkpoint_id=checkpoint_id,
                train_seed=train_seed,
                total_rollouts=total_rollouts,
                code_commit=code_commit,
            )
            for result in rollout_results
        ]
        replan_rows = [
            row
            for result in rollout_results
            for row in _replan_rows(
                result,
                checkpoint_id=checkpoint_id,
                train_seed=train_seed,
                total_rollouts=total_rollouts,
                code_commit=code_commit,
            )
        ]
        condition_rows: list[dict[str, Any]] = []
        for scheduler, steps in resolved_conditions:
            condition_id = _condition_id(scheduler, steps)
            condition_results = [
                result for result in rollout_results if result.condition_id == condition_id
            ]
            interval = wilson_interval(
                sum(result.success for result in condition_results),
                len(condition_results),
            )
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "scheduler": scheduler,
                    "num_inference_steps": steps,
                    "num_rollouts": len(condition_results),
                    "success_rate": interval.estimate,
                    "success_ci95_lower": interval.lower,
                    "success_ci95_upper": interval.upper,
                    "mean_max_reward": float(
                        np.mean([result.max_reward for result in condition_results])
                    ),
                    **latency_summaries[condition_id],
                }
            )

        registered_reference = _E3_CONFIG["reference_condition"]
        reference = next(
            (
                row
                for row in condition_rows
                if row["scheduler"] == str(registered_reference["scheduler"]).upper()
                and row["num_inference_steps"] == int(registered_reference["num_inference_steps"])
            ),
            None,
        )
        if reference is None:
            raise ValueError("E3 conditions do not include the registered reference")
        qualifying: list[dict[str, Any]] = []
        for row in condition_rows:
            row["p95_speedup_vs_ddpm100"] = float(reference["end_to_end_p95_ms"]) / float(
                row["end_to_end_p95_ms"]
            )
            row["success_drop_vs_ddpm100"] = float(reference["success_rate"]) - float(
                row["success_rate"]
            )
            row["meets_target"] = (
                row["p95_speedup_vs_ddpm100"] >= 2.0 and row["success_drop_vs_ddpm100"] <= 0.05
            )
            if row["meets_target"]:
                qualifying.append(row)

        write_records_parquet(temporary / "rollouts.parquet", rollout_rows)
        write_records_parquet(temporary / "replans.parquet", replan_rows)
        write_records_parquet(temporary / "latency.parquet", latency_rows)
        write_records_parquet(temporary / "condition_metrics.parquet", condition_rows)
        best = (
            min(qualifying, key=lambda row: float(row["end_to_end_p95_ms"])) if qualifying else None
        )
        metrics = {
            "reference_condition": _condition_id(
                str(registered_reference["scheduler"]).upper(),
                int(registered_reference["num_inference_steps"]),
            ),
            "reference_success_rate": reference["success_rate"],
            "reference_end_to_end_p95_ms": reference["end_to_end_p95_ms"],
            "num_conditions": len(condition_rows),
            "num_rollouts": total_rollouts,
            "num_qualifying_conditions": len(qualifying),
            "best_qualifying_condition": None if best is None else best["condition_id"],
        }
        write_metrics(
            temporary,
            metrics,
            comments={
                "reference_condition": "High-quality pre-registered comparison",
                "reference_success_rate": "Closed-loop success across paired final seeds",
                "reference_end_to_end_p95_ms": "Raw observation to denormalized 8-action chunk",
                "num_conditions": "Distinct scheduler and denoising-step settings",
                "num_rollouts": "Actual independent environment episodes",
                "num_qualifying_conditions": "At least 2x faster and <=5 percentage-point drop",
                "best_qualifying_condition": "Lowest p95 among conditions meeting the criterion",
            },
        )
        write_json_atomic(
            temporary / "condition_metrics.json",
            {"conditions": condition_rows},
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 2,
                "experiment_schema": "pusht-e3-sampling-steps-v2",
                "label": "Closed-loop",
                "experiment": "E3",
                "conditions": [
                    {"scheduler": scheduler, "num_inference_steps": steps}
                    for scheduler, steps in resolved_conditions
                ],
                "reference_condition": registered_reference,
                "ddim_eta": 0.0,
                "environment_seeds": list(environment_seeds),
                "policy_seed": policy_seed,
                "num_action_samples": 1,
                "num_rollouts": total_rollouts,
                "checkpoint_id": checkpoint_id,
                "parameter_hashes": parameter_hashes,
                "num_gpus": num_gpus,
                "environment_seed_shards": [list(shard) for shard in gpu_shards],
                "recordings": {
                    "layout": "one-environment-synchronized-condition-panels",
                    "directory": "recordings",
                    "count": len(expected_recordings),
                    "conditions_per_recording": len(condition_ids),
                    "all_rollouts_include_simulator_frames": True,
                    "blueprint_active": True,
                    "blueprint_default": True,
                },
                "train_seed": train_seed,
                "dataset_revision": DATASET_REVISION,
                "latency": {
                    "batch_size": 1,
                    "device_names": list(latency_device_names),
                    "execution": latency_execution,
                    "num_gpus": latency_num_gpus,
                    "warmup_iterations_per_gpu": latency_warmup_iterations,
                    "measurement_iterations_per_condition": latency_iterations,
                    "measurement_iterations_per_gpu": list(latency_iteration_counts),
                    "progress_directory": "latency_progress",
                    "cuda_synchronized": True,
                    "rendering": False,
                },
                "runtime": runtime,
            },
        )
        _plot_results(temporary, condition_rows, latency_rows)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    registered_conditions = _E3_CONFIG["conditions"]
    registered_ddim = ",".join(
        str(item["num_inference_steps"])
        for item in registered_conditions
        if str(item["scheduler"]).upper() == "DDIM"
    )
    registered_ddpm = ",".join(
        str(item["num_inference_steps"])
        for item in registered_conditions
        if str(item["scheduler"]).upper() == "DDPM"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-policy", required=True)
    parser.add_argument("--schedulers", default="ddim,ddpm")
    parser.add_argument("--ddim-steps", default=registered_ddim)
    parser.add_argument("--ddpm-steps", default=registered_ddpm)
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=int(_E3_CONFIG["num_environment_seeds"]),
    )
    parser.add_argument("--policy-seed", type=int, default=int(_E3_CONFIG["policy_seed"]))
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--latency-warmup-iterations",
        type=int,
        default=int(_E3_CONFIG["latency_warmup_iterations"]),
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=int(_E3_CONFIG["latency_measurement_iterations"]),
    )
    parser.add_argument(
        "--videos-per-condition",
        type=int,
        default=int(_E3_CONFIG["videos_per_condition"]),
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e3_sampling_steps",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    conditions = _conditions(args.schedulers, args.ddim_steps, args.ddpm_steps)
    result = evaluate_sampling_steps(
        diffusion_policy_path=args.diffusion_policy,
        output_dir=args.output_dir.expanduser().resolve(),
        conditions=conditions,
        num_episodes=args.n_episodes,
        policy_seed=args.policy_seed,
        train_seed=args.train_seed,
        latency_warmup_iterations=args.latency_warmup_iterations,
        latency_iterations=args.latency_iterations,
        videos_per_condition=args.videos_per_condition,
        episode_length=args.episode_length,
        overwrite=args.overwrite,
        num_gpus=args.num_gpus,
    )
    print(f"E3 output: {result}")


if __name__ == "__main__":
    main()
