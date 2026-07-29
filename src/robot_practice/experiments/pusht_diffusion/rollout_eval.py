"""E2 closed-loop PushT sampling experiments."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any

import matplotlib
import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy

from .config import (
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_output_root,
    load_named_config,
    practice_root,
)
from .dataset_split import load_split
from .diversity_metrics import (
    complete_linkage_mode_labels,
    pairwise_rms_distance,
    resample_agent_block_trajectory,
    trajectory_distance_matrix,
    upper_triangle_mean,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rerun_visualization import record_single_rollout
from .rollout import (
    DIFFUSION_SAMPLING_PROTOCOLS,
    FULL_DDPM_SAMPLING,
    INITIAL_NOISE_SENSITIVITY,
    RolloutResult,
    run_rollout,
    save_rollout_video,
)
from .statistics import environment_cluster_bootstrap, wilson_interval

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_E2_CONFIG = load_named_config("e2_rollout.json")
EXPERIMENT_SCHEMA = str(_E2_CONFIG["experiment_schema"])
PRIMARY_MODE_THRESHOLD = float(_E2_CONFIG["mode_distance_threshold"])
SENSITIVITY_THRESHOLDS = tuple(float(value) for value in _E2_CONFIG["mode_distance_sensitivity"])
TRAJECTORY_RESAMPLE_POINTS = int(_E2_CONFIG["trajectory_resample_points"])
BOOTSTRAP_ITERATIONS = int(_E2_CONFIG["bootstrap_iterations"])
BOOTSTRAP_CONFIDENCE = float(_E2_CONFIG["confidence"])
BOOTSTRAP_SEED = int(_E2_CONFIG["bootstrap_seed"])


@dataclass(frozen=True)
class ProtocolPlan:
    protocol: str
    name: str
    condition_id: str
    environment_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    include_bc_baseline: bool
    output_directory: str

    @property
    def num_diffusion_rollouts(self) -> int:
        return len(self.environment_seeds) * len(self.policy_seeds)

    @property
    def num_bc_rollouts(self) -> int:
        return len(self.environment_seeds) if self.include_bc_baseline else 0

    @property
    def num_rollouts(self) -> int:
        return self.num_diffusion_rollouts + self.num_bc_rollouts


@dataclass(frozen=True)
class RolloutWorkerConfig:
    gpu_index: int
    protocol: str
    condition_id: str
    environment_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    include_bc_baseline: bool
    bc_policy_path: str | None
    diffusion_policy_path: str
    episode_length: int
    output_root: str
    video_policy_seeds: tuple[int, ...]


@dataclass(frozen=True)
class RolloutWorkerEvent:
    kind: str
    gpu_index: int
    result: RolloutResult | None = None
    error: str | None = None


def _configured_protocol(protocol: str) -> ProtocolPlan:
    if protocol not in DIFFUSION_SAMPLING_PROTOCOLS:
        raise ValueError(f"protocol must be one of {sorted(DIFFUSION_SAMPLING_PROTOCOLS)}")
    raw = _E2_CONFIG["protocols"][protocol]
    return ProtocolPlan(
        protocol=protocol,
        name=str(raw["name"]),
        condition_id=str(raw["condition_id"]),
        environment_seeds=tuple(int(value) for value in raw["environment_seeds"]),
        policy_seeds=tuple(int(value) for value in raw["policy_seeds"]),
        include_bc_baseline=bool(raw["include_bc_baseline"]),
        output_directory=str(raw["output_directory"]),
    )


def resolve_protocol_plan(
    *,
    protocol: str,
    label: str,
    num_environment_seeds: int | None = None,
    num_policy_seeds: int | None = None,
) -> ProtocolPlan:
    """Resolve an exact Formal plan or a reduced closed-loop Smoke plan."""

    if label not in {"Smoke", "Closed-loop"}:
        raise ValueError("label must be Smoke or Closed-loop")
    configured = _configured_protocol(protocol)
    environment_count = (
        len(configured.environment_seeds)
        if num_environment_seeds is None
        else num_environment_seeds
    )
    policy_count = len(configured.policy_seeds) if num_policy_seeds is None else num_policy_seeds
    if not 1 <= environment_count <= len(configured.environment_seeds):
        raise ValueError(
            f"num_environment_seeds must be in [1,{len(configured.environment_seeds)}] "
            f"for {protocol}"
        )
    if not 1 <= policy_count <= len(configured.policy_seeds):
        raise ValueError(
            f"num_policy_seeds must be in [1,{len(configured.policy_seeds)}] for {protocol}"
        )
    if label == "Closed-loop" and (
        environment_count != len(configured.environment_seeds)
        or policy_count != len(configured.policy_seeds)
    ):
        raise ValueError(
            f"Closed-loop {protocol} requires exactly "
            f"{len(configured.environment_seeds)} environment seeds × "
            f"{len(configured.policy_seeds)} policy seeds; use label=Smoke for reduced runs"
        )
    return replace(
        configured,
        environment_seeds=configured.environment_seeds[:environment_count],
        policy_seeds=configured.policy_seeds[:policy_count],
    )


def partition_environment_seeds(
    environment_seeds: Sequence[int],
    *,
    num_gpus: int,
) -> tuple[tuple[int, ...], ...]:
    seeds = tuple(int(seed) for seed in environment_seeds)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    if num_gpus > len(seeds):
        raise ValueError("num_gpus cannot exceed the number of environment seeds")
    return tuple(seeds[gpu_index::num_gpus] for gpu_index in range(num_gpus))


def rollout_recording_identity(
    *,
    protocol: str,
    model: str,
    environment_seed: int,
    policy_seed: int,
    success: bool,
) -> tuple[str, str]:
    protocol_labels = {
        INITIAL_NOISE_SENSITIVITY: "E2a",
        FULL_DDPM_SAMPLING: "E2b",
    }
    if protocol not in protocol_labels:
        raise ValueError(f"Unknown E2 recording protocol: {protocol}")
    protocol_label = protocol_labels[protocol]
    outcome = "success" if success else "failure"
    if model == "bc_chunk":
        return (
            f"{protocol_label.lower()}_bc_env_{environment_seed:04d}",
            f"{protocol_label} | BC | env {environment_seed} | {outcome}",
        )
    if model == "diffusion":
        return (
            (f"{protocol_label.lower()}_dp_env_{environment_seed:04d}_seed_{policy_seed:02d}"),
            (
                f"{protocol_label} | DP | env {environment_seed} | "
                f"seed {policy_seed:02d} | {outcome}"
            ),
        )
    raise ValueError(f"Unknown E2 recording model: {model}")


def _validate_num_gpus(*, num_gpus: int, num_environment_seeds: int) -> None:
    partition_environment_seeds(
        tuple(range(num_environment_seeds)),
        num_gpus=num_gpus,
    )
    if num_gpus == 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU E2 evaluation requires CUDA")
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise RuntimeError(
            f"num_gpus={num_gpus} requested, but only {available} CUDA device(s) are visible"
        )


def _checkpoint_sha256(path: str | Path) -> str:
    model_path = Path(path).expanduser().resolve() / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint weights are missing: {model_path}")
    return sha256_file(model_path)


def _resampled(result: RolloutResult) -> np.ndarray:
    return resample_agent_block_trajectory(
        result.agent_path,
        result.block_path,
        num_points=TRAJECTORY_RESAMPLE_POINTS,
    )


def _distance_summary(results: Sequence[RolloutResult]) -> float | None:
    if len(results) < 2:
        return None
    matrix = trajectory_distance_matrix([_resampled(item) for item in results])
    return upper_triangle_mean(matrix)


def _bootstrap_diversity(
    grouped: Mapping[int, Sequence[RolloutResult]],
) -> dict[str, Any] | None:
    """Average per-environment pairwise means with environments weighted equally."""

    per_environment = {
        environment_seed: value
        for environment_seed, results in grouped.items()
        if (value := _distance_summary(results)) is not None
    }
    if not per_environment:
        return None
    interval = environment_cluster_bootstrap(
        {seed: [value] for seed, value in per_environment.items()},
        iterations=BOOTSTRAP_ITERATIONS,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_SEED,
    )
    return {
        **interval.to_dict(),
        "eligible_environment_count": len(per_environment),
        "per_environment": per_environment,
    }


def _first_replan_diversity(
    grouped: Mapping[int, Sequence[RolloutResult]],
) -> dict[str, Any] | None:
    per_environment: dict[int, float] = {}
    for environment_seed, results in grouped.items():
        chunks = [result.replans[0].normalized_chunk for result in results if result.replans]
        if len(chunks) < 2:
            continue
        per_environment[environment_seed] = upper_triangle_mean(
            pairwise_rms_distance(np.stack(chunks))
        )
    if not per_environment:
        return None
    interval = environment_cluster_bootstrap(
        {seed: [value] for seed, value in per_environment.items()},
        iterations=BOOTSTRAP_ITERATIONS,
        confidence=BOOTSTRAP_CONFIDENCE,
        seed=BOOTSTRAP_SEED,
    )
    return {
        **interval.to_dict(),
        "eligible_environment_count": len(per_environment),
        "per_environment": per_environment,
    }


def success_variance(
    successes_by_environment: Mapping[int, Sequence[bool | int | float]],
) -> tuple[float, float]:
    """Return mean within-environment Bernoulli variance and variance across rates."""

    if not successes_by_environment:
        raise ValueError("At least one environment is required")
    rates = []
    for outcomes in successes_by_environment.values():
        values = np.asarray(outcomes, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError("Every environment must contain non-empty binary outcomes")
        rates.append(float(np.mean(values)))
    rate_array = np.asarray(rates, dtype=np.float64)
    return (
        float(np.mean(rate_array * (1.0 - rate_array))),
        float(np.var(rate_array, ddof=0)),
    )


def _mode_labels(
    grouped: Mapping[int, Sequence[RolloutResult]],
    *,
    threshold: float,
) -> tuple[dict[tuple[int, int], int | None], dict[int, int]]:
    labels_by_rollout: dict[tuple[int, int], int | None] = {}
    counts: dict[int, int] = {}
    for environment_seed, results in grouped.items():
        successful = [result for result in results if result.success]
        for result in results:
            labels_by_rollout[(environment_seed, result.policy_seed)] = None
        if not successful:
            counts[environment_seed] = 0
            continue
        matrix = trajectory_distance_matrix([_resampled(item) for item in successful])
        labels = complete_linkage_mode_labels(matrix, threshold=threshold)
        counts[environment_seed] = len({int(value) for value in labels})
        for result, label in zip(successful, labels, strict=True):
            labels_by_rollout[(environment_seed, result.policy_seed)] = int(label)
    return labels_by_rollout, counts


def _validate_closed_loop_result(result: RolloutResult) -> None:
    if result.episode_length <= 0 or result.action_targets.shape != (result.episode_length, 2):
        raise RuntimeError("Every E2 sample must contain executed closed-loop actions")
    if result.rewards.shape != (result.episode_length,):
        raise RuntimeError("Every E2 action must have one environment reward")
    if result.agent_path.shape != (result.episode_length + 1, 2):
        raise RuntimeError("Agent path must contain the reset state and every closed-loop state")
    if result.block_path.shape != (result.episode_length + 1, 3):
        raise RuntimeError("Block path must contain the reset state and every closed-loop state")
    if not result.replans:
        raise RuntimeError("Every E2 sample must execute at least one policy replan")
    if result.termination_reason == "unknown":
        raise RuntimeError("Every E2 sample must record its termination reason")


def _rollout_row(
    result: RolloutResult,
    *,
    checkpoint_id: str,
    train_seed: int,
    mode_label: int | None,
    code_commit: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": "E2",
        "experiment_schema": EXPERIMENT_SCHEMA,
        "sampling_protocol": result.sampling_protocol,
        "condition_id": result.condition_id,
        "model": result.model,
        "checkpoint_id": checkpoint_id,
        "train_seed": train_seed,
        "environment_seed": result.environment_seed,
        "policy_seed": result.policy_seed,
        "replan_index": None,
        "initial_noise_seed": None,
        "initial_noise_sha256": None,
        "reverse_noise_seed": None,
        "reverse_generator_state_sha256": None,
        "num_inference_steps": result.num_inference_steps,
        "num_action_samples": 1,
        "num_rollouts": 1,
        "dataset_revision": DATASET_REVISION,
        "code_commit": code_commit,
        "success": result.success,
        "sum_reward": result.sum_reward,
        "max_reward": result.max_reward,
        "final_reward": result.final_reward,
        "episode_length": result.episode_length,
        "termination_reason": result.termination_reason,
        "successful_mode_label": mode_label,
        "agent_path": result.agent_path.tolist(),
        "block_path": result.block_path.tolist(),
        "action_targets": result.action_targets.tolist(),
        "rewards": result.rewards.tolist(),
    }


def _replan_rows(
    result: RolloutResult,
    *,
    checkpoint_id: str,
    train_seed: int,
    code_commit: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "E2",
            "experiment_schema": EXPERIMENT_SCHEMA,
            "sampling_protocol": replan.sampling_protocol,
            "condition_id": result.condition_id,
            "model": result.model,
            "checkpoint_id": checkpoint_id,
            "train_seed": train_seed,
            "environment_seed": result.environment_seed,
            "policy_seed": result.policy_seed,
            "replan_index": replan.replan_index,
            "initial_noise_seed": replan.initial_noise_seed,
            "initial_noise_sha256": replan.initial_noise_sha256,
            "reverse_noise_seed": replan.reverse_noise_seed,
            "reverse_generator_state_sha256": replan.reverse_generator_state_sha256,
            "num_inference_steps": replan.num_inference_steps,
            "num_action_samples": 1,
            "num_rollouts": 1,
            "dataset_revision": DATASET_REVISION,
            "code_commit": code_commit,
            "scheduler": replan.scheduler,
            "chunk_latency_ms": replan.latency_ms,
            "normalized_chunk": replan.normalized_chunk.tolist(),
        }
        for replan in result.replans
    ]


def _save_static_figures(
    output_dir: Path,
    diffusion_results: Sequence[RolloutResult],
    diversity: Mapping[str, float | None],
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    for result in diffusion_results:
        if result.success:
            axis.plot(result.agent_path[:, 0], result.agent_path[:, 1], alpha=0.4)
    axis.set(
        xlim=(0, 512),
        ylim=(512, 0),
        xlabel="x",
        ylabel="y",
        title="Successful closed-loop agent paths",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "successful_rollout_modes.png", dpi=160)
    plt.close(figure)

    labels = ["all", "successful", "failed"]
    values = [diversity.get(label) or 0.0 for label in labels]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#4e79a7", "#59a14f", "#e15759"])
    axis.set_ylabel("joint trajectory RMS distance")
    axis.set_title("Closed-loop trajectory diversity")
    figure.tight_layout()
    figure.savefig(output_dir / "success_conditioned_diversity.png", dpi=160)
    plt.close(figure)


def _run_one(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    environment_seed: int,
    policy_seed: int,
    episode_length: int,
    condition_id: str,
    sampling_protocol: str,
    record_frames: bool,
) -> RolloutResult:
    result = run_rollout(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        record_frames=record_frames,
        episode_length=episode_length,
        condition_id=condition_id,
        sampling_protocol=sampling_protocol,
    )
    _validate_closed_loop_result(result)
    return result


def _load_rollout_policies(
    *,
    diffusion_policy_path: str | Path,
    bc_policy_path: str | Path | None,
    include_bc_baseline: bool,
    device: str | None = None,
) -> tuple[Any | None, Any]:
    diffusion_loaded = load_policy_with_processors(
        diffusion_policy_path,
        local_files_only=True,
        device=device,
    )
    if not isinstance(diffusion_loaded.policy, DiffusionPolicy):
        raise TypeError("diffusion-policy must contain policy.type=diffusion")
    replace_scheduler(
        diffusion_loaded.policy,
        scheduler_name=str(_E2_CONFIG["scheduler"]),
        num_inference_steps=int(_E2_CONFIG["num_inference_steps"]),
    )
    if diffusion_loaded.config.n_action_steps != int(_E2_CONFIG["n_action_steps"]):
        raise ValueError("Loaded checkpoint does not use the registered E2 action horizon")

    bc_loaded = None
    if include_bc_baseline:
        if bc_policy_path is None:
            raise ValueError("Full DDPM sampling requires --bc-policy")
        bc_loaded = load_policy_with_processors(
            bc_policy_path,
            local_files_only=True,
            device=device,
        )
        if not isinstance(bc_loaded.policy, BCChunkPolicy):
            raise TypeError("bc-policy must contain policy.type=bc_chunk")
        for field in ("n_obs_steps", "horizon", "n_action_steps"):
            if getattr(bc_loaded.config, field) != getattr(diffusion_loaded.config, field):
                raise ValueError(f"BCChunk and Diffusion disagree on {field}")
    return bc_loaded, diffusion_loaded


def _execute_assigned_rollouts(
    *,
    plan: ProtocolPlan,
    bc_loaded: Any | None,
    diffusion_loaded: Any,
    output_root: Path,
    episode_length: int,
    video_policy_seeds: set[int],
    result_callback: Any | None = None,
) -> tuple[list[RolloutResult], list[RolloutResult]]:
    video_dir = output_root / "videos"
    recordings_dir = output_root / "recordings"
    bc_results: list[RolloutResult] = []
    diffusion_results: list[RolloutResult] = []

    def store(result: RolloutResult, destination: list[RolloutResult]) -> None:
        lightweight = replace(result, frames=())
        destination.append(lightweight)
        if result_callback is not None:
            result_callback(lightweight)

    if bc_loaded is not None:
        bc_recordings_dir = output_root / "bc_recordings"
        for environment_seed in plan.environment_seeds:
            result = _run_one(
                policy=bc_loaded.policy,
                preprocessor=bc_loaded.preprocessor,
                postprocessor=bc_loaded.postprocessor,
                environment_seed=environment_seed,
                policy_seed=0,
                episode_length=episode_length,
                condition_id="bc_chunk",
                sampling_protocol=plan.protocol,
                record_frames=True,
            )
            recording_id, recording_name = rollout_recording_identity(
                protocol=plan.protocol,
                model=result.model,
                environment_seed=environment_seed,
                policy_seed=result.policy_seed,
                success=result.success,
            )
            record_single_rollout(
                output_path=bc_recordings_dir / f"env_{environment_seed:04d}.rrd",
                rollout=result,
                recording_id=recording_id,
                recording_name=recording_name,
            )
            save_rollout_video(
                result,
                video_dir / f"bc_chunk_env_{environment_seed:04d}.mp4",
            )
            store(result, bc_results)

    for environment_seed in plan.environment_seeds:
        for policy_seed in plan.policy_seeds:
            result = _run_one(
                policy=diffusion_loaded.policy,
                preprocessor=diffusion_loaded.preprocessor,
                postprocessor=diffusion_loaded.postprocessor,
                environment_seed=environment_seed,
                policy_seed=policy_seed,
                episode_length=episode_length,
                condition_id=plan.condition_id,
                sampling_protocol=plan.protocol,
                record_frames=True,
            )
            policy_recordings_dir = recordings_dir / f"policy_seed_{policy_seed:02d}"
            policy_recordings_dir.mkdir(exist_ok=True)
            recording_id, recording_name = rollout_recording_identity(
                protocol=plan.protocol,
                model=result.model,
                environment_seed=environment_seed,
                policy_seed=policy_seed,
                success=result.success,
            )
            record_single_rollout(
                output_path=policy_recordings_dir / f"env_{environment_seed:04d}.rrd",
                rollout=result,
                recording_id=recording_id,
                recording_name=recording_name,
            )
            if policy_seed in video_policy_seeds:
                save_rollout_video(
                    result,
                    video_dir
                    / f"diffusion_env_{environment_seed:04d}_policy_{policy_seed:02d}.mp4",
                )
            store(result, diffusion_results)
    return bc_results, diffusion_results


def _rollout_worker(
    config: RolloutWorkerConfig,
    event_queue: Any,
) -> None:
    try:
        torch.cuda.set_device(config.gpu_index)
        bc_loaded, diffusion_loaded = _load_rollout_policies(
            diffusion_policy_path=config.diffusion_policy_path,
            bc_policy_path=config.bc_policy_path,
            include_bc_baseline=config.include_bc_baseline,
            device=f"cuda:{config.gpu_index}",
        )
        plan = ProtocolPlan(
            protocol=config.protocol,
            name=config.protocol,
            condition_id=config.condition_id,
            environment_seeds=config.environment_seeds,
            policy_seeds=config.policy_seeds,
            include_bc_baseline=config.include_bc_baseline,
            output_directory="internal-worker",
        )

        def publish(result: RolloutResult) -> None:
            event_queue.put(
                RolloutWorkerEvent(
                    kind="result",
                    gpu_index=config.gpu_index,
                    result=result,
                )
            )

        _execute_assigned_rollouts(
            plan=plan,
            bc_loaded=bc_loaded,
            diffusion_loaded=diffusion_loaded,
            output_root=Path(config.output_root),
            episode_length=config.episode_length,
            video_policy_seeds=set(config.video_policy_seeds),
            result_callback=publish,
        )
        event_queue.put(
            RolloutWorkerEvent(
                kind="done",
                gpu_index=config.gpu_index,
            )
        )
    except (Exception, KeyboardInterrupt):
        event_queue.put(
            RolloutWorkerEvent(
                kind="error",
                gpu_index=config.gpu_index,
                error=traceback.format_exc(),
            )
        )
        raise


def _execute_parallel_rollouts(
    *,
    plan: ProtocolPlan,
    bc_policy_path: str | Path | None,
    diffusion_policy_path: str | Path,
    output_root: Path,
    episode_length: int,
    video_policy_seeds: set[int],
    num_gpus: int,
) -> tuple[list[RolloutResult], list[RolloutResult], tuple[tuple[int, ...], ...]]:
    shards = partition_environment_seeds(plan.environment_seeds, num_gpus=num_gpus)
    context = mp.get_context("spawn")
    event_queue = context.Queue()
    workers = []
    results_by_key: dict[tuple[str, int, int], RolloutResult] = {}
    completed_workers: set[int] = set()
    try:
        for gpu_index, environment_seeds in enumerate(shards):
            config = RolloutWorkerConfig(
                gpu_index=gpu_index,
                protocol=plan.protocol,
                condition_id=plan.condition_id,
                environment_seeds=environment_seeds,
                policy_seeds=plan.policy_seeds,
                include_bc_baseline=plan.include_bc_baseline,
                bc_policy_path=None if bc_policy_path is None else str(bc_policy_path),
                diffusion_policy_path=str(diffusion_policy_path),
                episode_length=episode_length,
                output_root=str(output_root),
                video_policy_seeds=tuple(sorted(video_policy_seeds)),
            )
            worker = context.Process(
                target=_rollout_worker,
                args=(config, event_queue),
                name=f"pusht-e2-gpu-{gpu_index}",
            )
            worker.start()
            workers.append(worker)

        while len(results_by_key) < plan.num_rollouts or len(completed_workers) < num_gpus:
            try:
                event: RolloutWorkerEvent = event_queue.get(timeout=1)
            except Empty:
                failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if failed:
                    details = ", ".join(
                        f"{worker.name} exitcode={worker.exitcode}" for worker in failed
                    )
                    raise RuntimeError(f"E2 rollout worker failed: {details}")
                exited_without_done = [
                    worker
                    for gpu_index, worker in enumerate(workers)
                    if worker.exitcode == 0 and gpu_index not in completed_workers
                ]
                if exited_without_done:
                    details = ", ".join(worker.name for worker in exited_without_done)
                    raise RuntimeError(
                        f"E2 rollout worker exited without a completion event: {details}"
                    )
                if (
                    len(results_by_key) < plan.num_rollouts
                    and workers
                    and all(worker.exitcode is not None for worker in workers)
                ):
                    raise RuntimeError(
                        "E2 rollout workers completed with only "
                        f"{len(results_by_key)}/{plan.num_rollouts} results"
                    )
                continue
            if event.kind == "error":
                raise RuntimeError(
                    f"E2 rollout worker on GPU {event.gpu_index} failed:\n{event.error}"
                )
            if event.kind == "done":
                if event.gpu_index in completed_workers:
                    raise RuntimeError(
                        f"Duplicate E2 worker completion event from GPU {event.gpu_index}"
                    )
                completed_workers.add(event.gpu_index)
                continue
            if event.kind != "result" or event.result is None:
                raise RuntimeError(f"Unexpected E2 worker event: {event.kind}")
            result = event.result
            key = (result.model, result.environment_seed, result.policy_seed)
            if key in results_by_key:
                raise RuntimeError(f"Duplicate E2 rollout result from workers: {key}")
            results_by_key[key] = result
            print(
                f"E2 multi-GPU progress: {len(results_by_key)}/{plan.num_rollouts}",
                flush=True,
            )

        for worker in workers:
            worker.join(timeout=30)
        alive = [worker for worker in workers if worker.is_alive()]
        if alive:
            details = ", ".join(worker.name for worker in alive)
            raise RuntimeError(f"E2 rollout workers did not shut down: {details}")
        failed = [worker for worker in workers if worker.exitcode != 0]
        if failed:
            details = ", ".join(f"{worker.name} exitcode={worker.exitcode}" for worker in failed)
            raise RuntimeError(f"E2 rollout worker shutdown failed: {details}")
    except (Exception, KeyboardInterrupt):
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        raise
    finally:
        event_queue.close()
        event_queue.join_thread()

    bc_results = sorted(
        (result for result in results_by_key.values() if result.model == "bc_chunk"),
        key=lambda result: result.environment_seed,
    )
    diffusion_results = sorted(
        (result for result in results_by_key.values() if result.model == "diffusion"),
        key=lambda result: (result.environment_seed, result.policy_seed),
    )
    return bc_results, diffusion_results, shards


def _per_environment_rows(
    *,
    plan: ProtocolPlan,
    grouped: Mapping[int, Sequence[RolloutResult]],
    bc_results: Mapping[int, RolloutResult],
    mode_counts: Mapping[int, int],
    all_diversity: Mapping[str, Any] | None,
    successful_diversity: Mapping[str, Any] | None,
    failed_diversity: Mapping[str, Any] | None,
    first_replan_diversity: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []
    for environment_seed in plan.environment_seeds:
        results = grouped[environment_seed]
        success_count = sum(result.success for result in results)
        success_rate = success_count / len(results)
        rows.append(
            {
                "experiment_id": "E2",
                "experiment_schema": EXPERIMENT_SCHEMA,
                "sampling_protocol": plan.protocol,
                "environment_seed": environment_seed,
                "num_policy_seeds": len(results),
                "success_count": success_count,
                "success_rate": success_rate,
                "within_environment_success_variance": success_rate * (1.0 - success_rate),
                "success_any": success_count > 0,
                "bc_success": (
                    bc_results[environment_seed].success if environment_seed in bc_results else None
                ),
                "all_rollout_diversity": (
                    None
                    if all_diversity is None
                    else all_diversity["per_environment"].get(environment_seed)
                ),
                "successful_rollout_diversity": (
                    None
                    if successful_diversity is None
                    else successful_diversity["per_environment"].get(environment_seed)
                ),
                "failed_rollout_diversity": (
                    None
                    if failed_diversity is None
                    else failed_diversity["per_environment"].get(environment_seed)
                ),
                "first_replan_chunk_diversity": (
                    None
                    if first_replan_diversity is None
                    else first_replan_diversity["per_environment"].get(environment_seed)
                ),
                "successful_mode_count": mode_counts[environment_seed],
            }
        )
    return rows


def evaluate_multi_seed_rollouts(
    *,
    diffusion_policy_path: str | Path,
    output_dir: Path,
    protocol: str = FULL_DDPM_SAMPLING,
    bc_policy_path: str | Path | None = None,
    num_environment_seeds: int | None = None,
    num_policy_seeds: int | None = None,
    train_seed: int = DEFAULT_TRAIN_SEED,
    episode_length: int = 300,
    label: str = "Closed-loop",
    overwrite: bool = False,
    num_gpus: int = 1,
) -> Path:
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    plan = resolve_protocol_plan(
        protocol=protocol,
        label=label,
        num_environment_seeds=num_environment_seeds,
        num_policy_seeds=num_policy_seeds,
    )
    split = load_split()
    configured_e2b = _configured_protocol(FULL_DDPM_SAMPLING)
    if configured_e2b.environment_seeds != split.e2_environment_seeds:
        raise ValueError("E2b environment seeds disagree with split.json")
    if plan.include_bc_baseline and bc_policy_path is None:
        raise ValueError("Full DDPM sampling requires --bc-policy")
    _validate_num_gpus(
        num_gpus=num_gpus,
        num_environment_seeds=len(plan.environment_seeds),
    )
    resolved_diffusion_path = Path(diffusion_policy_path).expanduser().resolve()
    resolved_bc_path = (
        None if bc_policy_path is None else Path(bc_policy_path).expanduser().resolve()
    )
    bc_loaded = None
    diffusion_loaded = None
    if num_gpus == 1:
        bc_loaded, diffusion_loaded = _load_rollout_policies(
            diffusion_policy_path=resolved_diffusion_path,
            bc_policy_path=resolved_bc_path,
            include_bc_baseline=plan.include_bc_baseline,
        )

    checkpoint_ids = {"diffusion": _checkpoint_sha256(resolved_diffusion_path)}
    if resolved_bc_path is not None and plan.include_bc_baseline:
        checkpoint_ids["bc_chunk"] = _checkpoint_sha256(resolved_bc_path)
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    video_policy_seeds = {int(value) for value in _E2_CONFIG["video_policy_seeds"]}

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        video_dir = temporary / "videos"
        video_dir.mkdir()
        recordings_dir = temporary / "recordings"
        recordings_dir.mkdir()
        if plan.include_bc_baseline:
            bc_recordings_dir = temporary / "bc_recordings"
            bc_recordings_dir.mkdir()
        if num_gpus == 1:
            if diffusion_loaded is None:
                raise RuntimeError("Single-GPU Diffusion policy was not loaded")
            bc_results, diffusion_results = _execute_assigned_rollouts(
                plan=plan,
                bc_loaded=bc_loaded,
                diffusion_loaded=diffusion_loaded,
                output_root=temporary,
                episode_length=episode_length,
                video_policy_seeds=video_policy_seeds,
            )
            environment_shards = (plan.environment_seeds,)
        else:
            bc_results, diffusion_results, environment_shards = _execute_parallel_rollouts(
                plan=plan,
                bc_policy_path=resolved_bc_path,
                diffusion_policy_path=resolved_diffusion_path,
                output_root=temporary,
                episode_length=episode_length,
                video_policy_seeds=video_policy_seeds,
                num_gpus=num_gpus,
            )

        if len(diffusion_results) != plan.num_diffusion_rollouts:
            raise RuntimeError("Diffusion rollout count does not match the resolved protocol plan")
        if len(bc_results) != plan.num_bc_rollouts:
            raise RuntimeError("BC rollout count does not match the resolved protocol plan")
        bc_results.sort(key=lambda result: result.environment_seed)
        diffusion_results.sort(key=lambda result: (result.environment_seed, result.policy_seed))
        all_replan_rows = []
        for result in (*bc_results, *diffusion_results):
            checkpoint_id = checkpoint_ids[result.model]
            all_replan_rows.extend(
                _replan_rows(
                    result,
                    checkpoint_id=checkpoint_id,
                    train_seed=train_seed,
                    code_commit=code_commit,
                )
            )

        grouped: dict[int, list[RolloutResult]] = defaultdict(list)
        for result in diffusion_results:
            grouped[result.environment_seed].append(result)
        mode_labels, primary_mode_counts = _mode_labels(
            grouped,
            threshold=PRIMARY_MODE_THRESHOLD,
        )
        sensitivity_counts = {
            str(threshold): _mode_labels(grouped, threshold=threshold)[1]
            for threshold in SENSITIVITY_THRESHOLDS
        }

        rollout_rows = [
            _rollout_row(
                result,
                checkpoint_id=checkpoint_ids["bc_chunk"],
                train_seed=train_seed,
                mode_label=None,
                code_commit=code_commit,
            )
            for result in bc_results
        ]
        rollout_rows.extend(
            _rollout_row(
                result,
                checkpoint_id=checkpoint_ids["diffusion"],
                train_seed=train_seed,
                mode_label=mode_labels[(result.environment_seed, result.policy_seed)],
                code_commit=code_commit,
            )
            for result in diffusion_results
        )
        write_records_parquet(temporary / "rollouts.parquet", rollout_rows)
        write_records_parquet(temporary / "replans.parquet", all_replan_rows)

        successful_grouped = {
            seed: [result for result in results if result.success]
            for seed, results in grouped.items()
        }
        failed_grouped = {
            seed: [result for result in results if not result.success]
            for seed, results in grouped.items()
        }
        all_diversity = _bootstrap_diversity(grouped)
        successful_diversity = _bootstrap_diversity(successful_grouped)
        failed_diversity = _bootstrap_diversity(failed_grouped)
        first_replan_diversity = _first_replan_diversity(grouped)
        diversity_points = {
            "all": None if all_diversity is None else all_diversity["estimate"],
            "successful": (
                None if successful_diversity is None else successful_diversity["estimate"]
            ),
            "failed": None if failed_diversity is None else failed_diversity["estimate"],
        }

        successes_per_environment = {
            seed: sum(result.success for result in results) for seed, results in grouped.items()
        }
        mean_within_success_variance, between_success_variance = success_variance(
            {seed: [result.success for result in grouped[seed]] for seed in plan.environment_seeds}
        )
        policy_zero = [result for result in diffusion_results if result.policy_seed == 0]
        single_interval = wilson_interval(
            sum(result.success for result in policy_zero),
            len(policy_zero),
            confidence=BOOTSTRAP_CONFIDENCE,
        )
        repeated_success_interval = environment_cluster_bootstrap(
            {
                seed: [float(result.success) for result in results]
                for seed, results in grouped.items()
            },
            iterations=BOOTSTRAP_ITERATIONS,
            confidence=BOOTSTRAP_CONFIDENCE,
            seed=BOOTSTRAP_SEED,
        )
        any_success = [
            int(successes_per_environment[environment_seed] > 0)
            for environment_seed in plan.environment_seeds
        ]
        any_interval = wilson_interval(
            sum(any_success),
            len(any_success),
            confidence=BOOTSTRAP_CONFIDENCE,
        )
        two_success = [
            int(successes_per_environment[environment_seed] >= 2)
            for environment_seed in plan.environment_seeds
        ]
        two_modes = [
            int(primary_mode_counts[environment_seed] >= 2)
            for environment_seed in plan.environment_seeds
        ]
        two_success_interval = wilson_interval(
            sum(two_success),
            len(two_success),
            confidence=BOOTSTRAP_CONFIDENCE,
        )
        two_modes_interval = wilson_interval(
            sum(two_modes),
            len(two_modes),
            confidence=BOOTSTRAP_CONFIDENCE,
        )
        bc_interval = None
        if bc_results:
            bc_interval = wilson_interval(
                sum(result.success for result in bc_results),
                len(bc_results),
                confidence=BOOTSTRAP_CONFIDENCE,
            )

        metrics = {
            "sampling_protocol": plan.protocol,
            "protocol_name": plan.name,
            "all_samples_closed_loop": True,
            "bc_success_rate": None if bc_interval is None else bc_interval.estimate,
            "single_sample_success": single_interval.estimate,
            "single_sample_success_ci95_lower": single_interval.lower,
            "single_sample_success_ci95_upper": single_interval.upper,
            "all_diffusion_rollout_success": repeated_success_interval.estimate,
            "all_diffusion_rollout_success_ci95_lower": repeated_success_interval.lower,
            "all_diffusion_rollout_success_ci95_upper": repeated_success_interval.upper,
            "mean_within_environment_success_variance": float(mean_within_success_variance),
            "between_environment_success_rate_variance": between_success_variance,
            "success_any_at_m": any_interval.estimate,
            "success_any_at_10": (any_interval.estimate if len(plan.policy_seeds) == 10 else None),
            "mean_successes_per_environment_seed": float(
                np.mean(list(successes_per_environment.values()))
            ),
            "fraction_with_2plus_successful_rollouts": two_success_interval.estimate,
            "fraction_with_2plus_distinct_success_modes": two_modes_interval.estimate,
            "mean_successful_mode_count": float(np.mean(list(primary_mode_counts.values()))),
            "max_successful_mode_count": int(max(primary_mode_counts.values())),
            "first_replan_chunk_diversity": (
                None if first_replan_diversity is None else first_replan_diversity["estimate"]
            ),
            "successful_rollout_diversity": diversity_points["successful"],
            "failed_rollout_diversity": diversity_points["failed"],
            "all_rollout_diversity": diversity_points["all"],
            "num_environment_seeds": len(plan.environment_seeds),
            "num_policy_seeds": len(plan.policy_seeds),
            "num_diffusion_rollouts": plan.num_diffusion_rollouts,
            "num_bc_rollouts": plan.num_bc_rollouts,
            "num_rollouts": plan.num_rollouts,
        }
        write_metrics(
            temporary,
            metrics,
            comments={
                "sampling_protocol": "Registered E2 closed-loop RNG protocol",
                "protocol_name": "Human-readable experiment name",
                "all_samples_closed_loop": "Every sample executed observe-policy-step feedback",
                "bc_success_rate": "One deterministic BCChunk rollout per E2b environment",
                "single_sample_success": "Diffusion deployment estimate at policy_seed=0",
                "all_diffusion_rollout_success": "All samples; environment-cluster bootstrap",
                "mean_within_environment_success_variance": "Mean of p_env*(1-p_env)",
                "between_environment_success_rate_variance": "Population variance of p_env",
                "success_any_at_m": "Oracle success coverage at the executed sample count",
                "success_any_at_10": "E2b oracle coverage; null outside exact 10-seed runs",
                "mean_successes_per_environment_seed": "Successful samples per reset seed",
                "fraction_with_2plus_successful_rollouts": "Environments with >=2 successes",
                "fraction_with_2plus_distinct_success_modes": (
                    "Environments with >=2 complete-linkage success modes at 0.05"
                ),
                "mean_successful_mode_count": "Mean within-environment successful mode count",
                "max_successful_mode_count": "Maximum within-environment successful mode count",
                "first_replan_chunk_diversity": "Initial closed-loop policy response diversity",
                "successful_rollout_diversity": "Equal-environment joint-path distance, successes",
                "failed_rollout_diversity": "Equal-environment joint-path distance, failures",
                "all_rollout_diversity": "Equal-environment joint-path distance, all samples",
                "num_environment_seeds": "Independent PushT reset seeds",
                "num_policy_seeds": "Sampling seeds per environment",
                "num_diffusion_rollouts": "Executed Diffusion closed-loop episodes",
                "num_bc_rollouts": "Executed BCChunk closed-loop episodes",
                "num_rollouts": "All executed closed-loop episodes",
            },
        )

        bc_by_environment = {result.environment_seed: result for result in bc_results}
        per_environment_rows = _per_environment_rows(
            plan=plan,
            grouped=grouped,
            bc_results=bc_by_environment,
            mode_counts=primary_mode_counts,
            all_diversity=all_diversity,
            successful_diversity=successful_diversity,
            failed_diversity=failed_diversity,
            first_replan_diversity=first_replan_diversity,
        )
        write_records_parquet(
            temporary / "per_environment_metrics.parquet",
            per_environment_rows,
        )
        write_json_atomic(
            temporary / "mode_analysis.json",
            {
                "experiment_schema": EXPERIMENT_SCHEMA,
                "sampling_protocol": plan.protocol,
                "primary_threshold": PRIMARY_MODE_THRESHOLD,
                "primary_mode_counts": primary_mode_counts,
                "sensitivity_mode_counts": sensitivity_counts,
                "all_diversity_ci95": all_diversity,
                "successful_diversity_ci95": successful_diversity,
                "failed_diversity_ci95": failed_diversity,
                "first_replan_chunk_diversity_ci95": first_replan_diversity,
                "success_any_at_m_ci95": any_interval.to_dict(),
                "fraction_with_2plus_success_ci95": two_success_interval.to_dict(),
                "fraction_with_2plus_modes_ci95": two_modes_interval.to_dict(),
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 2,
                "experiment_schema": EXPERIMENT_SCHEMA,
                "label": label,
                "experiment": "E2",
                "sampling_protocol": plan.protocol,
                "protocol_name": plan.name,
                "all_samples_closed_loop": True,
                "scheduler": str(_E2_CONFIG["scheduler"]),
                "num_inference_steps": int(_E2_CONFIG["num_inference_steps"]),
                "num_action_samples": 1,
                "num_rollouts": plan.num_rollouts,
                "num_diffusion_rollouts": plan.num_diffusion_rollouts,
                "num_bc_rollouts": plan.num_bc_rollouts,
                "environment_seeds": list(plan.environment_seeds),
                "policy_seeds": list(plan.policy_seeds),
                "video_policy_seeds": sorted(video_policy_seeds & set(plan.policy_seeds)),
                "recording_layout": (
                    "recordings/policy_seed_XX/env_YYYY.rrd; "
                    "viewer order is environment seed then policy seed"
                ),
                "execution": {
                    "num_gpus": num_gpus,
                    "parallelism": (
                        "environment_seed_process_shards" if num_gpus > 1 else "sequential"
                    ),
                    "environment_shards": [list(shard) for shard in environment_shards],
                },
                "checkpoint_ids": checkpoint_ids,
                "train_seed": train_seed,
                "dataset_revision": DATASET_REVISION,
                "trajectory_resample_points": TRAJECTORY_RESAMPLE_POINTS,
                "mode_distance_threshold": PRIMARY_MODE_THRESHOLD,
                "mode_distance_sensitivity": list(SENSITIVITY_THRESHOLDS),
                "bootstrap": {
                    "iterations": BOOTSTRAP_ITERATIONS,
                    "confidence": BOOTSTRAP_CONFIDENCE,
                    "seed": BOOTSTRAP_SEED,
                    "environment_weighting": "equal",
                },
                "runtime": runtime,
            },
        )
        _save_static_figures(temporary, diffusion_results, diversity_points)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=tuple(sorted(DIFFUSION_SAMPLING_PROTOCOLS)),
        default=FULL_DDPM_SAMPLING,
    )
    parser.add_argument("--bc-policy")
    parser.add_argument("--diffusion-policy", required=True)
    parser.add_argument("--num-environment-seeds", type=int)
    parser.add_argument("--num-policy-seeds", type=int)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--label", choices=("Smoke", "Closed-loop"), default="Closed-loop")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    configured = _configured_protocol(args.protocol)
    output_dir = args.output_dir or (default_output_root() / "pilot" / configured.output_directory)
    result = evaluate_multi_seed_rollouts(
        bc_policy_path=args.bc_policy,
        diffusion_policy_path=args.diffusion_policy,
        output_dir=output_dir.expanduser().resolve(),
        protocol=args.protocol,
        num_environment_seeds=args.num_environment_seeds,
        num_policy_seeds=args.num_policy_seeds,
        num_gpus=args.num_gpus,
        train_seed=args.train_seed,
        episode_length=args.episode_length,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(f"E2 {configured.name} output: {result}")


if __name__ == "__main__":
    main()
