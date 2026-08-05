"""Offline RF7 route-condition ablation gate across every run checkpoint.

This module implements the gate described in
plans/2026-08-03-RF7零场消融与条件敏感性诊断.md: hold the dataset middle-frame
observation, the DDPM initial noise, and the reverse noise stream fixed for
each episode, then vary only the commanded route and measure action-chunk
distances in denormalized PushT pixels. No rollout, no model selection, and no
training are performed here. With ``--num-gpus>1`` checkpoints are distributed
across the requested GPUs through a process pool; each worker pins one GPU and
reproduces the same per-episode seeds as the single-GPU path.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION
from tqdm import tqdm

from robot_practice.policies.route_field_bc_chunk.modeling_route_field_bc_chunk import (  # noqa: F401
    RouteFieldBCChunkPolicy,
)
from robot_practice.policies.route_field_diffusion.modeling_route_field_diffusion import (  # noqa: F401
    RouteFieldDiffusionPolicy,
)

from .checkpoint_selection import (
    _batched_inputs,
    _frame_lookup,
    _stack_images,
    _validate_num_gpus,
    discover_checkpoints,
)
from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    default_dataset_root,
    default_output_root,
    practice_root,
)
from .dataset_split import load_split
from .e5_checkpoint_selection import POLICY_SEED
from .e5_runtime import move_processed_batch
from .noise import (
    fixed_observation_e1a_reverse_seed,
    fixed_observation_initial_noise_seed,
    make_initial_noise,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import controlled_sample, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .route_field_checkpoint_selection import (
    MODEL_BC,
    MODEL_DIFFUSION,
    validate_route_field_policy,
)
from .route_field_data import ROUTE_FIELD_CHANNELS, ROUTE_FIELD_GRID_SIZE, reverse_macro_route
from .route_field_dataset import load_route_field_sidecar
from .route_field_episodes import route_field_tensor
from .route_field_runtime import add_route_field

CONDITION_ORACLE = "oracle"
CONDITION_ZERO = "zero"
CONDITION_REVERSED = "reversed"
CONDITION_SHUFFLED = "shuffled"
CONDITION_OBSERVATION_SWAP = "observation_swap"
CONDITION_ORDER = (
    CONDITION_ORACLE,
    CONDITION_ZERO,
    CONDITION_REVERSED,
    CONDITION_SHUFFLED,
    CONDITION_OBSERVATION_SWAP,
)
MIN_REVERSED_ROUTE_CELLS = 2
ABSOLUTE_THRESHOLD_PX = 1.0
RATIO_THRESHOLD = 0.05
DEFAULT_NUM_INFERENCE_STEPS = 100
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AblationTask:
    model_name: str
    step: int
    path: Path


@dataclass(frozen=True)
class AblationWorkerConfig:
    dataset_root: Path
    episode_indices: tuple[int, ...]
    route_cells_by_episode: Mapping[int, tuple[int, ...]]
    num_inference_steps: int


@dataclass(frozen=True)
class AblationWorkerEvent:
    kind: str
    gpu_index: int
    model: str | None = None
    step: int | None = None
    rows: tuple[dict[str, Any], ...] = ()
    error: str | None = None


def _repo_relative(path: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(practice_root()))
    except ValueError:
        return str(path.expanduser().resolve())


def _paired_episode_index(ordinal: int, episode_count: int) -> int:
    """Return the deterministic pairing used by shuffled and observation_swap."""

    if episode_count < 2:
        raise ValueError("Condition pairing requires at least two episodes")
    return (ordinal + episode_count // 2) % episode_count


def _conditions_for_episode(
    *,
    route_cells: Sequence[int],
    paired_route_cells: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Build the five commanded conditions for one episode row.

    ``reversed`` is skipped when the route has a single cell because reversing
    it would reproduce the identical field. Every other condition is always
    present, including ``observation_swap`` which keeps the oracle route but
    substitutes the paired episode's observation.
    """

    oracle_cells = tuple(int(cell) for cell in route_cells)
    paired_cells = tuple(int(cell) for cell in paired_route_cells)
    if not oracle_cells:
        raise ValueError("An ablation episode requires a non-empty macro route")
    if not paired_cells:
        raise ValueError("An ablation episode requires a non-empty paired route")
    conditions: list[dict[str, Any]] = [
        {
            "label": CONDITION_ORACLE,
            "field": route_field_tensor(oracle_cells),
            "route_length": len(oracle_cells),
            "observation_source": "own",
        },
        {
            "label": CONDITION_ZERO,
            "field": torch.zeros(
                len(ROUTE_FIELD_CHANNELS),
                ROUTE_FIELD_GRID_SIZE,
                ROUTE_FIELD_GRID_SIZE,
                dtype=torch.float32,
            ),
            "route_length": 0,
            "observation_source": "own",
        },
    ]
    if len(oracle_cells) >= MIN_REVERSED_ROUTE_CELLS:
        conditions.append(
            {
                "label": CONDITION_REVERSED,
                "field": route_field_tensor(reverse_macro_route(oracle_cells)),
                "route_length": len(oracle_cells),
                "observation_source": "own",
            }
        )
    conditions.extend(
        [
            {
                "label": CONDITION_SHUFFLED,
                "field": route_field_tensor(paired_cells),
                "route_length": len(paired_cells),
                "observation_source": "own",
            },
            {
                "label": CONDITION_OBSERVATION_SWAP,
                "field": route_field_tensor(oracle_cells),
                "route_length": len(oracle_cells),
                "observation_source": "paired",
            },
        ]
    )
    return tuple(conditions)


def _prepare_middle_frame(
    *,
    dataset: LeRobotDataset,
    local_indices: Sequence[int],
    policy: torch.nn.Module,
    preprocessor: Any,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], int]:
    """Preprocess one episode's dataset middle frame into a policy-ready batch."""

    if not local_indices:
        raise ValueError("Episode has no frames")
    local_index = int(local_indices[len(local_indices) // 2])
    item = dataset[local_index]
    raw_batch = _batched_inputs(item, policy)
    raw_batch[ACTION] = item[ACTION].unsqueeze(0)
    processed = preprocessor(raw_batch)
    if not isinstance(processed, dict):
        raise TypeError("Policy preprocessor must return a dictionary")
    processed = move_processed_batch(processed, device=device)
    prepared = _stack_images(policy, processed)
    frame_index = int(dataset.hf_dataset[local_index]["frame_index"])
    return prepared, frame_index


def _sample_action_chunk(
    *,
    loaded: Any,
    prepared: dict[str, torch.Tensor],
    initial_noise: torch.Tensor | None,
    reverse_seed: int | None,
) -> np.ndarray:
    """Run one forward pass and return the denormalized [8,2] action chunk."""

    with torch.inference_mode():
        if isinstance(loaded.policy, DiffusionPolicy):
            if initial_noise is None or reverse_seed is None:
                raise RuntimeError("Diffusion sampling requires initial noise and reverse seed")
            sampled = controlled_sample(
                loaded.policy,
                prepared,
                initial_noise=initial_noise.clone(),
                reverse_seed=reverse_seed,
            )
            normalized = sampled.actions
        else:
            start = int(loaded.config.n_obs_steps) - 1
            end = start + int(loaded.config.n_action_steps)
            normalized = loaded.policy.predict_full_action_horizon(prepared)[:, start:end]
    denormalized = loaded.postprocessor(normalized)[0].detach().cpu().numpy()
    if denormalized.ndim != 2 or denormalized.shape[-1] != 2:
        raise ValueError(f"Expected a denormalized [steps,2] chunk, got {denormalized.shape}")
    return denormalized


def _chunk_distances(
    condition_chunk: np.ndarray,
    oracle_chunk: np.ndarray,
) -> dict[str, float]:
    """Measure chunk-level differences in denormalized PushT pixels."""

    if condition_chunk.shape != oracle_chunk.shape:
        raise ValueError(
            f"Condition and oracle chunks disagree: {condition_chunk.shape} vs {oracle_chunk.shape}"
        )
    per_step = np.linalg.norm(condition_chunk - oracle_chunk, axis=-1)
    return {
        "mean_step_l2": float(per_step.mean()),
        "chunk_l2": float(np.linalg.norm(condition_chunk - oracle_chunk)),
        "endpoint_l2": float(per_step[-1]),
    }


def _evaluate_checkpoint(
    *,
    model_name: str,
    step: int,
    path: Path,
    dataset_root: Path,
    episode_indices: Sequence[int],
    route_cells_by_episode: Mapping[int, tuple[int, ...]],
    num_inference_steps: int,
    device: str,
    on_sample: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Measure every condition for every episode of one checkpoint."""

    loaded = load_policy_with_processors(path, local_files_only=True, device=device)
    try:
        validate_route_field_policy(loaded.policy, expected_model=model_name)
        is_diffusion = isinstance(loaded.policy, DiffusionPolicy)
        if is_diffusion:
            replace_scheduler(
                loaded.policy,
                scheduler_name="DDPM",
                num_inference_steps=num_inference_steps,
            )

        metadata = LeRobotDatasetMetadata(
            DATASET_REPO_ID,
            root=dataset_root,
            revision=DATASET_REVISION,
        )
        delta_timestamps = resolve_delta_timestamps(loaded.config, metadata)
        dataset = LeRobotDataset(
            DATASET_REPO_ID,
            root=dataset_root,
            episodes=list(episode_indices),
            delta_timestamps=delta_timestamps,
            revision=DATASET_REVISION,
        )
        lookup = _frame_lookup(dataset)
        missing = sorted(set(episode_indices) - set(lookup))
        if missing:
            raise ValueError(f"Dataset is missing requested episodes: {missing}")

        checkpoint_id = sha256_file(path / "model.safetensors")
        policy_device = get_device_from_parameters(loaded.policy)
        dtype = get_dtype_from_parameters(loaded.policy)
        rows: list[dict[str, Any]] = []
        for ordinal, episode_index in enumerate(episode_indices):
            paired_ordinal = _paired_episode_index(ordinal, len(episode_indices))
            paired_episode = int(episode_indices[paired_ordinal])
            if paired_episode == int(episode_index):
                raise ValueError(f"Episode {episode_index} was paired with itself")

            own_prepared, own_frame = _prepare_middle_frame(
                dataset=dataset,
                local_indices=lookup[int(episode_index)],
                policy=loaded.policy,
                preprocessor=loaded.preprocessor,
                device=policy_device,
            )
            paired_prepared, _ = _prepare_middle_frame(
                dataset=dataset,
                local_indices=lookup[paired_episode],
                policy=loaded.policy,
                preprocessor=loaded.preprocessor,
                device=policy_device,
            )

            initial_noise: torch.Tensor | None = None
            initial_noise_seed: int | None = None
            reverse_seed: int | None = None
            if is_diffusion:
                initial_noise_seed = fixed_observation_initial_noise_seed(
                    episode_index=int(episode_index),
                    frame_index=own_frame,
                    sample_ordinal=POLICY_SEED,
                )
                reverse_seed = fixed_observation_e1a_reverse_seed(
                    episode_index=int(episode_index),
                    frame_index=own_frame,
                )
                initial_noise = make_initial_noise(
                    seed=initial_noise_seed,
                    batch_size=1,
                    device=policy_device,
                    dtype=dtype,
                    horizon=int(loaded.config.horizon),
                    action_dim=int(loaded.config.action_feature.shape[0]),
                ).tensor

            conditions = _conditions_for_episode(
                route_cells=route_cells_by_episode[int(episode_index)],
                paired_route_cells=route_cells_by_episode[paired_episode],
            )
            chunks: dict[str, np.ndarray] = {}
            for condition in conditions:
                source = (
                    own_prepared
                    if condition["observation_source"] == "own"
                    else paired_prepared
                )
                prepared = add_route_field(source, field=condition["field"])
                chunks[str(condition["label"])] = _sample_action_chunk(
                    loaded=loaded,
                    prepared=prepared,
                    initial_noise=initial_noise,
                    reverse_seed=reverse_seed,
                )
                if on_sample is not None:
                    on_sample()

            oracle_chunk = chunks[CONDITION_ORACLE]
            oracle_chunk_norm = float(np.linalg.norm(oracle_chunk))
            for condition in conditions:
                label = str(condition["label"])
                distances = _chunk_distances(chunks[label], oracle_chunk)
                rows.append(
                    {
                        "model": model_name,
                        "step": step,
                        "checkpoint_id": checkpoint_id,
                        "episode_index": int(episode_index),
                        "frame_index": own_frame,
                        "route_length": int(condition["route_length"]),
                        "condition": label,
                        "mean_step_l2": distances["mean_step_l2"],
                        "chunk_l2": distances["chunk_l2"],
                        "endpoint_l2": distances["endpoint_l2"],
                        "oracle_chunk_norm": oracle_chunk_norm,
                        "policy_seed": POLICY_SEED,
                        "initial_noise_seed": initial_noise_seed,
                        "reverse_seed": reverse_seed,
                        "scheduler": "DDPM" if is_diffusion else None,
                        "num_inference_steps": (
                            num_inference_steps if is_diffusion else None
                        ),
                    }
                )
        return rows
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _checkpoint_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per (checkpoint, condition) and attach the two ratio criteria."""

    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["step"]), str(row["condition"]))].append(row)
    checkpoint_keys = sorted(
        {(str(row["model"]), int(row["step"])) for row in rows},
        key=lambda key: (key[0], key[1]),
    )

    def mean_condition(key: tuple[str, int], condition: str) -> float | None:
        group = grouped.get((key[0], key[1], condition))
        if not group:
            return None
        return float(np.mean([float(row["mean_step_l2"]) for row in group]))

    ratios: dict[tuple[str, int], dict[str, float | None]] = {}
    for key in checkpoint_keys:
        observation_swap = mean_condition(key, CONDITION_OBSERVATION_SWAP)
        ratios[key] = {
            "zero": (
                None
                if observation_swap in (None, 0.0)
                else (
                    mean_condition(key, CONDITION_ZERO) / observation_swap
                    if mean_condition(key, CONDITION_ZERO) is not None
                    else None
                )
            ),
            "reversed": (
                None
                if observation_swap in (None, 0.0)
                else (
                    mean_condition(key, CONDITION_REVERSED) / observation_swap
                    if mean_condition(key, CONDITION_REVERSED) is not None
                    else None
                )
            ),
        }

    result: list[dict[str, Any]] = []
    for key in checkpoint_keys:
        for condition in CONDITION_ORDER:
            group = grouped.get((key[0], key[1], condition))
            if not group:
                continue
            values = [float(row["mean_step_l2"]) for row in group]
            chunk_values = [float(row["chunk_l2"]) for row in group]
            endpoint_values = [float(row["endpoint_l2"]) for row in group]
            oracle_norms = [float(row["oracle_chunk_norm"]) for row in group]
            result.append(
                {
                    "model": key[0],
                    "step": key[1],
                    "condition": condition,
                    "count": len(group),
                    "mean_step_l2": float(np.mean(values)),
                    "std_step_l2": float(np.std(values)),
                    "min_step_l2": float(np.min(values)),
                    "max_step_l2": float(np.max(values)),
                    "mean_chunk_l2": float(np.mean(chunk_values)),
                    "mean_endpoint_l2": float(np.mean(endpoint_values)),
                    "mean_oracle_chunk_norm": float(np.mean(oracle_norms)),
                    "zero_observation_swap_ratio": ratios[key]["zero"],
                    "reversed_observation_swap_ratio": ratios[key]["reversed"],
                }
            )
    return result


def _evaluate_gate(
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the pre-registered thresholds without modifying them."""

    by_checkpoint: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in checkpoint_rows:
        by_checkpoint[(str(row["model"]), int(row["step"]))][str(row["condition"])] = float(
            row["mean_step_l2"]
        )

    zero_means = [values[CONDITION_ZERO] for values in by_checkpoint.values()]
    reversed_means = [values[CONDITION_REVERSED] for values in by_checkpoint.values()]
    observation_swap_means = [
        values[CONDITION_OBSERVATION_SWAP] for values in by_checkpoint.values()
    ]
    zero_absent = bool(zero_means) and all(
        value < ABSOLUTE_THRESHOLD_PX for value in zero_means
    )
    reversed_absent = bool(reversed_means) and all(
        value < ABSOLUTE_THRESHOLD_PX for value in reversed_means
    )

    ratios: list[tuple[tuple[str, int], float]] = []
    for key, values in by_checkpoint.items():
        observation_swap = values.get(CONDITION_OBSERVATION_SWAP)
        reversed_value = values.get(CONDITION_REVERSED)
        if (
            observation_swap is not None
            and reversed_value is not None
            and observation_swap > 0.0
        ):
            ratios.append((key, reversed_value / observation_swap))
    best_key, best_ratio = (
        max(ratios, key=lambda item: item[1]) if ratios else (None, None)
    )
    ratio_negligible = best_ratio is not None and best_ratio < RATIO_THRESHOLD

    absolute_negative = zero_absent and reversed_absent
    if absolute_negative:
        absolute_verdict = "unused"
    elif zero_absent or reversed_absent:
        absolute_verdict = "mixed"
    else:
        absolute_verdict = "used"
    ratio_verdict = (
        "negligible"
        if ratio_negligible
        else ("non_negligible" if best_ratio is not None else "missing")
    )
    if absolute_negative and ratio_negligible:
        gate_conclusion = "route_not_read"
    elif (not absolute_negative) and (not ratio_negligible):
        gate_conclusion = "route_read"
    else:
        gate_conclusion = "inconclusive"

    zero_ratios = []
    for key, values in by_checkpoint.items():
        observation_swap = values.get(CONDITION_OBSERVATION_SWAP)
        zero_value = values.get(CONDITION_ZERO)
        if observation_swap is not None and zero_value is not None and observation_swap > 0.0:
            zero_ratios.append((key, zero_value / observation_swap))
    best_zero_key, best_zero_ratio = (
        max(zero_ratios, key=lambda item: item[1]) if zero_ratios else (None, None)
    )

    return {
        "absolute_threshold_px": ABSOLUTE_THRESHOLD_PX,
        "ratio_threshold": RATIO_THRESHOLD,
        "zero_mean_step_l2_min_px": float(np.min(zero_means)) if zero_means else None,
        "zero_mean_step_l2_max_px": float(np.max(zero_means)) if zero_means else None,
        "reversed_mean_step_l2_min_px": (
            float(np.min(reversed_means)) if reversed_means else None
        ),
        "reversed_mean_step_l2_max_px": (
            float(np.max(reversed_means)) if reversed_means else None
        ),
        "observation_swap_mean_step_l2_min_px": (
            float(np.min(observation_swap_means)) if observation_swap_means else None
        ),
        "observation_swap_mean_step_l2_max_px": (
            float(np.max(observation_swap_means)) if observation_swap_means else None
        ),
        "zero_absent_all_checkpoints": zero_absent,
        "reversed_absent_all_checkpoints": reversed_absent,
        "absolute_verdict": absolute_verdict,
        "reversed_observation_swap_ratio_best": best_ratio,
        "reversed_observation_swap_ratio_best_model": best_key[0] if best_key else None,
        "reversed_observation_swap_ratio_best_step": best_key[1] if best_key else None,
        "zero_observation_swap_ratio_best": best_zero_ratio,
        "zero_observation_swap_ratio_best_model": (
            best_zero_key[0] if best_zero_key else None
        ),
        "zero_observation_swap_ratio_best_step": (
            best_zero_key[1] if best_zero_key else None
        ),
        "ratio_verdict": ratio_verdict,
        "gate_conclusion": gate_conclusion,
    }


def _ablation_worker(
    gpu_index: int,
    task_queue: Any,
    event_queue: Any,
    config: AblationWorkerConfig,
) -> None:
    torch.cuda.set_device(gpu_index)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    while True:
        task = task_queue.get()
        if task is None:
            return
        if not isinstance(task, AblationTask):
            event_queue.put(
                AblationWorkerEvent(
                    kind="error",
                    gpu_index=gpu_index,
                    error=f"Worker received invalid task: {task!r}",
                )
            )
            return
        try:
            rows = _evaluate_checkpoint(
                model_name=task.model_name,
                step=task.step,
                path=task.path,
                dataset_root=config.dataset_root,
                episode_indices=config.episode_indices,
                route_cells_by_episode=config.route_cells_by_episode,
                num_inference_steps=config.num_inference_steps,
                device=f"cuda:{gpu_index}",
                on_sample=lambda task=task: event_queue.put(
                    AblationWorkerEvent(
                        kind="progress",
                        gpu_index=gpu_index,
                        model=task.model_name,
                        step=task.step,
                    )
                ),
            )
        except Exception:  # noqa: BLE001 - relay the complete worker failure to the parent.
            event_queue.put(
                AblationWorkerEvent(
                    kind="error",
                    gpu_index=gpu_index,
                    model=task.model_name,
                    step=task.step,
                    error=traceback.format_exc(),
                )
            )
            return
        event_queue.put(
            AblationWorkerEvent(
                kind="result",
                gpu_index=gpu_index,
                model=task.model_name,
                step=task.step,
                rows=tuple(rows),
            )
        )


def _run_parallel(
    *,
    checkpoints: Mapping[str, tuple[tuple[int, Path], ...]],
    model_order: Sequence[str],
    config: AblationWorkerConfig,
    total_samples: int,
    num_gpus: int,
) -> list[dict[str, Any]]:
    """Distribute one process per GPU; each task is one checkpoint."""

    tasks = [
        AblationTask(model_name=model, step=step, path=path)
        for model in model_order
        for step, path in checkpoints[model]
    ]
    if not tasks:
        raise RuntimeError("No ablation tasks were created")
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    event_queue = context.Queue()
    worker_count = min(num_gpus, len(tasks))
    for task in tasks:
        task_queue.put(task)
    for _ in range(worker_count):
        task_queue.put(None)
    workers = [
        context.Process(
            target=_ablation_worker,
            args=(gpu_index, task_queue, event_queue, config),
            name=f"ablation-gpu-{gpu_index}",
        )
        for gpu_index in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    rows: list[dict[str, Any]] = []
    completed = 0
    try:
        with tqdm(
            total=total_samples,
            desc="RF7 condition ablation",
            unit="sample",
            dynamic_ncols=True,
        ) as progress:
            while completed < len(tasks):
                try:
                    event = event_queue.get(timeout=1.0)
                except Empty:
                    failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                    if failed:
                        details = ", ".join(
                            f"{worker.name} exitcode={worker.exitcode}" for worker in failed
                        )
                        raise RuntimeError(f"Ablation worker exited unexpectedly: {details}")
                    if not any(worker.is_alive() for worker in workers):
                        raise RuntimeError(
                            "All ablation workers exited before producing results"
                        )
                    continue
                if not isinstance(event, AblationWorkerEvent):
                    raise TypeError(f"Received invalid worker event: {event!r}")
                if event.kind == "error":
                    raise RuntimeError(
                        f"GPU {event.gpu_index} failed on {event.model} "
                        f"checkpoint {event.step:06d}:\n{event.error}"
                    )
                if event.kind == "progress":
                    progress.update(1)
                    if event.model is not None and event.step is not None:
                        progress.set_postfix(
                            gpu=event.gpu_index,
                            model=event.model,
                            checkpoint=f"{event.step:06d}",
                            refresh=True,
                        )
                    continue
                if event.kind == "result":
                    if not event.rows:
                        raise RuntimeError("Ablation worker returned an empty result")
                    rows.extend(event.rows)
                    completed += 1
                    continue
                raise RuntimeError(f"Unknown worker event kind: {event.kind!r}")
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        raise
    finally:
        for worker in workers:
            worker.join(timeout=10)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        task_queue.close()
        event_queue.close()

    failed = [worker for worker in workers if worker.exitcode != 0]
    if failed:
        details = ", ".join(f"{worker.name} exitcode={worker.exitcode}" for worker in failed)
        raise RuntimeError(f"Ablation worker shutdown failed: {details}")
    return rows


def run_route_field_condition_ablation(
    *,
    bc_run: Path | None,
    diffusion_run: Path | None,
    route_manifest: Path,
    dataset_root: Path,
    split_name: str,
    num_inference_steps: int,
    device: str | None,
    num_gpus: int,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    """Run the offline condition ablation for every discovered checkpoint."""

    if bc_run is None and diffusion_run is None:
        raise ValueError("At least one of --bc-run / --diffusion-run is required")
    if split_name not in {"development", "audit"}:
        raise ValueError("split_name must be development or audit")
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if isinstance(num_gpus, bool) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    if num_gpus > 1:
        if device is not None:
            raise ValueError("device cannot be set when num_gpus > 1")
        _validate_num_gpus(num_gpus)
        resolved_device: str | None = None
    else:
        resolved_device = device or "cuda:0"
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {resolved_device!r} but CUDA is unavailable")

    sidecar = load_route_field_sidecar(route_manifest)
    split = load_split()
    episode_indices = tuple(
        split.development if split_name == "development" else split.audit
    )
    route_cells_by_episode: dict[int, tuple[int, ...]] = {}
    for episode in episode_indices:
        assignment = sidecar.assignments.get(int(episode))
        if assignment is None:
            raise KeyError(f"Route-field sidecar has no episode {episode}")
        if assignment.split != split_name:
            raise ValueError(
                f"Episode {episode} sidecar split is {assignment.split!r}, "
                f"expected {split_name!r}"
            )
        if not assignment.route_cells:
            raise ValueError(f"Episode {episode} has an empty route field")
        route_cells_by_episode[int(episode)] = tuple(assignment.route_cells)

    checkpoints: dict[str, tuple[tuple[int, Path], ...]] = {}
    run_dirs: dict[str, Path] = {}
    if diffusion_run is not None:
        run_dirs[MODEL_DIFFUSION] = diffusion_run.expanduser().resolve()
        checkpoints[MODEL_DIFFUSION] = discover_checkpoints(run_dirs[MODEL_DIFFUSION])
    if bc_run is not None:
        run_dirs[MODEL_BC] = bc_run.expanduser().resolve()
        checkpoints[MODEL_BC] = discover_checkpoints(run_dirs[MODEL_BC])
    model_order = tuple(
        model for model in (MODEL_BC, MODEL_DIFFUSION) if model in checkpoints
    )

    single_cell_count = sum(
        1 for cells in route_cells_by_episode.values() if len(cells) < MIN_REVERSED_ROUTE_CELLS
    )
    total_samples = sum(
        len(checkpoints[model]) * (len(episode_indices) * len(CONDITION_ORDER) - single_cell_count)
        for model in model_order
    )
    runtime = collect_runtime_provenance(practice_root())

    destination = output_dir.expanduser().resolve()
    with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
        all_rows: list[dict[str, Any]] = []
        if num_gpus == 1:
            with tqdm(
                total=total_samples,
                desc="RF7 condition ablation",
                unit="sample",
                dynamic_ncols=True,
            ) as progress:
                for model_name in model_order:
                    for step, path in checkpoints[model_name]:
                        progress.set_postfix(
                            model=model_name,
                            checkpoint=f"{step:06d}",
                            refresh=True,
                        )
                        rows = _evaluate_checkpoint(
                            model_name=model_name,
                            step=step,
                            path=path,
                            dataset_root=dataset_root.expanduser().resolve(),
                            episode_indices=episode_indices,
                            route_cells_by_episode=route_cells_by_episode,
                            num_inference_steps=num_inference_steps,
                            device=resolved_device or "cuda:0",
                            on_sample=lambda: progress.update(1),
                        )
                        all_rows.extend(rows)
        else:
            all_rows = _run_parallel(
                checkpoints=checkpoints,
                model_order=model_order,
                config=AblationWorkerConfig(
                    dataset_root=dataset_root.expanduser().resolve(),
                    episode_indices=tuple(episode_indices),
                    route_cells_by_episode=route_cells_by_episode,
                    num_inference_steps=num_inference_steps,
                ),
                total_samples=total_samples,
                num_gpus=num_gpus,
            )
        all_rows.sort(
            key=lambda row: (
                str(row["model"]),
                int(row["step"]),
                int(row["episode_index"]),
                CONDITION_ORDER.index(str(row["condition"])),
            )
        )
        if not all_rows:
            raise RuntimeError("The ablation produced no rows")

        checkpoint_rows = _checkpoint_metrics(all_rows)
        gate = _evaluate_gate(checkpoint_rows)
        write_records_parquet(temporary / "condition_ablation.parquet", all_rows)
        write_records_parquet(temporary / "checkpoint_metrics.parquet", checkpoint_rows)
        metrics = {
            "experiment": "route_field_condition_ablation",
            "label": "Offline",
            "split": split_name,
            "episode_count": len(episode_indices),
            "checkpoint_count": sum(len(values) for values in checkpoints.values()),
            "sample_count": len(all_rows),
            "scheduler": "DDPM" if MODEL_DIFFUSION in checkpoints else None,
            "num_inference_steps": (
                num_inference_steps if MODEL_DIFFUSION in checkpoints else None
            ),
            "num_gpus": num_gpus,
            **gate,
        }
        comments = {
            "experiment": "Fixed-name identifier for this offline gate",
            "label": "Offline means no rollout was executed; never report success",
            "split": "Committed dataset split measured by this run",
            "episode_count": "Episodes per checkpoint in this split",
            "checkpoint_count": "Discovered checkpoints measured by this run",
            "sample_count": "Forward samples; reversed is skipped for single-cell routes",
            "scheduler": "Diffusion scheduler used at inference",
            "num_inference_steps": "DDPM reverse steps, matching the closed-loop protocol",
            "num_gpus": "GPUs used; >1 distributes one checkpoint per task",
            "absolute_threshold_px": "Pre-registered 1.0 px absolute difference threshold",
            "ratio_threshold": "Pre-registered 0.05 ratio threshold vs observation_swap",
            "zero_mean_step_l2_min_px": "Minimum checkpoint mean zero-vs-oracle step L2",
            "zero_mean_step_l2_max_px": "Maximum checkpoint mean zero-vs-oracle step L2",
            "reversed_mean_step_l2_min_px": "Minimum checkpoint mean reversed-vs-oracle step L2",
            "reversed_mean_step_l2_max_px": "Maximum checkpoint mean reversed-vs-oracle step L2",
            "observation_swap_mean_step_l2_min_px": (
                "Minimum checkpoint mean observation-swap step L2 (upper-bound scale reference)"
            ),
            "observation_swap_mean_step_l2_max_px": (
                "Maximum checkpoint mean observation-swap step L2 (upper-bound scale reference)"
            ),
            "zero_absent_all_checkpoints": (
                "True when every checkpoint has zero mean_step_l2 < 1.0 px"
            ),
            "reversed_absent_all_checkpoints": (
                "True when every checkpoint has reversed mean_step_l2 < 1.0 px"
            ),
            "absolute_verdict": (
                "unused/mixed/used from the two absolute thresholds only"
            ),
            "reversed_observation_swap_ratio_best": (
                "Largest reversed/observation_swap ratio across checkpoints"
            ),
            "reversed_observation_swap_ratio_best_model": (
                "Model of the best-ratio checkpoint"
            ),
            "reversed_observation_swap_ratio_best_step": (
                "Training step of the best-ratio checkpoint"
            ),
            "zero_observation_swap_ratio_best": (
                "Largest zero/observation_swap ratio across checkpoints (second ratio criterion)"
            ),
            "zero_observation_swap_ratio_best_model": (
                "Model of the best zero-ratio checkpoint"
            ),
            "zero_observation_swap_ratio_best_step": (
                "Training step of the best zero-ratio checkpoint"
            ),
            "ratio_verdict": (
                "negligible when the best reversed/observation_swap ratio is below 0.05"
            ),
            "gate_conclusion": (
                "route_not_read blocks RF7; route_read continues; inconclusive reports both"
            ),
        }
        write_metrics(temporary, metrics, comments=comments)
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "label": "Offline",
                "experiment": "route_field_condition_ablation",
                "dataset_revision": DATASET_REVISION,
                "split": split_name,
                "episode_indices": list(episode_indices),
                "pairing_rule": (
                    "episode ordinal i pairs with episode ordinal (i + n // 2) mod n "
                    "for shuffled routes and swapped observations"
                ),
                "observation_source": "dataset middle frame",
                "seeds": {
                    "policy_seed": POLICY_SEED,
                    "initial_noise_seed": (
                        "fixed_observation_initial_noise_seed(episode_index, frame_index, "
                        "sample_ordinal=policy_seed)"
                    ),
                    "reverse_seed": (
                        "fixed_observation_e1a_reverse_seed(episode_index, frame_index)"
                    ),
                    "shared_across_conditions": True,
                },
                "execution": {
                    "num_gpus": num_gpus,
                    "device": resolved_device,
                    "parallelism": (
                        "checkpoint_process_pool" if num_gpus > 1 else "sequential"
                    ),
                    "task_granularity": "one_model_checkpoint_all_split_episodes",
                },
                "conditions": [
                    {
                        "label": CONDITION_ORACLE,
                        "definition": "episode's own GT macro route",
                    },
                    {
                        "label": CONDITION_ZERO,
                        "definition": "all-zero [4,7,7] field",
                    },
                    {
                        "label": CONDITION_REVERSED,
                        "definition": (
                            "same-cell reversed route; skipped when the route has one cell"
                        ),
                    },
                    {
                        "label": CONDITION_SHUFFLED,
                        "definition": "paired episode's real route",
                    },
                    {
                        "label": CONDITION_OBSERVATION_SWAP,
                        "definition": (
                            "oracle route with the paired episode's observation"
                        ),
                    },
                ],
                "thresholds": {
                    "absolute_px": ABSOLUTE_THRESHOLD_PX,
                    "ratio": RATIO_THRESHOLD,
                },
                "interpretation_asymmetry": [
                    (
                        "zero distance ~= 0 is a strong conclusion that the route field is "
                        "definitively not read"
                    ),
                    (
                        "zero distance large is not a conclusion: the all-zero field is OOD "
                        "and was never seen in training; reversed/shuffled stay in-distribution"
                    ),
                    (
                        "reversed distance large shows the order/direction channels are read, "
                        "not that they are used correctly; signed correctness requires closed-loop"
                    ),
                    (
                        "chunk distance magnitude is not a goodness criterion: an episode-"
                        "fingerprint memorizer also produces a nonzero reversed distance"
                    ),
                    (
                        "observation_swap distance is an upper-bound scale reference only; it "
                        "includes state differences and is not the expected route effect size"
                    ),
                ],
                "gate": gate,
                "route_manifest": _repo_relative(route_manifest),
                "route_manifest_sha256": sha256_file(route_manifest),
                "models": {
                    model: {
                        "run_dir": _repo_relative(run_dirs[model]),
                        "checkpoints": [
                            {
                                "step": int(step),
                                "checkpoint_id": next(
                                    row["checkpoint_id"]
                                    for row in all_rows
                                    if row["model"] == model and int(row["step"]) == int(step)
                                ),
                            }
                            for step, _ in checkpoints[model]
                        ],
                    }
                    for model in model_order
                },
                "runtime": runtime,
            },
        )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bc-run", type=Path, default=None)
    parser.add_argument("--diffusion-run", type=Path, default=None)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--split",
        choices=("development", "audit"),
        default="development",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "route_field_condition_ablation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_route_field_condition_ablation(
        bc_run=(
            None if args.bc_run is None else args.bc_run.expanduser().resolve()
        ),
        diffusion_run=(
            None
            if args.diffusion_run is None
            else args.diffusion_run.expanduser().resolve()
        ),
        route_manifest=args.route_manifest.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        split_name=args.split,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        num_gpus=args.num_gpus,
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(f"RF7 route-condition ablation output: {result}")


if __name__ == "__main__":
    main()
