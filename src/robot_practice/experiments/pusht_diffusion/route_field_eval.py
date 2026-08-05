"""Held-out RF7 validation: oracle route, Route A/B, and condition response."""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
import re
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_dtype_from_parameters
from lerobot.utils.constants import ACTION
from tqdm.auto import tqdm

from robot_practice.policies.route_field_bc_chunk.modeling_route_field_bc_chunk import (
    RouteFieldBCChunkPolicy,
)
from robot_practice.policies.route_field_diffusion.modeling_route_field_diffusion import (  # noqa: F401
    RouteFieldDiffusionPolicy,
)

from .checkpoint_selection import _batched_inputs, _frame_lookup, _stack_images
from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    default_dataset_root,
    default_output_root,
    load_json_object,
    practice_root,
)
from .dataset_split import load_split
from .e5_checkpoint_selection import EPISODE_LENGTH, NUM_INFERENCE_STEPS, POLICY_SEED
from .e5_oracle_mode import preflight_reconstruction
from .noise import fixed_observation_noise_seed, fixed_observation_reverse_seed, make_initial_noise
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import controlled_sample, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rerun_visualization import record_single_rollout
from .route_field_checkpoint_selection import (
    MODEL_BC,
    MODEL_DIFFUSION,
    validate_route_field_policy,
)
from .route_field_data import ROUTE_FIELD_GRID_SIZE, position_to_cell
from .route_field_episodes import (
    CONDITION_ORACLE,
    CONDITION_ORDER,
    CONDITION_PERMUTED,
    CONDITION_REVERSED,
    CONDITION_ROLES,
    OBEDIENCE_PROBE_ROLE,
    RouteFieldEpisode,
    achieved_route_metrics,
    episode_condition_labels,
    episode_route_conditions,
    load_route_field_episodes,
    run_route_field_rollout,
)
from .route_field_figures import (
    write_condition_pair_figure,
    write_contact_sheet,
    write_rollout_figure,
)
from .route_field_runtime import add_route_field

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MODEL_DISPLAY_NAMES = {
    MODEL_BC: "BC",
    MODEL_DIFFUSION: "DP",
}
_PROBE_CONDITIONS = tuple(
    condition for condition in CONDITION_ORDER if CONDITION_ROLES[condition] == OBEDIENCE_PROBE_ROLE
)


@dataclass(frozen=True)
class RouteFieldEvalWorkerPlan:
    gpu_index: int
    model: str
    episode_indices: tuple[int, ...]
    leader: bool


@dataclass(frozen=True)
class RouteFieldEvalWorkerConfig:
    selected: Mapping[str, Mapping[str, Any]]
    route_manifest: Path
    dataset_root: Path
    temporary: Path
    episodes: tuple[RouteFieldEpisode, ...]
    episode_length: int
    num_inference_steps: int
    fixed_observation_episodes: int
    figures_enabled: bool


@dataclass(frozen=True)
class RouteFieldEvalWorkerEvent:
    kind: str
    gpu_index: int
    model: str
    phase: str | None = None
    episode_index: int | None = None
    condition: str | None = None
    closed_loop_rows: tuple[dict[str, Any], ...] = ()
    condition_response_rows: tuple[dict[str, Any], ...] = ()
    error: str | None = None


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _repository_checkpoint_path(
    *,
    repository_root: Path,
    model: str,
    step: int,
) -> Path:
    return (
        repository_root
        / "outputs"
        / "pusht_diffusion"
        / "pilot"
        / "route_field_7x7"
        / "training"
        / model
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    ).resolve()


def _resolve_selected_row(
    model: str,
    row: object,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError(f"Selected {model} entry must be an object")
    if row.get("model") != model:
        raise ValueError(f"Selected {model} entry must declare model={model!r}")
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError(f"Selected {model} step must be a positive integer")
    checkpoint_id = row.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or _SHA256_PATTERN.fullmatch(checkpoint_id) is None:
        raise ValueError(f"Selected {model} checkpoint_id must be a lowercase SHA256 digest")
    recorded_value = row.get("checkpoint_path")
    if not isinstance(recorded_value, str) or not recorded_value.strip():
        raise ValueError(f"Selected {model} checkpoint_path must be a non-empty string")

    recorded_path = Path(recorded_value).expanduser().resolve()
    candidates = (
        ("recorded", recorded_path),
        (
            "repository_relative",
            _repository_checkpoint_path(
                repository_root=repository_root,
                model=model,
                step=step,
            ),
        ),
    )
    seen: set[Path] = set()
    attempts = []
    hash_mismatch = False
    for resolution, path in candidates:
        if path in seen:
            continue
        seen.add(path)
        weights = path / "model.safetensors"
        if not weights.is_file():
            attempts.append(f"{resolution}={path} (missing model.safetensors)")
            continue
        actual_hash = sha256_file(weights)
        if actual_hash != checkpoint_id:
            hash_mismatch = True
            attempts.append(f"{resolution}={path} (SHA256 mismatch)")
            continue
        resolved = dict(row)
        resolved["recorded_checkpoint_path"] = recorded_value
        resolved["checkpoint_path"] = str(path)
        resolved["checkpoint_path_resolution"] = resolution
        return resolved

    detail = "; ".join(attempts)
    message = f"No selected {model} checkpoint matched checkpoint_id; {detail}"
    if hash_mismatch:
        raise ValueError(message)
    raise FileNotFoundError(message)


def _selected_rows(selection_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json_object(selection_dir / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Checkpoint selection is not complete: {selection_dir}")
    if manifest.get("experiment") != "route_field_oracle_development_checkpoint_selection":
        raise ValueError(f"Not an RF7 oracle checkpoint selection: {selection_dir}")
    payload = load_json_object(selection_dir / "selected_checkpoints.json")
    allowed = {MODEL_BC, MODEL_DIFFUSION}
    if not isinstance(payload, dict) or not set(payload) or not set(payload) <= allowed:
        raise ValueError(f"Selected checkpoints must be a non-empty subset of {sorted(allowed)}")
    result = {}
    for model in sorted(payload):
        result[model] = _resolve_selected_row(
            model,
            payload[model],
            repository_root=practice_root(),
        )
    return result


def _validate_execution_request(
    *,
    num_gpus: int,
    device: str | None,
) -> str | None:
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    if num_gpus == 1:
        return device or "cuda:0"
    if device not in (None, "cuda:0"):
        raise ValueError("device cannot be set when num_gpus > 1")
    if not torch.cuda.is_available():
        raise RuntimeError("multi-GPU RF7 validation requires CUDA")
    visible = torch.cuda.device_count()
    if num_gpus > visible:
        raise RuntimeError(
            f"num_gpus={num_gpus} requested, but only {visible} CUDA device(s) are visible"
        )
    return None


def _configure_loaded_policy(
    loaded: Any,
    *,
    model: str,
    num_inference_steps: int,
) -> None:
    validate_route_field_policy(loaded.policy, expected_model=model)
    if model == MODEL_DIFFUSION:
        replace_scheduler(
            loaded.policy,
            scheduler_name="DDPM",
            num_inference_steps=num_inference_steps,
        )


def _recording_path(
    root: Path,
    *,
    model: str,
    condition: str,
    episode_index: int,
) -> Path:
    return root / model / f"condition_{condition}" / f"episode_{episode_index:06d}.rrd"


def _recording_name(
    *,
    model: str,
    condition: str,
    episode_index: int,
    success: bool,
) -> str:
    outcome = "success" if success else "failure"
    return (
        f"RF7 eval | {_MODEL_DISPLAY_NAMES[model]} | command {condition} | "
        f"episode {episode_index} | {outcome}"
    )


def _recording_metadata(
    *,
    model: str,
    condition: str,
    episode_index: int,
    success: bool,
) -> dict[str, object]:
    return {
        "model": f"{_MODEL_DISPLAY_NAMES[model]} ({model})",
        "commanded_condition": condition,
        "episode": episode_index,
        "success": success,
    }


def _closed_loop_row(
    *,
    result: Any,
    episode: RouteFieldEpisode,
    selected: Mapping[str, Any],
    condition: str,
    requested_cells: Sequence[int],
) -> dict[str, Any]:
    latencies = [float(record.latency_ms) for record in result.replans]
    route_metrics = achieved_route_metrics(
        result.block_path,
        requested_cells=requested_cells,
        oracle_cells=episode.route_cells,
    )
    recording_name = _recording_name(
        model=result.model,
        condition=condition,
        episode_index=episode.episode_index,
        success=bool(result.success),
    )
    recording_metadata = _recording_metadata(
        model=result.model,
        condition=condition,
        episode_index=episode.episode_index,
        success=bool(result.success),
    )
    return {
        "model": result.model,
        "selected_step": int(selected["step"]),
        "checkpoint_id": selected["checkpoint_id"],
        "episode_index": episode.episode_index,
        "split": episode.split,
        "commanded_condition": condition,
        "condition_role": CONDITION_ROLES[condition],
        "initial_block_cell": position_to_cell(
            episode.block_center_workspace,
            grid_size=ROUTE_FIELD_GRID_SIZE,
        ),
        "initial_state_source": "dataset_first_frame_reconstruction",
        "policy_seed": POLICY_SEED,
        "success": result.success,
        "sum_reward": result.sum_reward,
        "max_reward": result.max_reward,
        "final_reward": result.final_reward,
        "episode_length": result.episode_length,
        "termination_reason": result.termination_reason,
        "num_replans": len(result.replans),
        "replan_latencies_ms": latencies,
        "mean_replan_latency_ms": float(np.mean(latencies)) if latencies else None,
        "p95_replan_latency_ms": (float(np.percentile(latencies, 95)) if latencies else None),
        "scheduler": result.scheduler,
        "num_inference_steps": result.num_inference_steps,
        "recording_name": recording_name,
        "recording_metadata": "\n".join(
            f"{key}: {value}" for key, value in recording_metadata.items()
        ),
        "recording_has_identity_panel": True,
        **route_metrics,
    }


def _fixed_observation_batch(
    loaded: Any,
    *,
    dataset: Any,
    lookup: Mapping[int, Sequence[int]],
    episode_index: int,
) -> dict[str, torch.Tensor]:
    """Build one batched middle-frame observation for a fixed-input probe.

    ``lookup`` must map an episode index to all of its dataset row indices, so
    the middle frame can be selected. The returned batch always carries an
    explicit leading batch dimension, which the policies require.
    """

    indices = list(lookup[int(episode_index)])
    if not indices:
        raise ValueError(f"Episode {episode_index} has no dataset frames")
    item = dataset[indices[len(indices) // 2]]
    raw_batch = _batched_inputs(item, loaded.policy)
    action = item[ACTION]
    if not isinstance(action, torch.Tensor):
        raise TypeError("Dataset item action must be a tensor")
    raw_batch[ACTION] = action.unsqueeze(0)
    return raw_batch


def _predict_normalized_chunk(
    loaded: Any,
    *,
    prepared: Mapping[str, torch.Tensor],
    ordinal: int,
    device: str,
    start: int,
    end: int,
) -> torch.Tensor:
    with torch.inference_mode():
        if isinstance(loaded.policy, RouteFieldBCChunkPolicy):
            return loaded.policy.predict_full_action_horizon(prepared)[:, start:end]
        if isinstance(loaded.policy, DiffusionPolicy):
            explicit = make_initial_noise(
                seed=fixed_observation_noise_seed(
                    candidate_ordinal=ordinal,
                    sample_ordinal=POLICY_SEED,
                ),
                batch_size=1,
                device=device,
                dtype=get_dtype_from_parameters(loaded.policy),
                horizon=loaded.config.horizon,
                action_dim=loaded.config.action_feature.shape[0],
            )
            return controlled_sample(
                loaded.policy,
                prepared,
                initial_noise=explicit.tensor,
                reverse_seed=fixed_observation_reverse_seed(candidate_ordinal=ordinal),
            ).actions
        raise TypeError(f"Unsupported RF7 policy: {type(loaded.policy).__name__}")


def _evaluate_condition_response(
    loaded: Any,
    *,
    dataset_root: Path,
    episodes: Sequence[RouteFieldEpisode],
    cap: int,
    num_inference_steps: int,
    device: str,
) -> tuple[dict[str, Any], ...]:
    """Fixed-observation forward-vs-reversed action-chunk gate.

    This is the cheapest decisive check in the protocol: the observation, the
    Diffusion initial noise, and the reverse noise stream are all held fixed,
    and only the commanded route is replaced by its reversal. The reversal has
    the same occupancy support, so a near-zero chunk distance means the ordering
    and direction channels are not read at all, and no amount of closed-loop
    rollouts can then demonstrate route control.
    """

    if cap <= 0:
        return ()
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_root,
        revision=DATASET_REVISION,
    )
    delta_timestamps = resolve_delta_timestamps(loaded.config, metadata)
    eligible = tuple(
        episode for episode in episodes if CONDITION_REVERSED in episode_condition_labels(episode)
    )[:cap]
    if not eligible:
        return ()
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        episodes=[episode.episode_index for episode in eligible],
        delta_timestamps=delta_timestamps,
        revision=DATASET_REVISION,
    )
    lookup = _frame_lookup(dataset)
    start = loaded.config.n_obs_steps - 1
    end = start + loaded.config.n_action_steps
    rows = []
    for ordinal, episode in enumerate(eligible):
        raw_batch = _fixed_observation_batch(
            loaded,
            dataset=dataset,
            lookup=lookup,
            episode_index=episode.episode_index,
        )
        base = loaded.preprocessor(raw_batch)
        conditions = {condition.label: condition for condition in episode_route_conditions(episode)}
        chunks: dict[str, np.ndarray] = {}
        for label in (CONDITION_ORACLE, CONDITION_REVERSED):
            processed = add_route_field(base, field=conditions[label].field)
            prepared = _stack_images(loaded.policy, processed)
            normalized_prediction = _predict_normalized_chunk(
                loaded,
                prepared=prepared,
                ordinal=ordinal,
                device=device,
                start=start,
                end=end,
            )
            chunks[label] = loaded.postprocessor(normalized_prediction).detach().cpu().numpy()[0]
        difference = chunks[CONDITION_REVERSED] - chunks[CONDITION_ORACLE]
        rows.append(
            {
                "model": loaded.policy.config.type,
                "episode_index": episode.episode_index,
                "split": episode.split,
                "comparison": f"{CONDITION_REVERSED}_minus_{CONDITION_ORACLE}",
                "observation_source": "dataset_middle_frame",
                "oracle_route_cells": list(conditions[CONDITION_ORACLE].route_cells),
                "reversed_route_cells": list(conditions[CONDITION_REVERSED].route_cells),
                "route_length": len(conditions[CONDITION_ORACLE].route_cells),
                "policy_seed": POLICY_SEED,
                "initial_noise_seed": (
                    fixed_observation_noise_seed(
                        candidate_ordinal=ordinal,
                        sample_ordinal=POLICY_SEED,
                    )
                    if isinstance(loaded.policy, DiffusionPolicy)
                    else None
                ),
                "reverse_seed": (
                    fixed_observation_reverse_seed(candidate_ordinal=ordinal)
                    if isinstance(loaded.policy, DiffusionPolicy)
                    else None
                ),
                "scheduler": (
                    str(loaded.policy.diffusion.noise_scheduler.__class__.__name__)
                    if isinstance(loaded.policy, DiffusionPolicy)
                    else None
                ),
                "num_inference_steps": (
                    num_inference_steps if isinstance(loaded.policy, DiffusionPolicy) else None
                ),
                "chunk_l2_distance": float(np.linalg.norm(difference)),
                "endpoint_distance": float(np.linalg.norm(difference[-1])),
                "max_chunk_distance": float(np.max(np.linalg.norm(difference, axis=-1))),
            }
        )
    return tuple(rows)


def _evaluate_episode_conditions(
    loaded: Any,
    *,
    episode: RouteFieldEpisode,
    selected: Mapping[str, Any],
    temporary: Path,
    episode_length: int,
    figures_enabled: bool,
) -> tuple[dict[str, Any], ...]:
    """Roll out every condition this episode supports from one initial state.

    All conditions share the environment seed and the policy seed, so for
    Diffusion the initial noise and the reverse noise stream are identical
    across conditions. The commanded route is the only difference.
    """

    rows = []
    results_by_condition: dict[str, tuple[Any, dict[str, Any]]] = {}
    for condition in episode_route_conditions(episode):
        result = run_route_field_rollout(
            loaded,
            episode=episode,
            field=condition.field,
            condition_label=condition.label,
            record_frames=True,
            episode_length=episode_length,
            policy_seed=POLICY_SEED,
        )
        row = _closed_loop_row(
            result=result,
            episode=episode,
            selected=selected,
            condition=condition.label,
            requested_cells=condition.route_cells,
        )
        record_single_rollout(
            output_path=_recording_path(
                temporary / "recordings",
                model=result.model,
                condition=condition.label,
                episode_index=episode.episode_index,
            ),
            rollout=result,
            recording_id=(
                f"rf7_eval_{result.model}_{condition.label}_episode_{episode.episode_index:06d}"
            ),
            recording_name=str(row["recording_name"]),
            metadata=_recording_metadata(
                model=result.model,
                condition=condition.label,
                episode_index=episode.episode_index,
                success=bool(result.success),
            ),
            show_metadata=True,
        )
        rows.append(row)
        results_by_condition[condition.label] = (result, row)
        if figures_enabled:
            write_rollout_figure(
                _figure_path(
                    temporary / "figures",
                    model=result.model,
                    condition=condition.label,
                    episode_index=episode.episode_index,
                    filename=f"episode_{episode.episode_index:06d}.png",
                ),
                model_display=_MODEL_DISPLAY_NAMES[result.model],
                episode_index=episode.episode_index,
                condition=condition.label,
                row=row,
                block_path=result.block_path,
            )
    if figures_enabled and CONDITION_ORACLE in results_by_condition:
        reference_result, reference_row = results_by_condition[CONDITION_ORACLE]
        for probe in _PROBE_CONDITIONS:
            if probe not in results_by_condition:
                continue
            probe_result, probe_row = results_by_condition[probe]
            write_condition_pair_figure(
                temporary
                / "figures"
                / reference_result.model
                / f"pair_{CONDITION_ORACLE}_vs_{probe}"
                / f"episode_{episode.episode_index:06d}.png",
                model_display=_MODEL_DISPLAY_NAMES[reference_result.model],
                episode_index=episode.episode_index,
                reference_label=CONDITION_ORACLE,
                probe_label=probe,
                reference_row=reference_row,
                probe_row=probe_row,
                reference_block_path=reference_result.block_path,
                probe_block_path=probe_result.block_path,
            )
    return tuple(rows)


def _figure_path(
    figures_root: Path,
    *,
    model: str,
    condition: str,
    episode_index: int,
    filename: str,
) -> Path:
    return figures_root / model / f"condition_{condition}" / filename


def _build_worker_plans(
    *,
    episodes: Sequence[RouteFieldEpisode],
    models: Sequence[str],
    num_gpus: int,
) -> tuple[RouteFieldEvalWorkerPlan, ...]:
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 2:
        raise ValueError("multi-GPU validation requires num_gpus >= 2")
    if not models:
        raise ValueError("no selected RF7 models")
    plans = []
    gpu_index = 0
    ordered_models = tuple(sorted(models))
    base_workers = num_gpus // len(ordered_models)
    extra_workers = num_gpus % len(ordered_models)
    for model_index, model in enumerate(ordered_models):
        worker_count = base_workers + (1 if model_index < extra_workers else 0)
        for worker_index in range(worker_count):
            assigned = tuple(
                episode.episode_index
                for episode_index, episode in enumerate(episodes)
                if episode_index % worker_count == worker_index
            )
            if not assigned:
                raise RuntimeError(f"GPU worker {gpu_index} received no validation episodes")
            plans.append(
                RouteFieldEvalWorkerPlan(
                    gpu_index=gpu_index,
                    model=model,
                    episode_indices=assigned,
                    leader=worker_index == 0,
                )
            )
            gpu_index += 1
    return tuple(plans)


def _evaluate_worker(
    plan: RouteFieldEvalWorkerPlan,
    config: RouteFieldEvalWorkerConfig,
    start_event: Any,
    event_queue: Any,
) -> None:
    torch.cuda.set_device(plan.gpu_index)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    phase = "loading"
    episode_index = None
    condition = None
    loaded = None
    try:
        selection = config.selected[plan.model]
        loaded = load_policy_with_processors(
            selection["checkpoint_path"],
            local_files_only=True,
            device=f"cuda:{plan.gpu_index}",
        )
        phase = "validating"
        _configure_loaded_policy(
            loaded,
            model=plan.model,
            num_inference_steps=config.num_inference_steps,
        )
        event_queue.put(
            RouteFieldEvalWorkerEvent(
                kind="ready",
                gpu_index=plan.gpu_index,
                model=plan.model,
                phase="validated",
            )
        )
        phase = "waiting"
        start_event.wait()

        response_rows = ()
        if plan.leader:
            phase = "condition_response"
            response_rows = _evaluate_condition_response(
                loaded,
                dataset_root=config.dataset_root,
                episodes=config.episodes,
                cap=config.fixed_observation_episodes,
                num_inference_steps=config.num_inference_steps,
                device=f"cuda:{plan.gpu_index}",
            )

        by_episode = {episode.episode_index: episode for episode in config.episodes}
        closed_loop_rows = []
        for episode_index in plan.episode_indices:
            episode = by_episode[episode_index]
            condition = None
            phase = "closed_loop"
            event_queue.put(
                RouteFieldEvalWorkerEvent(
                    kind="progress",
                    gpu_index=plan.gpu_index,
                    model=plan.model,
                    phase=phase,
                    episode_index=episode_index,
                )
            )
            episode_rows = _evaluate_episode_conditions(
                loaded,
                episode=episode,
                selected=selection,
                temporary=config.temporary,
                episode_length=config.episode_length,
                figures_enabled=config.figures_enabled,
            )
            closed_loop_rows.extend(episode_rows)
            for row in episode_rows:
                condition = str(row["commanded_condition"])
                event_queue.put(
                    RouteFieldEvalWorkerEvent(
                        kind="progress",
                        gpu_index=plan.gpu_index,
                        model=plan.model,
                        phase="rollout_complete",
                        episode_index=episode_index,
                        condition=condition,
                    )
                )

        event_queue.put(
            RouteFieldEvalWorkerEvent(
                kind="result",
                gpu_index=plan.gpu_index,
                model=plan.model,
                phase="complete",
                closed_loop_rows=tuple(closed_loop_rows),
                condition_response_rows=response_rows,
            )
        )
    except BaseException:  # noqa: BLE001 - relay the complete worker failure to the parent.
        event_queue.put(
            RouteFieldEvalWorkerEvent(
                kind="error",
                gpu_index=plan.gpu_index,
                model=plan.model,
                phase=phase,
                episode_index=episode_index,
                condition=condition,
                error=traceback.format_exc(),
            )
        )
    finally:
        if loaded is not None:
            del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _next_worker_event(
    event_queue: Any,
    workers: Sequence[Any],
    *,
    worker_gpu_indices: Sequence[int],
    completed_gpu_indices: set[int],
) -> RouteFieldEvalWorkerEvent:
    try:
        event = event_queue.get(timeout=1.0)
    except Empty:
        unexpected_exits = [
            (gpu_index, worker)
            for gpu_index, worker in zip(worker_gpu_indices, workers, strict=True)
            if worker.exitcode is not None and gpu_index not in completed_gpu_indices
        ]
        if unexpected_exits:
            details = ", ".join(
                f"{worker.name} exitcode={worker.exitcode}" for _, worker in unexpected_exits
            )
            raise RuntimeError(f"RF7 eval worker exited unexpectedly: {details}") from None
        if not any(worker.is_alive() for worker in workers):
            raise RuntimeError("All RF7 eval workers exited before producing results") from None
        raise
    if not isinstance(event, RouteFieldEvalWorkerEvent):
        raise TypeError(f"Received invalid RF7 eval worker event: {event!r}")
    return event


def _raise_worker_error(event: RouteFieldEvalWorkerEvent) -> None:
    raise RuntimeError(
        f"GPU {event.gpu_index} failed for {event.model} during {event.phase}"
        + (f" episode {event.episode_index}" if event.episode_index is not None else "")
        + (f" condition {event.condition}" if event.condition is not None else "")
        + f":\n{event.error}"
    )


def _evaluate_multi_gpu(
    *,
    plans: Sequence[RouteFieldEvalWorkerPlan],
    config: RouteFieldEvalWorkerConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan_by_gpu = {plan.gpu_index: plan for plan in plans}
    if len(plan_by_gpu) != len(plans):
        raise ValueError("RF7 eval worker plans contain duplicate GPU indices")
    context = mp.get_context("spawn")
    event_queue = context.Queue()
    start_event = context.Event()
    workers = [
        context.Process(
            target=_evaluate_worker,
            args=(plan, config, start_event, event_queue),
            name=f"rf7-eval-gpu-{plan.gpu_index}",
        )
        for plan in plans
    ]
    for worker in workers:
        worker.start()
    worker_gpu_indices = [plan.gpu_index for plan in plans]

    ready: set[int] = set()
    results: dict[int, RouteFieldEvalWorkerEvent] = {}
    try:
        while len(ready) < len(workers):
            try:
                event = _next_worker_event(
                    event_queue,
                    workers,
                    worker_gpu_indices=worker_gpu_indices,
                    completed_gpu_indices=set(),
                )
            except Empty:
                continue
            if event.kind == "error":
                _raise_worker_error(event)
            if event.kind != "ready":
                raise RuntimeError(
                    f"GPU {event.gpu_index} sent {event.kind!r} before the ready barrier"
                )
            expected_plan = plan_by_gpu.get(event.gpu_index)
            if expected_plan is None or event.model != expected_plan.model:
                raise RuntimeError(
                    f"Unexpected ready event from GPU {event.gpu_index} model {event.model}"
                )
            if event.gpu_index in ready:
                raise RuntimeError(f"GPU {event.gpu_index} reported ready more than once")
            ready.add(event.gpu_index)
        start_event.set()

        expected_rollouts = _expected_rollout_count(
            episodes=config.episodes,
            plans=plans,
        )
        with tqdm(
            total=expected_rollouts,
            desc="RF7 multi-GPU validation",
            unit="rollout",
            dynamic_ncols=True,
        ) as progress:
            while len(results) < len(workers):
                try:
                    event = _next_worker_event(
                        event_queue,
                        workers,
                        worker_gpu_indices=worker_gpu_indices,
                        completed_gpu_indices=set(results),
                    )
                except Empty:
                    continue
                expected_plan = plan_by_gpu.get(event.gpu_index)
                if expected_plan is None or event.model != expected_plan.model:
                    raise RuntimeError(
                        f"Unexpected event from GPU {event.gpu_index} model {event.model}"
                    )
                if event.kind == "error":
                    _raise_worker_error(event)
                if event.kind == "progress":
                    if event.phase == "rollout_complete":
                        progress.update(1)
                    else:
                        values: dict[str, Any] = {
                            "gpu": event.gpu_index,
                            "model": event.model,
                            "phase": event.phase,
                        }
                        if event.episode_index is not None:
                            values["episode_index"] = event.episode_index
                        if event.condition is not None:
                            values["condition"] = event.condition
                        progress.set_postfix(**values, refresh=True)
                    continue
                if event.kind == "result":
                    if event.gpu_index in results:
                        raise RuntimeError(f"GPU {event.gpu_index} returned more than one result")
                    results[event.gpu_index] = event
                    continue
                raise RuntimeError(
                    f"GPU {event.gpu_index} sent unexpected event kind {event.kind!r}"
                )
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
        event_queue.close()

    failed = [worker for worker in workers if worker.exitcode != 0]
    if failed:
        details = ", ".join(f"{worker.name} exitcode={worker.exitcode}" for worker in failed)
        raise RuntimeError(f"RF7 eval worker shutdown failed: {details}")

    closed_loop_rows = []
    condition_response_rows = []
    for gpu_index in sorted(results):
        event = results[gpu_index]
        closed_loop_rows.extend(event.closed_loop_rows)
        condition_response_rows.extend(event.condition_response_rows)
    return closed_loop_rows, condition_response_rows


def _expected_rollout_count(
    *,
    episodes: Sequence[RouteFieldEpisode],
    plans: Sequence[RouteFieldEvalWorkerPlan] | None = None,
) -> int:
    """Count the rollouts one pass will produce.

    Conditions are episode-dependent, so the total cannot be derived from a
    fixed conditions-per-episode constant.
    """

    per_episode = {
        episode.episode_index: len(episode_condition_labels(episode)) for episode in episodes
    }
    if plans is None:
        return sum(per_episode.values())
    return sum(
        per_episode[episode_index] for plan in plans for episode_index in plan.episode_indices
    )


def _evaluate_single_device(
    *,
    selected: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    temporary: Path,
    episodes: Sequence[RouteFieldEpisode],
    episode_length: int,
    num_inference_steps: int,
    fixed_observation_episodes: int,
    figures_enabled: bool,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    closed_loop_rows = []
    condition_response_rows = []
    expected_rollouts = _expected_rollout_count(episodes=episodes)
    for model in sorted(selected):
        selection = selected[model]
        loaded = load_policy_with_processors(
            selection["checkpoint_path"],
            local_files_only=True,
            device=device,
        )
        try:
            _configure_loaded_policy(
                loaded,
                model=model,
                num_inference_steps=num_inference_steps,
            )
            condition_response_rows.extend(
                _evaluate_condition_response(
                    loaded,
                    dataset_root=dataset_root,
                    episodes=episodes,
                    cap=fixed_observation_episodes,
                    num_inference_steps=num_inference_steps,
                    device=device,
                )
            )
            progress = tqdm(
                total=expected_rollouts,
                desc=f"RF7 validation closed-loop {model}",
                unit="rollout",
            )
            try:
                for episode in episodes:
                    episode_rows = _evaluate_episode_conditions(
                        loaded,
                        episode=episode,
                        selected=selection,
                        temporary=temporary,
                        episode_length=episode_length,
                        figures_enabled=figures_enabled,
                    )
                    closed_loop_rows.extend(episode_rows)
                    progress.update(len(episode_rows))
            finally:
                progress.close()
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return closed_loop_rows, condition_response_rows


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _column(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _condition_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate one row per model and commanded condition.

    ``success_rate`` is only a performance number for the task-reference
    condition. For obedience probes the commanded route deliberately conflicts
    with completing the task, so low probe success is expected and carries no
    information on its own; read ``mean_commanded_minus_oracle_adherence`` and
    ``mean_commanded_first_segment_progress_cells`` instead.
    """

    result = []
    for model in sorted(models):
        for condition in CONDITION_ORDER:
            selected_rows = [
                row
                for row in rows
                if row["model"] == model and row["commanded_condition"] == condition
            ]
            if not selected_rows:
                # A condition is absent when no episode's route is long enough
                # to support it. That is a property of the split, not a failure.
                continue
            successes = sum(bool(row["success"]) for row in selected_rows)
            total = len(selected_rows)
            lower, upper = wilson_interval(successes, total)
            latencies = [
                float(value)
                for row in selected_rows
                for value in row.get("replan_latencies_ms", [])
            ]
            result.append(
                {
                    "model": model,
                    "commanded_condition": condition,
                    "condition_role": CONDITION_ROLES[condition],
                    "success_count": successes,
                    "episode_count": total,
                    "failure_count": total - successes,
                    "success_rate": successes / total,
                    "success_wilson95_low": lower,
                    "success_wilson95_high": upper,
                    "mean_max_reward": float(
                        np.mean([float(row["max_reward"]) for row in selected_rows])
                    ),
                    "mean_ordered_waypoint_adherence": _mean_or_none(
                        _column(selected_rows, "ordered_waypoint_adherence")
                    ),
                    "mean_normalized_edit_distance": _mean_or_none(
                        _column(selected_rows, "normalized_edit_distance")
                    ),
                    "mean_oracle_ordered_waypoint_adherence": _mean_or_none(
                        _column(selected_rows, "oracle_ordered_waypoint_adherence")
                    ),
                    "mean_commanded_minus_oracle_adherence": _mean_or_none(
                        _column(selected_rows, "commanded_minus_oracle_adherence")
                    ),
                    "mean_commanded_first_segment_progress_cells": _mean_or_none(
                        _column(selected_rows, "commanded_first_segment_progress_cells")
                    ),
                    "mean_oracle_first_segment_progress_cells": _mean_or_none(
                        _column(selected_rows, "oracle_first_segment_progress_cells")
                    ),
                    "empty_achieved_route_count": sum(
                        1 for row in selected_rows if row.get("achieved_route_is_empty")
                    ),
                    "mean_final_rotation_error_deg": _mean_or_none(
                        _column(selected_rows, "final_rotation_error_deg")
                    ),
                    "mean_replan_latency_ms": _mean_or_none(latencies),
                    "p95_replan_latency_ms": (
                        float(np.percentile(latencies, 95)) if latencies else None
                    ),
                }
            )
    return result


def _paired_condition_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
) -> list[dict[str, Any]]:
    """Pair every obedience probe against the task reference on the same episode.

    Each pair shares the initial state, the environment seed, and the policy
    seed, so the commanded route is the only difference. ``obeyed_count`` is the
    strict readout: the executed path matched the commanded route better than it
    matched the route the policy would traverse anyway.
    """

    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(str(row["model"]), int(row["episode_index"]))][str(row["commanded_condition"])] = (
            row
        )
    result = []
    for model in sorted(models):
        for probe in _PROBE_CONDITIONS:
            pairs = [
                values
                for (model_name, _episode), values in sorted(by_key.items())
                if model_name == model and CONDITION_ORACLE in values and probe in values
            ]
            if not pairs:
                continue
            probe_rows = [values[probe] for values in pairs]
            reference_rows = [values[CONDITION_ORACLE] for values in pairs]
            advantage = _column(probe_rows, "commanded_minus_oracle_adherence")
            obeyed = sum(1 for value in advantage if value > 0.0)
            progress = _column(probe_rows, "commanded_first_segment_progress_cells")
            forward_progress = sum(1 for value in progress if value > 0.0)
            obeyed_low, obeyed_high = (
                wilson_interval(obeyed, len(advantage)) if advantage else (None, None)
            )
            undecidable_pairs = sum(1 for value in advantage if value == 0.0)
            decidable_pairs = len(advantage) - undecidable_pairs
            # A positive difference is by construction a decidable obey, so the
            # decidable obeyed count equals the original obeyed count; only the
            # denominator changes.
            obeyed_decidable = obeyed
            obeyed_decidable_low, obeyed_decidable_high = (
                wilson_interval(obeyed_decidable, decidable_pairs)
                if decidable_pairs
                else (None, None)
            )
            two_cell_probes = sum(
                1 for row in probe_rows if int(row.get("requested_route_length") or 0) == 2
            )
            result.append(
                {
                    "model": model,
                    "reference_condition": CONDITION_ORACLE,
                    "probe_condition": probe,
                    "pair_count": len(pairs),
                    "reference_success_count": sum(bool(row["success"]) for row in reference_rows),
                    "probe_success_count": sum(bool(row["success"]) for row in probe_rows),
                    "scored_pair_count": len(advantage),
                    "obeyed_count": obeyed,
                    "obeyed_rate": obeyed / len(advantage) if advantage else None,
                    "obeyed_wilson95_low": obeyed_low,
                    "obeyed_wilson95_high": obeyed_high,
                    "undecidable_pair_count": undecidable_pairs,
                    "decidable_pair_count": decidable_pairs,
                    "obeyed_count_decidable": obeyed_decidable,
                    "obeyed_rate_decidable": (
                        obeyed_decidable / decidable_pairs if decidable_pairs else None
                    ),
                    "obeyed_wilson95_low_decidable": obeyed_decidable_low,
                    "obeyed_wilson95_high_decidable": obeyed_decidable_high,
                    "two_cell_probe_count": two_cell_probes,
                    "mean_commanded_minus_oracle_adherence": _mean_or_none(advantage),
                    "scored_progress_count": len(progress),
                    "commanded_first_segment_positive_count": forward_progress,
                    "commanded_first_segment_positive_rate": (
                        forward_progress / len(progress) if progress else None
                    ),
                    "mean_commanded_first_segment_progress_cells": _mean_or_none(progress),
                    "mean_reference_first_segment_progress_cells": _mean_or_none(
                        _column(reference_rows, "commanded_first_segment_progress_cells")
                    ),
                }
            )
    return result


def _validate_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    episodes: Sequence[RouteFieldEpisode],
    models: Sequence[str],
    recordings_root: Path,
    figures_root: Path | None,
    figures_enabled: bool,
) -> list[dict[str, Any]]:
    conditions_by_episode = {
        episode.episode_index: episode_condition_labels(episode) for episode in episodes
    }
    expected = {
        (model, int(episode_index), condition)
        for model in models
        for episode_index, conditions in conditions_by_episode.items()
        for condition in conditions
    }
    actual = [
        (
            str(row["model"]),
            int(row["episode_index"]),
            str(row["commanded_condition"]),
        )
        for row in rows
    ]
    duplicates = sorted({key for key in actual if actual.count(key) > 1})
    if duplicates:
        raise ValueError(
            f"closed-loop contains duplicate model/episode/condition rows: {duplicates}"
        )
    actual_set = set(actual)
    missing = sorted(expected - actual_set)
    unexpected = sorted(actual_set - expected)
    if missing or unexpected:
        raise ValueError(
            "closed-loop model/episode/condition coverage mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    model_order = {MODEL_BC: 0, MODEL_DIFFUSION: 1}
    for row in rows:
        model = str(row["model"])
        episode_index = int(row["episode_index"])
        condition = str(row["commanded_condition"])
        expected_name = _recording_name(
            model=model,
            condition=condition,
            episode_index=episode_index,
            success=bool(row["success"]),
        )
        expected_metadata = "\n".join(
            f"{key}: {value}"
            for key, value in _recording_metadata(
                model=model,
                condition=condition,
                episode_index=episode_index,
                success=bool(row["success"]),
            ).items()
        )
        if row.get("recording_name") != expected_name:
            raise ValueError("Closed-loop recording name does not expose model/condition identity")
        if row.get("recording_metadata") != expected_metadata:
            raise ValueError("Closed-loop recording metadata identity is inconsistent")
        if row.get("recording_has_identity_panel") is not True:
            raise ValueError("Closed-loop recording must expose the identity metadata panel")
        recording = _recording_path(
            recordings_root,
            model=model,
            condition=condition,
            episode_index=episode_index,
        )
        if not recording.is_file() or recording.stat().st_size == 0:
            raise FileNotFoundError(f"RF7 recording is missing or empty: {recording}")
    if figures_enabled:
        if figures_root is None:
            raise ValueError("figures_root is required when figures are enabled")
        for row in rows:
            figure = _figure_path(
                figures_root,
                model=str(row["model"]),
                condition=str(row["commanded_condition"]),
                episode_index=int(row["episode_index"]),
                filename=f"episode_{int(row['episode_index']):06d}.png",
            )
            if not figure.is_file() or figure.stat().st_size == 0:
                raise FileNotFoundError(f"RF7 rollout figure is missing or empty: {figure}")
        present_conditions = {condition for _model, _episode, condition in actual_set}
        for model in models:
            for condition in CONDITION_ORDER:
                if condition not in present_conditions:
                    continue
                contact_sheet = (
                    figures_root / model / f"condition_{condition}" / ("contact_sheet.png")
                )
                if not contact_sheet.is_file() or contact_sheet.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"RF7 contact sheet is missing or empty: {contact_sheet}"
                    )
    return sorted(
        rows,
        key=lambda row: (
            model_order.get(str(row["model"]), 99),
            CONDITION_ORDER.index(str(row["commanded_condition"])),
            int(row["episode_index"]),
        ),
    )


def _evaluate_gate_only(
    *,
    selected: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    episodes: Sequence[RouteFieldEpisode],
    num_inference_steps: int,
    fixed_observation_episodes: int,
    device: str,
) -> list[dict[str, Any]]:
    """Run only the fixed-observation forward-vs-reversed gate.

    This exists so the gate can be acted on before committing to closed-loop
    rollouts: if replacing the route by its reversal barely moves the action
    chunk, the ordering and direction channels are not read and no rollout
    budget can demonstrate route control.
    """

    if fixed_observation_episodes < 1:
        raise ValueError("gate-only validation requires fixed_observation_episodes >= 1")
    rows: list[dict[str, Any]] = []
    for model in sorted(selected):
        selection = selected[model]
        loaded = load_policy_with_processors(
            selection["checkpoint_path"],
            local_files_only=True,
            device=device,
        )
        try:
            _configure_loaded_policy(
                loaded,
                model=model,
                num_inference_steps=num_inference_steps,
            )
            rows.extend(
                _evaluate_condition_response(
                    loaded,
                    dataset_root=dataset_root,
                    episodes=episodes,
                    cap=fixed_observation_episodes,
                    num_inference_steps=num_inference_steps,
                    device=device,
                )
            )
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError(
            "gate-only validation produced no rows; no episode in this split has a "
            "route long enough to reverse"
        )
    return rows


def _write_gate_only_outputs(
    temporary: Path,
    *,
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
    split_name: str,
    episodes: Sequence[RouteFieldEpisode],
    route_manifest: Path,
    route_manifest_sha256: str,
    selection_dir: Path,
    selection_manifest: Mapping[str, Any],
    num_inference_steps: int,
    fixed_observation_episodes: int,
    device: str,
    runtime: Mapping[str, Any],
) -> None:
    write_records_parquet(temporary / "condition_response.parquet", rows)
    metrics: dict[str, Any] = {
        "validation_split": split_name,
        "episode_count": len(episodes),
        "condition_response_episode_cap": fixed_observation_episodes,
        "condition_response_row_count": len(rows),
    }
    for model in sorted(models):
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        metrics[f"{model}_condition_response_row_count"] = len(model_rows)
        metrics[f"{model}_condition_response_chunk_l2_mean"] = float(
            np.mean([float(row["chunk_l2_distance"]) for row in model_rows])
        )
        metrics[f"{model}_condition_response_chunk_l2_min"] = float(
            np.min([float(row["chunk_l2_distance"]) for row in model_rows])
        )
        metrics[f"{model}_condition_response_endpoint_mean"] = float(
            np.mean([float(row["endpoint_distance"]) for row in model_rows])
        )
    write_json_atomic(
        temporary / "run_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            # Offline, not Closed-loop: nothing is executed in the simulator, so
            # this run can never report task success.
            "label": "Offline",
            "experiment": "route_field_condition_gate",
            "dataset_revision": DATASET_REVISION,
            "validation_split": split_name,
            "episode_indices": [episode.episode_index for episode in episodes],
            "route_manifest": str(route_manifest),
            "route_manifest_sha256": route_manifest_sha256,
            "selection_dir": str(selection_dir),
            "selection_experiment": selection_manifest.get("experiment"),
            "comparison": f"{CONDITION_REVERSED}_minus_{CONDITION_ORACLE}",
            "observation_source": "dataset_middle_frame",
            "policy_seed": POLICY_SEED,
            "diffusion_scheduler": "DDPM",
            "num_inference_steps": num_inference_steps,
            "fixed_observation_episode_cap": fixed_observation_episodes,
            "gate_criterion": (
                "The observation, the initial noise, and the reverse noise stream are "
                "identical across the two conditions, and the reversed route has the "
                "same occupancy support. A near-zero chunk distance therefore means the "
                "ordering and direction channels are not read, and closed-loop rollouts "
                "cannot demonstrate route control."
            ),
            "execution": {"num_gpus": 1, "device": device, "parallelism": "sequential"},
            "runtime": runtime,
        },
    )
    write_metrics(
        temporary,
        metrics,
        comments={
            "condition_response_row_count": (
                "Fixed observation and fixed noise, ground-truth route versus its reversal"
            ),
        },
    )


def run_route_field_eval(
    *,
    selection_dir: Path,
    route_manifest: Path,
    dataset_root: Path,
    output_dir: Path,
    device: str | None,
    split_name: str,
    max_episodes: int | None,
    episode_length: int,
    num_inference_steps: int,
    fixed_observation_episodes: int,
    figures_enabled: bool,
    overwrite: bool,
    num_gpus: int = 1,
    gate_only: bool = False,
) -> Path:
    if gate_only and num_gpus != 1:
        raise ValueError("gate-only validation is single-device; use --num-gpus=1")
    actual_device = _validate_execution_request(
        num_gpus=num_gpus,
        device=device,
    )
    selected = _selected_rows(selection_dir)
    models = tuple(sorted(selected))
    selection_manifest = load_json_object(selection_dir / "run_manifest.json")
    expected_route_hash = selection_manifest.get("route_manifest_sha256")
    actual_route_hash = sha256_file(route_manifest)
    if expected_route_hash != actual_route_hash:
        raise ValueError(
            "RF7 route manifest does not match checkpoint selection: "
            f"{expected_route_hash!r} != {actual_route_hash!r}"
        )
    split = load_split()
    if split_name not in ("audit", "development"):
        raise ValueError("--split must be audit or development")
    episode_indices = tuple(split.audit if split_name == "audit" else split.development)
    if max_episodes is not None:
        if isinstance(max_episodes, bool) or not isinstance(max_episodes, int) or max_episodes < 1:
            raise ValueError("--max-episodes must be a positive integer when provided")
        episode_indices = episode_indices[:max_episodes]
    episodes = load_route_field_episodes(
        dataset_root=dataset_root,
        route_manifest=route_manifest,
        episode_indices=episode_indices,
        split_name=split_name,
    )
    reconstruction = preflight_reconstruction(episodes)
    runtime = collect_runtime_provenance(practice_root())
    plans = (
        ()
        if num_gpus == 1
        else _build_worker_plans(
            episodes=episodes,
            models=models,
            num_gpus=num_gpus,
        )
    )

    if gate_only:
        if actual_device is None:
            raise RuntimeError("Gate-only RF7 validation did not resolve a device")
        gate_rows = _evaluate_gate_only(
            selected=selected,
            dataset_root=dataset_root,
            episodes=episodes,
            num_inference_steps=num_inference_steps,
            fixed_observation_episodes=fixed_observation_episodes,
            device=actual_device,
        )
        with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
            _write_gate_only_outputs(
                temporary,
                rows=gate_rows,
                models=models,
                split_name=split_name,
                episodes=episodes,
                route_manifest=route_manifest,
                route_manifest_sha256=actual_route_hash,
                selection_dir=selection_dir,
                selection_manifest=selection_manifest,
                num_inference_steps=num_inference_steps,
                fixed_observation_episodes=fixed_observation_episodes,
                device=actual_device,
                runtime=runtime,
            )
            write_records_parquet(
                temporary / "reconstruction_metrics.parquet",
                (metric.as_record() for metric in reconstruction),
            )
        return output_dir.resolve()

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        if num_gpus == 1:
            if actual_device is None:
                raise RuntimeError("Single-device RF7 validation did not resolve a device")
            closed_loop_rows, condition_response_rows = _evaluate_single_device(
                selected=selected,
                dataset_root=dataset_root,
                temporary=temporary,
                episodes=episodes,
                episode_length=episode_length,
                num_inference_steps=num_inference_steps,
                fixed_observation_episodes=fixed_observation_episodes,
                figures_enabled=figures_enabled,
                device=actual_device,
            )
        else:
            worker_config = RouteFieldEvalWorkerConfig(
                selected=selected,
                route_manifest=route_manifest,
                dataset_root=dataset_root,
                temporary=temporary,
                episodes=tuple(episodes),
                episode_length=episode_length,
                num_inference_steps=num_inference_steps,
                fixed_observation_episodes=fixed_observation_episodes,
                figures_enabled=figures_enabled,
            )
            closed_loop_rows, condition_response_rows = _evaluate_multi_gpu(
                plans=plans,
                config=worker_config,
            )
        figures_root = temporary / "figures"
        if figures_enabled:
            for model in models:
                for condition in CONDITION_ORDER:
                    condition_rows = [
                        row
                        for row in closed_loop_rows
                        if row["model"] == model and row["commanded_condition"] == condition
                    ]
                    if not condition_rows:
                        continue
                    write_contact_sheet(
                        figures_root / model / f"condition_{condition}" / "contact_sheet.png",
                        model_display=_MODEL_DISPLAY_NAMES[model],
                        condition=condition,
                        rows=condition_rows,
                    )
        closed_loop_rows = _validate_coverage(
            closed_loop_rows,
            episodes=episodes,
            models=models,
            recordings_root=temporary / "recordings",
            figures_root=figures_root if figures_enabled else None,
            figures_enabled=figures_enabled,
        )
        condition_metric_rows = _condition_metric_rows(
            closed_loop_rows,
            models=models,
        )
        paired_rows = _paired_condition_rows(closed_loop_rows, models=models)

        metrics: dict[str, Any] = {
            "validation_split": split_name,
            "episode_count": len(episodes),
            "rollout_count": len(closed_loop_rows),
            "condition_response_episode_cap": fixed_observation_episodes,
            "condition_response_row_count": len(condition_response_rows),
        }
        for row in condition_metric_rows:
            prefix = f"{row['model']}_{row['commanded_condition']}"
            for key, value in row.items():
                if key in ("model", "commanded_condition"):
                    continue
                metrics[f"{prefix}_{key}"] = value
        for row in paired_rows:
            prefix = f"{row['model']}_{row['reference_condition']}_vs_{row['probe_condition']}"
            for key, value in row.items():
                if key in ("model", "reference_condition", "probe_condition"):
                    continue
                metrics[f"{prefix}_{key}"] = value
        for model in models:
            response = [row for row in condition_response_rows if row["model"] == model]
            if response:
                metrics[f"{model}_condition_response_chunk_l2_mean"] = float(
                    np.mean([float(row["chunk_l2_distance"]) for row in response])
                )
                metrics[f"{model}_condition_response_endpoint_mean"] = float(
                    np.mean([float(row["endpoint_distance"]) for row in response])
                )
        write_records_parquet(temporary / "rollouts.parquet", closed_loop_rows)
        if condition_response_rows:
            write_records_parquet(
                temporary / "condition_response.parquet",
                condition_response_rows,
            )
        write_records_parquet(temporary / "condition_metrics.parquet", condition_metric_rows)
        if paired_rows:
            write_records_parquet(
                temporary / "paired_condition_metrics.parquet",
                paired_rows,
            )
        write_records_parquet(
            temporary / "reconstruction_metrics.parquet",
            (metric.as_record() for metric in reconstruction),
        )
        eval_info = [
            {
                key: value
                for key, value in row.items()
                if key not in ("replan_latencies_ms", "recording_metadata")
            }
            for row in closed_loop_rows
        ]
        write_json_atomic(temporary / "eval_info.json", eval_info)
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "label": "Closed-loop",
                "experiment": "route_field_held_out_validation",
                "dataset_revision": DATASET_REVISION,
                "validation_split": split_name,
                "episode_indices": episode_indices,
                "route_manifest": str(route_manifest),
                "route_manifest_sha256": actual_route_hash,
                "selection_dir": str(selection_dir),
                "selection_experiment": selection_manifest.get("experiment"),
                "condition_source": "per_episode_ground_truth_route_plus_occupancy_matched_probes",
                "condition_protocol": [
                    {
                        "condition": condition,
                        "role": CONDITION_ROLES[condition],
                        "definition": definition,
                    }
                    for condition, definition in (
                        (
                            CONDITION_ORACLE,
                            "the episode's own ground-truth macro route",
                        ),
                        (
                            CONDITION_REVERSED,
                            (
                                "the same cells traversed in reverse order (identical "
                                "occupancy, mirrored progress, recomputed direction); "
                                "requires at least 2 route cells"
                            ),
                        ),
                        (
                            CONDITION_PERMUTED,
                            (
                                "interior cells reversed with both endpoints kept "
                                "(identical occupancy and endpoints); requires at least "
                                "4 route cells"
                            ),
                        ),
                    )
                ],
                "condition_counts": {
                    condition: sum(
                        1 for episode in episodes if condition in episode_condition_labels(episode)
                    )
                    for condition in CONDITION_ORDER
                },
                "obedience_criterion": (
                    "For an obedience probe, task success is expected to fall because the "
                    "commanded route conflicts with reaching the goal. The decisive readouts "
                    "are commanded_minus_oracle_adherence > 0 and a positive "
                    "commanded_first_segment_progress_cells; probe success rate is not a "
                    "performance metric."
                ),
                "initial_state_source": "dataset_first_frame_reconstruction",
                "policy_seed": POLICY_SEED,
                "diffusion_scheduler": "DDPM",
                "diffusion_sampling_protocol": "full-ddpm-sampling",
                "num_inference_steps": num_inference_steps,
                "episode_length": episode_length,
                "fixed_observation_episode_cap": fixed_observation_episodes,
                "fixed_observation_comparison": (f"{CONDITION_REVERSED}_minus_{CONDITION_ORACLE}"),
                "figures_enabled": figures_enabled,
                "execution": {
                    "num_gpus": num_gpus,
                    "parallelism": ("worker_process_pool" if num_gpus > 1 else "sequential"),
                    "task_granularity": "model_episode_condition",
                },
                "runtime": runtime,
            },
        )
        comments = {
            "rollout_count": (
                "One ground-truth route rollout per episode/model, plus reversed and "
                "middle-permuted probes where the route is long enough"
            ),
            "condition_response_row_count": (
                "Fixed observation and fixed noise, ground-truth route versus its reversal"
            ),
        }
        for row in condition_metric_rows:
            prefix = f"{row['model']}_{row['commanded_condition']}"
            is_probe = row["condition_role"] == OBEDIENCE_PROBE_ROLE
            comments[f"{prefix}_success_rate"] = (
                "Obedience readout with inverted sign, NOT a performance metric"
                if is_probe
                else "Closed-loop task success under the episode's ground-truth route"
            )
            comments[f"{prefix}_mean_ordered_waypoint_adherence"] = (
                "LCS(commanded, achieved macro route) / len(commanded)"
            )
            comments[f"{prefix}_mean_commanded_minus_oracle_adherence"] = (
                "Commanded-route adherence minus ground-truth-route adherence; "
                "positive means the executed path followed what was commanded"
            )
            comments[f"{prefix}_mean_commanded_first_segment_progress_cells"] = (
                "Net T-center displacement projected onto the commanded first segment, "
                "in grid cells; positive means it moved the commanded way"
            )
        for row in paired_rows:
            prefix = f"{row['model']}_{row['reference_condition']}_vs_{row['probe_condition']}"
            comments[f"{prefix}_obeyed_rate"] = (
                "Share of same-state pairs where commanded-route adherence beat "
                "ground-truth-route adherence"
            )
            comments[f"{prefix}_commanded_first_segment_positive_rate"] = (
                "Share of probes whose net T-center displacement went the commanded way"
            )
        write_metrics(temporary, metrics, comments=comments)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=default_output_root() / "pilot" / "route_field_oracle_checkpoint_selection",
    )
    parser.add_argument(
        "--route-manifest",
        type=Path,
        default=default_output_root()
        / "pilot"
        / "route_field_7x7"
        / "sidecar"
        / "route_manifest.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, default="audit")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=EPISODE_LENGTH)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=NUM_INFERENCE_STEPS,
    )
    parser.add_argument("--fixed-observation-episodes", type=int, default=5)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "route_field_held_out_eval",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_route_field_eval(
        selection_dir=args.selection_dir.expanduser().resolve(),
        route_manifest=args.route_manifest.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        split_name=args.split,
        max_episodes=args.max_episodes,
        episode_length=args.episode_length,
        num_inference_steps=args.num_inference_steps,
        fixed_observation_episodes=args.fixed_observation_episodes,
        figures_enabled=not args.no_figures,
        overwrite=args.overwrite,
        num_gpus=args.num_gpus,
        gate_only=args.gate_only,
    )
    print(f"RF7 validation output: {result}")


if __name__ == "__main__":
    main()
