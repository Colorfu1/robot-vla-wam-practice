"""Evaluate every E5 condition for selected checkpoints on held-out audit episodes."""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
import re
import time
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
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION
from torch.utils.data import DataLoader
from tqdm import tqdm

from robot_practice.policies.mode_bc_chunk.modeling_mode_bc_chunk import ModeBCChunkPolicy
from robot_practice.policies.mode_conditioning import PROTOTYPE_ID
from robot_practice.policies.mode_diffusion.modeling_mode_diffusion import ModeDiffusionPolicy

from .checkpoint_selection import _stack_images
from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    default_dataset_root,
    default_output_root,
    load_json_object,
    practice_root,
)
from .dataset_split import load_split
from .e5_checkpoint_selection import (
    EPISODE_LENGTH,
    MODE_BC,
    MODE_DIFFUSION,
    NUM_INFERENCE_STEPS,
    POLICY_SEED,
)
from .e5_dataset import load_prototype_sidecar
from .e5_oracle_mode import (
    OracleModeEpisode,
    load_oracle_mode_episodes,
    preflight_reconstruction,
    run_oracle_mode_rollout,
    validate_mode_policy,
)
from .e5_rollout_eval import _load_classifier, classify_rollout
from .noise import fixed_observation_noise_seed, fixed_observation_reverse_seed, make_initial_noise
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import controlled_sample, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rerun_visualization import record_single_rollout

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MODEL_DISPLAY_NAMES = {
    MODE_BC: "BC",
    MODE_DIFFUSION: "DP",
}


@dataclass(frozen=True)
class AuditWorkerPlan:
    gpu_index: int
    model: str
    model_worker_index: int
    model_worker_count: int
    episode_indices: tuple[int, ...]
    open_loop_leader: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "gpu_index": self.gpu_index,
            "model": self.model,
            "model_worker_index": self.model_worker_index,
            "model_worker_count": self.model_worker_count,
            "episode_indices": list(self.episode_indices),
            "open_loop_leader": self.open_loop_leader,
        }


@dataclass(frozen=True)
class AuditWorkerConfig:
    selected: dict[str, dict[str, Any]]
    prototype_manifest: Path
    dataset_root: Path
    temporary: Path
    episodes: tuple[OracleModeEpisode, ...]
    selected_k: int
    null_prototype_id: int
    open_loop_batch_size: int


@dataclass(frozen=True)
class AuditWorkerEvent:
    kind: str
    gpu_index: int
    model: str
    phase: str | None = None
    episode_index: int | None = None
    commanded_condition_id: int | None = None
    closed_loop_rows: tuple[dict[str, Any], ...] = ()
    open_loop_row: dict[str, Any] | None = None
    open_loop_episode_rows: tuple[dict[str, Any], ...] = ()
    error: str | None = None


def _condition_ids(*, selected_k: int, null_prototype_id: int) -> tuple[int, ...]:
    if selected_k < 1:
        raise ValueError("selected_k must be positive")
    if null_prototype_id in range(selected_k):
        raise ValueError(f"NULL prototype ID must not overlap P0..P{selected_k - 1}")
    return (*range(selected_k), null_prototype_id)


def _condition_name(condition_id: int, *, null_prototype_id: int) -> str:
    return "NULL" if condition_id == null_prototype_id else f"P{condition_id}"


def _recording_path(
    root: Path,
    *,
    model: str,
    condition_name: str,
    episode_index: int,
) -> Path:
    return root / model / f"condition_{condition_name}" / f"episode_{episode_index:06d}.rrd"


def _recording_name(
    *,
    model: str,
    condition_name: str,
    episode_index: int,
    gt_mode_name: str,
    success: bool,
) -> str:
    outcome = "success" if success else "failure"
    return (
        f"E5 audit | {_MODEL_DISPLAY_NAMES[model]} | command {condition_name} | "
        f"GT {gt_mode_name} | episode {episode_index} | {outcome}"
    )


def _recording_metadata(
    *,
    model: str,
    condition_name: str,
    episode_index: int,
    gt_mode_name: str,
    success: bool,
) -> dict[str, object]:
    return {
        "model": f"{_MODEL_DISPLAY_NAMES[model]} ({model})",
        "commanded_condition": condition_name,
        "gt_mode": gt_mode_name,
        "episode": episode_index,
        "success": success,
    }


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
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


def _expanded_valid_mask(
    valid_steps: torch.Tensor,
    squared_error: torch.Tensor,
) -> torch.Tensor:
    if valid_steps.shape != squared_error.shape[:-1]:
        raise ValueError(
            "valid_steps must match the squared-error batch/action dimensions: "
            f"{tuple(valid_steps.shape)} != {tuple(squared_error.shape[:-1])}"
        )
    return (
        valid_steps.to(device=squared_error.device, dtype=torch.bool)
        .unsqueeze(-1)
        .expand_as(squared_error)
    )


def _build_worker_plans(
    *,
    episode_indices: Sequence[int],
    num_gpus: int,
) -> tuple[AuditWorkerPlan, ...]:
    episodes = tuple(int(value) for value in episode_indices)
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("episode_indices must be a non-empty unique sequence")
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 2:
        raise ValueError("multi-GPU audit requires num_gpus >= 2")
    if num_gpus > 2 * len(episodes):
        raise ValueError(f"num_gpus={num_gpus} exceeds {2 * len(episodes)} model/episode tasks")

    bc_workers = num_gpus // 2
    worker_counts = {
        MODE_BC: bc_workers,
        MODE_DIFFUSION: num_gpus - bc_workers,
    }
    plans = []
    gpu_index = 0
    for model in (MODE_BC, MODE_DIFFUSION):
        worker_count = worker_counts[model]
        for model_worker_index in range(worker_count):
            assigned = episodes[model_worker_index::worker_count]
            if not assigned:
                raise RuntimeError(f"GPU worker {gpu_index} received no audit episodes")
            plans.append(
                AuditWorkerPlan(
                    gpu_index=gpu_index,
                    model=model,
                    model_worker_index=model_worker_index,
                    model_worker_count=worker_count,
                    episode_indices=assigned,
                    open_loop_leader=model_worker_index == 0,
                )
            )
            gpu_index += 1
    return tuple(plans)


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
        / "e5_mode_conditioned"
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


def _selected_rows(
    selection_dir: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    manifest = load_json_object(selection_dir / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Checkpoint selection is not complete: {selection_dir}")
    if manifest.get("experiment") != "e5_oracle_mode_development_checkpoint_selection":
        raise ValueError(f"Not an E5 oracle checkpoint selection: {selection_dir}")
    payload = load_json_object(selection_dir / "selected_checkpoints.json")
    expected = {MODE_BC, MODE_DIFFUSION}
    if set(payload) != expected:
        raise ValueError(f"Selected checkpoints must contain exactly {sorted(expected)}")
    resolved_repository_root = (
        practice_root() if repository_root is None else repository_root.expanduser().resolve()
    )
    result = {}
    for model in sorted(expected):
        result[model] = _resolve_selected_row(
            model,
            payload[model],
            repository_root=resolved_repository_root,
        )
    return result


def _checkpoint_path_resolution_summary(
    selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        model: {
            "recorded_checkpoint_path": selected[model]["recorded_checkpoint_path"],
            "checkpoint_path": selected[model]["checkpoint_path"],
            "checkpoint_path_resolution": selected[model]["checkpoint_path_resolution"],
            "checkpoint_id": selected[model]["checkpoint_id"],
        }
        for model in (MODE_BC, MODE_DIFFUSION)
    }


def _raw_policy_batch(
    batch: Mapping[str, Any],
    policy: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    result = {}
    for key in policy.config.input_features:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Dataset batch {key!r} must be a tensor")
        result[key] = value
    action = batch.get(ACTION)
    if not isinstance(action, torch.Tensor):
        raise TypeError("Dataset batch action must be a tensor")
    result[ACTION] = action
    return result


def evaluate_full_open_loop(
    loaded: Any,
    *,
    dataset_root: Path,
    episodes: Sequence[OracleModeEpisode],
    selected_k: int,
    batch_size: int,
    show_progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure every valid audit action target with its episode oracle mode."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_root,
        revision=DATASET_REVISION,
    )
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        episodes=[episode.episode_index for episode in episodes],
        delta_timestamps=resolve_delta_timestamps(loaded.config, metadata),
        revision=DATASET_REVISION,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    oracle_ids = {episode.episode_index: episode.prototype_id for episode in episodes}
    invalid_ids = {
        episode_index: prototype_id
        for episode_index, prototype_id in oracle_ids.items()
        if not 0 <= prototype_id < selected_k
    }
    if invalid_ids:
        raise ValueError(f"Audit oracle prototype IDs are outside P0..P{selected_k - 1}")
    device = get_device_from_parameters(loaded.policy)
    start = loaded.config.n_obs_steps - 1
    end = start + loaded.config.n_action_steps
    normalized_sum = 0.0
    denormalized_sum = 0.0
    valid_elements = 0
    inference_latencies = []
    per_episode: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {"normalized_sum": 0.0, "denormalized_sum": 0.0, "valid_elements": 0}
    )

    for batch_ordinal, batch in enumerate(
        tqdm(
            loader,
            desc="Audit open-loop",
            unit="batch",
            disable=not show_progress,
        )
    ):
        raw = _raw_policy_batch(batch, loaded.policy)
        processed = loaded.preprocessor(raw)
        episode_tensor = batch["episode_index"]
        episode_values = [
            int(value) for value in episode_tensor.detach().cpu().numpy().reshape(-1).tolist()
        ]
        processed[PROTOTYPE_ID] = torch.as_tensor(
            [oracle_ids[episode] for episode in episode_values],
            dtype=torch.long,
            device=device,
        )
        prepared = _stack_images(loaded.policy, processed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            if isinstance(loaded.policy, ModeBCChunkPolicy):
                normalized_prediction = loaded.policy.predict_full_action_horizon(prepared)[
                    :, start:end
                ]
            elif isinstance(loaded.policy, ModeDiffusionPolicy):
                explicit = make_initial_noise(
                    seed=fixed_observation_noise_seed(
                        candidate_ordinal=batch_ordinal,
                        sample_ordinal=POLICY_SEED,
                    ),
                    batch_size=len(episode_values),
                    device=device,
                    dtype=get_dtype_from_parameters(loaded.policy),
                    horizon=loaded.config.horizon,
                    action_dim=loaded.config.action_feature.shape[0],
                )
                normalized_prediction = controlled_sample(
                    loaded.policy,
                    prepared,
                    initial_noise=explicit.tensor,
                    reverse_seed=fixed_observation_reverse_seed(candidate_ordinal=batch_ordinal),
                ).actions
            else:
                raise TypeError(f"Unsupported E5 policy: {type(loaded.policy).__name__}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_latencies.append((time.perf_counter() - started) * 1000.0)

        normalized_target = processed[ACTION][:, start:end]
        prediction = loaded.postprocessor(normalized_prediction)
        target = loaded.postprocessor(normalized_target)
        padding = batch.get("action_is_pad")
        if isinstance(padding, torch.Tensor):
            valid_steps = ~padding[:, start:end].to(dtype=torch.bool)
        else:
            valid_steps = torch.ones(
                normalized_target.shape[:2],
                dtype=torch.bool,
            )
        normalized_squared = (normalized_prediction - normalized_target) ** 2
        denormalized_squared = (prediction - target) ** 2
        normalized_mask = _expanded_valid_mask(valid_steps, normalized_squared)
        denormalized_mask = _expanded_valid_mask(valid_steps, denormalized_squared)
        normalized_values = normalized_squared.masked_select(normalized_mask)
        denormalized_values = denormalized_squared.masked_select(denormalized_mask)
        normalized_sum += float(normalized_values.sum().cpu())
        denormalized_sum += float(denormalized_values.sum().cpu())
        normalized_count = int(normalized_mask.sum().cpu())
        denormalized_count = int(denormalized_mask.sum().cpu())
        if normalized_count != denormalized_count:
            raise RuntimeError(
                "Normalized and denormalized open-loop masks have different element counts"
            )
        valid_elements += normalized_count

        for index, episode_index in enumerate(episode_values):
            normalized_sample_mask = normalized_mask[index]
            denormalized_sample_mask = denormalized_mask[index]
            normalized_sample_count = int(normalized_sample_mask.sum().cpu())
            denormalized_sample_count = int(denormalized_sample_mask.sum().cpu())
            if normalized_sample_count != denormalized_sample_count:
                raise RuntimeError(
                    "Normalized and denormalized per-episode masks have different element counts"
                )
            target_row = per_episode[episode_index]
            target_row["normalized_sum"] += float(
                normalized_squared[index].masked_select(normalized_sample_mask).sum().cpu()
            )
            target_row["denormalized_sum"] += float(
                denormalized_squared[index].masked_select(denormalized_sample_mask).sum().cpu()
            )
            target_row["valid_elements"] += normalized_sample_count

    if valid_elements == 0:
        raise RuntimeError("Audit open-loop evaluation found no valid action targets")
    episode_rows = []
    by_episode = {episode.episode_index: episode for episode in episodes}
    for episode_index in sorted(per_episode):
        values = per_episode[episode_index]
        count = int(values["valid_elements"])
        episode = by_episode[episode_index]
        episode_rows.append(
            {
                "model": loaded.policy.config.type,
                "episode_index": episode_index,
                "oracle_prototype_id": episode.prototype_id,
                "gt_mode_classifiable": episode.classifiable,
                "valid_action_elements": count,
                "normalized_action_mse": float(values["normalized_sum"]) / count,
                "denormalized_action_mse": float(values["denormalized_sum"]) / count,
            }
        )
    return (
        {
            "model": loaded.policy.config.type,
            "episode_count": len(episodes),
            "frame_count": len(dataset),
            "valid_action_elements": valid_elements,
            "normalized_action_mse": normalized_sum / valid_elements,
            "denormalized_action_mse": denormalized_sum / valid_elements,
            "mean_batch_inference_latency_ms": float(np.mean(inference_latencies)),
            "p95_batch_inference_latency_ms": float(np.percentile(inference_latencies, 95)),
            "batch_size": batch_size,
        },
        episode_rows,
    )


def _closed_loop_row(
    *,
    result: Any,
    episode: OracleModeEpisode,
    selected: Mapping[str, Any],
    commanded_condition_id: int,
    commanded_condition_name: str,
    null_prototype_id: int,
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    latencies = [float(record.latency_ms) for record in result.replans]
    gt_mode_name = f"P{episode.prototype_id}"
    nullable_classification = dict(classification)
    if commanded_condition_id == null_prototype_id:
        nullable_classification["prototype_adherent"] = None
        nullable_classification["controlled_success"] = None
    recording_name = _recording_name(
        model=result.model,
        condition_name=commanded_condition_name,
        episode_index=episode.episode_index,
        gt_mode_name=gt_mode_name,
        success=bool(result.success),
    )
    recording_metadata = _recording_metadata(
        model=result.model,
        condition_name=commanded_condition_name,
        episode_index=episode.episode_index,
        gt_mode_name=gt_mode_name,
        success=bool(result.success),
    )
    return {
        "model": result.model,
        "selected_step": int(selected["step"]),
        "checkpoint_id": selected["checkpoint_id"],
        "episode_index": episode.episode_index,
        "split": episode.split,
        "oracle_prototype_id": episode.prototype_id,
        "oracle_prototype_name": gt_mode_name,
        "commanded_condition_id": commanded_condition_id,
        "commanded_condition_name": commanded_condition_name,
        "is_null_condition": commanded_condition_id == null_prototype_id,
        "is_oracle_condition": commanded_condition_id == episode.prototype_id,
        "gt_mode_classifiable": episode.classifiable,
        "distance_to_medoid": episode.distance_to_medoid,
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
        "recording_name": recording_name,
        "recording_metadata": "\n".join(
            f"{key}: {value}" for key, value in recording_metadata.items()
        ),
        "recording_has_identity_panel": True,
        **nullable_classification,
    }


def _evaluate_closed_loop_episode(
    loaded: Any,
    *,
    episode: OracleModeEpisode,
    selected: Mapping[str, Any],
    commanded_condition_id: int,
    selected_k: int,
    null_prototype_id: int,
    classifier: Any,
    temporary: Path,
) -> dict[str, Any]:
    commanded_condition_name = _condition_name(
        commanded_condition_id,
        null_prototype_id=null_prototype_id,
    )
    result = run_oracle_mode_rollout(
        loaded,
        episode=episode,
        commanded_condition_id=commanded_condition_id,
        selected_k=selected_k,
        null_prototype_id=null_prototype_id,
        record_frames=True,
        episode_length=EPISODE_LENGTH,
        policy_seed=POLICY_SEED,
    )
    classification = classify_rollout(
        result,
        classifier=classifier,
        commanded_condition=commanded_condition_id,
        null_prototype_id=null_prototype_id,
    )
    row = _closed_loop_row(
        result=result,
        episode=episode,
        selected=selected,
        commanded_condition_id=commanded_condition_id,
        commanded_condition_name=commanded_condition_name,
        null_prototype_id=null_prototype_id,
        classification=classification,
    )
    gt_mode_name = f"P{episode.prototype_id}"
    record_single_rollout(
        output_path=_recording_path(
            temporary / "recordings",
            model=result.model,
            condition_name=commanded_condition_name,
            episode_index=episode.episode_index,
        ),
        rollout=result,
        recording_id=(
            f"e5_audit_{result.model}_{commanded_condition_name.lower()}"
            f"_episode_{episode.episode_index:06d}"
        ),
        recording_name=str(row["recording_name"]),
        metadata=_recording_metadata(
            model=result.model,
            condition_name=commanded_condition_name,
            episode_index=episode.episode_index,
            gt_mode_name=gt_mode_name,
            success=bool(result.success),
        ),
        show_metadata=True,
    )
    return row


def _model_metrics(
    model: str,
    rows: Sequence[Mapping[str, Any]],
    open_loop: Mapping[str, Any],
    *,
    selected_k: int,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if row["model"] == model]
    successes = sum(bool(row["success"]) for row in selected_rows)
    total = len(selected_rows)
    lower, upper = wilson_interval(successes, total)
    latencies = [
        float(value) for row in selected_rows for value in row.get("replan_latencies_ms", [])
    ]
    result: dict[str, Any] = {
        f"{model}_success_count": successes,
        f"{model}_episode_count": total,
        f"{model}_success_rate": successes / total,
        f"{model}_success_wilson95_low": lower,
        f"{model}_success_wilson95_high": upper,
        f"{model}_mean_max_reward": float(
            np.mean([float(row["max_reward"]) for row in selected_rows])
        ),
        f"{model}_open_loop_action_mse": open_loop["denormalized_action_mse"],
        f"{model}_prototype_classifiable_rate": float(
            np.mean([bool(row["classifiable"]) for row in selected_rows])
        ),
        f"{model}_prototype_adherence_rate": float(
            np.mean([bool(row["prototype_adherent"]) for row in selected_rows])
        ),
        f"{model}_controlled_success_rate": float(
            np.mean([bool(row["controlled_success"]) for row in selected_rows])
        ),
        f"{model}_failure_count": total - successes,
        f"{model}_mean_replan_latency_ms": (float(np.mean(latencies)) if latencies else None),
        f"{model}_p95_replan_latency_ms": (
            float(np.percentile(latencies, 95)) if latencies else None
        ),
    }
    for prototype_id in range(selected_k):
        subset = [row for row in selected_rows if int(row["oracle_prototype_id"]) == prototype_id]
        if subset:
            result[f"{model}_P{prototype_id}_success_rate"] = float(
                np.mean([bool(row["success"]) for row in subset])
            )
            result[f"{model}_P{prototype_id}_episode_count"] = len(subset)
    for label, classifiable in (("classifiable", True), ("unclassifiable", False)):
        subset = [row for row in selected_rows if bool(row["gt_mode_classifiable"]) is classifiable]
        if subset:
            result[f"{model}_{label}_success_rate"] = float(
                np.mean([bool(row["success"]) for row in subset])
            )
            result[f"{model}_{label}_episode_count"] = len(subset)
    return result


def _condition_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_k: int,
    null_prototype_id: int,
) -> list[dict[str, Any]]:
    result = []
    for model in (MODE_BC, MODE_DIFFUSION):
        for condition_id in _condition_ids(
            selected_k=selected_k,
            null_prototype_id=null_prototype_id,
        ):
            selected_rows = [
                row
                for row in rows
                if row["model"] == model and int(row["commanded_condition_id"]) == condition_id
            ]
            if not selected_rows:
                raise ValueError(f"No {model} rows found for commanded condition {condition_id}")
            successes = sum(bool(row["success"]) for row in selected_rows)
            total = len(selected_rows)
            lower, upper = wilson_interval(successes, total)
            latencies = [
                float(value)
                for row in selected_rows
                for value in row.get("replan_latencies_ms", [])
            ]
            is_null = condition_id == null_prototype_id
            result.append(
                {
                    "model": model,
                    "commanded_condition_id": condition_id,
                    "commanded_condition_name": _condition_name(
                        condition_id,
                        null_prototype_id=null_prototype_id,
                    ),
                    "is_null_condition": is_null,
                    "success_count": successes,
                    "episode_count": total,
                    "failure_count": total - successes,
                    "success_rate": successes / total,
                    "success_wilson95_low": lower,
                    "success_wilson95_high": upper,
                    "mean_max_reward": float(
                        np.mean([float(row["max_reward"]) for row in selected_rows])
                    ),
                    "prototype_classifiable_rate": float(
                        np.mean([bool(row["classifiable"]) for row in selected_rows])
                    ),
                    "prototype_adherence_rate": (
                        None
                        if is_null
                        else float(
                            np.mean([bool(row["prototype_adherent"]) for row in selected_rows])
                        )
                    ),
                    "controlled_success_rate": (
                        None
                        if is_null
                        else float(
                            np.mean([bool(row["controlled_success"]) for row in selected_rows])
                        )
                    ),
                    "mean_replan_latency_ms": (float(np.mean(latencies)) if latencies else None),
                    "p95_replan_latency_ms": (
                        float(np.percentile(latencies, 95)) if latencies else None
                    ),
                }
            )
    return result


def _gt_commanded_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_k: int,
    null_prototype_id: int,
) -> list[dict[str, Any]]:
    result = []
    for model in (MODE_BC, MODE_DIFFUSION):
        for gt_prototype_id in range(selected_k):
            for condition_id in _condition_ids(
                selected_k=selected_k,
                null_prototype_id=null_prototype_id,
            ):
                selected_rows = [
                    row
                    for row in rows
                    if row["model"] == model
                    and int(row["oracle_prototype_id"]) == gt_prototype_id
                    and int(row["commanded_condition_id"]) == condition_id
                ]
                if not selected_rows:
                    raise ValueError(
                        f"No {model} rows found for GT P{gt_prototype_id} "
                        f"and commanded condition {condition_id}"
                    )
                successes = sum(bool(row["success"]) for row in selected_rows)
                total = len(selected_rows)
                result.append(
                    {
                        "model": model,
                        "gt_prototype_id": gt_prototype_id,
                        "gt_prototype_name": f"P{gt_prototype_id}",
                        "commanded_condition_id": condition_id,
                        "commanded_condition_name": _condition_name(
                            condition_id,
                            null_prototype_id=null_prototype_id,
                        ),
                        "success_count": successes,
                        "episode_count": total,
                        "success_rate": successes / total,
                    }
                )
    return result


def _flatten_condition_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    excluded = {"model", "commanded_condition_id", "commanded_condition_name"}
    for row in rows:
        prefix = f"{row['model']}_{row['commanded_condition_name']}_all_tasks"
        for key, value in row.items():
            if key not in excluded:
                metrics[f"{prefix}_{key}"] = value
    return metrics


def _validate_execution_request(
    *,
    num_gpus: int,
    device: str | None,
    cuda_available: bool | None = None,
    visible_device_count: int | None = None,
) -> str | None:
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    if num_gpus == 1:
        return device or "cuda:0"
    if device not in (None, "cuda:0"):
        raise ValueError("device cannot be set when num_gpus > 1")
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if not available:
        raise RuntimeError("multi-GPU E5 audit requires CUDA")
    visible = torch.cuda.device_count() if visible_device_count is None else visible_device_count
    if num_gpus > visible:
        raise RuntimeError(
            f"num_gpus={num_gpus} requested, but only {visible} CUDA device(s) are visible"
        )
    return None


def _configure_loaded_policy(
    loaded: Any,
    *,
    model: str,
    selected_k: int,
) -> None:
    validate_mode_policy(
        loaded.policy,
        expected_model=model,
        selected_k=selected_k,
    )
    if model == MODE_DIFFUSION:
        replace_scheduler(
            loaded.policy,
            scheduler_name="DDPM",
            num_inference_steps=NUM_INFERENCE_STEPS,
        )


def _evaluate_single_device(
    *,
    selected: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    temporary: Path,
    episodes: Sequence[OracleModeEpisode],
    selected_k: int,
    null_prototype_id: int,
    classifier: Any,
    device: str,
    open_loop_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    closed_loop_rows = []
    open_loop_rows = []
    open_loop_episode_rows = []
    for model in (MODE_BC, MODE_DIFFUSION):
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
                selected_k=selected_k,
            )
            open_loop, per_episode = evaluate_full_open_loop(
                loaded,
                dataset_root=dataset_root,
                episodes=episodes,
                selected_k=selected_k,
                batch_size=open_loop_batch_size,
            )
            open_loop["selected_step"] = int(selection["step"])
            open_loop["checkpoint_id"] = selection["checkpoint_id"]
            open_loop_rows.append(open_loop)
            open_loop_episode_rows.extend(per_episode)

            conditions = _condition_ids(
                selected_k=selected_k,
                null_prototype_id=null_prototype_id,
            )
            progress = tqdm(
                total=len(episodes) * len(conditions),
                desc=f"Audit closed-loop {model}",
                unit="rollout",
            )
            try:
                for episode in episodes:
                    for commanded_condition_id in conditions:
                        closed_loop_rows.append(
                            _evaluate_closed_loop_episode(
                                loaded,
                                episode=episode,
                                selected=selection,
                                commanded_condition_id=commanded_condition_id,
                                selected_k=selected_k,
                                null_prototype_id=null_prototype_id,
                                classifier=classifier,
                                temporary=temporary,
                            )
                        )
                        progress.update(1)
            finally:
                progress.close()
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return closed_loop_rows, open_loop_rows, open_loop_episode_rows


def _audit_worker(
    plan: AuditWorkerPlan,
    config: AuditWorkerConfig,
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
    commanded_condition_id = None
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
            selected_k=config.selected_k,
        )
        classifier = _load_classifier(load_prototype_sidecar(config.prototype_manifest))
        event_queue.put(
            AuditWorkerEvent(
                kind="ready",
                gpu_index=plan.gpu_index,
                model=plan.model,
                phase="validated",
            )
        )
        phase = "waiting"
        start_event.wait()

        open_loop_row = None
        open_loop_episode_rows: tuple[dict[str, Any], ...] = ()
        if plan.open_loop_leader:
            phase = "open_loop"
            event_queue.put(
                AuditWorkerEvent(
                    kind="progress",
                    gpu_index=plan.gpu_index,
                    model=plan.model,
                    phase=phase,
                )
            )
            open_loop_row, per_episode = evaluate_full_open_loop(
                loaded,
                dataset_root=config.dataset_root,
                episodes=config.episodes,
                selected_k=config.selected_k,
                batch_size=config.open_loop_batch_size,
                show_progress=False,
            )
            open_loop_row["selected_step"] = int(selection["step"])
            open_loop_row["checkpoint_id"] = selection["checkpoint_id"]
            open_loop_episode_rows = tuple(per_episode)

        by_episode = {episode.episode_index: episode for episode in config.episodes}
        closed_loop_rows = []
        for assigned_episode_index in plan.episode_indices:
            episode_index = assigned_episode_index
            for condition_id in _condition_ids(
                selected_k=config.selected_k,
                null_prototype_id=config.null_prototype_id,
            ):
                commanded_condition_id = condition_id
                phase = "closed_loop"
                event_queue.put(
                    AuditWorkerEvent(
                        kind="progress",
                        gpu_index=plan.gpu_index,
                        model=plan.model,
                        phase=phase,
                        episode_index=episode_index,
                        commanded_condition_id=commanded_condition_id,
                    )
                )
                closed_loop_rows.append(
                    _evaluate_closed_loop_episode(
                        loaded,
                        episode=by_episode[episode_index],
                        selected=selection,
                        commanded_condition_id=commanded_condition_id,
                        selected_k=config.selected_k,
                        null_prototype_id=config.null_prototype_id,
                        classifier=classifier,
                        temporary=config.temporary,
                    )
                )
                event_queue.put(
                    AuditWorkerEvent(
                        kind="progress",
                        gpu_index=plan.gpu_index,
                        model=plan.model,
                        phase="rollout_complete",
                        episode_index=episode_index,
                        commanded_condition_id=commanded_condition_id,
                    )
                )
        event_queue.put(
            AuditWorkerEvent(
                kind="result",
                gpu_index=plan.gpu_index,
                model=plan.model,
                phase="complete",
                closed_loop_rows=tuple(closed_loop_rows),
                open_loop_row=open_loop_row,
                open_loop_episode_rows=open_loop_episode_rows,
            )
        )
    except BaseException:  # noqa: BLE001 - relay the complete worker failure to the parent.
        event_queue.put(
            AuditWorkerEvent(
                kind="error",
                gpu_index=plan.gpu_index,
                model=plan.model,
                phase=phase,
                episode_index=episode_index,
                commanded_condition_id=commanded_condition_id,
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
) -> AuditWorkerEvent:
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
            raise RuntimeError(f"Audit worker exited unexpectedly: {details}") from None
        if not any(worker.is_alive() for worker in workers):
            raise RuntimeError("All audit workers exited before producing results") from None
        raise
    if not isinstance(event, AuditWorkerEvent):
        raise TypeError(f"Received invalid audit worker event: {event!r}")
    return event


def _raise_worker_error(event: AuditWorkerEvent) -> None:
    raise RuntimeError(
        f"GPU {event.gpu_index} failed for {event.model} during {event.phase}"
        + (f" episode {event.episode_index}" if event.episode_index is not None else "")
        + (
            f" condition {event.commanded_condition_id}"
            if event.commanded_condition_id is not None
            else ""
        )
        + f":\n{event.error}"
    )


def _evaluate_multi_gpu(
    *,
    plans: Sequence[AuditWorkerPlan],
    config: AuditWorkerConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plan_by_gpu = {plan.gpu_index: plan for plan in plans}
    if len(plan_by_gpu) != len(plans):
        raise ValueError("Audit worker plans contain duplicate GPU indices")
    context = mp.get_context("spawn")
    event_queue = context.Queue()
    start_event = context.Event()
    workers = [
        context.Process(
            target=_audit_worker,
            args=(plan, config, start_event, event_queue),
            name=f"e5-audit-gpu-{plan.gpu_index}",
        )
        for plan in plans
    ]
    for worker in workers:
        worker.start()
    worker_gpu_indices = [plan.gpu_index for plan in plans]

    ready: set[int] = set()
    results: dict[int, AuditWorkerEvent] = {}
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

        condition_count = len(
            _condition_ids(
                selected_k=config.selected_k,
                null_prototype_id=config.null_prototype_id,
            )
        )
        with tqdm(
            total=sum(len(plan.episode_indices) for plan in plans) * condition_count,
            desc="E5 multi-GPU audit",
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
                        if event.commanded_condition_id is not None:
                            values["condition"] = _condition_name(
                                event.commanded_condition_id,
                                null_prototype_id=config.null_prototype_id,
                            )
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
        raise RuntimeError(f"Audit worker shutdown failed: {details}")

    closed_loop_rows = []
    open_loop_rows = []
    open_loop_episode_rows = []
    for gpu_index in sorted(results):
        event = results[gpu_index]
        closed_loop_rows.extend(event.closed_loop_rows)
        if event.open_loop_row is not None:
            open_loop_rows.append(event.open_loop_row)
            open_loop_episode_rows.extend(event.open_loop_episode_rows)
    return closed_loop_rows, open_loop_rows, open_loop_episode_rows


def _validate_exact_model_episode_rows(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_indices: Sequence[int],
) -> None:
    expected = {
        (model, int(episode_index))
        for model in (MODE_BC, MODE_DIFFUSION)
        for episode_index in episode_indices
    }
    actual = [(str(row["model"]), int(row["episode_index"])) for row in rows]
    duplicates = sorted({key for key in actual if actual.count(key) > 1})
    if duplicates:
        raise ValueError(f"{label} contains duplicate model/episode rows: {duplicates}")
    actual_set = set(actual)
    missing = sorted(expected - actual_set)
    unexpected = sorted(actual_set - expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} model/episode coverage mismatch; missing={missing}, unexpected={unexpected}"
        )


def _validate_exact_condition_sweep_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    episode_indices: Sequence[int],
    selected_k: int,
    null_prototype_id: int,
) -> None:
    conditions = _condition_ids(
        selected_k=selected_k,
        null_prototype_id=null_prototype_id,
    )
    expected = {
        (model, int(episode_index), condition_id)
        for model in (MODE_BC, MODE_DIFFUSION)
        for episode_index in episode_indices
        for condition_id in conditions
    }
    actual = [
        (
            str(row["model"]),
            int(row["episode_index"]),
            int(row["commanded_condition_id"]),
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


def _validate_audit_results(
    closed_loop_rows: Sequence[dict[str, Any]],
    open_loop_rows: Sequence[dict[str, Any]],
    open_loop_episode_rows: Sequence[dict[str, Any]],
    *,
    episode_indices: Sequence[int],
    selected_k: int,
    null_prototype_id: int,
    recordings_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_exact_condition_sweep_rows(
        closed_loop_rows,
        episode_indices=episode_indices,
        selected_k=selected_k,
        null_prototype_id=null_prototype_id,
    )
    _validate_exact_model_episode_rows(
        "open-loop per-episode",
        open_loop_episode_rows,
        episode_indices=episode_indices,
    )
    open_loop_models = [str(row["model"]) for row in open_loop_rows]
    expected_models = {MODE_BC, MODE_DIFFUSION}
    if len(open_loop_models) != len(set(open_loop_models)):
        raise ValueError("Open-loop summaries contain duplicate models")
    if set(open_loop_models) != expected_models:
        raise ValueError(
            "Open-loop summary model coverage mismatch; "
            f"expected={sorted(expected_models)}, actual={sorted(open_loop_models)}"
        )
    expected_episode_count = len(tuple(episode_indices))
    for row in open_loop_rows:
        if int(row["episode_count"]) != expected_episode_count:
            raise ValueError(
                f"Open-loop {row['model']} episode_count must be {expected_episode_count}"
            )
    for row in closed_loop_rows:
        model = str(row["model"])
        episode_index = int(row["episode_index"])
        commanded_condition_id = int(row["commanded_condition_id"])
        condition_name = _condition_name(
            commanded_condition_id,
            null_prototype_id=null_prototype_id,
        )
        gt_mode_name = f"P{int(row['oracle_prototype_id'])}"
        expected_name = _recording_name(
            model=model,
            condition_name=condition_name,
            episode_index=episode_index,
            gt_mode_name=gt_mode_name,
            success=bool(row["success"]),
        )
        expected_metadata = "\n".join(
            f"{key}: {value}"
            for key, value in _recording_metadata(
                model=model,
                condition_name=condition_name,
                episode_index=episode_index,
                gt_mode_name=gt_mode_name,
                success=bool(row["success"]),
            ).items()
        )
        if row.get("commanded_condition_name") != condition_name:
            raise ValueError("Closed-loop commanded condition name is inconsistent")
        if row.get("recording_name") != expected_name:
            raise ValueError("Closed-loop recording name does not expose model/condition identity")
        if row.get("recording_metadata") != expected_metadata:
            raise ValueError("Closed-loop recording metadata identity is inconsistent")
        if row.get("recording_has_identity_panel") is not True:
            raise ValueError("Closed-loop recording must expose the identity metadata panel")
        recording = _recording_path(
            recordings_root,
            model=model,
            condition_name=condition_name,
            episode_index=episode_index,
        )
        if not recording.is_file() or recording.stat().st_size == 0:
            raise FileNotFoundError(f"Audit recording is missing or empty: {recording}")

    model_order = {MODE_BC: 0, MODE_DIFFUSION: 1}
    sorted_closed = sorted(
        closed_loop_rows,
        key=lambda row: (
            model_order[str(row["model"])],
            int(row["commanded_condition_id"]),
            int(row["episode_index"]),
        ),
    )
    sorted_open = sorted(
        open_loop_rows,
        key=lambda row: model_order[str(row["model"])],
    )
    sorted_open_episodes = sorted(
        open_loop_episode_rows,
        key=lambda row: (model_order[str(row["model"])], int(row["episode_index"])),
    )
    return sorted_closed, sorted_open, sorted_open_episodes


def run_e5_oracle_audit(
    *,
    selection_dir: Path,
    prototype_manifest: Path,
    dataset_root: Path,
    output_dir: Path,
    device: str | None,
    open_loop_batch_size: int,
    overwrite: bool,
    num_gpus: int = 1,
) -> Path:
    actual_device = _validate_execution_request(
        num_gpus=num_gpus,
        device=device,
    )
    selected = _selected_rows(selection_dir)
    selection_manifest = load_json_object(selection_dir / "run_manifest.json")
    expected_prototype_hash = selection_manifest.get("prototype_manifest_sha256")
    actual_prototype_hash = sha256_file(prototype_manifest)
    if expected_prototype_hash != actual_prototype_hash:
        raise ValueError(
            "Audit prototype manifest does not match checkpoint selection: "
            f"{expected_prototype_hash!r} != {actual_prototype_hash!r}"
        )
    split = load_split()
    sidecar = load_prototype_sidecar(prototype_manifest)
    commanded_condition_ids = _condition_ids(
        selected_k=sidecar.selected_k,
        null_prototype_id=sidecar.null_prototype_id,
    )
    episodes = load_oracle_mode_episodes(
        dataset_root=dataset_root,
        prototype_manifest=prototype_manifest,
        episode_indices=split.audit,
        split_name="audit",
    )
    reconstruction = preflight_reconstruction(episodes)
    classifier = _load_classifier(sidecar)
    runtime = collect_runtime_provenance(practice_root())
    plans = (
        ()
        if num_gpus == 1
        else _build_worker_plans(
            episode_indices=split.audit,
            num_gpus=num_gpus,
        )
    )

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        if num_gpus == 1:
            if actual_device is None:
                raise RuntimeError("Single-device audit did not resolve an execution device")
            closed_loop_rows, open_loop_rows, open_loop_episode_rows = _evaluate_single_device(
                selected=selected,
                dataset_root=dataset_root,
                temporary=temporary,
                episodes=episodes,
                selected_k=sidecar.selected_k,
                null_prototype_id=sidecar.null_prototype_id,
                classifier=classifier,
                device=actual_device,
                open_loop_batch_size=open_loop_batch_size,
            )
        else:
            worker_config = AuditWorkerConfig(
                selected=selected,
                prototype_manifest=prototype_manifest,
                dataset_root=dataset_root,
                temporary=temporary,
                episodes=tuple(episodes),
                selected_k=sidecar.selected_k,
                null_prototype_id=sidecar.null_prototype_id,
                open_loop_batch_size=open_loop_batch_size,
            )
            closed_loop_rows, open_loop_rows, open_loop_episode_rows = _evaluate_multi_gpu(
                plans=plans,
                config=worker_config,
            )
        closed_loop_rows, open_loop_rows, open_loop_episode_rows = _validate_audit_results(
            closed_loop_rows,
            open_loop_rows,
            open_loop_episode_rows,
            episode_indices=split.audit,
            selected_k=sidecar.selected_k,
            null_prototype_id=sidecar.null_prototype_id,
            recordings_root=temporary / "recordings",
        )
        condition_metric_rows = _condition_metric_rows(
            closed_loop_rows,
            selected_k=sidecar.selected_k,
            null_prototype_id=sidecar.null_prototype_id,
        )
        gt_commanded_metric_rows = _gt_commanded_metric_rows(
            closed_loop_rows,
            selected_k=sidecar.selected_k,
            null_prototype_id=sidecar.null_prototype_id,
        )
        oracle_rows = [row for row in closed_loop_rows if bool(row["is_oracle_condition"])]
        expected_oracle_rows = len(episodes) * 2
        if len(oracle_rows) != expected_oracle_rows:
            raise ValueError(
                f"Oracle subset contains {len(oracle_rows)} rows, expected {expected_oracle_rows}"
            )

        write_records_parquet(temporary / "closed_loop_rollouts.parquet", closed_loop_rows)
        write_records_parquet(
            temporary / "condition_metrics.parquet",
            condition_metric_rows,
        )
        write_records_parquet(
            temporary / "gt_commanded_metrics.parquet",
            gt_commanded_metric_rows,
        )
        write_records_parquet(temporary / "open_loop_metrics.parquet", open_loop_rows)
        write_records_parquet(
            temporary / "open_loop_episode_metrics.parquet",
            open_loop_episode_rows,
        )
        write_records_parquet(
            temporary / "reconstruction_metrics.parquet",
            (metric.as_record() for metric in reconstruction),
        )
        write_json_atomic(temporary / "selected_checkpoints.json", selected)
        write_json_atomic(
            temporary / "rng_manifest.json",
            {
                "policy_seed": POLICY_SEED,
                "diffusion_scheduler": "DDPM",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "diffusion_sampling_protocol": "full-ddpm-sampling",
                "closed_loop_environment_seed_namespace": "dataset_episode_index",
                "open_loop_noise_namespace": "batch_ordinal/policy_seed",
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 2,
                "status": "complete",
                "label": "Closed-loop and Open-loop",
                "experiment": "e5_oracle_mode_audit",
                "evaluation_protocol": "held_out_all_condition_sweep",
                "dataset_revision": DATASET_REVISION,
                "audit_episodes": list(split.audit),
                "selection_dir": str(selection_dir),
                "prototype_manifest": str(prototype_manifest),
                "selected_k": sidecar.selected_k,
                "mode_source": "full_demonstration_nearest_train_medoid",
                "commanded_conditions": [
                    {
                        "condition_id": condition_id,
                        "condition_name": _condition_name(
                            condition_id,
                            null_prototype_id=sidecar.null_prototype_id,
                        ),
                    }
                    for condition_id in commanded_condition_ids
                ],
                "conditions_per_episode": len(commanded_condition_ids),
                "expected_closed_loop_rollouts": (len(episodes) * 2 * len(commanded_condition_ids)),
                "actual_closed_loop_rollouts": len(closed_loop_rows),
                "oracle_metrics_source": "is_oracle_condition=true sweep subset",
                "initial_state_source": "dataset_first_frame_reconstruction",
                "checkpoint_path_resolution": _checkpoint_path_resolution_summary(selected),
                "execution": {
                    "device": actual_device,
                    "parallelism": (
                        "single_device_sequential"
                        if num_gpus == 1
                        else "multi_gpu_model_episode_shards"
                    ),
                    "num_gpus": num_gpus,
                    "worker_plan": [plan.as_record() for plan in plans],
                    "open_loop_batch_size": open_loop_batch_size,
                },
                "runtime": runtime,
            },
        )
        metrics: dict[str, Any] = {
            "audit_episode_count": len(episodes),
            "closed_loop_rollout_count": len(closed_loop_rows),
            "oracle_closed_loop_rollout_count": len(oracle_rows),
            "reconstruction_preflight_passed": len(reconstruction),
        }
        metrics.update(_flatten_condition_metrics(condition_metric_rows))
        open_loop_by_model = {str(row["model"]): row for row in open_loop_rows}
        for model in (MODE_BC, MODE_DIFFUSION):
            metrics.update(
                _model_metrics(
                    model,
                    oracle_rows,
                    open_loop_by_model[model],
                    selected_k=sidecar.selected_k,
                )
            )
        write_metrics(
            temporary,
            metrics,
            comments={
                "audit_episode_count": "Held-out committed audit episodes",
                "closed_loop_rollout_count": ("Every model/audit-episode/P0-P1-P2-NULL rollout"),
                "oracle_closed_loop_rollout_count": (
                    "GT-command rows used for legacy oracle metrics"
                ),
                "reconstruction_preflight_passed": "No audit episode was silently excluded",
            },
        )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--prototype-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--open-loop-batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e5_oracle_audit",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_e5_oracle_audit(
        selection_dir=args.selection_dir.expanduser().resolve(),
        prototype_manifest=args.prototype_manifest.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        open_loop_batch_size=args.open_loop_batch_size,
        overwrite=args.overwrite,
        num_gpus=args.num_gpus,
    )
    print(f"E5 oracle audit output: {result}")


if __name__ == "__main__":
    main()
