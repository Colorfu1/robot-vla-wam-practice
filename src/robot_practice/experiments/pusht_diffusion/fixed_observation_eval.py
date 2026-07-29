"""E1 fixed-observation BCChunk and explicit-noise Diffusion evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from torch import Tensor

from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    PROTOCOL_NAMESPACE,
    default_dataset_root,
    default_output_root,
    load_named_config,
)
from .diversity_metrics import summarize_action_diversity
from .noise import (
    E1A_REVERSE_NOISE_STREAM,
    E1B_REVERSE_NOISE_STREAM,
    INITIAL_NOISE_STREAM,
    fixed_observation_e1a_reverse_seed,
    fixed_observation_e1b_reverse_seed,
    fixed_observation_initial_noise_seed,
    make_initial_noise,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import (
    controlled_sample,
    load_policy_with_processors,
    replace_scheduler,
)
from .provenance import (
    collect_runtime_provenance,
    sha256_file,
    sha256_tensor,
    write_json_atomic,
)
from .rerun_visualization import image_to_hwc_uint8, record_fixed_observation_samples

_E1_CONFIG = load_named_config("e1_sampling.json")
_FORMAL_NUM_ACTION_SAMPLES = 20
_EXPECTED_BC_REPEAT_CALLS = 3
_E1_SCHEMA_VERSION = 2
_E1_EXPERIMENT_SCHEMA = "pusht_e1_paired_sampling_v2"
_INITIAL_NOISE_ONLY = "initial_noise_only"
_FULL_SAMPLING_SEED = "full_sampling_seed"
_BC_DETERMINISTIC = "bc_deterministic"
_DIFFUSION_PROTOCOLS = (_INITIAL_NOISE_ONLY, _FULL_SAMPLING_SEED)
_RECORDING_FILENAMES = {
    _INITIAL_NOISE_ONLY: "recording_initial_noise_only.rrd",
    _FULL_SAMPLING_SEED: "recording_full_sampling_seed.rrd",
}


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path} does not contain a non-empty candidates array")
    return candidates


def _order_and_validate_candidates(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordinals = [int(candidate["candidate_ordinal"]) for candidate in candidates]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError(f"candidate_ordinal values must be unique, got {ordinals}")
    expected = list(range(len(candidates)))
    if sorted(ordinals) != expected:
        raise ValueError(
            "candidate_ordinal values must form a zero-based contiguous sequence; "
            f"expected {expected}, got {sorted(ordinals)}"
        )
    return sorted(candidates, key=lambda candidate: int(candidate["candidate_ordinal"]))


def _index_frames(dataset: LeRobotDataset) -> dict[tuple[int, int], int]:
    dataset._ensure_hf_dataset_loaded()
    grouped_counts: dict[int, int] = defaultdict(int)
    result: dict[tuple[int, int], int] = {}
    for local_index, raw_episode in enumerate(dataset.hf_dataset["episode_index"]):
        episode = int(raw_episode.item() if hasattr(raw_episode, "item") else raw_episode)
        frame = grouped_counts[episode]
        grouped_counts[episode] += 1
        result[(episode, frame)] = local_index
    return result


def _observation_only(item: dict[str, Any], policy: torch.nn.Module) -> dict[str, Tensor]:
    missing = [key for key in policy.config.input_features if key not in item]
    if missing:
        raise KeyError(f"Dataset item is missing policy inputs: {missing}")
    return {
        key: item[key].unsqueeze(0)
        for key in policy.config.input_features
        if isinstance(item[key], Tensor)
    }


def _stack_images(policy: torch.nn.Module, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    prepared = dict(batch)
    if policy.config.image_features:
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[key] for key in policy.config.image_features],
            dim=-4,
        )
    return prepared


def _postprocess_chunk(postprocessor: Any, chunk: Tensor) -> np.ndarray:
    value = postprocessor(chunk)
    if not isinstance(value, Tensor):
        raise TypeError("Policy postprocessor must return a Tensor")
    result = value.detach().cpu().numpy()
    if result.ndim != 3 or result.shape[0] != 1 or result.shape[-1] != 2:
        raise ValueError(f"Expected postprocessed chunk [1,steps,2], got {result.shape}")
    return result[0]


def _tensor_payload(batch: dict[str, Tensor]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for key, value in sorted(batch.items()):
        if isinstance(value, Tensor):
            arrays[key.replace(".", "__")] = value.detach().cpu().numpy()
            hashes[key] = sha256_tensor(value)
    return arrays, hashes


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _aggregate_e1_metrics(
    metric_rows: Sequence[dict[str, Any]],
    *,
    num_candidates: int,
    num_action_samples: int,
    bc_repeat_max_error: float,
) -> dict[str, Any]:
    fields = (
        "normalized_trajectory_pairwise_rms",
        "normalized_endpoint_pairwise_distance",
        "denormalized_trajectory_pairwise_rms",
        "denormalized_endpoint_pairwise_distance",
    )
    by_protocol = {
        protocol: [
            row for row in metric_rows if row["sampling_protocol"] == protocol
        ]
        for protocol in _DIFFUSION_PROTOCOLS
    }
    for protocol, rows in by_protocol.items():
        if len(rows) != num_candidates:
            raise ValueError(
                f"E1 protocol {protocol} has {len(rows)} diversity rows; "
                f"expected {num_candidates}"
            )

    result: dict[str, Any] = {
        "schema_version": _E1_SCHEMA_VERSION,
        "num_candidates": num_candidates,
        "num_action_samples_per_protocol": num_action_samples,
        "bc_repeat_max_error": float(bc_repeat_max_error),
        "num_rollouts": 0,
    }
    protocol_means: dict[str, dict[str, float]] = {}
    for protocol, rows in by_protocol.items():
        means = {
            field: float(np.mean([float(row[field]) for row in rows]))
            for field in fields
        }
        protocol_means[protocol] = means
        for field, value in means.items():
            result[f"{protocol}_mean_{field}"] = value
        result[f"{protocol}_repeat_max_error"] = float(
            max(float(row["repeat_max_error"]) for row in rows)
        )

    for field in fields:
        result[f"full_sampling_seed_to_initial_noise_only_{field}_ratio"] = _ratio_or_none(
            protocol_means[_FULL_SAMPLING_SEED][field],
            protocol_means[_INITIAL_NOISE_ONLY][field],
        )
    return result


def _validate_e1_rows(
    action_rows: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    *,
    candidates: Sequence[dict[str, Any]],
    num_action_samples: int,
    n_action_steps: int,
    label: str,
) -> None:
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    expected_rows_per_candidate = (
        _EXPECTED_BC_REPEAT_CALLS + 2 * num_action_samples
    ) * n_action_steps
    if label == "Offline" and expected_rows_per_candidate != 344:
        raise RuntimeError(
            f"Formal E1 row contract resolved to {expected_rows_per_candidate}, expected 344"
        )
    for candidate_id in candidate_ids:
        candidate_rows = [
            row for row in action_rows if row["candidate_id"] == candidate_id
        ]
        if len(candidate_rows) != expected_rows_per_candidate:
            raise RuntimeError(
                f"Candidate {candidate_id} has {len(candidate_rows)} action rows; "
                f"expected {expected_rows_per_candidate}"
            )
        bc_rows = [
            row
            for row in candidate_rows
            if row["sampling_protocol"] == _BC_DETERMINISTIC
        ]
        if len(bc_rows) != _EXPECTED_BC_REPEAT_CALLS * n_action_steps:
            raise RuntimeError(
                f"Candidate {candidate_id} violates the three-call BCChunk contract"
            )
        protocol_rows: dict[str, list[dict[str, Any]]] = {}
        for protocol in _DIFFUSION_PROTOCOLS:
            rows = [
                row
                for row in candidate_rows
                if row["sampling_protocol"] == protocol
            ]
            protocol_rows[protocol] = rows
            expected = num_action_samples * n_action_steps
            if len(rows) != expected:
                raise RuntimeError(
                    f"Candidate {candidate_id} protocol {protocol} has {len(rows)} rows; "
                    f"expected {expected}"
                )
            ordinals = sorted({int(row["sample_ordinal"]) for row in rows})
            if ordinals != list(range(num_action_samples)):
                raise RuntimeError(
                    f"Candidate {candidate_id} protocol {protocol} has invalid sample "
                    f"ordinals {ordinals}"
                )

        e1a_reverse = {
            int(row["reverse_noise_seed"])
            for row in protocol_rows[_INITIAL_NOISE_ONLY]
        }
        e1b_reverse = {
            int(row["reverse_noise_seed"])
            for row in protocol_rows[_FULL_SAMPLING_SEED]
        }
        if len(e1a_reverse) != 1:
            raise RuntimeError(
                f"Candidate {candidate_id} protocol {_INITIAL_NOISE_ONLY} must share "
                "one reverse seed"
            )
        if len(e1b_reverse) != num_action_samples:
            raise RuntimeError(
                f"Candidate {candidate_id} protocol {_FULL_SAMPLING_SEED} must have "
                f"{num_action_samples} unique reverse seeds"
            )
        for sample_ordinal in range(num_action_samples):
            paired = []
            for protocol in _DIFFUSION_PROTOCOLS:
                sample_rows = [
                    row
                    for row in protocol_rows[protocol]
                    if int(row["sample_ordinal"]) == sample_ordinal
                ]
                first = sample_rows[0]
                paired.append(
                    (
                        first["initial_noise_seed"],
                        first["initial_noise_sha256"],
                        first["initial_noise_file_sha256"],
                    )
                )
                if len(
                    {
                        (
                            row["initial_noise_seed"],
                            row["initial_noise_sha256"],
                            row["initial_noise_file_sha256"],
                            row["reverse_noise_seed"],
                            row["reverse_generator_state_sha256"],
                        )
                        for row in sample_rows
                    }
                ) != 1:
                    raise RuntimeError(
                        f"Candidate {candidate_id} protocol {protocol} sample "
                        f"{sample_ordinal} has inconsistent seed/hash metadata"
                    )
            if paired[0] != paired[1]:
                raise RuntimeError(
                    f"Candidate {candidate_id} sample {sample_ordinal} is not paired "
                    "across E1a and E1b"
                )

    if len(metric_rows) != 2 * len(candidates):
        raise RuntimeError(
            f"E1 has {len(metric_rows)} diversity rows; expected {2 * len(candidates)}"
        )


def evaluate_fixed_observations(
    *,
    bc_policy_path: str | Path,
    diffusion_policy_path: str | Path,
    candidates_path: Path,
    dataset_root: Path,
    output_dir: Path,
    num_action_samples: int,
    train_seed: int,
    label: str,
    overwrite: bool,
) -> Path:
    if num_action_samples <= 1:
        raise ValueError("num_action_samples must be greater than one")
    if label not in {"Smoke", "Offline"}:
        raise ValueError("label must be Smoke or Offline")
    protocol = _E1_CONFIG
    if int(protocol["bc_repeat_calls"]) != _EXPECTED_BC_REPEAT_CALLS:
        raise ValueError(
            "E1 requires exactly three BCChunk calls per candidate; "
            f"configured {protocol['bc_repeat_calls']}"
        )
    if label == "Offline" and num_action_samples != _FORMAL_NUM_ACTION_SAMPLES:
        raise ValueError(
            "Formal E1 requires exactly 20 Diffusion action samples; "
            f"got {num_action_samples}. Use label=Smoke for a reduced pipeline check."
        )
    candidates = _order_and_validate_candidates(_load_candidates(candidates_path))
    runtime = collect_runtime_provenance(Path(__file__).resolve().parents[4])
    code_commit = runtime["repository"]["commit"]
    checkpoint_ids = {
        "bc_chunk": sha256_file(Path(bc_policy_path).expanduser().resolve() / "model.safetensors"),
        "diffusion": sha256_file(
            Path(diffusion_policy_path).expanduser().resolve() / "model.safetensors"
        ),
    }
    bc_loaded = load_policy_with_processors(bc_policy_path, local_files_only=True)
    diffusion_loaded = load_policy_with_processors(
        diffusion_policy_path,
        local_files_only=True,
    )
    if not isinstance(bc_loaded.policy, BCChunkPolicy):
        raise TypeError("bc-policy must contain policy.type=bc_chunk")
    if not isinstance(diffusion_loaded.policy, DiffusionPolicy):
        raise TypeError("diffusion-policy must contain policy.type=diffusion")
    replace_scheduler(
        diffusion_loaded.policy,
        scheduler_name=str(protocol["scheduler"]),
        num_inference_steps=int(protocol["num_inference_steps"]),
    )
    for field in ("n_obs_steps", "horizon", "n_action_steps"):
        if getattr(bc_loaded.config, field) != getattr(diffusion_loaded.config, field):
            raise ValueError(f"BCChunk and Diffusion disagree on {field}")

    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_root,
        revision=DATASET_REVISION,
    )
    delta_timestamps = resolve_delta_timestamps(diffusion_loaded.config, metadata)
    episode_ids = sorted({int(candidate["episode_index"]) for candidate in candidates})
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        episodes=episode_ids,
        delta_timestamps=delta_timestamps,
        revision=DATASET_REVISION,
    )
    frame_lookup = _index_frames(dataset)
    action_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    bc_repeat_errors: list[float] = []
    visualization_candidates: dict[str, list[dict[str, Any]]] = {
        protocol_name: [] for protocol_name in _DIFFUSION_PROTOCOLS
    }

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        observation_dir = temporary / "observations"
        noise_dir = temporary / "noises"
        processor_dir = temporary / "processor_metadata"
        observation_dir.mkdir()
        noise_dir.mkdir()
        processor_dir.mkdir()
        processor_hashes: dict[str, str] = {}
        for model_name, policy_path in (
            ("bc_chunk", Path(bc_policy_path)),
            ("diffusion", Path(diffusion_policy_path)),
        ):
            for filename in (
                "config.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ):
                source = policy_path.expanduser().resolve() / filename
                if not source.is_file():
                    raise FileNotFoundError(f"Checkpoint metadata is missing: {source}")
                destination = processor_dir / f"{model_name}_{filename}"
                shutil.copy2(source, destination)
                processor_hashes[f"{model_name}/{filename}"] = sha256_file(destination)
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            ordinal = int(candidate["candidate_ordinal"])
            episode_index = int(candidate["episode_index"])
            frame_index = int(candidate["frame_index"])
            key = (episode_index, frame_index)
            if key not in frame_lookup:
                raise KeyError(f"Candidate frame is not present in dataset: {key}")
            item = dataset[frame_lookup[key]]

            bc_processed = bc_loaded.preprocessor(_observation_only(item, bc_loaded.policy))
            diffusion_processed = diffusion_loaded.preprocessor(
                _observation_only(item, diffusion_loaded.policy)
            )
            bc_batch = _stack_images(bc_loaded.policy, bc_processed)
            diffusion_batch = _stack_images(
                diffusion_loaded.policy,
                diffusion_processed,
            )
            for input_key in (OBS_STATE, OBS_IMAGES):
                if (
                    input_key in bc_batch
                    and input_key in diffusion_batch
                    and not torch.allclose(
                        bc_batch[input_key],
                        diffusion_batch[input_key],
                        rtol=0,
                        atol=1e-7,
                    )
                ):
                    raise RuntimeError(
                        f"BCChunk and Diffusion processed inputs differ for {input_key}"
                    )

            arrays, input_hashes = _tensor_payload(diffusion_batch)
            arrays["raw__observation_state"] = np.asarray(item[OBS_STATE])
            for image_key in diffusion_loaded.config.image_features:
                arrays[f"raw__{image_key.replace('.', '__')}"] = np.asarray(item[image_key])
            np.savez_compressed(observation_dir / f"{candidate_id}.npz", **arrays)
            with torch.inference_mode():
                start = bc_loaded.config.n_obs_steps - 1
                end = start + bc_loaded.config.n_action_steps
                normalized_bc_chunks = [
                    bc_loaded.policy.predict_full_action_horizon(bc_batch)[:, start:end]
                    for _ in range(_EXPECTED_BC_REPEAT_CALLS)
                ]
                bc_action_chunks = [
                    _postprocess_chunk(bc_loaded.postprocessor, chunk)
                    for chunk in normalized_bc_chunks
                ]
            repeat_errors = [
                float(torch.max(torch.abs(normalized_bc_chunks[0] - repeated)).detach().cpu())
                for repeated in normalized_bc_chunks[1:]
            ]
            bc_repeat_errors.extend(repeat_errors)
            stacked_bc_actions = np.stack(bc_action_chunks)
            if stacked_bc_actions.shape != (
                _EXPECTED_BC_REPEAT_CALLS,
                bc_loaded.config.n_action_steps,
                2,
            ):
                raise ValueError(
                    "BCChunk predictions must have shape "
                    f"[{_EXPECTED_BC_REPEAT_CALLS},{bc_loaded.config.n_action_steps},2], "
                    f"got {stacked_bc_actions.shape}"
                )
            if not np.isfinite(stacked_bc_actions).all():
                raise ValueError(
                    f"BCChunk predictions contain non-finite values for {candidate_id}"
                )
            for sample_ordinal, (normalized_chunk, action_chunk) in enumerate(
                zip(normalized_bc_chunks, bc_action_chunks, strict=True)
            ):
                for step, (normalized_action, action) in enumerate(
                    zip(
                        normalized_chunk[0].detach().cpu().numpy(),
                        action_chunk,
                        strict=True,
                    )
                ):
                    action_rows.append(
                        {
                            "schema_version": _E1_SCHEMA_VERSION,
                            "experiment_id": "E1",
                            "condition_id": "bc_chunk",
                            "sampling_protocol": _BC_DETERMINISTIC,
                            "candidate_id": candidate_id,
                            "episode_index": episode_index,
                            "frame_index": frame_index,
                            "model": "bc_chunk",
                            "checkpoint_id": checkpoint_ids["bc_chunk"],
                            "train_seed": train_seed,
                            "environment_seed": None,
                            "policy_seed": None,
                            "replan_index": None,
                            "sample_ordinal": sample_ordinal,
                            "step": step,
                            "normalized_action_x": float(normalized_action[0]),
                            "normalized_action_y": float(normalized_action[1]),
                            "action_x": float(action[0]),
                            "action_y": float(action[1]),
                            "initial_noise_seed": None,
                            "initial_noise_sha256": None,
                            "initial_noise_file_sha256": None,
                            "reverse_noise_seed": None,
                            "reverse_generator_state_sha256": None,
                            "num_inference_steps": None,
                            "num_action_samples": _EXPECTED_BC_REPEAT_CALLS,
                            "num_rollouts": 0,
                            "dataset_revision": DATASET_REVISION,
                            "code_commit": code_commit,
                        }
                    )

            explicit_noises: list[dict[str, Any]] = []
            for sample_ordinal in range(num_action_samples):
                seed = fixed_observation_initial_noise_seed(
                    episode_index=episode_index,
                    frame_index=frame_index,
                    sample_ordinal=sample_ordinal,
                )
                explicit = make_initial_noise(
                    seed=seed,
                    batch_size=1,
                    device=get_device_from_parameters(diffusion_loaded.policy),
                    dtype=get_dtype_from_parameters(diffusion_loaded.policy),
                    horizon=diffusion_loaded.config.horizon,
                    action_dim=diffusion_loaded.config.action_feature.shape[0],
                )
                noise_path = noise_dir / f"{candidate_id}_sample_{sample_ordinal:02d}.npy"
                np.save(noise_path, explicit.tensor.detach().cpu().numpy())
                file_sha256 = sha256_file(noise_path)
                reloaded_tensor = torch.from_numpy(np.load(noise_path, allow_pickle=False))
                reloaded_sha256 = sha256_tensor(reloaded_tensor)
                if reloaded_sha256 != explicit.tensor_sha256:
                    raise RuntimeError(
                        f"Candidate {candidate_id} sample {sample_ordinal} saved initial-noise "
                        "tensor hash does not match the canonical tensor"
                    )
                explicit_noises.append(
                    {
                        "sample_ordinal": sample_ordinal,
                        "seed": seed,
                        "explicit": explicit,
                        "tensor_sha256": explicit.tensor_sha256,
                        "file_sha256": file_sha256,
                    }
                )

            protocol_specs = (
                (
                    _INITIAL_NOISE_ONLY,
                    "diffusion_ddpm100_initial_noise_only",
                    [
                        fixed_observation_e1a_reverse_seed(
                            episode_index=episode_index,
                            frame_index=frame_index,
                        )
                    ]
                    * num_action_samples,
                ),
                (
                    _FULL_SAMPLING_SEED,
                    "diffusion_ddpm100_full_sampling_seed",
                    [
                        fixed_observation_e1b_reverse_seed(
                            episode_index=episode_index,
                            frame_index=frame_index,
                            sample_ordinal=sample_ordinal,
                        )
                        for sample_ordinal in range(num_action_samples)
                    ],
                ),
            )
            for sampling_protocol, condition_id, reverse_seeds in protocol_specs:
                diffusion_actions: list[np.ndarray] = []
                normalized_diffusion_actions: list[np.ndarray] = []
                repeat_max_error: float | None = None
                for noise_record, reverse_seed in zip(
                    explicit_noises,
                    reverse_seeds,
                    strict=True,
                ):
                    sample_ordinal = int(noise_record["sample_ordinal"])
                    seed = int(noise_record["seed"])
                    explicit = noise_record["explicit"]
                    canonical_sha256 = str(noise_record["tensor_sha256"])
                    with torch.inference_mode():
                        sampled = controlled_sample(
                            diffusion_loaded.policy,
                            diffusion_batch,
                            initial_noise=explicit.tensor.clone(),
                            reverse_seed=reverse_seed,
                        )
                    if sha256_tensor(explicit.tensor) != canonical_sha256:
                        raise RuntimeError(
                            f"Candidate {candidate_id} protocol {sampling_protocol} sample "
                            f"{sample_ordinal} mutated the canonical initial-noise tensor"
                        )
                    denormalized = _postprocess_chunk(
                        diffusion_loaded.postprocessor,
                        sampled.actions,
                    )
                    normalized = sampled.actions[0].detach().cpu().numpy()
                    diffusion_actions.append(denormalized)
                    normalized_diffusion_actions.append(normalized)
                    if sample_ordinal == 0:
                        with torch.inference_mode():
                            repeated = controlled_sample(
                                diffusion_loaded.policy,
                                diffusion_batch,
                                initial_noise=explicit.tensor.clone(),
                                reverse_seed=reverse_seed,
                            )
                        repeat_max_error = float(
                            torch.max(torch.abs(sampled.actions - repeated.actions)).cpu()
                        )
                        if sha256_tensor(explicit.tensor) != canonical_sha256:
                            raise RuntimeError(
                                f"Candidate {candidate_id} protocol {sampling_protocol} "
                                "reproducibility call mutated canonical initial noise"
                            )
                    for step, (normalized_action, action) in enumerate(
                        zip(normalized, denormalized, strict=True)
                    ):
                        action_rows.append(
                            {
                                "schema_version": _E1_SCHEMA_VERSION,
                                "experiment_id": "E1",
                                "condition_id": condition_id,
                                "sampling_protocol": sampling_protocol,
                                "candidate_id": candidate_id,
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "model": "diffusion",
                                "checkpoint_id": checkpoint_ids["diffusion"],
                                "train_seed": train_seed,
                                "environment_seed": None,
                                "policy_seed": None,
                                "replan_index": 0,
                                "sample_ordinal": sample_ordinal,
                                "step": step,
                                "normalized_action_x": float(normalized_action[0]),
                                "normalized_action_y": float(normalized_action[1]),
                                "action_x": float(action[0]),
                                "action_y": float(action[1]),
                                "initial_noise_seed": seed,
                                "initial_noise_sha256": canonical_sha256,
                                "initial_noise_file_sha256": noise_record["file_sha256"],
                                "reverse_noise_seed": reverse_seed,
                                "reverse_generator_state_sha256": (
                                    sampled.reverse_generator_state_sha256
                                ),
                                "num_inference_steps": sampled.num_inference_steps,
                                "num_action_samples": num_action_samples,
                                "num_rollouts": 0,
                                "dataset_revision": DATASET_REVISION,
                                "code_commit": code_commit,
                            }
                        )

                if repeat_max_error is None:
                    raise RuntimeError(
                        f"Candidate {candidate_id} protocol {sampling_protocol} did not "
                        "run its reproducibility check"
                    )
                stacked_actions = np.stack(diffusion_actions)
                stacked_normalized_actions = np.stack(normalized_diffusion_actions)
                expected_diffusion_shape = (
                    num_action_samples,
                    diffusion_loaded.config.n_action_steps,
                    2,
                )
                if stacked_actions.shape != expected_diffusion_shape:
                    raise ValueError(
                        f"Candidate {candidate_id} protocol {sampling_protocol} predictions "
                        f"must have shape {expected_diffusion_shape}, got "
                        f"{stacked_actions.shape}"
                    )
                if not np.isfinite(stacked_actions).all():
                    raise ValueError(
                        f"Candidate {candidate_id} protocol {sampling_protocol} contains "
                        "non-finite Diffusion predictions"
                    )
                if not np.isfinite(stacked_normalized_actions).all():
                    raise ValueError(
                        f"Candidate {candidate_id} protocol {sampling_protocol} contains "
                        "non-finite normalized Diffusion predictions"
                    )
                agent_position = np.asarray(item[OBS_STATE])[-1].reshape(-1)
                normalized_agent_position = (
                    diffusion_batch[OBS_STATE][0, -1].detach().cpu().numpy().reshape(-1)
                )
                denormalized_summary = summarize_action_diversity(
                    stacked_actions,
                    agent_position=agent_position,
                )
                normalized_summary = summarize_action_diversity(
                    stacked_normalized_actions,
                    agent_position=normalized_agent_position,
                    lower_bound=-1.0,
                    upper_bound=1.0,
                )
                metric_row = {
                    "schema_version": _E1_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "condition_id": condition_id,
                    "sampling_protocol": sampling_protocol,
                    **{
                        f"denormalized_{summary_key}": value
                        for summary_key, value in denormalized_summary.to_dict().items()
                    },
                    **{
                        f"normalized_{summary_key}": value
                        for summary_key, value in normalized_summary.to_dict().items()
                    },
                    "repeat_max_error": repeat_max_error,
                    "input_tensor_hashes_json": json.dumps(input_hashes, sort_keys=True),
                }
                metric_rows.append(metric_row)
                camera_key = next(iter(diffusion_loaded.config.image_features))
                raw_history = np.asarray(item[camera_key])
                visualization_candidates[sampling_protocol].append(
                    {
                        "candidate_id": candidate_id,
                        "previous_image": image_to_hwc_uint8(raw_history[0]),
                        "current_image": image_to_hwc_uint8(raw_history[-1]),
                        "candidate_ordinal": ordinal,
                        "agent_position": agent_position,
                        "bc_actions": stacked_bc_actions,
                        "diffusion_actions": stacked_actions,
                        "metrics": metric_row,
                    }
                )

            for noise_record in explicit_noises:
                explicit = noise_record["explicit"]
                if sha256_tensor(explicit.tensor) != noise_record["tensor_sha256"]:
                    raise RuntimeError(
                        f"Candidate {candidate_id} sample {noise_record['sample_ordinal']} "
                        "canonical initial noise changed after paired sampling"
                    )

        _validate_e1_rows(
            action_rows,
            metric_rows,
            candidates=candidates,
            num_action_samples=num_action_samples,
            n_action_steps=diffusion_loaded.config.n_action_steps,
            label=label,
        )
        expected_noise_files = len(candidates) * num_action_samples
        actual_noise_files = len(list(noise_dir.glob("*.npy")))
        if actual_noise_files != expected_noise_files:
            raise RuntimeError(
                f"E1 wrote {actual_noise_files} initial-noise files; "
                f"expected {expected_noise_files}"
            )
        write_records_parquet(temporary / "action_chunks.parquet", action_rows)
        write_records_parquet(temporary / "diversity_metrics.parquet", metric_rows)
        aggregate_metrics = _aggregate_e1_metrics(
            metric_rows,
            num_candidates=len(candidates),
            num_action_samples=num_action_samples,
            bc_repeat_max_error=max(bc_repeat_errors),
        )
        tolerance = float(protocol["same_noise_tolerance"])
        if aggregate_metrics["bc_repeat_max_error"] > tolerance:
            raise RuntimeError(f"BCChunk repeated-call determinism exceeded {tolerance}")
        for sampling_protocol in _DIFFUSION_PROTOCOLS:
            repeat_key = f"{sampling_protocol}_repeat_max_error"
            if aggregate_metrics[repeat_key] > tolerance:
                raise RuntimeError(
                    f"Diffusion protocol {sampling_protocol} reproducibility exceeded "
                    f"{tolerance}"
                )
        metric_comments = {
            "schema_version": "E1 paired-sampling artifact schema",
            "num_candidates": "Held-out frozen observation histories",
            "num_action_samples_per_protocol": (
                "Paired Diffusion samples for E1a and E1b per observation"
            ),
            "bc_repeat_max_error": "Deterministic BC repeated-call error",
            "num_rollouts": "E1 does not execute environment actions",
        }
        for sampling_protocol in _DIFFUSION_PROTOCOLS:
            label_name = (
                "E1a initial-noise-only"
                if sampling_protocol == _INITIAL_NOISE_ONLY
                else "E1b full-sampling-seed"
            )
            metric_comments.update(
                {
                    f"{sampling_protocol}_mean_normalized_trajectory_pairwise_rms": (
                        f"{label_name} mean trajectory diversity in normalized action space"
                    ),
                    f"{sampling_protocol}_mean_normalized_endpoint_pairwise_distance": (
                        f"{label_name} mean action[7] distance in normalized action space"
                    ),
                    f"{sampling_protocol}_mean_denormalized_trajectory_pairwise_rms": (
                        f"{label_name} mean trajectory diversity in 512-pixel coordinates"
                    ),
                    f"{sampling_protocol}_mean_denormalized_endpoint_pairwise_distance": (
                        f"{label_name} mean action[7] distance in 512-pixel coordinates"
                    ),
                    f"{sampling_protocol}_repeat_max_error": (
                        f"{label_name} identical-input reproducibility error"
                    ),
                }
            )
        for field in (
            "normalized_trajectory_pairwise_rms",
            "normalized_endpoint_pairwise_distance",
            "denormalized_trajectory_pairwise_rms",
            "denormalized_endpoint_pairwise_distance",
        ):
            metric_comments[
                f"full_sampling_seed_to_initial_noise_only_{field}_ratio"
            ] = "Aggregate E1b mean divided by aggregate E1a mean"
        write_metrics(temporary, aggregate_metrics, comments=metric_comments)
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": _E1_SCHEMA_VERSION,
                "experiment_schema": _E1_EXPERIMENT_SCHEMA,
                "label": label,
                "experiment": "E1",
                "dataset_repo_id": DATASET_REPO_ID,
                "dataset_revision": DATASET_REVISION,
                "candidate_source_sha256": sha256_file(candidates_path),
                "processor_metadata_sha256": processor_hashes,
                "checkpoint_ids": checkpoint_ids,
                "bc_checkpoint": str(Path(bc_policy_path)),
                "diffusion_checkpoint": str(Path(diffusion_policy_path)),
                "scheduler": protocol["scheduler"],
                "num_inference_steps": protocol["num_inference_steps"],
                "num_action_samples_per_protocol": num_action_samples,
                "num_rollouts": 0,
                "train_seed": train_seed,
                "sampling_protocols": list(_DIFFUSION_PROTOCOLS),
                "recordings": dict(_RECORDING_FILENAMES),
                "seed_derivation": {
                    "version": "numpy_seedsequence_uint32_v1",
                    "protocol_namespace": PROTOCOL_NAMESPACE,
                    "initial_noise_stream": INITIAL_NOISE_STREAM,
                    "e1a_reverse_stream": E1A_REVERSE_NOISE_STREAM,
                    "e1b_reverse_stream": E1B_REVERSE_NOISE_STREAM,
                    "candidate_key": ["episode_index", "frame_index"],
                },
                "candidate_keys": [
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "episode_index": int(candidate["episode_index"]),
                        "frame_index": int(candidate["frame_index"]),
                    }
                    for candidate in candidates
                ],
                "runtime": runtime,
            },
        )
        for sampling_protocol, filename in _RECORDING_FILENAMES.items():
            recording_path = record_fixed_observation_samples(
                output_path=temporary / filename,
                candidates=visualization_candidates[sampling_protocol],
                sampling_protocol=sampling_protocol,
            )
            if not recording_path.is_file():
                raise RuntimeError(
                    f"E1 protocol {sampling_protocol} did not write recording {recording_path}"
                )
        if (temporary / "recording.rrd").exists():
            raise RuntimeError(
                "Legacy recording.rrd must not be created for E1 schema version 2"
            )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-policy", required=True)
    parser.add_argument("--diffusion-policy", required=True)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=int(_E1_CONFIG["num_action_samples"]),
    )
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e1_fixed_observation",
    )
    parser.add_argument("--label", choices=("Smoke", "Offline"), default="Offline")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = evaluate_fixed_observations(
        bc_policy_path=args.bc_policy,
        diffusion_policy_path=args.diffusion_policy,
        candidates_path=args.candidates.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        num_action_samples=args.num_action_samples,
        train_seed=args.train_seed,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(f"E1 output: {result}")


if __name__ == "__main__":
    main()
