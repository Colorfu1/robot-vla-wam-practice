"""E5 fixed-observation counterfactual prototype-ID evaluation."""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_STATE

from robot_practice.policies.mode_bc_chunk.modeling_mode_bc_chunk import ModeBCChunkPolicy
from robot_practice.policies.mode_diffusion.modeling_mode_diffusion import ModeDiffusionPolicy

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    default_dataset_root,
    default_output_root,
    load_named_config,
)
from .e5_dataset import load_prototype_sidecar
from .e5_runtime import add_prototype_id, move_processed_batch
from .fixed_observation_eval import (
    _index_frames,
    _load_candidates,
    _observation_only,
    _order_and_validate_candidates,
    _postprocess_chunk,
    _stack_images,
)
from .noise import (
    fixed_observation_e1b_reverse_seed,
    fixed_observation_initial_noise_seed,
    make_initial_noise,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import controlled_sample, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rerun_visualization import image_to_hwc_uint8, record_fixed_observation_samples

_CONFIG = load_named_config("e5_mode_conditioned.json")
_SCHEMA_VERSION = 1
_EXPERIMENT_SCHEMA = str(_CONFIG["experiment_schema"])


def _rms(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(first) - np.asarray(second)))))


def _mean_pairwise(chunks: Sequence[np.ndarray]) -> float | None:
    if len(chunks) < 2:
        return None
    return float(
        np.mean(
            [
                _rms(chunks[left], chunks[right])
                for left in range(len(chunks))
                for right in range(left + 1, len(chunks))
            ]
        )
    )


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _validate_loaded_pair(bc_loaded: Any, diffusion_loaded: Any, *, selected_k: int) -> None:
    if not isinstance(bc_loaded.policy, ModeBCChunkPolicy):
        raise TypeError("bc-policy must contain policy.type=mode_bc_chunk")
    if not isinstance(diffusion_loaded.policy, ModeDiffusionPolicy):
        raise TypeError("diffusion-policy must contain policy.type=mode_diffusion")
    for loaded in (bc_loaded, diffusion_loaded):
        if int(loaded.config.num_prototypes) != selected_k:
            raise ValueError(
                f"Checkpoint K={loaded.config.num_prototypes} disagrees with sidecar K={selected_k}"
            )
    for field in ("n_obs_steps", "horizon", "n_action_steps", "mode_embedding_dim"):
        if getattr(bc_loaded.config, field) != getattr(diffusion_loaded.config, field):
            raise ValueError(f"Mode-BCChunk and Mode-Diffusion disagree on {field}")


def evaluate_e5_fixed_observations(
    *,
    bc_policy_path: str | Path,
    diffusion_policy_path: str | Path,
    prototype_manifest: str | Path,
    candidates_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    num_policy_seeds: int,
    num_inference_steps: int,
    label: str,
    overwrite: bool,
) -> Path:
    """Hold observation and complete sampling RNG fixed while changing only the ID."""

    formal_seeds = tuple(int(value) for value in _CONFIG["fixed_observation_policy_seeds"])
    if label not in {"Smoke", "Offline"}:
        raise ValueError("label must be Smoke or Offline")
    if not 1 <= num_policy_seeds <= len(formal_seeds):
        raise ValueError(f"num_policy_seeds must be in [1,{len(formal_seeds)}]")
    if label == "Offline" and num_policy_seeds != len(formal_seeds):
        raise ValueError(
            f"Offline E5 fixed-observation evaluation requires {len(formal_seeds)} seeds"
        )
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if label == "Offline" and num_inference_steps != int(_CONFIG["num_inference_steps"]):
        raise ValueError(f"Offline E5 requires {_CONFIG['num_inference_steps']} inference steps")
    policy_seeds = formal_seeds[:num_policy_seeds]
    sidecar = load_prototype_sidecar(prototype_manifest)
    candidates = _order_and_validate_candidates(
        _load_candidates(Path(candidates_path).expanduser().resolve())
    )

    bc_loaded = load_policy_with_processors(
        bc_policy_path,
        local_files_only=True,
    )
    diffusion_loaded = load_policy_with_processors(
        diffusion_policy_path,
        local_files_only=True,
    )
    _validate_loaded_pair(bc_loaded, diffusion_loaded, selected_k=sidecar.selected_k)
    replace_scheduler(
        diffusion_loaded.policy,
        scheduler_name=str(_CONFIG["scheduler"]),
        num_inference_steps=num_inference_steps,
    )

    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=Path(dataset_root).expanduser().resolve(),
        revision=DATASET_REVISION,
    )
    delta_timestamps = resolve_delta_timestamps(diffusion_loaded.config, metadata)
    episodes = sorted({int(candidate["episode_index"]) for candidate in candidates})
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=Path(dataset_root).expanduser().resolve(),
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=DATASET_REVISION,
    )
    frame_lookup = _index_frames(dataset)
    condition_ids = tuple(range(sidecar.selected_k + 1))
    condition_names = {
        value: ("NULL" if value == sidecar.null_prototype_id else f"P{value}")
        for value in condition_ids
    }
    action_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    visualization: dict[int, list[dict[str, Any]]] = {value: [] for value in condition_ids}

    destination = Path(output_dir).expanduser().resolve()
    with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            episode_index = int(candidate["episode_index"])
            frame_index = int(candidate["frame_index"])
            item = dataset[frame_lookup[(episode_index, frame_index)]]
            bc_base = _stack_images(
                bc_loaded.policy,
                move_processed_batch(
                    bc_loaded.preprocessor(_observation_only(item, bc_loaded.policy)),
                    device=get_device_from_parameters(bc_loaded.policy),
                ),
            )
            diffusion_base = _stack_images(
                diffusion_loaded.policy,
                move_processed_batch(
                    diffusion_loaded.preprocessor(_observation_only(item, diffusion_loaded.policy)),
                    device=get_device_from_parameters(diffusion_loaded.policy),
                ),
            )
            for key in set(bc_base) & set(diffusion_base):
                if isinstance(bc_base[key], torch.Tensor) and not torch.allclose(
                    bc_base[key],
                    diffusion_base[key],
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise RuntimeError(f"Processed Mode policy inputs differ for {key}")

            bc_chunks: dict[int, np.ndarray] = {}
            diffusion_chunks: dict[int, list[np.ndarray]] = {value: [] for value in condition_ids}
            normalized_diffusion_chunks: dict[int, list[np.ndarray]] = {
                value: [] for value in condition_ids
            }
            with torch.inference_mode():
                start = int(bc_loaded.config.n_obs_steps) - 1
                end = start + int(bc_loaded.config.n_action_steps)
                for condition_id in condition_ids:
                    conditioned = add_prototype_id(
                        bc_base,
                        prototype_id=condition_id,
                        num_prototypes=sidecar.selected_k,
                    )
                    normalized = bc_loaded.policy.predict_full_action_horizon(conditioned)[
                        :, start:end
                    ]
                    bc_chunks[condition_id] = _postprocess_chunk(
                        bc_loaded.postprocessor,
                        normalized,
                    )
                    for step, (normalized_action, action) in enumerate(
                        zip(
                            normalized[0].detach().cpu().numpy(),
                            bc_chunks[condition_id],
                            strict=True,
                        )
                    ):
                        action_rows.append(
                            {
                                "schema_version": _SCHEMA_VERSION,
                                "candidate_id": candidate_id,
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "model": "mode_bc_chunk",
                                "condition_id": condition_id,
                                "condition_name": condition_names[condition_id],
                                "policy_seed": None,
                                "step": step,
                                "normalized_action_x": float(normalized_action[0]),
                                "normalized_action_y": float(normalized_action[1]),
                                "action_x": float(action[0]),
                                "action_y": float(action[1]),
                                "initial_noise_seed": None,
                                "reverse_noise_seed": None,
                            }
                        )

            for policy_seed in policy_seeds:
                initial_seed = fixed_observation_initial_noise_seed(
                    episode_index=episode_index,
                    frame_index=frame_index,
                    sample_ordinal=policy_seed,
                )
                reverse_seed = fixed_observation_e1b_reverse_seed(
                    episode_index=episode_index,
                    frame_index=frame_index,
                    sample_ordinal=policy_seed,
                )
                explicit = make_initial_noise(
                    seed=initial_seed,
                    batch_size=1,
                    device=get_device_from_parameters(diffusion_loaded.policy),
                    dtype=get_dtype_from_parameters(diffusion_loaded.policy),
                    horizon=diffusion_loaded.config.horizon,
                    action_dim=diffusion_loaded.config.action_feature.shape[0],
                )
                for condition_id in condition_ids:
                    conditioned = add_prototype_id(
                        diffusion_base,
                        prototype_id=condition_id,
                        num_prototypes=sidecar.selected_k,
                    )
                    with torch.inference_mode():
                        sampled = controlled_sample(
                            diffusion_loaded.policy,
                            conditioned,
                            initial_noise=explicit.tensor.clone(),
                            reverse_seed=reverse_seed,
                        )
                    normalized = sampled.actions[0].detach().cpu().numpy()
                    action = _postprocess_chunk(
                        diffusion_loaded.postprocessor,
                        sampled.actions,
                    )
                    normalized_diffusion_chunks[condition_id].append(normalized)
                    diffusion_chunks[condition_id].append(action)
                    for step, (normalized_action, action_target) in enumerate(
                        zip(normalized, action, strict=True)
                    ):
                        action_rows.append(
                            {
                                "schema_version": _SCHEMA_VERSION,
                                "candidate_id": candidate_id,
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "model": "mode_diffusion",
                                "condition_id": condition_id,
                                "condition_name": condition_names[condition_id],
                                "policy_seed": policy_seed,
                                "step": step,
                                "normalized_action_x": float(normalized_action[0]),
                                "normalized_action_y": float(normalized_action[1]),
                                "action_x": float(action_target[0]),
                                "action_y": float(action_target[1]),
                                "initial_noise_seed": initial_seed,
                                "initial_noise_sha256": explicit.tensor_sha256,
                                "reverse_noise_seed": reverse_seed,
                                "reverse_generator_state_sha256": (
                                    sampled.reverse_generator_state_sha256
                                ),
                            }
                        )

            for condition_id in condition_ids:
                metric_rows.append(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "candidate_id": candidate_id,
                        "metric": "within_condition_pairwise_rms",
                        "model": "mode_diffusion",
                        "condition_a": condition_id,
                        "condition_b": condition_id,
                        "value": _mean_pairwise(diffusion_chunks[condition_id]),
                    }
                )
                camera_key = next(iter(diffusion_loaded.config.image_features))
                image_history = np.asarray(item[camera_key])
                visualization[condition_id].append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_ordinal": int(candidate["candidate_ordinal"]),
                        "previous_image": image_to_hwc_uint8(image_history[0]),
                        "current_image": image_to_hwc_uint8(image_history[-1]),
                        "agent_position": np.asarray(item[OBS_STATE])[-1].reshape(-1),
                        "bc_actions": np.repeat(
                            bc_chunks[condition_id][None, ...],
                            repeats=3,
                            axis=0,
                        ),
                        "diffusion_actions": np.stack(diffusion_chunks[condition_id]),
                        "metrics": {
                            "prototype_condition": condition_names[condition_id],
                        },
                    }
                )

            for first, second in itertools.combinations(range(sidecar.selected_k), 2):
                paired = [
                    _rms(
                        diffusion_chunks[first][seed_index],
                        diffusion_chunks[second][seed_index],
                    )
                    for seed_index in range(len(policy_seeds))
                ]
                metric_rows.extend(
                    [
                        {
                            "schema_version": _SCHEMA_VERSION,
                            "candidate_id": candidate_id,
                            "metric": "paired_between_condition_rms",
                            "model": "mode_diffusion",
                            "condition_a": first,
                            "condition_b": second,
                            "value": float(np.mean(paired)),
                        },
                        {
                            "schema_version": _SCHEMA_VERSION,
                            "candidate_id": candidate_id,
                            "metric": "between_condition_rms",
                            "model": "mode_bc_chunk",
                            "condition_a": first,
                            "condition_b": second,
                            "value": _rms(bc_chunks[first], bc_chunks[second]),
                        },
                    ]
                )

        write_records_parquet(temporary / "actions.parquet", action_rows)
        write_records_parquet(temporary / "condition_metrics.parquet", metric_rows)
        recordings = temporary / "recordings"
        for condition_id in condition_ids:
            record_fixed_observation_samples(
                output_path=recordings / f"condition_{condition_names[condition_id]}.rrd",
                candidates=visualization[condition_id],
                sampling_protocol="full_sampling_seed",
                recording_id=f"e5_fixed_{condition_names[condition_id].lower()}",
                recording_name=(
                    "E5 fixed observation | "
                    f"{condition_names[condition_id]} | {len(policy_seeds)} seeds"
                ),
            )
        between_dp = [
            float(row["value"])
            for row in metric_rows
            if row["model"] == "mode_diffusion" and row["metric"] == "paired_between_condition_rms"
        ]
        within_dp = [
            float(row["value"])
            for row in metric_rows
            if row["model"] == "mode_diffusion"
            and row["metric"] == "within_condition_pairwise_rms"
            and int(row["condition_a"]) < sidecar.selected_k
            and row["value"] is not None
        ]
        summary = {
            "num_candidates": len(candidates),
            "num_policy_seeds": len(policy_seeds),
            "num_prototypes": sidecar.selected_k,
            "mean_paired_between_prototype_rms": _mean_or_none(between_dp),
            "mean_within_prototype_pairwise_rms": _mean_or_none(within_dp),
            "between_to_within_ratio": (
                float(np.mean(between_dp) / np.mean(within_dp))
                if between_dp and within_dp and np.mean(within_dp) > 0.0
                else None
            ),
        }
        write_metrics(temporary, summary)
        write_json_atomic(
            temporary / "manifest.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "experiment_schema": _EXPERIMENT_SCHEMA,
                "experiment": "E5 fixed-observation condition swap",
                "status": "complete",
                "label": label,
                "prototype_manifest_sha256": sha256_file(sidecar.manifest_path),
                "candidate_source_sha256": sha256_file(
                    Path(candidates_path).expanduser().resolve()
                ),
                "checkpoint_sha256": {
                    "mode_bc_chunk": sha256_file(
                        Path(bc_policy_path).expanduser().resolve() / "model.safetensors"
                    ),
                    "mode_diffusion": sha256_file(
                        Path(diffusion_policy_path).expanduser().resolve() / "model.safetensors"
                    ),
                },
                "policy_seeds": list(policy_seeds),
                "scheduler": str(_CONFIG["scheduler"]),
                "num_inference_steps": num_inference_steps,
                "condition_ids": list(condition_ids),
                "runtime": collect_runtime_provenance(Path(__file__).resolve().parents[4]),
            },
        )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-policy", type=Path, required=True)
    parser.add_argument("--diffusion-policy", type=Path, required=True)
    parser.add_argument("--prototype-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=default_output_root() / "data_inspection" / "candidates.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e5_mode_conditioned" / "fixed_observation",
    )
    parser.add_argument("--num-policy-seeds", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--label", choices=("Smoke", "Offline"), default="Offline")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output = evaluate_e5_fixed_observations(
        bc_policy_path=args.bc_policy,
        diffusion_policy_path=args.diffusion_policy,
        prototype_manifest=args.prototype_manifest,
        candidates_path=args.candidates,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_policy_seeds=args.num_policy_seeds,
        num_inference_steps=args.num_inference_steps,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(f"E5 fixed-observation output: {output}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_e5_fixed_observations"]
