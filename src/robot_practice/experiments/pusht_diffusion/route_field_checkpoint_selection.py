"""Select RouteField-BCChunk and RouteField-Diffusion checkpoints on development data."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION
from tqdm import tqdm

from robot_practice.policies.route_field_bc_chunk.modeling_route_field_bc_chunk import (  # noqa: F401,E501
    RouteFieldBCChunkPolicy,
)
from robot_practice.policies.route_field_diffusion.modeling_route_field_diffusion import (  # noqa: F401,E501
    RouteFieldDiffusionPolicy,
)

from .checkpoint_selection import (
    CheckpointEventCallback,
    CheckpointTask,
    WorkerConfig,
    _batched_inputs,
    _checkpoint_summary,
    _checkpoint_tasks,
    _evaluate_parallel,
    _frame_lookup,
    _record_selection,
    _select,
    _stack_images,
    _validate_num_gpus,
    discover_checkpoints,
)
from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_dataset_root,
    default_output_root,
    practice_root,
)
from .dataset_split import load_split
from .e5_checkpoint_selection import EPISODE_LENGTH, NUM_INFERENCE_STEPS, POLICY_SEED
from .e5_oracle_mode import preflight_reconstruction
from .noise import fixed_observation_noise_seed, fixed_observation_reverse_seed, make_initial_noise
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import controlled_sample, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .route_field_dataset import load_route_field_sidecar
from .route_field_episodes import (
    RouteFieldEpisode,
    achieved_route_metrics,
    load_route_field_episodes,
    run_route_field_rollout,
)

MODEL_BC = "route_field_bc_chunk"
MODEL_DIFFUSION = "route_field_diffusion"
ORACLE_CONDITION_LABEL = "oracle"


def validate_route_field_policy(policy: Any, *, expected_model: str) -> None:
    expected_type: type[Any]
    if expected_model == MODEL_BC:
        expected_type = RouteFieldBCChunkPolicy
    elif expected_model == MODEL_DIFFUSION:
        expected_type = RouteFieldDiffusionPolicy
    else:
        raise ValueError(f"Unsupported RF7 model: {expected_model}")
    if not isinstance(policy, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__} for {expected_model}, "
            f"got {type(policy).__name__}"
        )
    if int(policy.config.route_grid_size) != 7:
        raise ValueError("RF7 checkpoint must use route_grid_size=7")
    if int(policy.config.route_field_channels) != 4:
        raise ValueError("RF7 checkpoint must use route_field_channels=4")


def _route_field_offline_error(
    loaded: Any,
    *,
    dataset_root: Path,
    episodes: Sequence[RouteFieldEpisode],
) -> tuple[float, float]:
    metadata = LeRobotDatasetMetadata(
        DATASET_REPO_ID,
        root=dataset_root,
        revision=DATASET_REVISION,
    )
    delta_timestamps = resolve_delta_timestamps(loaded.config, metadata)
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        episodes=[episode.episode_index for episode in episodes],
        delta_timestamps=delta_timestamps,
        revision=DATASET_REVISION,
    )
    lookup = _frame_lookup(dataset)
    device = get_device_from_parameters(loaded.policy)
    normalized_errors = []
    denormalized_errors = []
    for ordinal, episode in enumerate(episodes):
        indices = lookup[episode.episode_index]
        item = dataset[indices[len(indices) // 2]]
        raw_batch = _batched_inputs(item, loaded.policy)
        raw_batch[ACTION] = item[ACTION].unsqueeze(0)
        from .route_field_runtime import FixedRouteFieldPreprocessor

        preprocessor = FixedRouteFieldPreprocessor(
            loaded.preprocessor,
            field=episode.field,
            device=device,
        )
        processed = preprocessor(raw_batch)
        prepared = _stack_images(loaded.policy, processed)
        start = loaded.config.n_obs_steps - 1
        end = start + loaded.config.n_action_steps
        with torch.inference_mode():
            if isinstance(loaded.policy, RouteFieldBCChunkPolicy):
                normalized_prediction = loaded.policy.predict_full_action_horizon(prepared)[
                    :, start:end
                ]
            elif isinstance(loaded.policy, DiffusionPolicy):
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
                normalized_prediction = controlled_sample(
                    loaded.policy,
                    prepared,
                    initial_noise=explicit.tensor,
                    reverse_seed=fixed_observation_reverse_seed(
                        candidate_ordinal=ordinal
                    ),
                ).actions
            else:
                raise TypeError(f"Unsupported RF7 policy: {type(loaded.policy).__name__}")

        normalized_target = processed[ACTION][:, start:end]
        normalized_errors.append(
            float(torch.mean((normalized_prediction - normalized_target) ** 2).cpu())
        )
        prediction = loaded.postprocessor(normalized_prediction)
        target = loaded.postprocessor(normalized_target)
        denormalized_errors.append(float(torch.mean((prediction - target) ** 2).cpu()))
    return float(np.mean(normalized_errors)), float(np.mean(denormalized_errors))


def _rollout_row(
    *,
    result: Any,
    episode: RouteFieldEpisode,
    model_name: str,
    step: int,
) -> dict[str, Any]:
    latencies = [float(record.latency_ms) for record in result.replans]
    route_metrics = achieved_route_metrics(
        result.block_path,
        requested_cells=episode.route_cells,
    )
    return {
        "model": model_name,
        "step": step,
        "episode_index": episode.episode_index,
        "split": episode.split,
        "commanded_condition": ORACLE_CONDITION_LABEL,
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
        **route_metrics,
    }


def evaluate_route_field_checkpoint(
    *,
    model_name: str,
    step: int,
    path: Path,
    dataset_root: Path,
    route_episodes: Sequence[RouteFieldEpisode],
    route_manifest: Path,
    train_seed: int,
    code_commit: str | None,
    device: str | None = None,
    event_callback: CheckpointEventCallback | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint; one development episode receives its exact route."""

    if event_callback is not None:
        event_callback("loading", None)
    loaded = load_policy_with_processors(path, local_files_only=True, device=device)
    try:
        validate_route_field_policy(loaded.policy, expected_model=model_name)
        if model_name == MODEL_DIFFUSION:
            replace_scheduler(
                loaded.policy,
                scheduler_name="DDPM",
                num_inference_steps=NUM_INFERENCE_STEPS,
            )

        if event_callback is not None:
            event_callback("offline", None)
        normalized_mse, denormalized_mse = _route_field_offline_error(
            loaded,
            dataset_root=dataset_root,
            episodes=route_episodes,
        )

        rollout_rows = []
        for episode in route_episodes:
            if event_callback is not None:
                event_callback("rollout", episode.episode_index)
            result = run_route_field_rollout(
                loaded,
                episode=episode,
                field=episode.field,
                condition_label=ORACLE_CONDITION_LABEL,
                # Selection only consumes block_path, replans, and reward, so
                # rendering every frame would cost time for nothing.
                record_frames=False,
                episode_length=EPISODE_LENGTH,
                policy_seed=POLICY_SEED,
            )
            rollout_rows.append(
                _rollout_row(
                    result=result,
                    episode=episode,
                    model_name=model_name,
                    step=step,
                )
            )
            if event_callback is not None:
                event_callback("rollout_complete", episode.episode_index)

        successes = [bool(row["success"]) for row in rollout_rows]
        max_rewards = [float(row["max_reward"]) for row in rollout_rows]
        return {
            "model": model_name,
            "step": step,
            "checkpoint_path": str(path),
            "checkpoint_id": sha256_file(path / "model.safetensors"),
            "train_seed": train_seed,
            "development_episode_count": len(route_episodes),
            "offline_normalized_action_mse": normalized_mse,
            "offline_denormalized_action_mse": denormalized_mse,
            "closed_loop_success_rate": float(np.mean(successes)),
            "mean_max_reward": float(np.mean(max_rewards)),
            "num_rollouts": len(rollout_rows),
            "scheduler": "DDPM" if model_name == MODEL_DIFFUSION else None,
            "num_inference_steps": (
                NUM_INFERENCE_STEPS if model_name == MODEL_DIFFUSION else None
            ),
            "num_action_samples": 1,
            "policy_seed": POLICY_SEED,
            "initial_state_source": "dataset_first_frame_reconstruction",
            "condition_source": "exact_episode_macro_route",
            "dataset_revision": DATASET_REVISION,
            "code_commit": code_commit,
            "_rollouts": rollout_rows,
        }
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluate_sequential(
    *,
    tasks: Sequence[CheckpointTask],
    config: WorkerConfig,
    device: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    total_rollouts = len(tasks) * len(config.oracle_episodes)
    with tqdm(
        total=total_rollouts,
        desc="RF7 oracle checkpoint selection",
        unit="rollout",
        dynamic_ncols=True,
    ) as progress:
        for task in tasks:

            def report_event(
                phase: str,
                episode_index: int | None,
                *,
                current_task: CheckpointTask = task,
            ) -> None:
                if phase == "rollout_complete":
                    progress.update(1)
                    return
                values: dict[str, Any] = {
                    "model": current_task.model_name,
                    "checkpoint": f"{current_task.step:06d}",
                    "phase": phase,
                }
                if episode_index is not None:
                    values["episode_index"] = episode_index
                progress.set_postfix(**values, refresh=True)

            row = evaluate_route_field_checkpoint(
                model_name=task.model_name,
                step=task.step,
                path=task.path,
                dataset_root=config.dataset_root,
                route_episodes=config.oracle_episodes,
                route_manifest=config.route_manifest,
                train_seed=config.train_seed,
                code_commit=config.code_commit,
                device=device,
                event_callback=report_event,
            )
            rows.append(row)
            progress.write(_checkpoint_summary(row))
    return rows


def _separate_rollout_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_rows = []
    rollout_rows = []
    for source in rows:
        row = dict(source)
        nested = row.pop("_rollouts", None)
        if not isinstance(nested, list) or not nested:
            raise RuntimeError(
                f"RF7 checkpoint {row.get('model')}:{row.get('step')} returned no rollouts"
            )
        checkpoint_rows.append(row)
        rollout_rows.extend(nested)
    return checkpoint_rows, rollout_rows


def select_route_field_checkpoints(
    *,
    bc_run: Path | None,
    diffusion_run: Path | None,
    route_manifest: Path,
    dataset_root: Path,
    output_dir: Path,
    train_seed: int,
    overwrite: bool,
    num_gpus: int = 1,
    device: str | None = None,
) -> Path:
    """Run strict development selection for the available RF7 model families."""

    _validate_num_gpus(num_gpus)
    if bc_run is None and diffusion_run is None:
        raise ValueError("At least one of bc_run or diffusion_run is required")
    if num_gpus > 1 and device is not None:
        raise ValueError("device cannot be set when num_gpus > 1")
    split = load_split()
    sidecar = load_route_field_sidecar(route_manifest)
    episodes = load_route_field_episodes(
        dataset_root=dataset_root,
        route_manifest=route_manifest,
        episode_indices=split.development,
        split_name="development",
    )
    reconstruction = preflight_reconstruction(episodes)
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]

    checkpoints: dict[str, tuple[tuple[int, Path], ...]] = {}
    if bc_run is not None:
        checkpoints[MODEL_BC] = discover_checkpoints(bc_run)
    if diffusion_run is not None:
        checkpoints[MODEL_DIFFUSION] = discover_checkpoints(diffusion_run)
    tasks = _checkpoint_tasks(
        checkpoints.get(MODEL_BC, ()),
        checkpoints.get(MODEL_DIFFUSION, ()),
        bc_model_name=MODEL_BC,
        diffusion_model_name=MODEL_DIFFUSION,
    )
    config = WorkerConfig(
        dataset_root=dataset_root,
        development_episodes=tuple(split.development),
        # RF7 drives one rollout per development episode rather than per
        # environment seed, so the seed list carries the episode indices and
        # rollouts_per_task keeps the shared progress accounting exact.
        environment_seeds=tuple(episode.episode_index for episode in episodes),
        rollouts_per_task=len(episodes),
        train_seed=train_seed,
        code_commit=code_commit,
        evaluation_protocol="route_field_oracle",
        route_manifest=route_manifest,
        oracle_episodes=episodes,
    )
    if num_gpus == 1:
        raw_rows = _evaluate_sequential(tasks=tasks, config=config, device=device)
    else:
        raw_rows = _evaluate_parallel(tasks=tasks, config=config, num_gpus=num_gpus)
    rows, rollout_rows = _separate_rollout_rows(raw_rows)
    selected = {
        model: _select(rows, model)
        for model in checkpoints
    }

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        write_records_parquet(temporary / "selection.parquet", rows)
        write_records_parquet(temporary / "rollouts.parquet", rollout_rows)
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
                "environment_seed_namespace": "dataset_episode_index",
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "label": "Closed-loop",
                "experiment": "route_field_oracle_development_checkpoint_selection",
                "dataset_revision": DATASET_REVISION,
                "development_episodes": list(split.development),
                "route_manifest": str(route_manifest),
                "route_manifest_sha256": sha256_file(route_manifest),
                "condition_source": "exact_episode_macro_route",
                "route_field_schema_version": sidecar.manifest.get("schema_version"),
                "route_field_grid_size": sidecar.manifest.get("grid_size"),
                "initial_state_source": "dataset_first_frame_reconstruction",
                "selection_order": [
                    "closed_loop_success_rate_desc",
                    "mean_max_reward_desc",
                    "training_step_asc",
                ],
                "execution": {
                    "num_gpus": num_gpus,
                    "device": device,
                    "parallelism": (
                        "checkpoint_process_pool" if num_gpus > 1 else "sequential"
                    ),
                    "task_granularity": "one_model_checkpoint_all_development_episodes",
                },
                "runtime": runtime,
            },
        )
        metrics: dict[str, Any] = {
            "development_episode_count": len(episodes),
            "development_rollout_count": len(rollout_rows),
            "reconstruction_preflight_passed": len(reconstruction),
        }
        for model, row in selected.items():
            metrics[f"selected_{model}_step"] = row["step"]
            metrics[f"selected_{model}_success_rate"] = row["closed_loop_success_rate"]
        comments = {}
        for model in selected:
            comments[f"selected_{model}_step"] = (
                "Success, reward, then earliest-step ordering"
            )
            comments[f"selected_{model}_success_rate"] = (
                "Development exact-route closed-loop success"
            )
        write_metrics(temporary, metrics, comments=comments)
        _record_selection(
            temporary / "recording.rrd",
            rows,
            models=tuple(checkpoints),
        )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", type=Path, default=None)
    parser.add_argument("--diffusion-run", type=Path, default=None)
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "route_field_oracle_checkpoint_selection",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = select_route_field_checkpoints(
        bc_run=(
            None
            if args.bc_run is None
            else args.bc_run.expanduser().resolve()
        ),
        diffusion_run=(
            None
            if args.diffusion_run is None
            else args.diffusion_run.expanduser().resolve()
        ),
        route_manifest=args.route_manifest.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        train_seed=args.train_seed,
        overwrite=args.overwrite,
        num_gpus=args.num_gpus,
        device=args.device,
    )
    print(f"RF7 checkpoint selection output: {result}")


if __name__ == "__main__":
    main()
