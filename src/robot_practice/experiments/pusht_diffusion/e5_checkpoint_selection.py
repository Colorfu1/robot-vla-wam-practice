"""Select E5 Mode-BCChunk and Mode-Diffusion checkpoints with oracle-mode rollouts."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint_selection import (
    CheckpointEventCallback,
    CheckpointTask,
    WorkerConfig,
    _checkpoint_summary,
    _checkpoint_tasks,
    _evaluate_parallel,
    _offline_error,
    _record_selection,
    _select,
    _validate_num_gpus,
    discover_checkpoints,
)
from .config import (
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_dataset_root,
    default_output_root,
    practice_root,
)
from .dataset_split import load_split
from .e5_dataset import load_prototype_sidecar
from .e5_oracle_mode import (
    OracleModeEpisode,
    load_oracle_mode_episodes,
    preflight_reconstruction,
    run_oracle_mode_rollout,
    validate_mode_policy,
)
from .e5_rollout_eval import _load_classifier, classify_rollout
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic

MODE_BC = "mode_bc_chunk"
MODE_DIFFUSION = "mode_diffusion"
EPISODE_LENGTH = 300
POLICY_SEED = 0
NUM_INFERENCE_STEPS = 100


def _rollout_row(
    *,
    result: Any,
    episode: OracleModeEpisode,
    model_name: str,
    step: int,
    classification: dict[str, Any],
) -> dict[str, Any]:
    latencies = [float(record.latency_ms) for record in result.replans]
    return {
        "model": model_name,
        "step": step,
        "episode_index": episode.episode_index,
        "split": episode.split,
        "oracle_prototype_id": episode.prototype_id,
        "oracle_prototype_name": f"P{episode.prototype_id}",
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
        "scheduler": result.scheduler,
        "num_inference_steps": result.num_inference_steps,
        **classification,
    }


def evaluate_e5_checkpoint(
    *,
    model_name: str,
    step: int,
    path: Path,
    dataset_root: Path,
    oracle_episodes: Sequence[OracleModeEpisode],
    prototype_manifest: Path,
    selected_k: int,
    train_seed: int,
    code_commit: str | None,
    device: str | None = None,
    event_callback: CheckpointEventCallback | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint; one development episode receives one oracle mode."""

    if event_callback is not None:
        event_callback("loading", None)
    loaded = load_policy_with_processors(path, local_files_only=True, device=device)
    try:
        validate_mode_policy(loaded.policy, expected_model=model_name, selected_k=selected_k)
        if model_name == MODE_DIFFUSION:
            replace_scheduler(
                loaded.policy,
                scheduler_name="DDPM",
                num_inference_steps=NUM_INFERENCE_STEPS,
            )

        episode_ids = tuple(episode.episode_index for episode in oracle_episodes)
        oracle_ids = {episode.episode_index: episode.prototype_id for episode in oracle_episodes}
        if event_callback is not None:
            event_callback("offline", None)
        normalized_mse, denormalized_mse = _offline_error(
            loaded,
            dataset_root=dataset_root,
            development_episodes=episode_ids,
            oracle_prototype_ids=oracle_ids,
            selected_k=selected_k,
        )

        classifier = _load_classifier(load_prototype_sidecar(prototype_manifest))
        rollout_rows = []
        for episode in oracle_episodes:
            if event_callback is not None:
                event_callback("rollout", episode.episode_index)
            result = run_oracle_mode_rollout(
                loaded,
                episode=episode,
                commanded_condition_id=episode.prototype_id,
                selected_k=selected_k,
                null_prototype_id=selected_k,
                record_frames=True,
                episode_length=EPISODE_LENGTH,
                policy_seed=POLICY_SEED,
            )
            classification = classify_rollout(
                result,
                classifier=classifier,
                commanded_condition=episode.prototype_id,
                null_prototype_id=selected_k,
            )
            rollout_rows.append(
                _rollout_row(
                    result=result,
                    episode=episode,
                    model_name=model_name,
                    step=step,
                    classification=classification,
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
            "development_episode_count": len(oracle_episodes),
            "offline_normalized_action_mse": normalized_mse,
            "offline_denormalized_action_mse": denormalized_mse,
            "closed_loop_success_rate": float(np.mean(successes)),
            "mean_max_reward": float(np.mean(max_rewards)),
            "num_rollouts": len(rollout_rows),
            "scheduler": "DDPM" if model_name == MODE_DIFFUSION else None,
            "num_inference_steps": (NUM_INFERENCE_STEPS if model_name == MODE_DIFFUSION else None),
            "num_action_samples": 1,
            "policy_seed": POLICY_SEED,
            "initial_state_source": "dataset_first_frame_reconstruction",
            "mode_source": "full_demonstration_nearest_train_medoid",
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
) -> list[dict[str, Any]]:
    rows = []
    total_rollouts = len(tasks) * len(config.oracle_episodes)
    with tqdm(
        total=total_rollouts,
        desc="E5 oracle checkpoint selection",
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

            row = evaluate_e5_checkpoint(
                model_name=task.model_name,
                step=task.step,
                path=task.path,
                dataset_root=config.dataset_root,
                oracle_episodes=config.oracle_episodes,
                prototype_manifest=config.prototype_manifest,
                selected_k=config.selected_k,
                train_seed=config.train_seed,
                code_commit=config.code_commit,
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
                f"E5 checkpoint {row.get('model')}:{row.get('step')} returned no rollouts"
            )
        checkpoint_rows.append(row)
        rollout_rows.extend(nested)
    return checkpoint_rows, rollout_rows


def select_e5_checkpoints(
    *,
    bc_run: Path,
    diffusion_run: Path,
    prototype_manifest: Path,
    dataset_root: Path,
    output_dir: Path,
    train_seed: int,
    overwrite: bool,
    num_gpus: int = 1,
) -> Path:
    """Run strict development selection for both E5 model families."""

    _validate_num_gpus(num_gpus)
    split = load_split()
    sidecar = load_prototype_sidecar(prototype_manifest)
    oracle_episodes = load_oracle_mode_episodes(
        dataset_root=dataset_root,
        prototype_manifest=prototype_manifest,
        episode_indices=split.development,
        split_name="development",
    )
    reconstruction = preflight_reconstruction(oracle_episodes)
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]

    bc_checkpoints = discover_checkpoints(bc_run)
    diffusion_checkpoints = discover_checkpoints(diffusion_run)
    tasks = _checkpoint_tasks(
        bc_checkpoints,
        diffusion_checkpoints,
        bc_model_name=MODE_BC,
        diffusion_model_name=MODE_DIFFUSION,
    )
    config = WorkerConfig(
        dataset_root=dataset_root,
        development_episodes=tuple(split.development),
        environment_seeds=tuple(split.development),
        train_seed=train_seed,
        code_commit=code_commit,
        evaluation_protocol="e5_oracle_mode",
        prototype_manifest=prototype_manifest,
        selected_k=sidecar.selected_k,
        oracle_episodes=oracle_episodes,
    )
    if num_gpus == 1:
        raw_rows = _evaluate_sequential(tasks=tasks, config=config)
    else:
        raw_rows = _evaluate_parallel(tasks=tasks, config=config, num_gpus=num_gpus)
    rows, rollout_rows = _separate_rollout_rows(raw_rows)
    selected = {model: _select(rows, model) for model in (MODE_BC, MODE_DIFFUSION)}

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
                "experiment": "e5_oracle_mode_development_checkpoint_selection",
                "dataset_revision": DATASET_REVISION,
                "development_episodes": list(split.development),
                "prototype_manifest": str(prototype_manifest),
                "prototype_manifest_sha256": sha256_file(prototype_manifest),
                "selected_k": sidecar.selected_k,
                "mode_source": "full_demonstration_nearest_train_medoid",
                "initial_state_source": "dataset_first_frame_reconstruction",
                "selection_order": [
                    "closed_loop_success_rate_desc",
                    "mean_max_reward_desc",
                    "training_step_asc",
                ],
                "execution": {
                    "num_gpus": num_gpus,
                    "parallelism": ("checkpoint_process_pool" if num_gpus > 1 else "sequential"),
                    "task_granularity": "one_model_checkpoint_all_development_episodes",
                },
                "runtime": runtime,
            },
        )
        write_metrics(
            temporary,
            {
                "selected_mode_bc_step": selected[MODE_BC]["step"],
                "selected_mode_bc_success_rate": selected[MODE_BC]["closed_loop_success_rate"],
                "selected_mode_diffusion_step": selected[MODE_DIFFUSION]["step"],
                "selected_mode_diffusion_success_rate": selected[MODE_DIFFUSION][
                    "closed_loop_success_rate"
                ],
                "development_episode_count": len(oracle_episodes),
                "development_rollout_count": len(rollout_rows),
                "reconstruction_preflight_passed": len(reconstruction),
            },
            comments={
                "selected_mode_bc_step": "Success, reward, then earliest-step ordering",
                "selected_mode_bc_success_rate": "Development oracle-mode closed-loop success",
                "selected_mode_diffusion_step": "Success, reward, then earliest-step ordering",
                "selected_mode_diffusion_success_rate": (
                    "Development oracle-mode closed-loop success"
                ),
                "development_episode_count": "One rollout per development episode/checkpoint",
                "development_rollout_count": "Across both model checkpoint sets",
                "reconstruction_preflight_passed": "No episode is silently excluded",
            },
        )
        _record_selection(
            temporary / "recording.rrd",
            rows,
            models=(MODE_BC, MODE_DIFFUSION),
        )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--diffusion-run", type=Path, required=True)
    parser.add_argument("--prototype-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e5_oracle_checkpoint_selection",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = select_e5_checkpoints(
        bc_run=args.bc_run.expanduser().resolve(),
        diffusion_run=args.diffusion_run.expanduser().resolve(),
        prototype_manifest=args.prototype_manifest.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        train_seed=args.train_seed,
        overwrite=args.overwrite,
        num_gpus=args.num_gpus,
    )
    print(f"E5 checkpoint selection output: {result}")


if __name__ == "__main__":
    main()
