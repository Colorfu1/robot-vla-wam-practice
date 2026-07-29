"""Compare the selected BCChunk and Diffusion checkpoints on development resets."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from tqdm import tqdm

from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy

from .config import (
    DATASET_REVISION,
    HORIZON,
    N_ACTION_STEPS,
    N_OBS_STEPS,
    default_output_root,
    load_json_object,
    practice_root,
)
from .dataset_split import load_split
from .development_recordings import (
    CATEGORY_ORDER,
    DEVELOPMENT_ENVIRONMENT_SEEDS,
    DEVELOPMENT_EXPERIMENT_ID,
    PairedTaskRecordings,
    task_metadata_from_result,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import LoadedPolicy, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rollout import RolloutResult, run_rollout, save_rollout_video

EXPERIMENT_ID = DEVELOPMENT_EXPERIMENT_ID


@dataclass(frozen=True)
class SelectedPolicy:
    model_name: str
    step: int
    path: Path
    checkpoint_id: str
    train_seed: int
    recorded_success_rate: float
    recorded_mean_max_reward: float


def _selection_entry(
    payload: Mapping[str, Any],
    *,
    model_name: str,
) -> SelectedPolicy:
    raw = payload.get(model_name)
    if not isinstance(raw, dict):
        raise TypeError(f"Selection must contain a {model_name!r} object")
    step = raw.get("step")
    path = raw.get("checkpoint_path")
    checkpoint_id = raw.get("checkpoint_id")
    train_seed = raw.get("train_seed")
    success_rate = raw.get("closed_loop_success_rate")
    mean_max_reward = raw.get("mean_max_reward")
    if raw.get("model") != model_name:
        raise ValueError(f"{model_name}.model must be {model_name!r}")
    if raw.get("dataset_revision") != DATASET_REVISION:
        raise ValueError(f"{model_name}.dataset_revision does not match this experiment")
    expected_rollouts = len(DEVELOPMENT_ENVIRONMENT_SEEDS)
    if raw.get("development_environment_seed_count") != expected_rollouts:
        raise ValueError(
            f"{model_name}.development_environment_seed_count must be {expected_rollouts}"
        )
    if raw.get("num_rollouts") != expected_rollouts:
        raise ValueError(f"{model_name}.num_rollouts must be {expected_rollouts}")
    expected_scheduler = "DDPM" if model_name == "diffusion" else None
    expected_inference_steps = 100 if model_name == "diffusion" else None
    if raw.get("scheduler") != expected_scheduler:
        raise ValueError(f"{model_name}.scheduler must be {expected_scheduler!r}")
    if raw.get("num_inference_steps") != expected_inference_steps:
        raise ValueError(f"{model_name}.num_inference_steps must be {expected_inference_steps!r}")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"{model_name}.step must be a non-negative integer")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{model_name}.checkpoint_path must be a non-empty string")
    if (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_id)
    ):
        raise ValueError(f"{model_name}.checkpoint_id must be a lowercase SHA256 digest")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int):
        raise TypeError(f"{model_name}.train_seed must be an integer")
    if (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or not 0.0 <= float(success_rate) <= 1.0
    ):
        raise ValueError(f"{model_name}.closed_loop_success_rate must be in [0,1]")
    if isinstance(mean_max_reward, bool) or not isinstance(mean_max_reward, (int, float)):
        raise TypeError(f"{model_name}.mean_max_reward must be numeric")
    checkpoint_path = Path(path).expanduser().resolve()
    model_path = checkpoint_path / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {model_path}")
    observed_id = sha256_file(model_path)
    if observed_id != checkpoint_id:
        raise RuntimeError(
            f"{model_name} checkpoint SHA256 mismatch: expected {checkpoint_id}, "
            f"observed {observed_id}"
        )
    return SelectedPolicy(
        model_name=model_name,
        step=step,
        path=checkpoint_path,
        checkpoint_id=checkpoint_id,
        train_seed=train_seed,
        recorded_success_rate=float(success_rate),
        recorded_mean_max_reward=float(mean_max_reward),
    )


def load_selected_policies(path: str | Path) -> tuple[SelectedPolicy, SelectedPolicy]:
    payload = load_json_object(path)
    bc = _selection_entry(payload, model_name="bc_chunk")
    diffusion = _selection_entry(payload, model_name="diffusion")
    if bc.train_seed != diffusion.train_seed:
        raise ValueError("Selected checkpoints disagree on train_seed")
    return bc, diffusion


def _resolve_device(value: str) -> str:
    try:
        device = torch.device(value)
    except RuntimeError as error:
        raise ValueError(f"Invalid device: {value!r}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is outside the "
                f"{torch.cuda.device_count()} visible device(s)"
            )
        torch.cuda.set_device(index)
        return f"cuda:{index}"
    return str(device)


def _validate_loaded_policy(
    loaded: LoadedPolicy,
    selected: SelectedPolicy,
) -> None:
    if selected.model_name == "bc_chunk":
        if not isinstance(loaded.policy, BCChunkPolicy):
            raise TypeError("Selected bc_chunk path is not a BCChunk checkpoint")
    elif selected.model_name == "diffusion":
        if not isinstance(loaded.policy, DiffusionPolicy):
            raise TypeError("Selected diffusion path is not a Diffusion checkpoint")
        replace_scheduler(
            loaded.policy,
            scheduler_name="DDPM",
            num_inference_steps=100,
        )
    else:
        raise ValueError(f"Unsupported selected model: {selected.model_name}")
    expected = {
        "n_obs_steps": N_OBS_STEPS,
        "horizon": HORIZON,
        "n_action_steps": N_ACTION_STEPS,
    }
    for field, value in expected.items():
        if getattr(loaded.config, field) != value:
            raise ValueError(
                f"{selected.model_name} checkpoint has {field}="
                f"{getattr(loaded.config, field)!r}, expected {value}"
            )


def _rollout_row(
    result: RolloutResult,
    *,
    selected: SelectedPolicy,
    num_rollouts: int,
    code_commit: str | None,
    video_path: str,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "condition_id": result.condition_id,
        "model": result.model,
        "checkpoint_step": selected.step,
        "checkpoint_id": selected.checkpoint_id,
        "train_seed": selected.train_seed,
        "environment_seed": result.environment_seed,
        "policy_seed": result.policy_seed,
        "replan_index": None,
        "initial_noise_seed": None,
        "initial_noise_sha256": None,
        "reverse_noise_seed": None,
        "reverse_generator_state_sha256": None,
        "num_inference_steps": result.num_inference_steps,
        "num_action_samples": 1,
        "num_rollouts": num_rollouts,
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
        "video_path": video_path,
    }


def _replan_rows(
    result: RolloutResult,
    *,
    selected: SelectedPolicy,
    num_rollouts: int,
    code_commit: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "condition_id": result.condition_id,
            "model": result.model,
            "checkpoint_step": selected.step,
            "checkpoint_id": selected.checkpoint_id,
            "train_seed": selected.train_seed,
            "environment_seed": result.environment_seed,
            "policy_seed": result.policy_seed,
            "replan_index": replan.replan_index,
            "initial_noise_seed": replan.initial_noise_seed,
            "initial_noise_sha256": replan.initial_noise_sha256,
            "reverse_noise_seed": replan.reverse_noise_seed,
            "reverse_generator_state_sha256": replan.reverse_generator_state_sha256,
            "num_inference_steps": replan.num_inference_steps,
            "num_action_samples": 1,
            "num_rollouts": num_rollouts,
            "dataset_revision": DATASET_REVISION,
            "code_commit": code_commit,
            "scheduler": replan.scheduler,
            "chunk_latency_ms": replan.latency_ms,
            "normalized_chunk": replan.normalized_chunk.tolist(),
        }
        for replan in result.replans
    ]


def _result_map(
    results: Sequence[RolloutResult],
    *,
    model_name: str,
    expected_seeds: Sequence[int],
) -> dict[int, RolloutResult]:
    mapped = {result.environment_seed: result for result in results}
    if len(mapped) != len(results):
        raise RuntimeError(f"{model_name} produced duplicate environment seeds")
    if set(mapped) != set(expected_seeds):
        raise RuntimeError(f"{model_name} did not produce the complete development seed set")
    if any(result.model != model_name for result in results):
        raise RuntimeError(f"{model_name} produced a rollout with the wrong model label")
    if any(result.policy_seed != 0 for result in results):
        raise RuntimeError(f"{model_name} comparison rollouts must use policy_seed=0")
    return mapped


def classify_results(
    bc_results: Sequence[RolloutResult],
    diffusion_results: Sequence[RolloutResult],
    *,
    environment_seeds: Sequence[int],
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    bc_by_seed = _result_map(
        bc_results,
        model_name="bc_chunk",
        expected_seeds=environment_seeds,
    )
    diffusion_by_seed = _result_map(
        diffusion_results,
        model_name="diffusion",
        expected_seeds=environment_seeds,
    )
    categories = {name: [] for name in CATEGORY_ORDER}
    cases = []
    for seed in environment_seeds:
        bc = bc_by_seed[seed]
        diffusion = diffusion_by_seed[seed]
        if bc.success and not diffusion.success:
            category = "bc_pass_diffusion_fail"
        elif diffusion.success and not bc.success:
            category = "diffusion_pass_bc_fail"
        elif not bc.success and not diffusion.success:
            category = "both_fail"
        else:
            category = "both_pass"
        categories[category].append(seed)
        if not np.array_equal(bc.agent_path[0], diffusion.agent_path[0]):
            raise RuntimeError(f"Models started from different agent positions for seed {seed}")
        if not np.array_equal(bc.block_path[0], diffusion.block_path[0]):
            raise RuntimeError(f"Models started from different block poses for seed {seed}")
        cases.append(
            {
                "environment_seed": seed,
                "category": category,
                "initial_state": {
                    "agent_position": bc.agent_path[0].tolist(),
                    "block_pose": bc.block_path[0].tolist(),
                },
                "bc_chunk": _compact_result(bc),
                "diffusion": _compact_result(diffusion),
            }
        )
    return categories, cases


def _compact_result(result: RolloutResult) -> dict[str, Any]:
    video_path = f"videos/{result.model}_env_{result.environment_seed:04d}.mp4"
    recording_name = f"env_{result.environment_seed:04d}.rrd"
    return {
        "success": result.success,
        "sum_reward": result.sum_reward,
        "max_reward": result.max_reward,
        "final_reward": result.final_reward,
        "episode_length": result.episode_length,
        "video_path": video_path,
        "rerun_recording_path": f"recordings/{result.model}/{recording_name}",
        "rerun_entity_path": "/rollout",
    }


def _evaluate_selected_policy(
    selected: SelectedPolicy,
    *,
    device: str,
    environment_seeds: Sequence[int],
    output_dir: Path,
    recorder: PairedTaskRecordings,
    progress: tqdm,
    num_rollouts: int,
    code_commit: str | None,
) -> tuple[list[RolloutResult], list[dict[str, Any]], list[dict[str, Any]]]:
    progress.set_postfix(
        model=selected.model_name,
        checkpoint=f"{selected.step:06d}",
        phase="loading",
        refresh=True,
    )
    loaded = load_policy_with_processors(
        selected.path,
        local_files_only=True,
        device=device,
    )
    try:
        _validate_loaded_policy(loaded, selected)
        results = []
        rollout_rows = []
        replan_rows = []
        for environment_seed in environment_seeds:
            progress.set_postfix(
                model=selected.model_name,
                checkpoint=f"{selected.step:06d}",
                environment_seed=environment_seed,
                phase="rollout",
                refresh=True,
            )
            result = run_rollout(
                policy=loaded.policy,
                preprocessor=loaded.preprocessor,
                postprocessor=loaded.postprocessor,
                environment_seed=environment_seed,
                policy_seed=0,
                record_frames=True,
                episode_length=300,
                condition_id=EXPERIMENT_ID,
            )
            progress.set_postfix(
                model=selected.model_name,
                checkpoint=f"{selected.step:06d}",
                environment_seed=environment_seed,
                phase="saving",
                refresh=True,
            )
            recorder.add(
                result,
                task_metadata_from_result(
                    result,
                    checkpoint_step=selected.step,
                    checkpoint_id=selected.checkpoint_id,
                ),
            )
            video_name = f"{selected.model_name}_env_{environment_seed:04d}.mp4"
            save_rollout_video(result, output_dir / "videos" / video_name)
            rollout_rows.append(
                _rollout_row(
                    result,
                    selected=selected,
                    num_rollouts=num_rollouts,
                    code_commit=code_commit,
                    video_path=f"videos/{video_name}",
                )
            )
            replan_rows.extend(
                _replan_rows(
                    result,
                    selected=selected,
                    num_rollouts=num_rollouts,
                    code_commit=code_commit,
                )
            )
            results.append(replace(result, frames=()))
            progress.update(1)
        return results, rollout_rows, replan_rows
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _observed_summary(
    results: Sequence[RolloutResult],
    selected: SelectedPolicy,
) -> dict[str, Any]:
    success_rate = float(np.mean([result.success for result in results]))
    mean_max_reward = float(np.mean([result.max_reward for result in results]))
    return {
        "checkpoint_step": selected.step,
        "checkpoint_id": selected.checkpoint_id,
        "success_count": int(sum(result.success for result in results)),
        "success_rate": success_rate,
        "mean_max_reward": mean_max_reward,
        "selection_recorded_success_rate": selected.recorded_success_rate,
        "selection_recorded_mean_max_reward": selected.recorded_mean_max_reward,
        "success_rate_delta_from_selection": success_rate - selected.recorded_success_rate,
        "mean_max_reward_delta_from_selection": (
            mean_max_reward - selected.recorded_mean_max_reward
        ),
    }


def compare_development_checkpoints(
    *,
    selection_path: Path,
    output_dir: Path,
    device: str,
    overwrite: bool,
) -> Path:
    resolved_device = _resolve_device(device)
    print("Validating selection metadata and checkpoint hashes...")
    selection_sha256 = sha256_file(selection_path)
    bc_selected, diffusion_selected = load_selected_policies(selection_path)
    print(
        "Selected checkpoints: "
        f"BCChunk {bc_selected.step:06d}, Diffusion {diffusion_selected.step:06d}"
    )
    split = load_split()
    environment_seeds = split.development_environment_seeds
    if environment_seeds != DEVELOPMENT_ENVIRONMENT_SEEDS:
        raise RuntimeError("Development comparison requires environment seeds 1000..1019")
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    rollouts_per_policy = len(environment_seeds)
    total_rollouts = len(environment_seeds) * 2

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        (temporary / "videos").mkdir()
        recorder = PairedTaskRecordings(temporary / "recordings")
        with tqdm(
            total=total_rollouts,
            desc="Development comparison",
            unit="rollout",
            dynamic_ncols=True,
        ) as progress:
            bc_results, bc_rows, bc_replans = _evaluate_selected_policy(
                bc_selected,
                device=resolved_device,
                environment_seeds=environment_seeds,
                output_dir=temporary,
                recorder=recorder,
                progress=progress,
                num_rollouts=rollouts_per_policy,
                code_commit=code_commit,
            )
            diffusion_results, diffusion_rows, diffusion_replans = _evaluate_selected_policy(
                diffusion_selected,
                device=resolved_device,
                environment_seeds=environment_seeds,
                output_dir=temporary,
                recorder=recorder,
                progress=progress,
                num_rollouts=rollouts_per_policy,
                code_commit=code_commit,
            )

        categories, cases = classify_results(
            bc_results,
            diffusion_results,
            environment_seeds=environment_seeds,
        )
        category_by_seed = {
            seed: category for category, seeds in categories.items() for seed in seeds
        }
        rollout_rows = bc_rows + diffusion_rows
        replan_rows = bc_replans + diffusion_replans
        for row in rollout_rows:
            row["comparison_category"] = category_by_seed[int(row["environment_seed"])]
        for row in replan_rows:
            row["comparison_category"] = category_by_seed[int(row["environment_seed"])]

        summaries = {
            "bc_chunk": _observed_summary(bc_results, bc_selected),
            "diffusion": _observed_summary(diffusion_results, diffusion_selected),
        }
        comparison = {
            "schema_version": 1,
            "label": "Closed-loop",
            "result_class": "Formal",
            "experiment": EXPERIMENT_ID,
            "device": resolved_device,
            "environment_seeds": list(environment_seeds),
            "policy_seed": 0,
            "diffusion_scheduler": "DDPM",
            "num_inference_steps": 100,
            "num_rollouts_per_policy": rollouts_per_policy,
            "num_rollouts": total_rollouts,
            "summaries": summaries,
            "category_counts": {category: len(categories[category]) for category in CATEGORY_ORDER},
            "categories": categories,
            "cases": cases,
        }
        write_json_atomic(temporary / "comparison.json", comparison)
        write_records_parquet(temporary / "rollouts.parquet", rollout_rows)
        write_records_parquet(temporary / "replans.parquet", replan_rows)
        write_metrics(
            temporary,
            {
                "bc_success_rate": summaries["bc_chunk"]["success_rate"],
                "diffusion_success_rate": summaries["diffusion"]["success_rate"],
                "bc_pass_diffusion_fail": len(categories["bc_pass_diffusion_fail"]),
                "diffusion_pass_bc_fail": len(categories["diffusion_pass_bc_fail"]),
                "both_fail": len(categories["both_fail"]),
                "both_pass": len(categories["both_pass"]),
                "num_environment_seeds": len(environment_seeds),
                "num_rollouts_per_policy": rollouts_per_policy,
                "num_rollouts": total_rollouts,
            },
            comments={
                "bc_success_rate": "Selected BCChunk checkpoint on development resets",
                "diffusion_success_rate": "Selected Diffusion checkpoint at policy_seed=0",
                "bc_pass_diffusion_fail": "Seeds where only BCChunk succeeds",
                "diffusion_pass_bc_fail": "Seeds where only Diffusion succeeds",
                "both_fail": "Seeds where neither policy succeeds",
                "both_pass": "Seeds where both policies succeed",
                "num_environment_seeds": "Committed development reset seeds",
                "num_rollouts_per_policy": "One rollout per development seed for each policy",
                "num_rollouts": "Two policies times twenty reset seeds",
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "label": "Closed-loop",
                "result_class": "Formal",
                "experiment": EXPERIMENT_ID,
                "selection_sha256": selection_sha256,
                "checkpoint_steps": {
                    "bc_chunk": bc_selected.step,
                    "diffusion": diffusion_selected.step,
                },
                "checkpoint_ids": {
                    "bc_chunk": bc_selected.checkpoint_id,
                    "diffusion": diffusion_selected.checkpoint_id,
                },
                "train_seed": bc_selected.train_seed,
                "environment_seeds": list(environment_seeds),
                "policy_seed": 0,
                "diffusion_scheduler": "DDPM",
                "num_inference_steps": 100,
                "episode_length": 300,
                "device": resolved_device,
                "dataset_revision": DATASET_REVISION,
                "num_rollouts_per_policy": rollouts_per_policy,
                "num_rollouts": total_rollouts,
                "runtime": runtime,
                "rerun_recording_directories": {
                    "bc_chunk": "recordings/bc_chunk",
                    "diffusion": "recordings/diffusion",
                },
                "rerun_recordings_per_model": rollouts_per_policy,
            },
        )
        recorder.finalize(
            comparison_path=temporary / "comparison.json",
        )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=default_output_root()
        / "pilot"
        / "checkpoint_selection"
        / "selected_checkpoints.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "development_comparison",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = compare_development_checkpoints(
        selection_path=args.selection.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"Development comparison output: {result}")


if __name__ == "__main__":
    main()
