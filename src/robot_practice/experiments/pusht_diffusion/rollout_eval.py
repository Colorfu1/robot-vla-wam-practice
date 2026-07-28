"""E2 paired BCChunk and multi-seed Diffusion closed-loop PushT evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
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
    resample_agent_block_trajectory,
    trajectory_distance_matrix,
    upper_triangle_mean,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rerun_visualization import RolloutRecording
from .rollout import RolloutResult, run_rollout, save_rollout_video
from .statistics import environment_cluster_bootstrap, wilson_interval

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_E2_CONFIG = load_named_config("e2_rollout.json")
PRIMARY_MODE_THRESHOLD = float(_E2_CONFIG["mode_distance_threshold"])
SENSITIVITY_THRESHOLDS = tuple(float(value) for value in _E2_CONFIG["mode_distance_sensitivity"])
TRAJECTORY_RESAMPLE_POINTS = int(_E2_CONFIG["trajectory_resample_points"])


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
    return upper_triangle_mean(trajectory_distance_matrix([_resampled(item) for item in results]))


def _bootstrap_diversity(
    grouped: Mapping[int, Sequence[RolloutResult]],
) -> dict[str, float] | None:
    pairwise_groups: dict[int, np.ndarray] = {}
    for environment_seed, results in grouped.items():
        if len(results) < 2:
            continue
        matrix = trajectory_distance_matrix([_resampled(item) for item in results])
        pairwise_groups[environment_seed] = matrix[np.triu_indices(len(results), k=1)]
    if not pairwise_groups:
        return None
    return environment_cluster_bootstrap(pairwise_groups).to_dict()


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


def _rollout_row(
    result: RolloutResult,
    *,
    checkpoint_id: str,
    train_seed: int,
    mode_label: int | None,
    num_rollouts: int,
    code_commit: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": "E2",
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
        "num_rollouts": num_rollouts,
        "dataset_revision": DATASET_REVISION,
        "code_commit": code_commit,
        "success": result.success,
        "sum_reward": result.sum_reward,
        "max_reward": result.max_reward,
        "final_reward": result.final_reward,
        "episode_length": result.episode_length,
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
    num_rollouts: int,
    code_commit: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "E2",
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
            "num_rollouts": num_rollouts,
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
        if not result.success:
            continue
        axis.plot(
            result.agent_path[:, 0],
            result.agent_path[:, 1],
            alpha=0.55,
            label=f"env {result.environment_seed} / policy {result.policy_seed}",
        )
    axis.set(xlim=(0, 512), ylim=(512, 0), xlabel="x", ylabel="y", title="Successful agent paths")
    if any(result.success for result in diffusion_results):
        axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "successful_rollout_modes.png", dpi=160)
    plt.close(figure)

    labels = ["all", "successful", "failed"]
    values = [
        diversity.get("all") or 0.0,
        diversity.get("successful") or 0.0,
        diversity.get("failed") or 0.0,
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#4e79a7", "#59a14f", "#e15759"])
    axis.set_ylabel("joint trajectory RMS distance")
    axis.set_title("Success-conditioned Diffusion rollout diversity")
    figure.tight_layout()
    figure.savefig(output_dir / "success_conditioned_diversity.png", dpi=160)
    plt.close(figure)


def evaluate_multi_seed_rollouts(
    *,
    bc_policy_path: str | Path,
    diffusion_policy_path: str | Path,
    output_dir: Path,
    num_environment_seeds: int,
    num_policy_seeds: int,
    train_seed: int,
    episode_length: int,
    label: str,
    overwrite: bool,
) -> Path:
    if label not in {"Smoke", "Closed-loop"}:
        raise ValueError("label must be Smoke or Closed-loop")
    split = load_split()
    if not 1 <= num_environment_seeds <= len(split.e2_environment_seeds):
        raise ValueError(f"num_environment_seeds must be in [1,{len(split.e2_environment_seeds)}]")
    if not 1 <= num_policy_seeds <= 20:
        raise ValueError("num_policy_seeds must be in [1,20]")
    environment_seeds = split.e2_environment_seeds[:num_environment_seeds]
    configured_environment_seeds = tuple(int(value) for value in _E2_CONFIG["environment_seeds"])
    if configured_environment_seeds != split.e2_environment_seeds:
        raise ValueError("e2_rollout.json environment seeds disagree with split.json")
    policy_seeds = tuple(range(num_policy_seeds))
    num_diffusion_rollouts = len(environment_seeds) * len(policy_seeds)
    num_bc_rollouts = len(environment_seeds)

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
        scheduler_name="DDPM",
        num_inference_steps=100,
    )
    for field in ("n_obs_steps", "horizon", "n_action_steps"):
        if getattr(bc_loaded.config, field) != getattr(diffusion_loaded.config, field):
            raise ValueError(f"BCChunk and Diffusion disagree on {field}")
    if diffusion_loaded.config.n_action_steps != int(_E2_CONFIG["n_action_steps"]):
        raise ValueError("Loaded checkpoints do not use the registered E2 action horizon")

    checkpoint_ids = {
        "bc_chunk": _checkpoint_sha256(bc_policy_path),
        "diffusion": _checkpoint_sha256(diffusion_policy_path),
    }
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    bc_results: list[RolloutResult] = []
    diffusion_results: list[RolloutResult] = []
    all_replan_rows: list[dict[str, Any]] = []

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        video_dir = temporary / "videos"
        video_dir.mkdir()
        recorder = RolloutRecording(temporary / "recording.rrd")
        try:
            for environment_seed in environment_seeds:
                result = run_rollout(
                    policy=bc_loaded.policy,
                    preprocessor=bc_loaded.preprocessor,
                    postprocessor=bc_loaded.postprocessor,
                    environment_seed=environment_seed,
                    policy_seed=0,
                    record_frames=True,
                    episode_length=episode_length,
                    condition_id="bc_chunk",
                )
                recorder.add(result)
                save_rollout_video(
                    result,
                    video_dir / f"bc_chunk_env_{environment_seed:04d}.mp4",
                )
                all_replan_rows.extend(
                    _replan_rows(
                        result,
                        checkpoint_id=checkpoint_ids["bc_chunk"],
                        train_seed=train_seed,
                        num_rollouts=num_bc_rollouts,
                        code_commit=code_commit,
                    )
                )
                bc_results.append(replace(result, frames=()))

            for environment_seed in environment_seeds:
                for policy_seed in policy_seeds:
                    result = run_rollout(
                        policy=diffusion_loaded.policy,
                        preprocessor=diffusion_loaded.preprocessor,
                        postprocessor=diffusion_loaded.postprocessor,
                        environment_seed=environment_seed,
                        policy_seed=policy_seed,
                        record_frames=True,
                        episode_length=episode_length,
                        condition_id="diffusion_ddpm100",
                    )
                    recorder.add(result)
                    save_rollout_video(
                        result,
                        video_dir
                        / f"diffusion_env_{environment_seed:04d}_policy_{policy_seed:02d}.mp4",
                    )
                    all_replan_rows.extend(
                        _replan_rows(
                            result,
                            checkpoint_id=checkpoint_ids["diffusion"],
                            train_seed=train_seed,
                            num_rollouts=num_diffusion_rollouts,
                            code_commit=code_commit,
                        )
                    )
                    diffusion_results.append(replace(result, frames=()))
        finally:
            recorder.close()

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
                num_rollouts=num_bc_rollouts,
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
                num_rollouts=num_diffusion_rollouts,
                code_commit=code_commit,
            )
            for result in diffusion_results
        )
        write_records_parquet(temporary / "rollouts.parquet", rollout_rows)
        write_records_parquet(temporary / "replans.parquet", all_replan_rows)

        successes_per_environment = {
            seed: sum(result.success for result in results) for seed, results in grouped.items()
        }
        policy_zero = [result for result in diffusion_results if result.policy_seed == 0]
        any_success = [
            int(successes_per_environment[environment_seed] > 0)
            for environment_seed in environment_seeds
        ]
        two_success = [
            int(successes_per_environment[environment_seed] >= 2)
            for environment_seed in environment_seeds
        ]
        two_modes = [
            int(primary_mode_counts[environment_seed] >= 2)
            for environment_seed in environment_seeds
        ]
        all_grouped = grouped
        successful_grouped = {
            seed: [result for result in results if result.success]
            for seed, results in grouped.items()
        }
        failed_grouped = {
            seed: [result for result in results if not result.success]
            for seed, results in grouped.items()
        }
        all_diversity_ci = _bootstrap_diversity(all_grouped)
        successful_diversity_ci = _bootstrap_diversity(successful_grouped)
        failed_diversity_ci = _bootstrap_diversity(failed_grouped)
        diversity_points = {
            "all": None if all_diversity_ci is None else all_diversity_ci["estimate"],
            "successful": (
                None if successful_diversity_ci is None else successful_diversity_ci["estimate"]
            ),
            "failed": (None if failed_diversity_ci is None else failed_diversity_ci["estimate"]),
        }
        single_interval = wilson_interval(
            sum(result.success for result in policy_zero),
            len(policy_zero),
        )
        bc_interval = wilson_interval(
            sum(result.success for result in bc_results),
            len(bc_results),
        )
        any_interval = wilson_interval(sum(any_success), len(any_success))
        two_success_interval = wilson_interval(sum(two_success), len(two_success))
        two_modes_interval = wilson_interval(sum(two_modes), len(two_modes))
        repeated_success_interval = environment_cluster_bootstrap(
            {
                seed: [float(result.success) for result in results]
                for seed, results in grouped.items()
            }
        )
        metrics = {
            "bc_success_rate": bc_interval.estimate,
            "bc_success_ci95_lower": bc_interval.lower,
            "bc_success_ci95_upper": bc_interval.upper,
            "single_sample_success": single_interval.estimate,
            "single_sample_success_ci95_lower": single_interval.lower,
            "single_sample_success_ci95_upper": single_interval.upper,
            "all_diffusion_rollout_success": repeated_success_interval.estimate,
            "all_diffusion_rollout_success_ci95_lower": repeated_success_interval.lower,
            "all_diffusion_rollout_success_ci95_upper": repeated_success_interval.upper,
            "success_any_at_m": any_interval.estimate,
            "success_any_at_5": (
                any_interval.estimate if len(policy_seeds) == 5 else None
            ),
            "mean_successes_per_environment_seed": float(
                np.mean(list(successes_per_environment.values()))
            ),
            "fraction_with_2plus_successful_rollouts": two_success_interval.estimate,
            "fraction_with_2plus_distinct_success_modes": two_modes_interval.estimate,
            "successful_rollout_diversity": diversity_points["successful"],
            "failed_rollout_diversity": diversity_points["failed"],
            "all_rollout_diversity": diversity_points["all"],
            "num_environment_seeds": len(environment_seeds),
            "num_policy_seeds": len(policy_seeds),
            "num_rollouts": num_diffusion_rollouts + num_bc_rollouts,
        }
        write_metrics(
            temporary,
            metrics,
            comments={
                "bc_success_rate": "One deterministic BCChunk rollout per environment seed",
                "single_sample_success": "Diffusion deployment metric at fixed policy_seed=0",
                "all_diffusion_rollout_success": "Repeated samples; environment-cluster bootstrap",
                "success_any_at_m": "Oracle coverage for the configured number of policy seeds",
                "success_any_at_5": "Pre-registered oracle coverage; null unless M=5",
                "mean_successes_per_environment_seed": "Successful Diffusion samples per reset seed",
                "fraction_with_2plus_successful_rollouts": "Seeds with at least two successes",
                "fraction_with_2plus_distinct_success_modes": (
                    "Seeds with at least two successful complete-linkage modes at 0.05"
                ),
                "successful_rollout_diversity": "Joint agent/block distance among successes",
                "failed_rollout_diversity": "Joint agent/block distance among failures",
                "all_rollout_diversity": "Joint agent/block distance across all samples",
                "num_environment_seeds": "Independent initial PushT states",
                "num_policy_seeds": "Independent initial-noise streams per initial state",
                "num_rollouts": "BCChunk plus Diffusion environment executions",
            },
        )
        write_json_atomic(
            temporary / "mode_analysis.json",
            {
                "primary_threshold": PRIMARY_MODE_THRESHOLD,
                "primary_mode_counts": primary_mode_counts,
                "sensitivity_mode_counts": sensitivity_counts,
                "all_diversity_ci95": all_diversity_ci,
                "successful_diversity_ci95": successful_diversity_ci,
                "failed_diversity_ci95": failed_diversity_ci,
                "success_any_at_5_ci95": any_interval.to_dict(),
                "fraction_with_2plus_success_ci95": two_success_interval.to_dict(),
                "fraction_with_2plus_modes_ci95": two_modes_interval.to_dict(),
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "label": label,
                "experiment": "E2",
                "scheduler": "DDPM",
                "num_inference_steps": 100,
                "num_action_samples": 1,
                "num_rollouts": num_diffusion_rollouts + num_bc_rollouts,
                "environment_seeds": list(environment_seeds),
                "policy_seeds": list(policy_seeds),
                "checkpoint_ids": checkpoint_ids,
                "train_seed": train_seed,
                "dataset_revision": DATASET_REVISION,
                "runtime": runtime,
            },
        )
        _save_static_figures(temporary, diffusion_results, diversity_points)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-policy", required=True)
    parser.add_argument("--diffusion-policy", required=True)
    parser.add_argument(
        "--num-environment-seeds",
        type=int,
        default=len(_E2_CONFIG["environment_seeds"]),
    )
    parser.add_argument(
        "--num-policy-seeds",
        type=int,
        default=len(_E2_CONFIG["policy_seeds"]),
    )
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--label", choices=("Smoke", "Closed-loop"), default="Closed-loop")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e2_rollouts",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = evaluate_multi_seed_rollouts(
        bc_policy_path=args.bc_policy,
        diffusion_policy_path=args.diffusion_policy,
        output_dir=args.output_dir.expanduser().resolve(),
        num_environment_seeds=args.num_environment_seeds,
        num_policy_seeds=args.num_policy_seeds,
        train_seed=args.train_seed,
        episode_length=args.episode_length,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(f"E2 output: {result}")


if __name__ == "__main__":
    main()
