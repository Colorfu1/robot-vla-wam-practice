"""E4 paired DDIM/DDPM sampling-step, success, and latency evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_IMAGES

from .config import (
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_output_root,
    load_named_config,
    practice_root,
)
from .dataset_split import load_split
from .latency import benchmark_latency
from .noise import initial_noise_seed, make_initial_noise, reverse_noise_seed
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import (
    controlled_sample,
    load_policy_with_processors,
    replace_scheduler,
)
from .provenance import (
    collect_runtime_provenance,
    sha256_file,
    state_dict_hashes,
    write_json_atomic,
)
from .rerun_visualization import RolloutRecording
from .rollout import make_pusht_env, run_rollout, save_rollout_video
from .statistics import wilson_interval

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_E4_CONFIG = load_named_config("e4_sampling_steps.json")


def _parse_steps(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed) or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{name} must contain unique positive integers")
    return parsed


def _conditions(
    schedulers: str,
    ddim_steps: str,
    ddpm_steps: str,
) -> tuple[tuple[str, int], ...]:
    selected = tuple(item.strip().upper() for item in schedulers.split(",") if item.strip())
    if not selected or any(item not in {"DDIM", "DDPM"} for item in selected):
        raise ValueError("schedulers must be a comma-separated subset of ddim,ddpm")
    if len(selected) != len(set(selected)):
        raise ValueError("schedulers must not contain duplicates")
    resolved: list[tuple[str, int]] = []
    if "DDIM" in selected:
        resolved.extend(("DDIM", step) for step in _parse_steps(ddim_steps, name="ddim-steps"))
    if "DDPM" in selected:
        resolved.extend(("DDPM", step) for step in _parse_steps(ddpm_steps, name="ddpm-steps"))
    return tuple(resolved)


def _condition_id(scheduler: str, steps: int) -> str:
    return f"{scheduler.lower()}_{steps:03d}"


def _checkpoint_sha256(path: str | Path) -> str:
    weights = Path(path).expanduser().resolve() / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint weights are missing: {weights}")
    return sha256_file(weights)


def _freeze_history(
    policy: DiffusionPolicy,
    preprocessor: Any,
    observation: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    processed = preprocessor(preprocess_observation(observation))
    history: dict[str, torch.Tensor] = {}
    for key in policy.config.input_features:
        if key not in processed:
            raise KeyError(f"Preprocessor output is missing {key!r}")
        value = processed[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Preprocessor output {key!r} must be a tensor")
        history[key] = value.unsqueeze(1).repeat(
            1,
            policy.config.n_obs_steps,
            *(1 for _ in value.shape[1:]),
        )
    if policy.config.image_features:
        history[OBS_IMAGES] = torch.stack(
            [history[key] for key in policy.config.image_features],
            dim=-4,
        )
    return history


def _rollout_row(
    result: Any,
    *,
    checkpoint_id: str,
    train_seed: int,
    total_rollouts: int,
    code_commit: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": "E4",
        "condition_id": result.condition_id,
        "model": "diffusion",
        "checkpoint_id": checkpoint_id,
        "train_seed": train_seed,
        "environment_seed": result.environment_seed,
        "policy_seed": result.policy_seed,
        "replan_index": None,
        "initial_noise_seed": None,
        "initial_noise_sha256": None,
        "reverse_noise_seed": None,
        "reverse_generator_state_sha256": None,
        "scheduler": result.scheduler,
        "num_inference_steps": result.num_inference_steps,
        "num_action_samples": 1,
        "num_rollouts": total_rollouts,
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
    }


def _replan_rows(
    result: Any,
    *,
    checkpoint_id: str,
    train_seed: int,
    total_rollouts: int,
    code_commit: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "E4",
            "condition_id": result.condition_id,
            "model": "diffusion",
            "checkpoint_id": checkpoint_id,
            "train_seed": train_seed,
            "environment_seed": result.environment_seed,
            "policy_seed": result.policy_seed,
            "replan_index": replan.replan_index,
            "initial_noise_seed": replan.initial_noise_seed,
            "initial_noise_sha256": replan.initial_noise_sha256,
            "reverse_noise_seed": replan.reverse_noise_seed,
            "reverse_generator_state_sha256": replan.reverse_generator_state_sha256,
            "scheduler": replan.scheduler,
            "num_inference_steps": replan.num_inference_steps,
            "num_action_samples": 1,
            "num_rollouts": total_rollouts,
            "dataset_revision": DATASET_REVISION,
            "code_commit": code_commit,
            "chunk_latency_ms": replan.latency_ms,
            "normalized_chunk": replan.normalized_chunk.tolist(),
        }
        for replan in result.replans
    ]


def _plot_results(
    output_dir: Path,
    condition_rows: Sequence[dict[str, Any]],
    latency_rows: Sequence[dict[str, Any]],
) -> None:
    labels = [str(row["condition_id"]) for row in condition_rows]
    successes = [float(row["success_rate"]) for row in condition_rows]
    p95_values = [float(row["end_to_end_p95_ms"]) for row in condition_rows]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(labels, successes, marker="o")
    axis.set(ylabel="success rate", xlabel="scheduler / inference steps", ylim=(0, 1.05))
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_dir / "sampling_steps_vs_success.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(p95_values, successes, s=70)
    for label, latency, success in zip(labels, p95_values, successes, strict=True):
        axis.annotate(label, (latency, success), xytext=(5, 4), textcoords="offset points")
    axis.set(
        xlabel="p95 end-to-end chunk latency (ms)",
        ylabel="success rate",
        ylim=(0, 1.05),
    )
    figure.tight_layout()
    figure.savefig(output_dir / "success_latency_pareto.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for condition_id in labels:
        values = [
            float(row["latency_ms"])
            for row in latency_rows
            if row["condition_id"] == condition_id and row["latency_kind"] == "end_to_end"
        ]
        axis.hist(values, bins=40, alpha=0.35, label=condition_id)
    axis.set(xlabel="end-to-end chunk latency (ms)", ylabel="count")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "latency_distribution.png", dpi=160)
    plt.close(figure)


def evaluate_sampling_steps(
    *,
    diffusion_policy_path: str | Path,
    output_dir: Path,
    conditions: Sequence[tuple[str, int]],
    num_episodes: int,
    policy_seed: int,
    train_seed: int,
    latency_warmup_iterations: int,
    latency_iterations: int,
    videos_per_condition: int,
    episode_length: int,
    overwrite: bool,
) -> Path:
    split = load_split()
    if not 1 <= num_episodes <= len(split.final_environment_seeds):
        raise ValueError(f"num_episodes must be in [1,{len(split.final_environment_seeds)}]")
    if videos_per_condition < 0 or videos_per_condition > num_episodes:
        raise ValueError("videos_per_condition must lie in [0,num_episodes]")
    environment_seeds = split.final_environment_seeds[:num_episodes]
    loaded = load_policy_with_processors(diffusion_policy_path, local_files_only=True)
    if not isinstance(loaded.policy, DiffusionPolicy):
        raise TypeError("diffusion-policy must contain policy.type=diffusion")
    if loaded.config.n_action_steps != 8:
        raise ValueError("E4 requires n_action_steps=8")
    if any(steps > loaded.config.num_train_timesteps for _, steps in conditions):
        raise ValueError(
            f"Inference steps cannot exceed num_train_timesteps={loaded.config.num_train_timesteps}"
        )
    checkpoint_id = _checkpoint_sha256(diffusion_policy_path)
    parameter_hashes = state_dict_hashes(loaded.policy.state_dict())
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    total_rollouts = len(conditions) * num_episodes

    rollout_rows: list[dict[str, Any]] = []
    replan_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        video_dir = temporary / "videos"
        video_dir.mkdir()
        recorder = RolloutRecording(temporary / "recording.rrd")
        try:
            for scheduler, steps in conditions:
                condition_id = _condition_id(scheduler, steps)
                replace_scheduler(
                    loaded.policy,
                    scheduler_name=scheduler,
                    num_inference_steps=steps,
                )
                if state_dict_hashes(loaded.policy.state_dict()) != parameter_hashes:
                    raise RuntimeError(f"{condition_id} changed model parameter hashes")

                condition_results = []
                for ordinal, environment_seed in enumerate(environment_seeds):
                    record_frames = ordinal < videos_per_condition
                    result = run_rollout(
                        policy=loaded.policy,
                        preprocessor=loaded.preprocessor,
                        postprocessor=loaded.postprocessor,
                        environment_seed=environment_seed,
                        policy_seed=policy_seed,
                        record_frames=record_frames,
                        episode_length=episode_length,
                        condition_id=condition_id,
                    )
                    recorder.add(result)
                    if record_frames:
                        save_rollout_video(
                            result,
                            video_dir / f"{condition_id}_env_{environment_seed:04d}.mp4",
                        )
                    rollout_rows.append(
                        _rollout_row(
                            result,
                            checkpoint_id=checkpoint_id,
                            train_seed=train_seed,
                            total_rollouts=total_rollouts,
                            code_commit=code_commit,
                        )
                    )
                    replan_rows.extend(
                        _replan_rows(
                            result,
                            checkpoint_id=checkpoint_id,
                            train_seed=train_seed,
                            total_rollouts=total_rollouts,
                            code_commit=code_commit,
                        )
                    )
                    condition_results.append(replace(result, frames=()))

                environment = make_pusht_env(episode_length=episode_length)
                raw_observation, _ = environment.reset(seed=environment_seeds[0])
                environment.close()
                frozen_history = _freeze_history(
                    loaded.policy,
                    loaded.preprocessor,
                    raw_observation,
                )
                device = get_device_from_parameters(loaded.policy)
                dtype = get_dtype_from_parameters(loaded.policy)
                first_seed = initial_noise_seed(
                    environment_seed=environment_seeds[0],
                    policy_seed=policy_seed,
                    replan_index=0,
                )
                second_seed = reverse_noise_seed(
                    environment_seed=environment_seeds[0],
                    replan_index=0,
                )
                explicit = make_initial_noise(
                    seed=first_seed,
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                    horizon=loaded.config.horizon,
                    action_dim=loaded.config.action_feature.shape[0],
                )

                def model_only(
                    frozen_batch: dict[str, torch.Tensor] = frozen_history,
                    initial_noise: torch.Tensor = explicit.tensor,
                    reverse_seed_value: int = second_seed,
                ) -> object:
                    return controlled_sample(
                        loaded.policy,
                        frozen_batch,
                        initial_noise=initial_noise,
                        reverse_seed=reverse_seed_value,
                    )

                def end_to_end(
                    raw: dict[str, np.ndarray] = raw_observation,
                    initial_noise: torch.Tensor = explicit.tensor,
                    reverse_seed_value: int = second_seed,
                ) -> object:
                    history = _freeze_history(
                        loaded.policy,
                        loaded.preprocessor,
                        raw,
                    )
                    sample = controlled_sample(
                        loaded.policy,
                        history,
                        initial_noise=initial_noise,
                        reverse_seed=reverse_seed_value,
                    )
                    return loaded.postprocessor(sample.actions)

                with torch.inference_mode():
                    model_summary, model_samples = benchmark_latency(
                        model_only,
                        device=device,
                        warmup_iterations=latency_warmup_iterations,
                        measurement_iterations=latency_iterations,
                    )
                    end_summary, end_samples = benchmark_latency(
                        end_to_end,
                        device=device,
                        warmup_iterations=latency_warmup_iterations,
                        measurement_iterations=latency_iterations,
                    )
                for kind, samples in (
                    ("model_only", model_samples),
                    ("end_to_end", end_samples),
                ):
                    latency_rows.extend(
                        {
                            "experiment_id": "E4",
                            "condition_id": condition_id,
                            "scheduler": scheduler,
                            "num_inference_steps": steps,
                            "latency_kind": kind,
                            "iteration": index,
                            "latency_ms": float(value),
                            "checkpoint_id": checkpoint_id,
                            "environment_seed": None,
                            "policy_seed": policy_seed,
                            "replan_index": None,
                            "initial_noise_seed": first_seed,
                            "reverse_noise_seed": second_seed,
                            "num_action_samples": 1,
                            "num_rollouts": total_rollouts,
                            "dataset_revision": DATASET_REVISION,
                            "code_commit": code_commit,
                        }
                        for index, value in enumerate(samples)
                    )
                interval = wilson_interval(
                    sum(result.success for result in condition_results),
                    len(condition_results),
                )
                condition_rows.append(
                    {
                        "condition_id": condition_id,
                        "scheduler": scheduler,
                        "num_inference_steps": steps,
                        "num_rollouts": len(condition_results),
                        "success_rate": interval.estimate,
                        "success_ci95_lower": interval.lower,
                        "success_ci95_upper": interval.upper,
                        "mean_max_reward": float(
                            np.mean([result.max_reward for result in condition_results])
                        ),
                        **{
                            f"model_only_{key}": value
                            for key, value in model_summary.to_dict().items()
                        },
                        **{
                            f"end_to_end_{key}": value
                            for key, value in end_summary.to_dict().items()
                        },
                        "amortized_end_to_end_p95_ms": (
                            end_summary.p95_ms / loaded.config.n_action_steps
                        ),
                    }
                )
        finally:
            recorder.close()

        registered_reference = _E4_CONFIG["reference_condition"]
        reference = next(
            (
                row
                for row in condition_rows
                if row["scheduler"] == str(registered_reference["scheduler"]).upper()
                and row["num_inference_steps"] == int(registered_reference["num_inference_steps"])
            ),
            None,
        )
        if reference is None:
            raise ValueError("E4 conditions do not include the registered reference")
        qualifying: list[dict[str, Any]] = []
        for row in condition_rows:
            row["p95_speedup_vs_ddpm100"] = float(reference["end_to_end_p95_ms"]) / float(
                row["end_to_end_p95_ms"]
            )
            row["success_drop_vs_ddpm100"] = float(reference["success_rate"]) - float(
                row["success_rate"]
            )
            row["meets_target"] = (
                row["p95_speedup_vs_ddpm100"] >= 2.0 and row["success_drop_vs_ddpm100"] <= 0.05
            )
            if row["meets_target"]:
                qualifying.append(row)

        write_records_parquet(temporary / "rollouts.parquet", rollout_rows)
        write_records_parquet(temporary / "replans.parquet", replan_rows)
        write_records_parquet(temporary / "latency.parquet", latency_rows)
        write_records_parquet(temporary / "condition_metrics.parquet", condition_rows)
        best = (
            min(qualifying, key=lambda row: float(row["end_to_end_p95_ms"])) if qualifying else None
        )
        metrics = {
            "reference_condition": _condition_id(
                str(registered_reference["scheduler"]).upper(),
                int(registered_reference["num_inference_steps"]),
            ),
            "reference_success_rate": reference["success_rate"],
            "reference_end_to_end_p95_ms": reference["end_to_end_p95_ms"],
            "num_conditions": len(condition_rows),
            "num_rollouts": total_rollouts,
            "num_qualifying_conditions": len(qualifying),
            "best_qualifying_condition": None if best is None else best["condition_id"],
        }
        write_metrics(
            temporary,
            metrics,
            comments={
                "reference_condition": "High-quality pre-registered comparison",
                "reference_success_rate": "Closed-loop success across paired final seeds",
                "reference_end_to_end_p95_ms": "Raw observation to denormalized 8-action chunk",
                "num_conditions": "Distinct scheduler and denoising-step settings",
                "num_rollouts": "Actual independent environment episodes",
                "num_qualifying_conditions": "At least 2x faster and <=5 percentage-point drop",
                "best_qualifying_condition": "Lowest p95 among conditions meeting the criterion",
            },
        )
        write_json_atomic(
            temporary / "condition_metrics.json",
            {"conditions": condition_rows},
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "label": "Closed-loop",
                "experiment": "E4",
                "conditions": [
                    {"scheduler": scheduler, "num_inference_steps": steps}
                    for scheduler, steps in conditions
                ],
                "reference_condition": registered_reference,
                "ddim_eta": 0.0,
                "environment_seeds": list(environment_seeds),
                "policy_seed": policy_seed,
                "num_action_samples": 1,
                "num_rollouts": total_rollouts,
                "checkpoint_id": checkpoint_id,
                "parameter_hashes": parameter_hashes,
                "train_seed": train_seed,
                "dataset_revision": DATASET_REVISION,
                "latency": {
                    "batch_size": 1,
                    "warmup_iterations": latency_warmup_iterations,
                    "measurement_iterations": latency_iterations,
                    "cuda_synchronized": True,
                    "rendering": False,
                },
                "runtime": runtime,
            },
        )
        _plot_results(temporary, condition_rows, latency_rows)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    registered_conditions = _E4_CONFIG["conditions"]
    registered_ddim = ",".join(
        str(item["num_inference_steps"])
        for item in registered_conditions
        if str(item["scheduler"]).upper() == "DDIM"
    )
    registered_ddpm = ",".join(
        str(item["num_inference_steps"])
        for item in registered_conditions
        if str(item["scheduler"]).upper() == "DDPM"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-policy", required=True)
    parser.add_argument("--schedulers", default="ddim,ddpm")
    parser.add_argument("--ddim-steps", default=registered_ddim)
    parser.add_argument("--ddpm-steps", default=registered_ddpm)
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=int(_E4_CONFIG["num_environment_seeds"]),
    )
    parser.add_argument("--policy-seed", type=int, default=int(_E4_CONFIG["policy_seed"]))
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--latency-warmup-iterations",
        type=int,
        default=int(_E4_CONFIG["latency_warmup_iterations"]),
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=int(_E4_CONFIG["latency_measurement_iterations"]),
    )
    parser.add_argument(
        "--videos-per-condition",
        type=int,
        default=int(_E4_CONFIG["videos_per_condition"]),
    )
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e4_sampling_steps",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    conditions = _conditions(args.schedulers, args.ddim_steps, args.ddpm_steps)
    result = evaluate_sampling_steps(
        diffusion_policy_path=args.diffusion_policy,
        output_dir=args.output_dir.expanduser().resolve(),
        conditions=conditions,
        num_episodes=args.n_episodes,
        policy_seed=args.policy_seed,
        train_seed=args.train_seed,
        latency_warmup_iterations=args.latency_warmup_iterations,
        latency_iterations=args.latency_iterations,
        videos_per_condition=args.videos_per_condition,
        episode_length=args.episode_length,
        overwrite=args.overwrite,
    )
    print(f"E4 output: {result}")


if __name__ == "__main__":
    main()
