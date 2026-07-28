"""Select BCChunk and Diffusion checkpoints on development data and reset seeds."""

from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION, OBS_IMAGES

from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_dataset_root,
    default_output_root,
    practice_root,
)
from .dataset_split import load_split
from .noise import fixed_observation_noise_seed, fixed_observation_reverse_seed, make_initial_noise
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import controlled_sample, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rollout import run_rollout


def discover_checkpoints(run_root: str | Path) -> tuple[tuple[int, Path], ...]:
    root = Path(run_root).expanduser().resolve()
    checkpoints_root = root / "checkpoints"
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {checkpoints_root}")
    result = []
    for child in checkpoints_root.iterdir():
        policy_path = child / "pretrained_model"
        if child.name.isdigit() and (policy_path / "model.safetensors").is_file():
            result.append((int(child.name), policy_path))
    if not result:
        raise FileNotFoundError(f"No numeric LeRobot checkpoints found under {checkpoints_root}")
    return tuple(sorted(result))


def _frame_lookup(dataset: LeRobotDataset) -> dict[int, list[int]]:
    dataset._ensure_hf_dataset_loaded()
    result: dict[int, list[int]] = defaultdict(list)
    for index, raw_episode in enumerate(dataset.hf_dataset["episode_index"]):
        episode = int(raw_episode.item() if hasattr(raw_episode, "item") else raw_episode)
        result[episode].append(index)
    return dict(result)


def _batched_inputs(
    item: dict[str, Any],
    policy: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    result = {}
    for key in policy.config.input_features:
        value = item.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Dataset item {key!r} must be a tensor")
        result[key] = value.unsqueeze(0)
    return result


def _stack_images(
    policy: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = dict(batch)
    if policy.config.image_features:
        result[OBS_IMAGES] = torch.stack(
            [result[key] for key in policy.config.image_features],
            dim=-4,
        )
    return result


def _offline_error(
    loaded: Any,
    *,
    dataset_root: Path,
    development_episodes: Sequence[int],
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
        episodes=list(development_episodes),
        delta_timestamps=delta_timestamps,
        revision=DATASET_REVISION,
    )
    lookup = _frame_lookup(dataset)
    normalized_errors = []
    denormalized_errors = []
    for ordinal, episode in enumerate(development_episodes):
        indices = lookup[episode]
        item = dataset[indices[len(indices) // 2]]
        raw_batch = _batched_inputs(item, loaded.policy)
        raw_batch[ACTION] = item[ACTION].unsqueeze(0)
        processed = loaded.preprocessor(raw_batch)
        prepared = _stack_images(loaded.policy, processed)
        start = loaded.config.n_obs_steps - 1
        end = start + loaded.config.n_action_steps
        with torch.inference_mode():
            if isinstance(loaded.policy, BCChunkPolicy):
                normalized_prediction = loaded.policy.predict_full_action_horizon(prepared)[
                    :, start:end
                ]
            elif isinstance(loaded.policy, DiffusionPolicy):
                explicit = make_initial_noise(
                    seed=fixed_observation_noise_seed(
                        candidate_ordinal=ordinal,
                        sample_ordinal=0,
                    ),
                    batch_size=1,
                    device=get_device_from_parameters(loaded.policy),
                    dtype=get_dtype_from_parameters(loaded.policy),
                    horizon=loaded.config.horizon,
                    action_dim=loaded.config.action_feature.shape[0],
                )
                normalized_prediction = controlled_sample(
                    loaded.policy,
                    prepared,
                    initial_noise=explicit.tensor,
                    reverse_seed=fixed_observation_reverse_seed(candidate_ordinal=ordinal),
                ).actions
            else:
                raise TypeError(f"Unsupported policy type: {type(loaded.policy).__name__}")

        normalized_target = processed[ACTION][:, start:end]
        normalized_errors.append(
            float(torch.mean((normalized_prediction - normalized_target) ** 2).cpu())
        )
        prediction = loaded.postprocessor(normalized_prediction)
        target = loaded.postprocessor(normalized_target)
        denormalized_errors.append(float(torch.mean((prediction - target) ** 2).cpu()))
    return float(np.mean(normalized_errors)), float(np.mean(denormalized_errors))


def _evaluate_run(
    *,
    model_name: str,
    checkpoints: Sequence[tuple[int, Path]],
    dataset_root: Path,
    development_episodes: Sequence[int],
    environment_seeds: Sequence[int],
    train_seed: int,
    code_commit: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for step, path in checkpoints:
        loaded = load_policy_with_processors(path, local_files_only=True)
        if model_name == "bc_chunk" and not isinstance(loaded.policy, BCChunkPolicy):
            raise TypeError(f"{path} is not a BCChunk checkpoint")
        if model_name == "diffusion":
            if not isinstance(loaded.policy, DiffusionPolicy):
                raise TypeError(f"{path} is not a Diffusion checkpoint")
            replace_scheduler(
                loaded.policy,
                scheduler_name="DDPM",
                num_inference_steps=100,
            )
        normalized_mse, denormalized_mse = _offline_error(
            loaded,
            dataset_root=dataset_root,
            development_episodes=development_episodes,
        )
        rollout_successes = []
        max_rewards = []
        for environment_seed in environment_seeds:
            result = run_rollout(
                policy=loaded.policy,
                preprocessor=loaded.preprocessor,
                postprocessor=loaded.postprocessor,
                environment_seed=environment_seed,
                policy_seed=0,
                record_frames=False,
                condition_id=f"development_{model_name}_{step:06d}",
            )
            rollout_successes.append(result.success)
            max_rewards.append(result.max_reward)
        rows.append(
            {
                "model": model_name,
                "step": step,
                "checkpoint_path": str(path),
                "checkpoint_id": sha256_file(path / "model.safetensors"),
                "train_seed": train_seed,
                "development_episode_count": len(development_episodes),
                "development_environment_seed_count": len(environment_seeds),
                "offline_normalized_action_mse": normalized_mse,
                "offline_denormalized_action_mse": denormalized_mse,
                "closed_loop_success_rate": float(np.mean(rollout_successes)),
                "mean_max_reward": float(np.mean(max_rewards)),
                "num_rollouts": len(environment_seeds),
                "scheduler": "DDPM" if model_name == "diffusion" else None,
                "num_inference_steps": 100 if model_name == "diffusion" else None,
                "num_action_samples": 1,
                "dataset_revision": DATASET_REVISION,
                "code_commit": code_commit,
            }
        )
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _select(rows: Sequence[dict[str, Any]], model: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["model"] == model]
    if not candidates:
        raise ValueError(f"No checkpoint rows available for {model}")
    return min(
        candidates,
        key=lambda row: (
            -float(row["closed_loop_success_rate"]),
            -float(row["mean_max_reward"]),
            int(row["step"]),
        ),
    )


def _record_selection(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    recording = rr.RecordingStream(
        "robot_practice_pusht_checkpoint_selection",
        recording_id="pusht_development_checkpoints",
    )
    recording.save(path)
    try:
        for model in ("bc_chunk", "diffusion"):
            model_rows = sorted(
                (row for row in rows if row["model"] == model),
                key=lambda row: int(row["step"]),
            )
            for row in model_rows:
                recording.set_time("training_step", sequence=int(row["step"]))
                recording.log(
                    f"checkpoints/{model}/success_rate",
                    rr.Scalars(float(row["closed_loop_success_rate"])),
                )
                recording.log(
                    f"checkpoints/{model}/mean_max_reward",
                    rr.Scalars(float(row["mean_max_reward"])),
                )
                recording.log(
                    f"checkpoints/{model}/offline_action_mse",
                    rr.Scalars(float(row["offline_denormalized_action_mse"])),
                )
    finally:
        recording.disconnect()


def select_checkpoints(
    *,
    bc_run: Path,
    diffusion_run: Path,
    dataset_root: Path,
    output_dir: Path,
    num_development_seeds: int,
    train_seed: int,
    overwrite: bool,
) -> Path:
    split = load_split()
    if not 1 <= num_development_seeds <= len(split.development_environment_seeds):
        raise ValueError("num_development_seeds must fit the committed development seed list")
    environment_seeds = split.development_environment_seeds[:num_development_seeds]
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    rows = _evaluate_run(
        model_name="bc_chunk",
        checkpoints=discover_checkpoints(bc_run),
        dataset_root=dataset_root,
        development_episodes=split.development,
        environment_seeds=environment_seeds,
        train_seed=train_seed,
        code_commit=code_commit,
    )
    rows.extend(
        _evaluate_run(
            model_name="diffusion",
            checkpoints=discover_checkpoints(diffusion_run),
            dataset_root=dataset_root,
            development_episodes=split.development,
            environment_seeds=environment_seeds,
            train_seed=train_seed,
            code_commit=code_commit,
        )
    )
    selected = {model: _select(rows, model) for model in ("bc_chunk", "diffusion")}
    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        write_records_parquet(temporary / "selection.parquet", rows)
        write_json_atomic(temporary / "selected_checkpoints.json", selected)
        write_json_atomic(
            temporary / "rng_manifest.json",
            {
                "environment_seeds": list(environment_seeds),
                "policy_seed": 0,
                "diffusion_scheduler": "DDPM",
                "num_inference_steps": 100,
                "noise_protocol": "PROTOCOL_NAMESPACE/env_seed/policy_seed/replan_index/stream",
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "label": "Closed-loop",
                "experiment": "development_checkpoint_selection",
                "dataset_revision": DATASET_REVISION,
                "development_episodes": list(split.development),
                "development_environment_seeds": list(environment_seeds),
                "train_seed": train_seed,
                "runtime": runtime,
            },
        )
        write_metrics(
            temporary,
            {
                "selected_bc_step": selected["bc_chunk"]["step"],
                "selected_bc_success_rate": selected["bc_chunk"]["closed_loop_success_rate"],
                "selected_diffusion_step": selected["diffusion"]["step"],
                "selected_diffusion_success_rate": selected["diffusion"][
                    "closed_loop_success_rate"
                ],
                "num_development_environment_seeds": len(environment_seeds),
            },
            comments={
                "selected_bc_step": "Highest success, then reward, then earliest step",
                "selected_bc_success_rate": "Development closed-loop success",
                "selected_diffusion_step": "Highest success, then reward, then earliest step",
                "selected_diffusion_success_rate": "Development closed-loop success",
                "num_development_environment_seeds": "Seeds used only for checkpoint selection",
            },
        )
        _record_selection(temporary / "recording.rrd", rows)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--diffusion-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--num-development-seeds", type=int, default=20)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "checkpoint_selection",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = select_checkpoints(
        bc_run=args.bc_run.expanduser().resolve(),
        diffusion_run=args.diffusion_run.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        num_development_seeds=args.num_development_seeds,
        train_seed=args.train_seed,
        overwrite=args.overwrite,
    )
    print(f"Checkpoint selection output: {result}")


if __name__ == "__main__":
    main()
