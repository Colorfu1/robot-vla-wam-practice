"""Select BCChunk and Diffusion checkpoints on development data and reset seeds."""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import rerun as rr
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION, OBS_IMAGES
from tqdm import tqdm

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
from .e5_runtime import add_prototype_id
from .noise import fixed_observation_noise_seed, fixed_observation_reverse_seed, make_initial_noise
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import controlled_sample, load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rollout import run_rollout

CheckpointEventCallback = Callable[[str, int | None], None]


@dataclass(frozen=True)
class CheckpointTask:
    model_name: str
    step: int
    path: Path


@dataclass(frozen=True)
class WorkerConfig:
    dataset_root: Path
    development_episodes: tuple[int, ...]
    environment_seeds: tuple[int, ...]
    train_seed: int
    code_commit: str | None
    evaluation_protocol: str = "standard"
    prototype_manifest: Path | None = None
    selected_k: int | None = None
    oracle_episodes: tuple[Any, ...] = ()


@dataclass(frozen=True)
class WorkerEvent:
    kind: str
    gpu_index: int
    model_name: str
    step: int
    phase: str | None = None
    environment_seed: int | None = None
    row: dict[str, Any] | None = None
    error: str | None = None


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
    oracle_prototype_ids: Mapping[int, int] | None = None,
    selected_k: int | None = None,
) -> tuple[float, float]:
    if (oracle_prototype_ids is None) != (selected_k is None):
        raise ValueError("oracle_prototype_ids and selected_k must be provided together")
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
        if oracle_prototype_ids is not None and selected_k is not None:
            processed = add_prototype_id(
                processed,
                prototype_id=oracle_prototype_ids[episode],
                num_prototypes=selected_k,
            )
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


def _evaluate_checkpoint(
    *,
    model_name: str,
    step: int,
    path: Path,
    dataset_root: Path,
    development_episodes: Sequence[int],
    environment_seeds: Sequence[int],
    train_seed: int,
    code_commit: str | None,
    device: str | None = None,
    event_callback: CheckpointEventCallback | None = None,
) -> dict[str, Any]:
    if event_callback is not None:
        event_callback("loading", None)
    loaded = load_policy_with_processors(
        path,
        local_files_only=True,
        device=device,
    )
    try:
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
        if event_callback is not None:
            event_callback("offline", None)
        normalized_mse, denormalized_mse = _offline_error(
            loaded,
            dataset_root=dataset_root,
            development_episodes=development_episodes,
        )
        rollout_successes = []
        max_rewards = []
        for environment_seed in environment_seeds:
            if event_callback is not None:
                event_callback("rollout", environment_seed)
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
            if event_callback is not None:
                event_callback("rollout_complete", environment_seed)
        return {
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
    finally:
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _checkpoint_summary(row: dict[str, Any]) -> str:
    return (
        f"{row['model']} checkpoint {int(row['step']):06d}: "
        f"success={float(row['closed_loop_success_rate']):.3f}, "
        f"mean_max_reward={float(row['mean_max_reward']):.3f}, "
        f"offline_mse={float(row['offline_denormalized_action_mse']):.6f}"
    )


def _evaluate_run(
    *,
    model_name: str,
    checkpoints: Sequence[tuple[int, Path]],
    dataset_root: Path,
    development_episodes: Sequence[int],
    environment_seeds: Sequence[int],
    train_seed: int,
    code_commit: str | None,
    progress: tqdm | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for step, path in checkpoints:

        def report_event(
            phase: str,
            environment_seed: int | None,
            *,
            current_step: int = step,
        ) -> None:
            if progress is None:
                return
            if phase == "rollout_complete":
                progress.update(1)
                return
            values: dict[str, Any] = {
                "model": model_name,
                "checkpoint": f"{current_step:06d}",
                "phase": phase,
            }
            if environment_seed is not None:
                values["environment_seed"] = environment_seed
            progress.set_postfix(**values, refresh=True)

        row = _evaluate_checkpoint(
            model_name=model_name,
            step=step,
            path=path,
            dataset_root=dataset_root,
            development_episodes=development_episodes,
            environment_seeds=environment_seeds,
            train_seed=train_seed,
            code_commit=code_commit,
            event_callback=report_event,
        )
        rows.append(row)
        if progress is not None:
            progress.write(_checkpoint_summary(row))
    return rows


def _checkpoint_tasks(
    bc_checkpoints: Sequence[tuple[int, Path]],
    diffusion_checkpoints: Sequence[tuple[int, Path]],
    *,
    bc_model_name: str = "bc_chunk",
    diffusion_model_name: str = "diffusion",
) -> tuple[CheckpointTask, ...]:
    # Diffusion checkpoints are slower, so enqueue them first for better load balancing.
    return tuple(
        [CheckpointTask(diffusion_model_name, step, path) for step, path in diffusion_checkpoints]
        + [CheckpointTask(bc_model_name, step, path) for step, path in bc_checkpoints]
    )


def _validate_num_gpus(num_gpus: int) -> None:
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")
    if num_gpus == 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("multi-GPU checkpoint selection requires CUDA")
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise RuntimeError(
            f"num_gpus={num_gpus} requested, but only {available} CUDA device(s) are visible"
        )


def _checkpoint_worker(
    gpu_index: int,
    task_queue: Any,
    event_queue: Any,
    config: WorkerConfig,
) -> None:
    torch.cuda.set_device(gpu_index)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    while True:
        task = task_queue.get()
        if task is None:
            return
        if not isinstance(task, CheckpointTask):
            event_queue.put(
                WorkerEvent(
                    kind="error",
                    gpu_index=gpu_index,
                    model_name="unknown",
                    step=-1,
                    error=f"Worker received invalid task: {task!r}",
                )
            )
            return

        def report_event(
            phase: str,
            environment_seed: int | None,
            *,
            current_task: CheckpointTask = task,
        ) -> None:
            event_queue.put(
                WorkerEvent(
                    kind="progress",
                    gpu_index=gpu_index,
                    model_name=current_task.model_name,
                    step=current_task.step,
                    phase=phase,
                    environment_seed=environment_seed,
                )
            )

        try:
            row = _evaluate_task(
                task=task,
                config=config,
                device=f"cuda:{gpu_index}",
                event_callback=report_event,
            )
        except Exception:  # noqa: BLE001 - relay the complete worker failure to the parent.
            event_queue.put(
                WorkerEvent(
                    kind="error",
                    gpu_index=gpu_index,
                    model_name=task.model_name,
                    step=task.step,
                    error=traceback.format_exc(),
                )
            )
            return
        event_queue.put(
            WorkerEvent(
                kind="result",
                gpu_index=gpu_index,
                model_name=task.model_name,
                step=task.step,
                row=row,
            )
        )


def _evaluate_task(
    *,
    task: CheckpointTask,
    config: WorkerConfig,
    device: str | None,
    event_callback: CheckpointEventCallback | None,
) -> dict[str, Any]:
    if config.evaluation_protocol == "standard":
        return _evaluate_checkpoint(
            model_name=task.model_name,
            step=task.step,
            path=task.path,
            dataset_root=config.dataset_root,
            development_episodes=config.development_episodes,
            environment_seeds=config.environment_seeds,
            train_seed=config.train_seed,
            code_commit=config.code_commit,
            device=device,
            event_callback=event_callback,
        )
    if config.evaluation_protocol == "e5_oracle_mode":
        if config.prototype_manifest is None or config.selected_k is None:
            raise ValueError("E5 worker config requires prototype_manifest and selected_k")
        from .e5_checkpoint_selection import evaluate_e5_checkpoint

        return evaluate_e5_checkpoint(
            model_name=task.model_name,
            step=task.step,
            path=task.path,
            dataset_root=config.dataset_root,
            oracle_episodes=config.oracle_episodes,
            prototype_manifest=config.prototype_manifest,
            selected_k=config.selected_k,
            train_seed=config.train_seed,
            code_commit=config.code_commit,
            device=device,
            event_callback=event_callback,
        )
    raise ValueError(f"Unsupported checkpoint evaluation protocol: {config.evaluation_protocol}")


def _evaluate_parallel(
    *,
    tasks: Sequence[CheckpointTask],
    config: WorkerConfig,
    num_gpus: int,
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    event_queue = context.Queue()
    worker_count = min(num_gpus, len(tasks))
    for task in tasks:
        task_queue.put(task)
    for _ in range(worker_count):
        task_queue.put(None)
    workers = [
        context.Process(
            target=_checkpoint_worker,
            args=(gpu_index, task_queue, event_queue, config),
            name=f"checkpoint-gpu-{gpu_index}",
        )
        for gpu_index in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    rows: list[dict[str, Any]] = []
    total_rollouts = len(tasks) * len(config.environment_seeds)
    try:
        with tqdm(
            total=total_rollouts,
            desc="Checkpoint selection",
            unit="rollout",
            dynamic_ncols=True,
        ) as progress:
            while len(rows) < len(tasks):
                try:
                    event = event_queue.get(timeout=1.0)
                except Empty:
                    failed = [worker for worker in workers if worker.exitcode not in (None, 0)]
                    if failed:
                        details = ", ".join(
                            f"{worker.name} exitcode={worker.exitcode}" for worker in failed
                        )
                        raise RuntimeError(f"Checkpoint worker exited unexpectedly: {details}")
                    if not any(worker.is_alive() for worker in workers):
                        raise RuntimeError("All checkpoint workers exited before producing results")
                    continue
                if not isinstance(event, WorkerEvent):
                    raise TypeError(f"Received invalid worker event: {event!r}")
                if event.kind == "error":
                    raise RuntimeError(
                        f"GPU {event.gpu_index} failed on {event.model_name} "
                        f"checkpoint {event.step:06d}:\n{event.error}"
                    )
                if event.kind == "progress":
                    if event.phase == "rollout_complete":
                        progress.update(1)
                        continue
                    values: dict[str, Any] = {
                        "gpu": event.gpu_index,
                        "model": event.model_name,
                        "checkpoint": f"{event.step:06d}",
                        "phase": event.phase,
                    }
                    if event.environment_seed is not None:
                        values["environment_seed"] = event.environment_seed
                    progress.set_postfix(**values, refresh=True)
                    continue
                if event.kind == "result":
                    if event.row is None:
                        raise RuntimeError("Checkpoint worker returned an empty result")
                    rows.append(event.row)
                    progress.write(_checkpoint_summary(event.row))
                    continue
                raise RuntimeError(f"Unknown worker event kind: {event.kind!r}")
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
        task_queue.close()
        event_queue.close()

    failed = [worker for worker in workers if worker.exitcode != 0]
    if failed:
        details = ", ".join(f"{worker.name} exitcode={worker.exitcode}" for worker in failed)
        raise RuntimeError(f"Checkpoint worker shutdown failed: {details}")
    model_order = {
        "bc_chunk": 0,
        "mode_bc_chunk": 0,
        "diffusion": 1,
        "mode_diffusion": 1,
    }
    return sorted(
        rows,
        key=lambda row: (model_order.get(str(row["model"]), 99), int(row["step"])),
    )


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


def _record_selection(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    models: Sequence[str] = ("bc_chunk", "diffusion"),
) -> None:
    recording = rr.RecordingStream(
        "robot_practice_pusht_checkpoint_selection",
        recording_id="pusht_development_checkpoints",
    )
    recording.save(path)
    try:
        for model in models:
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
    num_gpus: int = 1,
) -> Path:
    _validate_num_gpus(num_gpus)
    split = load_split()
    if not 1 <= num_development_seeds <= len(split.development_environment_seeds):
        raise ValueError("num_development_seeds must fit the committed development seed list")
    environment_seeds = split.development_environment_seeds[:num_development_seeds]
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    bc_checkpoints = discover_checkpoints(bc_run)
    diffusion_checkpoints = discover_checkpoints(diffusion_run)
    if num_gpus == 1:
        total_rollouts = (len(bc_checkpoints) + len(diffusion_checkpoints)) * len(environment_seeds)
        with tqdm(
            total=total_rollouts,
            desc="Checkpoint selection",
            unit="rollout",
            dynamic_ncols=True,
        ) as progress:
            rows = _evaluate_run(
                model_name="bc_chunk",
                checkpoints=bc_checkpoints,
                dataset_root=dataset_root,
                development_episodes=split.development,
                environment_seeds=environment_seeds,
                train_seed=train_seed,
                code_commit=code_commit,
                progress=progress,
            )
            rows.extend(
                _evaluate_run(
                    model_name="diffusion",
                    checkpoints=diffusion_checkpoints,
                    dataset_root=dataset_root,
                    development_episodes=split.development,
                    environment_seeds=environment_seeds,
                    train_seed=train_seed,
                    code_commit=code_commit,
                    progress=progress,
                )
            )
    else:
        tasks = _checkpoint_tasks(bc_checkpoints, diffusion_checkpoints)
        rows = _evaluate_parallel(
            tasks=tasks,
            config=WorkerConfig(
                dataset_root=dataset_root,
                development_episodes=tuple(split.development),
                environment_seeds=tuple(environment_seeds),
                train_seed=train_seed,
                code_commit=code_commit,
            ),
            num_gpus=num_gpus,
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
                "execution": {
                    "num_gpus": num_gpus,
                    "parallelism": ("checkpoint_process_pool" if num_gpus > 1 else "sequential"),
                },
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
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "checkpoint_selection",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.num_gpus < 1:
        parser.error("--num-gpus must be at least 1")
    result = select_checkpoints(
        bc_run=args.bc_run.expanduser().resolve(),
        diffusion_run=args.diffusion_run.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        num_development_seeds=args.num_development_seeds,
        train_seed=args.train_seed,
        overwrite=args.overwrite,
        num_gpus=args.num_gpus,
    )
    print(f"Checkpoint selection output: {result}")


if __name__ == "__main__":
    main()
