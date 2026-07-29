"""E5 closed-loop prototype-controlled and NULL PushT rollouts."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
from lerobot.policies.utils import get_device_from_parameters
from tqdm.auto import tqdm

from robot_practice.policies.mode_bc_chunk.modeling_mode_bc_chunk import ModeBCChunkPolicy
from robot_practice.policies.mode_diffusion.modeling_mode_diffusion import ModeDiffusionPolicy

from .config import default_output_root, load_named_config
from .e0_geometry import extract_moving_block_pose
from .e5_dataset import PrototypeSidecar, load_prototype_sidecar
from .e5_prototypes import (
    EpisodeSketch,
    build_episode_sketch,
    episode_sketch_from_record,
    trajectory_distance,
)
from .e5_runtime import FixedPrototypePreprocessor
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rerun_visualization import record_single_rollout
from .rollout import FULL_DDPM_SAMPLING, RolloutResult, run_rollout

_CONFIG = load_named_config("e5_mode_conditioned.json")
_SCHEMA_VERSION = 1
_MIN_NULL_MODE_SUPPORT = int(_CONFIG["minimum_null_mode_support"])


@dataclass(frozen=True)
class PrototypeClassifier:
    prototypes: tuple[EpisodeSketch, ...]
    radii: tuple[float, ...]
    scales: Mapping[str, float]

    @property
    def num_prototypes(self) -> int:
        return len(self.prototypes)


@dataclass(frozen=True)
class WorkerConfig:
    worker_index: int
    device: str
    environment_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    bc_policy_path: str
    diffusion_policy_path: str
    prototype_manifest: str
    output_root: str
    episode_length: int
    num_inference_steps: int


def partition_environment_seeds(
    environment_seeds: Sequence[int],
    *,
    num_gpus: int,
) -> tuple[tuple[int, ...], ...]:
    seeds = tuple(int(value) for value in environment_seeds)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    if num_gpus > len(seeds):
        raise ValueError("num_gpus cannot exceed the number of environment seeds")
    return tuple(seeds[index::num_gpus] for index in range(num_gpus))


def _load_classifier(sidecar: PrototypeSidecar) -> PrototypeClassifier:
    sketch_path = sidecar.root / "prototype_sketches.json"
    expected_hash = sidecar.manifest["artifact_hashes"].get("prototype_sketches.json")
    actual_hash = sha256_file(sketch_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"E5 prototype sketch hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    with sketch_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    rows = sorted(payload["prototypes"], key=lambda row: int(row["prototype_id"]))
    expected_ids = list(range(sidecar.selected_k))
    observed_ids = [int(row["prototype_id"]) for row in rows]
    if observed_ids != expected_ids:
        raise ValueError(f"Prototype sketch IDs must be {expected_ids}, got {observed_ids}")
    return PrototypeClassifier(
        prototypes=tuple(episode_sketch_from_record(row["sketch"]) for row in rows),
        radii=tuple(float(row["rejection_radius"]) for row in rows),
        scales={
            str(name): float(value) for name, value in sidecar.manifest["component_scales"].items()
        },
    )


def _rollout_sketch(result: RolloutResult) -> EpisodeSketch:
    if len(result.frames) != len(result.agent_path):
        raise ValueError(
            "Prototype classification requires one rendered frame per measured path point"
        )
    records: list[dict[str, Any]] = []
    for frame_index, (frame, agent) in enumerate(
        zip(result.frames, result.agent_path, strict=True)
    ):
        pose = extract_moving_block_pose(frame, agent_position=agent)
        records.append(
            {
                "episode_index": result.environment_seed,
                "frame_index": frame_index,
                "split": "rollout",
                "agent_position": np.asarray(agent, dtype=np.float64).tolist(),
                "block_pose_valid": pose.valid,
                "block_position": list(pose.position_workspace),
                "block_angle_rad": pose.angle_rad,
                "agent_block_surface_distance": pose.agent_surface_distance,
            }
        )
    return build_episode_sketch(records)


def classify_rollout(
    result: RolloutResult,
    *,
    classifier: PrototypeClassifier,
    commanded_condition: int,
    null_prototype_id: int,
) -> dict[str, Any]:
    sketch = _rollout_sketch(result)
    nearest_id: int | None = None
    nearest_distance: float | None = None
    classifiable = False
    if sketch.eligible:
        distances = [
            trajectory_distance(sketch, prototype, scales=classifier.scales)
            for prototype in classifier.prototypes
        ]
        nearest_id = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_id])
        classifiable = nearest_distance <= classifier.radii[nearest_id]
    is_null = commanded_condition == null_prototype_id
    adherent = bool(not is_null and classifiable and nearest_id == commanded_condition)
    return {
        "sketch_eligible": sketch.eligible,
        "sketch_exclusion_reason": sketch.exclusion_reason,
        "nearest_prototype_id": nearest_id,
        "nearest_prototype_distance": nearest_distance,
        "nearest_prototype_radius": (
            classifier.radii[nearest_id] if nearest_id is not None else None
        ),
        "classifiable": classifiable,
        "prototype_adherent": adherent,
        "controlled_success": bool(result.success and adherent),
    }


def _rollout_row(
    result: RolloutResult,
    *,
    condition_id: int,
    condition_name: str,
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    latencies = [float(replan.latency_ms) for replan in result.replans]
    return {
        "schema_version": _SCHEMA_VERSION,
        "model": result.model,
        "environment_seed": result.environment_seed,
        "policy_seed": result.policy_seed,
        "condition_id": condition_id,
        "condition_name": condition_name,
        "is_null_condition": condition_name == "NULL",
        "success": result.success,
        "sum_reward": result.sum_reward,
        "max_reward": result.max_reward,
        "final_reward": result.final_reward,
        "episode_length": result.episode_length,
        "termination_reason": result.termination_reason,
        "num_replans": len(result.replans),
        "mean_replan_latency_ms": float(np.mean(latencies)) if latencies else None,
        "max_replan_latency_ms": max(latencies) if latencies else None,
        "agent_path": np.asarray(result.agent_path).tolist(),
        "block_path": np.asarray(result.block_path).tolist(),
        "action_targets": np.asarray(result.action_targets).tolist(),
        "rewards": np.asarray(result.rewards).tolist(),
        **classification,
    }


def _condition_name(condition_id: int, *, null_id: int) -> str:
    return "NULL" if condition_id == null_id else f"P{condition_id}"


def _recording_path(
    root: Path,
    *,
    model: str,
    condition_name: str,
    environment_seed: int,
    policy_seed: int,
) -> Path:
    base = root / "recordings" / model / f"condition_{condition_name}"
    if model == "mode_diffusion":
        base = base / f"policy_seed_{policy_seed:02d}"
    return base / f"env_{environment_seed:04d}.rrd"


def _run_and_record(
    *,
    loaded: Any,
    classifier: PrototypeClassifier,
    output_root: Path,
    environment_seed: int,
    policy_seed: int,
    condition_id: int,
    null_id: int,
    episode_length: int,
) -> dict[str, Any]:
    name = _condition_name(condition_id, null_id=null_id)
    preprocessor = FixedPrototypePreprocessor(
        loaded.preprocessor,
        prototype_id=condition_id,
        num_prototypes=classifier.num_prototypes,
        device=get_device_from_parameters(loaded.policy),
    )
    result = run_rollout(
        policy=loaded.policy,
        preprocessor=preprocessor,
        postprocessor=loaded.postprocessor,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        record_frames=True,
        episode_length=episode_length,
        condition_id=name,
        sampling_protocol=FULL_DDPM_SAMPLING,
    )
    classification = classify_rollout(
        result,
        classifier=classifier,
        commanded_condition=condition_id,
        null_prototype_id=null_id,
    )
    outcome = "success" if result.success else "failure"
    record_single_rollout(
        output_path=_recording_path(
            output_root,
            model=result.model,
            condition_name=name,
            environment_seed=environment_seed,
            policy_seed=policy_seed,
        ),
        rollout=result,
        recording_id=(
            f"e5_{result.model}_{name.lower()}_env_{environment_seed:04d}_seed_{policy_seed:02d}"
        ),
        recording_name=(
            f"E5 | {result.model} | {name} | env {environment_seed} | "
            f"seed {policy_seed:02d} | {outcome}"
        ),
    )
    return _rollout_row(
        result,
        condition_id=condition_id,
        condition_name=name,
        classification=classification,
    )


def _load_worker_policies(config: WorkerConfig, *, selected_k: int) -> tuple[Any, Any]:
    bc_loaded = load_policy_with_processors(
        config.bc_policy_path,
        local_files_only=True,
        device=config.device,
    )
    diffusion_loaded = load_policy_with_processors(
        config.diffusion_policy_path,
        local_files_only=True,
        device=config.device,
    )
    if not isinstance(bc_loaded.policy, ModeBCChunkPolicy):
        raise TypeError("bc-policy must contain policy.type=mode_bc_chunk")
    if not isinstance(diffusion_loaded.policy, ModeDiffusionPolicy):
        raise TypeError("diffusion-policy must contain policy.type=mode_diffusion")
    for loaded in (bc_loaded, diffusion_loaded):
        if int(loaded.config.num_prototypes) != selected_k:
            raise ValueError("Checkpoint prototype count disagrees with manifest")
    replace_scheduler(
        diffusion_loaded.policy,
        scheduler_name=str(_CONFIG["scheduler"]),
        num_inference_steps=config.num_inference_steps,
    )
    return bc_loaded, diffusion_loaded


def _worker(config: WorkerConfig, queue: Any) -> None:
    try:
        sidecar = load_prototype_sidecar(config.prototype_manifest)
        classifier = _load_classifier(sidecar)
        bc_loaded, diffusion_loaded = _load_worker_policies(
            config,
            selected_k=sidecar.selected_k,
        )
        conditions = tuple(range(sidecar.selected_k + 1))
        output_root = Path(config.output_root)
        for environment_seed in config.environment_seeds:
            for condition_id in conditions:
                bc_row = _run_and_record(
                    loaded=bc_loaded,
                    classifier=classifier,
                    output_root=output_root,
                    environment_seed=environment_seed,
                    policy_seed=0,
                    condition_id=condition_id,
                    null_id=sidecar.null_prototype_id,
                    episode_length=config.episode_length,
                )
                queue.put(("result", config.worker_index, bc_row))
                for policy_seed in config.policy_seeds:
                    diffusion_row = _run_and_record(
                        loaded=diffusion_loaded,
                        classifier=classifier,
                        output_root=output_root,
                        environment_seed=environment_seed,
                        policy_seed=policy_seed,
                        condition_id=condition_id,
                        null_id=sidecar.null_prototype_id,
                        episode_length=config.episode_length,
                    )
                    queue.put(("result", config.worker_index, diffusion_row))
        queue.put(("done", config.worker_index, None))
    except Exception:  # noqa: BLE001 - worker must report every failure to its parent
        queue.put(("error", config.worker_index, traceback.format_exc()))


def _validate_num_gpus(*, num_gpus: int, num_environment_seeds: int) -> None:
    partition_environment_seeds(range(num_environment_seeds), num_gpus=num_gpus)
    if num_gpus == 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU E5 evaluation requires CUDA")
    if num_gpus > torch.cuda.device_count():
        raise RuntimeError(
            f"num_gpus={num_gpus}, but only {torch.cuda.device_count()} CUDA devices are visible"
        )


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_k: int,
    environment_seeds: Sequence[int],
    policy_seeds: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows: list[dict[str, Any]] = []
    for model in ("mode_bc_chunk", "mode_diffusion"):
        model_rows = [row for row in rows if row["model"] == model]
        for condition_id in range(selected_k + 1):
            selected = [row for row in model_rows if int(row["condition_id"]) == condition_id]
            strict_eligible = [row for row in selected if bool(row["classifiable"])]
            is_null = condition_id == selected_k
            group_rows.append(
                {
                    "model": model,
                    "condition_id": condition_id,
                    "condition_name": _condition_name(condition_id, null_id=selected_k),
                    "num_rollouts": len(selected),
                    "success_rate": float(np.mean([row["success"] for row in selected])),
                    "classifiable_rate": float(np.mean([row["classifiable"] for row in selected])),
                    "prototype_adherence_rate": (
                        None
                        if is_null
                        else float(np.mean([row["prototype_adherent"] for row in selected]))
                    ),
                    "conditional_adherence_rate": (
                        None
                        if is_null or not strict_eligible
                        else float(np.mean([row["prototype_adherent"] for row in strict_eligible]))
                    ),
                    "controlled_success_rate": (
                        None
                        if is_null
                        else float(np.mean([row["controlled_success"] for row in selected]))
                    ),
                    "mean_replan_latency_ms": _mean_non_null(
                        row["mean_replan_latency_ms"] for row in selected
                    ),
                }
            )

    null_rows = [
        row
        for row in rows
        if row["model"] == "mode_diffusion" and int(row["condition_id"]) == selected_k
    ]
    null_mode_rows: list[dict[str, Any]] = []
    for environment_seed in environment_seeds:
        selected = [row for row in null_rows if int(row["environment_seed"]) == environment_seed]
        counts = Counter(
            int(row["nearest_prototype_id"])
            for row in selected
            if bool(row["success"])
            and bool(row["classifiable"])
            and row["nearest_prototype_id"] is not None
        )
        supported = [
            prototype_id
            for prototype_id in range(selected_k)
            if counts[prototype_id] >= _MIN_NULL_MODE_SUPPORT
        ]
        null_mode_rows.append(
            {
                "environment_seed": environment_seed,
                "successful_classifiable_count_by_prototype": [
                    counts[prototype_id] for prototype_id in range(selected_k)
                ],
                "supported_prototype_ids": supported,
                "num_supported_modes": len(supported),
                "multi_mode_success": len(supported) >= 2,
            }
        )

    conditional_dp = [
        row
        for row in rows
        if row["model"] == "mode_diffusion" and int(row["condition_id"]) < selected_k
    ]
    feasible_counts = []
    for environment_seed in environment_seeds:
        feasible = 0
        for condition_id in range(selected_k):
            if any(
                bool(row["controlled_success"])
                for row in conditional_dp
                if int(row["environment_seed"]) == environment_seed
                and int(row["condition_id"]) == condition_id
            ):
                feasible += 1
        feasible_counts.append(feasible)

    summary = {
        "num_prototypes": selected_k,
        "num_environment_seeds": len(environment_seeds),
        "num_policy_seeds": len(policy_seeds),
        "num_rollouts": len(rows),
        "mode_diffusion_rollouts": sum(row["model"] == "mode_diffusion" for row in rows),
        "mode_bc_chunk_rollouts": sum(row["model"] == "mode_bc_chunk" for row in rows),
        "mean_feasible_successful_prototypes_per_environment": float(np.mean(feasible_counts)),
        "environments_with_all_prototypes_controlled_success": sum(
            value == selected_k for value in feasible_counts
        ),
        "null_environments_with_multi_mode_success": sum(
            bool(row["multi_mode_success"]) for row in null_mode_rows
        ),
    }
    return summary, group_rows, null_mode_rows


def _mean_non_null(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def evaluate_e5_rollouts(
    *,
    bc_policy_path: str | Path,
    diffusion_policy_path: str | Path,
    prototype_manifest: str | Path,
    output_dir: str | Path,
    num_environment_seeds: int,
    num_policy_seeds: int,
    num_gpus: int,
    episode_length: int,
    num_inference_steps: int,
    label: str,
    overwrite: bool,
) -> Path:
    formal_environments = tuple(int(value) for value in _CONFIG["closed_loop_environment_seeds"])
    formal_policies = tuple(int(value) for value in _CONFIG["closed_loop_policy_seeds"])
    if label not in {"Smoke", "Closed-loop"}:
        raise ValueError("label must be Smoke or Closed-loop")
    if not 1 <= num_environment_seeds <= len(formal_environments):
        raise ValueError("num_environment_seeds is outside the registered E5 range")
    if not 1 <= num_policy_seeds <= len(formal_policies):
        raise ValueError("num_policy_seeds is outside the registered E5 range")
    if label == "Closed-loop" and (
        num_environment_seeds != len(formal_environments)
        or num_policy_seeds != len(formal_policies)
        or episode_length != int(_CONFIG["episode_length"])
        or num_inference_steps != int(_CONFIG["num_inference_steps"])
    ):
        raise ValueError(
            "Closed-loop E5 requires 20 environments × 10 policy seeds × 300 steps × DDPM100"
        )
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    _validate_num_gpus(
        num_gpus=num_gpus,
        num_environment_seeds=num_environment_seeds,
    )
    environments = formal_environments[:num_environment_seeds]
    policies = formal_policies[:num_policy_seeds]
    sidecar = load_prototype_sidecar(prototype_manifest)
    _load_classifier(sidecar)
    partitions = partition_environment_seeds(environments, num_gpus=num_gpus)
    expected = len(environments) * (sidecar.selected_k + 1) * (len(policies) + 1)
    destination = Path(output_dir).expanduser().resolve()
    with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
        context = mp.get_context("spawn")
        queue = context.Queue()
        workers = []
        for worker_index, partition in enumerate(partitions):
            device = f"cuda:{worker_index}" if torch.cuda.is_available() else "cpu"
            config = WorkerConfig(
                worker_index=worker_index,
                device=device,
                environment_seeds=partition,
                policy_seeds=policies,
                bc_policy_path=str(Path(bc_policy_path).expanduser().resolve()),
                diffusion_policy_path=str(Path(diffusion_policy_path).expanduser().resolve()),
                prototype_manifest=str(sidecar.manifest_path),
                output_root=str(temporary),
                episode_length=episode_length,
                num_inference_steps=num_inference_steps,
            )
            process = context.Process(target=_worker, args=(config, queue))
            process.start()
            workers.append(process)

        rows: list[dict[str, Any]] = []
        finished: set[int] = set()
        error: str | None = None
        with tqdm(total=expected, desc="E5 closed-loop rollout progress", unit="rollout") as bar:
            while len(finished) < len(workers) and error is None:
                try:
                    kind, worker_index, payload = queue.get(timeout=5.0)
                except Empty:
                    failed = [
                        process.exitcode for process in workers if process.exitcode not in (None, 0)
                    ]
                    if failed:
                        error = f"E5 rollout worker exited unexpectedly: {failed}"
                    continue
                if kind == "result":
                    rows.append(payload)
                    bar.update(1)
                elif kind == "done":
                    finished.add(int(worker_index))
                elif kind == "error":
                    error = str(payload)
                else:
                    error = f"Unknown worker event: {kind}"
        if error is not None:
            for process in workers:
                if process.is_alive():
                    process.terminate()
        for process in workers:
            process.join(timeout=30)
        if error is not None:
            raise RuntimeError(error)
        if any(process.exitcode != 0 for process in workers):
            raise RuntimeError(
                f"E5 rollout workers did not exit cleanly: "
                f"{[process.exitcode for process in workers]}"
            )
        if len(rows) != expected:
            raise RuntimeError(f"E5 produced {len(rows)} rollouts, expected {expected}")

        summary, group_rows, null_rows = _aggregate(
            rows,
            selected_k=sidecar.selected_k,
            environment_seeds=environments,
            policy_seeds=policies,
        )
        write_records_parquet(temporary / "rollouts.parquet", rows)
        write_records_parquet(temporary / "condition_metrics.parquet", group_rows)
        write_records_parquet(temporary / "null_mode_support.parquet", null_rows)
        write_metrics(temporary, summary)
        write_json_atomic(
            temporary / "manifest.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "experiment_schema": _CONFIG["experiment_schema"],
                "experiment": "E5 closed-loop prototype control",
                "status": "complete",
                "label": label,
                "environment_seeds": list(environments),
                "policy_seeds": list(policies),
                "episode_length": episode_length,
                "scheduler": str(_CONFIG["scheduler"]),
                "num_inference_steps": num_inference_steps,
                "num_gpus": num_gpus,
                "prototype_manifest_sha256": sha256_file(sidecar.manifest_path),
                "checkpoint_sha256": {
                    "mode_bc_chunk": sha256_file(
                        Path(bc_policy_path).expanduser().resolve() / "model.safetensors"
                    ),
                    "mode_diffusion": sha256_file(
                        Path(diffusion_policy_path).expanduser().resolve() / "model.safetensors"
                    ),
                },
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
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e5_mode_conditioned" / "closed_loop",
    )
    parser.add_argument("--num-environment-seeds", type=int, default=20)
    parser.add_argument("--num-policy-seeds", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--label", choices=("Smoke", "Closed-loop"), default="Closed-loop")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output = evaluate_e5_rollouts(
        bc_policy_path=args.bc_policy,
        diffusion_policy_path=args.diffusion_policy,
        prototype_manifest=args.prototype_manifest,
        output_dir=args.output_dir,
        num_environment_seeds=args.num_environment_seeds,
        num_policy_seeds=args.num_policy_seeds,
        num_gpus=args.num_gpus,
        episode_length=args.episode_length,
        num_inference_steps=args.num_inference_steps,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(f"E5 closed-loop output: {output}")


if __name__ == "__main__":
    main()


__all__ = [
    "PrototypeClassifier",
    "classify_rollout",
    "evaluate_e5_rollouts",
    "partition_environment_seeds",
]
