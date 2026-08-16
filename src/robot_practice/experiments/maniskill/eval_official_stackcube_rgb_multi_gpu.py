"""Evaluate one official StackCube RGB checkpoint across multiple GPUs.

Each worker owns one physical GPU and one CPU ManiSkill vector environment.
Workers use disjoint deterministic reset seeds, then the parent merges the
official per-episode metrics.  This is an external evaluator; it does not
modify the vendored official training/evaluation implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .rollout_contract import OFFICIAL_DATA_PROVENANCE, require_rollout_backend

DATA_PROVENANCE = OFFICIAL_DATA_PROVENANCE


def _parse_gpus(value: str) -> list[int]:
    gpus = []
    for token in value.split(","):
        token = token.strip()
        if not token or not token.isdigit():
            raise ValueError("--gpus must be a comma-separated list of non-negative integers")
        gpu = int(token)
        if gpu in gpus:
            raise ValueError(f"duplicate GPU id: {gpu}")
        gpus.append(gpu)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    return gpus


def _partition(total: int, count: int) -> list[int]:
    if total < count:
        raise ValueError("--num-episodes must be at least the number of GPUs")
    base, remainder = divmod(total, count)
    return [base + int(index < remainder) for index in range(count)]


def _run_workers(
    *,
    official_repo: Path,
    checkpoint: Path,
    output_dir: Path,
    gpus: list[int],
    num_episodes: int,
    max_episode_steps: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    require_rollout_backend(DATA_PROVENANCE, "physx_cpu")
    counts = _partition(num_episodes, len(gpus))
    worker_root = output_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, int, Path, Any]] = []
    episode_offset = 0

    for worker_index, (gpu, worker_episodes) in enumerate(zip(gpus, counts)):
        worker_dir = worker_root / f"gpu_{gpu}"
        metrics_path = worker_dir / "metrics.json"
        log_path = worker_dir / "run.log"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_seed = seed_start + episode_offset
        episode_offset += worker_episodes

        command = [
            sys.executable,
            "-m",
            "robot_practice.experiments.maniskill.official_stackcube_rgb_verify",
            "--official-repo",
            str(official_repo),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(worker_dir),
            "--num-episodes",
            str(worker_episodes),
            "--num-envs",
            "1",
            "--max-episode-steps",
            str(max_episode_steps),
            "--device",
            "cuda",
            "--seed",
            str(worker_seed),
            "--env-seed",
            str(worker_seed),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTHONUNBUFFERED"] = "1"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(Path.cwd()),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu, worker_episodes, metrics_path, log_handle))
        processes[-1] = (*processes[-1], process)
        print(
            f"[official-eval] started worker={worker_index} gpu={gpu} "
            f"episodes={worker_episodes} seed_start={worker_seed} log={log_path}",
            flush=True,
        )

    failures: list[str] = []
    try:
        for gpu, worker_episodes, metrics_path, log_handle, process in processes:
            status = process.wait()
            log_handle.close()
            if status != 0:
                log_path = metrics_path.parent / "run.log"
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = "\n".join(lines[-40:])
                except OSError as exc:
                    tail = f"<unable to read worker log: {exc}>"
                failures.append(
                    f"gpu={gpu} status={status} log={log_path}\n--- worker tail ---\n{tail}"
                )
    except BaseException:
        for _gpu, _episodes, _metrics, log_handle, process in processes:
            if process.poll() is None:
                process.terminate()
            log_handle.close()
        raise

    if failures:
        raise RuntimeError("official evaluation worker failed: " + "; ".join(failures))

    return [json.loads(metrics_path.read_text(encoding="utf-8")) for _gpu, _episodes, metrics_path, _handle, _process in processes]


def _merge_worker_results(
    *,
    checkpoint: Path,
    output_dir: Path,
    gpus: list[int],
    seed_start: int,
    worker_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(worker_results) != len(gpus):
        raise RuntimeError(
            "worker result count does not match GPU count: "
            f"results={len(worker_results)} gpus={len(gpus)}"
        )
    first = worker_results[0]
    for worker_index, worker_result in enumerate(worker_results):
        for key in ("data_provenance", "required_rollout_backend", "actual_rollout_backend"):
            if worker_result.get(key) != first.get(key):
                raise RuntimeError(
                    f"official worker {worker_index} disagrees on rollout contract field {key}"
                )
    expected_seed = seed_start
    environment_seed_ranges: list[dict[str, int]] = []
    for index, (gpu, worker_result) in enumerate(zip(gpus, worker_results)):
        worker_episodes = int(worker_result["num_episodes"])
        if worker_episodes <= 0:
            raise RuntimeError(
                f"worker={index} gpu={gpu} reported invalid episode count: {worker_episodes}"
            )
        actual_seed = worker_result.get("official_protocol", {}).get("environment_seed")
        if actual_seed != expected_seed:
            raise RuntimeError(
                "worker environment seed assignment mismatch: "
                f"worker={index} gpu={gpu} expected={expected_seed} actual={actual_seed}"
            )
        environment_seed_ranges.append(
            {
                "worker_index": index,
                "gpu": gpu,
                "start": expected_seed,
                "end": expected_seed + worker_episodes - 1,
                "num_episodes": worker_episodes,
            }
        )
        expected_seed += worker_episodes

    expected_episodes = sum(item["num_episodes"] for item in worker_results)
    if expected_episodes <= 0:
        raise RuntimeError("worker results contain no episodes")

    merged_evaluation: dict[str, dict[str, Any]] = {}
    metric_keys = sorted(
        {
            key
            for result in worker_results
            for key in result.get("evaluation", {})
        }
    )
    for key in metric_keys:
        values = [
            value
            for result in worker_results
            for value in result.get("evaluation", {}).get(key, {}).get("values", [])
        ]
        merged_evaluation[key] = {
            "values": values,
            "num_episodes": len(values),
            "mean": sum(values) / len(values) if values else None,
        }

    result = {
        "schema_version": 2,
        "label": "Official ManiSkill Diffusion Policy RGB verification (multi-GPU)",
        "task": first["task"],
        "control_mode": first["control_mode"],
        "obs_mode": first["obs_mode"],
        "sim_backend": first["sim_backend"],
        "data_provenance": first["data_provenance"],
        "required_rollout_backend": first["required_rollout_backend"],
        "actual_rollout_backend": first["actual_rollout_backend"],
        "num_episodes": sum(item["num_episodes"] for item in worker_results),
        "num_envs_per_worker": 1,
        "max_episode_steps": first["max_episode_steps"],
        "policy_device": "one logical cuda:0 per worker",
        "evaluation": merged_evaluation,
        "primary_metric": "success_once",
        "parallelism": {
            "type": "one_process_per_physical_gpu",
            "physical_gpus": gpus,
            "worker_count": len(gpus),
            # Keep this legacy field for consumers of schema_version=2.
            "seed_start": seed_start,
            "environment_seed_start": seed_start,
            "environment_seed_assignment": (
                "contiguous disjoint per-worker ranges; episode j uses start+j"
            ),
            "environment_seed_ranges": environment_seed_ranges,
        },
        "checkpoint": first["checkpoint"],
        "official_repo": first["official_repo"],
        "workers": [
            {
                "gpu": gpu,
                "metrics": str(output_dir / "workers" / f"gpu_{gpu}" / "metrics.json"),
                "run_log": str(output_dir / "workers" / f"gpu_{gpu}" / "run.log"),
                "num_episodes": worker_results[index]["num_episodes"],
            }
            for index, gpu in enumerate(gpus)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_episodes <= 0 or args.max_episode_steps <= 0 or args.seed_start < 0:
        raise SystemExit("episode count, horizon, and seed-start must be valid")
    official_repo = args.official_repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    if (output_dir / "metrics.json").exists() and not args.overwrite:
        raise SystemExit(f"output already exists; use --overwrite: {output_dir}")
    gpus = _parse_gpus(args.gpus)
    worker_results = _run_workers(
        official_repo=official_repo,
        checkpoint=checkpoint,
        output_dir=output_dir,
        gpus=gpus,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        seed_start=args.seed_start,
    )
    result = _merge_worker_results(
        checkpoint=checkpoint,
        output_dir=output_dir,
        gpus=gpus,
        seed_start=args.seed_start,
        worker_results=worker_results,
    )
    summary = {
        key: value["mean"]
        for key, value in result["evaluation"].items()
        if isinstance(value, dict)
    }
    print(json.dumps({"primary_metric": "success_once", "evaluation": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
