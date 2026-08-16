"""Merge disjoint multi-GPU ManiSkill evaluation workers offline.

The workers deliberately reset the complete frozen state batch on every GPU,
then retain only their policy shard.  This module merges their episode rows
without creating an environment or loading a policy.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

from .eval_closed_loop import summarize_rows


def _portable(path: str | Path) -> str:
    value = Path(path).expanduser().resolve()
    try:
        return value.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _write_episodes_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_latency(payloads: list[dict[str, Any]], key: str) -> dict[str, Any]:
    summaries = [payload.get("latency", {}).get(key, {}) for payload in payloads]
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    weighted_mean = sum(
        int(summary.get("count", 0)) * float(summary["mean_ms"])
        for summary in summaries
        if summary.get("mean_ms") is not None
    )
    mean = weighted_mean / count if count else None
    return {
        "count": count,
        "mean_ms": mean,
        "p50_ms": None,
        "p95_ms": None,
        "aggregation_note": "mean is count-weighted; percentiles are unavailable after worker-level aggregation",
    }


def _worker_index(path: Path) -> int:
    try:
        return int(path.name.removeprefix("gpu_"))
    except ValueError as exc:
        raise ValueError(f"Worker directory must be named gpu_N: {path}") from exc


def merge_workers(
    *,
    workers_root: str | Path,
    output_dir: str | Path,
    expected_episodes: int,
    expected_workers: int,
) -> dict[str, Any]:
    """Validate and merge worker outputs into one normal evaluation directory."""

    if expected_episodes <= 0 or expected_workers <= 0:
        raise ValueError("expected_episodes and expected_workers must be positive")
    workers_root = Path(workers_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    worker_dirs = sorted(
        (path for path in workers_root.glob("gpu_*") if path.is_dir()),
        key=_worker_index,
    )
    if len(worker_dirs) != expected_workers:
        raise FileNotFoundError(
            f"Expected {expected_workers} worker directories under {workers_root}, "
            f"found {len(worker_dirs)}"
        )

    payloads: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_worker_indices: set[int] = set()
    for worker_dir in worker_dirs:
        metrics_path = worker_dir / "metrics.json"
        manifest_path = worker_dir / "run_manifest.json"
        episodes_path = worker_dir / "episodes.parquet"
        metrics = _read_json(metrics_path)
        manifest = _read_json(manifest_path)
        worker = metrics.get("worker")
        if not isinstance(worker, Mapping):
            raise TypeError(f"{metrics_path} is missing worker metadata")
        index = int(worker.get("index", -1))
        count = int(worker.get("count", -1))
        if index in seen_worker_indices or count != expected_workers:
            raise ValueError(f"Invalid or duplicate worker metadata in {metrics_path}")
        if index != _worker_index(worker_dir):
            raise ValueError(f"Worker directory and metadata disagree: {worker_dir}")
        seen_worker_indices.add(index)
        if metrics.get("num_episodes") != metrics.get("evaluation", {}).get("num_episodes"):
            raise ValueError(f"Episode count mismatch in {metrics_path}")
        for key in (
            "system",
            "task",
            "control_mode",
            "seed_set",
            "reset_batches",
            "data_provenance",
            "required_rollout_backend",
            "actual_rollout_backend",
        ):
            if payloads and metrics.get(key) != payloads[0].get(key):
                raise ValueError(f"Worker metrics disagree on {key}: {metrics_path}")
        if manifests:
            for key in (
                "policy_path",
                "data_provenance",
                "required_rollout_backend",
                "actual_rollout_backend",
            ):
                if manifest.get(key) != manifests[0].get(key):
                    raise ValueError(f"Worker manifests disagree on {key}: {manifest_path}")
        import pyarrow.parquet as pq

        worker_rows = pq.read_table(episodes_path).to_pylist()
        if len(worker_rows) != int(metrics["num_episodes"]):
            raise ValueError(f"Episode parquet/count mismatch in {worker_dir}")
        payloads.append(metrics)
        manifests.append(manifest)
        rows.extend(dict(row) for row in worker_rows)

    if seen_worker_indices != set(range(expected_workers)):
        raise ValueError(f"Worker indices are not complete: {sorted(seen_worker_indices)}")
    if len(rows) != expected_episodes:
        raise ValueError(f"Merged episode count is {len(rows)}, expected {expected_episodes}")
    rows.sort(key=lambda row: int(row["episode_index"]))
    episode_indices = [int(row["episode_index"]) for row in rows]
    if episode_indices != list(range(expected_episodes)):
        raise ValueError("Worker episode indices do not form the complete frozen evaluation set")

    base = payloads[0]
    base_manifest = manifests[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(base)
    metrics["num_episodes"] = expected_episodes
    metrics["evaluation"] = summarize_rows(rows)
    metrics["latency"] = {
        "policy_forward": _merge_latency(payloads, "policy_forward"),
        "environment_step": _merge_latency(payloads, "environment_step"),
    }
    metrics["worker"] = {
        "count": expected_workers,
        "state_batch_strategy": "replicated_full_reset_batch_policy_shard",
        "worker_metrics": [_portable(worker_dir / "metrics.json") for worker_dir in worker_dirs],
    }
    metrics["execution"] = {
        "parallelism": "multi_gpu_worker_pool",
        "num_gpus": expected_workers,
        "worker_outputs": _portable(workers_root),
    }

    manifest = dict(base_manifest)
    manifest["num_episodes"] = expected_episodes
    manifest["worker"] = {
        "count": expected_workers,
        "state_batch_strategy": "replicated_full_reset_batch_policy_shard",
        "worker_manifests": [
            _portable(worker_dir / "run_manifest.json") for worker_dir in worker_dirs
        ],
    }
    manifest["execution"] = {
        "parallelism": "multi_gpu_worker_pool",
        "num_gpus": expected_workers,
        "worker_outputs": _portable(workers_root),
    }
    manifest["created_at"] = iso_timestamp()

    write_json_atomic(output_dir / "metrics.json", metrics)
    _write_episodes_parquet(output_dir / "episodes.parquet", rows)
    write_json_atomic(output_dir / "run_manifest.json", manifest)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = merge_workers(
        workers_root=args.workers_root,
        output_dir=args.output_dir,
        expected_episodes=args.expected_episodes,
        expected_workers=args.expected_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
