"""Re-derive corrected RF7 metrics from an existing held-out run.

This module fixes two metric defects without re-running any rollout:

1. ``final_rotation_error_deg`` is recomputed against the gym_pusht goal angle
   ``pi/4`` instead of zero, and each row carries ``goal_block_angle_rad``.
2. Paired condition metrics add decidable/undecidable counting so that
   structurally tied ``[a,b] -> [b,a]`` probes no longer inflate the
   non-obedience count.

The source run directory is opened read-only; every corrected artifact is
written to a new output directory with ``label = "Re-derived"``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .config import (
    PUSHT_GOAL_BLOCK_ANGLE_RAD,
    default_output_root,
    load_json_object,
    practice_root,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .route_field_episodes import final_rotation_error_degrees
from .route_field_eval import _condition_metric_rows, _paired_condition_rows

SCHEMA_VERSION = 2
SOURCE_EXPERIMENT = "route_field_held_out_validation"
EXPECTED_CONDITION_COUNTS = {"oracle": 21, "reversed": 19, "permuted": 5}
REQUIRED_COLUMNS = (
    "final_block_angle_rad",
    "commanded_minus_oracle_adherence",
    "requested_route_length",
    "ordered_waypoint_adherence",
    "oracle_ordered_waypoint_adherence",
)
_REMOTE_REPO_MARKER = "/practice/practice/"
_METRIC_COMMENTS = {
    "validation_split": "Committed split of the source validation run",
    "episode_count": "Audit episodes in the source run",
    "rollout_count": "Source rollouts; no new rollout was executed",
    "goal_block_angle_rad": "Reference angle used by the corrected rotation error",
    "derived_from_rollouts_sha256": "SHA256 of the source rollouts.parquet",
    "source_repository_commit": "Source run commit; None means the source manifest had no commit",
    "undecidable_pair_count": "Pairs whose commanded-vs-oracle adherence difference is exactly 0",
    "decidable_pair_count": "Pairs with a non-zero adherence difference",
    "obeyed_count_decidable": "Obeyed count restricted to decidable pairs",
    "obeyed_rate_decidable": "Obeyed rate over decidable pairs only; None when none are decidable",
    "obeyed_wilson95_low_decidable": "Wilson 95% lower bound over decidable pairs",
    "obeyed_wilson95_high_decidable": "Wilson 95% upper bound over decidable pairs",
    "two_cell_probe_count": "Probes whose requested route has exactly two cells (structurally tied)",
}


def _repo_relative(path: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(practice_root()))
    except ValueError:
        return str(path.expanduser().resolve())


def _repo_relative_or_none(value: Any) -> str | None:
    """Rewrite a recorded absolute path to a repository-relative path or None."""

    if not isinstance(value, str) or not value:
        return None
    try:
        return str(Path(value).expanduser().resolve().relative_to(practice_root()))
    except ValueError:
        pass
    marker = value.find(_REMOTE_REPO_MARKER)
    if marker >= 0:
        return value[marker + len(_REMOTE_REPO_MARKER) :]
    return None


def _build_metrics(
    source_metrics: dict[str, Any],
    condition_rows: Sequence[dict[str, Any]],
    paired_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "validation_split": source_metrics.get("validation_split"),
        "episode_count": source_metrics.get("episode_count"),
        "rollout_count": source_metrics.get("rollout_count"),
        "goal_block_angle_rad": PUSHT_GOAL_BLOCK_ANGLE_RAD,
    }
    for key in (
        "condition_response_episode_cap",
        "condition_response_row_count",
        "route_field_diffusion_condition_response_chunk_l2_mean",
        "route_field_diffusion_condition_response_endpoint_mean",
    ):
        if key in source_metrics:
            metrics[key] = source_metrics[key]
    for row in condition_rows:
        prefix = f"{row['model']}_{row['commanded_condition']}"
        for key, value in row.items():
            if key in ("model", "commanded_condition"):
                continue
            metrics[f"{prefix}_{key}"] = value
    for row in paired_rows:
        prefix = f"{row['model']}_{row['reference_condition']}_vs_{row['probe_condition']}"
        for key, value in row.items():
            if key in ("model", "reference_condition", "probe_condition"):
                continue
            metrics[f"{prefix}_{key}"] = value
    return metrics


def _build_manifest(
    *,
    source_dir: Path,
    source_manifest: dict[str, Any],
    source_manifest_path: Path,
    rollouts_path: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    preserved_keys = (
        "dataset_revision",
        "episode_indices",
        "policy_seed",
        "num_inference_steps",
        "episode_length",
        "diffusion_scheduler",
        "diffusion_sampling_protocol",
        "condition_protocol",
        "condition_source",
        "initial_state_source",
        "obedience_criterion",
        "fixed_observation_episode_cap",
        "figures_enabled",
        "validation_split",
        "condition_counts",
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "label": "Re-derived",
        "experiment": "route_field_metrics_rederive",
        "goal_block_angle_rad": PUSHT_GOAL_BLOCK_ANGLE_RAD,
        "derived_from": _repo_relative(source_dir),
        "derived_from_rollouts_sha256": sha256_file(rollouts_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_repository_commit": (
            source_manifest.get("runtime", {}).get("repository", {}).get("commit")
        ),
        "route_manifest": _repo_relative_or_none(source_manifest.get("route_manifest")),
        "selection_dir": _repo_relative_or_none(source_manifest.get("selection_dir")),
    }
    for key in preserved_keys:
        if key in source_manifest:
            manifest[key] = source_manifest[key]
    manifest["runtime"] = runtime
    return manifest


def rederive_route_field_metrics(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool,
) -> Path:
    """Recompute corrected metrics from the original rollouts and write new artifacts."""

    source = Path(source_dir).expanduser().resolve()
    rollouts_path = source / "rollouts.parquet"
    manifest_path = source / "run_manifest.json"
    metrics_path = source / "metrics.json"
    if not rollouts_path.is_file():
        raise FileNotFoundError(f"Source rollouts are missing: {rollouts_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest is missing: {manifest_path}")
    source_manifest = load_json_object(manifest_path)
    if source_manifest.get("experiment") != SOURCE_EXPERIMENT:
        raise ValueError(f"Source manifest is not {SOURCE_EXPERIMENT!r}: {manifest_path}")
    if source_manifest.get("status") != "complete":
        raise ValueError(f"Source run is not complete: {manifest_path}")

    table = pq.read_table(rollouts_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.column_names]
    if missing:
        raise ValueError(f"Source rollouts are missing required columns: {missing}")
    rows = table.to_pylist()
    counts = Counter(str(row["commanded_condition"]) for row in rows)
    if dict(counts) != EXPECTED_CONDITION_COUNTS:
        raise ValueError(
            f"Source condition counts {dict(counts)} differ from "
            f"expected {EXPECTED_CONDITION_COUNTS}"
        )

    for row in rows:
        angle = row.get("final_block_angle_rad")
        if angle is None or not np.isfinite(float(angle)):
            row["final_rotation_error_deg"] = None
        else:
            row["final_rotation_error_deg"] = final_rotation_error_degrees(float(angle))
        row["goal_block_angle_rad"] = PUSHT_GOAL_BLOCK_ANGLE_RAD

    models = tuple(sorted({str(row["model"]) for row in rows}))
    condition_rows = _condition_metric_rows(rows, models=models)
    paired_rows = _paired_condition_rows(rows, models=models)
    source_metrics = load_json_object(metrics_path) if metrics_path.is_file() else {}
    metrics = _build_metrics(source_metrics, condition_rows, paired_rows)
    runtime = collect_runtime_provenance(practice_root())

    destination = Path(output_dir).expanduser().resolve()
    with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
        write_records_parquet(temporary / "rollouts.parquet", rows)
        write_records_parquet(temporary / "condition_metrics.parquet", condition_rows)
        write_records_parquet(temporary / "paired_condition_metrics.parquet", paired_rows)
        write_metrics(temporary, metrics, comments=_METRIC_COMMENTS)
        write_json_atomic(
            temporary / "run_manifest.json",
            _build_manifest(
                source_dir=source,
                source_manifest=source_manifest,
                source_manifest_path=manifest_path,
                rollouts_path=rollouts_path,
                runtime=runtime,
            ),
        )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "route_field_held_out_eval_corrected",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = rederive_route_field_metrics(
        source_dir=args.source_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(f"Corrected RF7 metrics output: {result}")


if __name__ == "__main__":
    main()
