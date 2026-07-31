"""Build the canonical 7x7 route-field sidecar for the PushT route pilot."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .config import PUSHT_WORKSPACE_SIZE, default_output_root, default_split_path, practice_root
from .dataset_split import EpisodeSplit, load_split
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .provenance import sha256_file, write_json_atomic

ROUTE_FIELD_SCHEMA_VERSION = 2
ROUTE_FIELD_NAME = "RF7"
ROUTE_FIELD_GRID_SIZE = 7
ROUTE_FIELD_CHANNELS = ("occupancy", "progress", "direction_x", "direction_y")
ROUTE_FIELD_STABLE_FRAMES = 3
PILOT_ROUTE_A = (16, 17, 24)
PILOT_ROUTE_B = (16, 23, 24)


def position_to_cell(
    position: Sequence[float],
    *,
    grid_size: int = ROUTE_FIELD_GRID_SIZE,
    workspace_size: float = PUSHT_WORKSPACE_SIZE,
) -> int:
    """Map one PushT world-space position into a row-major grid cell."""

    if len(position) != 2:
        raise ValueError(f"Expected one (x,y) position, got {position!r}")
    x, y = (float(position[0]), float(position[1]))
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"Position must be finite, got {position!r}")
    if grid_size <= 1 or workspace_size <= 0:
        raise ValueError("grid_size must exceed one and workspace_size must be positive")
    cell_size = workspace_size / grid_size
    column = min(max(int(x / cell_size), 0), grid_size - 1)
    row = min(max(int(y / cell_size), 0), grid_size - 1)
    return row * grid_size + column


def extract_stable_route(
    positions: Iterable[Sequence[float]],
    *,
    grid_size: int = ROUTE_FIELD_GRID_SIZE,
    stable_frames: int = ROUTE_FIELD_STABLE_FRAMES,
) -> tuple[int, ...]:
    """Debounce grid transitions and compress consecutive duplicate cells."""

    if stable_frames <= 0:
        raise ValueError("stable_frames must be positive")
    route: list[int] = []
    confirmed: int | None = None
    candidate: int | None = None
    candidate_frames = 0
    for position in positions:
        cell = position_to_cell(position, grid_size=grid_size)
        if cell == confirmed:
            candidate = None
            candidate_frames = 0
            continue
        if cell == candidate:
            candidate_frames += 1
        else:
            candidate = cell
            candidate_frames = 1
        if candidate_frames >= stable_frames:
            confirmed = cell
            route.append(cell)
            candidate = None
            candidate_frames = 0
    return tuple(route)


def route_to_field(
    route: Sequence[int],
    *,
    grid_size: int = ROUTE_FIELD_GRID_SIZE,
) -> np.ndarray:
    """Encode one simple route as occupancy, progress, dx, and dy maps."""

    route_cells = tuple(int(cell) for cell in route)
    if not route_cells:
        raise ValueError("A route field requires at least one cell")
    if len(route_cells) != len(set(route_cells)):
        raise ValueError("A route field cannot represent repeated cells")
    invalid = [cell for cell in route_cells if not 0 <= cell < grid_size * grid_size]
    if invalid:
        raise ValueError(f"Route contains out-of-range cells: {invalid}")

    field = np.zeros((len(ROUTE_FIELD_CHANNELS), grid_size, grid_size), dtype=np.float32)
    for index, cell in enumerate(route_cells):
        row, column = divmod(cell, grid_size)
        field[0, row, column] = 1.0
        field[1, row, column] = 1.0 if len(route_cells) == 1 else index / (len(route_cells) - 1)
        if index + 1 == len(route_cells):
            continue
        next_row, next_column = divmod(route_cells[index + 1], grid_size)
        dx = float(next_column - column)
        dy = float(next_row - row)
        norm = math.hypot(dx, dy)
        field[2, row, column] = dx / norm
        field[3, row, column] = dy / norm
    return field


def simplify_macro_route(route: Sequence[int]) -> tuple[tuple[int, ...], int, int]:
    """Remove immediate backtracks, then erase only residual closed loops."""

    raw = tuple(int(cell) for cell in route)
    if not raw:
        raise ValueError("Cannot simplify an empty route")

    backtrack_clean: list[int] = []
    backtrack_removed = 0
    for cell in raw:
        backtrack_clean.append(cell)
        while len(backtrack_clean) >= 3 and backtrack_clean[-1] == backtrack_clean[-3]:
            del backtrack_clean[-2:]
            backtrack_removed += 2

    macro: list[int] = []
    locations: dict[int, int] = {}
    loop_removed = 0
    for cell in backtrack_clean:
        if cell not in locations:
            locations[cell] = len(macro)
            macro.append(cell)
            continue
        keep_index = locations[cell]
        loop_removed += len(macro) - keep_index
        for removed_cell in macro[keep_index + 1 :]:
            locations.pop(removed_cell, None)
        del macro[keep_index + 1 :]

    result = tuple(macro)
    if len(result) != len(set(result)):
        raise RuntimeError(f"Macro-route simplification left repeated cells: {result}")
    if len(raw) - len(result) != backtrack_removed + loop_removed:
        raise RuntimeError("Macro-route simplification accounting mismatch")
    return result, backtrack_removed, loop_removed


def _episode_split_name(split: EpisodeSplit, episode: int) -> str:
    if episode in set(split.train):
        return "train"
    if episode in set(split.development):
        return "development"
    if episode in set(split.audit):
        return "audit"
    raise ValueError(f"Episode {episode} is absent from the frozen split")


def _source_label(path: Path) -> str:
    try:
        return str(path.relative_to(practice_root()))
    except ValueError:
        return path.name


def _grid_feasibility(
    positions_by_episode: dict[int, list[Sequence[float]]],
    *,
    train_episodes: Sequence[int],
    stable_frames: int,
) -> list[dict[str, Any]]:
    """Record the train-only evidence used to reject the 3x3 representation."""

    results = []
    for grid_size in (3, 4, 5, 7):
        routes = [
            extract_stable_route(
                positions_by_episode[episode],
                grid_size=grid_size,
                stable_frames=stable_frames,
            )
            for episode in train_episodes
        ]
        lengths = np.asarray([len(route) for route in routes], dtype=np.int64)
        results.append(
            {
                "grid_size": grid_size,
                "at_most_one_stable_cell_train_episodes": int((lengths <= 1).sum()),
                "median_route_cells": float(np.median(lengths)),
                "p90_route_cells": float(np.quantile(lengths, 0.90)),
                "repeated_cell_train_episodes": sum(
                    len(route) != len(set(route)) for route in routes
                ),
            }
        )
    return results


def build_route_field_sidecar(
    *,
    trajectories_path: str | Path,
    output_dir: str | Path,
    split_file: str | Path | None = None,
    grid_size: int = ROUTE_FIELD_GRID_SIZE,
    stable_frames: int = ROUTE_FIELD_STABLE_FRAMES,
    pilot_routes: Sequence[Sequence[int]] = (PILOT_ROUTE_A, PILOT_ROUTE_B),
    overwrite: bool = False,
) -> Path:
    """Mine train-only route families and write a complete immutable sidecar."""

    if grid_size != ROUTE_FIELD_GRID_SIZE:
        raise ValueError(
            f"This pilot is frozen to a {ROUTE_FIELD_GRID_SIZE}x{ROUTE_FIELD_GRID_SIZE} grid"
        )
    trajectories = Path(trajectories_path).expanduser().resolve()
    if not trajectories.is_file():
        raise FileNotFoundError(f"Trajectory artifact is missing: {trajectories}")
    split = load_split(split_file)

    records_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in pq.read_table(trajectories).to_pylist():
        records_by_episode[int(record["episode_index"])].append(record)
    expected_episodes = set(range(split.num_episodes))
    if set(records_by_episode) != expected_episodes:
        missing = sorted(expected_episodes - set(records_by_episode))
        extra = sorted(set(records_by_episode) - expected_episodes)
        raise ValueError(
            f"Trajectory episodes do not match the split: missing={missing}, extra={extra}"
        )

    episode_raw_routes: dict[int, tuple[int, ...]] = {}
    episode_routes: dict[int, tuple[int, ...]] = {}
    simplification_counts: dict[int, tuple[int, int]] = {}
    episode_positions: dict[int, list[Sequence[float]]] = {}
    frame_counts: dict[int, int] = {}
    for episode, records in records_by_episode.items():
        ordered = sorted(records, key=lambda row: int(row["frame_index"]))
        frame_counts[episode] = len(ordered)
        valid_positions = [
            row["block_position"]
            for row in ordered
            if bool(row["block_pose_valid"]) and row.get("block_position") is not None
        ]
        episode_positions[episode] = valid_positions
        raw_route = extract_stable_route(
            valid_positions,
            grid_size=grid_size,
            stable_frames=stable_frames,
        )
        if not raw_route:
            raise ValueError(f"Episode {episode} has no stable RF7 cell")
        route, backtrack_removed, loop_removed = simplify_macro_route(raw_route)
        episode_raw_routes[episode] = raw_route
        episode_routes[episode] = route
        simplification_counts[episode] = (backtrack_removed, loop_removed)

    train_counts = Counter(episode_routes[episode] for episode in split.train)
    canonical_routes = tuple(sorted(train_counts))
    if not canonical_routes:
        raise ValueError("No route family satisfies the frozen training eligibility rules")
    route_to_family = {route: family_id for family_id, route in enumerate(canonical_routes)}

    requested_pilot_routes = tuple(tuple(int(cell) for cell in route) for route in pilot_routes)
    missing_pilot_routes = [
        list(route) for route in requested_pilot_routes if route not in route_to_family
    ]
    if missing_pilot_routes:
        raise ValueError(
            f"Requested pilot routes are absent from the train macro routes: {missing_pilot_routes}"
        )

    fields = np.stack(
        [route_to_field(route, grid_size=grid_size) for route in canonical_routes],
        axis=0,
    )
    assignment_rows: list[dict[str, Any]] = []
    train_episode_ids: list[int] = []
    training_frames = 0
    for episode in range(split.num_episodes):
        route = episode_routes[episode]
        raw_route = episode_raw_routes[episode]
        split_name = _episode_split_name(split, episode)
        family_id = route_to_family.get(route)
        reason = None if family_id is not None else "macro_route_not_observed_in_train"
        eligible = family_id is not None
        training_condition_id = family_id if eligible and split_name == "train" else None
        if training_condition_id is not None:
            train_episode_ids.append(episode)
            training_frames += frame_counts[episode]
        backtrack_removed, loop_removed = simplification_counts[episode]
        assignment_rows.append(
            {
                "episode_index": episode,
                "split": split_name,
                "raw_route_cells": list(raw_route),
                "route_cells": list(route),
                "family_id": family_id,
                "training_condition_id": training_condition_id,
                "eligible": eligible,
                "exclusion_reason": reason,
                "backtrack_removed_cells": backtrack_removed,
                "loop_removed_cells": loop_removed,
                "total_removed_cells": backtrack_removed + loop_removed,
                "num_frames": frame_counts[episode],
            }
        )
    if tuple(train_episode_ids) != tuple(sorted(split.train)):
        missing = sorted(set(split.train) - set(train_episode_ids))
        raise RuntimeError(f"RF7 must include every train episode, missing: {missing}")

    destination = Path(output_dir).expanduser().resolve()
    with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
        assignments_path = write_records_parquet(
            temporary / "episode_assignments.parquet",
            assignment_rows,
        )
        fields_path = temporary / "route_fields.npz"
        np.savez_compressed(
            fields_path,
            family_ids=np.arange(len(canonical_routes), dtype=np.int64),
            fields=fields,
        )
        artifact_hashes = {
            assignments_path.name: sha256_file(assignments_path),
            fields_path.name: sha256_file(fields_path),
        }
        family_records = [
            {
                "family_id": family_id,
                "route_cells": list(route),
                "train_episode_support": train_counts[route],
            }
            for family_id, route in enumerate(canonical_routes)
        ]
        pilot_pair = [
            {
                "label": chr(ord("A") + index),
                "family_id": route_to_family[route],
                "route_cells": list(route),
                "train_episode_support": train_counts[route],
            }
            for index, route in enumerate(requested_pilot_routes)
        ]
        manifest = {
            "schema_version": ROUTE_FIELD_SCHEMA_VERSION,
            "status": "complete",
            "experiment": "PushT RouteField pilot",
            "source_trajectories": _source_label(trajectories),
            "source_trajectories_sha256": sha256_file(trajectories),
            "split_manifest_sha256": split.manifest_sha256,
            "grid": {
                "size": grid_size,
                "workspace_size": PUSHT_WORKSPACE_SIZE,
                "row_major": True,
                "stable_frames": stable_frames,
            },
            "field": {
                "name": ROUTE_FIELD_NAME,
                "channels": list(ROUTE_FIELD_CHANNELS),
                "shape": [len(ROUTE_FIELD_CHANNELS), grid_size, grid_size],
                "outside_route_value": 0.0,
            },
            "eligibility": {
                "minimum_route_cells": 1,
                "allow_single_cell_route": True,
                "allow_repeated_macro_cells": False,
                "minimum_train_family_support": None,
                "all_train_episodes_required": True,
            },
            "route_simplification": {
                "immediate_backtrack": "iteratively collapse A-B-A to A",
                "residual_closed_loops": "chronological loop erasure",
                "train_episodes_with_backtrack_removal": sum(
                    simplification_counts[episode][0] > 0 for episode in split.train
                ),
                "train_episodes_with_residual_loop_removal": sum(
                    simplification_counts[episode][1] > 0 for episode in split.train
                ),
                "train_episodes_with_any_removal": sum(
                    sum(simplification_counts[episode]) > 0 for episode in split.train
                ),
                "train_route_cells_removed": sum(
                    sum(simplification_counts[episode]) for episode in split.train
                ),
            },
            "grid_feasibility": _grid_feasibility(
                episode_positions,
                train_episodes=split.train,
                stable_frames=stable_frames,
            ),
            "canonical_families": family_records,
            "pilot_pair": pilot_pair,
            "training": {
                "episode_ids": train_episode_ids,
                "num_episodes": len(train_episode_ids),
                "num_frames": training_frames,
                "all_train_episodes_included": True,
                "excluded_train_episodes_by_reason": {},
            },
            "artifact_hashes": artifact_hashes,
        }
        write_json_atomic(temporary / "route_manifest.json", manifest)
        write_metrics(
            temporary,
            {
                "grid_size": grid_size,
                "num_canonical_families": len(canonical_routes),
                "eligible_train_episodes": len(train_episode_ids),
                "excluded_train_episodes": len(split.train) - len(train_episode_ids),
                "training_frames": training_frames,
                "train_episodes_with_route_simplification": sum(
                    sum(simplification_counts[episode]) > 0 for episode in split.train
                ),
                "train_route_cells_removed": sum(
                    sum(simplification_counts[episode]) for episode in split.train
                ),
                "pilot_route_a_support": (
                    train_counts[requested_pilot_routes[0]] if requested_pilot_routes else 0
                ),
                "pilot_route_b_support": (
                    train_counts[requested_pilot_routes[1]]
                    if len(requested_pilot_routes) > 1
                    else 0
                ),
            },
        )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=default_output_root() / "data_inspection" / "trajectories.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "route_field_7x7" / "sidecar",
    )
    parser.add_argument("--split-file", type=Path, default=default_split_path())
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output = build_route_field_sidecar(
        trajectories_path=args.trajectories,
        output_dir=args.output_dir,
        split_file=args.split_file,
        overwrite=args.overwrite,
    )
    print(f"Route-field artifacts: {output}")


if __name__ == "__main__":
    main()


__all__ = [
    "PILOT_ROUTE_A",
    "PILOT_ROUTE_B",
    "ROUTE_FIELD_CHANNELS",
    "ROUTE_FIELD_GRID_SIZE",
    "ROUTE_FIELD_NAME",
    "ROUTE_FIELD_SCHEMA_VERSION",
    "ROUTE_FIELD_STABLE_FRAMES",
    "build_route_field_sidecar",
    "extract_stable_route",
    "position_to_cell",
    "route_to_field",
    "simplify_macro_route",
]
