"""Frozen ManiSkill state batches and metadata-only split helpers.

ManiSkill 3's PickCube reset path uses the first seed in a seed list to seed a
global RNG and uses the row number inside that reset batch for the remaining
states.  The values after the first seed are therefore labels, not
independent state seeds.  This module keeps that unusual but measured contract
explicit instead of silently treating a batch as independent scalar seeds.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SEED_SET_NAMES = ("train", "development", "test")
TASK_ID = "PickCube-v1"
DEFAULT_SIM_BACKEND = "physx_cuda"
DEFAULT_STATE_MATCH_TOLERANCE = 1e-4


def _as_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer, got {value!r}") from exc
    if result != value:
        raise ValueError(f"{field} must be an exact integer, got {value!r}")
    return result


def _as_positive_int(value: Any, *, field: str) -> int:
    result = _as_int(value, field=field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def expand_seed_spec(spec: Any, *, field: str) -> tuple[int, ...]:
    """Expand an explicit list or a legacy half-open range specification."""

    if isinstance(spec, (list, tuple)):
        values = tuple(
            _as_int(value, field=f"{field}[{index}]") for index, value in enumerate(spec)
        )
    elif isinstance(spec, Mapping):
        start = _as_int(spec.get("start"), field=f"{field}.start")
        stop = _as_int(spec.get("stop"), field=f"{field}.stop")
        step = _as_int(spec.get("step", 1), field=f"{field}.step")
        if step == 0:
            raise ValueError(f"{field}.step must not be zero")
        values = tuple(range(start, stop, step))
        declared_count = spec.get("count")
        if declared_count is not None:
            count = _as_int(declared_count, field=f"{field}.count")
            if count != len(values):
                raise ValueError(
                    f"{field}.count={count} does not match expanded length {len(values)}"
                )
    else:
        raise TypeError(f"{field} must be a list or range object, got {type(spec).__name__}")

    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate seeds")
    return values


def _batch(head_seed: int, count: int, *, field: str) -> dict[str, Any]:
    return {
        "head_seed": _as_int(head_seed, field=f"{field}.head_seed"),
        "count": _as_positive_int(count, field=f"{field}.count"),
    }


def _parse_batch_spec(spec: Any, *, field: str) -> tuple[dict[str, Any], ...]:
    """Parse the canonical ``head_seeds + count`` or explicit block form."""

    if not isinstance(spec, Mapping):
        raise TypeError(f"{field} must be an object")

    if "blocks" in spec:
        raw_blocks = spec["blocks"]
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise ValueError(f"{field}.blocks must be a non-empty list")
        blocks = []
        for index, raw_block in enumerate(raw_blocks):
            if not isinstance(raw_block, Mapping):
                raise TypeError(f"{field}.blocks[{index}] must be an object")
            blocks.append(
                _batch(
                    raw_block.get("head_seed"),
                    raw_block.get("count"),
                    field=f"{field}.blocks[{index}]",
                )
            )
        return tuple(blocks)

    raw_heads = spec.get("head_seeds")
    if raw_heads is None and "head_seed" in spec:
        raw_heads = [spec["head_seed"]]
    if not isinstance(raw_heads, (list, tuple)) or not raw_heads:
        raise ValueError(f"{field}.head_seeds must be a non-empty list")
    count = _as_positive_int(spec.get("count"), field=f"{field}.count")
    return tuple(
        _batch(head, count, field=f"{field}.head_seeds[{index}]")
        for index, head in enumerate(raw_heads)
    )


def _legacy_batches(spec: Any, *, field: str) -> tuple[dict[str, Any], ...]:
    values = expand_seed_spec(spec, field=field)
    if not values:
        raise ValueError(f"{field} must not be empty")
    # Legacy explicit lists are retained for small unit-test/config callers.
    # They are not used by the frozen M0 config, whose semantics are batches.
    return ({"head_seed": values[0], "count": len(values), "seeds": values},)


def _materialize_batches(
    specs: Iterable[Mapping[str, Any]], *, legacy: bool = False
) -> tuple[dict[str, Any], ...]:
    materialized = []
    for spec in specs:
        head_seed = int(spec["head_seed"])
        count = int(spec["count"])
        seeds = tuple(
            int(value) for value in spec.get("seeds", range(head_seed, head_seed + count))
        )
        if len(seeds) != count:
            raise ValueError("A state batch's seed labels must match its count")
        if len(set(seeds)) != len(seeds):
            raise ValueError("A state batch contains duplicate seed labels")
        materialized.append(
            {
                "head_seed": head_seed,
                "count": count,
                "seeds": seeds,
                "batch_index": len(materialized),
                "semantics": "explicit_labels" if legacy else "head_seed_plus_row",
            }
        )
    return tuple(materialized)


def _parse_sets(payload: Mapping[str, Any]) -> tuple[dict[str, tuple[dict[str, Any], ...]], bool]:
    raw_sets = payload.get("sets")
    if isinstance(raw_sets, Mapping):
        parsed: dict[str, tuple[dict[str, Any], ...]] = {}
        for name in SEED_SET_NAMES:
            if name not in raw_sets:
                raise KeyError(f"Seed split is missing sets.{name}")
            parsed[name] = _parse_batch_spec(raw_sets[name], field=f"sets.{name}")
        return parsed, False

    parsed = {}
    for name in SEED_SET_NAMES:
        key = f"{name}_seeds"
        if key not in payload:
            raise KeyError(f"Seed split is missing {key}")
        parsed[name] = _legacy_batches(payload[key], field=key)
    return parsed, True


def load_seed_sets(path: str | Path, *, task: str = TASK_ID) -> dict[str, Any]:
    """Load and validate the frozen state-batch partition.

    ``sets`` contains flattened labels for compatibility.  ``batches`` is the
    authoritative representation for ManiSkill resets and preserves batch
    boundaries.
    """

    split_path = Path(path)
    with split_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Seed split must be a JSON object: {split_path}")
    if payload.get("task") != task:
        raise ValueError(f"Seed split task must be {task!r}, got {payload.get('task')!r}")

    parsed, legacy = _parse_sets(payload)
    batches = {name: _materialize_batches(specs, legacy=legacy) for name, specs in parsed.items()}
    sets = {
        name: tuple(seed for batch in batches[name] for seed in batch["seeds"])
        for name in SEED_SET_NAMES
    }
    if any(not values for values in sets.values()):
        raise ValueError("All state sets must contain at least one state")

    overlaps: dict[str, list[int]] = {}
    for left_index, left_name in enumerate(SEED_SET_NAMES):
        for right_name in SEED_SET_NAMES[left_index + 1 :]:
            overlap = sorted(set(sets[left_name]) & set(sets[right_name]))
            if overlap:
                overlaps[f"{left_name}∩{right_name}"] = overlap
    if overlaps:
        raise ValueError(f"State split labels overlap: {overlaps}")

    head_seeds = [int(batch["head_seed"]) for name in SEED_SET_NAMES for batch in batches[name]]
    if len(head_seeds) != len(set(head_seeds)):
        raise ValueError("State-batch head seeds must be unique")

    sim_backend = str(payload.get("sim_backend", DEFAULT_SIM_BACKEND))
    tolerance = float(payload.get("state_match_tolerance", DEFAULT_STATE_MATCH_TOLERANCE))
    if tolerance <= 0:
        raise ValueError("state_match_tolerance must be positive")
    return {
        "path": split_path,
        "task": task,
        "sets": sets,
        "batches": batches,
        "sim_backend": sim_backend,
        "state_match_tolerance": tolerance,
        "payload": payload,
    }


def seeds_for_set(
    split: Mapping[str, Any],
    seed_set: str,
    *,
    limit: int | None = None,
) -> tuple[int, ...]:
    """Return flattened labels, optionally as a deterministic prefix."""

    if seed_set not in SEED_SET_NAMES:
        raise ValueError(f"seed_set must be one of {SEED_SET_NAMES}, got {seed_set!r}")
    sets = split.get("sets")
    if not isinstance(sets, Mapping) or seed_set not in sets:
        raise TypeError("split must be the object returned by load_seed_sets")
    values = tuple(int(seed) for seed in sets[seed_set])
    if limit is None:
        return values
    if limit <= 0:
        raise ValueError("limit must be positive")
    return values[:limit]


def reset_batches_for_set(
    split: Mapping[str, Any],
    seed_set: str,
    *,
    limit: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return reset seed lists while preserving the frozen batch boundaries."""

    if seed_set not in SEED_SET_NAMES:
        raise ValueError(f"seed_set must be one of {SEED_SET_NAMES}, got {seed_set!r}")
    batches = split.get("batches")
    if not isinstance(batches, Mapping) or seed_set not in batches:
        raise TypeError("split must be the object returned by load_seed_sets")
    result: list[tuple[int, ...]] = []
    remaining = None if limit is None else int(limit)
    if remaining is not None and remaining <= 0:
        raise ValueError("limit must be positive")
    for batch in batches[seed_set]:
        values = tuple(int(seed) for seed in batch["seeds"])
        if remaining is None:
            result.append(values)
            continue
        if remaining <= 0:
            break
        result.append(values[:remaining])
        remaining -= len(result[-1])
    return tuple(batch for batch in result if batch)


def _metadata_seed(record: Mapping[str, Any]) -> int | None:
    episode_seed = record.get("episode_seed")
    if episode_seed is not None:
        return _as_int(episode_seed, field="episode_seed")
    reset_kwargs = record.get("reset_kwargs")
    if not isinstance(reset_kwargs, Mapping):
        return None
    seed = reset_kwargs.get("seed")
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(f"Expected one demonstration seed, got {seed!r}")
        seed = seed[0]
    return None if seed is None else _as_int(seed, field="reset_kwargs.seed")


def collect_demonstration_seeds(
    paths: Iterable[str | Path],
    *,
    task: str = TASK_ID,
) -> tuple[int, ...]:
    """Collect metadata seeds for auxiliary reporting only.

    These values are intentionally not used to prove state isolation for the
    RL demonstrations.  The actual audit compares initial-state fingerprints.
    """

    seeds: set[int] = set()
    found_metadata = False
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            continue
        env_info = payload.get("env_info")
        env_id = env_info.get("env_id") if isinstance(env_info, Mapping) else None
        if env_id is not None and env_id != task:
            continue
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            continue
        found_metadata = True
        for record in episodes:
            if isinstance(record, Mapping):
                seed = _metadata_seed(record)
                if seed is not None:
                    seeds.add(seed)
    if not found_metadata:
        raise FileNotFoundError("No ManiSkill trajectory metadata with episodes was found")
    return tuple(sorted(seeds))


def discover_demo_metadata(root: str | Path) -> tuple[Path, ...]:
    """Find trajectory metadata below a file or demo directory."""

    path = Path(root)
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise FileNotFoundError(f"Demonstration metadata root does not exist: {path}")
    return tuple(sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file()))


def seed_isolation_report(
    split: Mapping[str, Any],
    demonstration_seeds: Iterable[int],
) -> dict[str, Any]:
    """Return an explicitly limited, seed-only overlap report."""

    demo = {int(seed) for seed in demonstration_seeds}
    sets = split["sets"]
    intersections = {name: sorted(demo & set(sets[name])) for name in ("development", "test")}
    return {
        "method": "metadata_seed_overlap_only",
        "warning": "Seed integers are not a valid state-isolation proof for RL demonstrations.",
        "demonstration_seed_count": len(demo),
        "demonstration_seeds": sorted(demo),
        "intersections": intersections,
        "status": "disjoint" if not any(intersections.values()) else "overlap_present",
    }


def evaluation_seed_plan(
    split: Mapping[str, Any],
    seed_set: str,
    demonstration_seeds: Iterable[int] = (),
    *,
    limit: int | None = None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Compatibility helper that never mutates a frozen state batch.

    Unlike the old implementation, this function does not remove or replace
    seeds based on metadata.  Callers should use ``reset_batches_for_set`` and
    the initial-state fingerprint audit for leakage decisions.
    """

    batches = reset_batches_for_set(split, seed_set, limit=limit)
    seeds = tuple(seed for batch in batches for seed in batch)
    return seeds, {
        "seed_set": seed_set,
        "requested_seeds": list(seeds),
        "effective_seeds": list(seeds),
        "reset_batches": [list(batch) for batch in batches],
        "demonstration_seeds_ignored_for_state_isolation": sorted(
            {int(seed) for seed in demonstration_seeds}
        ),
        "status": "state_fingerprint_required",
    }
