"""Validation helpers for episode-level paired evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

MANIFEST_KEYS = ("episode_key", "reset_state_id")
PAIRED_CONTRACT_KEYS = (
    "reset_seed",
    "policy_seed",
    "requested_env_id",
    "runtime_env_id",
    "action_dim",
    "n_obs_steps",
    "action_horizon",
    "execution_action_steps",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _key(row: Mapping[str, Any], *, source: str) -> str:
    value = row.get("episode_key")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} row has no non-empty episode_key")
    return value


def validate_manifest(rows: Iterable[Mapping[str, Any]], *, source: str = "manifest") -> None:
    seen: set[str] = set()
    for row in rows:
        key = _key(row, source=source)
        if key in seen:
            raise ValueError(f"{source} contains duplicate episode_key {key!r}")
        seen.add(key)
        for field in MANIFEST_KEYS:
            if field not in row or row[field] in (None, ""):
                raise ValueError(f"{source} row {key!r} is missing {field}")


def validate_paired_records(
    base: Iterable[Mapping[str, Any]], treatment: Iterable[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Require exact episode and reset-state identity before comparison."""

    base_rows = list(base)
    treatment_rows = list(treatment)
    validate_manifest(base_rows, source="base")
    validate_manifest(treatment_rows, source="treatment")
    base_by_key = {_key(row, source="base"): row for row in base_rows}
    treatment_by_key = {_key(row, source="treatment"): row for row in treatment_rows}
    if set(base_by_key) != set(treatment_by_key):
        raise ValueError("base and treatment episode_key sets are not identical")
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for key in sorted(base_by_key):
        left = base_by_key[key]
        right = treatment_by_key[key]
        if left["reset_state_id"] != right["reset_state_id"]:
            raise ValueError(f"paired reset_state_id mismatch for {key!r}")
        for field in PAIRED_CONTRACT_KEYS:
            # Older generic RobotWin rows do not carry every S1 field.  When
            # a field exists on both sides, however, silently comparing rows
            # with different contracts would invalidate the treatment gap.
            if field in left and field in right and left[field] != right[field]:
                raise ValueError(f"paired {field} mismatch for {key!r}")
        pairs.append((left, right))
    return pairs


def paired_bootstrap_ci(
    differences: Iterable[float], *, seed: int = 20260829, repeats: int = 2000
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=np.float64)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one difference")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[sample_indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_paired(
    base_rows: Iterable[Mapping[str, Any]],
    treatment_rows: Iterable[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260829,
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    pairs = validate_paired_records(base_rows, treatment_rows)
    base_success = np.asarray(
        [bool(left.get("success", False)) for left, _ in pairs], dtype=np.float64
    )
    treatment_success = np.asarray(
        [bool(right.get("success", False)) for _, right in pairs], dtype=np.float64
    )
    differences = treatment_success - base_success
    ci = paired_bootstrap_ci(differences, seed=bootstrap_seed, repeats=bootstrap_repeats)
    result = {
        "num_paired_episodes": int(len(pairs)),
        "base_success_rate": float(base_success.mean()),
        "treatment_success_rate": float(treatment_success.mean()),
        "observed_gap": float(differences.mean()),
        "paired_bootstrap_95_ci": [ci[0], ci[1]],
        "paired_success_delta_counts": {
            "base_only": int(np.sum((base_success == 1) & (treatment_success == 0))),
            "treatment_only": int(np.sum((base_success == 0) & (treatment_success == 1))),
            "both_success": int(np.sum((base_success == 1) & (treatment_success == 1))),
            "both_failure": int(np.sum((base_success == 0) & (treatment_success == 0))),
        },
    }
    seed_groups: dict[int, list[tuple[float, float]]] = {}
    for (left, right), base_value, treatment_value in zip(
        pairs, base_success, treatment_success
    ):
        policy_seed = left.get("policy_seed")
        if policy_seed is None:
            seed_groups = {}
            break
        seed_groups.setdefault(int(policy_seed), []).append((base_value, treatment_value))
    if seed_groups:
        per_seed: dict[str, dict[str, Any]] = {}
        for policy_seed in sorted(seed_groups):
            values = seed_groups[policy_seed]
            seed_base = np.asarray([value[0] for value in values], dtype=np.float64)
            seed_treatment = np.asarray([value[1] for value in values], dtype=np.float64)
            seed_differences = seed_treatment - seed_base
            seed_ci = paired_bootstrap_ci(
                seed_differences,
                seed=bootstrap_seed + policy_seed,
                repeats=bootstrap_repeats,
            )
            per_seed[str(policy_seed)] = {
                "num_paired_episodes": int(len(values)),
                "base_success_rate": float(seed_base.mean()),
                "treatment_success_rate": float(seed_treatment.mean()),
                "observed_gap": float(seed_differences.mean()),
                "paired_bootstrap_95_ci": [seed_ci[0], seed_ci[1]],
            }
        result["per_policy_seed"] = per_seed
    return result
