"""Pure-Python analysis helpers for the W0 Fast-WAM protocol.

The module intentionally has no simulator, torch, numpy, or LeRobot import.  It
can therefore be used on a login node to aggregate evaluation JSON files and to
run the pre-registered bootstrap/gate without loading a model.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _success_indices(payload: Mapping[str, Any]) -> tuple[set[int], set[int], int]:
    """Extract success/failure episode indices from one upstream result JSON."""

    successes = {int(x) for x in payload.get("success_episodes", [])}
    failures = {int(x) for x in payload.get("failure_episodes", [])}
    total = payload.get("total_episodes", payload.get("num_trials"))
    if total is None:
        total = len(successes | failures)
    total = int(total)
    if total < len(successes | failures):
        raise ValueError("result reports fewer episodes than success/failure indices")
    if successes & failures:
        raise ValueError("an episode occurs in both success_episodes and failure_episodes")
    return successes, failures, total


def _result_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"summary.json", "manager_config.yaml", "run_manifest.json"}:
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and {
            "success_episodes",
            "failure_episodes",
        }.issubset(payload):
            files.append(path)
    return files


def aggregate_eval_directory(
    root: str | Path,
    *,
    expected_task_keys: Sequence[tuple[str, int]] | None = None,
    expected_trials_per_task: int = 50,
) -> dict[str, Any]:
    """Aggregate official per-task JSON files without double-counting summaries.

    The default is the formal 40-task/50-trial protocol. Callers running a
    smoke or single-task probe must pass their requested task keys so a
    complete probe is not incorrectly reported as ``incomplete``.
    """

    root_path = Path(root)
    merged_tasks: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_keys: list[list[Any]] = []
    duplicate_trial_indices: list[dict[str, Any]] = []
    for path in _result_files(root_path):
        payload = _read_json(path)
        local_successes, local_failures, reported_total = _success_indices(payload)
        # FastWAM's official evaluator writes ``task_suite``; the repository
        # fixtures and some earlier local wrappers used ``task_suite_name`` or
        # ``suite``.  Accept all three so a successful worker is not reported
        # as an incomplete run merely because of the result schema spelling.
        suite = str(
            payload.get(
                "task_suite",
                payload.get("task_suite_name", payload.get("suite", "unknown")),
            )
        )
        task_id = int(payload.get("task_id", -1))
        key = (suite, task_id)
        trial_start = int(payload.get("trial_start", 0))
        trial_count = int(payload.get("trial_count", reported_total))
        if trial_start < 0 or trial_count <= 0:
            raise ValueError(f"invalid trial shard in {path}: start={trial_start} count={trial_count}")
        if bool(payload.get("episode_indices_are_global", False)):
            successes = local_successes
            failures = local_failures
        else:
            successes = {trial_start + index for index in local_successes}
            failures = {trial_start + index for index in local_failures}
        shard_indices = successes | failures
        shard_end = trial_start + trial_count
        if any(index < trial_start or index >= shard_end for index in shard_indices):
            raise ValueError(
                f"trial indices outside shard range in {path}: "
                f"start={trial_start} count={trial_count} indices={sorted(shard_indices)}"
            )
        entry = merged_tasks.setdefault(
            key,
            {
                "suite": suite,
                "task_id": task_id,
                "success_indices": set(),
                "failure_indices": set(),
                "span_end": 0,
                "sources": [],
            },
        )
        previous_indices = entry["success_indices"] | entry["failure_indices"]
        overlap = sorted(previous_indices & shard_indices)
        if overlap:
            duplicate_keys.append([suite, task_id])
            duplicate_trial_indices.append(
                {"suite": suite, "task_id": task_id, "indices": overlap, "source": path.as_posix()}
            )
        entry["success_indices"].update(successes)
        entry["failure_indices"].update(failures)
        entry["span_end"] = max(int(entry["span_end"]), shard_end)
        entry["sources"].append(path.as_posix())

    task_rows: list[dict[str, Any]] = []
    for key in sorted(merged_tasks):
        entry = merged_tasks[key]
        successes = entry["success_indices"]
        failures = entry["failure_indices"]
        total = int(entry["span_end"])
        covered = successes | failures
        task_rows.append(
            {
                "suite": entry["suite"],
                "task_id": entry["task_id"],
                "successes": len(successes),
                "failures": len(failures),
                "total_episodes": total,
                "covered_episodes": len(covered),
                "missing_trial_count": max(total - len(covered), 0),
                "success_rate": len(successes) / total if total else 0.0,
                "success_indices": sorted(successes),
                "failure_indices": sorted(failures),
                "source": entry["sources"][0],
                "sources": entry["sources"],
            }
        )

    seen_keys = set(merged_tasks)

    by_suite: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"successes": 0, "total_episodes": 0, "tasks": 0}
    )
    for row in task_rows:
        suite = by_suite[row["suite"]]
        suite["successes"] += row["successes"]
        suite["total_episodes"] += row["total_episodes"]
        suite["tasks"] += 1
    suite_rows = {
        suite: {
            **values,
            "success_rate": values["successes"] / values["total_episodes"]
            if values["total_episodes"]
            else 0.0,
        }
        for suite, values in sorted(by_suite.items())
    }
    successes = sum(row["successes"] for row in task_rows)
    total = sum(row["total_episodes"] for row in task_rows)
    expected_keys = (
        {(str(suite), int(task_id)) for suite, task_id in expected_task_keys}
        if expected_task_keys is not None
        else None
    )
    expected_tasks = len(expected_keys) if expected_keys is not None else 40
    expected_trials = int(expected_trials_per_task)
    actual_keys = set(seen_keys)
    complete = (
        (actual_keys == expected_keys if expected_keys is not None else len(task_rows) == expected_tasks)
        and all(row["total_episodes"] == expected_trials for row in task_rows)
        and all(row["covered_episodes"] == expected_trials for row in task_rows)
        and not duplicate_keys
    )
    return {
        "schema_version": 1,
        "source_root": root_path.as_posix(),
        "status": "complete" if complete else "incomplete",
        "num_tasks": len(task_rows),
        "num_episodes": total,
        "successes": successes,
        "overall_success_rate": successes / total if total else 0.0,
        "expected_tasks": expected_tasks,
        "expected_trials_per_task": expected_trials,
        "duplicate_task_keys": duplicate_keys,
        "duplicate_trial_indices": duplicate_trial_indices,
        "missing_task_count": (
            len(expected_keys - actual_keys)
            if expected_keys is not None
            else max(expected_tasks - len(task_rows), 0)
        ),
        "per_suite": suite_rows,
        "tasks": sorted(task_rows, key=lambda row: (row["suite"], row["task_id"])),
    }


def paired_rows_from_summaries(
    with_video: Mapping[str, Any], no_video: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Pair task/trial outcomes from two aggregate summaries."""

    left = {(row["suite"], int(row["task_id"])): row for row in with_video.get("tasks", [])}
    right = {(row["suite"], int(row["task_id"])): row for row in no_video.get("tasks", [])}
    keys = sorted(set(left) & set(right))
    rows: list[dict[str, Any]] = []
    for suite, task_id in keys:
        a = left[(suite, task_id)]
        b = right[(suite, task_id)]
        a_success = {int(x) for x in a.get("success_indices", [])}
        b_success = {int(x) for x in b.get("success_indices", [])}
        total = max(int(a.get("total_episodes", 0)), int(b.get("total_episodes", 0)))
        for trial in range(total):
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "trial": trial,
                    "with_video": trial in a_success,
                    "no_video": trial in b_success,
                }
            )
    return rows


def paired_bootstrap(
    pairs: Sequence[Mapping[str, Any]], *, repeats: int = 10000, seed: int = 20260817
) -> dict[str, Any]:
    """Bootstrap paired success differences, with deterministic local RNG."""

    if not pairs:
        raise ValueError("paired bootstrap requires at least one episode")
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    deltas = [int(bool(row["with_video"])) - int(bool(row["no_video"])) for row in pairs]
    left = sum(int(bool(row["with_video"])) for row in pairs)
    right = sum(int(bool(row["no_video"])) for row in pairs)
    observed = (left - right) / len(pairs)
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(deltas)
    for _ in range(repeats):
        samples.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()

    def quantile(q: float) -> float:
        position = q * (len(samples) - 1)
        low = int(position)
        high = min(low + 1, len(samples) - 1)
        weight = position - low
        return samples[low] * (1.0 - weight) + samples[high] * weight

    return {
        "pairs": len(pairs),
        "with_video_successes": left,
        "no_video_successes": right,
        "observed_gap": observed,
        "ci95": [quantile(0.025), quantile(0.975)],
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
    }


def suite_differences(pairs: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        buckets[str(row["suite"])].append(row)
    result: dict[str, dict[str, float]] = {}
    for suite, rows in sorted(buckets.items()):
        left = sum(int(bool(row["with_video"])) for row in rows)
        right = sum(int(bool(row["no_video"])) for row in rows)
        result[suite] = {
            "pairs": len(rows),
            "with_video_rate": left / len(rows) if rows else 0.0,
            "no_video_rate": right / len(rows) if rows else 0.0,
            "difference": (left - right) / len(rows) if rows else 0.0,
        }
    return result


def compare_causal(
    with_video: Mapping[str, Any],
    no_video: Mapping[str, Any],
    *,
    repeats: int = 10000,
    seed: int = 20260817,
    contract_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pairs = paired_rows_from_summaries(with_video, no_video)
    bootstrap = paired_bootstrap(pairs, repeats=repeats, seed=seed)
    suite = suite_differences(pairs)
    invalid = int(with_video.get("invalid_rollouts", 0)) + int(
        no_video.get("invalid_rollouts", 0)
    )
    return {
        "schema_version": 1,
        "with_video": {
            "overall_success_rate": with_video.get("overall_success_rate"),
            "status": with_video.get("status"),
        },
        "no_video": {
            "overall_success_rate": no_video.get("overall_success_rate"),
            "status": no_video.get("status"),
        },
        "pairs": pairs,
        "bootstrap": bootstrap,
        "per_suite": suite,
        "invalid_rollouts": invalid,
        "contract": dict(contract_fields or {}),
    }


def deployment_gate(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    required = {
        "output_shape": ["B", 32, 7],
        "future_noisy_video_token_allocations": 0,
        "future_video_denoise_steps": 0,
        "future_vae_decode_calls": 0,
        "n_action_steps": 10,
        "gripper_contract": "toggle_last_dimension",
    }
    reasons: list[str] = []
    if payload.get("status") != "pass":
        reasons.append("deployment evidence status is not pass")
    for key, expected in required.items():
        if payload.get(key) != expected:
            reasons.append(f"{key} != {expected!r}")
    return not reasons, reasons


def gate_w0(
    released: Mapping[str, Any],
    causal: Mapping[str, Any],
    deployment: Mapping[str, Any],
    *,
    lerobot: Mapping[str, Any] | None = None,
    causal_min_gap: float = 0.02,
    suite_reference: Mapping[str, float] | None = None,
    suite_tolerance: float = 0.05,
    expected_bootstrap_repeats: int | None = None,
    expected_bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    artifact_reasons: list[str] = []
    if released.get("status") != "complete":
        artifact_reasons.append("released evaluation is incomplete")
    if int(released.get("num_tasks", 0)) != 40 or int(released.get("num_episodes", 0)) != 2000:
        artifact_reasons.append("released evaluation is not 40 tasks x 50 trials")
    if float(released.get("overall_success_rate", 0.0)) < 0.946:
        artifact_reasons.append("released overall success rate is below 94.6%")
    references = dict(
        suite_reference
        or {
            "libero_spatial": 0.982,
            "libero_object": 1.0,
            "libero_goal": 0.97,
            "libero_10": 0.952,
        }
    )
    for suite, reference in references.items():
        observed = float(released.get("per_suite", {}).get(suite, {}).get("success_rate", -1.0))
        if observed < reference - suite_tolerance:
            artifact_reasons.append(f"{suite} is below its reference floor")
    artifact_pass = not artifact_reasons

    causal_reasons: list[str] = []
    bootstrap = causal.get("bootstrap", {})
    if expected_bootstrap_repeats is not None and int(
        bootstrap.get("bootstrap_repeats", -1)
    ) != expected_bootstrap_repeats:
        causal_reasons.append("paired bootstrap repeat count does not match the preregistration")
    if expected_bootstrap_seed is not None and int(bootstrap.get("bootstrap_seed", -1)) != expected_bootstrap_seed:
        causal_reasons.append("paired bootstrap seed does not match the preregistration")
    if float(bootstrap.get("observed_gap", -1.0)) < causal_min_gap:
        causal_reasons.append("causal gap is below 2 percentage points")
    ci = bootstrap.get("ci95", [-1.0, 0.0])
    if not ci or float(ci[0]) <= 0.0:
        causal_reasons.append("paired bootstrap 95% CI lower bound is not positive")
    nonnegative = sum(
        1 for row in causal.get("per_suite", {}).values() if float(row.get("difference", -1.0)) >= 0
    )
    if nonnegative < 3:
        causal_reasons.append("fewer than 3 suites have nonnegative point differences")
    contract = causal.get("contract", {})
    if contract.get("equal_init_data_noise_update_batch") is not True:
        causal_reasons.append("causal pair contract hashes/batch are not verified equal")
    if int(causal.get("invalid_rollouts", 1)) != 0:
        causal_reasons.append("causal evaluation contains invalid rollouts")
    causal_pass = not causal_reasons

    deployment_pass, deployment_reasons = deployment_gate(deployment)
    lerobot_pass = bool(lerobot and lerobot.get("api_pass") is True)
    if not lerobot_pass:
        reasons.append("LeRobot integration is not marked api_pass")
    if not artifact_pass:
        reasons.extend(f"artifact: {item}" for item in artifact_reasons)
    if not causal_pass:
        reasons.extend(f"causal: {item}" for item in causal_reasons)
    if not deployment_pass:
        reasons.extend(f"deployment: {item}" for item in deployment_reasons)
    return {
        "schema_version": 1,
        "passed": artifact_pass and causal_pass and deployment_pass and lerobot_pass,
        "artifact_pass": artifact_pass,
        "causal_pass": causal_pass,
        "deployment_pass": deployment_pass,
        "lerobot_api_pass": lerobot_pass,
        "artifact_reasons": artifact_reasons,
        "causal_reasons": causal_reasons,
        "deployment_reasons": deployment_reasons,
        "reasons": reasons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--root", required=True)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "aggregate":
        _write_json(args.output, aggregate_eval_directory(args.root))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
