#!/usr/bin/env python3
"""Measure how much of each LIBERO task's eval init-state set is covered by training demos.

Why this exists
---------------
The W0 closed-loop protocol runs 50 trials per task, and LIBERO supplies exactly 50
initial states per task via ``<task>.pruned_init``. LIBERO ships 50 demonstrations per
task, so the eval initial configurations are almost certainly the demonstration starting
configurations -- meaning the eval is "re-execute from a seen initial configuration",
not held-out generalization. That framing matters for how UP-REL / LR-REL numbers are
worded.

But the released training set is the ``_no_noops`` subset: 1,712 of 2,000 demos survive
the filter. So for each task, ``50 - n_train`` of the eval trials start from a
configuration whose demonstration was dropped. This script quantifies that per task and
joins it against the measured per-task success rates, which gives a cheap test of one
candidate explanation for the unexplained ``libero_spatial`` task 5 gap
(UP 96% vs LR 28%).

Caveat this script does NOT resolve
-----------------------------------
It does not prove ``pruned_init[i]`` equals demo ``i``'s ``states[0]`` byte-for-byte;
LIBERO ships no generation script and the raw HDF5 demos are not part of the pinned
artifacts. Coverage here is a *count* comparison (episodes retained vs init states
available), not a per-state identity match. Treat a coverage/success correlation as a
lead, not a proof.

Join key
--------
Task language string, not index. The LeRobot conversion orders tasks alphabetically while
LIBERO uses its own ``libero_task_map`` order, so ``task_index != task_id``.

Run inside an env where ``libero`` imports (``robot-practice``).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _libero_task_languages(suite: str) -> tuple[list[str], list[int]]:
    """Return per-task-id language strings and eval init-state counts, as eval sees them."""

    from libero.libero.benchmark import get_benchmark

    bench = get_benchmark(suite)()
    languages: list[str] = []
    init_counts: list[int] = []
    for task_id in range(bench.n_tasks):
        languages.append(bench.tasks[task_id].language)
        init_counts.append(int(len(bench.get_task_init_states(task_id))))
    return languages, init_counts


def _train_episode_counts(dataset_root: Path, suite: str) -> dict[str, int]:
    """Count retained training episodes per task language from the LeRobot metadata."""

    path = dataset_root / f"{suite}_no_noops_lerobot/meta/episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tasks = record.get("tasks") or []
            if not tasks:
                raise ValueError(f"episode without task language in {path}: {record}")
            counts[tasks[0]] = counts.get(tasks[0], 0) + 1
    return counts


def _up_rel_rates(summary: Path) -> dict[tuple[str, int], float]:
    if not summary.is_file():
        return {}
    payload = json.loads(summary.read_text(encoding="utf-8"))
    rates: dict[tuple[str, int], float] = {}
    for entry in payload.get("tasks", []):
        rates[(entry["suite"], int(entry["task_id"]))] = float(entry["success_rate"]) * 100.0
    return rates


def _lr_rel_rates(summary: Path) -> dict[tuple[str, int], float]:
    if not summary.is_file():
        return {}
    payload = json.loads(summary.read_text(encoding="utf-8"))
    rates: dict[tuple[str, int], float] = {}
    for suite, metrics in (payload.get("lerobot_metrics") or {}).items():
        for key, entry in (metrics.get("per_task") or {}).items():
            # keys look like "libero_spatial_task07"
            task_id = int(key.rsplit("task", 1)[-1])
            rates[(suite, task_id)] = float(entry["pc_success"])
    return rates


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def build_report(dataset_root: Path, up_summary: Path, lr_summary: Path) -> dict[str, Any]:
    up = _up_rel_rates(up_summary)
    lr = _lr_rel_rates(lr_summary)
    rows: list[dict[str, Any]] = []
    for suite in SUITES:
        languages, init_counts = _libero_task_languages(suite)
        counts = _train_episode_counts(dataset_root, suite)
        unmatched = set(counts) - set(languages)
        if unmatched:
            raise ValueError(
                f"{suite}: dataset languages absent from LIBERO task list: {sorted(unmatched)}"
            )
        for task_id, language in enumerate(languages):
            n_eval = init_counts[task_id]
            n_train = counts.get(language, 0)
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "language": language,
                    "eval_init_states": n_eval,
                    "train_episodes": n_train,
                    "uncovered_init_states": max(n_eval - n_train, 0),
                    "coverage": (n_train / n_eval) if n_eval else None,
                    "up_rel_pc": up.get((suite, task_id)),
                    "lr_rel_pc": lr.get((suite, task_id)),
                }
            )
    for row in rows:
        if row["up_rel_pc"] is not None and row["lr_rel_pc"] is not None:
            row["up_minus_lr_pp"] = round(row["up_rel_pc"] - row["lr_rel_pc"], 2)
        else:
            row["up_minus_lr_pp"] = None

    def _corr(field: str, rows_subset: list[dict[str, Any]] | None = None) -> float | None:
        source = rows if rows_subset is None else rows_subset
        pairs = [
            (row["coverage"], row[field])
            for row in source
            if row["coverage"] is not None and row[field] is not None
        ]
        if not pairs:
            return None
        value = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
        return None if value is None else round(value, 4)

    # Pooled correlation mixes suite difficulty with coverage (libero_10 has both the
    # lowest coverage and the hardest tasks), so the per-suite figures are the ones to
    # read. n=10 per suite: treat any of these as a lead, never as a result.
    per_suite_corr = {
        suite: {
            field: _corr(field, [row for row in rows if row["suite"] == suite])
            for field in ("up_rel_pc", "lr_rel_pc", "up_minus_lr_pp")
        }
        for suite in SUITES
    }

    total_train = sum(row["train_episodes"] for row in rows)
    total_eval = sum(row["eval_init_states"] for row in rows)
    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "join_key": "task language string (dataset task_index != LIBERO task_id)",
        "totals": {
            "train_episodes": total_train,
            "eval_init_states": total_eval,
            "uncovered_init_states": sum(row["uncovered_init_states"] for row in rows),
            "coverage": round(total_train / total_eval, 4) if total_eval else None,
        },
        "correlation_coverage_vs": {
            "up_rel_pc": _corr("up_rel_pc"),
            "lr_rel_pc": _corr("lr_rel_pc"),
            "up_minus_lr_pp": _corr("up_minus_lr_pp"),
        },
        "correlation_coverage_vs_per_suite": per_suite_corr,
        "tasks": rows,
    }


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "n/a"
    return format(value, spec) if spec else str(value)


def print_report(report: dict[str, Any], sort_by_coverage: bool) -> None:
    rows = list(report["tasks"])
    if sort_by_coverage:
        rows.sort(key=lambda r: (r["coverage"] if r["coverage"] is not None else 1.0, r["suite"]))
    header = (
        f"{'suite':15s} {'id':>2s} {'train':>5s} {'eval':>4s} {'cover':>6s} "
        f"{'UP%':>6s} {'LR%':>6s} {'UP-LR':>7s}  language"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['suite']:15s} {row['task_id']:2d} {row['train_episodes']:5d} "
            f"{row['eval_init_states']:4d} "
            f"{_fmt(row['coverage'], '6.3f')} "
            f"{_fmt(row['up_rel_pc'], '6.1f')} {_fmt(row['lr_rel_pc'], '6.1f')} "
            f"{_fmt(row['up_minus_lr_pp'], '7.1f')}  {row['language'][:58]}"
        )
    totals = report["totals"]
    print()
    print(
        f"totals: train={totals['train_episodes']} eval={totals['eval_init_states']} "
        f"uncovered={totals['uncovered_init_states']} coverage={_fmt(totals['coverage'], '.4f')}"
    )
    corr = report["correlation_coverage_vs"]
    print(
        "pearson(coverage, .) pooled: "
        f"UP%={_fmt(corr['up_rel_pc'])}  LR%={_fmt(corr['lr_rel_pc'])}  "
        f"UP-LR={_fmt(corr['up_minus_lr_pp'])}"
    )
    print("pearson(coverage, .) per suite (n=10 each; read these, not the pooled row):")
    for suite, values in report["correlation_coverage_vs_per_suite"].items():
        print(
            f"  {suite:15s} UP%={_fmt(values['up_rel_pc']):>8s}  "
            f"LR%={_fmt(values['lr_rel_pc']):>8s}  UP-LR={_fmt(values['up_minus_lr_pp']):>8s}"
        )
    print(
        "notes: coverage is a count comparison, not per-state identity. It is also "
        "identical for UP-REL and LR-REL (same checkpoint, same training set), so it "
        "cannot explain any UP-minus-LR difference by construction."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2",
        help="directory holding <suite>_no_noops_lerobot",
    )
    parser.add_argument(
        "--up-summary",
        default="outputs/literature/w0_fastwam/upstream_released/summary.json",
    )
    parser.add_argument(
        "--lr-summary",
        default="outputs/literature/w0_fastwam/lerobot_released_visualfix/summary.json",
    )
    parser.add_argument("--output", help="write the JSON report here")
    parser.add_argument(
        "--sort-by-coverage",
        action="store_true",
        help="sort rows by ascending coverage instead of suite order",
    )
    args = parser.parse_args()

    report = build_report(
        Path(args.dataset_root), Path(args.up_summary), Path(args.lr_summary)
    )
    print_report(report, args.sort_by_coverage)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
