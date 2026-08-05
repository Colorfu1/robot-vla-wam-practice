"""Paired background-task regression analysis between B0 and a finetuned policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .b0_checkpoint_selection import EXPECTED_TASK_IDS
from .b0_training import write_json_atomic
from .b0_validation_analysis import exact_mcnemar_pvalue

SCHEMA_VERSION = 1
RESULT_SET = "background_regression"


def _load_episode_map(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["task_id"]), int(row["init_state_id"]))
        if key in result:
            raise ValueError(f"Duplicate episode record for {key}")
        result[key] = row
    return result


def _success(row: dict[str, Any]) -> bool:
    return bool(row["success"])


def analyze_regression(
    *,
    baseline_path: Path,
    finetuned_path: Path,
    baseline_label: str,
    finetuned_label: str,
) -> dict[str, Any]:
    baseline = _load_episode_map(baseline_path)
    finetuned = _load_episode_map(finetuned_path)
    if set(baseline) != set(finetuned):
        missing_baseline = sorted(set(finetuned) - set(baseline))
        missing_finetuned = sorted(set(baseline) - set(finetuned))
        raise ValueError(
            "Episode maps are not paired; missing from baseline: "
            f"{missing_baseline[:5]}; missing from finetuned: {missing_finetuned[:5]}"
        )

    per_task: dict[str, Any] = {}
    total_gains = 0
    total_regressions = 0
    common_success_deltas: list[float] = []
    regression_cases: list[dict[str, Any]] = []
    termination_transitions: Counter[tuple[str, str]] = Counter()
    for task_id in EXPECTED_TASK_IDS:
        keys = [key for key in baseline if key[0] == task_id]
        keys.sort()
        task_gains = 0
        task_regressions = 0
        task_common_deltas: list[float] = []
        for key in keys:
            base = baseline[key]
            fine = finetuned[key]
            base_ok = _success(base)
            fine_ok = _success(fine)
            if base_ok and not fine_ok:
                task_regressions += 1
                total_regressions += 1
                regression_cases.append(
                    {
                        "task_id": task_id,
                        "init_state_id": key[1],
                        "baseline_success": True,
                        "finetuned_success": False,
                        "baseline_termination_reason": str(base["termination_reason"]),
                        "finetuned_termination_reason": str(fine["termination_reason"]),
                    }
                )
                termination_transitions[
                    (
                        str(base["termination_reason"]),
                        str(fine["termination_reason"]),
                    )
                ] += 1
            if not base_ok and fine_ok:
                task_gains += 1
                total_gains += 1
            if base_ok and fine_ok:
                delta = float(fine["num_steps"]) - float(base["num_steps"])
                task_common_deltas.append(delta)
                common_success_deltas.append(delta)
        discordant = task_gains + task_regressions
        per_task[str(task_id)] = {
            "num_cases": len(keys),
            "baseline_successes": sum(1 for key in keys if _success(baseline[key])),
            "finetuned_successes": sum(1 for key in keys if _success(finetuned[key])),
            "gains": task_gains,
            "regressions": task_regressions,
            "discordant": discordant,
            "exact_mcnemar_pvalue": (
                exact_mcnemar_pvalue(task_gains, task_regressions) if discordant else 1.0
            ),
            "common_success_step_delta_mean": (
                float(np.mean(task_common_deltas)) if task_common_deltas else None
            ),
            "common_success_cases": len(task_common_deltas),
        }
    discordant = total_gains + total_regressions
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "result_set": RESULT_SET,
        "baseline_label": baseline_label,
        "finetuned_label": finetuned_label,
        "baseline_path": str(baseline_path),
        "finetuned_path": str(finetuned_path),
        "case_count": len(baseline),
        "per_task": per_task,
        "totals": {
            "baseline_successes": sum(1 for key in baseline if _success(baseline[key])),
            "finetuned_successes": sum(1 for key in finetuned if _success(finetuned[key])),
            "gains": total_gains,
            "regressions": total_regressions,
            "discordant": discordant,
            "exact_mcnemar_pvalue": (
                exact_mcnemar_pvalue(total_gains, total_regressions) if discordant else 1.0
            ),
            "common_success_step_delta_mean": (
                float(np.mean(common_success_deltas)) if common_success_deltas else None
            ),
            "common_success_cases": len(common_success_deltas),
        },
        "regression_cases": regression_cases,
        "termination_transitions": {
            " | ".join(pair): count for pair, count in sorted(termination_transitions.items())
        },
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    totals = result["totals"]
    lines = [
        "# C1 背景任务回退配对分析",
        "",
        "结果类别：Formal / Closed-loop / Test（配对分析）。",
        "",
        (
            f"{result['baseline_label']} vs {result['finetuned_label']}："
            f"{totals['baseline_successes']}/{result['case_count']} → "
            f"{totals['finetuned_successes']}/{result['case_count']}，"
            f"gains={totals['gains']}，regressions={totals['regressions']}，"
            f"exact McNemar p={totals['exact_mcnemar_pvalue']:.6f}。"
        ),
        "",
        "| Task | B0 SR | C1 SR | Gains | Regressions | McNemar p |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task_id in EXPECTED_TASK_IDS:
        task = result["per_task"][str(task_id)]
        lines.append(
            f"| {task_id} | "
            f"{task['baseline_successes']}/{task['num_cases']} | "
            f"{task['finetuned_successes']}/{task['num_cases']} | "
            f"{task['gains']} | {task['regressions']} | "
            f"{task['exact_mcnemar_pvalue']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-parquet", required=True, type=Path)
    parser.add_argument("--finetuned-parquet", required=True, type=Path)
    parser.add_argument("--baseline-label", default="b0_selected")
    parser.add_argument("--finetuned-label", default="c1_selected")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    return parser


def main() -> None:
    namespace = build_argument_parser().parse_args()
    result = analyze_regression(
        baseline_path=namespace.baseline_parquet,
        finetuned_path=namespace.finetuned_parquet,
        baseline_label=namespace.baseline_label,
        finetuned_label=namespace.finetuned_label,
    )
    write_json_atomic(namespace.output_json, result)
    _write_markdown(namespace.output_markdown, result)
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
