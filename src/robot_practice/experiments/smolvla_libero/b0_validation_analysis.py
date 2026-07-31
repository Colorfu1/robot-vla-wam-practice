"""Paired diagnostic of the 20k/25k/30k B0 validation rollouts."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_practice.experiments.smolvla_libero.b0_checkpoint_selection import (
    EXPECTED_TASK_IDS,
    EXPECTED_VALIDATION_STATES,
    SelectionJob,
    _completed_job_metrics,
    _read_json,
    _selection_protocol,
    _sha256_file,
    checkpoint_rank_key,
    discover_checkpoints,
    load_selection_split,
)
from robot_practice.experiments.smolvla_libero.b0_training import (
    portable_path,
    resolve_repository_path,
    write_json_atomic,
)

LOGGER = logging.getLogger(__name__)
COMPARISON_STEPS = (20000, 25000, 30000)
RESULT_SET = "background_validation_paired_diagnostic"


@dataclass(frozen=True)
class ValidationAnalysisArguments:
    run_dir: Path
    selection_dir: Path
    split_file: Path
    env_seed: int
    policy_seed: int
    n_action_steps: int
    output_json: Path
    output_markdown: Path


def exact_mcnemar_pvalue(gains: int, regressions: int) -> float:
    """Two-sided exact McNemar p-value over discordant paired outcomes."""

    if gains < 0 or regressions < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _load_selection_protocol(
    arguments: ValidationAnalysisArguments,
    split: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = arguments.selection_dir / "selected_checkpoint.json"
    results_path = arguments.selection_dir / "checkpoint_results.json"
    if not selection_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            "Validation analysis requires selected_checkpoint.json and checkpoint_results.json "
            f"under {arguments.selection_dir}"
        )
    selection = _read_json(selection_path)
    results = _read_json(results_path)
    if selection.get("status") != "complete" or selection.get("selection_set") != "background_validation":
        raise ValueError("Validation analysis requires a complete background validation selection")
    if selection.get("split_sha256") != _sha256_file(arguments.split_file):
        raise ValueError("Checkpoint selection and validation analysis split hashes differ")
    expected_steps, frozen_policy_seed = _selection_protocol(split)
    if arguments.policy_seed != frozen_policy_seed:
        raise ValueError(
            f"--policy-seed={arguments.policy_seed} violates the frozen value {frozen_policy_seed}"
        )
    if not set(COMPARISON_STEPS) <= set(expected_steps):
        raise ValueError(f"Frozen selection does not contain comparison steps {COMPARISON_STEPS}")
    if selection.get("selected_checkpoint_step") != 25000:
        raise ValueError(
            "The registered paired diagnostic expects selected checkpoint 025000; "
            f"got {selection.get('selected_checkpoint_step')!r}"
        )
    ranking = results.get("ranking")
    if results.get("status") != "complete" or not isinstance(ranking, list) or not ranking:
        raise ValueError("Validation analysis requires a complete checkpoint ranking")
    if ranking[0].get("checkpoint_step") != selection["selected_checkpoint_step"]:
        raise ValueError("Selected checkpoint does not match the first ranked checkpoint")
    return selection, results


def _load_episode_maps(
    arguments: ValidationAnalysisArguments,
    split: dict[str, Any],
) -> dict[int, dict[tuple[int, int], dict[str, Any]]]:
    checkpoints = discover_checkpoints(arguments.run_dir, COMPARISON_STEPS)
    by_step: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
    for step in COMPARISON_STEPS:
        episodes: dict[tuple[int, int], dict[str, Any]] = {}
        for task_id in EXPECTED_TASK_IDS:
            init_state_ids = tuple(split["tasks"][str(task_id)]["validation_init_state_ids"])
            job = SelectionJob(
                checkpoint_step=step,
                checkpoint_path=checkpoints[step],
                task_id=task_id,
                init_state_ids=init_state_ids,
                output_dir=(
                    arguments.selection_dir
                    / "jobs"
                    / f"checkpoint_{step:06d}"
                    / f"task_{task_id:02d}"
                ),
                log_path=(
                    arguments.selection_dir
                    / "logs"
                    / f"checkpoint_{step:06d}_task_{task_id:02d}.log"
                ),
            )
            metrics = _completed_job_metrics(
                job,
                env_seed=arguments.env_seed,
                policy_seed=arguments.policy_seed,
                n_action_steps=arguments.n_action_steps,
            )
            if metrics is None:
                raise RuntimeError(
                    f"Validation job checkpoint {step:06d} task {task_id} is incomplete or "
                    "does not match the frozen policy/seed/state protocol"
                )
            for episode in metrics["per_episode"]:
                key = (task_id, int(episode["init_state_id"]))
                if key in episodes:
                    raise ValueError(f"Duplicate validation episode key: {key}")
                episodes[key] = dict(episode)
        expected_count = len(EXPECTED_TASK_IDS) * EXPECTED_VALIDATION_STATES
        if len(episodes) != expected_count:
            raise ValueError(
                f"Checkpoint {step:06d} has {len(episodes)} paired cases, expected {expected_count}"
            )
        by_step[step] = episodes
    reference_keys = set(by_step[COMPARISON_STEPS[0]])
    for step, episodes in by_step.items():
        if set(episodes) != reference_keys:
            raise ValueError(f"Checkpoint {step:06d} does not contain the same paired validation cases")
    return by_step


def _checkpoint_summary(
    step: int,
    episodes: Mapping[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    per_task = {}
    for task_id in EXPECTED_TASK_IDS:
        task_episodes = [episode for (task, _), episode in episodes.items() if task == task_id]
        successes = sum(bool(episode["success"]) for episode in task_episodes)
        per_task[str(task_id)] = {
            "num_episodes": len(task_episodes),
            "num_successes": successes,
            "success_rate": successes / len(task_episodes),
        }
    num_successes = sum(bool(episode["success"]) for episode in episodes.values())
    successful_steps = [
        int(episode["num_steps"]) for episode in episodes.values() if episode["success"]
    ]
    return {
        "checkpoint_step": step,
        "num_episodes": len(episodes),
        "num_successes": num_successes,
        "macro_success_rate": float(
            np.mean([task["success_rate"] for task in per_task.values()])
        ),
        "mean_successful_episode_steps": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "per_task": per_task,
    }


def compare_episode_maps(
    *,
    from_step: int,
    to_step: int,
    from_episodes: Mapping[tuple[int, int], dict[str, Any]],
    to_episodes: Mapping[tuple[int, int], dict[str, Any]],
    split: dict[str, Any],
) -> dict[str, Any]:
    if set(from_episodes) != set(to_episodes):
        raise ValueError("Paired checkpoint comparisons require identical episode keys")
    gains: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    common_success_deltas: list[int] = []
    both_success = 0
    both_failure = 0
    per_task: dict[str, dict[str, int]] = {
        str(task_id): {
            "both_success": 0,
            "both_failure": 0,
            "gains": 0,
            "regressions": 0,
        }
        for task_id in EXPECTED_TASK_IDS
    }
    for task_id, init_state_id in sorted(from_episodes):
        before = from_episodes[(task_id, init_state_id)]
        after = to_episodes[(task_id, init_state_id)]
        before_success = bool(before["success"])
        after_success = bool(after["success"])
        task_counts = per_task[str(task_id)]
        case = {
            "task_id": task_id,
            "canonical": split["tasks"][str(task_id)]["canonical"],
            "init_state_id": init_state_id,
            "episode_index": int(before["episode_index"]),
            "from_num_steps": int(before["num_steps"]),
            "to_num_steps": int(after["num_steps"]),
        }
        if before_success and after_success:
            both_success += 1
            task_counts["both_success"] += 1
            common_success_deltas.append(int(after["num_steps"]) - int(before["num_steps"]))
        elif not before_success and not after_success:
            both_failure += 1
            task_counts["both_failure"] += 1
        elif not before_success and after_success:
            gains.append(case)
            task_counts["gains"] += 1
        else:
            regressions.append(case)
            task_counts["regressions"] += 1
    return {
        "from_step": from_step,
        "to_step": to_step,
        "num_paired_cases": len(from_episodes),
        "both_success": both_success,
        "both_failure": both_failure,
        "gains": len(gains),
        "regressions": len(regressions),
        "net_success_gain": len(gains) - len(regressions),
        "discordant_cases": len(gains) + len(regressions),
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(len(gains), len(regressions)),
        "common_success_cases": len(common_success_deltas),
        "mean_common_success_step_delta": (
            float(np.mean(common_success_deltas)) if common_success_deltas else None
        ),
        "median_common_success_step_delta": (
            float(np.median(common_success_deltas)) if common_success_deltas else None
        ),
        "per_task": per_task,
        "gained_cases": gains,
        "regressed_cases": regressions,
    }


def _ranking_audit(results: dict[str, Any]) -> dict[str, Any]:
    ranking = results["ranking"]
    reported_steps = [int(item["checkpoint_step"]) for item in ranking]
    recomputed = sorted(ranking, key=checkpoint_rank_key)
    recomputed_steps = [int(item["checkpoint_step"]) for item in recomputed]
    return {
        "reported_steps": reported_steps,
        "recomputed_steps": recomputed_steps,
        "matches_registered_tie_breakers": reported_steps == recomputed_steps,
        "full_precision": [
            {
                "checkpoint_step": int(item["checkpoint_step"]),
                "macro_success_rate": item["macro_success_rate"],
                "mean_max_reward": item["mean_max_reward"],
                "mean_successful_episode_steps": item["mean_successful_episode_steps"],
            }
            for item in ranking
        ],
    }


def _case_labels(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "None"
    return ", ".join(f"task-{case['task_id']}/state-{case['init_state_id']}" for case in cases)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# SmolVLA B0 validation paired analysis",
        "",
        "结果类别：Formal / Closed-loop / Validation diagnostic。",
        "",
        "该分析只复用 checkpoint selection 已产生的逐 episode 结果，不运行新 rollout，",
        "也不改变已经冻结的 selected checkpoint。",
        "",
        "## Checkpoint summaries",
        "",
        "| Step | Successes | Macro SR | Successful steps |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for step in COMPARISON_STEPS:
        summary = result["checkpoint_summaries"][str(step)]
        lines.append(
            f"| {step:06d} | {summary['num_successes']}/{summary['num_episodes']} | "
            f"{summary['macro_success_rate']:.4f} | "
            f"{summary['mean_successful_episode_steps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired transitions",
            "",
            (
                "| From → To | Gains | Regressions | Net | Discordant | McNemar p | "
                "Common-success Δsteps |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair in result["pairwise"]:
        delta = pair["mean_common_success_step_delta"]
        lines.append(
            f"| {pair['from_step']:06d} → {pair['to_step']:06d} | {pair['gains']} | "
            f"{pair['regressions']} | {pair['net_success_gain']:+d} | "
            f"{pair['discordant_cases']} | {pair['exact_mcnemar_pvalue']:.6f} | "
            f"{'N/A' if delta is None else f'{delta:+.2f}'} |"
        )
    for pair in result["pairwise"]:
        lines.extend(
            [
                "",
                f"### {pair['from_step']:06d} → {pair['to_step']:06d}",
                "",
                f"Gained cases: {_case_labels(pair['gained_cases'])}.",
                "",
                f"Regressed cases: {_case_labels(pair['regressed_cases'])}.",
            ]
        )
    audit = result["ranking_audit"]
    lines.extend(
        [
            "",
            "## Ranking audit",
            "",
            f"Reported order: `{audit['reported_steps']}`.",
            "",
            f"Recomputed order: `{audit['recomputed_steps']}`.",
            "",
            (
                "Registered tie-breakers match the stored order: "
                f"`{audit['matches_registered_tie_breakers']}`."
            ),
            "",
            "完整逐任务迁移和 case 明细保存在配套 JSON 中。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_validation(
    arguments: ValidationAnalysisArguments,
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    split = load_selection_split(arguments.split_file)
    selection, checkpoint_results = _load_selection_protocol(arguments, split)
    episode_maps = _load_episode_maps(arguments, split)
    pairwise = [
        compare_episode_maps(
            from_step=from_step,
            to_step=to_step,
            from_episodes=episode_maps[from_step],
            to_episodes=episode_maps[to_step],
            split=split,
        )
        for from_step, to_step in ((20000, 25000), (25000, 30000), (20000, 30000))
    ]
    result = {
        "schema_version": 1,
        "status": "complete",
        "result_set": RESULT_SET,
        "selected_checkpoint_step": int(selection["selected_checkpoint_step"]),
        "comparison_steps": list(COMPARISON_STEPS),
        "num_paired_cases": len(episode_maps[COMPARISON_STEPS[0]]),
        "checkpoint_summaries": {
            str(step): _checkpoint_summary(step, episode_maps[step]) for step in COMPARISON_STEPS
        },
        "pairwise": pairwise,
        "ranking_audit": _ranking_audit(checkpoint_results),
        "provenance": {
            "selection_file": portable_path(arguments.selection_dir / "selected_checkpoint.json"),
            "selection_sha256": _sha256_file(arguments.selection_dir / "selected_checkpoint.json"),
            "split_file": portable_path(arguments.split_file),
            "split_sha256": _sha256_file(arguments.split_file),
            "env_seed": arguments.env_seed,
            "policy_seed": arguments.policy_seed,
            "n_action_steps": arguments.n_action_steps,
        },
    }
    if write_outputs:
        write_json_atomic(arguments.output_json, result)
        arguments.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(arguments.output_markdown, result)
        LOGGER.info("B0 validation paired analysis: %s", arguments.output_markdown)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/background_closed_loop_split.json"),
    )
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--policy-seed", type=int, default=2000)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--print-json", action="store_true")
    return parser


def _resolve_arguments(namespace: argparse.Namespace) -> ValidationAnalysisArguments:
    run_dir = resolve_repository_path(namespace.run_dir)
    selection_dir = resolve_repository_path(namespace.selection_dir or run_dir / "checkpoint_selection")
    split_file = resolve_repository_path(namespace.split_file)
    if namespace.env_seed < 0 or namespace.policy_seed < 0:
        raise ValueError("--env-seed and --policy-seed must be non-negative")
    if namespace.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    return ValidationAnalysisArguments(
        run_dir=run_dir,
        selection_dir=selection_dir,
        split_file=split_file,
        env_seed=namespace.env_seed,
        policy_seed=namespace.policy_seed,
        n_action_steps=namespace.n_action_steps,
        output_json=resolve_repository_path(
            namespace.output_json or selection_dir / "validation_paired_analysis.json"
        ),
        output_markdown=resolve_repository_path(
            namespace.output_markdown or selection_dir / "validation_paired_analysis.md"
        ),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    namespace = build_argument_parser().parse_args()
    result = analyze_validation(_resolve_arguments(namespace))
    if namespace.print_json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
