"""Formal held-out closed-loop test for the selected SmolVLA LIBERO B0 checkpoint."""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_practice.evaluation.libero_assets import validate_libero_assets_dir
from robot_practice.experiments.smolvla_libero.b0_checkpoint_selection import (
    EXPECTED_TASK_IDS,
    EXPECTED_TEST_STATES,
    REQUIRED_CHECKPOINT_FILES,
    SelectionJob,
    _read_json,
    _selection_protocol,
    _sha256_file,
    build_evaluation_command,
    load_selection_split,
    run_evaluation_jobs,
    summarize_checkpoint,
)
from robot_practice.experiments.smolvla_libero.b0_training import (
    portable_path,
    resolve_repository_path,
    write_json_atomic,
)
from robot_practice.experiments.smolvla_libero.b0_validation_analysis import (
    ValidationAnalysisArguments,
    analyze_validation,
)

LOGGER = logging.getLogger(__name__)
RESULT_SET = "background_test"


@dataclass(frozen=True)
class TestArguments:
    run_dir: Path
    selection_dir: Path
    split_file: Path
    num_gpus: int
    env_seed: int
    policy_seed: int
    n_action_steps: int
    libero_assets_dir: Path
    output_dir: Path
    resume: bool
    overwrite: bool
    print_command: bool


@dataclass(frozen=True)
class SelectedPolicy:
    step: int
    checkpoint_dir: Path
    policy_dir: Path
    selection: dict[str, Any]


def load_selected_policy(arguments: TestArguments, split: dict[str, Any]) -> SelectedPolicy:
    """Resolve the frozen selection artifact without depending on checkpoints/selected."""

    selection_path = arguments.selection_dir / "selected_checkpoint.json"
    results_path = arguments.selection_dir / "checkpoint_results.json"
    if not selection_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            "Checkpoint selection is incomplete; expected selected_checkpoint.json and "
            f"checkpoint_results.json under {arguments.selection_dir}"
        )
    selection = _read_json(selection_path)
    results = _read_json(results_path)
    if selection.get("status") != "complete" or selection.get("selection_set") != "background_validation":
        raise ValueError(f"Not a complete B0 background validation selection: {selection_path}")
    if results.get("status") != "complete" or not isinstance(results.get("ranking"), list):
        raise ValueError(f"Not a complete B0 checkpoint ranking: {results_path}")

    split_sha256 = _sha256_file(arguments.split_file)
    if selection.get("split_sha256") != split_sha256:
        raise ValueError("Selection split hash does not match the frozen test split")
    expected_steps, frozen_policy_seed = _selection_protocol(split)
    if arguments.policy_seed != frozen_policy_seed:
        raise ValueError(
            f"--policy-seed={arguments.policy_seed} violates the frozen value {frozen_policy_seed}"
        )
    step = selection.get("selected_checkpoint_step")
    if isinstance(step, bool) or not isinstance(step, int) or step not in expected_steps:
        raise ValueError(f"Invalid selected checkpoint step: {step!r}")
    ranking = results["ranking"]
    if not ranking or ranking[0].get("checkpoint_step") != step:
        raise ValueError("selected_checkpoint.json does not match checkpoint_results.json winner")

    expected_checkpoint_dir = (arguments.run_dir / "checkpoints" / f"{step:06d}").resolve()
    source_checkpoint = selection.get("source_checkpoint")
    if not isinstance(source_checkpoint, str):
        raise TypeError("Selection source_checkpoint must be a path string")
    checkpoint_dir = resolve_repository_path(source_checkpoint)
    if checkpoint_dir != expected_checkpoint_dir:
        raise ValueError(
            "Selection source checkpoint does not belong to the requested run: "
            f"{portable_path(checkpoint_dir)}"
        )
    policy_dir = checkpoint_dir / "pretrained_model"
    missing = [filename for filename in REQUIRED_CHECKPOINT_FILES if not (policy_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Selected checkpoint {step:06d} is incomplete; missing {', '.join(missing)}"
        )
    return SelectedPolicy(
        step=step,
        checkpoint_dir=checkpoint_dir,
        policy_dir=policy_dir,
        selection=selection,
    )


def build_test_jobs(
    *,
    selected: SelectedPolicy,
    split: dict[str, Any],
    output_dir: Path,
) -> list[SelectionJob]:
    jobs: list[SelectionJob] = []
    for task_id in EXPECTED_TASK_IDS:
        init_state_ids = tuple(split["tasks"][str(task_id)]["test_init_state_ids"])
        if len(init_state_ids) != EXPECTED_TEST_STATES:
            raise ValueError(f"Task {task_id} must have exactly {EXPECTED_TEST_STATES} test states")
        jobs.append(
            SelectionJob(
                checkpoint_step=selected.step,
                checkpoint_path=selected.policy_dir,
                task_id=task_id,
                init_state_ids=init_state_ids,
                output_dir=output_dir / "jobs" / f"task_{task_id:02d}",
                log_path=output_dir / "logs" / f"task_{task_id:02d}.log",
            )
        )
    return jobs


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_test(
    *,
    selected_step: int,
    split: dict[str, Any],
    task_metrics: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    checkpoint_summary = summarize_checkpoint(selected_step, task_metrics)
    per_task: dict[str, Any] = {}
    all_termination_reasons: Counter[str] = Counter()
    task_aggregate_metric_names = (
        "action_tv",
        "action_jerk",
        "mean_gripper_toggles",
        "policy_step_latency_p50_ms",
        "policy_step_latency_p95_ms",
        "model_query_latency_p50_ms",
        "model_query_latency_p95_ms",
    )
    task_aggregate_values: dict[str, list[float]] = {
        name: [] for name in task_aggregate_metric_names
    }
    for task_id in EXPECTED_TASK_IDS:
        metrics = task_metrics[task_id]
        aggregate = metrics["aggregate"]
        episodes = metrics["per_episode"]
        successful_steps = [int(item["num_steps"]) for item in episodes if item["success"]]
        termination_reasons = Counter(str(item["termination_reason"]) for item in episodes)
        all_termination_reasons.update(termination_reasons)
        for name in task_aggregate_metric_names:
            value = aggregate.get(name)
            if value is not None:
                task_aggregate_values[name].append(float(value))
        per_task[str(task_id)] = {
            "canonical": split["tasks"][str(task_id)]["canonical"],
            "num_episodes": int(aggregate["num_episodes"]),
            "num_successes": int(aggregate["num_successes"]),
            "success_rate": float(aggregate["success_rate"]),
            "avg_max_reward": float(aggregate["avg_max_reward"]),
            "mean_successful_episode_steps": _mean(successful_steps),
            "termination_reasons": dict(sorted(termination_reasons.items())),
        }

    aggregate = {
        key: value
        for key, value in checkpoint_summary.items()
        if key not in {"checkpoint_step", "per_task"}
    }
    aggregate["termination_reasons"] = dict(sorted(all_termination_reasons.items()))
    for name, values in task_aggregate_values.items():
        aggregate[f"macro_task_{name}"] = _mean(values)
    return {
        "schema_version": 1,
        "status": "complete",
        "result_set": RESULT_SET,
        "suite": "libero_goal",
        "selected_checkpoint_step": selected_step,
        "aggregate": aggregate,
        "per_task": per_task,
    }


def _write_metrics_markdown(path: Path, result: dict[str, Any]) -> None:
    aggregate = result["aggregate"]
    lines = [
        "# SmolVLA B0 held-out test",
        "",
        "结果类别：Formal / Closed-loop / Test。",
        "",
        f"Selected checkpoint: `{result['selected_checkpoint_step']:06d}`.",
        "",
        (
            f"Overall macro success rate: `{aggregate['macro_success_rate']:.4f}` "
            f"({aggregate['num_successes']}/{aggregate['num_episodes']})."
        ),
        "",
        "| Task | Instruction | Successes | SR | Mean max reward | Successful steps |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for task_id in EXPECTED_TASK_IDS:
        task = result["per_task"][str(task_id)]
        successful_steps = task["mean_successful_episode_steps"]
        lines.append(
            f"| {task_id} | {task['canonical']} | "
            f"{task['num_successes']}/{task['num_episodes']} | "
            f"{task['success_rate']:.4f} | {task['avg_max_reward']:.4f} | "
            f"{'N/A' if successful_steps is None else f'{successful_steps:.2f}'} |"
        )
    lines.extend(
        [
            "",
            "This held-out test result is descriptive only and must not be used to reselect a checkpoint.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_output(arguments: TestArguments) -> None:
    if arguments.output_dir.exists():
        if arguments.resume:
            return
        if not arguments.overwrite:
            raise FileExistsError(
                f"Test output directory already exists: {arguments.output_dir}; "
                "use --resume or --overwrite"
            )
        shutil.rmtree(arguments.output_dir)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)


def evaluate_selected_checkpoint(arguments: TestArguments) -> dict[str, Any]:
    assets_dir = validate_libero_assets_dir(arguments.libero_assets_dir)
    split = load_selection_split(arguments.split_file)
    selected = load_selected_policy(arguments, split)
    jobs = build_test_jobs(selected=selected, split=split, output_dir=arguments.output_dir)
    total_rollouts = sum(job.rollout_count for job in jobs)
    validation_analysis_arguments = ValidationAnalysisArguments(
        run_dir=arguments.run_dir,
        selection_dir=arguments.selection_dir,
        split_file=arguments.split_file,
        env_seed=arguments.env_seed,
        policy_seed=arguments.policy_seed,
        n_action_steps=arguments.n_action_steps,
        output_json=arguments.selection_dir / "validation_paired_analysis.json",
        output_markdown=arguments.selection_dir / "validation_paired_analysis.md",
    )

    if arguments.print_command:
        validation_analysis = analyze_validation(
            validation_analysis_arguments,
            write_outputs=False,
        )
        payload = {
            "result_set": RESULT_SET,
            "selected_checkpoint_step": selected.step,
            "policy_path": portable_path(selected.policy_dir),
            "tasks": list(EXPECTED_TASK_IDS),
            "test_rollouts": total_rollouts,
            "num_gpus": arguments.num_gpus,
            "output_dir": portable_path(arguments.output_dir),
            "validation_pairwise": [
                {
                    key: pair[key]
                    for key in (
                        "from_step",
                        "to_step",
                        "gains",
                        "regressions",
                        "net_success_gain",
                        "exact_mcnemar_pvalue",
                    )
                }
                for pair in validation_analysis["pairwise"]
            ],
            "first_command": build_evaluation_command(
                jobs[0],
                env_seed=arguments.env_seed,
                policy_seed=arguments.policy_seed,
                n_action_steps=arguments.n_action_steps,
                libero_assets_dir=assets_dir,
                overwrite=arguments.overwrite,
            ),
        }
        print(__import__("json").dumps(payload, indent=2))
        return payload

    _prepare_output(arguments)
    validation_analysis = analyze_validation(validation_analysis_arguments)
    manifest_path = arguments.output_dir / "test_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "status": "running",
            "result_set": RESULT_SET,
            "arguments": {
                **asdict(arguments),
                "run_dir": portable_path(arguments.run_dir),
                "selection_dir": portable_path(arguments.selection_dir),
                "split_file": portable_path(arguments.split_file),
                "libero_assets_dir": portable_path(assets_dir),
                "output_dir": portable_path(arguments.output_dir),
            },
            "selected_checkpoint_step": selected.step,
            "policy_path": portable_path(selected.policy_dir),
            "selection_file": portable_path(arguments.selection_dir / "selected_checkpoint.json"),
            "selection_sha256": _sha256_file(arguments.selection_dir / "selected_checkpoint.json"),
            "validation_analysis_file": portable_path(
                validation_analysis_arguments.output_json
            ),
            "validation_analysis_sha256": _sha256_file(
                validation_analysis_arguments.output_json
            ),
            "validation_analysis_result_set": validation_analysis["result_set"],
            "split_sha256": _sha256_file(arguments.split_file),
            "task_ids": list(EXPECTED_TASK_IDS),
            "rollouts_per_task": EXPECTED_TEST_STATES,
            "total_rollouts": total_rollouts,
        },
    )
    collected = run_evaluation_jobs(
        jobs,
        num_gpus=arguments.num_gpus,
        env_seed=arguments.env_seed,
        policy_seed=arguments.policy_seed,
        n_action_steps=arguments.n_action_steps,
        libero_assets_dir=assets_dir,
        resume=arguments.resume,
        overwrite=arguments.overwrite,
        progress_description="B0 held-out test",
    )
    result = summarize_test(
        selected_step=selected.step,
        split=split,
        task_metrics=collected[selected.step],
    )
    write_json_atomic(arguments.output_dir / "test_results.json", result)
    _write_metrics_markdown(arguments.output_dir / "metrics.md", result)
    manifest = _read_json(manifest_path)
    manifest["status"] = "complete"
    manifest["macro_success_rate"] = result["aggregate"]["macro_success_rate"]
    write_json_atomic(manifest_path, manifest)
    LOGGER.info("B0 held-out test complete: %s", arguments.output_dir / "metrics.md")
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
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--policy-seed", type=int, default=2000)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument(
        "--libero-assets-dir",
        type=Path,
        default=Path("datasets/raw/libero_assets"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser


def _resolve_arguments(namespace: argparse.Namespace) -> TestArguments:
    run_dir = resolve_repository_path(namespace.run_dir)
    selection_dir = resolve_repository_path(namespace.selection_dir or run_dir / "checkpoint_selection")
    split_file = resolve_repository_path(namespace.split_file)
    libero_assets_dir = resolve_repository_path(namespace.libero_assets_dir)
    output_dir = resolve_repository_path(namespace.output_dir or run_dir / "background_test")
    if namespace.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive")
    if namespace.env_seed < 0 or namespace.policy_seed < 0:
        raise ValueError("--env-seed and --policy-seed must be non-negative")
    if namespace.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    if namespace.resume and namespace.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if output_dir in {run_dir, selection_dir}:
        raise ValueError("Test output must not replace the run or checkpoint selection directory")
    return TestArguments(
        run_dir=run_dir,
        selection_dir=selection_dir,
        split_file=split_file,
        num_gpus=namespace.num_gpus,
        env_seed=namespace.env_seed,
        policy_seed=namespace.policy_seed,
        n_action_steps=namespace.n_action_steps,
        libero_assets_dir=libero_assets_dir,
        output_dir=output_dir,
        resume=namespace.resume,
        overwrite=namespace.overwrite,
        print_command=namespace.print_command,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    namespace = build_argument_parser().parse_args()
    evaluate_selected_checkpoint(_resolve_arguments(namespace))


if __name__ == "__main__":
    main()
