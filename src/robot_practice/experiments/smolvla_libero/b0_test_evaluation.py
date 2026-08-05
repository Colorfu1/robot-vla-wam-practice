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
import pyarrow as pa

from robot_practice.evaluation.common import write_parquet_atomic
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
    policy_path: Path | None = None
    model_label: str | None = None


@dataclass(frozen=True)
class SelectedPolicy:
    step: int
    checkpoint_dir: Path
    policy_dir: Path
    selection: dict[str, Any]


def _load_direct_policy(arguments: TestArguments, split: dict[str, Any]) -> SelectedPolicy:
    """Resolve an explicitly supplied policy without any selection provenance."""

    if arguments.policy_path is None or arguments.model_label is None:
        raise ValueError("Direct policy mode requires both --policy-path and --model-label")
    _expected_steps, frozen_policy_seed = _selection_protocol(split)
    if arguments.policy_seed != frozen_policy_seed:
        raise ValueError(
            f"--policy-seed={arguments.policy_seed} violates the frozen value {frozen_policy_seed}"
        )
    if arguments.env_seed != 1000:
        raise ValueError("--env-seed must be 1000 for the frozen background test protocol")
    if arguments.n_action_steps != 10:
        raise ValueError("--n-action-steps must be 10 for the frozen background test protocol")
    policy_dir = resolve_repository_path(arguments.policy_path)
    missing = [
        filename for filename in REQUIRED_CHECKPOINT_FILES if not (policy_dir / filename).is_file()
    ]
    if missing:
        # LoRA checkpoints (C2/C3) replace the full model file with a complete adapter pair.
        missing_without_weights = [name for name in missing if name != "model.safetensors"]
        adapter_complete = all(
            (policy_dir / name).is_file()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        )
        if missing_without_weights or not adapter_complete:
            raise FileNotFoundError(
                f"Policy checkpoint is incomplete; missing {', '.join(missing)}"
            )
    return SelectedPolicy(
        step=0,
        checkpoint_dir=policy_dir.parent,
        policy_dir=policy_dir,
        selection={
            "model_label": arguments.model_label,
            "policy_path": portable_path(policy_dir),
        },
    )


def _policy_run_is_diagnostic(policy_dir: Path) -> bool:
    run_config = policy_dir.parent.parent / "run_config.json"
    if not run_config.is_file():
        return False
    payload = _read_json(run_config)
    training = payload.get("training")
    return isinstance(training, dict) and training.get("mode") == "diagnostic"


def background_result_set_name(*, policy_dir: Path) -> str:
    return "diagnostic_background_test" if _policy_run_is_diagnostic(policy_dir) else RESULT_SET


def load_selected_policy(arguments: TestArguments, split: dict[str, Any]) -> SelectedPolicy:
    """Resolve the selected B0 policy or an explicitly supplied policy path."""

    if arguments.policy_path is not None:
        return _load_direct_policy(arguments, split)
    return _load_selection_policy(arguments, split)


def _load_selection_policy(arguments: TestArguments, split: dict[str, Any]) -> SelectedPolicy:
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
    if (
        selection.get("status") != "complete"
        or selection.get("selection_set") != "background_validation"
    ):
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
    missing = [
        filename for filename in REQUIRED_CHECKPOINT_FILES if not (policy_dir / filename).is_file()
    ]
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
    result_set: str = RESULT_SET,
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
        "result_set": result_set,
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


def collect_background_episode_records(
    jobs: list[SelectionJob],
    task_metrics: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect (task, init_state) granular records aligned with target eval."""

    records: list[dict[str, Any]] = []
    for job in jobs:
        metrics = task_metrics[job.task_id]
        for episode in metrics["per_episode"]:
            records.append(
                {
                    "task_id": int(job.task_id),
                    "init_state_id": int(episode["init_state_id"]),
                    "episode_index": int(episode["episode_index"]),
                    "seed": int(episode["seed"]),
                    "num_steps": int(episode["num_steps"]),
                    "success": bool(episode["success"]),
                    "termination_reason": str(episode["termination_reason"]),
                }
            )
    return records


def rebuild_episodes_from_jobs(
    *,
    output_dir: Path,
    split: dict[str, Any],
) -> Path:
    """Rebuild episodes.parquet from the original per-task background test jobs.

    The original B0 background test did not write a consolidated episode table,
    but every task job stores its own 30-row ``per_episode`` record together
    with the frozen protocol in ``run_config.json``. This rebuild validates the
    protocol and re-emits the exact same 210 rows the paired regression analysis
    needs, so the B0 baseline does not have to be rerun.
    """

    records: list[dict[str, Any]] = []
    for task_id in EXPECTED_TASK_IDS:
        job_dir = output_dir / "jobs" / f"task_{task_id:02d}"
        metrics_path = job_dir / "metrics.json"
        run_config_path = job_dir / "run_config.json"
        if not metrics_path.is_file() or not run_config_path.is_file():
            raise FileNotFoundError(
                f"Background test job is incomplete for task {task_id}: {job_dir}"
            )
        metrics = _read_json(metrics_path)
        run_config = _read_json(run_config_path)
        if metrics.get("status") != "complete":
            raise ValueError(f"Background test job {task_id} is not complete")
        aggregate = metrics.get("aggregate")
        episodes = metrics.get("per_episode")
        if (
            not isinstance(aggregate, dict)
            or aggregate.get("num_episodes") != EXPECTED_TEST_STATES
            or not isinstance(episodes, list)
            or len(episodes) != EXPECTED_TEST_STATES
        ):
            raise ValueError(
                f"Background test job {task_id} must have {EXPECTED_TEST_STATES} measured episodes"
            )
        run_input = run_config.get("input")
        policy_execution = (
            run_input.get("policy_execution") if isinstance(run_input, dict) else None
        )
        if not isinstance(run_input, dict) or not isinstance(policy_execution, dict):
            raise TypeError(f"Background test job {task_id} run_config is invalid")
        if run_input.get("policy_seed") != 2000:
            raise ValueError(f"Background test job {task_id} policy_seed is not frozen 2000")
        if run_input.get("seeds") != list(range(1000, 1030)):
            raise ValueError(f"Background test job {task_id} env seeds are not frozen 1000..1029")
        if policy_execution.get("effective_n_action_steps") != 10:
            raise ValueError(f"Background test job {task_id} n_action_steps is not frozen 10")
        expected_ids = tuple(split["tasks"][str(task_id)]["test_init_state_ids"])
        observed_ids = tuple(int(item["init_state_id"]) for item in episodes)
        if observed_ids != expected_ids:
            raise ValueError(
                f"Background test job {task_id} init states do not match the frozen split"
            )
        for episode in episodes:
            records.append(
                {
                    "task_id": int(task_id),
                    "init_state_id": int(episode["init_state_id"]),
                    "episode_index": int(episode["episode_index"]),
                    "seed": int(episode["seed"]),
                    "num_steps": int(episode["num_steps"]),
                    "success": bool(episode["success"]),
                    "termination_reason": str(episode["termination_reason"]),
                }
            )
    if len(records) != EXPECTED_TEST_STATES * len(EXPECTED_TASK_IDS):
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_STATES * len(EXPECTED_TASK_IDS)} episode records, "
            f"collected {len(records)}"
        )
    destination = output_dir / "episodes.parquet"
    write_parquet_atomic(destination, pa.Table.from_pylist(records))
    return destination


def evaluate_selected_checkpoint(arguments: TestArguments) -> dict[str, Any]:
    assets_dir = validate_libero_assets_dir(arguments.libero_assets_dir)
    split = load_selection_split(arguments.split_file)
    selected = load_selected_policy(arguments, split)
    direct_policy = arguments.policy_path is not None
    jobs = build_test_jobs(selected=selected, split=split, output_dir=arguments.output_dir)
    total_rollouts = sum(job.rollout_count for job in jobs)
    result_set = background_result_set_name(policy_dir=selected.policy_dir)
    validation_analysis_arguments = (
        ValidationAnalysisArguments(
            run_dir=arguments.run_dir,
            selection_dir=arguments.selection_dir,
            split_file=arguments.split_file,
            env_seed=arguments.env_seed,
            policy_seed=arguments.policy_seed,
            n_action_steps=arguments.n_action_steps,
            output_json=arguments.selection_dir / "validation_paired_analysis.json",
            output_markdown=arguments.selection_dir / "validation_paired_analysis.md",
        )
        if not direct_policy
        else None
    )

    if arguments.print_command:
        validation_pairwise = (
            [
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
                for pair in analyze_validation(
                    validation_analysis_arguments,
                    write_outputs=False,
                )["pairwise"]
            ]
            if not direct_policy
            else []
        )
        payload = {
            "result_set": result_set,
            "selected_checkpoint_step": selected.step,
            "model_label": arguments.model_label,
            "policy_path": portable_path(selected.policy_dir),
            "tasks": list(EXPECTED_TASK_IDS),
            "test_rollouts": total_rollouts,
            "num_gpus": arguments.num_gpus,
            "output_dir": portable_path(arguments.output_dir),
            "validation_pairwise": validation_pairwise,
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
    validation_analysis = (
        analyze_validation(validation_analysis_arguments)
        if validation_analysis_arguments is not None
        else None
    )
    manifest_path = arguments.output_dir / "test_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "status": "running",
            "result_set": result_set,
            "arguments": {
                **asdict(arguments),
                "run_dir": portable_path(arguments.run_dir),
                "selection_dir": portable_path(arguments.selection_dir),
                "split_file": portable_path(arguments.split_file),
                "libero_assets_dir": portable_path(assets_dir),
                "output_dir": portable_path(arguments.output_dir),
            },
            "selected_checkpoint_step": selected.step,
            "model_label": arguments.model_label,
            "policy_path": portable_path(selected.policy_dir),
            "selection_mode": "direct" if direct_policy else "selection",
            "selection_file": portable_path(arguments.selection_dir / "selected_checkpoint.json"),
            "selection_sha256": _sha256_file(arguments.selection_dir / "selected_checkpoint.json"),
            "validation_analysis_file": (
                portable_path(validation_analysis_arguments.output_json)
                if validation_analysis_arguments is not None
                else None
            ),
            "validation_analysis_sha256": (
                _sha256_file(validation_analysis_arguments.output_json)
                if validation_analysis_arguments is not None
                else None
            ),
            "validation_analysis_result_set": (
                validation_analysis["result_set"] if validation_analysis is not None else None
            ),
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
        progress_description=(
            f"Background test ({arguments.model_label})"
            if arguments.model_label is not None
            else "B0 held-out test"
        ),
    )
    result = summarize_test(
        selected_step=selected.step,
        split=split,
        task_metrics=collected[selected.step],
        result_set=result_set,
    )
    result["model_label"] = arguments.model_label
    records = collect_background_episode_records(jobs, collected[selected.step])
    if len(records) != total_rollouts:
        raise RuntimeError(f"Expected {total_rollouts} episode records, collected {len(records)}")
    write_parquet_atomic(
        arguments.output_dir / "episodes.parquet",
        pa.Table.from_pylist(records),
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
        "--policy-path",
        type=Path,
        help=(
            "Evaluate an arbitrary policy directly; requires --model-label and "
            "skips B0 selection provenance"
        ),
    )
    parser.add_argument(
        "--model-label",
        help="Policy label written into the manifest and results",
    )
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
    parser.add_argument(
        "--rebuild-episodes",
        action="store_true",
        help=(
            "Rebuild episodes.parquet from the original per-task job metrics "
            "without running any rollout"
        ),
    )
    return parser


def _resolve_arguments(namespace: argparse.Namespace) -> TestArguments:
    if namespace.policy_path is not None and namespace.model_label is None:
        raise ValueError("--policy-path requires --model-label")
    if namespace.model_label is not None and namespace.policy_path is None:
        raise ValueError("--model-label requires --policy-path")
    run_dir = resolve_repository_path(namespace.run_dir)
    selection_dir = resolve_repository_path(
        namespace.selection_dir or run_dir / "checkpoint_selection"
    )
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
        policy_path=(
            None
            if namespace.policy_path is None
            else resolve_repository_path(namespace.policy_path)
        ),
        model_label=namespace.model_label,
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
    if namespace.rebuild_episodes:
        split = load_selection_split(resolve_repository_path(namespace.split_file))
        output_dir = resolve_repository_path(
            namespace.output_dir or resolve_repository_path(namespace.run_dir) / "background_test"
        )
        destination = rebuild_episodes_from_jobs(output_dir=output_dir, split=split)
        print(f"Rebuilt background test episodes: {destination}")
        return
    evaluate_selected_checkpoint(_resolve_arguments(namespace))


if __name__ == "__main__":
    main()
