from __future__ import annotations

import json
from pathlib import Path

import pytest

import robot_practice.experiments.smolvla_libero.b0_checkpoint_selection as selection
import robot_practice.experiments.smolvla_libero.b0_validation_analysis as analysis

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = (
    PRACTICE_ROOT
    / "configs"
    / "experiments"
    / "smolvla_libero"
    / "background_closed_loop_split.json"
)


def _complete_checkpoint(run_dir: Path, step: int) -> Path:
    policy = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    policy.mkdir(parents=True)
    for filename in selection.REQUIRED_CHECKPOINT_FILES:
        (policy / filename).write_text(filename, encoding="utf-8")
    return policy


def _success_indices(step: int) -> set[int]:
    if step == 20000:
        return set(range(10))
    if step == 25000:
        return set(range(12))
    if step == 30000:
        return set(range(11)) | {12}
    raise ValueError(step)


def _write_validation_job(
    *,
    selection_dir: Path,
    policy: Path,
    step: int,
    task_id: int,
    init_state_ids: list[int],
) -> None:
    output = selection_dir / "jobs" / f"checkpoint_{step:06d}" / f"task_{task_id:02d}"
    output.mkdir(parents=True)
    success_indices = _success_indices(step)
    episodes = [
        {
            "episode_index": index,
            "seed": 1000 + index,
            "init_state_id": init_state_id,
            "num_steps": 100 + index if index in success_indices else 300,
            "success": index in success_indices,
            "max_reward": 1.0 if index in success_indices else 0.0,
            "termination_reason": "success" if index in success_indices else "time_limit",
        }
        for index, init_state_id in enumerate(init_state_ids)
    ]
    successes = len(success_indices)
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "task": {"task_id": task_id},
                "aggregate": {
                    "num_episodes": len(episodes),
                    "num_successes": successes,
                    "success_rate": successes / len(episodes),
                    "avg_max_reward": successes / len(episodes),
                },
                "per_episode": episodes,
            }
        ),
        encoding="utf-8",
    )
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "policy": {"path": str(policy)},
                "input": {
                    "policy_seed": 2000,
                    "seeds": [1000 + index for index in range(20)],
                    "policy_execution": {"effective_n_action_steps": 10},
                },
            }
        ),
        encoding="utf-8",
    )


def _analysis_arguments(tmp_path: Path) -> analysis.ValidationAnalysisArguments:
    run_dir = tmp_path / "run"
    selection_dir = run_dir / "checkpoint_selection"
    selection_dir.mkdir(parents=True)
    split = selection.load_selection_split(SPLIT_FILE)
    for step in analysis.COMPARISON_STEPS:
        policy = _complete_checkpoint(run_dir, step)
        for task_id in selection.EXPECTED_TASK_IDS:
            _write_validation_job(
                selection_dir=selection_dir,
                policy=policy,
                step=step,
                task_id=task_id,
                init_state_ids=split["tasks"][str(task_id)]["validation_init_state_ids"],
            )
    selected = {
        "status": "complete",
        "selection_set": "background_validation",
        "selected_checkpoint_step": 25000,
        "source_checkpoint": str(run_dir / "checkpoints" / "025000"),
        "split_sha256": selection._sha256_file(SPLIT_FILE),
    }
    ranking = [
        {
            "checkpoint_step": 25000,
            "macro_success_rate": 0.6,
            "mean_max_reward": 0.6,
            "mean_successful_episode_steps": 100.0,
        },
        {
            "checkpoint_step": 30000,
            "macro_success_rate": 0.6,
            "mean_max_reward": 0.6,
            "mean_successful_episode_steps": 101.0,
        },
        {
            "checkpoint_step": 20000,
            "macro_success_rate": 0.5,
            "mean_max_reward": 0.5,
            "mean_successful_episode_steps": 110.0,
        },
    ]
    (selection_dir / "selected_checkpoint.json").write_text(json.dumps(selected), encoding="utf-8")
    (selection_dir / "checkpoint_results.json").write_text(
        json.dumps({"status": "complete", "ranking": ranking}), encoding="utf-8"
    )
    return analysis.ValidationAnalysisArguments(
        run_dir=run_dir,
        selection_dir=selection_dir,
        split_file=SPLIT_FILE,
        env_seed=1000,
        policy_seed=2000,
        n_action_steps=10,
        output_json=selection_dir / "validation_paired_analysis.json",
        output_markdown=selection_dir / "validation_paired_analysis.md",
    )


def test_exact_mcnemar_pvalue() -> None:
    assert analysis.exact_mcnemar_pvalue(0, 0) == 1.0
    assert analysis.exact_mcnemar_pvalue(10, 0) == pytest.approx(0.001953125)
    assert analysis.exact_mcnemar_pvalue(7, 7) == 1.0


def test_validation_analysis_compares_20k_25k_and_30k_without_rollouts(
    tmp_path: Path,
) -> None:
    arguments = _analysis_arguments(tmp_path)

    result = analysis.analyze_validation(arguments)

    assert result["num_paired_cases"] == 140
    assert result["selected_checkpoint_step"] == 25000
    pairs = {(item["from_step"], item["to_step"]): item for item in result["pairwise"]}
    assert pairs[(20000, 25000)]["gains"] == 14
    assert pairs[(20000, 25000)]["regressions"] == 0
    assert pairs[(25000, 30000)]["gains"] == 7
    assert pairs[(25000, 30000)]["regressions"] == 7
    assert pairs[(25000, 30000)]["net_success_gain"] == 0
    assert pairs[(25000, 30000)]["exact_mcnemar_pvalue"] == 1.0
    assert result["ranking_audit"]["matches_registered_tie_breakers"] is True
    assert arguments.output_json.is_file()
    assert arguments.output_markdown.is_file()
