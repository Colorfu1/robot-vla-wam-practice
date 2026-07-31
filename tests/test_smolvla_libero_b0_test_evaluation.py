from __future__ import annotations

import json
from pathlib import Path

import pytest

import robot_practice.experiments.smolvla_libero.b0_checkpoint_selection as selection
import robot_practice.experiments.smolvla_libero.b0_test_evaluation as test_evaluation
from robot_practice.evaluation.libero_assets import REQUIRED_ASSET_DIRECTORIES

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


def _complete_assets(root: Path) -> Path:
    for directory in REQUIRED_ASSET_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "scenes" / "libero_tabletop_base_style.xml").write_text(
        "<mujoco/>", encoding="utf-8"
    )
    return root


def _complete_selection(run_dir: Path, step: int) -> Path:
    selection_dir = run_dir / "checkpoint_selection"
    selection_dir.mkdir(parents=True)
    checkpoint_dir = run_dir / "checkpoints" / f"{step:06d}"
    payload = {
        "status": "complete",
        "selection_set": "background_validation",
        "selected_checkpoint_step": step,
        "source_checkpoint": str(checkpoint_dir),
        "policy_seed": 2000,
        "split_sha256": selection._sha256_file(SPLIT_FILE),
    }
    (selection_dir / "selected_checkpoint.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (selection_dir / "checkpoint_results.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "ranking": [{"checkpoint_step": step}],
            }
        ),
        encoding="utf-8",
    )
    return selection_dir


def _arguments(tmp_path: Path) -> test_evaluation.TestArguments:
    run_dir = tmp_path / "run"
    _complete_checkpoint(run_dir, 25000)
    selection_dir = _complete_selection(run_dir, 25000)
    assets = _complete_assets(tmp_path / "assets")
    return test_evaluation.TestArguments(
        run_dir=run_dir,
        selection_dir=selection_dir,
        split_file=SPLIT_FILE,
        num_gpus=8,
        env_seed=1000,
        policy_seed=2000,
        n_action_steps=10,
        libero_assets_dir=assets,
        output_dir=run_dir / "background_test",
        resume=False,
        overwrite=False,
        print_command=True,
    )


def test_print_command_freezes_selected_checkpoint_and_210_test_rollouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    monkeypatch.setattr(
        test_evaluation,
        "analyze_validation",
        lambda *args, **kwargs: {
            "pairwise": [
                {
                    "from_step": 20000,
                    "to_step": 25000,
                    "gains": 10,
                    "regressions": 0,
                    "net_success_gain": 10,
                    "exact_mcnemar_pvalue": 0.001953125,
                }
            ]
        },
    )

    payload = test_evaluation.evaluate_selected_checkpoint(arguments)

    assert payload["selected_checkpoint_step"] == 25000
    assert payload["test_rollouts"] == 210
    assert payload["tasks"] == [0, 2, 3, 5, 6, 7, 9]
    assert "--n-episodes=30" in payload["first_command"]
    assert "--policy-seed=2000" in payload["first_command"]
    assert payload["validation_pairwise"][0]["net_success_gain"] == 10
    assert not arguments.output_dir.exists()


def test_test_jobs_use_only_the_30_held_out_states(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    split = selection.load_selection_split(SPLIT_FILE)
    selected = test_evaluation.load_selected_policy(arguments, split)

    jobs = test_evaluation.build_test_jobs(
        selected=selected,
        split=split,
        output_dir=arguments.output_dir,
    )

    assert len(jobs) == 7
    assert sum(job.rollout_count for job in jobs) == 210
    for job in jobs:
        expected = tuple(split["tasks"][str(job.task_id)]["test_init_state_ids"])
        validation = set(split["tasks"][str(job.task_id)]["validation_init_state_ids"])
        assert job.init_state_ids == expected
        assert set(job.init_state_ids).isdisjoint(validation)


def test_selection_must_match_the_frozen_split(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    selected_path = arguments.selection_dir / "selected_checkpoint.json"
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    payload["split_sha256"] = "wrong"
    selected_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="split hash"):
        test_evaluation.load_selected_policy(
            arguments,
            selection.load_selection_split(SPLIT_FILE),
        )


def test_test_summary_keeps_per_task_and_failure_counts() -> None:
    split = selection.load_selection_split(SPLIT_FILE)
    task_metrics = {}
    for task_id in selection.EXPECTED_TASK_IDS:
        episodes = [
            {
                "success": index < 24,
                "max_reward": 1.0 if index < 24 else 0.0,
                "num_steps": 100 + index,
                "termination_reason": "success" if index < 24 else "time_limit",
            }
            for index in range(30)
        ]
        task_metrics[task_id] = {
            "aggregate": {
                "num_episodes": 30,
                "num_successes": 24,
                "success_rate": 0.8,
                "avg_max_reward": 0.8,
                "action_tv": 1.0,
                "action_jerk": 2.0,
                "mean_gripper_toggles": 3.0,
                "policy_step_latency_p50_ms": 4.0,
                "policy_step_latency_p95_ms": 5.0,
                "model_query_latency_p50_ms": 6.0,
                "model_query_latency_p95_ms": 7.0,
            },
            "per_episode": episodes,
        }

    result = test_evaluation.summarize_test(
        selected_step=25000,
        split=split,
        task_metrics=task_metrics,
    )

    assert result["aggregate"]["num_episodes"] == 210
    assert result["aggregate"]["num_successes"] == 168
    assert result["aggregate"]["macro_success_rate"] == pytest.approx(0.8)
    assert result["aggregate"]["termination_reasons"] == {
        "success": 168,
        "time_limit": 42,
    }
    assert result["per_task"]["0"]["canonical"] == "open the middle drawer of the cabinet"
