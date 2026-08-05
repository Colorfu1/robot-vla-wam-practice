from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import robot_practice.experiments.smolvla_libero.b0_checkpoint_selection as selection
import robot_practice.experiments.smolvla_libero.b0_test_evaluation as test_evaluation
import robot_practice.experiments.smolvla_libero.background_regression_analysis as regression
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


def _adapter_checkpoint(run_dir: Path, step: int) -> Path:
    """LoRA checkpoint shape: no model.safetensors, but a complete adapter pair."""

    policy = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    policy.mkdir(parents=True)
    for filename in selection.REQUIRED_CHECKPOINT_FILES:
        if filename == "model.safetensors":
            continue
        (policy / filename).write_text(filename, encoding="utf-8")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        (policy / filename).write_text(filename, encoding="utf-8")
    return policy


def _complete_assets(root: Path) -> Path:
    for directory in REQUIRED_ASSET_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "scenes" / "libero_tabletop_base_style.xml").write_text("<mujoco/>", encoding="utf-8")
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
    (selection_dir / "selected_checkpoint.json").write_text(json.dumps(payload), encoding="utf-8")
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


def test_direct_policy_mode_skips_selection_provenance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    policy = _complete_checkpoint(run_dir, 25000)
    assets = _complete_assets(tmp_path / "assets")
    arguments = test_evaluation.TestArguments(
        run_dir=run_dir,
        selection_dir=tmp_path / "missing_selection",
        policy_path=policy,
        model_label="c1_selected",
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
    split = selection.load_selection_split(SPLIT_FILE)

    selected = test_evaluation.load_selected_policy(arguments, split)
    assert selected.step == 0
    assert selected.policy_dir == policy

    payload = test_evaluation.evaluate_selected_checkpoint(arguments)
    assert payload["model_label"] == "c1_selected"
    assert payload["selected_checkpoint_step"] == 0
    assert payload["validation_pairwise"] == []
    assert payload["test_rollouts"] == 210


def test_direct_policy_mode_accepts_lora_adapter_checkpoint(tmp_path: Path) -> None:
    """C2/C3 checkpoints carry an adapter pair instead of model.safetensors."""

    run_dir = tmp_path / "run"
    policy = _adapter_checkpoint(run_dir, 20000)
    assert not (policy / "model.safetensors").exists()
    arguments = test_evaluation.TestArguments(
        run_dir=run_dir,
        selection_dir=tmp_path / "missing_selection",
        policy_path=policy,
        model_label="c2_selected",
        split_file=SPLIT_FILE,
        num_gpus=8,
        env_seed=1000,
        policy_seed=2000,
        n_action_steps=10,
        libero_assets_dir=_complete_assets(tmp_path / "assets"),
        output_dir=run_dir / "background_test",
        resume=False,
        overwrite=False,
        print_command=True,
    )
    split = selection.load_selection_split(SPLIT_FILE)

    selected = test_evaluation.load_selected_policy(arguments, split)
    assert selected.policy_dir == policy


def test_direct_policy_mode_rejects_incomplete_adapter_pair(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    policy = _adapter_checkpoint(run_dir, 20000)
    (policy / "adapter_model.safetensors").unlink()
    arguments = test_evaluation.TestArguments(
        run_dir=run_dir,
        selection_dir=tmp_path / "missing_selection",
        policy_path=policy,
        model_label="c2_selected",
        split_file=SPLIT_FILE,
        num_gpus=8,
        env_seed=1000,
        policy_seed=2000,
        n_action_steps=10,
        libero_assets_dir=_complete_assets(tmp_path / "assets"),
        output_dir=run_dir / "background_test",
        resume=False,
        overwrite=False,
        print_command=True,
    )
    split = selection.load_selection_split(SPLIT_FILE)

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        test_evaluation.load_selected_policy(arguments, split)


def test_diagnostic_policy_result_set_is_prefixed(tmp_path: Path) -> None:
    policy = _complete_checkpoint(tmp_path / "run", 25000)
    run_dir = policy.parent.parent
    (run_dir / "run_config.json").write_text(
        json.dumps({"training": {"mode": "diagnostic", "experiment_tag": "a_same_config"}}),
        encoding="utf-8",
    )

    assert test_evaluation._policy_run_is_diagnostic(policy) is True
    assert (
        test_evaluation.background_result_set_name(policy_dir=policy)
        == "diagnostic_background_test"
    )

    (run_dir / "run_config.json").write_text(
        json.dumps({"training": {"mode": "formal"}}),
        encoding="utf-8",
    )
    assert test_evaluation._policy_run_is_diagnostic(policy) is False
    assert test_evaluation.background_result_set_name(policy_dir=policy) == "background_test"


def test_cli_requires_policy_and_label_together(tmp_path: Path) -> None:
    parser = test_evaluation.build_argument_parser()
    policy = tmp_path / "pretrained_model"
    base = [
        f"--run-dir={tmp_path / 'run'}",
        f"--split-file={SPLIT_FILE}",
        f"--output-dir={tmp_path / 'out'}",
    ]

    with pytest.raises(ValueError, match="--policy-path requires --model-label"):
        test_evaluation._resolve_arguments(parser.parse_args([*base, f"--policy-path={policy}"]))
    with pytest.raises(ValueError, match="--model-label requires --policy-path"):
        test_evaluation._resolve_arguments(parser.parse_args([*base, "--model-label=c1_selected"]))
    namespace = parser.parse_args([*base, f"--policy-path={policy}", "--model-label=c1_selected"])
    arguments = test_evaluation._resolve_arguments(namespace)
    assert arguments.policy_path == policy.resolve()
    assert arguments.model_label == "c1_selected"


def test_legacy_mode_still_requires_selection(tmp_path: Path) -> None:
    arguments = dataclasses.replace(
        _arguments(tmp_path),
        selection_dir=tmp_path / "missing_selection",
    )
    with pytest.raises(FileNotFoundError, match="selection is incomplete"):
        test_evaluation.load_selected_policy(
            arguments,
            selection.load_selection_split(SPLIT_FILE),
        )


def test_episode_records_collect_210_paired_rows(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    split = selection.load_selection_split(SPLIT_FILE)
    selected = test_evaluation.load_selected_policy(arguments, split)
    jobs = test_evaluation.build_test_jobs(
        selected=selected,
        split=split,
        output_dir=arguments.output_dir,
    )
    task_metrics: dict[int, dict[str, object]] = {}
    for job in jobs:
        task_metrics[job.task_id] = {
            "aggregate": {},
            "per_episode": [
                {
                    "episode_index": index,
                    "seed": arguments.env_seed + index,
                    "init_state_id": state,
                    "num_steps": 100 + index,
                    "success": index % 2 == 0,
                    "termination_reason": "success" if index % 2 == 0 else "max_steps",
                }
                for index, state in enumerate(job.init_state_ids)
            ],
        }

    records = test_evaluation.collect_background_episode_records(jobs, task_metrics)

    assert len(records) == 210
    assert {
        "task_id",
        "init_state_id",
        "episode_index",
        "seed",
        "num_steps",
        "success",
        "termination_reason",
    } <= set(records[0])
    assert len({(record["task_id"], record["init_state_id"]) for record in records}) == 210


def _fake_background_job(
    output_dir: Path,
    task_id: int,
    *,
    policy_seed: int = 2000,
) -> None:
    job_dir = output_dir / "jobs" / f"task_{task_id:02d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    split = selection.load_selection_split(SPLIT_FILE)
    init_state_ids = split["tasks"][str(task_id)]["test_init_state_ids"]
    per_episode = [
        {
            "episode_index": index,
            "seed": 1000 + index,
            "init_state_id": state,
            "num_steps": 120 + index,
            "success": index % 3 != 0,
            "termination_reason": "success" if index % 3 != 0 else "max_steps",
        }
        for index, state in enumerate(init_state_ids)
    ]
    (job_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "aggregate": {"num_episodes": 30},
                "per_episode": per_episode,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "run_config.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "input": {
                    "policy_seed": policy_seed,
                    "seeds": list(range(1000, 1030)),
                    "init_state_ids": list(init_state_ids),
                    "policy_execution": {"effective_n_action_steps": 10},
                },
            }
        ),
        encoding="utf-8",
    )


def test_rebuild_episodes_from_original_jobs(tmp_path: Path) -> None:
    output_dir = tmp_path / "background_test"
    for task_id in (0, 2, 3, 5, 6, 7, 9):
        _fake_background_job(output_dir, task_id)
    split = selection.load_selection_split(SPLIT_FILE)

    destination = test_evaluation.rebuild_episodes_from_jobs(
        output_dir=output_dir,
        split=split,
    )
    rows = pq.read_table(destination).to_pylist()

    assert len(rows) == 210
    assert {
        "task_id",
        "init_state_id",
        "episode_index",
        "seed",
        "num_steps",
        "success",
        "termination_reason",
    } <= set(rows[0])
    assert len({(row["task_id"], row["init_state_id"]) for row in rows}) == 210

    _fake_background_job(output_dir, 0, policy_seed=1999)
    with pytest.raises(ValueError, match="policy_seed"):
        test_evaluation.rebuild_episodes_from_jobs(
            output_dir=output_dir,
            split=split,
        )


def test_regression_analysis_pairs_episodes(tmp_path: Path) -> None:
    baseline = []
    finetuned = []
    for task in (0, 2, 3, 5, 6, 7, 9):
        for state in range(2):
            baseline.append(
                {
                    "task_id": task,
                    "init_state_id": state,
                    "episode_index": task * 10 + state,
                    "seed": 1000 + state,
                    "num_steps": 120 if state == 0 else 300,
                    "success": state == 0,
                    "termination_reason": "success" if state == 0 else "max_steps",
                }
            )
            finetuned.append(
                {
                    "task_id": task,
                    "init_state_id": state,
                    "episode_index": task * 10 + state,
                    "seed": 1000 + state,
                    "num_steps": 300 if state == 0 else 130,
                    "success": state == 1,
                    "termination_reason": "max_steps" if state == 0 else "success",
                }
            )
    baseline_path = tmp_path / "baseline.parquet"
    finetuned_path = tmp_path / "finetuned.parquet"
    pq.write_table(pa.Table.from_pylist(baseline), baseline_path)
    pq.write_table(pa.Table.from_pylist(finetuned), finetuned_path)

    result = regression.analyze_regression(
        baseline_path=baseline_path,
        finetuned_path=finetuned_path,
        baseline_label="b0_selected",
        finetuned_label="c1_selected",
    )

    totals = result["totals"]
    assert totals["baseline_successes"] == 7
    assert totals["finetuned_successes"] == 7
    assert totals["gains"] == 7
    assert totals["regressions"] == 7
    assert totals["discordant"] == 14
    assert totals["exact_mcnemar_pvalue"] == 1.0
    assert len(result["regression_cases"]) == 7


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
