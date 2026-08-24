from __future__ import annotations

import json
from pathlib import Path

from robot_practice.experiments.literature.w0_fastwam_analysis import (
    aggregate_eval_directory,
    compare_causal,
    deployment_gate,
    gate_w0,
    paired_bootstrap,
)


def _write_task(root: Path, suite: str, task_id: int, success: list[int], total: int = 2) -> None:
    target = root / suite
    target.mkdir(parents=True, exist_ok=True)
    failures = [idx for idx in range(total) if idx not in success]
    (target / f"gpu0_task{task_id}_results.json").write_text(
        json.dumps(
            {
                "task_suite_name": suite,
                "task_id": task_id,
                "total_episodes": total,
                "successes": len(success),
                "success_episodes": success,
                "failure_episodes": failures,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_uses_task_results_without_counting_summary(tmp_path: Path) -> None:
    _write_task(tmp_path, "libero_spatial", 0, [0])
    _write_task(tmp_path, "libero_object", 0, [0, 1])
    (tmp_path / "summary.json").write_text(json.dumps({"successes": 999}), encoding="utf-8")
    result = aggregate_eval_directory(tmp_path)
    assert result["num_tasks"] == 2
    assert result["num_episodes"] == 4
    assert result["successes"] == 3
    assert result["status"] == "incomplete"


def test_aggregate_marks_requested_single_task_probe_complete(tmp_path: Path) -> None:
    _write_task(tmp_path, "libero_spatial", 6, [0], total=1)
    result = aggregate_eval_directory(
        tmp_path,
        expected_task_keys=[("libero_spatial", 6)],
        expected_trials_per_task=1,
    )
    assert result["status"] == "complete"
    assert result["num_tasks"] == 1
    assert result["missing_task_count"] == 0


def test_aggregate_accepts_official_fastwam_task_suite_field(tmp_path: Path) -> None:
    target = tmp_path / "libero_spatial"
    target.mkdir(parents=True)
    (target / "gpu0_task0_results.json").write_text(
        json.dumps(
            {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "total_episodes": 1,
                "successes": 1,
                "success_episodes": [0],
                "failure_episodes": [],
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_eval_directory(
        tmp_path,
        expected_task_keys=[("libero_spatial", 0)],
        expected_trials_per_task=1,
    )

    assert result["status"] == "complete"
    assert result["tasks"][0]["suite"] == "libero_spatial"


def test_aggregate_merges_trial_shards_for_one_task(tmp_path: Path) -> None:
    target = tmp_path / "libero_spatial"
    target.mkdir(parents=True)
    for trial_start, success in ((0, True), (1, False)):
        (target / f"worker_trial{trial_start}.json").write_text(
            json.dumps(
                {
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "trial_start": trial_start,
                    "trial_count": 1,
                    "episode_indices_are_global": True,
                    "total_episodes": 1,
                    "successes": int(success),
                    "success_episodes": [trial_start] if success else [],
                    "failure_episodes": [] if success else [trial_start],
                }
            ),
            encoding="utf-8",
        )

    result = aggregate_eval_directory(
        tmp_path,
        expected_task_keys=[("libero_spatial", 0)],
        expected_trials_per_task=2,
    )

    assert result["status"] == "complete"
    assert result["num_tasks"] == 1
    assert result["num_episodes"] == 2
    assert result["successes"] == 1
    assert result["tasks"][0]["success_indices"] == [0]
    assert result["tasks"][0]["failure_indices"] == [1]


def test_paired_bootstrap_is_deterministic() -> None:
    pairs = [
        {"with_video": True, "no_video": False},
        {"with_video": True, "no_video": True},
        {"with_video": False, "no_video": False},
        {"with_video": True, "no_video": False},
    ]
    first = paired_bootstrap(pairs, repeats=100, seed=7)
    second = paired_bootstrap(pairs, repeats=100, seed=7)
    assert first == second
    assert first["observed_gap"] == 0.5


def test_causal_comparison_reports_suite_differences() -> None:
    left = {
        "status": "complete",
        "overall_success_rate": 0.75,
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 0,
                "total_episodes": 2,
                "success_indices": [0, 1],
            }
        ],
    }
    right = {
        "status": "complete",
        "overall_success_rate": 0.25,
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 0,
                "total_episodes": 2,
                "success_indices": [0],
            }
        ],
    }
    result = compare_causal(
        left,
        right,
        repeats=100,
        seed=7,
        contract_fields={"equal_init_data_noise_update_batch": True},
    )
    assert result["bootstrap"]["observed_gap"] == 0.5
    assert result["per_suite"]["libero_spatial"]["difference"] == 0.5


def test_gate_rejects_unmeasured_deployment() -> None:
    deployment = {"status": "pending_runtime_trace"}
    passed, reasons = deployment_gate(deployment)
    assert passed is False
    assert reasons


def test_gate_can_pass_synthetic_protocol_payload() -> None:
    released = {
        "status": "complete",
        "num_tasks": 40,
        "num_episodes": 2000,
        "overall_success_rate": 0.96,
        "per_suite": {
            "libero_spatial": {"success_rate": 0.96},
            "libero_object": {"success_rate": 0.96},
            "libero_goal": {"success_rate": 0.96},
            "libero_10": {"success_rate": 0.96},
        },
    }
    causal = {
        "bootstrap": {"observed_gap": 0.04, "ci95": [0.01, 0.07]},
        "per_suite": {
            suite: {"difference": 0.01} for suite in ("a", "b", "c", "d")
        },
        "contract": {"equal_init_data_noise_update_batch": True},
        "invalid_rollouts": 0,
    }
    deployment = {
        "status": "pass",
        "output_shape": ["B", 32, 7],
        "future_noisy_video_token_allocations": 0,
        "future_video_denoise_steps": 0,
        "future_vae_decode_calls": 0,
        "n_action_steps": 10,
        "gripper_contract": "toggle_last_dimension",
    }
    result = gate_w0(
        released,
        causal,
        deployment,
        lerobot={"api_pass": True},
    )
    assert result["passed"] is True
