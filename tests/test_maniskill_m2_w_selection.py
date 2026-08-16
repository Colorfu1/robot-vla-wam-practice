from __future__ import annotations

import pytest

from robot_practice.experiments.maniskill.m2_w_action_flow import (
    evaluate_development_gates,
    paired_success_bootstrap_ci,
    selection_sort_key,
)


def _row(index: int, success: bool, *, fingerprint: str | None = None) -> dict[str, object]:
    return {
        "index": index,
        "start_state_fingerprint": fingerprint or f"fp-{index}",
        "success_at_end": success,
        "success_once": success,
        "is_cubeA_grasped": True,
        "is_cubeA_on_cubeB": success,
        "is_cubeA_static": success,
    }


def test_selection_order_is_mechanical() -> None:
    better_success = {
        "summary": {
            "success_at_end": 0.5,
            "success_once": 0.7,
            "ever_on_cube": 0.5,
            "amortized_policy_latency_p95_ms": 10,
        },
        "normalized_action_mse": 2,
        "checkpoint_step": 20000,
        "n_action_steps": 8,
    }
    worse_success = {
        "summary": {
            "success_at_end": 0.4,
            "success_once": 0.9,
            "ever_on_cube": 0.9,
            "amortized_policy_latency_p95_ms": 1,
        },
        "normalized_action_mse": 0.1,
        "checkpoint_step": 10000,
        "n_action_steps": 4,
    }
    assert selection_sort_key(better_success) < selection_sort_key(worse_success)


def test_paired_bootstrap_requires_matching_fingerprints() -> None:
    candidate = [_row(0, True)]
    reference = [_row(0, False, fingerprint="different")]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        paired_success_bootstrap_ci(candidate, reference, samples=100)


def test_development_gates_are_fixed_before_results() -> None:
    candidate_rows = [_row(index, index < 50) for index in range(100)]
    reference_rows = [_row(index, False) for index in range(100)]
    selected = {
        "summary": {
            "split": "development",
            "test_access_count": 0,
            "success_at_end": 0.50,
            "success_once": 0.60,
            "ever_grasped": 0.80,
            "ever_on_cube": 0.50,
            "finite_rate": 1.0,
            "action_clip_fraction": 0.001,
            "amortized_policy_latency_p95_ms": 10.0,
        },
        "normalized_action_mse": 0.8,
        "train_action_mean_mse": 1.0,
        "rows": candidate_rows,
    }
    strongest_reference = {
        "summary": {"success_at_end": 0.0},
        "rows": reference_rows,
    }
    config = {
        "evaluation": {
            "paired_bootstrap_seed": 1000,
            "paired_bootstrap_samples": 100,
        }
    }
    result = evaluate_development_gates(
        selected,
        strongest_reference=strongest_reference,
        reference_rows=reference_rows,
        config=config,
    )
    assert result["gate_pass"] is True
    assert result["gates"]["flow_learning"]["passed"] is True
    assert result["gates"]["above_reference"]["bootstrap_95_ci"][0] > 0
