from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_practice.evaluation.metrics_report import (
    backfill_metrics_reports,
    main,
    render_metrics_markdown,
    write_metrics_report,
)


def _offline_metrics() -> dict:
    return {
        "schema_version": 2,
        "mode": "offline",
        "status": "complete",
        "aggregate": {
            "num_episodes": 1,
            "num_frames": 2,
            "num_model_queries": 2,
            "batch_size": 1,
            "first_action_mae": 0.123456,
            "per_dimension_mae": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            "chunk_mae_by_horizon": [0.123456],
            "delta_action_mae": None,
            "teacher_forced_action_tv": 0.123456,
            "teacher_forced_action_jerk": None,
            "inference_latency_p50_ms": 1.234,
            "inference_latency_p95_ms": 2.345,
        },
        "per_episode": [
            {
                "episode_index": 823,
                "num_frames": 2,
                "first_action_mae": 0.123456,
                "delta_action_mae": None,
                "teacher_forced_action_tv": 0.123456,
                "teacher_forced_action_jerk": None,
            }
        ],
        "task": {
            "suite": "libero_object",
            "task_id": 8,
            "task_name": "pick | place\nname",
        },
    }


def _libero_metrics() -> dict:
    return {
        "schema_version": 2,
        "mode": "libero",
        "status": "complete",
        "aggregate": {
            "is_formal": True,
            "num_episodes": 1,
            "num_successes": 1,
            "success_rate": 1.0,
            "avg_sum_reward": 1.0,
            "avg_max_reward": 1.0,
            "action_tv": 0.0123456,
            "action_jerk": None,
            "mean_gripper_toggles": 2.0,
            "policy_step_latency_p50_ms": 1.234,
            "policy_step_latency_p95_ms": 2.345,
            "model_query_latency_p50_ms": 3.456,
            "model_query_latency_p95_ms": 4.567,
            "num_policy_steps": 3,
            "num_model_queries": 1,
        },
        "per_episode": [
            {
                "episode_index": 0,
                "seed": 1000,
                "init_state_id": 0,
                "num_steps": 3,
                "success": True,
                "sum_reward": 1.0,
                "max_reward": 1.0,
                "termination_reason": "success|done\nnow",
                "action_tv": 0.0123456,
                "action_jerk": None,
                "gripper_toggles": 2,
                "policy_step_latency_p50_ms": 1.234,
                "policy_step_latency_p95_ms": 2.345,
                "model_query_latency_p50_ms": 3.456,
                "model_query_latency_p95_ms": 4.567,
                "num_model_queries": 1,
            }
        ],
        "task": {
            "suite": "libero_object",
            "task_id": 8,
            "task_name": "pick_up_the_object",
        },
    }


def test_offline_metrics_markdown_matches_golden_output() -> None:
    assert render_metrics_markdown(_offline_metrics()) == """\
# Offline evaluation metrics

## Run context

| Field | Value |
|---|---|
| Mode | offline |
| Status | complete |
| Task suite | libero_object |
| Task ID | 8 |
| Task name | pick \\| place name |
| Episodes | 1 |

## Aggregate metrics

| Metric | Value | Meaning |
|---|---:|---|
| Episodes | 1 | 离线评测的 demonstration 数量 |
| Frames | 2 | 参与预测的数据帧数量 |
| Model queries | 2 | 模型批量推理调用次数 |
| Batch size | 1 | 每次模型调用的样本数 |
| First-action MAE | 0.1235 | 当前动作预测与真值的平均绝对误差 |
| Delta-action MAE | N/A | 预测动作变化与真值动作变化的误差 |
| Action TV | 0.12346 | 相邻预测动作的平均变化量 |
| Action Jerk | N/A | 预测动作二阶差分的抖动代理 |
| Inference latency p50 | 1.23 ms | 一半模型批调用不超过的耗时 |
| Inference latency p95 | 2.35 ms | 95% 模型批调用不超过的耗时 |

## Per-dimension first-action MAE

| Dimension | MAE |
|---|---:|
| x | 0.0100 |
| y | 0.0200 |
| z | 0.0300 |
| roll | 0.0400 |
| pitch | 0.0500 |
| yaw | 0.0600 |
| gripper | 0.0700 |

## Per-episode metrics

| Episode | Frames | First-action MAE | Delta-action MAE | Action TV | Action Jerk |
|---:|---:|---:|---:|---:|---:|
| 823 | 2 | 0.1235 | N/A | 0.12346 | N/A |
"""


def test_libero_metrics_markdown_matches_golden_output() -> None:
    assert render_metrics_markdown(_libero_metrics()) == """\
# LIBERO closed-loop evaluation metrics

## Run context

| Field | Value |
|---|---|
| Mode | libero |
| Status | complete |
| Task suite | libero_object |
| Task ID | 8 |
| Task name | pick_up_the_object |
| Episodes | 1 |

## Aggregate metrics

| Metric | Value | Meaning |
|---|---:|---|
| Formal evaluation | 是 | 是否使用完整任务步数上限 |
| Episodes | 1 | 闭环 rollout 数量 |
| Successes | 1 | LIBERO 判定成功的 rollout 数量 |
| Success rate | 100.0% | 成功完成任务的 rollout 比例 |
| Average sum reward | 1.0000 | 每个 rollout 累计 reward 的平均值 |
| Average max reward | 1.0000 | 每个 rollout 最大单步 reward 的平均值 |
| Action TV | 0.01235 | 相邻执行动作的平均变化量 |
| Action Jerk | N/A | 执行动作二阶差分的抖动代理 |
| Mean gripper toggles | 2.00 | 平均每次 rollout 的夹爪切换次数 |
| Policy latency p50 | 1.23 ms | 一半控制步骤不超过的策略耗时 |
| Policy latency p95 | 2.35 ms | 95% 控制步骤不超过的策略耗时 |
| Model latency p50 | 3.46 ms | 一半模型调用不超过的推理耗时 |
| Model latency p95 | 4.57 ms | 95% 模型调用不超过的推理耗时 |
| Policy steps | 3 | 发送给模拟器的控制动作数 |
| Model queries | 1 | 实际模型推理次数 |

## Per-episode metrics

| Episode | Seed | Init state | Success | Steps | Sum reward | Action TV | Action Jerk | Policy p95 | Termination |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---|
| 0 | 1000 | 0 | 是 | 3 | 1.0000 | 0.01235 | N/A | 2.35 ms | success\\|done now |
"""


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=1), "schema_version must equal 2"),
        (lambda payload: payload.update(mode="unknown"), "mode must be"),
        (
            lambda payload: payload["aggregate"].update(first_action_mae="0.1"),
            "first_action_mae must be a number",
        ),
        (
            lambda payload: payload["aggregate"].update(first_action_mae=float("nan")),
            "first_action_mae must be finite",
        ),
        (
            lambda payload: payload["aggregate"].update(per_dimension_mae=[0.1]),
            "must contain exactly 7",
        ),
        (
            lambda payload: payload["per_episode"].append(payload["per_episode"][0]),
            "num_episodes must equal",
        ),
        (lambda payload: payload.update(per_episode=[]), "must not be empty"),
    ],
)
def test_metrics_markdown_rejects_invalid_contract(mutation, message: str) -> None:
    payload = _offline_metrics()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        render_metrics_markdown(payload)


def test_metrics_markdown_sorts_episode_rows_and_rejects_duplicates() -> None:
    payload = _offline_metrics()
    second = dict(payload["per_episode"][0], episode_index=7)
    payload["per_episode"] = [payload["per_episode"][0], second]
    payload["aggregate"]["num_episodes"] = 2

    markdown = render_metrics_markdown(payload)
    assert markdown.index("| 7 | 2 |") < markdown.index("| 823 | 2 |")

    payload["per_episode"][0]["episode_index"] = 7
    with pytest.raises(ValueError, match="duplicate episode_index"):
        render_metrics_markdown(payload)


def test_write_metrics_report_is_repeatable_and_overwrites(tmp_path: Path) -> None:
    report_path = tmp_path / "metrics.md"
    report_path.write_text("stale", encoding="utf-8")

    assert write_metrics_report(tmp_path, _offline_metrics()) == report_path
    first = report_path.read_bytes()
    write_metrics_report(tmp_path, _offline_metrics())

    assert report_path.read_bytes() == first
    assert not list(tmp_path.glob(".metrics.md.*.tmp"))


def test_atomic_write_failure_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_replace(self: Path, target: Path) -> Path:
        raise PermissionError(f"cannot replace {target}")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="cannot replace"):
        write_metrics_report(tmp_path, _offline_metrics())

    assert not (tmp_path / "metrics.md").exists()
    assert not list(tmp_path.glob(".metrics.md.*.tmp"))


def test_backfill_continues_after_invalid_json_and_reports_cli_failure(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    (good / "metrics.json").write_text(
        json.dumps(_offline_metrics(), ensure_ascii=False),
        encoding="utf-8",
    )
    (good / "metrics.md").write_text("stale", encoding="utf-8")
    (bad / "metrics.json").write_text("{broken", encoding="utf-8")

    result = backfill_metrics_reports(tmp_path)

    assert result.generated == 1
    assert result.skipped == 1
    assert (good / "metrics.md").read_text(encoding="utf-8") == render_metrics_markdown(
        _offline_metrics()
    )
    assert not (bad / "metrics.md").exists()
    assert main([str(tmp_path)]) == 1
