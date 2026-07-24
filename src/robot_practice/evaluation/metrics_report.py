"""Render compact Markdown summaries from evaluator ``metrics.json`` payloads."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


@dataclass(frozen=True)
class MetricRow:
    label: str
    key: str
    kind: str
    description: str
    nullable: bool = False


@dataclass(frozen=True)
class BackfillResult:
    generated: int
    skipped: int


OFFLINE_AGGREGATE_ROWS = (
    MetricRow("Episodes", "num_episodes", "integer", "离线评测的 demonstration 数量"),
    MetricRow("Frames", "num_frames", "integer", "参与预测的数据帧数量"),
    MetricRow("Model queries", "num_model_queries", "integer", "模型批量推理调用次数"),
    MetricRow("Batch size", "batch_size", "integer", "每次模型调用的样本数"),
    MetricRow(
        "First-action MAE",
        "first_action_mae",
        "float4",
        "当前动作预测与真值的平均绝对误差",
    ),
    MetricRow(
        "Delta-action MAE",
        "delta_action_mae",
        "float5",
        "预测动作变化与真值动作变化的误差",
        nullable=True,
    ),
    MetricRow(
        "Action TV",
        "teacher_forced_action_tv",
        "float5",
        "相邻预测动作的平均变化量",
        nullable=True,
    ),
    MetricRow(
        "Action Jerk",
        "teacher_forced_action_jerk",
        "float5",
        "预测动作二阶差分的抖动代理",
        nullable=True,
    ),
    MetricRow(
        "Inference latency p50",
        "inference_latency_p50_ms",
        "latency",
        "一半模型批调用不超过的耗时",
        nullable=True,
    ),
    MetricRow(
        "Inference latency p95",
        "inference_latency_p95_ms",
        "latency",
        "95% 模型批调用不超过的耗时",
        nullable=True,
    ),
)

LIBERO_AGGREGATE_ROWS = (
    MetricRow("Formal evaluation", "is_formal", "boolean", "是否使用完整任务步数上限"),
    MetricRow("Episodes", "num_episodes", "integer", "闭环 rollout 数量"),
    MetricRow("Successes", "num_successes", "integer", "LIBERO 判定成功的 rollout 数量"),
    MetricRow(
        "Success rate",
        "success_rate",
        "percent",
        "成功完成任务的 rollout 比例",
        nullable=True,
    ),
    MetricRow(
        "Average sum reward",
        "avg_sum_reward",
        "float4",
        "每个 rollout 累计 reward 的平均值",
        nullable=True,
    ),
    MetricRow(
        "Average max reward",
        "avg_max_reward",
        "float4",
        "每个 rollout 最大单步 reward 的平均值",
        nullable=True,
    ),
    MetricRow(
        "Action TV",
        "action_tv",
        "float5",
        "相邻执行动作的平均变化量",
        nullable=True,
    ),
    MetricRow(
        "Action Jerk",
        "action_jerk",
        "float5",
        "执行动作二阶差分的抖动代理",
        nullable=True,
    ),
    MetricRow(
        "Mean gripper toggles",
        "mean_gripper_toggles",
        "float2",
        "平均每次 rollout 的夹爪切换次数",
    ),
    MetricRow(
        "Policy latency p50",
        "policy_step_latency_p50_ms",
        "latency",
        "一半控制步骤不超过的策略耗时",
        nullable=True,
    ),
    MetricRow(
        "Policy latency p95",
        "policy_step_latency_p95_ms",
        "latency",
        "95% 控制步骤不超过的策略耗时",
        nullable=True,
    ),
    MetricRow(
        "Model latency p50",
        "model_query_latency_p50_ms",
        "latency",
        "一半模型调用不超过的推理耗时",
        nullable=True,
    ),
    MetricRow(
        "Model latency p95",
        "model_query_latency_p95_ms",
        "latency",
        "95% 模型调用不超过的推理耗时",
        nullable=True,
    ),
    MetricRow("Policy steps", "num_policy_steps", "integer", "发送给模拟器的控制动作数"),
    MetricRow("Model queries", "num_model_queries", "integer", "实际模型推理次数"),
)


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _require_sequence(value: Any, *, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be an array")
    return value


def _require_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _require_boolean(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_integer(
    value: Any,
    *,
    path: str,
    positive: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{path} must be {qualifier}")
    return value


def _require_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _field(mapping: Mapping[str, Any], key: str, *, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{path}.{key} is required")
    return mapping[key]


def _format_metric(value: Any, row: MetricRow, *, path: str) -> str:
    if value is None:
        if row.nullable:
            return "N/A"
        raise ValueError(f"{path} must not be null")
    if row.kind == "integer":
        return str(_require_integer(value, path=path))
    if row.kind == "boolean":
        return "是" if _require_boolean(value, path=path) else "否"

    number = _require_number(value, path=path)
    if row.kind == "float2":
        return f"{number:.2f}"
    if row.kind == "float4":
        return f"{number:.4f}"
    if row.kind == "float5":
        return f"{number:.5f}"
    if row.kind == "latency":
        return f"{number:.2f} ms"
    if row.kind == "percent":
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{path} must be between 0 and 1")
        return f"{number:.1%}"
    raise ValueError(f"Unsupported metric format: {row.kind}")


def _escape_markdown_cell(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _validate_common(metrics: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any], list]:
    schema_version = _field(metrics, "schema_version", path="metrics")
    if _require_integer(schema_version, path="metrics.schema_version") != 2:
        raise ValueError("metrics.schema_version must equal 2")
    mode = _require_string(_field(metrics, "mode", path="metrics"), path="metrics.mode")
    if mode not in {"offline", "libero"}:
        raise ValueError("metrics.mode must be 'offline' or 'libero'")
    status = _require_string(
        _field(metrics, "status", path="metrics"),
        path="metrics.status",
    )
    if status not in {"complete", "smoke"}:
        raise ValueError("metrics.status must be 'complete' or 'smoke'")
    aggregate = _require_mapping(
        _field(metrics, "aggregate", path="metrics"),
        path="metrics.aggregate",
    )
    episode_values = _require_sequence(
        _field(metrics, "per_episode", path="metrics"),
        path="metrics.per_episode",
    )
    if not episode_values:
        raise ValueError("metrics.per_episode must not be empty")
    episodes = [
        _require_mapping(value, path=f"metrics.per_episode[{index}]")
        for index, value in enumerate(episode_values)
    ]
    expected_count = _require_integer(
        _field(aggregate, "num_episodes", path="metrics.aggregate"),
        path="metrics.aggregate.num_episodes",
    )
    if expected_count != len(episodes):
        raise ValueError(
            "metrics.aggregate.num_episodes must equal the number of per_episode rows"
        )
    return mode, status, aggregate, episodes


def _ordered_episodes(episodes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for row_index, episode in enumerate(episodes):
        episode_index = _require_integer(
            _field(episode, "episode_index", path=f"metrics.per_episode[{row_index}]"),
            path=f"metrics.per_episode[{row_index}].episode_index",
        )
        indexed.append((episode_index, episode))
    indices = [index for index, _ in indexed]
    if len(indices) != len(set(indices)):
        raise ValueError("metrics.per_episode contains duplicate episode_index values")
    return [episode for _, episode in sorted(indexed, key=lambda item: item[0])]


def _context_lines(metrics: Mapping[str, Any], mode: str, status: str, count: int) -> list[str]:
    lines = [
        "## Run context",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Mode | {mode} |",
        f"| Status | {status} |",
    ]
    task_value = metrics.get("task")
    if task_value is not None:
        task = _require_mapping(task_value, path="metrics.task")
        suite = _escape_markdown_cell(
            _require_string(_field(task, "suite", path="metrics.task"), path="metrics.task.suite")
        )
        task_id = _require_integer(
            _field(task, "task_id", path="metrics.task"),
            path="metrics.task.task_id",
        )
        task_name = _escape_markdown_cell(
            _require_string(
                _field(task, "task_name", path="metrics.task"),
                path="metrics.task.task_name",
            )
        )
        lines.extend(
            (
                f"| Task suite | {suite} |",
                f"| Task ID | {task_id} |",
                f"| Task name | {task_name} |",
            )
        )
    lines.append(f"| Episodes | {count} |")
    return lines


def _aggregate_lines(
    aggregate: Mapping[str, Any],
    rows: Sequence[MetricRow],
) -> list[str]:
    lines = [
        "## Aggregate metrics",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
    ]
    for row in rows:
        path = f"metrics.aggregate.{row.key}"
        rendered = _format_metric(
            _field(aggregate, row.key, path="metrics.aggregate"),
            row,
            path=path,
        )
        lines.append(f"| {row.label} | {rendered} | {row.description} |")
    return lines


def _offline_lines(
    aggregate: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
) -> list[str]:
    dimensions = _require_sequence(
        _field(aggregate, "per_dimension_mae", path="metrics.aggregate"),
        path="metrics.aggregate.per_dimension_mae",
    )
    if len(dimensions) != len(ACTION_NAMES):
        raise ValueError("metrics.aggregate.per_dimension_mae must contain exactly 7 values")
    lines = [
        "## Per-dimension first-action MAE",
        "",
        "| Dimension | MAE |",
        "|---|---:|",
    ]
    for index, (name, value) in enumerate(zip(ACTION_NAMES, dimensions, strict=True)):
        number = _require_number(
            value,
            path=f"metrics.aggregate.per_dimension_mae[{index}]",
        )
        lines.append(f"| {name} | {number:.4f} |")

    lines.extend(
        (
            "",
            "## Per-episode metrics",
            "",
            "| Episode | Frames | First-action MAE | Delta-action MAE | "
            "Action TV | Action Jerk |",
            "|---:|---:|---:|---:|---:|---:|",
        )
    )
    first_action_row = MetricRow("", "", "float4", "")
    nullable_float5 = MetricRow("", "", "float5", "", nullable=True)
    for row_index, episode in enumerate(_ordered_episodes(episodes)):
        path = f"metrics.per_episode[{row_index}]"
        episode_index = _require_integer(
            _field(episode, "episode_index", path=path),
            path=f"{path}.episode_index",
        )
        frames = _require_integer(
            _field(episode, "num_frames", path=path),
            path=f"{path}.num_frames",
        )
        first_action = _format_metric(
            _field(episode, "first_action_mae", path=path),
            first_action_row,
            path=f"{path}.first_action_mae",
        )
        delta = _format_metric(
            _field(episode, "delta_action_mae", path=path),
            nullable_float5,
            path=f"{path}.delta_action_mae",
        )
        tv = _format_metric(
            _field(episode, "teacher_forced_action_tv", path=path),
            nullable_float5,
            path=f"{path}.teacher_forced_action_tv",
        )
        jerk = _format_metric(
            _field(episode, "teacher_forced_action_jerk", path=path),
            nullable_float5,
            path=f"{path}.teacher_forced_action_jerk",
        )
        lines.append(
            f"| {episode_index} | {frames} | {first_action} | {delta} | {tv} | {jerk} |"
        )
    return lines


def _libero_lines(episodes: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "## Per-episode metrics",
        "",
        "| Episode | Seed | Init state | Success | Steps | Sum reward | "
        "Action TV | Action Jerk | Policy p95 | Termination |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    reward_row = MetricRow("", "", "float4", "")
    nullable_float5 = MetricRow("", "", "float5", "", nullable=True)
    nullable_latency = MetricRow("", "", "latency", "", nullable=True)
    for row_index, episode in enumerate(_ordered_episodes(episodes)):
        path = f"metrics.per_episode[{row_index}]"
        episode_index = _require_integer(
            _field(episode, "episode_index", path=path),
            path=f"{path}.episode_index",
        )
        seed = _require_integer(
            _field(episode, "seed", path=path),
            path=f"{path}.seed",
        )
        init_state_id = _require_integer(
            _field(episode, "init_state_id", path=path),
            path=f"{path}.init_state_id",
        )
        success = _require_boolean(
            _field(episode, "success", path=path),
            path=f"{path}.success",
        )
        steps = _require_integer(
            _field(episode, "num_steps", path=path),
            path=f"{path}.num_steps",
            positive=True,
        )
        _require_integer(
            _field(episode, "num_model_queries", path=path),
            path=f"{path}.num_model_queries",
        )
        _require_integer(
            _field(episode, "gripper_toggles", path=path),
            path=f"{path}.gripper_toggles",
        )
        reward = _format_metric(
            _field(episode, "sum_reward", path=path),
            reward_row,
            path=f"{path}.sum_reward",
        )
        _require_number(
            _field(episode, "max_reward", path=path),
            path=f"{path}.max_reward",
        )
        tv = _format_metric(
            _field(episode, "action_tv", path=path),
            nullable_float5,
            path=f"{path}.action_tv",
        )
        jerk = _format_metric(
            _field(episode, "action_jerk", path=path),
            nullable_float5,
            path=f"{path}.action_jerk",
        )
        for latency_key in (
            "policy_step_latency_p50_ms",
            "model_query_latency_p50_ms",
            "model_query_latency_p95_ms",
        ):
            _format_metric(
                _field(episode, latency_key, path=path),
                nullable_latency,
                path=f"{path}.{latency_key}",
            )
        policy_p95 = _format_metric(
            _field(episode, "policy_step_latency_p95_ms", path=path),
            nullable_latency,
            path=f"{path}.policy_step_latency_p95_ms",
        )
        termination = _escape_markdown_cell(
            _require_string(
                _field(episode, "termination_reason", path=path),
                path=f"{path}.termination_reason",
            )
        )
        lines.append(
            f"| {episode_index} | {seed} | {init_state_id} | "
            f"{'是' if success else '否'} | {steps} | {reward} | {tv} | {jerk} | "
            f"{policy_p95} | {termination} |"
        )
    return lines


def render_metrics_markdown(metrics: Mapping[str, Any]) -> str:
    """Validate a schema-v2 metrics payload and render deterministic Markdown."""

    metrics_mapping = _require_mapping(metrics, path="metrics")
    mode, status, aggregate, episodes = _validate_common(metrics_mapping)
    title = (
        "# Offline evaluation metrics"
        if mode == "offline"
        else "# LIBERO closed-loop evaluation metrics"
    )
    aggregate_rows = (
        OFFLINE_AGGREGATE_ROWS if mode == "offline" else LIBERO_AGGREGATE_ROWS
    )
    lines = [title, ""]
    lines.extend(_context_lines(metrics_mapping, mode, status, len(episodes)))
    lines.extend(("", *_aggregate_lines(aggregate, aggregate_rows)))
    if mode == "offline":
        lines.extend(("", *_offline_lines(aggregate, episodes)))
    else:
        lines.extend(("", *_libero_lines(episodes)))
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_metrics_report(output_dir: Path, metrics: Mapping[str, Any]) -> Path:
    """Validate and atomically write ``metrics.md`` in an evaluator output."""

    report_path = output_dir / "metrics.md"
    _write_text_atomic(report_path, render_metrics_markdown(metrics))
    return report_path


def backfill_metrics_reports(root: Path) -> BackfillResult:
    """Regenerate every sibling ``metrics.md`` below a file or directory."""

    root = root.expanduser()
    if root.is_file():
        candidates = [root] if root.name == "metrics.json" else []
    elif root.is_dir():
        candidates = sorted(root.rglob("metrics.json"))
    else:
        raise FileNotFoundError(f"Metrics root does not exist: {root}")

    generated = 0
    skipped = 0
    for metrics_path in candidates:
        try:
            with metrics_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            write_metrics_report(metrics_path.parent, payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped += 1
            LOGGER.warning("[Metrics] skipped %s: %s", metrics_path, exc)
        else:
            generated += 1
            LOGGER.info("[Metrics] wrote %s", metrics_path.with_name("metrics.md"))
    return BackfillResult(generated=generated, skipped=skipped)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("outputs"),
        help="metrics.json file or directory to scan recursively (default: outputs)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_argument_parser().parse_args(argv)
    try:
        result = backfill_metrics_reports(args.root)
    except (OSError, ValueError) as exc:
        LOGGER.error("[Metrics] %s", exc)
        return 1
    LOGGER.info(
        "[Metrics] complete: generated=%d skipped=%d",
        result.generated,
        result.skipped,
    )
    return 1 if result.skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
