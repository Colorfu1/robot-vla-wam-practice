"""Merge corrected Prompt Swap rollouts into the original target-protocol results.

This is a publication step, not an evaluator. It never creates a LIBERO
environment and never launches a subprocess. It validates the three corrected
30-rollout outputs, replaces only the Prompt Swap artifacts in each original
210-rollout result, and optionally updates the human-facing reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from robot_practice.experiments.smolvla_libero.target_protocol_evaluation import (
    _write_metrics_markdown,
)

MODEL_LABELS = (
    "z0_b0_zero_shot",
    "c1_default_non_peft",
    "c2_canonical_lora",
)
GOAL_LABELS = ("stove", "cabinet", "plate")
FAILED_GOAL = "failed"
PROMPT_FIELDS = ("first_goal_reached", "first_goal_step", "terminal_goal")
PROMPT_PROTOCOL = "prompt_swap"
FULL_PROTOCOLS = ("canonical", "held_out_paraphrase", "prompt_swap")
PROMPT_ROLLOUTS = 30
FULL_ROLLOUTS = 210
PROMPT_STATES = (1, 2, 3, 4, 5, 7, 9, 10, 11, 12)

MERGE_NOTICE_BEGIN = "<!-- PROMPT_SWAP_MERGE_NOTICE:BEGIN -->"
MERGE_NOTICE_END = "<!-- PROMPT_SWAP_MERGE_NOTICE:END -->"
MERGE_LOG_BEGIN = "<!-- PROMPT_SWAP_MERGE_LOG:BEGIN -->"
MERGE_LOG_END = "<!-- PROMPT_SWAP_MERGE_LOG:END -->"


class MergeError(RuntimeError):
    """Raised when a source result cannot be safely merged."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MergeError(f"Missing JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MergeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"Expected a JSON object: {path}")
    return value


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise MergeError(f"Missing required file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(text)
    temporary_path.replace(path)


def _write_table_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
    pq.write_table(table, temporary_path)
    temporary_path.replace(path)


def _normalise_goal(value: Any) -> str:
    if value is None:
        return FAILED_GOAL
    value = str(value)
    return value if value in (*GOAL_LABELS, FAILED_GOAL) else FAILED_GOAL


def _confusion(records: Iterable[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    matrix = {
        expected: {predicted: 0 for predicted in (*GOAL_LABELS, FAILED_GOAL)}
        for expected in GOAL_LABELS
    }
    for record in records:
        expected = str(record.get("expected_goal"))
        if expected not in matrix:
            raise MergeError(f"Unexpected Prompt Swap expected goal: {expected!r}")
        matrix[expected][_normalise_goal(record.get(field))] += 1
    return matrix


def _summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != PROMPT_ROLLOUTS:
        raise MergeError(f"Prompt Swap episodes must contain 30 rows, got {len(records)}")
    counts = Counter(str(record.get("expected_goal")) for record in records)
    if counts != Counter({goal: 10 for goal in GOAL_LABELS}):
        raise MergeError(f"Prompt Swap expected-goal counts are not 10/goal: {counts}")
    for record in records:
        missing = [field for field in PROMPT_FIELDS if field not in record]
        if missing:
            raise MergeError("Prompt Swap episodes.parquet is missing fields " + ", ".join(missing))
        if record.get("num_steps") != 300:
            raise MergeError(
                "Prompt Swap must run the full 300-step horizon; "
                f"got {record.get('num_steps')} for {record.get('expected_goal')}/"
                f"{record.get('init_state_id')}"
            )
        first_step = record.get("first_goal_step")
        if first_step is not None and not 1 <= int(first_step) <= 300:
            raise MergeError(f"Invalid first_goal_step: {first_step!r}")
    first_matrix = _confusion(records, "first_goal_reached")
    terminal_matrix = _confusion(records, "terminal_goal")
    first_compliant = sum(first_matrix[goal][goal] for goal in GOAL_LABELS)
    terminal_compliant = sum(terminal_matrix[goal][goal] for goal in GOAL_LABELS)
    first_values = [_normalise_goal(record.get("first_goal_reached")) for record in records]
    terminal_values = [_normalise_goal(record.get("terminal_goal")) for record in records]
    return {
        "num_episodes": len(records),
        "num_compliant": first_compliant,
        "prompt_compliance": first_compliant / len(records),
        "first_goal_num_compliant": first_compliant,
        "first_goal_prompt_compliance": first_compliant / len(records),
        "terminal_goal_num_compliant": terminal_compliant,
        "terminal_goal_prompt_compliance": terminal_compliant / len(records),
        "num_full_horizon_episodes": sum(record.get("num_steps") == 300 for record in records),
        "num_first_terminal_disagreements": sum(
            first != terminal for first, terminal in zip(first_values, terminal_values, strict=True)
        ),
        "confusion_matrix": first_matrix,
        "terminal_goal_confusion_matrix": terminal_matrix,
    }


def _assert_same(mapping: dict[str, Any], other: dict[str, Any], fields: Iterable[str]) -> None:
    differences = {
        field: (mapping.get(field), other.get(field))
        for field in fields
        if mapping.get(field) != other.get(field)
    }
    if differences:
        raise MergeError(f"Frozen protocol fields differ: {differences}")


def _job_map(manifest: dict[str, Any], *, protocol: str) -> dict[str, dict[str, Any]]:
    jobs = manifest.get("contract", {}).get("jobs")
    if not isinstance(jobs, list):
        raise MergeError("Manifest contract.jobs must be a list")
    selected = [job for job in jobs if isinstance(job, dict) and job.get("protocol") == protocol]
    result = {str(job.get("job_id")): job for job in selected}
    if len(result) != len(selected):
        raise MergeError(f"Duplicate {protocol} job IDs in manifest")
    return result


def _validate_manifests(
    original_manifest: dict[str, Any],
    fixed_manifest: dict[str, Any],
    *,
    label: str,
) -> None:
    if original_manifest.get("status") != "complete":
        raise MergeError(f"Original result is not complete: {label}")
    if fixed_manifest.get("status") != "complete":
        raise MergeError(f"Fixed Prompt Swap result is not complete: {label}")
    original_contract = original_manifest.get("contract")
    fixed_contract = fixed_manifest.get("contract")
    if not isinstance(original_contract, dict) or not isinstance(fixed_contract, dict):
        raise MergeError(f"Missing manifest contract: {label}")
    if tuple(original_contract.get("protocols", ())) != FULL_PROTOCOLS:
        raise MergeError(f"Original result is not the full target protocol: {label}")
    if tuple(fixed_contract.get("protocols", ())) != (PROMPT_PROTOCOL,):
        raise MergeError(f"Fixed result is not Prompt Swap only: {label}")
    if original_contract.get("model_label") != label or fixed_contract.get("model_label") != label:
        raise MergeError(f"Model label mismatch in manifests: {label}")
    _assert_same(
        original_contract,
        fixed_contract,
        (
            "policy_path",
            "policy_config_sha256",
            "split_file",
            "split_sha256",
            "paraphrases",
            "paraphrases_sha256",
            "protocol_config",
            "protocol_config_sha256",
            "env_seed",
            "policy_seed",
            "n_action_steps",
            "max_steps",
        ),
    )
    if original_contract.get("total_rollouts") != FULL_ROLLOUTS:
        raise MergeError(f"Original result must contain 210 rollouts: {label}")
    if fixed_contract.get("total_rollouts") != PROMPT_ROLLOUTS:
        raise MergeError(f"Fixed result must contain 30 rollouts: {label}")
    original_jobs = _job_map(original_manifest, protocol=PROMPT_PROTOCOL)
    fixed_jobs = _job_map(fixed_manifest, protocol=PROMPT_PROTOCOL)
    if set(original_jobs) != set(fixed_jobs) or len(fixed_jobs) != 3:
        raise MergeError(f"Original/fixed Prompt Swap job sets differ: {label}")
    for job_id in fixed_jobs:
        original_job = original_jobs[job_id]
        fixed_job = fixed_jobs[job_id]
        _assert_same(
            original_job,
            fixed_job,
            (
                "protocol",
                "task_id",
                "init_state_ids",
                "episode_seeds",
                "instruction",
                "expected_goal",
            ),
        )


def _validate_fixed_result(
    fixed_root: Path,
    fixed_manifest: dict[str, Any],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_path = fixed_root / "metrics.json"
    episodes_path = fixed_root / "episodes.parquet"
    _require_file(metrics_path)
    _require_file(episodes_path)
    metrics = _load_json(metrics_path)
    if metrics.get("status") != "complete":
        raise MergeError(f"Fixed metrics are not complete: {label}")
    if metrics.get("model_label") != label:
        raise MergeError(f"Fixed metrics model label mismatch: {label}")
    if tuple(metrics.get("protocols", ())) != (PROMPT_PROTOCOL,):
        raise MergeError(f"Fixed metrics are not Prompt Swap only: {label}")
    if metrics.get("total_rollouts") != PROMPT_ROLLOUTS:
        raise MergeError(f"Fixed metrics must contain 30 rollouts: {label}")
    prompt_metrics = metrics.get("prompt_swap")
    if not isinstance(prompt_metrics, dict):
        raise MergeError(f"Fixed result lacks prompt_swap metrics: {label}")
    table = pq.read_table(episodes_path)
    records = [dict(row) for row in table.to_pylist()]
    summary = _summary_from_records(records)
    for field in (
        "num_episodes",
        "num_full_horizon_episodes",
        "num_compliant",
        "prompt_compliance",
        "first_goal_num_compliant",
        "first_goal_prompt_compliance",
        "terminal_goal_num_compliant",
        "terminal_goal_prompt_compliance",
    ):
        if prompt_metrics.get(field) != summary[field]:
            raise MergeError(
                f"Fixed metrics mismatch for {label}: prompt_swap.{field} "
                f"is {prompt_metrics.get(field)!r}, derived {summary[field]!r}"
            )
    if prompt_metrics.get("confusion_matrix") != summary["confusion_matrix"]:
        raise MergeError(f"Fixed first-goal confusion matrix mismatch: {label}")
    if (
        prompt_metrics.get("terminal_goal_confusion_matrix")
        != summary["terminal_goal_confusion_matrix"]
    ):
        raise MergeError(f"Fixed terminal-goal confusion matrix mismatch: {label}")
    jobs = _job_map(fixed_manifest, protocol=PROMPT_PROTOCOL)
    for job_id, job in jobs.items():
        episode_records = [record for record in records if record.get("job_id") == job_id]
        if len(episode_records) != 10:
            raise MergeError(f"Fixed root must contain 10 rows for {label}/{job_id}")
        if any(
            record.get("task_id") != job["task_id"]
            or record.get("expected_goal") != job["expected_goal"]
            or record.get("instruction") != job["instruction"]
            for record in episode_records
        ):
            raise MergeError(f"Fixed root job metadata differs from manifest: {label}/{job_id}")
        if tuple(int(record["init_state_id"]) for record in episode_records) != tuple(
            int(value) for value in job["init_state_ids"]
        ):
            raise MergeError(f"Fixed root init states differ from manifest: {label}/{job_id}")
        if tuple(int(record["seed"]) for record in episode_records) != tuple(
            int(value) for value in job["episode_seeds"]
        ):
            raise MergeError(f"Fixed root seeds differ from manifest: {label}/{job_id}")
        output_dir = str(job["output_dir"])
        if "/jobs/" in output_dir:
            job_root = fixed_root / "jobs" / Path(output_dir.split("/jobs/", 1)[-1])
        else:
            job_root = fixed_root / "jobs" / "prompt_swap" / str(job["expected_goal"])
        for filename in ("metrics.json", "run_config.json", "rollouts.parquet", "rollout.rrd"):
            _require_file(job_root / filename)
        _require_file(fixed_root / "logs" / "prompt_swap" / f"{job['expected_goal']}.log")
        job_metrics = _load_json(job_root / "metrics.json")
        per_episode = job_metrics.get("per_episode")
        if not isinstance(per_episode, list) or len(per_episode) != 10:
            raise MergeError(f"Fixed job must contain 10 per-episode rows: {label}/{job_id}")
        for episode in per_episode:
            if any(field not in episode for field in PROMPT_FIELDS):
                raise MergeError(f"Fixed job lacks first/terminal fields: {label}/{job_id}")
            if episode.get("num_steps") != 300:
                raise MergeError(f"Fixed job is not full horizon: {label}/{job_id}")
        if tuple(int(episode["init_state_id"]) for episode in per_episode) != tuple(
            int(value) for value in job["init_state_ids"]
        ):
            raise MergeError(f"Fixed job init states differ from manifest: {label}/{job_id}")
        if tuple(int(episode["seed"]) for episode in per_episode) != tuple(
            int(value) for value in job["episode_seeds"]
        ):
            raise MergeError(f"Fixed job seeds differ from manifest: {label}/{job_id}")
        if any(episode.get("expected_goal") != job["expected_goal"] for episode in per_episode):
            raise MergeError(f"Fixed job expected goals differ from manifest: {label}/{job_id}")
    return records, summary


def _build_merged_episodes(original_path: Path, fixed_path: Path) -> pa.Table:
    original_table = pq.read_table(original_path)
    fixed_table = pq.read_table(fixed_path)
    original_rows = [dict(row) for row in original_table.to_pylist()]
    fixed_rows = [dict(row) for row in fixed_table.to_pylist()]
    untouched = [row for row in original_rows if row.get("protocol") != PROMPT_PROTOCOL]
    if len(untouched) != 180:
        raise MergeError("Original episodes.parquet must retain 180 non-Prompt-Swap rows")
    for row in untouched:
        row.setdefault("first_goal_reached", None)
        row.setdefault("first_goal_step", None)
        row.setdefault("terminal_goal", None)
    rows = untouched + fixed_rows
    if len(rows) != FULL_ROLLOUTS:
        raise MergeError(f"Merged episodes.parquet must contain 210 rows, got {len(rows)}")
    names = list(original_table.schema.names)
    for field in PROMPT_FIELDS:
        if field not in names:
            names.append(field)
    field_types = {field.name: field.type for field in original_table.schema}
    for field in fixed_table.schema:
        field_types.setdefault(field.name, field.type)
    schema = pa.schema([pa.field(name, field_types[name], nullable=True) for name in names])
    return pa.Table.from_pylist(rows, schema=schema)


def _copy_with_replacement(source: Path, destination: Path, replacements: dict[str, str]) -> None:
    if source.suffix.lower() in {".json", ".md", ".log"}:
        text = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        _write_text_atomic(destination, text)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _backup_path(source: Path, backup_root: Path, root: Path) -> None:
    relative = source.resolve().relative_to(root.resolve())
    destination = backup_root / relative
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _replace_directory(source: Path, destination: Path, replacements: dict[str, str]) -> None:
    """Copy a result tree over the old tree without requiring same-device rename."""

    if destination.is_symlink():
        raise MergeError(f"Refusing to overwrite symlink destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    # Keep the destination directory inode/mount intact. This is intentional:
    # remote output trees may be mounted on a different device from their
    # parent, so replacing the directory itself with Path.replace() raises
    # EXDEV. The original tree was backed up by _merge_one before this call.
    for child in list(destination.iterdir()):
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)

    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        else:
            _copy_with_replacement(source_path, destination_path, replacements)


def _render_report_notice(
    *,
    short_notice: str,
    summaries: list[tuple[str, dict[str, Any]]],
    overall_report: Path,
    repository_root: Path,
) -> str:
    lines = [
        MERGE_NOTICE_BEGIN,
        "## Prompt Swap 修复后当前结果",
        "",
        f"**{short_notice}**",
        "",
        (
            "以下是当前有效的结果；本文其余位置中修复前的 Prompt Swap 数值、混淆矩阵和由此得到的结论，"
            "均只保留作历史审计，不得继续引用。"
        ),
        "",
        "| Model | First-goal compliance | Terminal-goal compliance | Full horizon |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, summary in summaries:
        lines.append(
            f"| {label} | {summary['first_goal_num_compliant']}/30 = "
            f"{summary['first_goal_prompt_compliance']:.1%} | "
            f"{summary['terminal_goal_num_compliant']}/30 = "
            f"{summary['terminal_goal_prompt_compliance']:.1%} | "
            f"{summary['num_full_horizon_episodes']}/30 |"
        )
    lines.extend(
        [
            "",
            (
                "主指标是每一步 simulator predicate 的 `first_goal_reached`；"
                "`terminal_goal` 只表示达成后的保持性诊断。"
            ),
            (
                "这次合并只替换 Prompt Swap 的 job/rollout/episode 与汇总结果，"
                "canonical、held-out paraphrase、训练和 checkpoint selection 结论不变。"
            ),
            f"详细合并报告：`{_relative(overall_report, repository_root)}`。",
        ]
    )
    lines.extend(
        [
            MERGE_NOTICE_END,
            "",
        ]
    )
    return "\n".join(lines)


def _insert_notice(text: str, notice: str, *, after_heading: str | None) -> str:
    pattern = re.compile(
        re.escape(MERGE_NOTICE_BEGIN) + r".*?" + re.escape(MERGE_NOTICE_END) + r"\n?",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(notice, text, count=1)
    if after_heading is not None:
        marker = after_heading + "\n"
        if marker not in text:
            raise MergeError(f"Report heading not found: {after_heading}")
        return text.replace(marker, marker + "\n" + notice, 1)
    first_line_end = text.find("\n")
    if first_line_end < 0:
        return text + "\n\n" + notice
    return text[: first_line_end + 1] + "\n" + notice + text[first_line_end + 1 :]


def _replace_prompt_compliance_row(text: str, summaries: list[tuple[str, dict[str, Any]]]) -> str:
    row = (
        "| prompt compliance | "
        + " | ".join(
            f"{summary['first_goal_num_compliant']}/30 = "
            f"{summary['first_goal_prompt_compliance']:.1%}"
            for _, summary in summaries
        )
        + " |"
    )
    if "| prompt compliance |" not in text:
        return text
    return re.sub(r"^\| prompt compliance \|.*$", row, text, count=1, flags=re.MULTILINE)


def _historical_prompt_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    merge_provenance = metrics.get("prompt_swap_merge")
    if isinstance(merge_provenance, dict):
        original_prompt_metrics = merge_provenance.get("original_prompt_swap")
        if isinstance(original_prompt_metrics, dict):
            return dict(original_prompt_metrics)

    prompt_metrics = metrics.get("prompt_swap")
    if not isinstance(prompt_metrics, dict):
        raise MergeError("Original metrics lacks prompt_swap")
    return dict(prompt_metrics)


def _render_model_report(
    *,
    label: str,
    summary: dict[str, Any],
    old_prompt_metrics: dict[str, Any],
    merged_root: Path,
    fixed_root: Path,
    repository_root: Path,
) -> str:
    matrix = summary["confusion_matrix"]
    terminal_matrix = summary["terminal_goal_confusion_matrix"]
    old_rate = old_prompt_metrics.get("prompt_compliance")
    old_rendered = "N/A" if old_rate is None else f"{float(old_rate):.1%}"

    def matrix_lines(value: dict[str, dict[str, int]]) -> list[str]:
        lines = [
            "| Input prompt | Stove | Cabinet | Plate | Failed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for goal in GOAL_LABELS:
            row = value[goal]
            lines.append(
                f"| {goal} | {row['stove']} | {row['cabinet']} | {row['plate']} | {row['failed']} |"
            )
        return lines

    lines = [
        f"# Prompt Swap 修复后合并结果：{label}",
        "",
        "## 重要进展",
        "",
        f"- 正式 Prompt Swap 30 条已从 `{_relative(fixed_root, repository_root)}` 合并到原始结果目录。",
        (
            f"- 主指标：首次达成目标 {summary['first_goal_num_compliant']}/30 = "
            f"**{summary['first_goal_prompt_compliance']:.1%}**。"
        ),
        (
            f"- 终态保持诊断：{summary['terminal_goal_num_compliant']}/30 = "
            f"**{summary['terminal_goal_prompt_compliance']:.1%}**。"
        ),
        (
            f"- 30/30 条 rollout 均跑满 300 步；首次目标与终态目标不同的 episode 为 "
            f"{summary['num_first_terminal_disagreements']} 条。"
        ),
        "- canonical 与 held-out paraphrase 原始 180 条记录未改动。",
        "",
        "## 判定口径",
        "",
        (
            "Prompt Swap 以每一步 simulator predicate 的**首次达成目标**作为主指标；"
            "`terminal_goal` 仅用于观察达成后是否保持。"
        ),
        "",
        "| 指标 | 修复前原始汇总 | 修复后合并值 |",
        "| --- | ---: | ---: |",
        (
            f"| Prompt compliance | {old_rendered} | "
            f"{summary['first_goal_num_compliant']}/30 = "
            f"{summary['first_goal_prompt_compliance']:.1%} |"
        ),
        (
            f"| Terminal goal compliance | 未记录 | "
            f"{summary['terminal_goal_num_compliant']}/30 = "
            f"{summary['terminal_goal_prompt_compliance']:.1%} |"
        ),
        "",
        "## 首次达成目标混淆矩阵（主指标）",
        "",
        *matrix_lines(matrix),
        "",
        "## 终态目标混淆矩阵（保持诊断）",
        "",
        *matrix_lines(terminal_matrix),
        "",
        "## 结论",
        "",
        (
            f"- 对 `{label}`，应引用首次达成主指标 {summary['first_goal_num_compliant']}/30，"
            f"而不是修复前的 {old_rendered}。"
        ),
        (
            f"- 终态与首次目标不一致的 {summary['num_first_terminal_disagreements']} 条，"
            "表示策略在达成目标后没有一直保持该状态；这是保持性诊断，不改变首次达成主指标。"
        ),
        "- 本次合并没有重跑、替换或重新解释 canonical / held-out paraphrase。",
        "",
        "## 结果位置",
        "",
        f"- 合并主题结果：`{_relative(merged_root, repository_root)}`",
        f"- 修复源：`{_relative(fixed_root, repository_root)}`",
        f"- 合并后的 episodes：`{_relative(merged_root / 'episodes.parquet', repository_root)}`",
        "",
        "结果类别：Formal / Closed-loop / Test。",
        "",
    ]
    return "\n".join(lines)


def _merge_one(
    *,
    label: str,
    original_root: Path,
    fixed_root: Path,
    backup_root: Path,
    repository_root: Path,
    apply: bool,
) -> dict[str, Any]:
    original_manifest_path = original_root / "run_manifest.json"
    fixed_manifest_path = fixed_root / "run_manifest.json"
    _require_file(original_manifest_path)
    _require_file(fixed_manifest_path)
    original_manifest = _load_json(original_manifest_path)
    fixed_manifest = _load_json(fixed_manifest_path)
    _validate_manifests(original_manifest, fixed_manifest, label=label)
    _, summary = _validate_fixed_result(fixed_root, fixed_manifest, label=label)

    original_metrics_path = original_root / "metrics.json"
    original_episodes_path = original_root / "episodes.parquet"
    _require_file(original_metrics_path)
    _require_file(original_episodes_path)
    for manifest, metrics_path, source_name in (
        (original_manifest, original_metrics_path, "original"),
        (fixed_manifest, fixed_root / "metrics.json", "fixed"),
    ):
        declared_sha256 = manifest.get("metrics_sha256")
        if declared_sha256 != _sha256(metrics_path):
            raise MergeError(f"{source_name} metrics sha256 mismatch: {label}")
    original_metrics = _load_json(original_metrics_path)
    original_table = pq.read_table(original_episodes_path)
    original_records = [dict(row) for row in original_table.to_pylist()]
    original_prompt_records = [
        row for row in original_records if row.get("protocol") == PROMPT_PROTOCOL
    ]
    if len(original_records) != FULL_ROLLOUTS or len(original_prompt_records) != PROMPT_ROLLOUTS:
        raise MergeError(f"Original episodes.parquet has unexpected shape: {label}")
    old_prompt_metrics = _historical_prompt_metrics(original_metrics)

    merged_table = _build_merged_episodes(original_episodes_path, fixed_root / "episodes.parquet")
    merged_records = [dict(row) for row in merged_table.to_pylist()]
    if Counter(row.get("protocol") for row in merged_records) != Counter(
        {"canonical": 90, "held_out_paraphrase": 90, PROMPT_PROTOCOL: 30}
    ):
        raise MergeError(f"Merged protocol counts are invalid: {label}")

    merged_metrics = dict(original_metrics)
    merged_metrics["schema_version"] = 2
    merged_metrics["status"] = "complete"
    merged_metrics["protocols"] = list(FULL_PROTOCOLS)
    merged_metrics["total_rollouts"] = FULL_ROLLOUTS
    merged_metrics["prompt_swap"] = summary
    merged_metrics["prompt_swap_merge"] = {
        "status": "merged",
        "source_output_dir": _relative(fixed_root, repository_root),
        "source_metrics_sha256": _sha256(fixed_root / "metrics.json"),
        "primary_metric": "first_goal_reached",
        "terminal_diagnostic": "terminal_goal",
        "canonical_and_paraphrase_reused": True,
        "original_prompt_swap": old_prompt_metrics,
    }

    if not apply:
        return {
            "label": label,
            "summary": summary,
            "old_prompt_metrics": old_prompt_metrics,
            "original_root": original_root,
            "fixed_root": fixed_root,
            "merged_report_path": original_root / "prompt_swap_merged.md",
        }

    model_backup = backup_root / label
    for relative in (
        "metrics.json",
        "metrics.md",
        "episodes.parquet",
        "prompt_goal_confusion.csv",
        "prompt_goal_confusion_terminal.csv",
        "run_manifest.json",
        "prompt_swap_merged.md",
    ):
        source = original_root / relative
        if source.exists():
            _backup_path(source, model_backup, original_root)
    for relative in ("jobs/prompt_swap", "logs/prompt_swap"):
        source = original_root / relative
        if source.exists():
            _backup_path(source, model_backup, original_root)

    replacement_paths = {
        _relative(fixed_root, repository_root): _relative(original_root, repository_root),
        str(fixed_root): str(original_root),
    }
    _replace_directory(
        fixed_root / "jobs" / "prompt_swap",
        original_root / "jobs" / "prompt_swap",
        replacement_paths,
    )
    _replace_directory(
        fixed_root / "logs" / "prompt_swap",
        original_root / "logs" / "prompt_swap",
        replacement_paths,
    )
    _write_table_atomic(original_episodes_path, merged_table)
    _write_json_atomic(original_metrics_path, merged_metrics)

    terminal_csv_path = original_root / "prompt_goal_confusion_terminal.csv"
    _write_confusion_csv(original_root / "prompt_goal_confusion.csv", summary["confusion_matrix"])
    _write_confusion_csv(terminal_csv_path, summary["terminal_goal_confusion_matrix"])

    metrics_markdown_path = original_root / "metrics.md"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=original_root, prefix=".metrics.md.", delete=False
    ) as stream:
        temporary_metrics_markdown = Path(stream.name)
    _write_metrics_markdown(temporary_metrics_markdown, merged_metrics)
    generated_metrics = temporary_metrics_markdown.read_text(encoding="utf-8")
    generated_metrics += (
        "\n## Prompt Swap merge provenance\n\n"
        f"- Corrected source: `{_relative(fixed_root, repository_root)}`\n"
        "- Primary metric: `first_goal_reached`; terminal diagnostic: `terminal_goal`.\n"
        "- Canonical and held-out paraphrase rows were reused from the original result.\n"
    )
    _write_text_atomic(metrics_markdown_path, generated_metrics)
    temporary_metrics_markdown.unlink(missing_ok=True)

    merged_manifest = dict(original_manifest)
    merged_manifest["metrics_sha256"] = _sha256(original_metrics_path)
    merged_manifest["prompt_swap_merge"] = {
        "status": "merged",
        "source_output_dir": _relative(fixed_root, repository_root),
        "source_manifest_sha256": _sha256(fixed_manifest_path),
        "primary_metric": "first_goal_reached",
        "terminal_diagnostic": "terminal_goal",
        "canonical_and_paraphrase_reused": True,
    }
    _write_json_atomic(original_manifest_path, merged_manifest)

    merged_report_path = original_root / "prompt_swap_merged.md"
    merged_report = _render_model_report(
        label=label,
        summary=summary,
        old_prompt_metrics=old_prompt_metrics,
        merged_root=original_root,
        fixed_root=fixed_root,
        repository_root=repository_root,
    )
    _write_text_atomic(merged_report_path, merged_report)
    return {
        "label": label,
        "summary": summary,
        "old_prompt_metrics": old_prompt_metrics,
        "original_root": original_root,
        "fixed_root": fixed_root,
        "merged_report_path": merged_report_path,
    }


def _write_confusion_csv(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary_path = Path(stream.name)
        writer = csv.writer(stream)
        writer.writerow(["input_prompt", *GOAL_LABELS, FAILED_GOAL])
        for goal in GOAL_LABELS:
            writer.writerow(
                [goal, *[matrix[goal][column] for column in (*GOAL_LABELS, FAILED_GOAL)]]
            )
    temporary_path.replace(path)


def _render_overall_report(
    results: list[dict[str, Any]],
    *,
    repository_root: Path,
) -> str:
    lines = [
        f"# {datetime.now(timezone.utc).date().isoformat()} Prompt Swap 修复后合并结果",
        "",
        "## 重要进展",
        "",
        "- Z0/C1/C2 的 30 条 Prompt Swap fixed rollout 已分别合并回原始 210 条主题结果目录。",
        "- 主指标统一使用每一步 predicate 的 `first_goal_reached`，不再使用 reset 后的终态作为主判定。",
        "- canonical 与 held-out paraphrase 原始 180 条记录未重跑、未替换。",
        "- 每个模型的 30 条 Prompt Swap episode 均通过 300 步完整性校验。",
        "",
        "## 汇总",
        "",
        "| Model | First-goal compliance | Terminal-goal compliance | First/terminal disagreement | Full horizon |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        summary = result["summary"]
        lines.append(
            f"| {result['label']} | {summary['first_goal_num_compliant']}/30 "
            f"({summary['first_goal_prompt_compliance']:.1%}) | "
            f"{summary['terminal_goal_num_compliant']}/30 "
            f"({summary['terminal_goal_prompt_compliance']:.1%}) | "
            f"{summary['num_first_terminal_disagreements']} | "
            f"{summary['num_full_horizon_episodes']}/30 |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 旧 Prompt Swap 汇总值因参考场景原生目标在 step 返回前触发 reset 而偏低；合并后的主指标应引用首次达成目标。",
            "- `terminal_goal` 低于 `first_goal_reached` 时，说明策略达成后没有保持目标状态；该差异是保持性诊断，不应重新解释为未达成。",
            "- 这次合并只改变 Prompt Swap 判定结果，不改变 canonical、held-out paraphrase、训练或 checkpoint selection 结论。",
            "",
            "## 结果位置",
            "",
        ]
    )
    for result in results:
        lines.append(
            f"- `{result['label']}`：`{_relative(result['original_root'], repository_root)}`；"
            f"合并说明：`{_relative(result['merged_report_path'], repository_root)}`"
        )
    lines.extend(["", "结果类别：Formal / Closed-loop / Test。", ""])
    return "\n".join(lines)


def _update_report_file(
    path: Path,
    *,
    notice: str,
    summaries: list[tuple[str, dict[str, Any]]],
    heading: str | None,
) -> None:
    _require_file(path)
    text = path.read_text(encoding="utf-8")
    text = _replace_prompt_compliance_row(text, summaries)
    text = _insert_notice(text, notice, after_heading=heading)
    _write_text_atomic(path, text)


def _update_experiment_log(
    path: Path,
    *,
    results: list[dict[str, Any]],
    overall_report: Path,
    repository_root: Path,
) -> None:
    _require_file(path)
    lines = [
        MERGE_LOG_BEGIN,
        f"## {datetime.now(timezone.utc).date().isoformat()} — SmolVLA LIBERO Prompt Swap 判定修复后合并",
        "",
        "- Date: " + datetime.now(timezone.utc).date().isoformat(),
        "- Task suite: LIBERO-Goal Prompt Swap reference task 1，stove/cabinet/plate 三目标",
        "- Mode: Formal / Closed-loop / Test（只补跑 Prompt Swap；canonical 与 held-out paraphrase 沿用原始结果）",
        "- 判定口径: `first_goal_reached` 为主指标，`terminal_goal` 为保持性诊断",
        "- Results:",
    ]
    for result in results:
        summary = result["summary"]
        lines.append(
            f"  - {result['label']}: first-goal "
            f"{summary['first_goal_num_compliant']}/30 ({summary['first_goal_prompt_compliance']:.1%}); "
            f"terminal-goal {summary['terminal_goal_num_compliant']}/30 "
            f"({summary['terminal_goal_prompt_compliance']:.1%}); "
            f"full horizon {summary['num_full_horizon_episodes']}/30"
        )
    lines.extend(
        [
            (
                "- Notes: 原始 Prompt Swap 旧值因参考任务成功后 reset 而偏低；合并脚本保留原始 180 条非 Prompt Swap 数据，"
                "并将 corrected job/rollout/episodes/metrics 写回原始主题结果目录。"
            ),
            "- Result artifacts:",
        ]
    )
    for result in results:
        lines.append(f"  - `{_relative(result['original_root'], repository_root)}`")
    lines.extend(
        [
            f"- Consolidated report: `{_relative(overall_report, repository_root)}`",
            MERGE_LOG_END,
            "",
        ]
    )
    block = "\n".join(lines)
    pattern = re.compile(
        re.escape(MERGE_LOG_BEGIN) + r".*?" + re.escape(MERGE_LOG_END) + r"\n?",
        flags=re.DOTALL,
    )
    if pattern.search(path.read_text(encoding="utf-8")):
        updated = pattern.sub(block, path.read_text(encoding="utf-8"), count=1)
    else:
        updated = path.read_text(encoding="utf-8")
        if not updated.endswith("\n"):
            updated += "\n"
        updated += "\n" + block
    _write_text_atomic(path, updated)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("outputs/smolvla_libero/eval"),
        help="Original full target-protocol result root",
    )
    parser.add_argument(
        "--fixed-root",
        type=Path,
        default=Path("outputs/smolvla_libero/eval/prompt_swap_fixed"),
        help="Corrected Prompt Swap-only result root",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_LABELS),
        help="Comma-separated model labels, default: all three corrected runs",
    )
    parser.add_argument(
        "--reports-root", type=Path, default=Path("reports"), help="Reports directory"
    )
    parser.add_argument(
        "--apply-reports",
        action="store_true",
        help="Update formal reports, experiment_log.md, and reports/README.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print derived summaries without changing output files",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    repository_root = Path.cwd().resolve()
    eval_root = arguments.eval_root.resolve()
    fixed_root = arguments.fixed_root.resolve()
    reports_root = arguments.reports_root.resolve()
    labels = tuple(value.strip() for value in arguments.models.split(",") if value.strip())
    if not labels or any(label not in MODEL_LABELS for label in labels):
        raise SystemExit(f"--models must be selected from {', '.join(MODEL_LABELS)}")
    if len(set(labels)) != len(labels):
        raise SystemExit("--models must not contain duplicates")
    if arguments.apply_reports and set(labels) != set(MODEL_LABELS):
        raise SystemExit("--apply-reports requires all three model labels")

    report_pairs = [
        (
            reports_root / "2026-08-04-Z0基线与C1目标适配正式评测结论.md",
            ("Z0/C1 Prompt Swap 已合并：以 first_goal_reached 为主指标。", "## 重要进展"),
            ("z0_b0_zero_shot", "c1_default_non_peft"),
        ),
        (
            reports_root / "2026-08-05-C2 Canonical LoRA 正式评测结论.md",
            ("Z0/C1/C2 Prompt Swap 已合并：以 first_goal_reached 为主指标。", "## 重要进展"),
            ("z0_b0_zero_shot", "c1_default_non_peft", "c2_canonical_lora"),
        ),
        (
            reports_root / "guides/smolvla_libero/03_c1_non_peft.md",
            ("C1/Z0 Prompt Swap 已合并；本文旧 Prompt Swap 数值仅为历史快照。", None),
            ("z0_b0_zero_shot", "c1_default_non_peft"),
        ),
        (
            reports_root / "guides/smolvla_libero/04_c2_canonical_lora.md",
            ("Z0/C1/C2 Prompt Swap 已合并；本文旧 Prompt Swap 数值仅为历史快照。", None),
            ("z0_b0_zero_shot", "c1_default_non_peft", "c2_canonical_lora"),
        ),
    ]
    if arguments.apply_reports:
        for path, (_, heading), _ in report_pairs:
            _require_file(path)
            if heading is not None:
                report_text = path.read_text(encoding="utf-8")
                if heading + "\n" not in report_text:
                    raise SystemExit(f"Report heading not found: {path}: {heading}")
        _require_file(reports_root / "experiment_log.md")
        _require_file(reports_root / "README.md")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = fixed_root / "_merge_backups" / timestamp
    try:
        # Validate every source before changing any original result directory.
        validated_results = [
            _merge_one(
                label=label,
                original_root=eval_root / label,
                fixed_root=fixed_root / label,
                backup_root=backup_root,
                repository_root=repository_root,
                apply=False,
            )
            for label in labels
        ]
        if arguments.dry_run:
            for result in validated_results:
                print(
                    json.dumps(
                        {"model_label": result["label"], **result["summary"]},
                        ensure_ascii=False,
                    )
                )
            return

        results = [
            _merge_one(
                label=label,
                original_root=eval_root / label,
                fixed_root=fixed_root / label,
                backup_root=backup_root,
                repository_root=repository_root,
                apply=True,
            )
            for label in labels
        ]
    except MergeError as exc:
        raise SystemExit(f"Prompt Swap merge aborted before completion: {exc}") from exc

    overall_report = reports_root / "2026-08-06-PromptSwap修复后合并结果.md"
    overall_text = _render_overall_report(results, repository_root=repository_root)
    _write_text_atomic(overall_report, overall_text)
    if arguments.apply_reports:
        by_label = {result["label"]: result for result in results}
        for path, (short_notice, heading), report_labels in report_pairs:
            selected = [(label, by_label[label]["summary"]) for label in report_labels]
            notice = _render_report_notice(
                short_notice=short_notice,
                summaries=selected,
                overall_report=overall_report,
                repository_root=repository_root,
            )
            _update_report_file(path, notice=notice, summaries=selected, heading=heading)
        _update_experiment_log(
            reports_root / "experiment_log.md",
            results=results,
            overall_report=overall_report,
            repository_root=repository_root,
        )
        readme_path = reports_root / "README.md"
        readme = readme_path.read_text(encoding="utf-8")
        link = "- [2026-08-06 Prompt Swap 修复后合并结果](2026-08-06-PromptSwap修复后合并结果.md)"
        if link not in readme:
            readme = readme.rstrip() + "\n\n" + link + "\n"
            _write_text_atomic(readme_path, readme)

    print(f"Merged Prompt Swap results for: {', '.join(labels)}")
    print(f"Backup: {_relative(backup_root, repository_root)}")
    print(f"Overall report: {_relative(overall_report, repository_root)}")
    for result in results:
        summary = result["summary"]
        print(
            f"{result['label']}: first={summary['first_goal_num_compliant']}/30 "
            f"({summary['first_goal_prompt_compliance']:.1%}), "
            f"terminal={summary['terminal_goal_num_compliant']}/30 "
            f"({summary['terminal_goal_prompt_compliance']:.1%}), "
            f"full_horizon={summary['num_full_horizon_episodes']}/30"
        )


if __name__ == "__main__":
    main()
