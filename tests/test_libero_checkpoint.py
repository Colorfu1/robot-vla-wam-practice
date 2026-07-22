from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_practice.evaluation.common import CheckpointManifest
from robot_practice.evaluation.libero_checkpoint import (
    assess_checkpoint_task,
    log_checkpoint_task_assessment,
)
from robot_practice.evaluation.libero_tasks import resolve_libero_task


def _manifest(tmp_path: Path, *, suite: str | None, task_id: int | None) -> CheckpointManifest:
    checkpoint = tmp_path / "run" / "checkpoints" / "000000" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    if suite is not None and task_id is not None:
        payload = {
            "dataset": {"task_suite": suite, "simulator_task_id": task_id},
        }
        (tmp_path / "run" / "initialization_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return CheckpointManifest(
        path=checkpoint,
        policy_config={"type": "act"},
        train_config=None,
        processor_state_files=(),
    )


def test_checkpoint_explicit_task_match_and_mismatch(tmp_path: Path) -> None:
    selected = resolve_libero_task("libero_object", 8)
    match = assess_checkpoint_task(
        _manifest(tmp_path / "match", suite="libero_object", task_id=8), selected
    )
    mismatch = assess_checkpoint_task(
        _manifest(tmp_path / "mismatch", suite="libero_goal", task_id=0), selected
    )
    assert match.status == "match"
    assert match.inferred_task == selected
    assert mismatch.status == "mismatch"
    assert mismatch.inferred_task == resolve_libero_task("libero_goal", 0)


def test_checkpoint_without_task_evidence_is_unknown(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    selected = resolve_libero_task("libero_object", 8)
    assessment = assess_checkpoint_task(
        _manifest(tmp_path, suite=None, task_id=None), selected
    )
    assert assessment.status == "unknown"
    with caplog.at_level("WARNING"):
        log_checkpoint_task_assessment(assessment)
    assert "[Warning]" in caplog.text
    assert "no language input" in caplog.text
