from __future__ import annotations

import json
from pathlib import Path

from robot_practice.experiments.maniskill.select_official_stackcube_checkpoint import (
    select_checkpoint,
)


def _write_metrics(path: Path, *, success_once: float, success_at_end: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "success_once": {"mean": success_once},
                    "success_at_end": {"mean": success_at_end},
                }
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_selection_ranks_exact_metrics_and_writes_selected_metrics(
    tmp_path: Path,
) -> None:
    train_run = tmp_path / "train" / "runs" / "run"
    checkpoint_dir = train_run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step in (10000, 20000, 30000):
        (checkpoint_dir / f"{step}.pt").write_bytes(f"checkpoint-{step}".encode())

    eval_root = tmp_path / "eval"
    _write_metrics(
        eval_root / "checkpoint_010000" / "metrics.json",
        success_once=0.4,
        success_at_end=0.4,
    )
    _write_metrics(
        eval_root / "checkpoint_020000" / "metrics.json",
        success_once=0.5,
        success_at_end=0.2,
    )
    _write_metrics(
        eval_root / "checkpoint_030000" / "metrics.json",
        success_once=0.5,
        success_at_end=0.3,
    )

    output_dir = tmp_path / "selected"
    result = select_checkpoint(
        train_run=train_run,
        evaluation_root=eval_root,
        output_dir=output_dir,
        selection_source="fixed state-bank development evaluation",
    )

    assert result["selected"]["step"] == 30000
    assert result["selection_protocol"]["selection_source"] == (
        "fixed state-bank development evaluation"
    )
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))["evaluation"][
        "success_once"
    ]["mean"] == 0.5


def test_checkpoint_selection_accepts_rate_metrics_from_state_bank_evaluator(
    tmp_path: Path,
) -> None:
    train_run = tmp_path / "train"
    checkpoint_dir = train_run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "100.pt").write_bytes(b"checkpoint")
    eval_root = tmp_path / "eval"
    metrics_path = eval_root / "checkpoint_000100" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "success_once": {"rate": 0.8},
                    "success_at_end": {"rate": 0.7},
                }
            }
        ),
        encoding="utf-8",
    )

    result = select_checkpoint(
        train_run=train_run,
        evaluation_root=eval_root,
        output_dir=tmp_path / "selected",
    )

    assert result["selected"]["success_once"] == 0.8
