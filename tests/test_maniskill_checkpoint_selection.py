from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_practice.experiments.maniskill import checkpoint_selection as selection_module
from robot_practice.experiments.maniskill.checkpoint_selection import (
    discover_checkpoint_steps,
    rank_candidates,
    select_checkpoint,
)


def _write_candidate(
    tmp_path: Path,
    *,
    step: int,
    success_at_end: float,
    success_once: float,
    num_episodes: int = 200,
    seed_set: str = "development",
) -> tuple[Path, Path]:
    run_dir = tmp_path / "train_main"
    policy_path = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    policy_path.mkdir(parents=True, exist_ok=True)
    (policy_path / "config.json").write_text("{}", encoding="utf-8")
    (policy_path / "model.safetensors").write_bytes(b"model")

    metrics_dir = tmp_path / "evaluations" / f"checkpoint_{step:06d}" / "development"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": 2,
        "system": "visual",
        "task": "PickCube-v1",
        "control_mode": "pd_ee_delta_pose",
        "seed_set": seed_set,
        "num_episodes": num_episodes,
        "evaluation": {
            "num_episodes": num_episodes,
            "success_at_end": {"rate": success_at_end},
            "success_once": {"rate": success_once},
        },
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    run_manifest = {"policy_path": str(policy_path.resolve()), "sim_backend": "physx_cuda"}
    (metrics_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest),
        encoding="utf-8",
    )
    return run_dir, tmp_path / "evaluations"


def test_relative_selection_paths_resolve_from_repository_root() -> None:
    assert selection_module._practice_root() == Path(__file__).resolve().parents[1]


def test_selection_ranks_terminal_success_then_once_then_earlier_step() -> None:
    candidates = [
        {
            "checkpoint_step": 20000,
            "success_at_end_rate": 0.5,
            "success_once_rate": 0.7,
        },
        {
            "checkpoint_step": 10000,
            "success_at_end_rate": 0.5,
            "success_once_rate": 0.8,
        },
        {
            "checkpoint_step": 5000,
            "success_at_end_rate": 0.5,
            "success_once_rate": 0.8,
        },
    ]

    ranked = rank_candidates(candidates)

    assert [row["checkpoint_step"] for row in ranked] == [5000, 10000, 20000]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_select_checkpoint_writes_a_machine_readable_winner(tmp_path: Path) -> None:
    run_dir, evaluation_root = _write_candidate(
        tmp_path,
        step=10000,
        success_at_end=0.4,
        success_once=0.6,
    )
    _write_candidate(
        tmp_path,
        step=20000,
        success_at_end=0.5,
        success_once=0.7,
    )

    selection = select_checkpoint(
        run_dir=run_dir,
        evaluation_root=evaluation_root,
        output_dir=tmp_path / "selection",
        checkpoint_steps=(10000, 20000),
    )

    assert selection["selected_checkpoint_step"] == 20000
    assert selection["primary_metric"] == "success_at_end_rate"
    saved = json.loads((tmp_path / "selection/selected_checkpoint.json").read_text())
    assert saved["selected_checkpoint_step"] == 20000
    assert len(saved["ranking"]) == 2


def test_select_checkpoint_publishes_self_contained_winner_copy(tmp_path: Path) -> None:
    run_dir, evaluation_root = _write_candidate(
        tmp_path,
        step=10000,
        success_at_end=0.4,
        success_once=0.6,
    )
    _write_candidate(
        tmp_path,
        step=20000,
        success_at_end=0.5,
        success_once=0.7,
    )

    selection = select_checkpoint(
        run_dir=run_dir,
        evaluation_root=evaluation_root,
        output_dir=tmp_path / "selection",
        checkpoint_steps=(10000, 20000),
    )

    selected_dir = tmp_path / "selection" / "selected"
    assert selected_dir.is_dir()
    assert not selected_dir.is_symlink()
    assert (selected_dir / "pretrained_model" / "config.json").is_file()
    assert (selected_dir / "pretrained_model" / "model.safetensors").read_bytes() == b"model"
    assert selection["selected_policy_path"] == str(
        (selected_dir / "pretrained_model").resolve()
    )
    published = json.loads((selected_dir / "selected_checkpoint.json").read_text())
    assert published["selected_checkpoint_step"] == 20000
    assert published["source_policy_path"].endswith("checkpoints/020000/pretrained_model")


def test_select_checkpoint_republishes_existing_selection_output(tmp_path: Path) -> None:
    run_dir, evaluation_root = _write_candidate(
        tmp_path,
        step=10000,
        success_at_end=0.4,
        success_once=0.6,
    )
    output_dir = tmp_path / "selection"
    select_checkpoint(
        run_dir=run_dir,
        evaluation_root=evaluation_root,
        output_dir=output_dir,
        checkpoint_steps=(10000,),
    )
    (output_dir / "selected" / "stale.txt").write_text("stale", encoding="utf-8")

    select_checkpoint(
        run_dir=run_dir,
        evaluation_root=evaluation_root,
        output_dir=output_dir,
        checkpoint_steps=(10000,),
    )

    assert not (output_dir / "selected" / "stale.txt").exists()
    assert (output_dir / "selected" / "pretrained_model" / "model.safetensors").is_file()


def test_select_checkpoint_rejects_non_development_or_incomplete_counts(tmp_path: Path) -> None:
    run_dir, evaluation_root = _write_candidate(
        tmp_path,
        step=10000,
        success_at_end=0.4,
        success_once=0.6,
        num_episodes=50,
        seed_set="development",
    )

    with pytest.raises(ValueError, match="num_episodes"):
        select_checkpoint(
            run_dir=run_dir,
            evaluation_root=evaluation_root,
            output_dir=tmp_path / "selection",
            checkpoint_steps=(10000,),
        )


def test_discover_checkpoint_steps_is_sorted(tmp_path: Path) -> None:
    run_dir, _ = _write_candidate(
        tmp_path,
        step=20000,
        success_at_end=0.5,
        success_once=0.5,
    )
    _write_candidate(
        tmp_path,
        step=10000,
        success_at_end=0.4,
        success_once=0.4,
    )

    assert discover_checkpoint_steps(run_dir) == (10000, 20000)
