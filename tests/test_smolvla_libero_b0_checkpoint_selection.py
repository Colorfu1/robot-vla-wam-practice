from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

import robot_practice.experiments.smolvla_libero.b0_checkpoint_selection as selection
from robot_practice.evaluation.libero_assets import (
    REQUIRED_ASSET_DIRECTORIES,
    validate_libero_assets_dir,
)

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
    training_state = policy.parent / "training_state"
    training_state.mkdir()
    (training_state / "state.txt").write_text("state", encoding="utf-8")
    return policy


def _complete_assets(root: Path) -> Path:
    for directory in REQUIRED_ASSET_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    scene = root / "scenes" / "libero_tabletop_base_style.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    return root


def test_frozen_background_closed_loop_split_is_complete_and_disjoint() -> None:
    payload = selection.load_selection_split(SPLIT_FILE)
    checkpoint_steps, policy_seed = selection._selection_protocol(payload)

    assert checkpoint_steps == (5000, 10000, 15000, 20000, 25000, 30000)
    assert policy_seed == 2000
    assert tuple(sorted(int(task_id) for task_id in payload["tasks"])) == (
        0,
        2,
        3,
        5,
        6,
        7,
        9,
    )
    for task in payload["tasks"].values():
        validation = set(task["validation_init_state_ids"])
        test = set(task["test_init_state_ids"])
        assert len(validation) == 20
        assert len(test) == 30
        assert validation.isdisjoint(test)
        assert validation | test == set(range(50))


def test_offline_assets_are_rejected_before_gpu_jobs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_libero_assets_dir(tmp_path / "missing")

    assets = _complete_assets(tmp_path / "assets")
    assert validate_libero_assets_dir(assets) == assets.resolve()


def test_discovery_requires_every_frozen_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_checkpoint(run_dir, 5000)

    assert selection.discover_checkpoints(run_dir, (5000,)) == {
        5000: run_dir / "checkpoints" / "005000" / "pretrained_model"
    }
    with pytest.raises(FileNotFoundError, match="010000"):
        selection.discover_checkpoints(run_dir, (5000, 10000))


def test_explicit_candidate_paths_cover_the_complete_frozen_set(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    policies = {
        step: _complete_checkpoint(run_dir, step) for step in (5000, 10000, 15000)
    }

    discovered = selection.discover_candidate_checkpoints(
        run_dir=run_dir,
        expected_steps=(5000, 10000, 15000),
        checkpoint_paths=(policies[15000], policies[5000], policies[10000]),
    )

    assert discovered == policies
    with pytest.raises(ValueError, match="complete frozen candidate set"):
        selection.discover_candidate_checkpoints(
            run_dir=run_dir,
            expected_steps=(5000, 10000, 15000),
            checkpoint_paths=(policies[5000], policies[10000]),
        )


def test_cli_rejects_absolute_explicit_checkpoint_paths(tmp_path: Path) -> None:
    namespace = selection.build_argument_parser().parse_args(
        ["--checkpoint-path", str(tmp_path / "checkpoints/005000/pretrained_model")]
    )

    with pytest.raises(ValueError, match="repository-relative"):
        selection._resolve_arguments(namespace)


def test_evaluation_command_binds_frozen_states_and_policy_seed(tmp_path: Path) -> None:
    job = selection.SelectionJob(
        checkpoint_step=5000,
        checkpoint_path=tmp_path / "run/checkpoints/005000/pretrained_model",
        task_id=2,
        init_state_ids=(4, 7, 9),
        output_dir=tmp_path / "selection/jobs/checkpoint_005000/task_02",
        log_path=tmp_path / "selection/logs/job.log",
    )

    command = selection.build_evaluation_command(
        job,
        env_seed=1000,
        policy_seed=2000,
        n_action_steps=10,
        libero_assets_dir=tmp_path / "assets",
        overwrite=False,
    )

    assert "--n-episodes=3" in command
    assert "--init-state-ids=4,7,9" in command
    assert "--policy-seed=2000" in command
    assert any(item.startswith("--libero-assets-dir=") for item in command)
    assert "--num-visualized-episodes=0" in command
    assert "--no-open-viewer" in command


def test_coordinator_accepts_the_flushed_rollout_progress_protocol() -> None:
    line = "[RolloutProgress] completed=8 total=30 init_state_id=12 success=True"

    match = selection.ROLLOUT_PROGRESS_PATTERN.search(line)

    assert match is not None
    assert match.groupdict() == {"completed": "8", "total": "30"}


def test_ranking_uses_macro_success_then_registered_tie_breakers() -> None:
    results = [
        {
            "checkpoint_step": 5000,
            "macro_success_rate": 0.5,
            "mean_max_reward": 0.7,
            "mean_successful_episode_steps": 120.0,
        },
        {
            "checkpoint_step": 10000,
            "macro_success_rate": 0.6,
            "mean_max_reward": 0.6,
            "mean_successful_episode_steps": 140.0,
        },
        {
            "checkpoint_step": 15000,
            "macro_success_rate": 0.6,
            "mean_max_reward": 0.6,
            "mean_successful_episode_steps": 130.0,
        },
    ]

    ranked = sorted(results, key=selection.checkpoint_rank_key)

    assert [item["checkpoint_step"] for item in ranked] == [15000, 10000, 5000]


def test_selected_checkpoint_is_a_real_self_contained_copy(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    policy = _complete_checkpoint(run_dir, 5000)
    selected_dir = run_dir / "checkpoints" / "selected"
    metadata = {"selected_checkpoint_step": 5000}

    selection._copy_selected_checkpoint(
        source_checkpoint_dir=policy.parent,
        selected_dir=selected_dir,
        selection=metadata,
    )

    assert not selected_dir.is_symlink()
    assert (selected_dir / "pretrained_model" / "model.safetensors").is_file()
    assert (selected_dir / "training_state" / "state.txt").is_file()
    assert (
        json.loads((selected_dir / "selected_checkpoint.json").read_text(encoding="utf-8"))
        == metadata
    )


def test_selected_checkpoint_publication_never_renames_a_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    policy = _complete_checkpoint(run_dir, 5000)
    selected_dir = run_dir / "checkpoints" / "selected"
    original_replace = Path.replace

    def reject_directory_replace(source: Path, target: Path) -> Path:
        if source.is_dir():
            raise AssertionError("selected checkpoint publication must not rename a directory")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", reject_directory_replace)
    selection._copy_selected_checkpoint(
        source_checkpoint_dir=policy.parent,
        selected_dir=selected_dir,
        selection={"selected_checkpoint_step": 5000},
    )

    assert (selected_dir / "pretrained_model" / "model.safetensors").is_file()


def test_selection_fails_fast_and_terminates_other_gpu_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = _complete_assets(tmp_path / "assets")
    run_dir = tmp_path / "run"
    checkpoint = _complete_checkpoint(run_dir, 5000)
    output_dir = run_dir / "checkpoint_selection"
    jobs = [
        selection.SelectionJob(
            checkpoint_step=5000,
            checkpoint_path=checkpoint,
            task_id=task_id,
            init_state_ids=(0,),
            output_dir=output_dir / f"task_{task_id}",
            log_path=output_dir / "logs" / f"task_{task_id}.log",
        )
        for task_id in (0, 2)
    ]
    monkeypatch.setattr(
        selection,
        "discover_checkpoints",
        lambda *args, **kwargs: {5000: checkpoint},
    )
    monkeypatch.setattr(selection, "build_jobs", lambda **kwargs: jobs)

    def fake_command(job: selection.SelectionJob, **kwargs: object) -> list[str]:
        del kwargs
        code = (
            "import time; time.sleep(0.2); raise SystemExit(7)"
            if job.task_id == 0
            else "import time; time.sleep(3)"
        )
        return [sys.executable, "-c", code]

    monkeypatch.setattr(selection, "build_evaluation_command", fake_command)
    arguments = selection.SelectionArguments(
        run_dir=run_dir,
        checkpoint_paths=None,
        split_file=SPLIT_FILE,
        num_gpus=2,
        env_seed=1000,
        policy_seed=2000,
        n_action_steps=10,
        libero_assets_dir=assets,
        output_dir=output_dir,
        selected_dir=run_dir / "checkpoints" / "selected",
        resume=False,
        overwrite=False,
        print_command=False,
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="task 0 failed"):
        selection.select_checkpoints(arguments)

    assert time.monotonic() - started < 2.0
