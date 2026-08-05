from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest

import robot_practice.experiments.smolvla_libero.target_checkpoint_selection as target
from robot_practice.evaluation.libero_assets import (
    REQUIRED_ASSET_DIRECTORIES,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_split.json"
SMOKE_CHECKPOINT = (
    PRACTICE_ROOT
    / "outputs/smolvla_libero/debug/c1_num_gpus_smoke/checkpoints/000002/pretrained_model"
)
DATASET_ROOT = PRACTICE_ROOT / "datasets/lerobot/lerobot/libero"


def _complete_c_checkpoint(
    run_dir: Path,
    step: int,
    *,
    adapter: bool = False,
    job_name: str = "smolvla_libero_c1_default_non_peft",
) -> Path:
    policy = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    policy.mkdir(parents=True)
    (policy / "config.json").write_text('{"type": "smolvla"}', encoding="utf-8")
    (policy / "train_config.json").write_text(
        json.dumps({"job_name": job_name}),
        encoding="utf-8",
    )
    if adapter:
        (policy / "adapter_config.json").write_text("{}", encoding="utf-8")
        (policy / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    else:
        (policy / "model.safetensors").write_text("model", encoding="utf-8")
    return policy


def _complete_assets(root: Path) -> Path:
    for directory in REQUIRED_ASSET_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    scene = root / "scenes" / "libero_tabletop_base_style.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    return root


def _arguments(tmp_path: Path, **overrides: object) -> target.TargetSelectionArguments:
    values: dict[str, object] = {
        "run_dir": tmp_path / "run",
        "checkpoint_steps": target.FROZEN_CHECKPOINT_STEPS,
        "split_file": SPLIT_FILE,
        "dataset_root": tmp_path / "dataset",
        "num_gpus": 1,
        "env_seed": 0,
        "policy_seed": target.DEFAULT_POLICY_SEED,
        "libero_assets_dir": tmp_path / "assets",
        "output_dir": tmp_path / "run" / "checkpoint_selection",
        "selected_dir": tmp_path / "run" / "checkpoints" / "selected",
        "resume": False,
        "overwrite": False,
        "print_command": False,
        "max_validation_states": None,
        "max_offline_frames_per_task": None,
    }
    values.update(overrides)
    return target.TargetSelectionArguments(**values)


def test_candidate_set_rejects_missing_duplicate_cross_run_and_absolute(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in target.FROZEN_CHECKPOINT_STEPS[:5]:
        _complete_c_checkpoint(run_dir, step)

    with pytest.raises(FileNotFoundError, match="incomplete"):
        target.resolve_candidate_checkpoints(run_dir=run_dir, checkpoint_steps=None)

    with pytest.raises(ValueError, match="duplicates"):
        target.resolve_candidate_checkpoints(
            run_dir=run_dir,
            checkpoint_steps=(5000, 5000),
        )

    _complete_c_checkpoint(
        run_dir,
        30000,
        job_name="smolvla_libero_b0_domain_adaptation",
    )
    with pytest.raises(ValueError, match="cross-run"):
        target.resolve_candidate_checkpoints(
            run_dir=run_dir,
            checkpoint_steps=(5000, 30000),
        )

    with pytest.raises(argparse.ArgumentTypeError):
        target._parse_checkpoint_steps("5000,/tmp/absolute")
    with pytest.raises(ValueError, match="subset"):
        target.resolve_candidate_checkpoints(
            run_dir=run_dir,
            checkpoint_steps=(5000, 9999),
        )


def test_closed_loop_jobs_cover_validation_states(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for step in target.FROZEN_CHECKPOINT_STEPS:
        _complete_c_checkpoint(run_dir, step)
    checkpoints = target.resolve_candidate_checkpoints(
        run_dir=run_dir,
        checkpoint_steps=None,
    )
    split = target.load_target_selection_split(SPLIT_FILE)
    jobs = target.build_target_jobs(
        checkpoints=checkpoints,
        split=split,
        output_dir=tmp_path / "selection",
        max_validation_states=None,
    )

    assert len(jobs) == 18
    assert sum(job.rollout_count for job in jobs) == 360
    for job in jobs:
        expected = tuple(
            split["tasks"][str(job.task_id)]["closed_loop_split"]["validation_init_state_ids"]
        )
        assert job.init_state_ids == expected
        test_states = split["tasks"][str(job.task_id)]["closed_loop_split"]["test_init_state_ids"]
        assert not set(job.init_state_ids) & set(test_states)


def test_ranking_uses_mse_then_step_and_never_reward() -> None:
    base = {
        "checkpoint_step": 10000,
        "macro_success_rate": 0.5,
        "offline_normalized_action_mse": 0.2,
        "num_episodes": 60,
        "num_successes": 30,
    }
    better_mse = dict(base, checkpoint_step=20000, offline_normalized_action_mse=0.1)
    earlier = dict(base, checkpoint_step=5000, offline_normalized_action_mse=0.2)
    key = target.checkpoint_rank_key(base)
    assert len(key) == 3
    assert "mean_max_reward" not in base

    ranked = sorted(
        [dict(base), better_mse, earlier],
        key=target.checkpoint_rank_key,
    )
    assert ranked[0]["checkpoint_step"] == 20000
    assert ranked[1]["checkpoint_step"] == 5000
    assert ranked[2]["checkpoint_step"] == 10000

    audit = target.build_ranking_audit(ranked)
    assert audit[0]["rank"] == 1
    assert audit[1]["tie_breaker_level"] == 2
    assert audit[2]["tie_breaker_level"] == 3


def test_validation_overlap_is_rejected(tmp_path: Path) -> None:
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    task = split["tasks"]["1"]
    validation = list(task["closed_loop_split"]["validation_init_state_ids"])
    test = list(task["closed_loop_split"]["test_init_state_ids"])
    validation[0] = test[0]
    task["closed_loop_split"]["validation_init_state_ids"] = validation
    corrupted = tmp_path / "corrupted_split.json"
    corrupted.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        target.load_target_selection_split(corrupted)


def test_adapter_checkpoint_is_accepted_without_special_branch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _complete_c_checkpoint(
        run_dir,
        5000,
        adapter=True,
        job_name="smolvla_libero_c2_canonical_lora",
    )
    policy = run_dir / "checkpoints" / "005000" / "pretrained_model"

    resolved = target._validate_checkpoint_dir(policy, 5000)
    assert resolved == policy


def test_print_command_reports_frozen_workload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for step in target.FROZEN_CHECKPOINT_STEPS:
        _complete_c_checkpoint(run_dir, step)
    assets = _complete_assets(tmp_path / "assets")
    arguments = _arguments(
        tmp_path,
        run_dir=run_dir,
        libero_assets_dir=assets,
        print_command=True,
    )

    payload = target.select_target_checkpoint(arguments)

    assert payload["checkpoint_steps"] == list(target.FROZEN_CHECKPOINT_STEPS)
    assert payload["diagnostic"] is False
    assert payload["tasks"] == list(target.TARGET_TASK_IDS)
    assert payload["jobs"] == 18
    assert payload["validation_rollouts"] == 360
    assert payload["offline_validation_frames"] == target.EXPECTED_OFFLINE_FRAMES
    assert payload["open_loop_total_frames"] == target.EXPECTED_OFFLINE_FRAMES * 6
    assert payload["open_loop_parallelism"] == 1


def test_diagnostic_run_selection_is_marked_fail_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "training": {
                    "mode": "diagnostic",
                    "experiment_tag": "a_same_config",
                }
            }
        ),
        encoding="utf-8",
    )

    assert target._run_is_diagnostic(run_dir) is True
    assert target.target_selection_set_name(run_dir=run_dir) == "diagnostic_target_validation"
    assert (
        target.target_result_set_name(run_dir=run_dir) == "diagnostic_target_checkpoint_selection"
    )

    (run_dir / "run_config.json").write_text(
        json.dumps({"training": {"mode": "formal"}}),
        encoding="utf-8",
    )
    assert target._run_is_diagnostic(run_dir) is False
    assert target.target_selection_set_name(run_dir=run_dir) == "target_validation"


def _open_loop_namespace(
    tmp_path: Path,
    *,
    step: int = 5000,
    output: Path | None = None,
    extra: list[str] | None = None,
) -> argparse.Namespace:
    args = [
        "--open-loop-only",
        f"--run-dir={tmp_path / 'run'}",
        f"--checkpoint-step={step}",
        f"--split-file={SPLIT_FILE}",
        "--dataset-root=datasets/lerobot/lerobot/libero",
        f"--open-loop-output={output or tmp_path / 'open_loop_result.json'}",
    ]
    args.extend(extra or [])
    return target.build_argument_parser().parse_args(args)


def test_open_loop_only_requires_step_output_and_rejects_num_gpus(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        target.main(
            [
                "--open-loop-only",
                f"--run-dir={tmp_path / 'run'}",
                f"--split-file={SPLIT_FILE}",
                f"--open-loop-output={tmp_path / 'x.json'}",
            ]
        )
    with pytest.raises(SystemExit):
        target.main(
            [
                "--open-loop-only",
                f"--run-dir={tmp_path / 'run'}",
                "--checkpoint-step=5000",
                f"--split-file={SPLIT_FILE}",
            ]
        )
    with pytest.raises(SystemExit):
        target.main(
            [
                "--open-loop-only",
                f"--run-dir={tmp_path / 'run'}",
                "--checkpoint-step=5000",
                "--num-gpus=2",
                f"--split-file={SPLIT_FILE}",
                f"--open-loop-output={tmp_path / 'x.json'}",
            ]
        )


def test_open_loop_only_writes_only_its_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _complete_c_checkpoint(run_dir, 5000)
    output = tmp_path / "selection" / "jobs" / "open_loop" / "checkpoint_005000.json"
    monkeypatch.setattr(
        target,
        "evaluate_open_loop",
        lambda **kwargs: {
            "frame_count": 3,
            "macro_normalized_mse": 0.25,
            "macro_denormalized_mse": 1.5,
            "per_task": {"1": {"frame_count": 1}, "4": {"frame_count": 1}, "8": {"frame_count": 1}},
        },
    )

    target.run_open_loop_only(_open_loop_namespace(tmp_path, output=output))

    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["checkpoint_step"] == 5000
    assert payload["macro_normalized_mse"] == 0.25
    assert not (tmp_path / "selection" / "checkpoint_results.json").exists()
    assert not (tmp_path / "selection" / "selected_checkpoint.json").exists()
    assert not (run_dir / "checkpoints" / "selected").exists()

    with pytest.raises(FileExistsError, match="--overwrite"):
        target.run_open_loop_only(_open_loop_namespace(tmp_path, output=output))


def test_open_loop_child_command_construction(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, num_gpus=3, max_offline_frames_per_task=8)
    output = tmp_path / "result.json"

    command = target._open_loop_command(
        arguments=arguments,
        step=5000,
        output_path=output,
        overwrite=False,
    )

    assert "--open-loop-only" in command
    assert "--checkpoint-step=5000" in command
    assert f"--open-loop-output={output}" in command
    assert "--max-offline-frames-per-task=8" in command
    assert not any(argument.startswith("--num-gpus") for argument in command)


class _FakeOpenLoopProcess:
    instances: ClassVar[list[_FakeOpenLoopProcess]] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def __init__(self, command: list[str], env: dict, stdout: object, stderr: object):
        self.command = command
        self.terminated = False
        self.released = threading.Event()
        self.block_until_terminated = False
        self.exitcode = 0
        with self.lock:
            self.instances.append(self)
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)

    def wait(self) -> int:
        if self.block_until_terminated and self.exitcode == 0:
            self.released.wait(5)
        time.sleep(0.03)
        with self.lock:
            type(self).active -= 1
        if self.terminated:
            return 1
        output_value = next(
            argument for argument in self.command if argument.startswith("--open-loop-output=")
        ).split("=", 1)[1]
        step_value = next(
            argument for argument in self.command if argument.startswith("--checkpoint-step=")
        ).split("=", 1)[1]
        Path(output_value).parent.mkdir(parents=True, exist_ok=True)
        Path(output_value).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "checkpoint_step": int(step_value),
                    "checkpoint_id": "fake",
                    "split_sha256": "fake-split",
                    "max_frames_per_task": None,
                    "frame_count": 1,
                    "macro_normalized_mse": 0.1,
                    "macro_denormalized_mse": 0.2,
                    "per_task": {},
                }
            ),
            encoding="utf-8",
        )
        return self.exitcode

    def terminate(self) -> None:
        self.terminated = True
        self.released.set()


def _fake_popen_factory(
    failing_step: int | None = None,
    *,
    block_until_terminated: bool = False,
):
    def factory(command, env, stdout, stderr):
        step = next(
            argument for argument in command if argument.startswith("--checkpoint-step=")
        ).split("=", 1)[1]
        process = _FakeOpenLoopProcess(command, env, stdout, stderr)
        process.block_until_terminated = block_until_terminated
        if failing_step is not None and int(step) == failing_step:
            process.exitcode = 1
        return process

    return factory


def test_gpu_pool_bounds_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    checkpoints = {
        step: _complete_c_checkpoint(run_dir, step) for step in target.FROZEN_CHECKPOINT_STEPS
    }
    split = target.load_target_selection_split(SPLIT_FILE)
    arguments = _arguments(
        tmp_path,
        run_dir=run_dir,
        num_gpus=3,
        output_dir=tmp_path / "selection",
        max_offline_frames_per_task=None,
    )
    _FakeOpenLoopProcess.instances.clear()
    _FakeOpenLoopProcess.active = 0
    _FakeOpenLoopProcess.max_active = 0
    monkeypatch.setattr(target.subprocess, "Popen", _fake_popen_factory())

    results = target._run_open_loop_parallel(
        checkpoints=checkpoints,
        split=split,
        arguments=arguments,
        split_sha256="fake-split",
        progress_total=6,
    )

    assert len(results) == 6
    assert 0 < _FakeOpenLoopProcess.max_active <= 3


def test_open_loop_failure_terminates_others_and_points_to_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoints = {step: _complete_c_checkpoint(run_dir, step) for step in (5000, 10000, 15000)}
    split = target.load_target_selection_split(SPLIT_FILE)
    arguments = _arguments(
        tmp_path,
        run_dir=run_dir,
        num_gpus=3,
        output_dir=tmp_path / "selection",
        max_offline_frames_per_task=None,
    )
    _FakeOpenLoopProcess.instances.clear()
    monkeypatch.setattr(
        target.subprocess,
        "Popen",
        _fake_popen_factory(failing_step=15000, block_until_terminated=True),
    )

    with pytest.raises(RuntimeError, match="open_loop_checkpoint_015000.log"):
        target._run_open_loop_parallel(
            checkpoints=checkpoints,
            split=split,
            arguments=arguments,
            split_sha256="fake-split",
            progress_total=3,
        )
    assert any(process.terminated for process in _FakeOpenLoopProcess.instances)


def test_open_loop_resume_reuses_matching_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoints = {step: _complete_c_checkpoint(run_dir, step) for step in (5000,)}
    split = target.load_target_selection_split(SPLIT_FILE)
    checkpoint_id = target._checkpoint_weights_sha256(checkpoints[5000])
    matching = {
        "status": "complete",
        "checkpoint_id": checkpoint_id,
        "split_sha256": "fake-split",
        "max_frames_per_task": 8,
        "frame_count": 3,
        "macro_normalized_mse": 0.5,
        "macro_denormalized_mse": 1.0,
        "per_task": {},
    }
    result_path = target._open_loop_result_path(tmp_path / "selection", 5000)
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(matching), encoding="utf-8")
    arguments = _arguments(
        tmp_path,
        run_dir=run_dir,
        num_gpus=2,
        output_dir=tmp_path / "selection",
        max_offline_frames_per_task=8,
        resume=True,
    )
    monkeypatch.setattr(
        target.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("matching open-loop result must be reused"),
    )

    results = target._run_open_loop_parallel(
        checkpoints=checkpoints,
        split=split,
        arguments=arguments,
        split_sha256="fake-split",
        progress_total=6,
    )

    assert results[5000]["macro_normalized_mse"] == 0.5
    assert results[5000]["frame_count"] == 3
    assert not target._open_loop_result_reusable(
        dict(matching, max_frames_per_task=4),
        checkpoint_id=checkpoint_id,
        split_sha256="fake-split",
        max_frames_per_task=8,
    )


def test_ranking_and_audit_are_unchanged_for_same_results() -> None:
    rows = [
        {
            "checkpoint_step": 5000,
            "macro_success_rate": 0.5,
            "offline_normalized_action_mse": 0.2,
            "num_episodes": 60,
            "num_successes": 30,
        },
        {
            "checkpoint_step": 10000,
            "macro_success_rate": 0.5,
            "offline_normalized_action_mse": 0.1,
            "num_episodes": 60,
            "num_successes": 30,
        },
    ]
    ranked = sorted(rows, key=target.checkpoint_rank_key)
    audit = target.build_ranking_audit(ranked)

    assert [row["checkpoint_step"] for row in ranked] == [10000, 5000]
    assert [entry.get("tie_breaker_level") for entry in audit] == [None, 2]
    assert audit == target.build_ranking_audit(sorted(rows, key=target.checkpoint_rank_key))


def test_open_loop_is_deterministic() -> None:
    if not SMOKE_CHECKPOINT.is_dir() or not DATASET_ROOT.is_dir():
        pytest.skip("C1 smoke checkpoint or dataset is not available")
    split = target.load_target_selection_split(SPLIT_FILE)

    first = target.evaluate_open_loop(
        checkpoint_path=SMOKE_CHECKPOINT,
        split=split,
        dataset_root=DATASET_ROOT,
        max_frames_per_task=2,
    )
    second = target.evaluate_open_loop(
        checkpoint_path=SMOKE_CHECKPOINT,
        split=split,
        dataset_root=DATASET_ROOT,
        max_frames_per_task=2,
    )

    assert first == second
    assert first["frame_count"] > 0


def test_publish_requires_sha_match_and_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source = _complete_c_checkpoint(run_dir, 25000)
    selection_file = tmp_path / "selected_checkpoint.json"
    selection_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "selection_set": "target_validation",
                "selected_checkpoint_step": 25000,
                "source_checkpoint": str(source.parent),
            }
        ),
        encoding="utf-8",
    )
    selected_dir = run_dir / "checkpoints" / "selected"

    target.publish_selected_checkpoint(
        selection_file=selection_file,
        selected_dir=selected_dir,
        overwrite=False,
        selection_set="target_validation",
    )
    assert (selected_dir / "pretrained_model" / "model.safetensors").is_file()

    shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="--overwrite"):
        target.publish_selected_checkpoint(
            selection_file=selection_file,
            selected_dir=selected_dir,
            overwrite=False,
            selection_set="target_validation",
        )

    shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True)
    wrong = selected_dir / "pretrained_model" / "model.safetensors"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("different weights", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        target.publish_selected_checkpoint(
            selection_file=selection_file,
            selected_dir=selected_dir,
            overwrite=False,
            selection_set="target_validation",
        )


def test_publish_only_cli_copies_adapter_and_marks_manifest_complete(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    source = _complete_c_checkpoint(run_dir, 25000, adapter=True)
    output_dir = run_dir / "checkpoint_selection"
    output_dir.mkdir(parents=True)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps({"status": "running", "selected_checkpoint_step": 0}),
        encoding="utf-8",
    )
    selection_file = tmp_path / "selected_checkpoint.json"
    selection_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "selection_set": "target_validation",
                "selected_checkpoint_step": 25000,
                "source_checkpoint": str(source.parent),
            }
        ),
        encoding="utf-8",
    )
    selected_dir = run_dir / "checkpoints" / "selected"

    target.main(
        [
            "--publish-only",
            f"--run-dir={run_dir}",
            f"--selection-file={selection_file}",
            f"--output-dir={output_dir}",
        ]
    )

    assert (selected_dir / "pretrained_model" / "adapter_model.safetensors").is_file()
    manifest = json.loads((output_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["selected_checkpoint_step"] == 25000
