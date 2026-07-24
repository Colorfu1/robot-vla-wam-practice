from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import robot_practice.evaluation.common as common
from robot_practice.evaluation.common import (
    ActionPlanTracker,
    INCOMPLETE_MARKER_TEXT,
    ModelQueryTimer,
    CheckpointManifest,
    LoadedACT,
    build_run_config,
    create_output_directory,
    mark_run_complete,
    release_output_directory_lock,
    validate_act_libero_contract,
    validate_complete_checkpoint,
    write_json_atomic,
)


def _checkpoint(
    tmp_path: Path,
    *,
    include_state: bool = True,
    checkpoint_type: str = "act",
) -> Path:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"type": checkpoint_type}))
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    processor = {
        "name": "test",
        "steps": [{"registry_name": "normalizer_processor", "state_file": "stats.safetensors"}],
    }
    (checkpoint / "policy_preprocessor.json").write_text(json.dumps(processor))
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps({"name": "post", "steps": []})
    )
    if include_state:
        (checkpoint / "stats.safetensors").write_bytes(b"stats")
    return checkpoint


def test_launch_rerun_viewer_prefers_project_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "rollout.rrd"
    recording.write_bytes(b"rrd")
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(
        common.shutil,
        "which",
        lambda name: {
            "robot-rerun": "/env/bin/robot-rerun",
            "rerun": "/env/bin/rerun",
        }.get(name),
    )
    monkeypatch.setattr(
        common.subprocess,
        "Popen",
        lambda argv, **kwargs: calls.append((argv, kwargs)),
    )

    assert common.launch_rerun_viewer(recording, enabled=True) is True
    assert calls[0][0] == ["/env/bin/robot-rerun", str(recording)]


def test_launch_rerun_viewer_falls_back_to_official_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "prediction.rrd"
    recording.write_bytes(b"rrd")
    calls: list[list[str]] = []

    monkeypatch.setattr(
        common.shutil,
        "which",
        lambda name: "/env/bin/rerun" if name == "rerun" else None,
    )
    monkeypatch.setattr(
        common.subprocess,
        "Popen",
        lambda argv, **kwargs: calls.append(argv),
    )

    assert common.launch_rerun_viewer(recording, enabled=True) is True
    assert calls == [["/env/bin/rerun", str(recording)]]


def test_complete_checkpoint_requires_referenced_processor_state(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, include_state=False)
    with pytest.raises(FileNotFoundError, match="Processor state file"):
        validate_complete_checkpoint(checkpoint)


def test_complete_checkpoint_returns_manifest(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    manifest = validate_complete_checkpoint(checkpoint)
    assert manifest.path == checkpoint.resolve()
    assert manifest.policy_config == {"type": "act"}
    assert manifest.processor_state_files == ((checkpoint / "stats.safetensors").resolve(),)


def test_checkpoint_path_rejects_single_weight_file(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    with pytest.raises(ValueError, match="complete pretrained_model directory"):
        validate_complete_checkpoint(weights)


def test_output_derives_from_training_run_and_uses_marker(tmp_path: Path) -> None:
    policy_path = tmp_path / "run" / "checkpoints" / "020000" / "pretrained_model"
    policy_path.mkdir(parents=True)
    output = create_output_directory(
        policy_path=policy_path,
        mode="offline",
        output_dir=None,
        suite="libero_object",
        task_id=8,
    )
    assert output.parent == tmp_path / "run" / "eval" / "offline" / "libero_object" / "task_008"
    assert (output / ".incomplete").is_file()
    mark_run_complete(output)
    assert not (output / ".incomplete").exists()


def test_output_override_must_not_exist(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        create_output_directory(
            policy_path=tmp_path,
            mode="libero",
            output_dir=output,
        )


def _run_config(*, mode: str = "offline", policy_type: str = "act") -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "mode": mode,
        "created_at": "2026-07-22T12:00:00+08:00",
        "policy": {"type": policy_type},
        "runtime": {},
        "input": {},
        "arguments": {},
    }


def test_output_overwrite_replaces_recognized_result(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "run_config.json").write_text(json.dumps(_run_config()))
    (output / "stale.txt").write_text("old")
    result = create_output_directory(
        policy_path=tmp_path,
        mode="offline",
        output_dir=output,
        overwrite=True,
    )
    try:
        assert result == output.resolve()
        assert (result / ".incomplete").read_text() == INCOMPLETE_MARKER_TEXT
        assert not (result / "stale.txt").exists()
    finally:
        release_output_directory_lock(result)


def test_output_overwrite_creates_missing_and_replaces_empty_target(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    created = create_output_directory(
        policy_path=tmp_path,
        mode="offline",
        output_dir=missing,
        overwrite=True,
    )
    release_output_directory_lock(created)
    assert (created / ".incomplete").is_file()

    empty = tmp_path / "empty"
    empty.mkdir()
    replaced = create_output_directory(
        policy_path=tmp_path,
        mode="offline",
        output_dir=empty,
        overwrite=True,
    )
    release_output_directory_lock(replaced)
    assert (replaced / ".incomplete").is_file()


def test_output_overwrite_requires_explicit_output_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an explicit --output-dir"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=None,
            overwrite=True,
        )


@pytest.mark.parametrize("kind", ["file", "unrelated_directory"])
def test_output_overwrite_rejects_unrecognized_targets(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "result"
    if kind == "file":
        output.write_text("not a directory")
    else:
        output.mkdir()
        (output / "notes.txt").write_text("not evaluator output")
    with pytest.raises(ValueError, match="Overwrite target|not a recognized"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=output,
            overwrite=True,
        )


@pytest.mark.parametrize("dangling", [False, True])
def test_output_overwrite_rejects_output_symlink(tmp_path: Path, dangling: bool) -> None:
    target = tmp_path / "target"
    if not dangling:
        target.mkdir()
    output = tmp_path / "result"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=output,
            overwrite=True,
        )


@pytest.mark.parametrize("sentinel", [".incomplete", "run_config.json"])
@pytest.mark.parametrize("dangling", [False, True])
def test_output_overwrite_rejects_sentinel_symlink(
    tmp_path: Path,
    sentinel: str,
    dangling: bool,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    target = tmp_path / "sentinel-target"
    if not dangling:
        if sentinel == ".incomplete":
            target.write_text(INCOMPLETE_MARKER_TEXT)
        else:
            target.write_text(json.dumps(_run_config()))
    (output / sentinel).symlink_to(target)
    with pytest.raises(ValueError, match="sentinel must not be a symlink"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=output,
            overwrite=True,
        )


@pytest.mark.parametrize(
    ("contents", "mode"),
    [
        ("not json", "offline"),
        (json.dumps({"schema_version": 1, "mode": "offline"}), "offline"),
        (json.dumps(_run_config(mode="libero")), "offline"),
    ],
)
def test_output_overwrite_rejects_invalid_or_mismatched_run_config(
    tmp_path: Path,
    contents: str,
    mode: str,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "run_config.json").write_text(contents)
    with pytest.raises(ValueError, match="Invalid|not a recognized"):
        create_output_directory(
            policy_path=tmp_path,
            mode=mode,
            output_dir=output,
            overwrite=True,
        )


def test_output_overwrite_rejects_invalid_marker_even_with_run_config(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / ".incomplete").write_text("not an evaluator marker\n")
    (output / "run_config.json").write_text(json.dumps(_run_config()))
    with pytest.raises(ValueError, match="Invalid evaluator incomplete marker"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=output,
            overwrite=True,
        )


def test_output_overwrite_rejects_invalid_run_config_even_with_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / ".incomplete").write_text(INCOMPLETE_MARKER_TEXT)
    (output / "run_config.json").write_text("not json")
    with pytest.raises(ValueError, match="Invalid or mismatched evaluator run config"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=output,
            overwrite=True,
        )


def test_complete_checkpoint_accepts_act_delta_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, checkpoint_type="act_delta")
    manifest = validate_complete_checkpoint(checkpoint)
    assert manifest.policy_config == {"type": "act_delta"}


def test_build_run_config_preserves_act_delta_type(tmp_path: Path) -> None:
    manifest = CheckpointManifest(
        path=_checkpoint(tmp_path, checkpoint_type="act_delta"),
        policy_config={"type": "act_delta"},
        train_config=None,
        processor_state_files=(),
    )
    loaded = LoadedACT(
        manifest=manifest,
        config=SimpleNamespace(device="cpu"),
        policy=object(),
        preprocessor=None,
        postprocessor=None,
        checkpoint_n_action_steps=5,
        checkpoint_temporal_ensemble_coeff=None,
    )
    run_config = build_run_config(
        status="complete",
        mode="offline",
        loaded=loaded,
        input_config={},
        arguments={},
    )
    assert run_config["policy"]["type"] == "act_delta"


def test_output_lock_blocks_active_overwrite_and_releases_on_completion(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    active = create_output_directory(
        policy_path=tmp_path,
        mode="offline",
        output_dir=output,
    )
    with pytest.raises(RuntimeError, match="active evaluator"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=output,
            overwrite=True,
        )
    mark_run_complete(active)

    replacement = create_output_directory(
        policy_path=tmp_path,
        mode="offline",
        output_dir=output,
        overwrite=True,
    )
    release_output_directory_lock(replacement)


def test_output_lock_can_be_released_after_failure(tmp_path: Path) -> None:
    output = create_output_directory(
        policy_path=tmp_path,
        mode="offline",
        output_dir=tmp_path / "result",
    )
    release_output_directory_lock(output)
    replacement = create_output_directory(
        policy_path=tmp_path,
        mode="offline",
        output_dir=output,
        overwrite=True,
    )
    release_output_directory_lock(replacement)


def test_atomic_json_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NaN or Infinity"):
        write_json_atomic(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_output_task_identity_must_be_complete(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both be supplied"):
        create_output_directory(
            policy_path=tmp_path,
            mode="offline",
            output_dir=None,
            suite="libero_object",
        )


def test_libero_contract_requires_exact_input_features() -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig(
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        pretrained_backbone_weights=None,
    )
    validate_act_libero_contract(config)
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (7,))
    with pytest.raises(ValueError, match="observation.state.*shape"):
        validate_act_libero_contract(config)


def test_act_execution_config_accepts_single_step_temporal_ensemble() -> None:
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig(
        chunk_size=20,
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
        pretrained_backbone_weights=None,
    )
    common.validate_act_execution_config(config)


@pytest.mark.parametrize(
    ("n_action_steps", "coefficient", "message"),
    [
        (5, 0.01, "must equal 1"),
        (1, float("nan"), "finite"),
        (1, -100.0, "non-finite float32"),
    ],
)
def test_act_execution_config_rejects_invalid_temporal_ensemble(
    n_action_steps: int,
    coefficient: float,
    message: str,
) -> None:
    from lerobot.policies.act.configuration_act import ACTConfig

    config = ACTConfig(
        chunk_size=20,
        n_action_steps=1,
        temporal_ensemble_coeff=None,
        pretrained_backbone_weights=None,
    )
    config.n_action_steps = n_action_steps
    config.temporal_ensemble_coeff = coefficient
    with pytest.raises(ValueError, match=message):
        common.validate_act_execution_config(config)



class FakeACT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.queries = 0

    def predict_action_chunk(self, batch: torch.Tensor) -> torch.Tensor:
        self.queries += 1
        return batch + 1

    def select_action(self, batch: torch.Tensor, *, query: bool) -> torch.Tensor:
        if query:
            return self.predict_action_chunk(batch)
        return batch


def test_model_query_timer_distinguishes_query_from_queue_hit() -> None:
    policy = FakeACT()
    original = policy.predict_action_chunk
    with ModelQueryTimer(policy, "cpu") as timer:
        timer.begin_step()
        policy.select_action(torch.tensor([1.0]), query=False)
        assert timer.current_latency_ms is None

        timer.begin_step()
        output = policy.select_action(torch.tensor([1.0]), query=True)
        assert output.item() == 2.0
        assert timer.current_latency_ms is not None
        assert timer.current_latency_ms >= 0
        torch.testing.assert_close(timer.current_chunk_policy, torch.tensor([2.0]))

    assert policy.predict_action_chunk.__func__ is original.__func__


def _policy_chunk(length: int = 20) -> torch.Tensor:
    return torch.arange(length * 7, dtype=torch.float32).reshape(1, length, 7)


def test_action_plan_tracker_partitions_queue_and_discarded_tail() -> None:
    tracker = ActionPlanTracker(n_action_steps=5, temporal_ensemble_coeff=None)
    chunk = _policy_chunk()

    first = tracker.record_step(
        selected_action_policy=chunk[:, 0],
        queried_chunk_policy=chunk,
    )
    assert first.model_queried
    assert first.plan_step_index == 0
    torch.testing.assert_close(first.queued_actions_policy, chunk[0, 1:5])
    torch.testing.assert_close(first.discarded_actions_policy, chunk[0, 5:])
    assert first.new_prediction_chunk_policy is None

    second = tracker.record_step(
        selected_action_policy=chunk[:, 1],
        queried_chunk_policy=None,
    )
    assert not second.model_queried
    assert second.plan_step_index == 1
    torch.testing.assert_close(second.queued_actions_policy, chunk[0, 2:5])
    torch.testing.assert_close(second.discarded_actions_policy, chunk[0, 5:])

    for index in range(2, 5):
        tracker.record_step(
            selected_action_policy=chunk[:, index],
            queried_chunk_policy=None,
        )
    replacement = chunk + 1000
    replanned = tracker.record_step(
        selected_action_policy=replacement[:, 0],
        queried_chunk_policy=replacement,
    )
    assert replanned.model_queried
    assert replanned.plan_step_index == 0


def test_action_plan_tracker_reset_requires_a_new_query() -> None:
    tracker = ActionPlanTracker(n_action_steps=5, temporal_ensemble_coeff=None)
    chunk = _policy_chunk()
    tracker.record_step(
        selected_action_policy=chunk[:, 0],
        queried_chunk_policy=chunk,
    )
    tracker.reset_episode()

    with pytest.raises(RuntimeError, match="without an active"):
        tracker.record_step(
            selected_action_policy=chunk[:, 1],
            queried_chunk_policy=None,
        )


def test_action_plan_tracker_rejects_policy_queue_mismatch() -> None:
    tracker = ActionPlanTracker(n_action_steps=5, temporal_ensemble_coeff=None)
    chunk = _policy_chunk()

    with pytest.raises(RuntimeError, match="does not match"):
        tracker.record_step(
            selected_action_policy=chunk[:, 0] + 1,
            queried_chunk_policy=chunk,
        )


def test_action_plan_tracker_temporal_ensemble_uses_new_chunk_semantics() -> None:
    tracker = ActionPlanTracker(n_action_steps=1, temporal_ensemble_coeff=0.01)
    chunk = _policy_chunk()
    selected = torch.full((1, 7), 42.0)

    step = tracker.record_step(
        selected_action_policy=selected,
        queried_chunk_policy=chunk,
    )

    assert step.model_queried
    assert step.plan_step_index is None
    torch.testing.assert_close(step.new_prediction_chunk_policy, chunk)
    assert step.queued_actions_policy.shape == (0, 7)
    assert step.discarded_actions_policy.shape == (0, 7)


def test_action_plan_tracker_temporal_ensemble_requires_query_every_step() -> None:
    tracker = ActionPlanTracker(n_action_steps=1, temporal_ensemble_coeff=0.01)

    with pytest.raises(RuntimeError, match="every control step"):
        tracker.record_step(
            selected_action_policy=torch.zeros((1, 7)),
            queried_chunk_policy=None,
        )
