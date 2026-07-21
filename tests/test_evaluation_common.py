from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from robot_practice.evaluation.common import (
    ModelQueryTimer,
    create_output_directory,
    mark_run_complete,
    validate_complete_checkpoint,
    write_json_atomic,
)


def _checkpoint(tmp_path: Path, *, include_state: bool = True) -> Path:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"type": "act"}))
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
    )
    assert output.parent == tmp_path / "run" / "eval" / "offline"
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


def test_atomic_json_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NaN or Infinity"):
        write_json_atomic(tmp_path / "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()


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

    assert policy.predict_action_chunk.__func__ is original.__func__
