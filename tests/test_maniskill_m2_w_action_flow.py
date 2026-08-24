from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_practice.experiments.maniskill import eval_stackcube_action_flow as evaluator
from robot_practice.experiments.maniskill import m2_w_action_flow as runner
from robot_practice.policies.wam_tiny.action_flow import (
    BASE_IMAGE_KEY,
    HAND_IMAGE_KEY,
    STATE_KEY,
    ActionOnlyFlowModel,
    ActionOnlyFlowPolicy,
    NormalizationStats,
    derive_flow_seed,
    load_action_flow_checkpoint,
    save_action_flow_checkpoint,
)

CONFIG = Path("configs/experiments/maniskill/m2_w_action_only_flow.json")


def test_m2_config_and_real_m1_inputs_pass_the_contract() -> None:
    result = runner.validate_config(CONFIG)
    assert result["task"] == "StackCube-v1"
    assert result["global_batch_size"] == 256
    assert result["test_access_allowed"] is False
    assert len(result["m1_frozen_protocol_sha256"]) == 64


def test_7d_action_normalization_profile_is_explicit_and_train_only() -> None:
    config = runner.load_config(CONFIG)
    normalization = config["normalization"]
    assert normalization["profile"] == "7d_action_per_channel_zscore"
    assert normalization["source_split"] == "train"
    assert normalization["action"]["method"] == "mean_std"
    assert normalization["action"]["per_channel"] is True
    assert normalization["action"]["clip_before_stats"] is False
    assert len(normalization["action"]["channel_names"]) == 7


def test_7d_action_normalization_rejects_global_or_leaky_stats() -> None:
    config = runner.load_config(CONFIG)
    config["normalization"]["source_split"] = "development"
    with pytest.raises(ValueError, match="normalization.*train"):
        runner.validate_config(config, check_inputs=False)

    config = runner.load_config(CONFIG)
    config["normalization"]["action"]["per_channel"] = False
    with pytest.raises(ValueError, match="per-channel"):
        runner.validate_config(config, check_inputs=False)


def test_training_checkpoint_eval_contract_is_explicit() -> None:
    config = runner.load_config(CONFIG)
    assert config["training"]["eval_during_training"] is True
    assert config["training"]["eval_num_envs"] == 64
    assert config["evaluation"]["n_action_steps"] == [4, 8]


def test_eval_maps_local_cuda_devices_to_physical_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("M2_W_PHYSICAL_GPU", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,7")
    assert evaluator._physical_gpu_label(torch.device("cuda:0")) == "5"
    assert evaluator._physical_gpu_label(torch.device("cuda:1")) == "7"


def test_training_eval_uses_in_process_policy_without_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModel:
        def eval(self) -> FakeModel:
            return self

    class FakePolicy:
        n_action_steps = 4

        def __init__(self) -> None:
            self.model = FakeModel()
            self.used = False
            self._chunk_index = 0

        @property
        def chunk_index(self) -> int:
            return self._chunk_index

        def reset(self) -> None:
            self._chunk_index = 0

        def select_action(self, observations: object, *, fingerprints: list[str]) -> torch.Tensor:
            del observations, fingerprints
            self.used = True
            self._chunk_index += 1
            return torch.zeros((1, 7), dtype=torch.float32)

    observation = {"observation": "fake"}
    signals = {
        "is_cubeA_grasped": np.array([False]),
        "is_cubeA_on_cubeB": np.array([False]),
        "is_cubeA_static": np.array([False]),
        "success": np.array([True]),
        "fail": np.array([False]),
    }
    policy = FakePolicy()
    monkeypatch.setattr(
        evaluator,
        "load_state_bank",
        lambda path, max_episodes=None: [
            {"index": 0, "start_state_fingerprint": "fake-fingerprint"}
        ],
    )
    monkeypatch.setattr(evaluator, "_stack_context", lambda history: {})
    monkeypatch.setattr(evaluator, "_reset_exact_state", lambda env, rows, device: observation)
    monkeypatch.setattr(evaluator, "_signals", lambda info, batch_size: signals)

    class FakeEnv:
        unwrapped = None

        def __init__(self) -> None:
            self.unwrapped = self
            self.device = "cpu"

        def get_info(self) -> dict[str, object]:
            return {}

        def step(self, action: np.ndarray) -> tuple[dict[str, str], None, np.ndarray, np.ndarray, dict]:
            del action
            return observation, None, np.array([True]), np.array([False]), {}

        def close(self) -> None:
            return None

    monkeypatch.setattr(evaluator, "_make_env", lambda *args, **kwargs: FakeEnv())
    payload = evaluator.run_development_rollout(
        state_bank=tmp_path / "development.jsonl",
        output_dir=tmp_path / "rollout",
        n_action_steps=4,
        checkpoint=tmp_path / "checkpoint_step_00000007.pt",
        policy=policy,
        checkpoint_step=7,
        max_episodes=1,
        max_steps=1,
        num_envs=1,
        device="cpu",
        smoke=True,
    )
    assert policy.used is True
    assert payload["summary"]["checkpoint_step"] == 7
    assert payload["summary"]["n_action_steps"] == 4
    assert payload["worker"]["policy_device"] == "cpu"


def test_config_rejects_global_batch_drift() -> None:
    config = runner.load_config(CONFIG)
    config["training"]["per_process_batch_size"] = 31
    with pytest.raises(ValueError, match="global batch"):
        runner.validate_config(config, check_inputs=False)


def test_evaluation_worker_count_need_not_divide_training_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.load_config(CONFIG)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 7)
    result = runner.validate_config(
        config,
        num_gpus=7,
        require_batch_divisor=False,
        check_inputs=False,
    )
    assert result["num_gpus"] == 7
    with pytest.raises(ValueError, match="positive divisor"):
        runner.validate_config(config, num_gpus=7, check_inputs=False)


def test_config_rejects_privileged_observation_key() -> None:
    config = runner.load_config(CONFIG)
    config["protocol"]["observation_keys"] = ["observation.state", "object_pose"]
    with pytest.raises(ValueError, match="privileged"):
        runner.validate_config(config, check_inputs=False)


def test_action_flow_model_shape_and_flow_path() -> None:
    model = ActionOnlyFlowModel()
    observations = {
        BASE_IMAGE_KEY: torch.zeros((1, 2, 3, 128, 128)),
        HAND_IMAGE_KEY: torch.zeros((1, 2, 3, 128, 128)),
        STATE_KEY: torch.zeros((1, 2, 18)),
    }
    actions = torch.zeros((1, 16, 7))
    time = torch.full((1,), 0.5)
    with torch.inference_mode():
        prediction = model(observations, actions, time)
        loss = model.flow_loss(observations, actions, time=time)
    assert tuple(prediction.shape) == (1, 16, 7)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_flow_seed_and_action_queue_are_deterministic() -> None:
    assert derive_flow_seed("fingerprint", 0) == derive_flow_seed("fingerprint", 0)
    assert derive_flow_seed("fingerprint", 0) != derive_flow_seed("fingerprint", 1)
    stats = NormalizationStats(
        image_mean=torch.zeros(3),
        image_std=torch.ones(3),
        state_mean=torch.zeros(18),
        state_std=torch.ones(18),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )
    model = ActionOnlyFlowModel().eval()
    policy = ActionOnlyFlowPolicy(model, stats, n_action_steps=4)
    observations = {
        BASE_IMAGE_KEY: torch.zeros((2, 2, 3, 128, 128)),
        HAND_IMAGE_KEY: torch.zeros((2, 2, 3, 128, 128)),
        STATE_KEY: torch.zeros((2, 2, 18)),
    }
    with torch.inference_mode():
        first = policy.predict_action_chunk(observations, fingerprints=["a", "b"])
        second = policy.predict_action_chunk(observations, fingerprints=["a", "b"])
    assert torch.equal(first, second)
    assert tuple(first.shape) == (2, 16, 7)


def test_normalization_stats_apply_the_frozen_std_floor() -> None:
    stats = NormalizationStats(
        image_mean=torch.zeros(3),
        image_std=torch.zeros(3),
        state_mean=torch.zeros(18),
        state_std=torch.zeros(18),
        action_mean=torch.zeros(7),
        action_std=torch.zeros(7),
    )
    assert torch.all(stats.image_std >= 1e-6)
    assert torch.all(stats.state_std >= 1e-6)
    assert torch.all(stats.action_std >= 1e-6)


def test_checkpoint_reload_is_regular_file_and_preserves_contract(tmp_path: Path) -> None:
    model = ActionOnlyFlowModel().eval()
    stats = NormalizationStats(
        image_mean=torch.zeros(3),
        image_std=torch.ones(3),
        state_mean=torch.zeros(18),
        state_std=torch.ones(18),
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )
    checkpoint = tmp_path / "checkpoint_step_00000020.pt"
    save_action_flow_checkpoint(checkpoint, model=model, stats=stats, step=20)
    loaded, payload = load_action_flow_checkpoint(checkpoint, n_action_steps=8)
    assert payload["step"] == 20
    assert loaded.n_action_steps == 8
    assert checkpoint.is_file() and not checkpoint.is_symlink()


def test_smoke_command_parser_uses_fixed_contract() -> None:
    parsed = runner.build_parser().parse_args(
        [
            "smoke",
            "--config",
            str(CONFIG),
            "--output-root",
            "outputs/maniskill/m2_task_base/StackCube-v1/smoke",
            "--num-gpus",
            "1",
        ]
    )
    assert parsed.command == "smoke"
    assert parsed.num_gpus == 1


def test_train_command_parser_accepts_explicit_overwrite() -> None:
    parsed = runner.build_parser().parse_args(
        [
            "train",
            "--config",
            str(CONFIG),
            "--output-root",
            "outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000",
            "--num-gpus",
            "8",
            "--overwrite",
        ]
    )
    assert parsed.overwrite is True


def test_training_log_helpers_flush_text_and_jsonl_records(tmp_path: Path) -> None:
    text_path = tmp_path / "training.log"
    jsonl_path = tmp_path / "loss_history.jsonl"
    with (
        text_path.open("w", encoding="utf-8") as text_stream,
        jsonl_path.open("w", encoding="utf-8") as jsonl_stream,
    ):
        runner._write_training_log_line(text_stream, "[M2-W] step=1 loss=1.0")
        runner._write_jsonl_record(
            jsonl_stream,
            {"event": "step", "step": 1, "loss": 1.0, "grad_norm": 2.0},
        )
        assert text_stream.tell() > 0
        assert jsonl_stream.tell() > 0

    assert text_path.read_text(encoding="utf-8") == "[M2-W] step=1 loss=1.0\n"
    assert json.loads(jsonl_path.read_text(encoding="utf-8")) == {
        "event": "step",
        "grad_norm": 2.0,
        "loss": 1.0,
        "step": 1,
    }


def test_rollout_gpu_assignment_respects_parent_cuda_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6,7")
    assert runner._visible_gpu_ids(2) == ["4", "6"]
    with pytest.raises(RuntimeError, match="only 3 are visible"):
        runner._visible_gpu_ids(4)


def test_prepare_output_dir_overwrite_replaces_only_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    practice = tmp_path / "practice"
    output_root = practice / "outputs"
    target = output_root / "m2" / "run"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(runner, "practice_root", lambda: practice)

    with pytest.raises(FileExistsError, match="Refusing existing"):
        runner._prepare_output_dir(target)

    prepared = runner._prepare_output_dir(target, overwrite=True)
    assert prepared == target
    assert prepared.is_dir()
    assert not (prepared / "stale.txt").exists()


def test_prepare_output_dir_overwrite_rejects_non_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    practice = tmp_path / "practice"
    target = tmp_path / "unrelated" / "run"
    target.mkdir(parents=True)
    monkeypatch.setattr(runner, "practice_root", lambda: practice)

    with pytest.raises(ValueError, match="restricted to an output directory"):
        runner._prepare_output_dir(target, overwrite=True)


def test_formal_train_rejects_runtime_world_size_drift() -> None:
    with pytest.raises(ValueError, match="frozen training world_size"):
        runner.train(CONFIG, "outputs/unused-m2-w", num_gpus=1)
