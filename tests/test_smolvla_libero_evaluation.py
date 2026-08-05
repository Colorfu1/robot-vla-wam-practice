from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import robot_practice.evaluation.libero_smolvla as smol

PRACTICE_ROOT = Path(__file__).resolve().parents[1]


class FakeSmolPolicy:
    def __init__(self, chunks: list[torch.Tensor]) -> None:
        self.chunks = list(chunks)
        self.query_count = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        del batch
        self.query_count += 1
        return self.chunks.pop(0)


def _loaded(
    policy: FakeSmolPolicy,
    *,
    n_action_steps: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        config=SimpleNamespace(
            n_action_steps=n_action_steps,
            device="cpu",
            use_amp=False,
        ),
        postprocessor=lambda actions: actions,
    )


def _env_postprocessor(transition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return transition


def test_smolvla_driver_queries_once_and_executes_the_same_chunk() -> None:
    first = torch.arange(28, dtype=torch.float32).reshape(1, 4, 7)
    second = first + 100
    policy = FakeSmolPolicy([first, second])
    driver = smol.SmolVLALiberoPolicyDriver(
        _loaded(policy, n_action_steps=2),
        _env_postprocessor,
    )

    driver.reset_episode()
    step0 = driver.select_step({})
    step1 = driver.select_step({})
    step2 = driver.select_step({})

    assert policy.reset_count == 1
    assert policy.query_count == 2
    np.testing.assert_array_equal(step0.queried_chunk, first.numpy()[0])
    np.testing.assert_array_equal(step0.executed_action, first.numpy()[0, 0])
    np.testing.assert_array_equal(step0.queued_actions, first.numpy()[0, 1:2])
    np.testing.assert_array_equal(step0.discarded_actions, first.numpy()[0, 2:])
    assert step0.model_queried is True
    assert step0.plan_step_index == 0
    assert step0.model_query_latency_ms is not None

    assert step1.queried_chunk is None
    np.testing.assert_array_equal(step1.executed_action, first.numpy()[0, 1])
    assert step1.queued_actions.shape == (0, 7)
    np.testing.assert_array_equal(step1.discarded_actions, first.numpy()[0, 2:])
    assert step1.model_queried is False
    assert step1.plan_step_index == 1
    assert step1.model_query_latency_ms is None

    np.testing.assert_array_equal(step2.queried_chunk, second.numpy()[0])
    np.testing.assert_array_equal(step2.executed_action, second.numpy()[0, 0])


def test_smolvla_driver_n_action_steps_one_discards_the_tail() -> None:
    chunk = torch.arange(21, dtype=torch.float32).reshape(1, 3, 7)
    policy = FakeSmolPolicy([chunk])
    driver = smol.SmolVLALiberoPolicyDriver(
        _loaded(policy, n_action_steps=1),
        _env_postprocessor,
    )

    driver.reset_episode()
    step = driver.select_step({})

    np.testing.assert_array_equal(step.executed_action, chunk.numpy()[0, 0])
    assert step.queued_actions.shape == (0, 7)
    np.testing.assert_array_equal(step.discarded_actions, chunk.numpy()[0, 1:])
    assert policy.query_count == 1


def test_smolvla_driver_rejects_nonfinite_chunks() -> None:
    chunk = torch.zeros((1, 2, 7))
    chunk[0, 0, 0] = torch.nan
    driver = smol.SmolVLALiberoPolicyDriver(
        _loaded(FakeSmolPolicy([chunk]), n_action_steps=1),
        _env_postprocessor,
    )
    driver.reset_episode()

    with pytest.raises(ValueError, match="finite shape"):
        driver.select_step({})


def test_policy_reference_distinguishes_existing_local_directory(
    tmp_path: Path,
) -> None:
    reference, local = smol._resolved_policy_reference(str(tmp_path))
    assert reference == str(tmp_path.resolve())
    assert local == tmp_path.resolve()

    reference, local = smol._resolved_policy_reference("HuggingFaceVLA/smolvla_libero")
    assert reference == "HuggingFaceVLA/smolvla_libero"
    assert local is None


def test_policy_reference_keeps_repository_checkpoint_relative() -> None:
    reference, local = smol._resolved_policy_reference("tests")

    assert reference == "tests"
    assert local == (PRACTICE_ROOT / "tests").resolve()


def test_local_manifest_accepts_ordinary_and_peft_checkpoints(
    tmp_path: Path,
) -> None:
    for name in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        payload = {"type": "smolvla"} if name == "config.json" else {"steps": []}
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    ordinary = smol._local_manifest(tmp_path)
    assert ordinary.policy_config["type"] == "smolvla"

    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    peft = smol._local_manifest(tmp_path)
    assert peft.policy_config["type"] == "smolvla"


def test_checkpoint_rename_map_recovers_saved_camera_mapping(tmp_path: Path) -> None:
    processor = {
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {
                    "rename_map": {
                        "observation.images.image": "observation.images.camera1",
                        "observation.images.image2": "observation.images.camera2",
                    }
                },
            },
            {"registry_name": "to_batch_processor", "config": {}},
        ]
    }
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(processor),
        encoding="utf-8",
    )

    assert smol._checkpoint_rename_map(tmp_path) == processor["steps"][0]["config"]["rename_map"]
    assert smol._checkpoint_rename_map(None) is None


def test_local_load_rebases_vlm_and_tokenizer_paths_without_rewriting_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    assets = (tmp_path / "local_assets").resolve()
    config = SmolVLAConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={
            "action": PolicyFeature(FeatureType.ACTION, (7,)),
        },
        device="cpu",
        use_amp=False,
        chunk_size=50,
        n_action_steps=10,
    )
    config.vlm_model_name = "/remote/practice/policies/smolvla_or_pi0/smolvla_base/vlm_assets"
    captured: dict[str, object] = {}
    manifest = SimpleNamespace(path=Path("relative/checkpoint"))

    monkeypatch.setattr(
        smol.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        smol,
        "_resolve_local_vlm_assets",
        lambda *args: (assets, "policies/smolvla_or_pi0/smolvla_base/vlm_assets"),
    )
    monkeypatch.setattr(
        smol,
        "_local_manifest",
        lambda path, **kwargs: (
            captured.update(
                {
                    "manifest_path": path,
                    "manifest_kwargs": kwargs,
                }
            )
            or manifest
        ),
    )
    monkeypatch.setattr(smol, "_checkpoint_rename_map", lambda _: None)
    monkeypatch.setattr(
        smol,
        "make_policy",
        lambda **kwargs: SimpleNamespace(eval=lambda: None),
    )
    monkeypatch.setattr(smol, "_validate_loaded_policy", lambda *args: None)

    def fake_processors(**kwargs: object) -> tuple[object, object]:
        captured["processor_kwargs"] = kwargs
        return object(), object()

    monkeypatch.setattr(smol, "make_pre_post_processors", fake_processors)
    loaded = smol._load_smolvla(
        policy_reference="outputs/run/checkpoints/030000/pretrained_model",
        local_path=checkpoint,
        env_config=SimpleNamespace(),
        n_action_steps_override=None,
    )

    assert loaded.manifest is manifest
    assert config.pretrained_path == checkpoint
    assert config.vlm_model_name == str(assets)
    assert captured["manifest_kwargs"] == {
        "policy_reference": "outputs/run/checkpoints/030000/pretrained_model",
        "vlm_assets_reference": ("policies/smolvla_or_pi0/smolvla_base/vlm_assets"),
    }
    processor_kwargs = captured["processor_kwargs"]
    assert processor_kwargs["pretrained_path"] == str(checkpoint)
    assert processor_kwargs["preprocessor_overrides"] == {
        "device_processor": {"device": "cpu"},
        "tokenizer_processor": {"tokenizer_name": str(assets)},
    }


def test_evaluate_smolvla_passes_hub_reference_to_shared_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def fake_evaluate(base_config: object, **kwargs: object) -> object:
        captured["config"] = base_config
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(smol, "evaluate_libero", fake_evaluate)
    result = smol.evaluate_smolvla_libero(
        smol.SmolVLAEvaluationConfig(
            policy_path="HuggingFaceVLA/smolvla_libero",
            suite="libero_goal",
            task_id=1,
            n_episodes=2,
            init_state_ids=(8, 3),
            policy_seed=2000,
            output_dir=tmp_path / "result",
        )
    )

    assert result is expected
    base = captured["config"]
    assert base.policy_path == Path("HuggingFaceVLA/smolvla_libero")
    assert base.output_dir == tmp_path / "result"
    assert base.init_state_ids == (8, 3)
    assert base.policy_seed == 2000
    assert callable(captured["policy_runtime_loader"])
