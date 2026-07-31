from __future__ import annotations

import dataclasses
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_practice.experiments.smolvla_libero.b0_training import CAMERA_RENAME_MAP
from robot_practice.experiments.smolvla_libero.c1_training import C1FreshArguments
from robot_practice.experiments.smolvla_libero.c1_training import (
    build_train_config as build_c1_train_config,
)
from robot_practice.experiments.smolvla_libero.c2_training import (
    DEFAULT_B0_SELECTED_REFERENCE,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_LORA_RANK,
    FORMAL_WORLD_SIZE,
    SMOLVLA_LORA_TARGET,
    C2FreshArguments,
    LaunchInspection,
    build_train_config,
    parse_arguments,
    requested_peft_config,
)
from robot_practice.experiments.smolvla_libero.target_training import (
    EXPECTED_TRAIN_EPISODES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_VALIDATION_EPISODES,
    EXPECTED_VALIDATION_FRAMES,
    load_target_split,
)
from robot_practice.experiments.smolvla_libero.train_c2 import (
    _load_trainable_peft_resume,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_split.json"
SCRIPT = PRACTICE_ROOT / "scripts/experiments/smolvla_libero/train_c2.sh"


def _policy_config():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    return SmolVLAConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.camera1": PolicyFeature(
                FeatureType.VISUAL,
                (3, 256, 256),
            ),
            "observation.images.camera2": PolicyFeature(
                FeatureType.VISUAL,
                (3, 256, 256),
            ),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        device="cuda",
        use_amp=False,
        use_peft=False,
        push_to_hub=False,
        chunk_size=50,
        n_action_steps=10,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
    )


def _fresh_arguments(tmp_path: Path) -> C2FreshArguments:
    return C2FreshArguments(
        mode="smoke",
        base_policy=DEFAULT_B0_SELECTED_REFERENCE,
        dataset_root=tmp_path / "dataset",
        split_file=SPLIT_FILE,
        num_processes=1,
        batch_size=1,
        steps=2,
        save_freq=2,
        learning_rate=1e-4,
        lora_rank=32,
        train_seed=1000,
        num_workers=0,
        mixed_precision="bf16",
        output_dir=tmp_path / "run",
    )


def test_frozen_target_split_has_exact_c2_counts() -> None:
    split = load_target_split(SPLIT_FILE)

    assert split.task_ids == (1, 4, 8)
    assert split.combined_task_indices == (17, 18, 10)
    assert len(split.train_episodes) == EXPECTED_TRAIN_EPISODES == 113
    assert split.train_frames == EXPECTED_TRAIN_FRAMES == 11022
    assert len(split.validation_episodes) == EXPECTED_VALIDATION_EPISODES == 30
    assert split.validation_frames == EXPECTED_VALIDATION_FRAMES == 2969
    assert not set(split.train_episodes) & set(split.validation_episodes)


def test_no_argument_cli_resolves_frozen_formal_launch(tmp_path: Path) -> None:
    inspection = parse_arguments(
        [
            f"--output-dir={tmp_path / 'formal'}",
            "--inspect-launch",
        ]
    )

    assert isinstance(inspection, LaunchInspection)
    assert inspection.num_processes == FORMAL_WORLD_SIZE == 8
    assert inspection.mixed_precision == "bf16"


@pytest.mark.parametrize(
    "override",
    [
        "--num-processes=1",
        "--batch-size=2",
        "--steps=29999",
        "--save-freq=4999",
        "--learning-rate=0.0002",
        "--lora-rank=16",
        "--train-seed=999",
    ],
)
def test_formal_cli_rejects_contract_drift(tmp_path: Path, override: str) -> None:
    with pytest.raises(ValueError):
        parse_arguments(
            [
                override,
                f"--output-dir={tmp_path / 'formal'}",
            ]
        )


def test_train_config_uses_only_target_train_episodes_and_rank_32_lora(
    tmp_path: Path,
) -> None:
    split = load_target_split(SPLIT_FILE)
    config = build_train_config(_fresh_arguments(tmp_path), split, _policy_config())
    resolved = config.to_dict()

    assert resolved["dataset"]["episodes"] == list(split.train_episodes)
    assert not set(resolved["dataset"]["episodes"]) & set(split.validation_episodes)
    assert resolved["rename_map"] == CAMERA_RENAME_MAP
    assert resolved["batch_size"] == 1
    assert resolved["policy"]["use_peft"] is False
    assert resolved["policy"]["train_expert_only"] is True
    assert resolved["policy"]["freeze_vision_encoder"] is True
    assert resolved["peft"] == dataclasses.asdict(requested_peft_config(FORMAL_LORA_RANK))
    assert resolved["peft"]["method_type"] == "LORA"
    assert resolved["peft"]["r"] == 32
    assert resolved["peft"]["target_modules"] is None
    assert resolved["peft"]["full_training_modules"] is None


def test_c2_matches_c1_except_for_experiment_name_and_peft(tmp_path: Path) -> None:
    split = load_target_split(SPLIT_FILE)
    c2_arguments = _fresh_arguments(tmp_path)
    c1_arguments = C1FreshArguments(
        mode=c2_arguments.mode,
        base_policy=c2_arguments.base_policy,
        dataset_root=c2_arguments.dataset_root,
        split_file=c2_arguments.split_file,
        num_processes=c2_arguments.num_processes,
        batch_size=c2_arguments.batch_size,
        steps=c2_arguments.steps,
        save_freq=c2_arguments.save_freq,
        learning_rate=c2_arguments.learning_rate,
        train_seed=c2_arguments.train_seed,
        num_workers=c2_arguments.num_workers,
        mixed_precision=c2_arguments.mixed_precision,
        output_dir=c2_arguments.output_dir,
    )
    c1 = build_c1_train_config(c1_arguments, split, _policy_config()).to_dict()
    c2 = build_train_config(c2_arguments, split, _policy_config()).to_dict()

    assert c1.pop("job_name") == "smolvla_libero_c1_default_non_peft"
    assert c2.pop("job_name") == "smolvla_libero_c2_canonical_lora"
    assert c1.pop("peft") is None
    assert c2.pop("peft")["r"] == 32
    assert c2 == c1


def test_trainable_resume_loads_adapter_once(tmp_path: Path) -> None:
    import torch
    from peft import LoraConfig, get_peft_model

    class TinyConfig(dict):
        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error

        def __setattr__(self, name: str, value) -> None:
            self[name] = value

    class TinyPolicy(torch.nn.Module):
        def __init__(self, config: TinyConfig) -> None:
            super().__init__()
            self.config = config
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.proj(value)

    base_dir = tmp_path / "selected_b0"
    adapter_dir = tmp_path / "adapter"
    base_dir.mkdir()
    (base_dir / "config.json").write_text("{}", encoding="utf-8")
    source_config = TinyConfig(use_peft=False, _name_or_path=str(base_dir))
    source = TinyPolicy(source_config)
    source.name_or_path = str(base_dir)
    adapter = get_peft_model(
        source,
        LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]),
    )
    adapter.save_pretrained(adapter_dir)

    runtime_config = TinyConfig(pretrained_path=adapter_dir, use_peft=True)
    calls: list[tuple[Path, bool]] = []

    def make_base_policy(*args, **kwargs):
        del args
        config = kwargs["cfg"]
        calls.append((Path(config.pretrained_path), config.use_peft))
        return TinyPolicy(config)

    module = SimpleNamespace(make_policy=make_base_policy)
    with _load_trainable_peft_resume(module, enabled=True):
        resumed = module.make_policy(cfg=runtime_config)
        assert resumed.wrap_with_peft() is resumed

    assert module.make_policy is make_base_policy
    assert calls == [(base_dir, False)]
    assert runtime_config.pretrained_path == adapter_dir
    assert runtime_config.use_peft is True
    trainable_names = [
        name for name, parameter in resumed.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)


def test_shell_prints_one_command_formal_accelerate_launch(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            f"--output-dir={tmp_path / 'formal'}",
            "--print-command",
        ],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:6] == [
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONUNBUFFERED=1",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
    ]
    assert "--num_processes=8" in command
    assert "--mixed_precision=bf16" in command
    assert "--multi_gpu" in command
    assert "robot_practice.experiments.smolvla_libero.train_c2" in command
    assert FORMAL_GLOBAL_BATCH_SIZE == 8
    assert FORMAL_LORA_RANK == 32
    assert "lm_expert" in SMOLVLA_LORA_TARGET
