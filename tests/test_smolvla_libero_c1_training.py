from __future__ import annotations

import dataclasses
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_practice.experiments.smolvla_libero.b0_training import CAMERA_RENAME_MAP
from robot_practice.experiments.smolvla_libero.c1_training import (
    DEFAULT_B0_SELECTED_REFERENCE,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_WORLD_SIZE,
    C1FreshArguments,
    LaunchInspection,
    build_train_config,
    parse_arguments,
)
from robot_practice.experiments.smolvla_libero.target_training import (
    EXPECTED_TRAIN_EPISODES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_VALIDATION_EPISODES,
    EXPECTED_VALIDATION_FRAMES,
    UniformTaskEpisodeSampler,
    load_target_split,
    override_target_training_sampler,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_split.json"
SCRIPT = PRACTICE_ROOT / "scripts/experiments/smolvla_libero/train_c1.sh"


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


def _fresh_arguments(tmp_path: Path) -> C1FreshArguments:
    return C1FreshArguments(
        mode="smoke",
        base_policy=DEFAULT_B0_SELECTED_REFERENCE,
        dataset_root=tmp_path / "dataset",
        split_file=SPLIT_FILE,
        num_processes=1,
        batch_size=1,
        steps=2,
        save_freq=2,
        learning_rate=1e-4,
        train_seed=1000,
        num_workers=0,
        mixed_precision="bf16",
        output_dir=tmp_path / "run",
    )


def test_frozen_target_split_has_exact_c1_counts() -> None:
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


def test_train_config_uses_only_target_train_episodes(tmp_path: Path) -> None:
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


def test_balanced_sampler_is_deterministic_and_task_balanced() -> None:
    split = load_target_split(SPLIT_FILE)
    one_frame_split = dataclasses.replace(
        split,
        train_frames=len(split.train_episodes),
    )
    dataset = type(
        "Dataset",
        (),
        {"hf_dataset": {"episode_index": list(split.train_episodes)}},
    )()
    first = list(
        UniformTaskEpisodeSampler(
            dataset,
            one_frame_split,
            seed=1000,
            num_samples=9000,
        )
    )
    second = list(
        UniformTaskEpisodeSampler(
            dataset,
            one_frame_split,
            seed=1000,
            num_samples=9000,
        )
    )
    assert first == second

    episode_to_task = {
        episode: task_id
        for task_id, episodes in split.train_episodes_by_task.items()
        for episode in episodes
    }
    relative_to_episode = list(split.train_episodes)
    counts = Counter(episode_to_task[relative_to_episode[index]] for index in first)
    assert set(counts) == {1, 4, 8}
    for count in counts.values():
        assert count == pytest.approx(3000, abs=150)


def test_sampler_override_restores_torch_dataloader_before_accelerate_prepare() -> None:
    import torch

    split = load_target_split(SPLIT_FILE)
    one_frame_split = dataclasses.replace(
        split,
        train_frames=len(split.train_episodes),
    )

    class Dataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.hf_dataset = {"episode_index": list(split.train_episodes)}

        def __len__(self) -> int:
            return len(split.train_episodes)

        def __getitem__(self, index: int) -> int:
            return index

    original = torch.utils.data.DataLoader
    module = SimpleNamespace(torch=torch)
    with override_target_training_sampler(module, one_frame_split, seed=1000):
        loader = module.torch.utils.data.DataLoader(Dataset(), batch_size=1, shuffle=True)
        assert module.torch.utils.data.DataLoader is original
        assert isinstance(loader, original)
        assert isinstance(loader.sampler, UniformTaskEpisodeSampler)
    assert torch.utils.data.DataLoader is original


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
    assert "robot_practice.experiments.smolvla_libero.train_c1" in command
    assert FORMAL_GLOBAL_BATCH_SIZE == 8
