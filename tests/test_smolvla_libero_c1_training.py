from __future__ import annotations

import dataclasses
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_practice.experiments.smolvla_libero import c1_training
from robot_practice.experiments.smolvla_libero.b0_training import (
    CAMERA_RENAME_MAP,
    BasePolicyIdentity,
    load_background_split,
)
from robot_practice.experiments.smolvla_libero.c1_training import (
    DEFAULT_B0_SELECTED_REFERENCE,
    DEFAULT_WORLD_SIZE,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_SAVE_FREQ,
    FORMAL_STEPS,
    C1FreshArguments,
    LaunchInspection,
    build_run_config,
    build_train_config,
    parse_arguments,
)
from robot_practice.experiments.smolvla_libero.target_training import (
    DUAL_POOL_SAMPLING_STRATEGY,
    EXPECTED_TRAIN_EPISODES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_VALIDATION_EPISODES,
    EXPECTED_VALIDATION_FRAMES,
    DualPoolTaskEpisodeSampler,
    UniformTaskEpisodeSampler,
    load_target_split,
    override_target_training_sampler,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_split.json"
BACKGROUND_SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/background_split.json"
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


def _formal_cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--mode=formal",
        "--dataset-root=datasets/lerobot/lerobot/libero",
        f"--split-file={SPLIT_FILE}",
        f"--output-dir={tmp_path / 'formal'}",
        *extra,
    ]


def _diagnostic_cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--mode=diagnostic",
        "--dataset-root=datasets/lerobot/lerobot/libero",
        f"--split-file={SPLIT_FILE}",
        f"--output-dir={tmp_path / 'diagnostic'}",
        *extra,
    ]


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
    inspection = parse_arguments([*_formal_cli_args(tmp_path), "--inspect-launch"])

    assert isinstance(inspection, LaunchInspection)
    assert inspection.num_processes == DEFAULT_WORLD_SIZE == 8
    assert inspection.mixed_precision == "bf16"


def test_fresh_cli_requires_mode_dataset_split_and_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_arguments([f"--output-dir={tmp_path / 'formal'}"])
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "--mode=formal",
                "--dataset-root=datasets/lerobot/lerobot/libero",
                f"--split-file={SPLIT_FILE}",
            ]
        )


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
        parse_arguments(_formal_cli_args(tmp_path, override))


def test_formal_accepts_four_gpus_two_per_gpu_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c1_training.torch.cuda, "device_count", lambda: 4)
    arguments = parse_arguments(
        _formal_cli_args(
            tmp_path,
            "--num-processes=4",
            "--batch-size=2",
        )
    )

    assert isinstance(arguments, C1FreshArguments)
    assert arguments.num_processes == 4
    assert arguments.batch_size == 2
    assert arguments.num_processes * arguments.batch_size == FORMAL_GLOBAL_BATCH_SIZE


def test_formal_rejects_global_batch_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c1_training.torch.cuda, "device_count", lambda: 4)
    with pytest.raises(ValueError, match=r"8x1, 4x2, 2x4, 1x8"):
        parse_arguments(
            _formal_cli_args(
                tmp_path,
                "--num-processes=4",
                "--batch-size=1",
            )
        )


def test_fresh_rejects_more_processes_than_visible_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c1_training.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        parse_arguments(
            _formal_cli_args(
                tmp_path,
                "--num-processes=2",
                "--batch-size=4",
            )
        )


def test_diagnostic_mode_requires_experiment_tag(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_arguments(_diagnostic_cli_args(tmp_path))
    with pytest.raises(SystemExit):
        parse_arguments(_formal_cli_args(tmp_path, "--experiment-tag=tag"))


def test_background_replay_is_diagnostic_only_and_defaults_to_k0(tmp_path: Path) -> None:
    frozen_diagnostic = [
        "--num-processes=1",
        "--batch-size=8",
        "--steps=30000",
        "--save-freq=5000",
    ]
    default = parse_arguments(
        _diagnostic_cli_args(tmp_path, "--experiment-tag=tag", *frozen_diagnostic)
    )
    assert isinstance(default, C1FreshArguments)
    assert default.background_replay_k == 0
    assert default.background_split_file == BACKGROUND_SPLIT_FILE

    replay = parse_arguments(
        _diagnostic_cli_args(
            tmp_path,
            "--experiment-tag=replay_k4",
            "--background-replay=4",
            *frozen_diagnostic,
        )
    )
    assert isinstance(replay, C1FreshArguments)
    assert replay.background_replay_k == 4
    assert replay.background_split_file == BACKGROUND_SPLIT_FILE

    with pytest.raises(SystemExit):
        parse_arguments(
            _formal_cli_args(
                tmp_path,
                "--background-replay=4",
            )
        )
    with pytest.raises(SystemExit):
        parse_arguments(
            _diagnostic_cli_args(
                tmp_path,
                "--experiment-tag=too_many",
                "--background-replay=8",
            )
        )


def test_diagnostic_allows_lr_and_steps_override(tmp_path: Path) -> None:
    arguments = parse_arguments(
        _diagnostic_cli_args(
            tmp_path,
            "--experiment-tag=a_same_config",
            "--num-processes=1",
            "--batch-size=8",
            "--steps=20000",
            "--save-freq=5000",
            "--learning-rate=0.0000125",
        )
    )

    assert isinstance(arguments, C1FreshArguments)
    assert arguments.mode == "diagnostic"
    assert arguments.experiment_tag == "a_same_config"
    assert arguments.learning_rate == pytest.approx(1.25e-5)
    assert arguments.steps == 20000


def test_diagnostic_enforces_frozen_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c1_training.torch.cuda, "device_count", lambda: 4)
    with pytest.raises(ValueError, match="train_seed"):
        parse_arguments(
            _diagnostic_cli_args(
                tmp_path,
                "--experiment-tag=tag",
                "--num-processes=1",
                "--batch-size=8",
                "--steps=30000",
                "--save-freq=5000",
                "--train-seed=999",
            )
        )
    with pytest.raises(ValueError, match="save_freq"):
        parse_arguments(
            _diagnostic_cli_args(
                tmp_path,
                "--experiment-tag=tag",
                "--num-processes=1",
                "--batch-size=8",
                "--steps=30000",
                "--save-freq=2",
            )
        )
    with pytest.raises(ValueError, match=r"8x1, 4x2, 2x4, 1x8"):
        parse_arguments(
            _diagnostic_cli_args(
                tmp_path,
                "--experiment-tag=tag",
                "--num-processes=4",
                "--batch-size=1",
                "--steps=30000",
                "--save-freq=5000",
            )
        )


def test_diagnostic_run_config_records_tag_and_label(tmp_path: Path) -> None:
    arguments = dataclasses.replace(
        _fresh_arguments(tmp_path),
        mode="diagnostic",
        experiment_tag="b_scaled_lr",
        learning_rate=1.25e-5,
        steps=30000,
        save_freq=5000,
        num_processes=1,
        batch_size=8,
    )
    split = load_target_split(SPLIT_FILE)
    dataset_audit = SimpleNamespace(
        fps=10,
        features={"observation.image": {"shape": [3, 256, 256]}},
    )
    train_config = SimpleNamespace(
        to_dict=lambda: {"optimizer": {"type": "adamw"}, "scheduler": {"type": "cosine"}},
        use_policy_training_preset=True,
    )
    base_identity = BasePolicyIdentity(
        requested="policies/smolvla_or_pi0/smolvla_base",
        source_type="local",
        resolved_revision=None,
        files_sha256={},
    )

    payload = build_run_config(
        arguments=arguments,
        split=split,
        dataset_audit=dataset_audit,
        base_identity=base_identity,
        base_selection=None,
        train_config=train_config,
    )

    assert payload["result_label"] == "Diagnostic"
    assert payload["training"]["mode"] == "diagnostic"
    assert payload["training"]["experiment_tag"] == "b_scaled_lr"


def test_diagnostic_learning_rate_passes_policy_preset_validation() -> None:
    config = _policy_config()
    config.optimizer_lr = 1.25e-5

    c1_training.validate_c1_policy_preset(
        config,
        expected_learning_rate=1.25e-5,
    )
    with pytest.raises(ValueError, match="Optimizer preset mismatch"):
        c1_training.validate_c1_policy_preset(config)

    config.optimizer_lr = 1e-4
    c1_training.validate_c1_policy_preset(config)


def test_run_config_records_data_parallel_layout(tmp_path: Path) -> None:
    arguments = dataclasses.replace(
        _fresh_arguments(tmp_path),
        mode="formal",
        num_processes=4,
        batch_size=2,
    )
    split = load_target_split(SPLIT_FILE)
    dataset_audit = SimpleNamespace(
        fps=10,
        features={"observation.image": {"shape": [3, 256, 256]}},
    )
    train_config = SimpleNamespace(
        to_dict=lambda: {"optimizer": {"type": "adamw"}, "scheduler": {"type": "cosine"}},
        use_policy_training_preset=True,
    )
    base_identity = BasePolicyIdentity(
        requested="policies/smolvla_or_pi0/smolvla_base",
        source_type="local",
        resolved_revision=None,
        files_sha256={},
    )

    payload = build_run_config(
        arguments=arguments,
        split=split,
        dataset_audit=dataset_audit,
        base_identity=base_identity,
        base_selection=None,
        train_config=train_config,
    )

    assert payload["training"]["data_parallel_layout"] == "4x2"
    assert payload["training"]["global_batch_size"] == 8


def test_resume_rejects_changed_world_size() -> None:
    training = {
        "world_size": 8,
        "per_gpu_batch_size": 1,
        "global_batch_size": 8,
    }
    with pytest.raises(ValueError, match="world_size"):
        c1_training.validate_resume_batch_contract(
            requested_world_size=4,
            training=training,
        )
    c1_training.validate_resume_batch_contract(
        requested_world_size=8,
        training=training,
    )


def test_resume_formal_contract_allows_any_legal_layout() -> None:
    training = {
        "world_size": 4,
        "per_gpu_batch_size": 2,
        "global_batch_size": 8,
        "mixed_precision": "bf16",
        "steps": FORMAL_STEPS,
        "save_freq": FORMAL_SAVE_FREQ,
    }
    c1_training.validate_resume_batch_contract(
        requested_world_size=4,
        training=training,
    )
    c1_training.validate_resume_formal_contract(training)
    with pytest.raises(ValueError, match="global_batch_size"):
        c1_training.validate_resume_formal_contract(dict(training, global_batch_size=4))


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


def test_replay_train_config_uses_union_and_records_provenance(tmp_path: Path) -> None:
    target = load_target_split(SPLIT_FILE)
    background = load_background_split(BACKGROUND_SPLIT_FILE)
    arguments = dataclasses.replace(
        _fresh_arguments(tmp_path),
        mode="diagnostic",
        experiment_tag="replay_k4",
        background_replay_k=4,
        background_split_file=BACKGROUND_SPLIT_FILE,
        num_processes=8,
        batch_size=1,
        steps=30000,
        save_freq=5000,
    )
    config = build_train_config(arguments, target, _policy_config(), background)
    resolved = config.to_dict()
    expected_episodes = sorted(set(target.train_episodes) | set(background.train_episodes))
    assert resolved["dataset"]["episodes"] == expected_episodes

    dataset_audit = SimpleNamespace(
        fps=10,
        features={"observation.image": {"shape": [3, 256, 256]}},
    )
    train_config = SimpleNamespace(
        to_dict=lambda: {"optimizer": {"type": "adamw"}, "scheduler": {"type": "cosine"}},
        use_policy_training_preset=True,
    )
    base_identity = BasePolicyIdentity(
        requested="policies/smolvla_or_pi0/smolvla_base",
        source_type="local",
        resolved_revision=None,
        files_sha256={},
    )
    payload = build_run_config(
        arguments=arguments,
        split=target,
        dataset_audit=dataset_audit,
        base_identity=base_identity,
        base_selection=None,
        train_config=train_config,
        background_split=background,
    )
    assert payload["dataset"]["sampler"] == DUAL_POOL_SAMPLING_STRATEGY
    assert payload["dataset"]["background_replay_k"] == 4
    assert payload["dataset"]["replay_fraction"] == pytest.approx(0.5)
    assert payload["dataset"]["target_train_episodes"] == 113
    assert payload["dataset"]["background_train_episodes"] == 1412
    assert payload["inputs"]["background_split_sha256"] == background.sha256


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


@pytest.mark.parametrize("num_processes,batch_size", [(8, 1), (4, 2), (2, 4), (1, 8)])
@pytest.mark.parametrize("background_replay_k", [1, 2, 4, 6, 7])
def test_dual_pool_sampler_has_exact_global_batch_ratio(
    num_processes: int,
    batch_size: int,
    background_replay_k: int,
) -> None:
    target = load_target_split(SPLIT_FILE)
    background = load_background_split(BACKGROUND_SPLIT_FILE)
    target = dataclasses.replace(target, train_frames=len(target.train_episodes))
    background = dataclasses.replace(background, train_frames=len(background.train_episodes))
    episodes = list(target.train_episodes) + list(background.train_episodes)
    dataset = type("Dataset", (), {"hf_dataset": {"episode_index": episodes}})()
    sampler = DualPoolTaskEpisodeSampler(
        dataset,
        target,
        background,
        background_replay_k=background_replay_k,
        seed=1000,
        num_processes=num_processes,
        batch_size=batch_size,
        num_samples=8 * 12,
    )
    first = list(sampler)
    second = list(
        DualPoolTaskEpisodeSampler(
            dataset,
            target,
            background,
            background_replay_k=background_replay_k,
            seed=1000,
            num_processes=num_processes,
            batch_size=batch_size,
            num_samples=8 * 12,
        )
    )
    assert first == second

    target_episodes = set(target.train_episodes)
    background_episodes = set(background.train_episodes)
    for offset in range(0, len(first), 8):
        global_batch = [episodes[index] for index in first[offset : offset + 8]]
        assert (
            sum(episode in background_episodes for episode in global_batch) == background_replay_k
        )
        assert (
            sum(episode in target_episodes for episode in global_batch) == 8 - background_replay_k
        )
        for local_offset in range(0, 8, batch_size):
            local_batch = global_batch[local_offset : local_offset + batch_size]
            assert all(episode in target_episodes | background_episodes for episode in local_batch)


def test_dual_pool_sampler_rejects_overlap() -> None:
    target = load_target_split(SPLIT_FILE)
    background = load_background_split(BACKGROUND_SPLIT_FILE)
    overlap_episode = target.train_episodes[0]
    background = dataclasses.replace(
        background,
        train_episodes=(overlap_episode, *background.train_episodes[1:]),
    )
    dataset = type(
        "Dataset",
        (),
        {
            "hf_dataset": {
                "episode_index": list(target.train_episodes) + list(background.train_episodes)
            }
        },
    )()
    with pytest.raises(ValueError, match="overlap"):
        DualPoolTaskEpisodeSampler(
            dataset,
            target,
            background,
            background_replay_k=4,
            seed=1000,
            num_processes=8,
            batch_size=1,
        )


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
            "--mode=formal",
            "--dataset-root=datasets/lerobot/lerobot/libero",
            f"--split-file={SPLIT_FILE}",
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


def test_shell_prints_four_gpu_two_batch_accelerate_launch(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--mode=formal",
            "--dataset-root=datasets/lerobot/lerobot/libero",
            f"--split-file={SPLIT_FILE}",
            "--num-processes=4",
            "--batch-size=2",
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
    assert "--num_processes=4" in command
    assert "--multi_gpu" in command
    assert "robot_practice.experiments.smolvla_libero.train_c1" in command


def test_shell_prints_replay_accelerate_launch(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--mode=diagnostic",
            "--experiment-tag=replay_k4",
            "--background-replay=4",
            f"--background-split-file={BACKGROUND_SPLIT_FILE}",
            "--dataset-root=datasets/lerobot/lerobot/libero",
            f"--split-file={SPLIT_FILE}",
            "--num-processes=8",
            "--batch-size=1",
            "--steps=30000",
            "--save-freq=5000",
            "--learning-rate=0.0001",
            f"--output-dir={tmp_path / 'replay_k4'}",
            "--print-command",
        ],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "--mode=diagnostic" in command
    assert "--experiment-tag=replay_k4" in command
    assert "--background-replay=4" in command
    assert f"--background-split-file={BACKGROUND_SPLIT_FILE}" in command
    assert "--num_processes=8" in command
    assert "--mixed_precision=bf16" in command
