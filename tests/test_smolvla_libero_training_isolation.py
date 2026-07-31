from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_practice.experiments.smolvla_libero.paraphrase_training import (
    InstructionSelectionTracker,
    load_paraphrase_training_spec,
    override_instruction_preprocessor,
    track_successful_instruction_updates,
)
from robot_practice.experiments.smolvla_libero.target_training import load_target_split
from robot_practice.experiments.smolvla_libero.train_c2 import (
    _load_trainable_peft_resume,
    _validate_selected_b0_peft_without_scratch_warning,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PRACTICE_ROOT / "src/robot_practice/experiments/smolvla_libero"
SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_split.json"
PARAPHRASES_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/paraphrases.json"


def test_training_dependency_direction_is_only_c3_to_c2_to_c1() -> None:
    c1_sources = [
        EXPERIMENT_ROOT / "c1_training.py",
        EXPERIMENT_ROOT / "train_c1.py",
    ]
    c2_sources = [
        EXPERIMENT_ROOT / "c2_training.py",
        EXPERIMENT_ROOT / "train_c2.py",
    ]

    for path in c1_sources:
        source = path.read_text(encoding="utf-8")
        assert ".c2_training" not in source
        assert ".train_c2" not in source
        assert ".c3_training" not in source
        assert ".train_c3" not in source
        assert ".paraphrase_training" not in source
    for path in c2_sources:
        source = path.read_text(encoding="utf-8")
        assert ".c3_training" not in source
        assert ".train_c3" not in source
        assert ".paraphrase_training" not in source


def test_importing_c3_has_no_global_lerobot_side_effects() -> None:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts import lerobot_train

    make_policy = lerobot_train.make_policy
    make_processors = lerobot_train.make_pre_post_processors
    update_policy = lerobot_train.update_policy
    validate_peft = SmolVLAPolicy._validate_peft_config

    importlib.import_module("robot_practice.experiments.smolvla_libero.c3_training")
    importlib.import_module("robot_practice.experiments.smolvla_libero.train_c3")

    assert lerobot_train.make_policy is make_policy
    assert lerobot_train.make_pre_post_processors is make_processors
    assert lerobot_train.update_policy is update_policy
    assert SmolVLAPolicy._validate_peft_config is validate_peft


def test_c2_peft_contexts_restore_state_after_failure() -> None:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    original_validate = SmolVLAPolicy._validate_peft_config
    with (
        pytest.raises(RuntimeError, match="stop"),
        _validate_selected_b0_peft_without_scratch_warning(),
    ):
        assert SmolVLAPolicy._validate_peft_config is not original_validate
        raise RuntimeError("stop")
    assert SmolVLAPolicy._validate_peft_config is original_validate

    def make_policy(*args, **kwargs):
        del args, kwargs

    module = SimpleNamespace(make_policy=make_policy)
    with (
        pytest.raises(RuntimeError, match="stop"),
        _load_trainable_peft_resume(module, enabled=True),
    ):
        assert module.make_policy is not make_policy
        raise RuntimeError("stop")
    assert module.make_policy is make_policy


def test_c3_instruction_contexts_restore_state_after_failure(tmp_path: Path) -> None:
    split = load_target_split(SPLIT_FILE)
    spec = load_paraphrase_training_spec(PARAPHRASES_FILE, split)
    tracker = InstructionSelectionTracker(
        spec=spec,
        split=split,
        seed=1000,
        rank=0,
        world_size=1,
        batch_size=1,
        initial_step=0,
        total_steps=1,
        save_freq=1,
        output_dir=tmp_path,
        attempt_id=0,
    )

    def make_processors(*args, **kwargs):
        del args, kwargs

    def update_policy(*args, **kwargs):
        del args, kwargs

    module = SimpleNamespace(
        make_pre_post_processors=make_processors,
        update_policy=update_policy,
    )
    with (
        pytest.raises(RuntimeError, match="stop"),
        override_instruction_preprocessor(module, tracker),
        track_successful_instruction_updates(module, tracker),
    ):
        assert module.make_pre_post_processors is not make_processors
        assert module.update_policy is not update_policy
        raise RuntimeError("stop")
    assert module.make_pre_post_processors is make_processors
    assert module.update_policy is update_policy
