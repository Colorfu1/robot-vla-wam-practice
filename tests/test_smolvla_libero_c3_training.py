from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_practice.experiments.smolvla_libero import c3_training
from robot_practice.experiments.smolvla_libero.b0_training import BasePolicyIdentity
from robot_practice.experiments.smolvla_libero.c1_training import DEFAULT_WORLD_SIZE
from robot_practice.experiments.smolvla_libero.c2_training import (
    C2FreshArguments,
    PeftInspection,
    requested_peft_config,
)
from robot_practice.experiments.smolvla_libero.c2_training import (
    build_train_config as build_c2_train_config,
)
from robot_practice.experiments.smolvla_libero.c3_training import (
    DEFAULT_B0_SELECTED_REFERENCE,
    DEFAULT_PARAPHRASES_FILE,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_LORA_RANK,
    C3FreshArguments,
    LaunchInspection,
    build_run_config,
    build_train_config,
    parse_arguments,
)
from robot_practice.experiments.smolvla_libero.paraphrase_training import (
    CANONICAL_PROBABILITY,
    INSTRUCTION_SAMPLING_CONTRACT,
    InstructionSelectionTracker,
    aggregate_instruction_counts,
    choose_instruction,
    instruction_snapshot_path,
    load_paraphrase_training_spec,
    load_rank_instruction_baseline,
    override_instruction_preprocessor,
    track_successful_instruction_updates,
    validate_instruction_snapshot,
)
from robot_practice.experiments.smolvla_libero.target_training import load_target_split

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_split.json"
PARAPHRASES_FILE = PRACTICE_ROOT / DEFAULT_PARAPHRASES_FILE
SCRIPT = PRACTICE_ROOT / "scripts/experiments/smolvla_libero/train_c3.sh"


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


def _fresh_arguments(tmp_path: Path) -> C3FreshArguments:
    return C3FreshArguments(
        mode="smoke",
        base_policy=DEFAULT_B0_SELECTED_REFERENCE,
        dataset_root=tmp_path / "dataset",
        split_file=SPLIT_FILE,
        paraphrases=PARAPHRASES_FILE,
        canonical_probability=0.25,
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


def _batch(spec, task_id: int, dataset_index: int) -> dict:
    task = spec.tasks[task_id]
    return {
        "task_index": [task.combined_task_index],
        "index": [dataset_index],
        "task": [task.canonical],
    }


def _batch_multi(spec, task_id: int, dataset_indices: list[int]) -> dict:
    task = spec.tasks[task_id]
    return {
        "task_index": [task.combined_task_index] * len(dataset_indices),
        "index": list(dataset_indices),
        "task": [task.canonical] * len(dataset_indices),
    }


def test_real_paraphrases_are_bound_and_selector_excludes_held_out_text() -> None:
    split = load_target_split(SPLIT_FILE)
    spec = load_paraphrase_training_spec(PARAPHRASES_FILE, split)
    payload = json.loads(PARAPHRASES_FILE.read_text(encoding="utf-8"))

    assert spec.sha256 == split.payload["paraphrases_sha256"]
    assert spec.canonical_probability == CANONICAL_PROBABILITY == 0.25
    assert set(spec.tasks) == {1, 4, 8}
    for task_id, task in spec.tasks.items():
        assert len(task.train_paraphrases) == 8
        assert len(task.held_out_sha256) == 5
        assert not hasattr(task, "held_out_paraphrases")
        selected = {
            choose_instruction(
                task,
                seed=1000,
                step=step,
                rank=0,
                batch_index=0,
                dataset_index=step * 7,
            )[0]
            for step in range(1, 2000)
        }
        assert not selected & set(payload["tasks"][str(task_id)]["held_out_paraphrases"])


def test_instruction_sampling_is_deterministic_and_matches_registered_mix() -> None:
    spec = load_paraphrase_training_spec(PARAPHRASES_FILE, load_target_split(SPLIT_FILE))
    task = spec.tasks[1]
    first = [
        choose_instruction(
            task,
            seed=1000,
            step=step,
            rank=3,
            batch_index=0,
            dataset_index=step * 11,
        )
        for step in range(1, 10001)
    ]
    second = [
        choose_instruction(
            task,
            seed=1000,
            step=step,
            rank=3,
            batch_index=0,
            dataset_index=step * 11,
        )
        for step in range(1, 10001)
    ]

    assert first == second
    canonical_ratio = sum(identifier == "canonical" for _, identifier in first) / len(first)
    assert canonical_ratio == pytest.approx(0.25, abs=0.015)
    assert {identifier for _, identifier in first if identifier != "canonical"} == {
        f"train_{index}" for index in range(8)
    }


def test_tracker_snapshots_are_cumulative_across_resume(tmp_path: Path) -> None:
    split = load_target_split(SPLIT_FILE)
    spec = load_paraphrase_training_spec(PARAPHRASES_FILE, split)
    first = InstructionSelectionTracker(
        spec=spec,
        split=split,
        seed=1000,
        rank=0,
        world_size=1,
        batch_size=1,
        initial_step=0,
        total_steps=2,
        save_freq=2,
        output_dir=tmp_path,
        attempt_id=0,
    )
    for step, task_id in enumerate((1, 4), start=1):
        batch = _batch(spec, task_id, dataset_index=100 + step)
        first.apply(batch)
        assert batch["task"][0] in {
            spec.tasks[task_id].canonical,
            *spec.tasks[task_id].train_paraphrases,
        }
        first.complete_step()
    first.ensure_complete()

    baseline = load_rank_instruction_baseline(
        run_dir=tmp_path,
        spec=spec,
        split=split,
        seed=1000,
        step=2,
        rank=0,
        world_size=1,
        batch_size=1,
    )
    resumed = InstructionSelectionTracker(
        spec=spec,
        split=split,
        seed=1000,
        rank=0,
        world_size=1,
        batch_size=1,
        initial_step=2,
        total_steps=4,
        save_freq=2,
        output_dir=tmp_path,
        attempt_id=1,
        baseline=baseline,
    )
    for step, task_id in enumerate((8, 1), start=3):
        resumed.apply(_batch(spec, task_id, dataset_index=100 + step))
        resumed.complete_step()
    resumed.ensure_complete()

    final_path = instruction_snapshot_path(tmp_path, step=4, rank=0)
    final = validate_instruction_snapshot(
        final_path,
        spec=spec,
        split=split,
        seed=1000,
        step=4,
        rank=0,
        world_size=1,
        batch_size=1,
    )
    assert final["total_selections"] == 4
    summary = aggregate_instruction_counts(
        run_dir=tmp_path,
        spec=spec,
        split=split,
        seed=1000,
        step=4,
        world_size=1,
        batch_size=1,
    )
    assert summary["total_selections"] == 4
    assert summary["sampling_contract"] == INSTRUCTION_SAMPLING_CONTRACT


def test_instruction_summary_aggregates_all_ranks(tmp_path: Path) -> None:
    split = load_target_split(SPLIT_FILE)
    spec = load_paraphrase_training_spec(PARAPHRASES_FILE, split)
    for rank in range(2):
        tracker = InstructionSelectionTracker(
            spec=spec,
            split=split,
            seed=1000,
            rank=rank,
            world_size=2,
            batch_size=1,
            initial_step=0,
            total_steps=3,
            save_freq=3,
            output_dir=tmp_path,
            attempt_id=0,
        )
        for step, task_id in enumerate((1, 4, 8), start=1):
            tracker.apply(_batch(spec, task_id, dataset_index=rank * 100 + step))
            tracker.complete_step()
        tracker.ensure_complete()

    summary = aggregate_instruction_counts(
        run_dir=tmp_path,
        spec=spec,
        split=split,
        seed=1000,
        step=3,
        world_size=2,
        batch_size=1,
    )
    assert summary["total_selections"] == 6
    assert len(summary["rank_snapshots"]) == 2
    assert {record["total"] for record in summary["tasks"].values()} == {2}


def test_instruction_hook_runs_before_processor_and_commits_after_update(
    tmp_path: Path,
) -> None:
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

    class Processor:
        def __init__(self) -> None:
            self.observed: list[str] | None = None

        def __call__(self, batch):
            self.observed = list(batch["task"])
            return batch

        def save_pretrained(self, path) -> Path:
            return Path(path)

    processor = Processor()

    def make_processors(*args, **kwargs):
        del args, kwargs
        return processor, "post"

    def update_policy(*args, **kwargs):
        del args, kwargs
        return "updated"

    module = SimpleNamespace(
        make_pre_post_processors=make_processors,
        update_policy=update_policy,
    )
    with (
        override_instruction_preprocessor(module, tracker),
        track_successful_instruction_updates(module, tracker),
    ):
        preprocessor, postprocessor = module.make_pre_post_processors()
        batch = _batch(spec, 1, dataset_index=123)
        preprocessor(batch)
        assert processor.observed == batch["task"]
        assert postprocessor == "post"
        assert preprocessor.save_pretrained(tmp_path) == tmp_path
        module.update_policy()
    tracker.ensure_complete()


def test_no_argument_cli_resolves_frozen_formal_launch(tmp_path: Path) -> None:
    inspection = parse_arguments(
        [
            f"--output-dir={tmp_path / 'formal'}",
            "--inspect-launch",
        ]
    )

    assert isinstance(inspection, LaunchInspection)
    assert inspection.num_processes == DEFAULT_WORLD_SIZE == 8
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
        "--canonical-probability=0.5",
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


def test_formal_accepts_four_gpus_two_per_gpu_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c3_training.torch.cuda, "device_count", lambda: 4)
    arguments = parse_arguments(
        [
            "--num-processes=4",
            "--batch-size=2",
            f"--output-dir={tmp_path / 'formal'}",
        ]
    )

    assert isinstance(arguments, C3FreshArguments)
    assert arguments.num_processes == 4
    assert arguments.batch_size == 2
    assert arguments.num_processes * arguments.batch_size == FORMAL_GLOBAL_BATCH_SIZE


def test_formal_rejects_global_batch_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c3_training.torch.cuda, "device_count", lambda: 4)
    with pytest.raises(ValueError, match=r"8x1, 4x2, 2x4, 1x8"):
        parse_arguments(
            [
                "--num-processes=4",
                "--batch-size=1",
                f"--output-dir={tmp_path / 'formal'}",
            ]
        )


def test_fresh_rejects_more_processes_than_visible_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c3_training.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        parse_arguments(
            [
                "--num-processes=2",
                "--batch-size=4",
                f"--output-dir={tmp_path / 'formal'}",
            ]
        )


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
        peft=requested_peft_config(32),
    )
    base_identity = BasePolicyIdentity(
        requested="policies/smolvla_or_pi0/smolvla_base",
        source_type="local",
        resolved_revision=None,
        files_sha256={},
    )
    peft_inspection = PeftInspection(
        total_parameters=1,
        trainable_parameters=1,
        trainable_parameter_names=(),
        effective_config={},
    )
    paraphrase_spec = load_paraphrase_training_spec(PARAPHRASES_FILE, split)

    payload = build_run_config(
        arguments=arguments,
        split=split,
        dataset_audit=dataset_audit,
        base_identity=base_identity,
        base_selection=None,
        train_config=train_config,
        peft_inspection=peft_inspection,
        paraphrase_spec=paraphrase_spec,
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
        c3_training.validate_resume_batch_contract(
            requested_world_size=4,
            training=training,
        )
    c3_training.validate_resume_batch_contract(
        requested_world_size=8,
        training=training,
    )


def test_resume_formal_contract_allows_any_legal_layout() -> None:
    training = {
        "world_size": 4,
        "per_gpu_batch_size": 2,
        "global_batch_size": 8,
        "mixed_precision": "bf16",
        "steps": 30000,
        "save_freq": 5000,
    }
    c3_training.validate_resume_batch_contract(
        requested_world_size=4,
        training=training,
    )
    c3_training.validate_resume_formal_contract(training)
    with pytest.raises(ValueError, match="global_batch_size"):
        c3_training.validate_resume_formal_contract(dict(training, global_batch_size=4))


def test_instruction_snapshots_scale_with_world_size_and_keep_global_total(
    tmp_path: Path,
) -> None:
    split = load_target_split(SPLIT_FILE)
    spec = load_paraphrase_training_spec(PARAPHRASES_FILE, split)
    for world_size, batch_size in ((4, 2), (8, 1)):
        run_dir = tmp_path / f"world_{world_size}"
        run_dir.mkdir()
        for rank in range(world_size):
            tracker = InstructionSelectionTracker(
                spec=spec,
                split=split,
                seed=1000,
                rank=rank,
                world_size=world_size,
                batch_size=batch_size,
                initial_step=0,
                total_steps=3,
                save_freq=3,
                output_dir=run_dir,
                attempt_id=0,
            )
            for step, task_id in enumerate((1, 4, 8), start=1):
                tracker.apply(
                    _batch_multi(
                        spec,
                        task_id,
                        dataset_indices=[
                            rank * 100 + step * 10 + batch_index
                            for batch_index in range(batch_size)
                        ],
                    )
                )
                tracker.complete_step()
            tracker.ensure_complete()

        summary = aggregate_instruction_counts(
            run_dir=run_dir,
            spec=spec,
            split=split,
            seed=1000,
            step=3,
            world_size=world_size,
            batch_size=batch_size,
        )
        assert summary["total_selections"] == 24
        assert len(summary["rank_snapshots"]) == world_size


def test_c3_train_config_matches_c2_except_for_job_name(tmp_path: Path) -> None:
    split = load_target_split(SPLIT_FILE)
    c3_arguments = _fresh_arguments(tmp_path)
    c2_arguments = C2FreshArguments(
        mode=c3_arguments.mode,
        base_policy=c3_arguments.base_policy,
        dataset_root=c3_arguments.dataset_root,
        split_file=c3_arguments.split_file,
        num_processes=c3_arguments.num_processes,
        batch_size=c3_arguments.batch_size,
        steps=c3_arguments.steps,
        save_freq=c3_arguments.save_freq,
        learning_rate=c3_arguments.learning_rate,
        lora_rank=c3_arguments.lora_rank,
        train_seed=c3_arguments.train_seed,
        num_workers=c3_arguments.num_workers,
        mixed_precision=c3_arguments.mixed_precision,
        output_dir=c3_arguments.output_dir,
    )
    c2 = build_c2_train_config(c2_arguments, split, _policy_config()).to_dict()
    c3 = build_train_config(c3_arguments, split, _policy_config()).to_dict()

    assert c2.pop("job_name") == "smolvla_libero_c2_canonical_lora"
    assert c3.pop("job_name") == "smolvla_libero_c3_paraphrase_lora"
    assert c3 == c2
    assert c3["peft"]["r"] == FORMAL_LORA_RANK == 32


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
    assert "robot_practice.experiments.smolvla_libero.train_c3" in command
    assert FORMAL_GLOBAL_BATCH_SIZE == 8


def test_shell_prints_four_gpu_two_batch_accelerate_launch(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
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
    assert "robot_practice.experiments.smolvla_libero.train_c3" in command
