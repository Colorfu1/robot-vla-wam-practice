from __future__ import annotations

import io
import json
import shlex
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robot_practice.experiments.smolvla_libero import b0_training
from robot_practice.experiments.smolvla_libero import train_b0 as train_b0_module
from robot_practice.experiments.smolvla_libero.b0_training import (
    BASE_POLICY_REPO_ID,
    CAMERA_RENAME_MAP,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_STEPS,
    LOCAL_BASE_POLICY_PATH,
    LOCAL_BASE_POLICY_REFERENCE,
    LOCAL_VLM_ASSETS_REFERENCE,
    OFFLINE_VLM_ASSETS_DIRNAME,
    REQUIRED_OFFLINE_VLM_ASSET_FILES,
    BackgroundSplit,
    DatasetAudit,
    FreshArguments,
    LaunchInspection,
    MetricsSink,
    ResumeArguments,
    build_train_config,
    configure_smolvla_policy,
    instrument_update_policy,
    load_background_split,
    parse_arguments,
    resolve_base_policy,
    validate_local_vlm_assets,
    validate_resume,
)
from robot_practice.experiments.smolvla_libero.train_b0 import (
    _attempt_artifacts,
    _override_offline_tokenizer_processor,
    _publish_artifacts_on_checkpoint,
    _resume_training,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = (
    PRACTICE_ROOT / "configs/experiments/smolvla_libero/background_split.json"
)
SCRIPT = PRACTICE_ROOT / "scripts/experiments/smolvla_libero/train_b0.sh"


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
        output_features={
            "action": PolicyFeature(FeatureType.ACTION, (7,)),
        },
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


def _fresh_arguments(tmp_path: Path, **overrides: Any) -> FreshArguments:
    values: dict[str, Any] = {
        "mode": "smoke",
        "base_policy": LOCAL_BASE_POLICY_REFERENCE,
        "dataset_root": tmp_path / "dataset",
        "split_file": SPLIT_FILE,
        "num_processes": 1,
        "batch_size": 1,
        "steps": 4,
        "save_freq": 2,
        "learning_rate": 1e-4,
        "train_seed": 1000,
        "num_workers": 0,
        "mixed_precision": "bf16",
        "output_dir": tmp_path / "run",
    }
    values.update(overrides)
    return FreshArguments(**values)


def test_frozen_background_split_has_exact_b0_counts() -> None:
    split = load_background_split(SPLIT_FILE)

    assert len(split.tasks) == 37
    assert len(split.train_episodes) == 1412
    assert split.train_frames == 236470
    assert len(split.validation_episodes) == 138
    assert split.validation_frames == 23004
    assert not {10, 17, 18} & set(split.tasks)


def test_background_split_rejects_global_duplicate_episode(tmp_path: Path) -> None:
    payload = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    payload["tasks"]["1"]["train_episode_indices"][0] = payload["tasks"]["0"][
        "train_episode_indices"
    ][0]
    mutated = tmp_path / "background_split.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="multiple B0 train tasks"):
        load_background_split(mutated)


def test_formal_cli_freezes_global_batch_and_steps(tmp_path: Path) -> None:
    inspection = parse_arguments(
        [
            "--mode=formal",
            "--dataset-root=datasets/lerobot/lerobot/libero",
            f"--split-file={SPLIT_FILE}",
            "--num-processes=8",
            "--batch-size=8",
            "--steps=30000",
            "--save-freq=5000",
            "--learning-rate=0.0001",
            "--train-seed=1000",
            "--mixed-precision=bf16",
            f"--output-dir={tmp_path / 'formal'}",
            "--inspect-launch",
        ]
    )

    assert inspection == LaunchInspection(num_processes=8, mixed_precision="bf16")
    assert 8 * 8 == FORMAL_GLOBAL_BATCH_SIZE
    assert FORMAL_STEPS == 30000


def test_fresh_cli_defaults_to_materialized_local_base_policy(tmp_path: Path) -> None:
    arguments = parse_arguments(
        [
            "--mode=smoke",
            "--dataset-root=datasets/lerobot/lerobot/libero",
            f"--split-file={SPLIT_FILE}",
            f"--output-dir={tmp_path / 'smoke'}",
        ]
    )

    assert isinstance(arguments, FreshArguments)
    assert arguments.base_policy == LOCAL_BASE_POLICY_REFERENCE


def test_explicit_missing_local_base_policy_fails_without_hub_lookup(tmp_path: Path) -> None:
    missing = tmp_path / "missing_base_policy"

    with pytest.raises(FileNotFoundError, match="materialize_base_policy"):
        resolve_base_policy(str(missing))


def _write_local_base_policy_skeleton(root: Path) -> Path:
    policy = _policy_config()
    policy.load_vlm_weights = True
    policy.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    policy.save_pretrained(root)
    (root / "model.safetensors").touch()
    (root / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (root / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    return root


def test_local_base_policy_requires_offline_vlm_assets(tmp_path: Path) -> None:
    base = _write_local_base_policy_skeleton(tmp_path / "base")

    with pytest.raises(FileNotFoundError, match="offline VLM assets"):
        resolve_base_policy(str(base))


def test_local_vlm_assets_are_resolved_without_hub_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = tmp_path / OFFLINE_VLM_ASSETS_DIRNAME
    assets.mkdir()
    for filename in REQUIRED_OFFLINE_VLM_ASSET_FILES:
        (assets / filename).write_text("{}", encoding="utf-8")
    calls: list[tuple[str, Path, bool]] = []

    from transformers import AutoConfig, AutoProcessor

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda path, *, local_files_only: calls.append(
            ("config", Path(path), local_files_only)
        )
        or SimpleNamespace(model_type="smolvlm"),
    )
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        lambda path, *, local_files_only: calls.append(
            ("processor", Path(path), local_files_only)
        )
        or object(),
    )

    assert validate_local_vlm_assets(assets) == assets.resolve()
    assert calls == [
        ("config", assets.resolve(), True),
        ("processor", assets.resolve(), True),
    ]


def test_local_base_config_uses_offline_vlm_assets_without_preloading_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = _write_local_base_policy_skeleton(tmp_path / "base")
    assets = base / OFFLINE_VLM_ASSETS_DIRNAME
    assets.mkdir()
    monkeypatch.setattr(
        b0_training,
        "validate_local_vlm_assets",
        lambda path: Path(path).resolve(),
    )

    policy = configure_smolvla_policy(
        base,
        device="cpu",
        learning_rate=1e-4,
    )

    assert Path(policy.vlm_model_name) == assets.resolve()
    assert policy.load_vlm_weights is False
    assert policy.pretrained_path == base


def test_b0_rejects_explicit_hub_base_for_offline_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="offline-only"):
        parse_arguments(
            [
                "--mode=smoke",
                f"--base-policy={BASE_POLICY_REPO_ID}",
                "--dataset-root=datasets/lerobot/lerobot/libero",
                f"--split-file={SPLIT_FILE}",
                f"--output-dir={tmp_path / 'smoke'}",
            ]
        )


def test_offline_tokenizer_override_merges_with_upstream_processor_overrides(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def make_pre_post_processors(*args: Any, **kwargs: Any):
        calls.append(kwargs)
        return "preprocessor", "postprocessor"

    module = SimpleNamespace(make_pre_post_processors=make_pre_post_processors)
    tokenizer_path = tmp_path / "vlm_assets"
    with _override_offline_tokenizer_processor(module, tokenizer_path):
        result = module.make_pre_post_processors(
            policy_cfg="policy",
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
            },
        )

    assert result == ("preprocessor", "postprocessor")
    assert calls == [
        {
            "policy_cfg": "policy",
            "preprocessor_overrides": {
                "device_processor": {"device": "cuda"},
                "tokenizer_processor": {
                    "tokenizer_name": str(tokenizer_path.resolve()),
                },
            },
        }
    ]
    assert module.make_pre_post_processors is make_pre_post_processors


def test_offline_tokenizer_override_serializes_repository_path_as_relative() -> None:
    calls: list[dict[str, Any]] = []

    def make_pre_post_processors(*args: Any, **kwargs: Any):
        calls.append(kwargs)
        return object(), object()

    module = SimpleNamespace(make_pre_post_processors=make_pre_post_processors)
    with _override_offline_tokenizer_processor(
        module,
        LOCAL_BASE_POLICY_PATH / OFFLINE_VLM_ASSETS_DIRNAME,
    ):
        module.make_pre_post_processors(policy_cfg="policy")

    assert calls[0]["preprocessor_overrides"]["tokenizer_processor"] == {
        "tokenizer_name": LOCAL_VLM_ASSETS_REFERENCE,
    }


@pytest.mark.parametrize(
    "override",
    [
        "--batch-size=7",
        "--steps=29999",
        "--learning-rate=0.0002",
        "--train-seed=999",
    ],
)
def test_formal_cli_rejects_contract_drift(
    tmp_path: Path,
    override: str,
) -> None:
    arguments = [
        "--mode=formal",
        "--dataset-root=datasets/lerobot/lerobot/libero",
        f"--split-file={SPLIT_FILE}",
        "--num-processes=8",
        "--batch-size=8",
        "--steps=30000",
        "--save-freq=5000",
        "--learning-rate=0.0001",
        "--train-seed=1000",
        "--mixed-precision=bf16",
        f"--output-dir={tmp_path / 'formal'}",
    ]
    option = override.split("=", maxsplit=1)[0]
    arguments = [argument for argument in arguments if not argument.startswith(f"{option}=")]

    with pytest.raises(ValueError):
        parse_arguments([*arguments, override])


def test_train_config_uses_native_frame_sampling_and_frozen_features(
    tmp_path: Path,
) -> None:
    split = load_background_split(SPLIT_FILE)
    config = build_train_config(
        _fresh_arguments(tmp_path),
        split,
        _policy_config(),
    )
    resolved = config.to_dict()

    assert resolved["dataset"]["episodes"] == list(split.train_episodes)
    assert resolved["dataset"]["use_imagenet_stats"] is True
    assert resolved["rename_map"] == CAMERA_RENAME_MAP
    assert resolved["batch_size"] == 1
    assert resolved["policy"]["use_peft"] is False
    assert resolved["policy"]["input_features"]["observation.state"]["shape"] == [8]
    assert resolved["policy"]["output_features"]["action"]["shape"] == [7]
    assert resolved["optimizer"]["grad_clip_norm"] == 10
    assert resolved["scheduler"]["num_decay_steps"] == 30000


def test_train_config_serializes_repository_paths_as_relative(tmp_path: Path) -> None:
    split = load_background_split(SPLIT_FILE)
    policy = _policy_config()
    policy.pretrained_path = Path(LOCAL_BASE_POLICY_REFERENCE)
    policy.vlm_model_name = LOCAL_VLM_ASSETS_REFERENCE
    arguments = _fresh_arguments(
        tmp_path,
        dataset_root=PRACTICE_ROOT / "datasets/lerobot/lerobot/libero",
        output_dir=PRACTICE_ROOT / "outputs/smolvla_libero/debug/relative-config",
    )

    resolved = build_train_config(arguments, split, policy).to_dict()

    assert resolved["dataset"]["root"] == "datasets/lerobot/lerobot/libero"
    assert resolved["output_dir"] == "outputs/smolvla_libero/debug/relative-config"
    assert resolved["policy"]["pretrained_path"] == LOCAL_BASE_POLICY_REFERENCE
    assert resolved["policy"]["vlm_model_name"] == LOCAL_VLM_ASSETS_REFERENCE


def test_update_policy_instrumentation_is_observational() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    tracker = SimpleNamespace(
        steps=5,
        samples=20,
        _batch_size=2,
        _num_frames=100,
        accelerator=SimpleNamespace(num_processes=4),
        to_dict=lambda use_avg: {
            "loss": 1.25,
            "grad_norm": 2.5,
            "lr": 1e-4,
            "dataloading_s": 0.01,
            "update_s": 0.02,
        },
    )
    expected = (tracker, {"aux": 3})

    def original(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        return expected

    module = SimpleNamespace(update_policy=original)
    stream = io.StringIO()
    sink = MetricsSink(
        stream=stream,
        attempt_id=2,
        initial_step=5,
        started_monotonic=time.perf_counter(),
    )

    with instrument_update_policy(module, sink):
        observed = module.update_policy("batch", accelerator="accelerator")

    assert observed is expected
    assert calls == [(("batch",), {"accelerator": "accelerator"})]
    assert module.update_policy is original
    record = json.loads(stream.getvalue())
    assert record["record_type"] == "train_step"
    assert record["attempt_id"] == 2
    assert record["step"] == 6
    assert record["samples"] == 28
    assert record["epochs"] == pytest.approx(0.28)
    assert record["loss"] == pytest.approx(1.25)


def test_failed_fresh_attempt_publishes_diagnostics_after_checkpoint_dir_exists(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="training failed"), _attempt_artifacts(
        output_dir=output_dir,
        attempt_id=0,
        initial_step=0,
        run_config={"experiment": "test"},
        resume=False,
        enabled=True,
    ) as sink:
        assert sink is not None
        sink.write({"record_type": "start", "step": 0})
        output_dir.mkdir()
        raise RuntimeError("training failed")

    assert (output_dir / "run_config.json").is_file()
    assert (output_dir / "train_metrics.jsonl").is_file()
    assert (output_dir / "train.log").is_file()


def test_first_checkpoint_publishes_open_run_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    for filename in ("run_config.json", "train_metrics.jsonl", "train.log"):
        (tmp_path / f".run.{filename}").write_text(filename, encoding="utf-8")
    calls: list[str] = []

    def save_checkpoint(value: str) -> str:
        calls.append(value)
        return "saved"

    module = SimpleNamespace(save_checkpoint=save_checkpoint)
    with _publish_artifacts_on_checkpoint(
        module,
        output_dir=output_dir,
        enabled=True,
    ):
        assert module.save_checkpoint("checkpoint") == "saved"

    assert calls == ["checkpoint"]
    assert module.save_checkpoint is save_checkpoint
    for filename in ("run_config.json", "train_metrics.jsonl", "train.log"):
        assert (output_dir / filename).read_text(encoding="utf-8") == filename


def _write_complete_checkpoint(
    checkpoint: Path,
    *,
    train_config: dict[str, Any],
    policy: Any,
    step: int,
) -> None:
    policy_dir = checkpoint / "pretrained_model"
    state_dir = checkpoint / "training_state"
    policy.save_pretrained(policy_dir)
    (policy_dir / "model.safetensors").touch()
    (policy_dir / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (policy_dir / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    (policy_dir / "train_config.json").write_text(
        json.dumps(train_config, default=str),
        encoding="utf-8",
    )
    state_dir.mkdir(parents=True)
    for filename in (
        "optimizer_param_groups.json",
        "optimizer_state.safetensors",
        "rng_state.safetensors",
        "scheduler_state.json",
    ):
        (state_dir / filename).touch()
    (state_dir / "training_step.json").write_text(
        json.dumps({"step": step}),
        encoding="utf-8",
    )


def test_resume_restores_latest_complete_checkpoint_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    split = BackgroundSplit(
        source=split_path,
        sha256="split-sha",
        tasks=(0,),
        train_episodes=(1, 2),
        validation_episodes=(3,),
        train_frames=20,
        validation_frames=10,
        payload={"tasks": {}},
    )
    audit = DatasetAudit(
        dataset_root=dataset_root,
        total_episodes=3,
        total_frames=30,
        train_episodes=2,
        train_frames=20,
        fps=10,
        features={},
    )
    arguments = _fresh_arguments(
        tmp_path,
        dataset_root=dataset_root,
        split_file=split_path,
    )
    policy = _policy_config()
    train_config = build_train_config(arguments, split, policy).to_dict()
    checkpoint = arguments.output_dir / "checkpoints/000002"
    _write_complete_checkpoint(
        checkpoint,
        train_config=train_config,
        policy=policy,
        step=2,
    )
    run_config = {
        "schema_version": 1,
        "experiment": "smolvla_libero_b0",
        "inputs": {
            "dataset_root": str(dataset_root),
            "split_file": str(split_path),
            "split_sha256": split.sha256,
            "base_policy": {
                "requested": "lerobot/smolvla_base",
                "source_type": "huggingface_hub",
                "resolved_revision": "a" * 40,
                "files_sha256": {
                    "config.json": "b" * 64,
                    "model.safetensors": "c" * 64,
                },
            },
        },
        "training": {
            "mode": "smoke",
            "world_size": 1,
            "per_gpu_batch_size": 1,
            "global_batch_size": 1,
            "mixed_precision": "bf16",
            "steps": 4,
            "save_freq": 2,
            "seed": 1000,
            "optimizer": train_config["optimizer"],
            "scheduler": train_config["scheduler"],
        },
    }
    (arguments.output_dir / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )
    (arguments.output_dir / "train_metrics.jsonl").write_text(
        json.dumps({"record_type": "train_step", "attempt_id": 0, "step": 2})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(b0_training, "load_background_split", lambda _: split)
    monkeypatch.setattr(b0_training, "validate_dataset", lambda *_: audit)

    context = validate_resume(
        ResumeArguments(
            resume_from=checkpoint.resolve(),
            num_processes=1,
            mixed_precision="bf16",
        )
    )

    assert context.saved_step == 2
    assert context.attempt_id == 1
    assert context.previous_metrics_tail_step == 2
    assert context.config_path == checkpoint / "pretrained_model/train_config.json"


def test_resume_rebases_legacy_paths_and_keeps_new_checkpoint_paths_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from lerobot.configs.train import TrainPipelineConfig

    config = SimpleNamespace(
        policy=SimpleNamespace(
            vlm_model_name=(
                "/remote/practice/policies/smolvla_or_pi0/"
                "smolvla_base/vlm_assets"
            ),
            pretrained_path=(
                "/remote/practice/policies/smolvla_or_pi0/smolvla_base"
            ),
        ),
        dataset=SimpleNamespace(
            root="/remote/practice/datasets/lerobot/lerobot/libero"
        ),
        output_dir=Path("/remote/practice/outputs/smolvla_libero/run"),
        resume=False,
    )
    run_dir = PRACTICE_ROOT / "outputs/smolvla_libero/run"
    config_path = (
        run_dir / "checkpoints/005000/pretrained_model/train_config.json"
    )
    context = SimpleNamespace(
        config_path=config_path,
        run_dir=run_dir,
        attempt_id=1,
        saved_step=5000,
        dataset_audit=SimpleNamespace(
            dataset_root=PRACTICE_ROOT / "datasets/lerobot/lerobot/libero"
        ),
    )
    monkeypatch.setattr(train_b0_module, "validate_resume", lambda _: context)
    monkeypatch.setattr(
        TrainPipelineConfig,
        "from_pretrained",
        lambda _: config,
    )
    monkeypatch.setattr(
        train_b0_module,
        "validate_local_vlm_assets",
        lambda _: PRACTICE_ROOT
        / "policies/smolvla_or_pi0/smolvla_base/vlm_assets",
    )

    (
        resolved,
        output_dir,
        attempt_id,
        saved_step,
        run_config,
        resume_config_path,
        returned_context,
    ) = _resume_training(
        ResumeArguments(
            resume_from=tmp_path / "checkpoint",
            num_processes=1,
            mixed_precision="bf16",
        )
    )

    assert resolved.policy.vlm_model_name == LOCAL_VLM_ASSETS_REFERENCE
    assert resolved.policy.pretrained_path == Path(LOCAL_BASE_POLICY_REFERENCE)
    assert resolved.dataset.root == "datasets/lerobot/lerobot/libero"
    assert resolved.output_dir == Path("outputs/smolvla_libero/run")
    assert resolved.resume is True
    assert output_dir == run_dir
    assert attempt_id == 1
    assert saved_step == 5000
    assert run_config is None
    assert resume_config_path == Path(
        "outputs/smolvla_libero/run/checkpoints/005000/"
        "pretrained_model/train_config.json"
    )
    assert returned_context is context


def test_shell_prints_pinned_formal_accelerate_command(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--mode=formal",
            "--dataset-root=datasets/lerobot/lerobot/libero",
            f"--split-file={SPLIT_FILE}",
            "--num-processes=8",
            "--batch-size=8",
            "--steps=30000",
            "--save-freq=5000",
            "--learning-rate=0.0001",
            "--train-seed=1000",
            "--num-workers=4",
            "--mixed-precision=bf16",
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
    assert command[:4] == [
        "env",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
    ]
    assert command[4:10] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "robot-practice",
        "accelerate",
    ]
    assert "--mixed_precision=bf16" in command
    assert "--num_processes=8" in command
    assert "--num_machines=1" in command
    assert "--multi_gpu" in command
    assert "--gpu_ids=all" in command
    assert "--dynamo_backend=no" in command
    assert "--main_process_port=0" in command
    assert "-m" in command
    assert "robot_practice.experiments.smolvla_libero.train_b0" in command


def test_shell_resume_restores_saved_world_size_and_bf16(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/005000"
    checkpoint.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "smolvla_libero_b0",
                "training": {
                    "world_size": 8,
                    "mixed_precision": "bf16",
                },
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            f"--resume-from={checkpoint}",
            "--print-command",
        ],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "--num_processes=8" in command
    assert "--mixed_precision=bf16" in command
    assert "--multi_gpu" in command
    assert f"--resume-from={checkpoint}" in command
    assert "--num-processes=8" in command
    assert "--mixed-precision=bf16" in command
