from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig

from robot_practice.experiments.act_a_sanity import (
    ACTION_SHAPE,
    BACKBONE_WEIGHTS,
    CAMERA_KEYS,
    CHUNK_SIZE,
    EPISODES,
    IMAGE_SHAPE,
    N_ACTION_STEPS,
    SEED,
    STATE_SHAPE,
    build_argument_parser,
    build_step_zero_train_config,
    create_initialization_checkpoint,
    validate_a_sanity_policy_path,
    validate_policy_contract,
    verify_pretrained_backbone,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PRACTICE_ROOT / "scripts" / "experiments" / "act" / "a_sanity.sh"
DATASET_ROOT = PRACTICE_ROOT / "datasets" / "lerobot" / "lerobot" / "libero"


@pytest.fixture(scope="module", autouse=True)
def _portable_shell_preflight(tmp_path_factory: pytest.TempPathFactory):
    """Keep launcher parsing tests independent of Conda and ignored local data."""

    tool_dir = tmp_path_factory.mktemp("a_sanity_tools")
    for name in ("conda", "flock"):
        executable = tool_dir / name
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    patch = pytest.MonkeyPatch()
    patch.setenv("PATH", f"{tool_dir}{os.pathsep}{os.environ['PATH']}")

    metadata_path = DATASET_ROOT / "meta" / "info.json"
    created_metadata = not metadata_path.exists()
    if created_metadata:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text("{}\n", encoding="utf-8")

    yield

    patch.undo()
    if created_metadata:
        metadata_path.unlink(missing_ok=True)
        for directory in (metadata_path.parent, DATASET_ROOT, DATASET_ROOT.parent):
            try:
                directory.rmdir()
            except OSError:
                break


def _compatible_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            CAMERA_KEYS[0]: PolicyFeature(FeatureType.VISUAL, IMAGE_SHAPE),
            CAMERA_KEYS[1]: PolicyFeature(FeatureType.VISUAL, IMAGE_SHAPE),
            "observation.state": PolicyFeature(FeatureType.STATE, STATE_SHAPE),
        },
        output_features={
            "action": PolicyFeature(FeatureType.ACTION, ACTION_SHAPE),
        },
        device="cpu",
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        use_vae=False,
        pretrained_backbone_weights=BACKBONE_WEIGHTS,
        push_to_hub=False,
    )


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    _compatible_config().save_pretrained(path)
    (path / "model.safetensors").write_bytes(b"shape validation is config-only")
    empty_pipeline = json.dumps({"name": "test", "steps": []})
    (path / "policy_preprocessor.json").write_text(empty_pipeline, encoding="utf-8")
    (path / "policy_postprocessor.json").write_text(empty_pipeline, encoding="utf-8")
    return path


def _run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *arguments],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_step_zero_config_matches_fixed_training_contract(tmp_path: Path) -> None:
    config = build_step_zero_train_config(
        output_dir=tmp_path / "run",
        dataset_root=DATASET_ROOT,
        device="cpu",
    )
    assert config.steps == 0
    assert config.seed == SEED
    assert config.batch_size == 8
    assert config.dataset.repo_id == "lerobot/libero"
    assert config.dataset.episodes == list(EPISODES)
    assert config.dataset.use_imagenet_stats is True
    assert isinstance(config.policy, ACTConfig)
    assert config.policy.chunk_size == CHUNK_SIZE
    assert config.policy.n_action_steps == N_ACTION_STEPS
    assert config.policy.use_vae is False
    assert config.policy.push_to_hub is False


def test_policy_contract_accepts_exact_libero_act() -> None:
    validate_policy_contract(_compatible_config())


def test_policy_contract_reports_all_interface_mismatches() -> None:
    config = _compatible_config()
    config.chunk_size = 21
    config.output_features = {
        "action": PolicyFeature(FeatureType.ACTION, (14,)),
    }
    with pytest.raises(ValueError, match="chunk_size=21") as error:
        validate_policy_contract(config)
    assert "action shape=(14,)" in str(error.value)


def test_complete_compatible_policy_path_is_accepted(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "pretrained_model")
    assert validate_a_sanity_policy_path(checkpoint) == checkpoint.resolve()


def test_pretrained_backbone_comparison_uses_every_actual_tensor() -> None:
    state = {
        "conv.weight": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2),
        "bn.running_mean": torch.tensor([0.0]),
    }
    policy = SimpleNamespace(
        model=SimpleNamespace(backbone=SimpleNamespace(state_dict=lambda: state))
    )
    assert (
        verify_pretrained_backbone(
            policy,
            reference_state={**state, "unused": torch.ones(1)},
        )
        == 2
    )

    wrong = {**state, "conv.weight": torch.zeros(1, 1, 2, 2)}
    with pytest.raises(ValueError, match="conv.weight"):
        verify_pretrained_backbone(policy, reference_state=wrong)


def test_python_cli_has_explicit_create_and_validate_commands(tmp_path: Path) -> None:
    parser = build_argument_parser()
    create = parser.parse_args(
        ["create", "--output-dir", str(tmp_path / "run"), "--device", "cpu"]
    )
    validate = parser.parse_args(
        ["validate-policy", "--policy-path", str(tmp_path / "checkpoint")]
    )
    assert create.command == "create"
    assert create.dataset_root == DATASET_ROOT
    assert validate.command == "validate-policy"


def test_shell_zero_prints_checkpoint_creation_without_mutation() -> None:
    result = _run_script("--steps=0", "--print-command")
    assert result.returncode == 0, result.stderr
    assert "robot_practice.experiments.act_a_sanity create" in result.stdout
    assert "lerobot-train" not in result.stdout
    assert "--output-dir=" in result.stdout


def test_shell_default_preserves_20000_step_training() -> None:
    result = _run_script("--print-command")
    assert result.returncode == 0, result.stderr
    assert "lerobot-train" in result.stdout
    assert "--steps=20000" in result.stdout
    assert "--save_freq=20000" in result.stdout
    assert "--policy.type=act" in result.stdout


def test_shell_short_run_reduces_log_frequency() -> None:
    result = _run_script("--print-command", "--steps=17")
    assert result.returncode == 0, result.stderr
    assert "--steps=17" in result.stdout
    assert "--log_freq=17" in result.stdout
    assert "--save_freq=17" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ("--steps=1", "--steps=2"),
        ("--policy-path=one", "--policy-path=two"),
        ("--print-command", "--print-command"),
        ("--unknown",),
    ],
)
def test_shell_rejects_duplicate_or_unknown_options(arguments: tuple[str, ...]) -> None:
    result = _run_script(*arguments)
    assert result.returncode == 64


@pytest.mark.parametrize(
    "argument",
    [
        "--steps=",
        "--steps=-1",
        "--steps=00",
        "--steps=08",
        "--steps=020000",
        "--steps=2147483648",
        "--steps=999999999999999999999999999999",
    ],
)
def test_shell_rejects_noncanonical_or_oversized_steps(argument: str) -> None:
    result = _run_script(argument, "--print-command")
    assert result.returncode == 64
    assert "--steps" in result.stderr


def test_shell_rejects_policy_path_for_step_zero(tmp_path: Path) -> None:
    result = _run_script(
        "--steps=0",
        f"--policy-path={tmp_path}",
        "--print-command",
    )
    assert result.returncode == 64
    assert "cannot be used with --steps=0" in result.stderr


def test_shell_canonicalizes_and_quotes_checkpoint_path_with_spaces(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "policy with spaces" / "pretrained_model")
    result = _run_script(
        "--steps=3",
        f"--policy-path={checkpoint}",
        "--print-command",
    )
    assert result.returncode == 0, result.stderr
    assert "--policy.path=" in result.stdout
    assert "policy\\ with\\ spaces" in result.stdout
    assert "--policy.type=act" not in result.stdout
    assert "--policy.push_to_hub=false" in result.stdout


@pytest.mark.skipif(
    os.environ.get("RUN_ACT_EVAL_INTEGRATION") != "1",
    reason="set RUN_ACT_EVAL_INTEGRATION=1 for the real dataset/model round trip",
)
def test_real_step_zero_checkpoint_round_trip(tmp_path: Path) -> None:
    result = create_initialization_checkpoint(
        output_dir=tmp_path / "run",
        dataset_root=DATASET_ROOT,
        device="cpu",
    )
    checkpoint = validate_a_sanity_policy_path(result.policy_path)
    manifest = json.loads(
        (result.run_root / "initialization_manifest.json").read_text(encoding="utf-8")
    )

    digest = hashlib.sha256()
    with (checkpoint / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    assert not (result.run_root / ".incomplete").exists()
    assert manifest["status"] == "complete"
    assert manifest["optimizer_updates"] == 0
    assert manifest["dataset"]["episodes"] == list(EPISODES)
    assert manifest["dataset"]["num_frames"] == 859
    assert manifest["checkpoint"]["path"] == "checkpoints/000000/pretrained_model"
    assert manifest["checkpoint"]["model_sha256"] == digest.hexdigest()
    assert result.model_sha256 == digest.hexdigest()
