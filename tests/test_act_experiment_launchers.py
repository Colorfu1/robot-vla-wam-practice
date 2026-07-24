from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig

from robot_practice.experiments.act_initialization import (
    ACTION_SHAPE,
    BACKBONE_WEIGHTS,
    CAMERA_KEYS,
    EPISODES,
    IMAGE_SHAPE,
    SEED,
    STATE_SHAPE,
    build_argument_parser,
    build_step_zero_train_config,
    validate_policy_contract,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PRACTICE_ROOT / "scripts" / "experiments" / "act"
PUBLIC_SCRIPT = SCRIPT_ROOT / "action_chunking.sh"
DATASET_ROOT = PRACTICE_ROOT / "datasets" / "lerobot" / "lerobot" / "libero"
SCRIPTS = {
    "a0": SCRIPT_ROOT / "a0_single_step.sh",
    "a1": SCRIPT_ROOT / "a1_chunking.sh",
}
EXPECTED = {
    "a0": ("1", "1", "a0_single_step"),
    "a1": ("20", "5", "a1_chunking"),
}


@pytest.fixture(scope="module", autouse=True)
def _portable_shell_preflight(tmp_path_factory: pytest.TempPathFactory):
    tool_dir = tmp_path_factory.mktemp("act_comparison_tools")
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


def _run(variant: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPTS[variant]), *arguments],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_public(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PUBLIC_SCRIPT), *arguments],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _compatible_config(variant: str) -> ACTConfig:
    chunk_size, n_action_steps, _ = EXPECTED[variant]
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
        chunk_size=int(chunk_size),
        n_action_steps=int(n_action_steps),
        use_vae=False,
        pretrained_backbone_weights=BACKBONE_WEIGHTS,
        push_to_hub=False,
    )


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_step_zero_config_matches_fixed_contract(variant: str, tmp_path: Path) -> None:
    config = build_step_zero_train_config(
        variant=variant,
        output_dir=tmp_path / variant,
        dataset_root=DATASET_ROOT,
        device="cpu",
    )
    chunk_size, n_action_steps, _ = EXPECTED[variant]
    assert config.steps == 0
    assert config.seed == SEED
    assert config.batch_size == 8
    assert config.dataset.episodes == list(EPISODES)
    assert config.dataset.use_imagenet_stats is True
    assert isinstance(config.policy, ACTConfig)
    assert config.policy.chunk_size == int(chunk_size)
    assert config.policy.n_action_steps == int(n_action_steps)
    assert config.policy.use_vae is False


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_policy_contract_accepts_only_matching_variant(variant: str) -> None:
    validate_policy_contract(_compatible_config(variant), variant=variant)
    other = "a1" if variant == "a0" else "a0"
    with pytest.raises(ValueError, match="Incompatible"):
        validate_policy_contract(_compatible_config(variant), variant=other)


def test_python_cli_requires_explicit_variant(tmp_path: Path) -> None:
    parser = build_argument_parser()
    create = parser.parse_args(
        [
            "create",
            "--variant=a0",
            f"--output-dir={tmp_path / 'run'}",
            "--device=cpu",
        ]
    )
    validate = parser.parse_args(
        ["validate-policy", "--variant=a1", f"--policy-path={tmp_path}"]
    )
    assert create.variant == "a0"
    assert validate.variant == "a1"


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_step_zero_prints_variant_initializer(variant: str) -> None:
    result = _run(variant, "--steps=0", "--print-command")
    assert result.returncode == 0, result.stderr
    assert "robot_practice.experiments.act_initialization" in result.stdout
    assert f"--variant={variant}" in result.stdout
    assert "lerobot-train" not in result.stdout


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_public_entry_matches_thin_variant_alias(variant: str) -> None:
    public = _run_public(f"--variant={variant}", "--steps=17", "--print-command")
    alias = _run(variant, "--steps=17", "--print-command")
    assert public.returncode == 0, public.stderr
    assert alias.returncode == 0, alias.stderr
    public_arguments = [
        value
        for value in shlex.split(public.stdout)
        if not value.startswith("--output_dir=")
    ]
    alias_arguments = [
        value
        for value in shlex.split(alias.stdout)
        if not value.startswith("--output_dir=")
    ]
    assert public_arguments == alias_arguments


def test_public_entry_requires_exactly_one_variant() -> None:
    missing = _run_public("--steps=17", "--print-command")
    duplicate = _run_public(
        "--variant=a0",
        "--variant=a1",
        "--steps=17",
        "--print-command",
    )
    unknown = _run_public("--variant=a2", "--steps=17", "--print-command")
    assert missing.returncode == 64
    assert duplicate.returncode == 64
    assert unknown.returncode == 64


def test_public_help_distinguishes_training_and_inference_horizons() -> None:
    result = _run_public("--help")
    assert result.returncode == 0
    assert "training chunk_size=20" in result.stdout
    assert "inference default n_action_steps=5" in result.stdout
    script = PUBLIC_SCRIPT.read_text(encoding="utf-8")
    assert "Training supervision horizon:" in script
    assert "not used by training loss" in script


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_alias_rejects_user_variant_override(variant: str) -> None:
    result = _run(variant, "--variant=a1", "--print-command")
    assert result.returncode == 64
    assert "specified only once" in result.stderr


def test_obsolete_internal_launcher_is_removed() -> None:
    assert not (SCRIPT_ROOT / "run_action_chunking_variant.sh").exists()


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_training_command_has_exact_shared_data_and_variant(variant: str) -> None:
    result = _run(variant, "--steps=20000", "--print-command")
    assert result.returncode == 0, result.stderr
    chunk_size, n_action_steps, output_name = EXPECTED[variant]
    episodes = ",".join(str(value) for value in EPISODES)
    arguments = shlex.split(result.stdout)
    assert "lerobot-train" in result.stdout
    assert f"--dataset.episodes=[{episodes}]" in arguments
    assert f"--policy.chunk_size={chunk_size}" in result.stdout
    assert f"--policy.n_action_steps={n_action_steps}" in result.stdout
    assert "--policy.use_vae=false" in result.stdout
    assert "--batch_size=8" in result.stdout
    assert "--steps=20000" in result.stdout
    assert "--seed=1000" in result.stdout
    assert f"outputs/act/{output_name}" in result.stdout


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_short_training_uses_short_log_and_save_frequency(variant: str) -> None:
    result = _run(variant, "--steps=17", "--print-command")
    assert result.returncode == 0, result.stderr
    assert "--log_freq=17" in result.stdout
    assert "--save_freq=17" in result.stdout


@pytest.mark.parametrize("variant", ["a0", "a1"])
@pytest.mark.parametrize(
    "arguments",
    [
        ("--steps=1", "--steps=2"),
        ("--policy-path=one", "--policy-path=two"),
        ("--print-command", "--print-command"),
        ("--unknown",),
        ("--steps=00",),
        ("--steps=-1",),
        ("--steps=2147483648",),
    ],
)
def test_launchers_reject_invalid_arguments(
    variant: str,
    arguments: tuple[str, ...],
) -> None:
    result = _run(variant, *arguments, "--print-command")
    assert result.returncode == 64


@pytest.mark.parametrize("variant", ["a0", "a1"])
def test_launchers_reject_policy_path_for_step_zero(
    variant: str,
    tmp_path: Path,
) -> None:
    result = _run(
        variant,
        "--steps=0",
        f"--policy-path={tmp_path}",
        "--print-command",
    )
    assert result.returncode == 64
