from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from robot_practice.experiments.act_initialization import EPISODES

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PRACTICE_ROOT / "scripts" / "experiments" / "act"
DATASET_ROOT = PRACTICE_ROOT / "datasets" / "lerobot" / "lerobot" / "libero"
SCRIPTS = {
    "obs_equal": SCRIPT_ROOT / "a1_observation_equal.sh",
    "vae": SCRIPT_ROOT / "a1_vae.sh",
}


@pytest.fixture(scope="module", autouse=True)
def _portable_shell_preflight(tmp_path_factory: pytest.TempPathFactory):
    tool_dir = tmp_path_factory.mktemp("act_followup_tools")
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


@pytest.mark.parametrize("variant", ["obs_equal", "vae"])
def test_launcher_prints_complete_shared_a1_command(variant: str, tmp_path: Path) -> None:
    output_dir = tmp_path / variant
    result = _run(
        variant,
        "--steps=17",
        f"--output-dir={output_dir}",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    arguments = shlex.split(result.stdout)
    episodes = ",".join(str(value) for value in EPISODES)
    assert "lerobot-train" in arguments
    assert "--dataset.repo_id=lerobot/libero" in arguments
    assert f"--dataset.root={DATASET_ROOT}" in arguments
    assert f"--dataset.episodes=[{episodes}]" in arguments
    assert "--policy.chunk_size=20" in arguments
    assert "--policy.n_action_steps=5" in arguments
    assert "--policy.push_to_hub=false" in arguments
    assert "--batch_size=8" in arguments
    assert "--steps=17" in arguments
    assert "--seed=1000" in arguments
    assert "--eval_freq=0" in arguments
    assert "--log_freq=17" in arguments
    assert "--save_checkpoint=true" in arguments
    assert "--save_freq=17" in arguments
    assert "--wandb.enable=false" in arguments
    assert f"--output_dir={output_dir}" in arguments
    assert not output_dir.exists()


def test_observation_equal_launcher_has_only_its_controlled_policy_change(
    tmp_path: Path,
) -> None:
    result = _run(
        "obs_equal",
        "--steps=20000",
        f"--output-dir={tmp_path / 'obs_equal'}",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    arguments = shlex.split(result.stdout)
    assert "--policy.type=act_obs_equal" in arguments
    assert "--policy.use_vae=false" in arguments
    assert (
        "--policy.discover_packages_path=robot_practice.policies.act_obs_equal"
        in arguments
    )
    assert "--log_freq=100" in arguments
    assert "--save_freq=20000" in arguments
    assert not any("delta_loss" in argument for argument in arguments)


def test_vae_launcher_has_only_standard_act_vae_change(tmp_path: Path) -> None:
    result = _run(
        "vae",
        "--steps=20000",
        f"--output-dir={tmp_path / 'vae'}",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    arguments = shlex.split(result.stdout)
    assert "--policy.type=act" in arguments
    assert "--policy.use_vae=true" in arguments
    assert "--policy.latent_dim=32" in arguments
    assert "--policy.kl_weight=10.0" in arguments
    assert "--log_freq=100" in arguments
    assert "--save_freq=20000" in arguments
    assert not any("discover_packages_path" in argument for argument in arguments)
    assert not any("delta_loss" in argument for argument in arguments)


@pytest.mark.parametrize("variant", ["obs_equal", "vae"])
@pytest.mark.parametrize(
    "arguments",
    [
        ("--steps=0",),
        ("--steps=00",),
        ("--steps=-1",),
        ("--steps=2147483648",),
        ("--steps=1", "--steps=2"),
        ("--output-dir=one", "--output-dir=two"),
        ("--print-command", "--print-command"),
        ("--unknown",),
    ],
)
def test_launchers_reject_invalid_arguments(
    variant: str,
    arguments: tuple[str, ...],
) -> None:
    result = _run(variant, *arguments, "--print-command")
    assert result.returncode == 64


@pytest.mark.parametrize("variant", ["obs_equal", "vae"])
def test_launcher_help_explains_controlled_change(variant: str) -> None:
    result = _run(variant, "--help")
    assert result.returncode == 0
    assert "chunk_size=20" in result.stdout
    if variant == "obs_equal":
        assert "normalized independently for each sampled observation" in result.stdout
    else:
        assert "latent_dim=32" in result.stdout
        assert "kl_weight=10.0" in result.stdout
