from __future__ import annotations

import subprocess
from pathlib import Path

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PRACTICE_ROOT / "scripts/experiments/maniskill/train_formal.sh"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_m0_formal_launcher_builds_explicit_eight_gpu_accelerate_command() -> None:
    result = _run("--num-gpus=8", "--print-command")

    assert result.returncode == 0, result.stderr
    assert "accelerate launch" in result.stdout
    assert "--multi_gpu" in result.stdout
    assert "--num_processes=8" in result.stdout
    assert "-m lerobot.scripts.lerobot_train" in result.stdout
    assert "--batch_size=8" in result.stdout
    assert "--dataset.repo_id=local/maniskill_pickcube_rl_7d_full" in result.stdout
    assert "--output_dir=outputs/maniskill/m0_probe/train_50k" in result.stdout
    assert "--job_name=m0_pickcube_50k" in result.stdout
    assert "--steps=50000" in result.stdout
    assert "--save_freq=5000" in result.stdout


def test_m0_formal_launcher_accepts_singular_num_gpu_alias_and_trim_dataset() -> None:
    result = _run(
        "--num-gpu",
        "4",
        "--dataset-variant=trim",
        "--batch-size=16",
        "--steps=10",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    assert "--num_processes=4" in result.stdout
    assert "--dataset.root=datasets/lerobot/maniskill/pickcube_rl_7d_trim" in result.stdout
    assert "--batch_size=16" in result.stdout
    assert "--steps=10" in result.stdout


def test_m0_formal_launcher_requires_num_gpus() -> None:
    result = _run("--print-command")

    assert result.returncode == 64
    assert "--num-gpus=N is required exactly once" in result.stderr


def test_m0_formal_launcher_rejects_duplicate_gpu_count() -> None:
    result = _run("--num-gpus=8", "--num-gpu=8", "--print-command")

    assert result.returncode == 64
    assert "duplicate --num-gpus/--num-gpu" in result.stderr
