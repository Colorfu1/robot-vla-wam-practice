from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from robot_practice.experiments.act_delta_checkpoint import (
    CHECKPOINT_DIR,
    create_act_delta_checkpoint,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
A2_LAUNCHER = PRACTICE_ROOT / "scripts" / "experiments" / "act" / "a2_delta_loss.sh"


def _act_checkpoint(path: Path) -> Path:
    config = ACTConfig(
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (8,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        device="cpu",
        chunk_size=20,
        n_action_steps=5,
        use_vae=False,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )
    config.save_pretrained(path)
    (path / "model.safetensors").write_bytes(b"weights")
    processor = {
        "name": "test",
        "steps": [],
    }
    (path / "policy_preprocessor.json").write_text(json.dumps(processor), encoding="utf-8")
    (path / "policy_postprocessor.json").write_text(json.dumps(processor), encoding="utf-8")
    return path


def test_create_act_delta_checkpoint_updates_config(tmp_path: Path) -> None:
    source = _act_checkpoint(tmp_path / "source")
    output_dir = tmp_path / "a2"
    converted = create_act_delta_checkpoint(
        source_policy=source,
        output_dir=output_dir,
        delta_loss_weight=0.1,
    )

    expected_output = (output_dir / CHECKPOINT_DIR).resolve()
    assert converted == expected_output
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    target_config = json.loads((converted / "config.json").read_text(encoding="utf-8"))
    assert source_config["type"] == "act"
    assert target_config["type"] == "act_delta"
    assert target_config["delta_loss_weight"] == 0.1
    assert target_config["chunk_size"] == 20
    assert target_config["n_action_steps"] == 5
    assert (converted / "model.safetensors").exists()


def test_create_act_delta_checkpoint_rejects_existing_output_directory(tmp_path: Path) -> None:
    source = _act_checkpoint(tmp_path / "source")
    output_root = tmp_path / "already_exists"
    output_root.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        create_act_delta_checkpoint(
            source_policy=source,
            output_dir=output_root,
            delta_loss_weight=0.1,
        )


def test_create_act_delta_checkpoint_requires_existing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        create_act_delta_checkpoint(
            source_policy=tmp_path / "missing",
            output_dir=tmp_path / "out",
            delta_loss_weight=0.1,
        )


def test_a2_launcher_keeps_training_output_absent_until_lerobot_starts(
    tmp_path: Path,
) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    fake_conda = tool_dir / "conda"
    fake_conda.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
for argument in "$@"; do
  case "${argument}" in
    --output_dir=*)
      output_dir="${argument#--output_dir=}"
      if [[ -e "${output_dir}" ]]; then
        echo "training output already exists before lerobot-train" >&2
        exit 42
      fi
      mkdir -p -- "${output_dir}"
      ;;
  esac
done
""",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)
    output_dir = tmp_path / "run"
    env = {
        **os.environ,
        "PATH": f"{tool_dir}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        [
            str(A2_LAUNCHER),
            "--steps=1",
            f"--output-dir={output_dir}",
        ],
        cwd=PRACTICE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_dir.is_dir()
    assert not (output_dir / ".launch.lock").exists()
    assert (tmp_path / ".run.launch.lock").is_file()


def test_a2_launcher_print_command_configures_component_loss_log(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    result = subprocess.run(
        [
            str(A2_LAUNCHER),
            "--steps=20000",
            f"--output-dir={output_dir}",
            "--print-command",
        ],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "ROBOT_PRACTICE_A2_LOSS_LOG_PATH="
        f"{output_dir}/train_loss_components.jsonl"
    ) in result.stdout
    assert "ROBOT_PRACTICE_A2_LOSS_LOG_FREQ=100" in result.stdout
