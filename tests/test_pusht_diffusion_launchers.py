from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PRACTICE_ROOT / "scripts" / "experiments" / "pusht"
MODULE_ROOT = PRACTICE_ROOT / "src" / "robot_practice" / "experiments" / "pusht_diffusion"


def _run(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT_ROOT / name), *arguments],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _training_plan(result: subprocess.CompletedProcess[str]) -> tuple[list[str], dict]:
    command_marker = "Command:\n"
    config_marker = "\nResolved configuration:\n"
    assert result.stdout.startswith(command_marker)
    command_text, config_text = result.stdout[len(command_marker) :].split(
        config_marker,
        maxsplit=1,
    )
    return shlex.split(command_text), json.loads(config_text)


@pytest.mark.parametrize(
    ("name", "arguments", "module"),
    [
        ("check_environment.sh", (), "interface_checks"),
        ("benchmark_pretrained.sh", (), "pretrained_benchmark"),
        (
            "inspect_demonstrations.sh",
            ("--num-episodes=5", "--num-candidates=2"),
            "data_inspection",
        ),
        (
            "select_checkpoints.sh",
            ("--bc-run=bc", "--diffusion-run=diff"),
            "checkpoint_selection",
        ),
        (
            "eval_fixed_observations.sh",
            ("--bc-policy=bc", "--diffusion-policy=diff"),
            "fixed_observation_eval",
        ),
        (
            "eval_multi_seed_rollouts.sh",
            ("--bc-policy=bc", "--diffusion-policy=diff"),
            "rollout_eval",
        ),
        (
            "benchmark_sampling_steps.sh",
            ("--diffusion-policy=diff",),
            "sampling_steps_eval",
        ),
        (
            "make_report.sh",
            ("--e0=e0", "--e1=e1", "--e2=e2", "--e4=e4"),
            "report",
        ),
    ],
)
def test_eval_launchers_print_direct_python_commands(
    name: str,
    arguments: tuple[str, ...],
    module: str,
) -> None:
    result = _run(name, *arguments, "--print-command")

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:6] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "robot-practice",
        "python",
    ]
    assert f"robot_practice.experiments.pusht_diffusion.{module}" in command


def test_smoke_eval_requires_candidates_before_loading_policies(tmp_path: Path) -> None:
    missing = tmp_path / "candidates.json"

    result = _run(
        "smoke_eval.sh",
        "--bc-policy=bc",
        "--diffusion-policy=diff",
        f"--candidates={missing}",
        "--print-command",
    )

    assert result.returncode == 66
    assert f"candidates file does not exist: {missing}" in result.stderr
    assert "inspect_demonstrations.sh" in result.stderr
    assert "--num-episodes=50" in result.stderr
    assert "--num-candidates=10" in result.stderr
    assert "conda run" not in result.stdout


def test_smoke_eval_prints_commands_when_candidates_exist(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text("{}\n", encoding="utf-8")

    result = _run(
        "smoke_eval.sh",
        "--bc-policy=bc",
        "--diffusion-policy=diff",
        f"--candidates={candidates}",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    assert (
        "robot_practice.experiments.pusht_diffusion.fixed_observation_eval" in commands[0]
    )
    assert "robot_practice.experiments.pusht_diffusion.rollout_eval" in commands[1]


def test_main_guide_exposes_required_candidate_generation_command() -> None:
    guide = (
        PRACTICE_ROOT / "reports" / "guides" / "pusht_diffusion" / "README.md"
    ).read_text(encoding="utf-8")

    assert "bash scripts/experiments/pusht/inspect_demonstrations.sh" in guide
    assert "--num-episodes=50" in guide
    assert "--num-candidates=10" in guide
    assert "--output-dir=outputs/pusht_diffusion/data_inspection" in guide


@pytest.mark.parametrize(
    ("name", "policy_type", "module", "config_module"),
    [
        (
            "train_bc_chunk.sh",
            "bc_chunk",
            "train_bc_chunk",
            "model_configs.bc_chunk",
        ),
        (
            "train_diffusion.sh",
            "diffusion",
            "train_diffusion",
            "model_configs.diffusion",
        ),
    ],
)
def test_training_launchers_expose_explicit_python_config(
    name: str,
    policy_type: str,
    module: str,
    config_module: str,
    tmp_path: Path,
) -> None:
    output = tmp_path / policy_type
    result = _run(
        name,
        "--mode=smoke",
        "--steps=17",
        "--device=cpu",
        f"--output-dir={output}",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command, resolved = _training_plan(result)
    python_index = command.index("python")
    assert command[python_index : python_index + 3] == [
        "python",
        "-m",
        f"robot_practice.experiments.pusht_diffusion.{module}",
    ]
    assert "--steps=17" in command
    assert f"--output-dir={output}" in command
    assert resolved["entry_module"].endswith(module)
    assert resolved["model"]["type"] == policy_type
    assert resolved["model"]["config_module"].endswith(config_module)
    assert resolved["model"]["horizon"] == 16
    assert resolved["model"]["n_action_steps"] == 8
    assert resolved["dataset"]["episodes"] == [205, 74, 105, 73, 42, 57, 134, 115]
    assert resolved["training"]["batch_size"] == 8
    assert resolved["training"]["steps"] == 17
    assert resolved["training"]["save_freq"] == 17
    assert resolved["training"]["initialization"].endswith(
        f"/initialization/smoke/seed_1000/{policy_type}"
    )
    assert not output.exists()


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("train_bc_chunk.sh", "train_bc_chunk"),
        ("train_diffusion.sh", "train_diffusion"),
    ],
)
def test_training_bash_is_only_an_environment_forwarder(name: str, module: str) -> None:
    text = (SCRIPT_ROOT / name).read_text(encoding="utf-8")

    assert text.count("\n") <= 6
    assert "exec conda run --no-capture-output -n robot-practice" in text
    assert f"robot_practice.experiments.pusht_diffusion.{module}" in text
    assert '"$@"' in text
    assert "case " not in text
    assert "dataset." not in text
    assert "policy." not in text


def test_bc_chunk_entry_registers_policy_before_lerobot_training() -> None:
    code = (
        "import robot_practice.experiments.pusht_diffusion.train_bc_chunk\n"
        "from lerobot.configs.policies import PreTrainedConfig\n"
        "config = PreTrainedConfig.get_choice_class('bc_chunk')\n"
        "assert config.__name__ == 'BCChunkConfig'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_model_specific_entries_directly_call_lerobot_training() -> None:
    initialization = (MODULE_ROOT / "initialization.py").read_text(encoding="utf-8")
    training_common = (MODULE_ROOT / "training_common.py").read_text(encoding="utf-8")
    entries = [
        (MODULE_ROOT / "train_bc_chunk.py").read_text(encoding="utf-8"),
        (MODULE_ROOT / "train_diffusion.py").read_text(encoding="utf-8"),
    ]

    assert "from .model_configs.bc_chunk import make_bc_chunk_config" in initialization
    assert "from .model_configs.diffusion import make_diffusion_config" in initialization
    assert "bc_chunk.json" not in initialization
    assert "diffusion.json" not in initialization
    assert "lerobot_train" not in training_common
    assert not (MODULE_ROOT / "train_config.py").exists()
    for entry in entries:
        assert "from lerobot.scripts import lerobot_train" in entry
        assert "from .training_common import" in entry
        assert "lerobot_train.train(resolved.config)" in entry
    assert not (
        PRACTICE_ROOT / "configs" / "experiments" / "pusht_diffusion" / "bc_chunk.json"
    ).exists()
    assert not (
        PRACTICE_ROOT / "configs" / "experiments" / "pusht_diffusion" / "diffusion.json"
    ).exists()


def test_vscode_debug_uses_the_same_model_specific_modules_as_bash() -> None:
    launch = json.loads((PRACTICE_ROOT / ".vscode" / "launch.json").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in launch["configurations"]}

    bc = by_name["PushT: BCChunk train (3-step debug)"]
    diffusion = by_name["PushT: Diffusion train (3-step debug)"]
    assert bc["module"].endswith("pusht_diffusion.train_bc_chunk")
    assert diffusion["module"].endswith("pusht_diffusion.train_diffusion")
    for config in (bc, diffusion):
        assert "--mode=smoke" in config["args"]
        assert "--steps=3" in config["args"]
        assert not any(argument.startswith("--dataset.") for argument in config["args"])
        assert not any(argument.startswith("--policy.") for argument in config["args"])


def test_pilot_training_defaults_are_full_split_and_batch_64(tmp_path: Path) -> None:
    result = _run(
        "train_diffusion.sh",
        "--mode=pilot",
        "--device=cpu",
        f"--output-dir={tmp_path / 'pilot'}",
        "--print-command",
    )
    assert result.returncode == 0, result.stderr
    _, resolved = _training_plan(result)
    assert len(resolved["dataset"]["episodes"]) == 165
    assert resolved["training"]["batch_size"] == 64
    assert resolved["training"]["steps"] == 200000
    assert resolved["training"]["save_freq"] == 25000


@pytest.mark.parametrize("name", ["train_bc_chunk.sh", "train_diffusion.sh"])
def test_training_launchers_reject_duplicates_unknowns_and_mode_mixing(name: str) -> None:
    duplicate = _run(
        name,
        "--mode=smoke",
        "--mode=smoke",
        "--device=cpu",
        "--print-command",
    )
    unknown = _run(name, "--mode=smoke", "--device=cpu", "--wat", "--print-command")
    mixed = _run(
        name,
        "--mode=smoke",
        "--device=cpu",
        "--output-dir=outputs/pusht_diffusion/pilot/not_allowed",
        "--print-command",
    )

    assert duplicate.returncode == 64
    assert unknown.returncode == 64
    assert mixed.returncode == 64


def test_prepare_dataset_prints_all_five_operations_without_writing() -> None:
    result = _run("prepare_dataset.sh", "--print-command")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("dataset_views") >= 1
    assert result.stdout.count("padding_stats") >= 2
    assert result.stdout.count("initialization") >= 2
