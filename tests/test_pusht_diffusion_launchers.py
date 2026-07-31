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


def test_all_vscode_debug_launches_pin_the_project_interpreter() -> None:
    vscode_root = PRACTICE_ROOT / ".vscode"
    launch = json.loads((vscode_root / "launch.json").read_text(encoding="utf-8"))
    settings = json.loads((vscode_root / "settings.json").read_text(encoding="utf-8"))
    expected = settings["python.defaultInterpreterPath"]
    debug_launches = [
        configuration
        for configuration in launch["configurations"]
        if configuration["type"] == "debugpy"
    ]

    assert debug_launches
    assert {
        configuration["name"]
        for configuration in debug_launches
        if configuration.get("python") != expected
    } == set()


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
            "compare_development_checkpoints.sh",
            (),
            "development_comparison",
        ),
        (
            "eval_fixed_observations.sh",
            ("--bc-policy=bc", "--diffusion-policy=diff"),
            "fixed_observation_eval",
        ),
        (
            "analyze_fixed_observation_action_modes.sh",
            (),
            "fixed_observation_action_modes",
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
            "eval_e5_fixed_observations.sh",
            (
                "--bc-policy=bc",
                "--diffusion-policy=diff",
                "--prototype-manifest=prototype_manifest.json",
                "--label=Smoke",
                "--num-policy-seeds=2",
                "--num-inference-steps=2",
            ),
            "e5_fixed_observation",
        ),
        (
            "eval_e5_rollouts.sh",
            (
                "--bc-policy=bc",
                "--diffusion-policy=diff",
                "--prototype-manifest=prototype_manifest.json",
                "--num-gpus=1",
                "--label=Smoke",
                "--num-environment-seeds=1",
                "--num-policy-seeds=1",
                "--episode-length=1",
                "--num-inference-steps=1",
            ),
            "e5_rollout_eval",
        ),
        (
            "select_e5_checkpoints.sh",
            ("--num-gpus=2",),
            "e5_checkpoint_selection",
        ),
        (
            "eval_e5_oracle_audit.sh",
            (),
            "e5_oracle_audit",
        ),
        (
            "make_report.sh",
            ("--e0=e0", "--e1=e1", "--e2=e2", "--e3=e3"),
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


def test_sampling_steps_launcher_uses_e3_default_output() -> None:
    result = _run(
        "benchmark_sampling_steps.sh",
        "--diffusion-policy=diff",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert (
        f"--output-dir={PRACTICE_ROOT}/outputs/pusht_diffusion/pilot/e3_sampling_steps" in command
    )
    assert "--latency-iterations=200" in command


def test_sampling_steps_launcher_forwards_num_gpus() -> None:
    result = _run(
        "benchmark_sampling_steps.sh",
        "--diffusion-policy=diff",
        "--num-gpus=8",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    assert "--num-gpus=8" in shlex.split(result.stdout)


def test_sampling_steps_launcher_rejects_zero_gpus() -> None:
    result = _run(
        "benchmark_sampling_steps.sh",
        "--diffusion-policy=diff",
        "--num-gpus=0",
        "--print-command",
    )

    assert result.returncode == 64
    assert "positive counts required" in result.stderr


def test_e3_recording_viewer_loads_scenes_in_environment_order(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    for seed in reversed(range(2000, 2050)):
        (recordings / f"env_{seed:04d}.rrd").write_bytes(b"rrd")

    result = _run(
        "view_e3_rollouts.sh",
        f"--recordings-dir={recordings}",
        "--port=9879",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "--port=9879" in command
    assert command[-50:] == [str(recordings / f"env_{seed:04d}.rrd") for seed in range(2000, 2050)]


def test_e5_viewers_keep_same_scene_and_conditions_adjacent(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    for condition in ("P0", "P1", "NULL"):
        (fixed / f"condition_{condition}.rrd").write_bytes(b"rrd")
    fixed_result = _run(
        "view_e5_fixed_observations.sh",
        f"--recordings-dir={fixed}",
        "--print-command",
    )
    assert fixed_result.returncode == 0, fixed_result.stderr
    assert shlex.split(fixed_result.stdout)[-3:] == [
        str(fixed / "condition_P0.rrd"),
        str(fixed / "condition_P1.rrd"),
        str(fixed / "condition_NULL.rrd"),
    ]

    rollouts = tmp_path / "rollouts"
    for condition in ("P0", "P1", "NULL"):
        bc = rollouts / "mode_bc_chunk" / f"condition_{condition}"
        dp = rollouts / "mode_diffusion" / f"condition_{condition}" / "policy_seed_00"
        bc.mkdir(parents=True)
        dp.mkdir(parents=True)
        (bc / "env_2000.rrd").write_bytes(b"rrd")
        (dp / "env_2000.rrd").write_bytes(b"rrd")
    rollout_result = _run(
        "view_e5_rollouts.sh",
        f"--recordings-dir={rollouts}",
        "--print-command",
    )
    assert rollout_result.returncode == 0, rollout_result.stderr
    expected = []
    for condition in ("P0", "P1", "NULL"):
        expected.extend(
            [
                str(rollouts / "mode_bc_chunk" / f"condition_{condition}" / "env_2000.rrd"),
                str(
                    rollouts
                    / "mode_diffusion"
                    / f"condition_{condition}"
                    / "policy_seed_00"
                    / "env_2000.rrd"
                ),
            ]
        )
    assert shlex.split(rollout_result.stdout)[-6:] == expected


def test_e5_rollout_viewer_supports_held_out_condition_sweep(tmp_path: Path) -> None:
    recordings = tmp_path / "audit_recordings"
    for model in ("mode_bc_chunk", "mode_diffusion"):
        for condition in ("P0", "P1", "P2", "NULL"):
            directory = recordings / model / f"condition_{condition}"
            directory.mkdir(parents=True)
            for episode_index in (16, 108):
                (directory / f"episode_{episode_index:06d}.rrd").write_bytes(b"rrd")

    result = _run(
        "view_e5_rollouts.sh",
        f"--recordings-dir={recordings}",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    expected = []
    for episode_index in (16, 108):
        for condition in ("P0", "P1", "P2", "NULL"):
            for model in ("mode_bc_chunk", "mode_diffusion"):
                expected.append(
                    str(
                        recordings
                        / model
                        / f"condition_{condition}"
                        / f"episode_{episode_index:06d}.rrd"
                    )
                )
    assert shlex.split(result.stdout)[-16:] == expected


def test_report_launcher_forwards_e3_input() -> None:
    result = _run(
        "make_report.sh",
        "--e0=e0",
        "--e1=e1",
        "--e2=e2",
        "--e3=e3",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert f"--e3={PRACTICE_ROOT}/e3" in command
    assert not any(argument.startswith("--e4=") for argument in command)


def test_checkpoint_selection_launcher_forwards_num_gpus() -> None:
    result = _run(
        "select_checkpoints.sh",
        "--bc-run=bc",
        "--diffusion-run=diff",
        "--num-gpus=8",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    assert "--num-gpus=8" in shlex.split(result.stdout)


def test_checkpoint_selection_launcher_rejects_zero_gpus() -> None:
    result = _run(
        "select_checkpoints.sh",
        "--bc-run=bc",
        "--diffusion-run=diff",
        "--num-gpus=0",
        "--print-command",
    )

    assert result.returncode == 64
    assert "invalid numeric argument" in result.stderr


def test_e5_oracle_audit_launcher_forwards_num_gpus() -> None:
    result = _run(
        "eval_e5_oracle_audit.sh",
        "--num-gpus=8",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "--num-gpus=8" in command
    assert not any(argument.startswith("--device=") for argument in command)


def test_e5_oracle_audit_launcher_rejects_zero_gpus() -> None:
    result = _run(
        "eval_e5_oracle_audit.sh",
        "--num-gpus=0",
        "--print-command",
    )

    assert result.returncode == 64
    assert "--num-gpus must be a positive integer" in result.stderr


def test_e5_oracle_audit_launcher_rejects_multi_gpu_device_override() -> None:
    result = _run(
        "eval_e5_oracle_audit.sh",
        "--num-gpus=2",
        "--device=cuda:3",
        "--print-command",
    )

    assert result.returncode == 64
    assert "--device cannot be set with --num-gpus>1" in result.stderr


def test_e5_oracle_audit_launcher_rejects_duplicate_num_gpus() -> None:
    result = _run(
        "eval_e5_oracle_audit.sh",
        "--num-gpus=2",
        "--num-gpus=8",
        "--print-command",
    )

    assert result.returncode == 64
    assert "duplicate --num-gpus" in result.stderr


def test_development_comparison_launcher_forwards_inputs() -> None:
    result = _run(
        "compare_development_checkpoints.sh",
        "--selection=selection.json",
        "--device=cuda:3",
        "--output-dir=comparison",
        "--overwrite",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert f"--selection={PRACTICE_ROOT}/selection.json" in command
    assert "--device=cuda:3" in command
    assert f"--output-dir={PRACTICE_ROOT}/comparison" in command
    assert "--overwrite" in command


def test_development_recording_viewer_expands_complete_directory(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    for seed in range(1000, 1020):
        (recordings / f"env_{seed:04d}.rrd").write_bytes(b"rrd")

    result = _run(
        "view_development_recordings.sh",
        f"--recordings-dir={recordings}",
        "--port=9877",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:6] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "robot-practice",
        "robot-rerun",
    ]
    assert "--persist-state" in command
    assert "--detach-process" in command
    assert "--port=9877" in command
    assert command[-20:] == [str(recordings / f"env_{seed:04d}.rrd") for seed in range(1000, 1020)]


def test_development_recording_viewer_rejects_incomplete_and_invalid_inputs(
    tmp_path: Path,
) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    for seed in range(1000, 1019):
        (recordings / f"env_{seed:04d}.rrd").write_bytes(b"rrd")
    incomplete = _run(
        "view_development_recordings.sh",
        f"--recordings-dir={recordings}",
        "--print-command",
    )
    duplicate = _run(
        "view_development_recordings.sh",
        f"--recordings-dir={recordings}",
        f"--recordings-dir={recordings}",
        "--print-command",
    )
    unknown = _run(
        "view_development_recordings.sh",
        f"--recordings-dir={recordings}",
        "--unknown",
        "--print-command",
    )
    invalid_port = _run(
        "view_development_recordings.sh",
        f"--recordings-dir={recordings}",
        "--port=0",
        "--print-command",
    )

    assert incomplete.returncode == 66
    assert "expected exactly 20 RRD files" in incomplete.stderr
    assert duplicate.returncode == 64
    assert "duplicate --recordings-dir" in duplicate.stderr
    assert unknown.returncode == 64
    assert "unknown argument" in unknown.stderr
    assert invalid_port.returncode == 64
    assert "--port must be an integer" in invalid_port.stderr


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
    assert "--num-episodes=206" in result.stderr
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
    assert len(commands) == 3
    assert "robot_practice.experiments.pusht_diffusion.fixed_observation_eval" in commands[0]
    assert "robot_practice.experiments.pusht_diffusion.rollout_eval" in commands[1]
    assert "--protocol=initial-noise-sensitivity" in commands[1]
    assert "robot_practice.experiments.pusht_diffusion.rollout_eval" in commands[2]
    assert "--protocol=full-ddpm-sampling" in commands[2]


def test_e2_launcher_defaults_to_exact_full_sampling_protocol() -> None:
    result = _run(
        "eval_multi_seed_rollouts.sh",
        "--bc-policy=bc",
        "--diffusion-policy=diff",
        "--overwrite",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "--protocol=full-ddpm-sampling" in command
    assert "--label=Closed-loop" in command
    assert "--num-environment-seeds" not in " ".join(command)
    assert "--num-policy-seeds" not in " ".join(command)
    assert "--num-gpus=1" in command
    assert "--overwrite" in command


def test_e2_launcher_forwards_explicit_gpu_count() -> None:
    result = _run(
        "eval_multi_seed_rollouts.sh",
        "--bc-policy=bc",
        "--diffusion-policy=diff",
        "--num-gpus=4",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    assert "--num-gpus=4" in shlex.split(result.stdout)


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_e2_launcher_rejects_invalid_gpu_count(value: str) -> None:
    result = _run(
        "eval_multi_seed_rollouts.sh",
        "--bc-policy=bc",
        "--diffusion-policy=diff",
        f"--num-gpus={value}",
        "--print-command",
    )

    assert result.returncode == 64
    assert (
        "numeric argument required" in result.stderr
        or "GPU count must be positive" in result.stderr
    )


def test_e2b_viewer_loads_environment_major_recording_order(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    bc_recordings = tmp_path / "bc_recordings"
    bc_recordings.mkdir()
    for policy_seed in range(10):
        seed_dir = recordings / f"policy_seed_{policy_seed:02d}"
        seed_dir.mkdir(parents=True)
        for environment_seed in range(2000, 2020):
            (seed_dir / f"env_{environment_seed:04d}.rrd").write_bytes(b"rrd")
    for environment_seed in range(2000, 2020):
        (bc_recordings / f"env_{environment_seed:04d}.rrd").write_bytes(b"rrd")

    result = _run(
        "view_e2b_rollouts.sh",
        f"--recordings-dir={recordings}",
        "--print-command",
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    rrd_paths = [Path(value) for value in command if value.endswith(".rrd")]
    assert "--persist-state" in command
    assert len(rrd_paths) == 220
    assert rrd_paths[:11] == [
        bc_recordings / "env_2000.rrd",
        *[
            recordings / f"policy_seed_{policy_seed:02d}" / "env_2000.rrd"
            for policy_seed in range(10)
        ],
    ]
    assert rrd_paths[11:22] == [
        bc_recordings / "env_2001.rrd",
        *[
            recordings / f"policy_seed_{policy_seed:02d}" / "env_2001.rrd"
            for policy_seed in range(10)
        ],
    ]


def test_main_guide_exposes_required_candidate_generation_command() -> None:
    guide = (PRACTICE_ROOT / "reports" / "guides" / "pusht_diffusion" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "bash scripts/experiments/pusht/inspect_demonstrations.sh" in guide
    assert "--num-episodes=206" in guide
    assert "--num-candidates=10" in guide
    assert "--output-dir=outputs/pusht_diffusion/data_inspection" in guide


def test_development_comparison_guide_exposes_complete_command_and_outputs() -> None:
    guide = (
        PRACTICE_ROOT / "reports" / "guides" / "pusht_diffusion" / "development_comparison.md"
    ).read_text(encoding="utf-8")

    assert "bash scripts/experiments/pusht/compare_development_checkpoints.sh" in guide
    assert "--selection=outputs/pusht_diffusion/pilot/checkpoint_selection/" in guide
    assert "--device=cuda:0" in guide
    assert "--output-dir=outputs/pusht_diffusion/pilot/development_comparison" in guide
    for filename in (
        "comparison.json",
        "rollouts.parquet",
        "recordings/bc_chunk",
        "recordings/diffusion",
        "env_1000.rrd",
        "env_1019.rrd",
    ):
        assert filename in guide
    assert "bash scripts/experiments/pusht/view_development_recordings.sh" in guide
    assert "--port=9876" in guide
    assert "--port=9877" in guide
    assert "Reset Blueprint" in guide


def test_e3_guide_exposes_multi_gpu_command_and_scene_viewer() -> None:
    guide = (
        PRACTICE_ROOT / "reports" / "guides" / "pusht_diffusion" / "e3_sampling_speed.md"
    ).read_text(encoding="utf-8")

    assert "bash scripts/experiments/pusht/benchmark_sampling_steps.sh" in guide
    assert "--num-gpus=8" in guide
    assert "--overwrite" in guide
    assert "bash scripts/experiments/pusht/view_e3_rollouts.sh" in guide
    assert "recording.rrd" not in guide
    assert "env_2000.rrd" in guide
    assert "DDIM-5/10/20/50/100" in guide
    assert "DDPM-100" in guide


def test_e1_guide_exposes_both_sampling_protocol_recordings() -> None:
    guide = (
        PRACTICE_ROOT / "reports" / "guides" / "pusht_diffusion" / "e1_fixed_observation.md"
    ).read_text(encoding="utf-8")

    assert "eval_fixed_observations.sh" in guide
    assert "--num-action-samples=20" in guide
    assert "recording_initial_noise_only.rrd" in guide
    assert "recording_full_sampling_seed.rrd" in guide
    assert "344" in guide
    assert "E1a" in guide
    assert "E1b" in guide


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
    assert resolved["training"]["initialization"] == "random_from_train_seed"
    assert resolved["training"]["initialization_seed"] == 1000
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


def test_prepare_dataset_prints_three_data_operations_without_writing() -> None:
    result = _run("prepare_dataset.sh", "--print-command")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("dataset_views") >= 1
    assert result.stdout.count("padding_stats") >= 2
    assert "pusht_diffusion.initialization" not in result.stdout
