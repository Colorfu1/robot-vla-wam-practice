from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from robot_practice.experiments.pusht_diffusion import distributed_training
from robot_practice.experiments.pusht_diffusion.distributed_training import (
    build_accelerate_command,
    print_multi_gpu_plan,
    resolve_multi_gpu_launch,
)
from robot_practice.experiments.pusht_diffusion.train_bc_chunk import (
    resolve_training as resolve_bc_chunk_training,
)
from robot_practice.experiments.pusht_diffusion.training_common import (
    DistributedProcessContext,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PRACTICE_ROOT / "scripts" / "experiments" / "pusht"


def _multi_gpu_arguments(output_dir: Path) -> list[str]:
    return [
        "--num-gpus=2",
        "--mode=smoke",
        "--steps=3",
        "--train-seed=1000",
        f"--output-dir={output_dir}",
        "--batch-size=7",
        "--num-workers=0",
        "--device=cuda",
        "--print-command",
    ]


def _resolve_plan(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
    arguments: list[str] | None = None,
):
    monkeypatch.setattr(distributed_training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_training.torch.cuda, "device_count", lambda: 4)
    return resolve_multi_gpu_launch(
        arguments or _multi_gpu_arguments(output_dir),
        description="BCChunk multi-GPU test",
        policy_type="bc_chunk",
        single_gpu_script="scripts/experiments/pusht/train_bc_chunk.sh",
        resolve_training=resolve_bc_chunk_training,
    )


def test_distributed_process_context_defaults_to_single_process() -> None:
    context = DistributedProcessContext.from_environment({})

    assert context == DistributedProcessContext(rank=0, local_rank=0, world_size=1)
    assert context.is_main_process


def test_distributed_process_context_reads_accelerate_environment() -> None:
    context = DistributedProcessContext.from_environment(
        {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
    )

    assert context.rank == 1
    assert context.local_rank == 1
    assert context.world_size == 4
    assert not context.is_main_process


@pytest.mark.parametrize(
    "environment",
    [
        {"WORLD_SIZE": "0"},
        {"WORLD_SIZE": "2", "RANK": "2"},
        {"WORLD_SIZE": "2", "LOCAL_RANK": "-1"},
        {"WORLD_SIZE": "two"},
    ],
)
def test_distributed_process_context_rejects_invalid_values(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        DistributedProcessContext.from_environment(environment)


def test_accelerate_command_pins_distributed_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(distributed_training.torch.cuda, "is_available", lambda: True)
    _, resolved = resolve_bc_chunk_training(
        [
            "--mode=smoke",
            "--steps=3",
            f"--output-dir={tmp_path / 'bc'}",
            "--batch-size=7",
            "--num-workers=0",
            "--device=cuda",
            "--print-command",
        ]
    )

    command = build_accelerate_command(
        resolved,
        num_gpus=2,
        python_executable="/environment/python",
    )

    assert command[:3] == (
        "/environment/python",
        "-m",
        "accelerate.commands.launch",
    )
    assert "--multi_gpu" in command
    assert "--num_processes=2" in command
    assert "--num_machines=1" in command
    assert "--gpu_ids=all" in command
    assert "--mixed_precision=no" in command
    assert "--dynamo_backend=no" in command
    assert "--main_process_port=0" in command
    module_index = command.index("-m", 3)
    assert command[module_index + 1].endswith("pusht_diffusion.train_bc_chunk")
    assert "--batch-size=7" in command
    assert f"--output-dir={tmp_path / 'bc'}" in command
    assert "--print-command" not in command


def test_multi_gpu_plan_uses_per_gpu_batch_and_one_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bc"
    plan = _resolve_plan(monkeypatch, output_dir)
    summary = plan.summary()

    assert plan.num_gpus == 2
    assert plan.available_cuda_devices == 4
    assert summary["training"]["batch_size"] == 7
    assert summary["training"]["per_gpu_batch_size"] == 7
    assert summary["training"]["world_size"] == 2
    assert summary["training"]["global_batch_size"] == 14
    assert summary["training"]["output_dir"] == str(output_dir)
    assert f"--output-dir={output_dir}" in plan.command
    assert not output_dir.exists()


def test_multi_gpu_plan_resolves_default_output_once_before_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = [
        "--num-gpus=2",
        "--mode=smoke",
        "--steps=3",
        "--batch-size=7",
        "--num-workers=0",
        "--device=cuda",
        "--print-command",
    ]
    plan = _resolve_plan(monkeypatch, Path("unused"), arguments)
    output_argument = f"--output-dir={plan.resolved.output_dir}"

    assert plan.command.count(output_argument) == 1
    assert plan.resolved.output_dir.parent.name == "bc_chunk"


def test_multi_gpu_print_plan_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _resolve_plan(monkeypatch, tmp_path / "bc")

    print_multi_gpu_plan(plan)

    stdout = capsys.readouterr().out
    command_text, config_text = stdout.removeprefix("Command:\n").split(
        "\nResolved configuration:\n",
        maxsplit=1,
    )
    command = shlex.split(command_text)
    config = json.loads(config_text)
    assert "accelerate.commands.launch" in command
    assert config["launcher"]["worker_module"].endswith("train_bc_chunk")
    assert config["training"]["global_batch_size"] == 14


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--mode=smoke", "--device=cuda", "--print-command"],
            "the following arguments are required: --num-gpus",
        ),
        (
            ["--num-gpus=1", "--mode=smoke", "--device=cuda", "--print-command"],
            "use scripts/experiments/pusht/train_bc_chunk.sh for one GPU",
        ),
        (
            [
                "--num-gpus=2",
                "--num-gpus=3",
                "--mode=smoke",
                "--device=cuda",
                "--print-command",
            ],
            "duplicate argument: --num-gpus",
        ),
        (
            [
                "--num-gpus=2",
                "--mode=smoke",
                "--device=cpu",
                "--print-command",
            ],
            "only supports --device=auto or --device=cuda",
        ),
        (
            [
                "--num-gpus=2",
                "--mode=smoke",
                "--device=cuda",
                "--unknown",
                "--print-command",
            ],
            "unrecognized arguments: --unknown",
        ),
    ],
)
def test_multi_gpu_launcher_rejects_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(distributed_training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_training.torch.cuda, "device_count", lambda: 4)

    with pytest.raises(SystemExit) as raised:
        _resolve_plan(monkeypatch, tmp_path / "bc", arguments)

    assert raised.value.code == 64
    assert message in capsys.readouterr().err


def test_multi_gpu_launcher_rejects_more_than_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(distributed_training.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_training.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(SystemExit) as raised:
        resolve_multi_gpu_launch(
            _multi_gpu_arguments(tmp_path / "bc"),
            description="BCChunk multi-GPU test",
            policy_type="bc_chunk",
            single_gpu_script="scripts/experiments/pusht/train_bc_chunk.sh",
            resolve_training=resolve_bc_chunk_training,
        )

    assert raised.value.code == 64
    assert "only 1 CUDA device(s) are visible" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("script_name", "launcher_module"),
    [
        ("train_bc_chunk_multi_gpu.sh", "train_bc_chunk_multi_gpu"),
        ("train_diffusion_multi_gpu.sh", "train_diffusion_multi_gpu"),
    ],
)
def test_multi_gpu_bash_is_only_a_model_specific_environment_forwarder(
    script_name: str,
    launcher_module: str,
) -> None:
    script = SCRIPT_ROOT / script_name
    text = script.read_text(encoding="utf-8")

    assert script.stat().st_mode & 0o111
    assert text.count("\n") <= 6
    assert "exec conda run --no-capture-output -n robot-practice" in text
    assert f"pusht_diffusion.{launcher_module}" in text
    assert '"$@"' in text
    assert "dataset." not in text
    assert "policy." not in text


def test_non_main_rank_does_not_touch_shared_training_log(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    code = (
        "from pathlib import Path\n"
        "from robot_practice.experiments.pusht_diffusion.training_runtime "
        "import persistent_training_log\n"
        f"output = Path({str(output_dir)!r})\n"
        "with persistent_training_log(output, enabled=False) as path:\n"
        "    assert path is None\n"
        "assert not output.exists()\n"
        "assert not output.with_name(output.name + '.train.log').exists()\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_main_rank_moves_training_log_into_completed_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    code = (
        "from pathlib import Path\n"
        "from robot_practice.experiments.pusht_diffusion.training_runtime "
        "import persistent_training_log\n"
        f"output = Path({str(output_dir)!r})\n"
        "with persistent_training_log(output, enabled=True):\n"
        "    output.mkdir(parents=True)\n"
        "    print('rank-zero-log-marker')\n"
        "assert (output / 'train.log').is_file()\n"
        "assert 'rank-zero-log-marker' in (output / 'train.log').read_text()\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_guides_keep_single_gpu_commands_and_add_compatible_multi_gpu_alternatives() -> None:
    guide_root = PRACTICE_ROOT / "reports" / "guides" / "pusht_diffusion"
    readme = (guide_root / "README.md").read_text(encoding="utf-8")
    smoke = (guide_root / "smoke.md").read_text(encoding="utf-8")
    benchmark = (guide_root / "benchmark.md").read_text(encoding="utf-8")

    for guide in (readme, smoke, benchmark):
        assert "train_bc_chunk_multi_gpu.sh" in guide
        assert "train_diffusion_multi_gpu.sh" in guide
    assert "train_bc_chunk.sh" in smoke
    assert "train_diffusion.sh" in smoke
    assert "--batch-size=4" in smoke
    assert "smoke_eval.sh" in smoke
    assert "train_bc_chunk.sh" in benchmark
    assert "train_diffusion.sh" in benchmark
    assert "--batch-size=32" in benchmark
    assert "checkpoint selection" in benchmark
