"""Verify the portable robot-practice runtime, GPU stack, simulators, and launchers."""

from __future__ import annotations

import argparse
import json
import os
import pty
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    return completed.stdout


def _basic_facts() -> dict[str, Any]:
    import torch

    packages = (
        "accelerate",
        "diffusers",
        "gym-pusht",
        "lerobot",
        "mani-skill",
        "mujoco",
        "num2words",
        "peft",
        "rerun-sdk",
        "torch",
        "torchvision",
    )
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "packages": {name: metadata.version(name) for name in packages},
        "driver": _nvidia_smi_driver(),
    }


def _nvidia_smi_driver() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    drivers = sorted(set(completed.stdout.split()))
    return ",".join(drivers) or None


def _verify_watch() -> None:
    wrapper = shutil.which("watch")
    expected = Path("/usr/local/bin/watch")
    if wrapper is None or Path(wrapper) != expected:
        raise RuntimeError(f"Expected watch wrapper at {expected}, found {wrapper}")

    help_result = subprocess.run(
        [wrapper, "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if help_result.returncode != 0:
        raise RuntimeError(
            f"watch --help failed with status {help_result.returncode}: {help_result.stdout}"
        )

    invalid_option = "--robot-practice-invalid-option"
    wrapper_invalid = subprocess.run(
        [wrapper, invalid_option],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    clean_environment = os.environ.copy()
    clean_environment.pop("LD_LIBRARY_PATH", None)
    clean_environment.pop("LD_PRELOAD", None)
    system_invalid = subprocess.run(
        ["/usr/bin/watch", invalid_option],
        check=False,
        env=clean_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if wrapper_invalid.returncode == 0:
        raise RuntimeError("watch unexpectedly accepted an invalid option")
    if wrapper_invalid.returncode != system_invalid.returncode:
        raise RuntimeError(
            "watch wrapper did not preserve the invalid-option exit status: "
            f"wrapper={wrapper_invalid.returncode} system={system_invalid.returncode}"
        )

    timeout = shutil.which("timeout")
    if timeout is None:
        raise RuntimeError("GNU timeout is required for the watch PTY probe")
    pty_environment = os.environ.copy()
    pty_environment["TERM"] = "xterm"
    master_fd, slave_fd = pty.openpty()
    try:
        probe = subprocess.run(
            [
                timeout,
                "--signal=TERM",
                "--kill-after=1s",
                "1s",
                wrapper,
                "-n",
                "0.1",
                "true",
            ],
            check=False,
            env=pty_environment,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            timeout=3,
        )
    finally:
        os.close(slave_fd)
        os.close(master_fd)
    if probe.returncode != 124:
        raise RuntimeError(
            "watch exited before the PTY probe deadline; "
            f"expected timeout status 124, got {probe.returncode}"
        )
    print("watch wrapper: path, exit forwarding, and PTY refresh passed")


def _verify_gpu(required_count: int, *, require_bf16: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Verify NVIDIA runtime/device-plugin injection and host driver."
        )
    actual_count = torch.cuda.device_count()
    if actual_count != required_count:
        raise RuntimeError(f"Expected exactly {required_count} CUDA GPUs, found {actual_count}")
    for index in range(actual_count):
        with torch.cuda.device(index):
            left = torch.arange(1024, device=f"cuda:{index}", dtype=torch.float32)
            result = (left * 2).sum().item()
            expected = float((1023 * 1024 // 2) * 2)
            if result != expected:
                raise RuntimeError(f"CUDA computation mismatch on device {index}")
            if require_bf16:
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"CUDA device {index} does not support BF16")
                matrix = torch.ones((16, 16), device=f"cuda:{index}", dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    bf16_result = matrix @ matrix
                if bf16_result.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"CUDA BF16 autocast produced {bf16_result.dtype} on device {index}"
                    )
        suffix = " (BF16 autocast passed)" if require_bf16 else ""
        print(f"CUDA device {index}: {torch.cuda.get_device_name(index)}{suffix}")


def _verify_smolvla() -> None:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from num2words import num2words
    from peft import LoraConfig

    config = SmolVLAConfig(
        device="cpu",
        use_peft=True,
        push_to_hub=False,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj"],
    )
    if config.type != "smolvla":
        raise RuntimeError(f"Unexpected SmolVLA policy type: {config.type}")
    if not config.use_peft:
        raise RuntimeError("SmolVLA PEFT configuration was not retained")
    if lora_config.r != 32 or lora_config.lora_alpha != 32:
        raise RuntimeError("LoRA configuration round-trip failed")
    if num2words(32) != "thirty-two":
        raise RuntimeError("num2words functional check failed")
    print(
        "SmolVLA/PEFT imports: "
        f"policy={SmolVLAPolicy.__name__} lora_rank={lora_config.r}"
    )


def _verify_pusht() -> None:
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    environment = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    try:
        observation, _ = environment.reset(seed=0)
        frame = environment.render()
        if set(observation) != {"pixels", "agent_pos"}:
            raise RuntimeError(f"Unexpected PushT observation keys: {observation.keys()}")
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.size == 0:
            raise RuntimeError(f"Unexpected PushT render shape: {frame.shape}")
        print(f"PushT headless render: shape={frame.shape}")
    finally:
        environment.close()


def _verify_mujoco() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=32, width=32)
    try:
        renderer.update_scene(data)
        frame = renderer.render()
        if frame.shape != (32, 32, 3):
            raise RuntimeError(f"Unexpected MuJoCo render shape: {frame.shape}")
        print(f"MuJoCo EGL render: shape={frame.shape}")
    finally:
        renderer.close()


def _verify_libero() -> None:
    import libero  # noqa: F401
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    required = {"libero_10", "libero_90", "libero_100", "libero_object"}
    missing = required - suites.keys()
    if missing:
        raise RuntimeError(f"LIBERO benchmark registry is incomplete: missing={sorted(missing)}")
    print(f"LIBERO benchmark suites: {sorted(suites)}")


def _verify_maniskill(*, use_gpu: bool) -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from sapien import physx

    backend = "gpu" if use_gpu else "cpu"
    if use_gpu:
        cache_path = (
            Path.home()
            / ".sapien"
            / "physx"
            / physx.version()
            / "libPhysXGpu_64.so"
        )
        if not cache_path.is_file():
            raise RuntimeError(f"SAPIEN PhysX GPU cache is missing: {cache_path}")
        physx.enable_gpu()
        if not physx.is_gpu_enabled():
            raise RuntimeError("SAPIEN PhysX GPU failed to enable")

    environment = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend=backend,
    )
    try:
        environment.reset(seed=0)
        frame = environment.render()
        shape = tuple(frame.shape)
        if len(shape) != 4 or shape[-1] != 3:
            raise RuntimeError(f"Unexpected ManiSkill render shape: {shape}")
        print(f"ManiSkill headless render: backend={backend} shape={shape}")
    finally:
        environment.close()


def _verify_torchcodec(repo: Path | None = None) -> None:
    from torchcodec.decoders import VideoDecoder

    print("TorchCodec shared libraries loaded")

    with tempfile.TemporaryDirectory(prefix="robot-practice-torchcodec-") as temporary:
        generated_video = Path(temporary) / "fixture.mp4"
        _run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=10",
                "-frames:v",
                "2",
                "-pix_fmt",
                "yuv420p",
                str(generated_video),
            ]
        )
        generated_frame = VideoDecoder(str(generated_video))[0]
        generated_shape = tuple(generated_frame.shape)
        if generated_shape != (3, 16, 16):
            raise RuntimeError(f"Unexpected generated TorchCodec frame shape: {generated_shape}")
        print(f"TorchCodec generated-video decode: shape={generated_shape}")

    if repo is None:
        return

    video_root = repo / "datasets" / "lerobot" / "lerobot" / "pusht" / "videos"
    video_path = next(video_root.rglob("*.mp4"), None) if video_root.is_dir() else None
    if video_path is None:
        print("TorchCodec dataset decode skipped: no PushT MP4 under repository")
        return

    frame = VideoDecoder(str(video_path))[0]
    shape = tuple(frame.shape)
    if len(shape) != 3 or shape[0] != 3:
        raise RuntimeError(f"Unexpected TorchCodec frame shape: {shape}")
    print(f"TorchCodec PushT decode: path={video_path.relative_to(repo)} shape={shape}")


def _verify_distributed_repo_import(repo: Path, world_size: int) -> None:
    if world_size < 2:
        return

    program = """
from pathlib import Path
import os

import robot_practice

repo = Path(os.environ["ROBOT_PRACTICE_VERIFY_REPO"]).resolve()
imported = Path(robot_practice.__file__).resolve()
if not imported.is_relative_to(repo):
    raise SystemExit(f"rank import came from {imported}, not {repo}")
print(f"rank={os.environ['RANK']} robot_practice={imported}", flush=True)
"""
    with tempfile.TemporaryDirectory(prefix="robot-practice-distributed-") as temporary:
        program_path = Path(temporary) / "verify_repo_import.py"
        program_path.write_text(program, encoding="utf-8")
        environment = os.environ.copy()
        environment["ROBOT_PRACTICE_VERIFY_REPO"] = str(repo)
        print(
            "+",
            "torchrun",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(program_path),
            flush=True,
        )
        completed = subprocess.run(
            [
                "torchrun",
                "--standalone",
                f"--nproc-per-node={world_size}",
                str(program_path),
            ],
            cwd=repo,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="")
        observed_ranks = {
            line.split()[0] for line in completed.stdout.splitlines() if line.startswith("rank=")
        }
        expected_ranks = {f"rank={rank}" for rank in range(world_size)}
        if observed_ranks != expected_ranks:
            raise RuntimeError(
                f"Distributed repository imports missing ranks: "
                f"expected={sorted(expected_ranks)} observed={sorted(observed_ranks)}"
            )


def _verify_repo(repo: Path, gpu_count: int) -> None:
    import robot_practice

    imported = Path(robot_practice.__file__).resolve()
    if not imported.is_relative_to(repo):
        raise RuntimeError(f"robot_practice imports from {imported}, not {repo}")
    _verify_torchcodec(repo)
    _verify_distributed_repo_import(repo, gpu_count)
    _run(["bash", "scripts/doctor.sh"], cwd=repo)

    device = "cuda" if gpu_count else "cpu"
    commands = (
        (
            "scripts/experiments/pusht/train_bc_chunk.sh",
            "robot_practice.experiments.pusht_diffusion.train_bc_chunk",
            "robot_practice.experiments.pusht_diffusion.model_configs.bc_chunk",
        ),
        (
            "scripts/experiments/pusht/train_diffusion.sh",
            "robot_practice.experiments.pusht_diffusion.train_diffusion",
            "robot_practice.experiments.pusht_diffusion.model_configs.diffusion",
        ),
    )
    for script, entry_module, config_module in commands:
        output = _run(
            [
                "bash",
                script,
                "--mode=smoke",
                f"--device={device}",
                "--print-command",
            ],
            cwd=repo,
        )
        for expected in (entry_module, config_module):
            if expected not in output:
                raise RuntimeError(f"{script} output is missing {expected}")

    if gpu_count >= 2:
        for script, entry_module, config_module in (
            (
                "scripts/experiments/pusht/train_bc_chunk_multi_gpu.sh",
                "robot_practice.experiments.pusht_diffusion.train_bc_chunk",
                "robot_practice.experiments.pusht_diffusion.model_configs.bc_chunk",
            ),
            (
                "scripts/experiments/pusht/train_diffusion_multi_gpu.sh",
                "robot_practice.experiments.pusht_diffusion.train_diffusion",
                "robot_practice.experiments.pusht_diffusion.model_configs.diffusion",
            ),
        ):
            output = _run(
                [
                    "bash",
                    script,
                    f"--num-gpus={gpu_count}",
                    "--mode=smoke",
                    "--device=cuda",
                    "--print-command",
                ],
                cwd=repo,
            )
            for expected in (entry_module, config_module, f'"world_size": {gpu_count}'):
                if expected not in output:
                    raise RuntimeError(f"{script} output is missing {expected}")


def _verify_nccl(world_size: int) -> None:
    shared_memory = shutil.disk_usage("/dev/shm").total
    if shared_memory < 1024**3:
        raise RuntimeError(
            f"/dev/shm is only {shared_memory / 1024**2:.0f} MiB; "
            "configure at least 1 GiB of shared memory for the NCCL probe"
        )
    _run(
        [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={world_size}",
            "/usr/local/lib/robot-practice/nccl_probe.py",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpus", type=int, default=0)
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--nccl-world-size", type=int, default=0)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    os.environ.pop("DISPLAY", None)
    _verify_watch()
    facts = _basic_facts()
    print(json.dumps(facts, indent=2, sort_keys=True))
    _run([sys.executable, "-m", "pip", "check"])
    _verify_torchcodec()
    _verify_smolvla()

    if arguments.require_bf16 and not arguments.require_gpus:
        raise RuntimeError("--require-bf16 requires --require-gpus")
    if arguments.require_gpus:
        _verify_gpu(arguments.require_gpus, require_bf16=arguments.require_bf16)
    if arguments.headless:
        _verify_pusht()
        _verify_mujoco()
        _verify_libero()
        _verify_maniskill(use_gpu=bool(arguments.require_gpus))
    if arguments.repo is not None:
        _verify_repo(arguments.repo.expanduser().resolve(strict=True), arguments.require_gpus)
    if arguments.nccl_world_size:
        if arguments.require_gpus != arguments.nccl_world_size:
            raise RuntimeError("--nccl-world-size must equal --require-gpus")
        _verify_nccl(arguments.nccl_world_size)
    print("robot-practice environment verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
