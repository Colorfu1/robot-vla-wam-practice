"""Verify the portable robot-practice runtime, GPU stack, simulators, and launchers."""

from __future__ import annotations

import argparse
import json
import os
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


def _verify_gpu(required_count: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Verify NVIDIA runtime/device-plugin injection and host driver."
        )
    actual_count = torch.cuda.device_count()
    if actual_count != required_count:
        raise RuntimeError(f"Expected exactly {required_count} CUDA GPUs, found {actual_count}")
    for index in range(actual_count):
        left = torch.arange(1024, device=f"cuda:{index}", dtype=torch.float32)
        result = (left * 2).sum().item()
        expected = float((1023 * 1024 // 2) * 2)
        if result != expected:
            raise RuntimeError(f"CUDA computation mismatch on device {index}")
        print(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")


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


def _verify_maniskill() -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import sapien  # noqa: F401

    environment = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )
    try:
        environment.reset(seed=0)
        frame = environment.render()
        shape = tuple(frame.shape)
        if len(shape) != 4 or shape[-1] != 3:
            raise RuntimeError(f"Unexpected ManiSkill render shape: {shape}")
        print(f"ManiSkill headless render: shape={shape}")
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
    parser.add_argument("--nccl-world-size", type=int, default=0)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    os.environ.pop("DISPLAY", None)
    facts = _basic_facts()
    print(json.dumps(facts, indent=2, sort_keys=True))
    _run([sys.executable, "-m", "pip", "check"])
    _verify_torchcodec()

    if arguments.require_gpus:
        _verify_gpu(arguments.require_gpus)
    if arguments.headless:
        _verify_pusht()
        _verify_mujoco()
        _verify_libero()
        _verify_maniskill()
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
