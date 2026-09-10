#!/usr/bin/env python3
"""Small, model-free CUDA preflight for the W0 Fast-WAM container.

This intentionally does not import Fast-WAM, load a checkpoint, touch the
dataset, or initialise a DataLoader.  It only checks the container's NVIDIA
device exposure, PyTorch CUDA discovery, and one small CUDA kernel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        return 127, "", f"{type(error).__name__}: {error}"
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _print_command_result(label: str, command: list[str], result: tuple[int, str, str]) -> bool:
    code, stdout, stderr = result
    print(f"{label}_command={' '.join(command)}")
    print(f"{label}_returncode={code}")
    if stdout:
        print(f"{label}_stdout={stdout}")
    if stderr:
        print(f"{label}_stderr={stderr}")
    return code == 0


def _read_first(paths: list[Path]) -> str:
    for path in paths:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return "unavailable"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Logical CUDA device index inside the container (default: 0)",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=2048,
        help="Side length for the small CUDA kernel smoke (default: 2048)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    failures: list[str] = []

    print("== W0 Fast-WAM CUDA preflight ==")
    print(f"python={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print(f"cwd={Path.cwd()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<unset>')}")
    print(f"NVIDIA_DRIVER_CAPABILITIES={os.environ.get('NVIDIA_DRIVER_CAPABILITIES', '<unset>')}")
    print(f"/dev/nvidia0={Path('/dev/nvidia0').exists()}")
    print(f"/dev/nvidiactl={Path('/dev/nvidiactl').exists()}")
    print(f"/dev/nvidia-uvm={Path('/dev/nvidia-uvm').exists()}")

    if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"].strip():
        failures.append("CUDA_VISIBLE_DEVICES is set but empty")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("nvidia_smi=missing")
        failures.append("nvidia-smi is not installed/in PATH")
    else:
        print(f"nvidia_smi={nvidia_smi}")
        result = _run([nvidia_smi, "-L"])
        if not _print_command_result("nvidia_smi_list", [nvidia_smi, "-L"], result):
            failures.append("nvidia-smi -L failed")
        query_command = [
            nvidia_smi,
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = _run(query_command)
        if not _print_command_result("nvidia_smi_query", query_command, result):
            failures.append("nvidia-smi query failed")

    try:
        import torch
    except Exception as error:  # pragma: no cover - environment failure path
        print(f"torch_import=FAIL {type(error).__name__}: {error}")
        failures.append("torch import failed")
    else:
        print(f"torch_version={torch.__version__}")
        print(f"torch_cuda_version={torch.version.cuda}")
        try:
            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count())
        except Exception as error:  # pragma: no cover - driver failure path
            print(f"torch_cuda_probe=FAIL {type(error).__name__}: {error}")
            failures.append("PyTorch CUDA probe raised an exception")
        else:
            print(f"torch_cuda_available={available}")
            print(f"torch_cuda_device_count={count}")
            if not available or count <= args.device_index:
                failures.append(
                    f"requested logical CUDA device {args.device_index} is unavailable"
                )
            else:
                device = torch.device(f"cuda:{args.device_index}")
                try:
                    properties = torch.cuda.get_device_properties(device)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    print(f"torch_device_name={properties.name}")
                    print(f"torch_device_capability={properties.major}.{properties.minor}")
                    print(f"torch_device_total_gib={total_bytes / 2**30:.2f}")
                    print(f"torch_device_free_gib={free_bytes / 2**30:.2f}")

                    size = max(1, int(args.matmul_size))
                    left = torch.ones((size, size), device=device, dtype=torch.float32)
                    right = torch.full((size, size), 2.0, device=device, dtype=torch.float32)
                    output = left @ right
                    torch.cuda.synchronize(device)
                    expected = 2.0 * size
                    max_error = float((output - expected).abs().max().item())
                    print(f"cuda_kernel_matmul_size={size}")
                    print(f"cuda_kernel_max_error={max_error}")
                    print(
                        "cuda_kernel_smoke="
                        + ("PASS" if max_error == 0.0 else "FAIL")
                    )
                    if max_error != 0.0:
                        failures.append("CUDA matmul result was incorrect")
                    del left, right, output
                    torch.cuda.synchronize(device)
                    print(
                        f"torch_memory_allocated_mib="
                        f"{torch.cuda.memory_allocated(device) / 2**20:.1f}"
                    )
                    print(
                        f"torch_memory_reserved_mib="
                        f"{torch.cuda.memory_reserved(device) / 2**20:.1f}"
                    )
                except Exception as error:  # pragma: no cover - driver/kernel failure path
                    print(f"cuda_kernel_smoke=FAIL {type(error).__name__}: {error}")
                    failures.append("CUDA device allocation/kernel failed")

    print(f"cgroup_memory_max={_read_first([Path('/sys/fs/cgroup/memory.max')])}")
    print(f"cgroup_memory_current={_read_first([Path('/sys/fs/cgroup/memory.current')])}")
    if failures:
        print("CUDA_PREFLIGHT=FAIL")
        for failure in failures:
            print(f"FAILURE={failure}")
        return 1

    print("CUDA_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
