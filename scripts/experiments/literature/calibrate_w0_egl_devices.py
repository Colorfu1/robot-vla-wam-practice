#!/usr/bin/env python
"""Calibrate which (CUDA device, EGL device index) pairs can render safely.

``robosuite`` indexes ``eglQueryDevicesEXT()`` with ``MUJOCO_EGL_DEVICE_ID``.
That list is not filtered by ``CUDA_VISIBLE_DEVICES``, its order need not match
CUDA's, and its length need not equal the GPU count -- on an 8x H20 node it has
10 entries.  A wrong index still initialises and still renders, but aborts
natively inside ``mjr_readPixels`` once the same process also drives CUDA.  A
native SIGABRT cannot be caught in-process, so the only reliable way to learn
the mapping is to try each pair in its own child process.

Each cell runs ``probe_w0_render_lifecycle.py`` for a short render loop with a
CUDA observation copy.  ``CUDA_VISIBLE_DEVICES`` is deliberately left unset so
robosuite's ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` assertion does not
restrict the sweep, and the CUDA device is chosen with ``--cuda-device``.

Writes a JSON report with the per-CUDA-device list of safe EGL indices so the
W0 launcher can set ``MUJOCO_EGL_DEVICE_ID`` from measurement instead of
assuming ``egl_index == cuda_index``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = REPO_ROOT / "scripts/experiments/literature/probe_w0_render_lifecycle.py"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-devices",
        default="",
        help="comma-separated CUDA indices (default: all visible)",
    )
    parser.add_argument(
        "--egl-indices",
        default="",
        help="comma-separated EGL indices (default: 0..egl_device_count-1)",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help=(
            "explicit cuda:egl pairs, e.g. '0:0,1:1,7:7' or 'diagonal' / 'shifted'. "
            "Overrides --cuda-devices/--egl-indices."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run each pair this many times; a pair is safe only if every repeat completes",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=405,
        help="render steps per cell; 405 matches one libero_object episode",
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--output",
        default="outputs/literature/w0_fastwam/egl_calibration.json",
        help="repository-relative JSON report path",
    )
    return parser.parse_args(argv)


def _detect_counts() -> tuple[int, int]:
    """Return (cuda_device_count, egl_device_count) from a throwaway child."""

    code = (
        "import os;os.environ.setdefault('PYOPENGL_PLATFORM','egl');"
        "import torch;from mujoco.egl import egl_ext as E;"
        "print(torch.cuda.device_count(), len(E.eglQueryDevicesEXT()))"
    )
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True, capture_output=True, check=True
    )
    cuda_count, egl_count = result.stdout.strip().split()[-2:]
    return int(cuda_count), int(egl_count)


def _resolve_pairs(args: argparse.Namespace, cuda_count: int, egl_count: int) -> list[tuple[int, int]]:
    spec = args.pairs.strip().lower()
    gpu_egl_count = min(egl_count, cuda_count)
    if spec == "diagonal":
        return [(index, index) for index in range(gpu_egl_count)]
    if spec == "shifted":
        return [(index, (index + 1) % gpu_egl_count) for index in range(gpu_egl_count)]
    if spec:
        pairs = []
        for item in spec.split(","):
            cuda_text, _, egl_text = item.partition(":")
            pairs.append((int(cuda_text), int(egl_text)))
        return pairs
    cuda_devices = [
        int(item) for item in args.cuda_devices.split(",") if item.strip()
    ] or list(range(cuda_count))
    egl_indices = [
        int(item) for item in args.egl_indices.split(",") if item.strip()
    ] or list(range(egl_count))
    return [(cuda, egl) for cuda in cuda_devices for egl in egl_indices]


def _last_step(text: str) -> int | None:
    """Highest cumulative step the probe reported before dying."""

    matches = re.findall(r"\[PROBE\] STATUS total=(\d+)", text)
    return int(matches[-1]) if matches else None


def _failure_reason(text: str) -> str:
    for pattern in (
        r"(free\(\)[^\n]*)",
        r"(malloc\(\)[^\n]*)",
        r"(double free[^\n]*)",
        r"(Assertion [^\n]*)",
        r"(mju_[A-Za-z]*: [^\n]*)",
        r"(terminate called[^\n]*)",
        r"(\w*Error: [^\n]*)",
        r"(Fatal Python error: [^\n]*)",
    ):
        found = re.search(pattern, text)
        if found:
            return found.group(1)[:160]
    return "<no message before abort>"


def _run_cell(
    *,
    cuda_device: int,
    egl_index: int,
    attempt: int,
    args: argparse.Namespace,
    log_dir: Path,
) -> dict[str, object]:
    env = os.environ.copy()
    # Unset so robosuite's assertion does not constrain MUJOCO_EGL_DEVICE_ID,
    # and so --cuda-device indexes absolute CUDA devices.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["MUJOCO_EGL_DEVICE_ID"] = str(egl_index)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"

    command = [
        sys.executable,
        str(PROBE),
        "--suite", args.suite,
        "--task-id", str(args.task_id),
        "--trials", "1",
        "--steps", str(args.steps),
        "--obs-device", "cuda",
        "--cuda-device", str(cuda_device),
        "--hard-reset", "1",
        "--cuda-init", "before",
        "--report-every", "25",
    ]

    started = time.time()
    try:
        result = subprocess.run(
            command,
            env=env,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
        output = result.stdout + result.stderr
        return_code = result.returncode
        completed = "[PROBE] DONE" in result.stdout
    except subprocess.TimeoutExpired as expired:
        output = (expired.stdout or "") + (expired.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return_code = -1
        completed = False

    log_path = log_dir / f"cuda{cuda_device}_egl{egl_index}_try{attempt}.log"
    log_path.write_text(output, encoding="utf-8")

    safe = bool(completed and return_code == 0)
    return {
        "cuda_device": cuda_device,
        "egl_index": egl_index,
        "attempt": attempt,
        "return_code": return_code,
        "safe": safe,
        "last_step": _last_step(output),
        "steps_requested": args.steps,
        "seconds": round(time.time() - started, 1),
        "reason": "" if safe else _failure_reason(output),
        "log": str(log_path.relative_to(REPO_ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not PROBE.is_file():
        raise FileNotFoundError(f"missing render probe: {PROBE}")

    cuda_count, egl_count = _detect_counts()
    pairs = _resolve_pairs(args, cuda_count, egl_count)
    output_path = REPO_ROOT / args.output
    log_dir = output_path.parent / "egl_calibration_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    total = len(pairs) * args.repeats
    print(
        f"[CALIB] START cuda_device_count={cuda_count} egl_device_count={egl_count} "
        f"pairs={len(pairs)} repeats={args.repeats} cells={total} steps={args.steps} "
        f"timeout={args.timeout}s logs={log_dir.relative_to(REPO_ROOT)}",
        flush=True,
    )

    cells: list[dict[str, object]] = []
    done = 0
    for cuda_device, egl_index in pairs:
        for attempt in range(1, args.repeats + 1):
            cell = _run_cell(
                cuda_device=cuda_device,
                egl_index=egl_index,
                attempt=attempt,
                args=args,
                log_dir=log_dir,
            )
            cells.append(cell)
            done += 1
            print(
                f"[CALIB] {done}/{total} cuda={cuda_device} egl={egl_index} "
                f"try={attempt} safe={cell['safe']} rc={cell['return_code']} "
                f"last_step={cell['last_step']}/{args.steps} {cell['seconds']}s "
                f":: {cell['reason']}",
                flush=True,
            )

    pair_safe: dict[tuple[int, int], bool] = {}
    for cuda_device, egl_index in pairs:
        attempts = [
            cell
            for cell in cells
            if cell["cuda_device"] == cuda_device and cell["egl_index"] == egl_index
        ]
        pair_safe[(cuda_device, egl_index)] = all(cell["safe"] for cell in attempts)

    cuda_devices = sorted({cuda for cuda, _ in pairs})
    egl_indices = sorted({egl for _, egl in pairs})
    print("[CALIB] MATRIX rows=cuda_device cols=egl_index (o=safe, X=fail, .=untested)", flush=True)
    print("[CALIB]         " + " ".join(f"{index:>2}" for index in egl_indices), flush=True)
    for cuda_device in cuda_devices:
        row = []
        for egl_index in egl_indices:
            state = pair_safe.get((cuda_device, egl_index))
            row.append(" ." if state is None else (" o" if state else " X"))
        print(f"[CALIB] cuda{cuda_device:>2} " + " ".join(row), flush=True)

    safe_by_cuda = {
        str(cuda_device): sorted(
            egl_index
            for (cuda, egl_index), state in pair_safe.items()
            if cuda == cuda_device and state
        )
        for cuda_device in cuda_devices
    }
    diagonal_tested = [
        (cuda, egl) for (cuda, egl) in pair_safe if cuda == egl
    ]
    report = {
        "schema_version": 2,
        "cuda_device_count": cuda_count,
        "egl_device_count": egl_count,
        "steps_per_cell": args.steps,
        "repeats": args.repeats,
        "suite": args.suite,
        "task_id": args.task_id,
        "cells": cells,
        "safe_egl_indices_by_cuda_device": safe_by_cuda,
        "identity_mapping_valid": all(
            cuda_device in safe_by_cuda.get(str(cuda_device), [])
            for cuda_device in cuda_devices
        ),
        "same_device_pairs_tested": len(diagonal_tested),
        "same_device_pairs_safe": sum(
            1 for pair in diagonal_tested if pair_safe[pair]
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[CALIB] DONE identity_mapping_valid={report['identity_mapping_valid']} "
        f"same_device_safe={report['same_device_pairs_safe']}/"
        f"{report['same_device_pairs_tested']} report={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
