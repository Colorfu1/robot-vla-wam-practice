"""Build-time gate for the W0 video-generation scoring dependencies.

Two independent things are checked, and either one failing must fail the build.

1.  Every module the scoring path imports is importable.  ``imaging_quality`` reaches
    ``pyiqa.archs.musiq_arch``, which executes ``pyiqa/__init__.py`` first, so importing
    the submodule -- not just the package -- is the real test.

2.  Nothing was upgraded underneath the training stack.  The packages are installed with
    ``--no-deps`` precisely because their declared requirements are incompatible with this
    image: pinned VBench asks for ``numpy<2.0.0`` and ``transformers==4.33.2``, and pyiqa
    0.1.16 asks for ``transformers>=5.0``.  Resolving any of those would replace numpy,
    transformers or torch and break FastWAM, DeepSpeed and lerobot.  ``--no-deps`` only
    helps if it is verified, so the versions are asserted exactly.

decord is additionally exercised on a real file, because its wheel carries its own bundled
FFmpeg and an import-only check would not prove that the bundle loads.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

# Versions this layer must not disturb.  Measured on the base image before installing.
EXPECTED = {
    "torch": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "numpy": "2.2.6",
    "transformers": "4.57.6",
    "cv2": "5.0.0",
}

# Modules the nine reported VBench dimensions plus the W0 LPIPS metric import.
REQUIRED_MODULES = (
    "cv2",  # i2v_subject, subject_consistency, motion_smoothness, dynamic_degree
    "decord",  # every load_video call in vbench/utils.py and vbench2_beta_i2v/utils.py
    "imageio.v3",  # W0 frame extraction in w0_video_generation_eval.py
    "lpips",  # W0-Video-40 LPIPS
    "clip",  # background_consistency, aesthetic_quality (openai-clip)
    "timm",  # pyiqa and dreamsim backbones
    "open_clip",  # dreamsim ensemble branch
    "pyiqa",  # imaging_quality
    "dreamsim",  # i2v_background
    "pyiqa.archs.musiq_arch",  # the exact symbol imaging_quality imports
)

failures: list[str] = []

for name in REQUIRED_MODULES:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
        failures.append(f"import {name}: {type(exc).__name__}: {exc}")
    else:
        print(f"import OK   {name}", flush=True)

for name, expected in EXPECTED.items():
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"version probe {name}: {type(exc).__name__}: {exc}")
        continue
    actual = getattr(module, "__version__", "?")
    if actual != expected:
        failures.append(f"{name} changed: expected {expected}, found {actual}")
    else:
        print(f"version OK  {name} {actual}", flush=True)

# Prove decord's bundled FFmpeg actually loads and returns frames.
with tempfile.TemporaryDirectory() as tmp:
    sample = Path(tmp) / "decord_probe.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
            "-pix_fmt", "yuv420p", str(sample),
        ],
        check=True,
    )
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(sample), ctx=cpu(0))
        frame = reader[3].asnumpy()
        if frame.shape != (64, 64, 3):
            raise AssertionError(f"unexpected decoded shape {frame.shape}")
        print(f"decode OK   decord read {len(reader)} frames, frame {frame.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"decord decode: {type(exc).__name__}: {exc}")

if failures:
    print("\nFAILED:", file=sys.stderr)
    for item in failures:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)

print("\nW0 video-generation scoring dependencies: OK", flush=True)
