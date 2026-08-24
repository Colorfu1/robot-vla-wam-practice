#!/usr/bin/env python
"""Probe the LIBERO/robosuite offscreen renderer against CUDA runtime state.

This reproduces the W0 Fast-WAM ``mjr_readPixels`` SIGABRT with the smallest
possible setup: no policy weights, no Hydra, no vendored FastWAM code.  Every
difference between this probe and the real evaluator is exposed as one flag so
an abort can be attributed to a single variable:

* ``--cuda-init {after,before}``  initialise CUDA before or after the EGL render
  context is created.  The real evaluator loads the model first, so CUDA comes
  first; the first version of this probe did the opposite.
* ``--gpu-ballast-gib``           keep N GiB resident on the GPU to mimic the
  checkpoint's footprint without loading it.
* ``--obs-pipeline {raw,fastwam}``  ``fastwam`` replays the evaluator's PIL
  centre-crop/resize to 224x224 per camera plus horizontal concat.
* ``--obs-device {cpu,cuda}``     whether the RGB observation is copied to CUDA.
* ``--hard-reset {0,1}``          robosuite rebuilds sim + render context on
  every ``reset()``.
* ``--new-env-per-trial``         construct a fresh ``OffScreenRenderEnv`` per
  trial instead of resetting one env.
* ``--accelerate-state``          instantiate ``accelerate.PartialState()``.
* ``--gc-every``                  force ``gc.collect()`` every N steps.
* ``W0_GL_CONTEXT_GUARD=1``       sitecustomize re-binds each render call to its
  own GL context.

Progress, RSS and free VRAM are reported every ``--report-every`` steps so a
native abort lands at a known cumulative step instead of being guessed.

Diagnostic only: it never writes results and never claims a success rate.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from typing import Any

import numpy as np


def _resident_mib() -> int:
    with open("/proc/self/statm", encoding="utf-8") as handle:
        pages = int(handle.read().split()[1])
    return pages * 4096 // (1 << 20)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=7)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument(
        "--steps",
        type=int,
        default=405,
        help="steps per trial; 405 matches libero_object max_steps + num_steps_wait",
    )
    parser.add_argument(
        "--cuda-init",
        choices=("after", "before"),
        default="after",
        help="initialise CUDA after (default) or before the first render context",
    )
    parser.add_argument(
        "--gpu-ballast-gib",
        type=float,
        default=0.0,
        help="keep this many GiB resident on the GPU to mimic model weights",
    )
    parser.add_argument("--obs-pipeline", choices=("raw", "fastwam"), default="raw")
    parser.add_argument("--obs-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help=(
            "torch CUDA index to use, relative to CUDA_VISIBLE_DEVICES. Lets the "
            "CUDA device and MUJOCO_EGL_DEVICE_ID be varied independently."
        ),
    )
    parser.add_argument("--hard-reset", type=int, choices=(0, 1), default=1)
    parser.add_argument("--new-env-per-trial", action="store_true")
    parser.add_argument("--accelerate-state", action="store_true")
    parser.add_argument(
        "--gc-every",
        type=int,
        default=0,
        help="force gc.collect() every N steps (0 disables)",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--model-hw", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=50)
    return parser.parse_args(argv)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Mirror the evaluator's PIL centre-crop/resize exactly."""

    from PIL import Image

    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    resized_w, resized_h = resized.size
    left = max((resized_w - width) // 2, 0)
    top = max((resized_h - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _build_observation(observation: dict, args: argparse.Namespace) -> np.ndarray:
    front = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    if args.obs_pipeline == "fastwam":
        model_h, model_w = args.model_hw
        front = _center_crop_resize(front, width=model_w, height=model_h)
        wrist = _center_crop_resize(wrist, width=model_w, height=model_h)
    return np.concatenate([front, wrist], axis=1)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    use_cuda = args.obs_device == "cuda"
    cuda_device = f"cuda:{args.cuda_device}"
    ballast: Any = None

    def init_cuda(reason: str) -> None:
        nonlocal ballast
        torch.cuda.set_device(args.cuda_device)
        torch.zeros(1, device=cuda_device)
        if args.gpu_ballast_gib > 0 and ballast is None:
            elements = int(args.gpu_ballast_gib * (1 << 30) // 2)
            ballast = torch.empty(elements, dtype=torch.bfloat16, device=cuda_device)
        free_mib = torch.cuda.mem_get_info(args.cuda_device)[0] >> 20
        print(
            f"[PROBE] CUDA-INIT reason={reason} device={cuda_device} "
            f"name={torch.cuda.get_device_name(args.cuda_device)} "
            f"ballast_gib={args.gpu_ballast_gib} cuda_free_mib={free_mib}",
            flush=True,
        )

    if args.accelerate_state:
        from accelerate import PartialState

        state = PartialState()
        print(f"[PROBE] ACCELERATE device={state.device}", flush=True)

    if args.cuda_init == "before":
        init_cuda("before_render_context")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = list(suite.get_task_init_states(args.task_id))
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    def make_env():
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=args.resolution,
            camera_widths=args.resolution,
            hard_reset=bool(args.hard_reset),
        )
        env.seed(args.seed)
        return env

    env = make_env()

    if args.cuda_init == "after":
        init_cuda("after_render_context")
    elif use_cuda:
        init_cuda("lazy_before_loop")

    print(
        "[PROBE] START "
        f"suite={args.suite} task={args.task_id} trials={args.trials} steps={args.steps} "
        f"requested_hard_reset={bool(args.hard_reset)} actual_hard_reset={env.env.hard_reset} "
        f"cuda_init={args.cuda_init} ballast_gib={args.gpu_ballast_gib} "
        f"obs_pipeline={args.obs_pipeline} obs_device={args.obs_device} "
        f"cuda_device={args.cuda_device} "
        f"egl_device_id={os.environ.get('MUJOCO_EGL_DEVICE_ID', 'unset')} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
        f"new_env_per_trial={args.new_env_per_trial} accelerate={args.accelerate_state} "
        f"gc_every={args.gc_every} "
        f"gl_context_guard={os.environ.get('W0_GL_CONTEXT_GUARD', 'unset')} "
        f"mujoco_gl={os.environ.get('MUJOCO_GL', 'unset')} pid={os.getpid()}",
        flush=True,
    )

    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    total_steps = 0

    for trial_index in range(args.trials):
        if args.new_env_per_trial and trial_index > 0:
            env.close()
            del env
            gc.collect()
            env = make_env()

        env.reset()
        env.set_init_state(initial_states[trial_index % len(initial_states)])
        inner = env.env
        render_context = inner.sim._render_context_offscreen
        print(
            f"[PROBE] TRIAL trial={trial_index} sim_id={id(inner.sim)} "
            f"render_context_id={id(render_context)} "
            f"gl_context_id={id(getattr(render_context, 'gl_ctx', None))}",
            flush=True,
        )

        for step_index in range(args.steps):
            observation = env.step(action)[0]
            rgb = _build_observation(observation, args)

            if use_cuda:
                # Same expression as the FastWAM eval observation path.
                image = (
                    torch.tensor(rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device=cuda_device, dtype=torch.bfloat16)
                )
                image = image * (2.0 / 255.0) - 1.0
                del image

            total_steps += 1
            if args.gc_every and total_steps % args.gc_every == 0:
                gc.collect()
            if total_steps % args.report_every == 0:
                free_mib = (
                    (torch.cuda.mem_get_info(args.cuda_device)[0] >> 20) if use_cuda else -1
                )
                print(
                    f"[PROBE] STATUS total={total_steps}/{args.trials * args.steps} "
                    f"trial={trial_index} step={step_index} "
                    f"rss_mib={_resident_mib()} cuda_free_mib={free_mib}",
                    flush=True,
                )

    print(f"[PROBE] DONE total={total_steps} trials={args.trials}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
