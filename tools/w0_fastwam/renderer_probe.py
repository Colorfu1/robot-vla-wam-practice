"""Minimal LIBERO EGL/MuJoCo reset-render stress probe without a policy model."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

def _snapshot(stage: str, episode: int, step: int) -> None:
    fields = {
        "stage": stage,
        "episode": episode,
        "step": step,
        "pid": os.getpid(),
        "hard_reset": os.environ.get("W0_HARD_RESET", "default_true"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID", "unset"),
    }
    print("[W0-RENDER-PROBE] " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0:
        raise ValueError("--episodes and --steps must be positive")

    vendor_root = Path(os.environ.get("W0_VENDOR_ROOT", ".")).resolve()
    fastwam_root = vendor_root / "FastWAM"
    if str(fastwam_root) not in os.sys.path:
        os.sys.path.insert(0, str(fastwam_root))

    from experiments.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict["libero_spatial"]()
    task = suite.get_task(args.task_id)
    initial_states = list(suite.get_task_init_states(args.task_id))
    print(
        f"[W0-RENDER-PROBE] start task={args.task_id} episodes={args.episodes} "
        f"steps={args.steps} hard_reset={os.environ.get('W0_HARD_RESET', 'default_true')}",
        flush=True,
    )
    env, _ = get_libero_env(task, args.resolution, args.seed)
    started = time.time()
    try:
        for episode in range(args.episodes):
            _snapshot("before_reset", episode, 0)
            env.reset()
            obs = env.set_init_state(initial_states[episode % len(initial_states)])
            _ = get_libero_image(obs)
            _snapshot("after_reset_and_initial_state", episode, 0)
            for step in range(args.steps):
                try:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    _ = get_libero_image(obs)
                except BaseException:
                    _snapshot("exception", episode, step)
                    raise
                if step == 0 or (step + 1) % 25 == 0:
                    _snapshot("progress", episode, step + 1)
                if done:
                    break
            _snapshot("episode_done", episode, step + 1)
        print(f"[W0-RENDER-PROBE] complete elapsed_s={time.time() - started:.2f}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
