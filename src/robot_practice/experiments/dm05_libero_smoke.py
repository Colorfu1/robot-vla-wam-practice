"""Small LIBERO closed-loop probe against the local DM05 HTTP service.

Use robot-practice Python; the model server runs in the isolated DM05 environment.
Image orientation, state layout, and continuous gripper follow dexbotic-benchmark.
"""

import argparse
import base64
import io
import json
import os
import time
from collections import deque
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7891/v1/infer")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--libero-assets-dir", type=Path, default=Path("datasets/raw/libero_assets")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dm05/libero_smoke"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import requests
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    from robosuite.utils.transform_utils import quat2axisangle

    from robot_practice.evaluation.libero_assets import configure_libero_assets_dir

    configure_libero_assets_dir(args.libero_assets_dir)

    args.output.mkdir(parents=True, exist_ok=True)

    def status(line):
        print(line, flush=True)
        with (args.output / "progress.log").open("a") as log:
            log.write(line + "\n")

    def encode(image):
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(image[::-1, ::-1])).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error("task-id is outside libero_spatial")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if args.episodes > len(initial_states):
        parser.error("Not enough official initial states")
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    np.random.seed(args.seed)
    session = requests.Session()
    session.trust_env = False  # Loopback requests must not use terminal proxies.
    results, latencies = [], []
    try:
        for episode in range(args.episodes):
            status(f"STATUS episodes={episode}/{args.episodes} steps=0/220")
            env.reset()
            obs = env.set_init_state(initial_states[episode])
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            queue = deque()
            success = False
            for step in range(220):
                if not queue:
                    state = np.concatenate(
                        [
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"].copy()),
                            obs["robot0_gripper_qpos"],
                        ]
                    )
                    payload = {
                        "observation": {
                            "prompt": task.language,
                            "state": state.tolist(),
                            "robot_type": "Franka",
                            "images": {
                                "1": encode(obs["agentview_image"]),
                                "2": encode(obs["robot0_eye_in_hand_image"]),
                            },
                        }
                    }
                    start = time.perf_counter()
                    response = session.post(args.url, json=payload, timeout=180)
                    response.raise_for_status()
                    actions = np.asarray(response.json()["actions"], dtype=np.float32)
                    if actions.shape != (10, 7) or not np.isfinite(actions).all():
                        raise RuntimeError("Invalid action response")
                    latencies.append((time.perf_counter() - start) * 1000)
                    queue.extend(actions)
                obs, _, done, _ = env.step(queue.popleft())
                if (step + 1) % 10 == 0:
                    status(f"STATUS episodes={episode}/{args.episodes} steps={step + 1}/220")
                if done:
                    success = True
                    break
            results.append({"episode": episode, "success": success, "steps": step + 1})
            status(f"DONE episodes={episode + 1}/{args.episodes} success={success}")
            metrics = {
                "category": "Smoke",
                "interaction": "Closed-loop",
                "suite": "libero_spatial",
                "task_id": args.task_id,
                "seed": args.seed,
                "episodes": results,
                "success_rate": sum(r["success"] for r in results) / len(results),
                "mean_http_latency_ms": float(np.mean(latencies)),
                "control_frequency_hz": env.env.control_freq,
                "action_horizon": 10,
                "max_steps": 220,
                "settle_steps": 10,
            }
            (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    finally:
        env.close()
        session.close()


if __name__ == "__main__":
    main()
