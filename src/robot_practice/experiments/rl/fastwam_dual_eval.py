"""Dual-view live closed-loop evaluation: predict 32, execute only 4, 600-step episodes."""

import argparse
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LOG = logging.getLogger(__name__)


def episode_selection(value, total):
    if total < 1:
        raise ValueError("episodes must be positive")
    if value is None:
        return list(range(total))
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part.isdecimal() for part in parts):
        raise ValueError("episode-indices must be comma-separated nonnegative integers")
    indices = [int(part) for part in parts]
    if len(set(indices)) != len(indices) or any(index >= total for index in indices):
        raise ValueError("episode-indices must be distinct and within episodes")
    return indices


def validate_rendering_info(info, backend, egl_device_id):
    expected = "EGLGLContext" if backend == "egl" else "OSMesaGLContext"
    if info["backend"] != backend or info["context_class"] != expected:
        raise ValueError(f"Simulator renderer mismatch: requested={backend}, observed={info}")
    if backend == "egl" and (
        info["egl_device_id"] != str(egl_device_id) or info["cuda_visible_devices"] is not None
    ):
        raise ValueError(f"Simulator EGL selection mismatch: {info}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--episode-indices",
                   help="Global zero-based episode indices assigned to this GPU")
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--render-backend", choices=("osmesa", "egl"), default="osmesa")
    p.add_argument("--egl-device-id", type=int, help="Mapped, unmasked EGL device id")
    p.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    p.add_argument("--normalization",
                   default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = p.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.max_steps % 4:
        p.error("Positive episodes and max-steps divisible by 4 required")
    if (args.render_backend == "egl" and (args.egl_device_id is None or args.egl_device_id < 0)
            or args.render_backend == "osmesa" and args.egl_device_id is not None):
        p.error("EGL requires --egl-device-id; OSMesa must not set it")
    try:
        selected = episode_selection(args.episode_indices, args.episodes)
    except ValueError as exc:
        p.error(str(exc))
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL, executed_actions
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime
    from robot_practice.experiments.rl.lightwam_train_scene_eval import (
        Simulator,
        file_hash,
        summarize,
        write_json,
    )

    training = json.loads((args.checkpoint.parent.parent / "metrics.json").read_text())
    if training["protocol"].get("observation_action") != PROTOCOL:
        raise ValueError("Not a dual-view 32/4/9/4 checkpoint")
    data = json.loads((Path(training["arguments"]["dual_data"]) / "manifest.json").read_text())
    for key in ("hdf5", "normalization"):
        if file_hash(getattr(args, key)) != data["source_sha256"][key]:
            raise ValueError(f"Changed dataset source {key}")
    cfg = OmegaConf.create({"data": {"npz": training["arguments"]["npz"]},
                           "lightwam": {"context": training["arguments"]["context"],
                                        "image_size": [224, 448]},
                           "evaluation": {"normalization_path": args.normalization}})
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")], force=True)
    result = {"status": "running", "protocol": PROTOCOL,
              "initialization": "seeded random reset; NOT demonstration state",
              "checkpoint_sha256": file_hash(args.checkpoint),
              "arguments": {"checkpoint": str(args.checkpoint), "episodes": args.episodes,
                            "episode_indices": selected, "seed": args.seed,
                            "max_steps": args.max_steps, "render_backend": args.render_backend,
                            "egl_device_id": args.egl_device_id},
              "episodes": [], "evaluated": 0, "requested": len(selected)}
    write_json(args.output / "metrics.json", result)
    worker_args = SimpleNamespace(hdf5=args.hdf5, npz=cfg.data.npz,
                                  normalization=args.normalization,
                                  trajectory_ids=[0], diagnostic_views=False)
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    started = time.monotonic()
    try:
        _model, adapter, provenance = load_fastwam(args.checkpoint, cfg)
        result["provenance"] = provenance
        write_json(args.output / "metrics.json", result)
        with Simulator(worker_args, args.output, python, root,
                       render_backend=args.render_backend,
                       egl_device_id=args.egl_device_id) as sim, torch.inference_mode():
            for completed, episode in enumerate(selected, 1):
                episode_started = time.monotonic()
                seed = args.seed + episode
                sim.call("reset_seed", seed)
                if "rendering" not in result:
                    result["rendering"] = sim.call("rendering_info", None)
                    validate_rendering_info(result["rendering"], args.render_backend,
                                            args.egl_device_id)
                    write_json(args.output / "metrics.json", result)
                rewards, successes, predictions, pictures = [], [], [], []
                for step in range(0, args.max_steps, 4):
                    obs = sim.call("dual_observation", None)
                    pictures.append(Image.fromarray(obs["rgb"].transpose(1, 2, 0)))
                    cond = {k: v[None, None] for k, v in obs.items()}
                    raw = adapter.predict_action(cond, seed=seed).float().cpu().numpy()[0]
                    if not np.isfinite(raw).all():
                        raise ValueError("Nonfinite prediction")
                    action = np.clip(executed_actions(raw), -1, 1)
                    predictions.append(raw)
                    reply = sim.call("step", (action, None))
                    rewards.extend(reply["rewards"])
                    successes.extend(reply["successes"])
                    if step % 40 == 0 or step + 4 == args.max_steps:
                        LOG.info("STATUS phase=rollout episode=%d seed=%d completed=%d total=%d",
                                 episode, seed, step + 4, args.max_steps)
                out = args.output / f"seed_{seed}"
                out.mkdir()
                pictures[0].save(out / "actual.gif", save_all=True,
                                 append_images=pictures[1:], duration=200, loop=0)
                np.savez_compressed(out / "actions.npz", predicted=predictions,
                                    rewards=rewards, successes=successes)
                result["episodes"].append(dict(summarize(rewards, successes),
                                               episode_index=episode, seed=seed,
                                               elapsed_seconds=time.monotonic() - episode_started,
                                               within_400=summarize(rewards[:400],
                                                                    successes[:400])))
                result["evaluated"] = completed
                write_json(args.output / "metrics.json", result)
                LOG.info("DONE phase=episodes completed=%d total=%d", completed, len(selected))
        result.update(status="complete", task_success_rate=float(np.mean(
            [row["task_success_any_step"] for row in result["episodes"]])),
            official_chunk_success_rate=float(np.mean(
                [row["official_chunk_success"] for row in result["episodes"]])))
        LOG.info("DONE phase=all completed=%d total=%d", len(selected), len(selected))
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        LOG.exception("Dual-view rollout failed; preserving partial metrics")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
