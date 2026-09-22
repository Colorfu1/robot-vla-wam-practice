"""Matched random-reset Square evaluation; no training-scene restoration or training."""

import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json

logger = logging.getLogger(__name__)


def summarize_batches(rows, requested):
    return {
        "evaluated": len(rows),
        "requested": requested,
        "success_rate_600": sum(r["success_600"] for r in rows) / len(rows) if rows else None,
        "success_rate_400": sum(r["success_400"] for r in rows) / len(rows) if rows else None,
        "per_episode": rows,
    }


def batch_rows(result, rewards):
    if rewards.shape[0] != 150:
        raise ValueError("Expected 150 action chunks for 600 control steps")
    rows = []
    for ep in result["per_episode"]:
        i = ep["env_index"]
        successful = rewards[:, i] >= 4.0
        indices = np.flatnonzero(successful)
        rows.append(
            {
                "environment_seed": ep["environment_seed"],
                "success_400": bool(successful[:100].any()),
                "success_600": bool(successful.any()),
                "first_success_chunk_end_control_step": int((indices[0] + 1) * 4)
                if len(indices)
                else None,
            }
        )
        if rows[-1]["success_600"] != bool(ep["success"]):
            raise ValueError("Episode summary disagrees with chunk reward criterion")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=("lightwam", "dp", "fastwam"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/experiments/rl/lightwam_base_robomimic_square.yaml")
    p.add_argument("--dp-vendor", default="third_party/dppo_official")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--batch-starts", help="Comma-separated global batch offsets for a GPU shard")
    args = p.parse_args()
    if args.episodes < 1 or args.batch_size < 1:
        p.error("episodes and batch-size must be positive")
    starts = list(range(0, args.episodes, args.batch_size))
    if args.batch_starts is not None:
        selected = [int(value) for value in args.batch_starts.split(",")]
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(s not in starts for s in selected)
        ):
            p.error("batch-starts must be unique valid global batch offsets")
        starts = sorted(selected)
    shard_total = sum(min(args.batch_size, args.episodes - s) for s in starts)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for key, value in {
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND": "osmesa",
        "ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES": "4",
        "ROBOT_PRACTICE_READPIXELS_PROBE": "1",
        "ROBOT_PRACTICE_ROLLOUT_START_METHOD": "spawn",
        "ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP": "1",
    }.items():
        os.environ[key] = value
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.lightwam_base import (
        LightWAMBaseAdapter,
        RobomimicLightWAMWindowDataset,
        load_official_lightwam,
    )
    from robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic import MODEL_DTYPES
    from robot_practice.experiments.rl.lightwam_square_diagnostic import rollout
    from robot_practice.experiments.rl.lightwam_train_scene_eval import load_dp

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
        force=True,
    )
    cfg = OmegaConf.load(args.config)
    if cfg.lightwam.action_horizon != 4 or cfg.action.dim != 7:
        raise ValueError("Requires Square 4x7 actions")
    if float(cfg.evaluation.success_threshold) != 1.0:
        raise ValueError("Requires official chunk success threshold 1")
    result = {
        "status": "running",
        "arguments": vars(args),
        "weight_variant": "model",
        "protocol": "random reset, 600 control steps, chunk reward sum >= 4; not official 400-step budget",
        "renderer": "osmesa",
        "checkpoint_sha256": file_hash(checkpoint),
        "runner_sha256": file_hash(__file__),
        "config_sha256": file_hash(args.config),
        "environment_seeds": list(range(args.seed, args.seed + args.episodes)),
        "batch_starts": starts,
        "shard_requested": shard_total,
    }
    rows = []
    write_json(output / "metrics.json", {**result, **summarize_batches(rows, args.episodes)})
    try:
        logger.info("START phase=load policy=%s completed=0 total=%d", args.policy, args.episodes)
        if args.policy == "dp":
            model, adapter = load_dp(checkpoint, args.dp_vendor, output, cfg.data.npz)
        elif args.policy == "fastwam":
            from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam

            model, adapter, provenance = load_fastwam(checkpoint, cfg)
            result["fastwam"] = provenance
            write_json(output / "metrics.json", {**result, **summarize_batches(rows, args.episodes)})
        else:
            dataset = RobomimicLightWAMWindowDataset(
                cfg.data.npz,
                context_path=cfg.lightwam.context,
                prompt=cfg.lightwam.prompt,
                image_size=cfg.lightwam.image_size,
                action_horizon=4,
                video_frames=cfg.lightwam.video_frames,
                normalized=True,
            )
            model = load_official_lightwam(
                vendor_root=Path(cfg.lightwam.vendor_root),
                model_config=Path(cfg.lightwam.model_config),
                local_model_root=Path(cfg.lightwam.local_model_root),
                device="cuda:0",
                model_dtype=MODEL_DTYPES[cfg.lightwam.model_dtype],
                checkpoint=checkpoint,
                allow_download=False,
            ).eval()
            adapter = LightWAMBaseAdapter(
                model,
                action_horizon=4,
                action_dim=7,
                image_size=cfg.lightwam.image_size,
                context=dataset.context,
                context_mask=dataset.context_mask,
                num_inference_steps=cfg.lightwam.inference_steps,
                clip_actions=False,
            ).eval()
        for start in starts:
            count = min(args.batch_size, args.episodes - start)
            destination = output / f"batch_{start:03d}"
            destination.mkdir()
            settings = SimpleNamespace(episodes=count, seed=args.seed + start, max_steps=600)
            batch = rollout(adapter, cfg, settings, destination)
            expected = list(range(settings.seed, settings.seed + count))
            if list(batch["environment_seeds"]) != expected:
                raise ValueError("Environment seed mismatch")
            with np.load(destination / "rollout_trace.npz") as trace:
                rows.extend(batch_rows(batch, trace["rewards"]))
            result.update(summarize_batches(rows, args.episodes))
            write_json(output / "metrics.json", result)
            logger.info(
                "STATUS phase=eval completed=%d total=%d rate600=%.4f rate400=%.4f",
                len(rows),
                shard_total,
                result["success_rate_600"],
                result["success_rate_400"],
            )
        result["status"] = "complete"
        logger.info("DONE phase=all completed=%d total=%d", len(rows), shard_total)
    except BaseException:
        result["status"] = "failed"
        logger.exception("Evaluation failed; partial results retained")
        raise
    finally:
        result.update(summarize_batches(rows, args.episodes))
        write_json(output / "metrics.json", result)


if __name__ == "__main__":
    main()
