"""Isolated Square diagnostics; never trains or changes the formal eval protocol.

Offline interventions are training-set sensitivity tests, NOT validation accuracy.
Rollout GIFs show policy observations at chunk boundaries, not every physics step.
Task stages must be reviewed from recordings; robot height alone is not grasp success.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from robot_practice.experiments.rl.lightwam_base import (
    LightWAMBaseAdapter,
    RobomimicLightWAMWindowDataset,
    _DPPOSquareEnvClient,
    _resolve_dppo_runtime,
    _summarize_official_square_rollout,
    load_official_lightwam,
)
from robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic import (
    MODEL_DTYPES,
    _compute_metrics,
    _cond_from_item,
    _sample_window_indices,
)
from robot_practice.experiments.rl.rollout_forensics import memory_snapshot, record

logger = logging.getLogger(__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def donor_indices(dataset, indices):
    """Fixed donor at matching relative time in the next nonempty trajectory."""
    lengths = np.asarray(dataset.traj_lengths)
    starts = np.r_[0, np.cumsum(lengths)]
    eligible = np.flatnonzero(lengths >= max(dataset.video_frames, dataset.action_horizon))
    if len(eligible) < 2:
        raise ValueError("Image swap needs at least two complete trajectories")
    lookup = {int(start): i for i, (start, _) in enumerate(dataset.windows)}
    result = []
    for index in indices:
        frame = dataset.windows[index][0]
        trajectory = int(np.searchsorted(starts, frame, side="right") - 1)
        position = int(np.flatnonzero(eligible == trajectory)[0])
        donor = int(eligible[(position + 1) % len(eligible)])
        span = int(lengths[trajectory] - max(dataset.video_frames, dataset.action_horizon))
        donor_span = int(lengths[donor] - max(dataset.video_frames, dataset.action_horizon))
        offset = round((frame - starts[trajectory]) / max(span, 1) * donor_span)
        result.append(lookup[int(starts[donor] + offset)])
    return result


def image_frame(rgb):
    array = np.asarray(rgb)
    if array.ndim == 4:
        array = array[-1]
    return Image.fromarray(np.clip(array.transpose(1, 2, 0), 0, 255).astype(np.uint8))


def offline(adapter, dataset, indices, donors, output):
    targets, originals, swapped, errors = [], [], [], []
    for count, (index, donor) in enumerate(zip(indices, donors), 1):
        item = dataset[index]
        cond = _cond_from_item(item)
        start = dataset.windows[index][0]
        online_cond = dict(cond, rgb=dataset.images[start])
        train_rgb, train_state = adapter._prepare_observation(cond)
        eval_rgb, eval_state = adapter._prepare_observation(online_cond)
        errors.append(float((train_rgb - eval_rgb).abs().max()))
        torch.testing.assert_close(train_state, eval_state, rtol=0, atol=0)
        # Store representative train/online inputs for camera/orientation review.
        if count <= 4:
            image_frame(dataset.images[start]).save(output / f"demo_{count:02d}.png")
        changed = dict(cond, rgb=_cond_from_item(dataset[donor])["rgb"])
        targets.append(item["action"].numpy())
        for condition, destination in ((cond, originals), (changed, swapped)):
            pred = adapter.predict_action(condition, seed=0).float().cpu().numpy()[0]
            if not np.isfinite(pred).all():
                raise ValueError("nonfinite offline prediction")
            destination.append(pred)
        if count % 16 == 0 or count == len(indices):
            logger.info("STATUS phase=offline completed=%d total=%d", count, len(indices))
    target, original, swapped = map(np.stack, (targets, originals, swapped))
    normal = _compute_metrics(target, original, action_dim=7, action_horizon=4)
    intervention = _compute_metrics(target, swapped, action_dim=7, action_horizon=4)
    mse = normal["overall"]["mse_clipped"]
    result = {
        "training_set_only": True,
        "original": normal,
        "other_trajectory_image": intervention,
        "swap_mse_ratio": intervention["overall"]["mse_clipped"] / mse if mse else None,
        "prediction_change_mse_raw": float(np.mean((original - swapped) ** 2)),
        "preprocessing_max_abs_error": max(errors),
        "preprocessing_pass_at_1e-5": max(errors) <= 1e-5,
        "limitations": "Sensitivity is not proof of task-relevant vision or generalization. "
        "Preprocessing equality does not verify simulator camera or rendering equality.",
    }
    np.savez_compressed(
        output / "offline_predictions.npz",
        target=target,
        original=original,
        swapped=swapped,
        indices=indices,
        donors=donors,
    )
    write_json(output / "offline_metrics.json", result)
    return result


def rollout(adapter, cfg, args, output):
    evidence = output / "forensics"
    previous = os.environ.get("ROBOT_PRACTICE_ROLLOUT_FORENSICS")
    previous_fault = os.environ.get("PYTHONFAULTHANDLER")
    os.environ["ROBOT_PRACTICE_ROLLOUT_FORENSICS"] = str(evidence.resolve())
    os.environ["PYTHONFAULTHANDLER"] = "1"
    record(evidence, "rollout_begin", memory=memory_snapshot())
    try:
        return _rollout(adapter, cfg, args, output)
    except BaseException:
        logger.exception("[DEBUG-rollout] rollout failed; evidence=%s", evidence)
        record(evidence, "rollout_failed", memory=memory_snapshot())
        raise
    finally:
        record(evidence, "rollout_end", memory=memory_snapshot())
        for key, value in (
            ("ROBOT_PRACTICE_ROLLOUT_FORENSICS", previous),
            ("PYTHONFAULTHANDLER", previous_fault),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _rollout(adapter, cfg, args, output):
    python, source = _resolve_dppo_runtime(
        dppo_python_value=cfg.evaluation.get("dppo_python"),
        dppo_root_value=cfg.evaluation.get("dppo_root"),
    )
    steps = args.max_steps // 4
    rewards = np.zeros((steps, args.episodes))
    firsts = np.zeros((steps + 1, args.episodes), dtype=bool)
    firsts[0] = True
    frames = [[] for _ in range(args.episodes)]
    states, actions, raw_actions = [], [], []
    evidence = output / "forensics"
    evidence.mkdir(exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with _DPPOSquareEnvClient(
        dppo_python=python,
        dppo_root=source,
        normalization_path=Path(cfg.evaluation.normalization_path).resolve(),
        num_envs=args.episodes,
        max_episode_steps=args.max_steps,
        action_horizon=4,
        policy_seed=args.seed,
        worker_log=output / "env_worker.log",
    ) as env:
        obs = env.reset()
        environment_seeds = env.environment_seeds
        for step in range(steps):
            states.append(np.asarray(obs["state"]).copy())
            for i in range(args.episodes):
                frames[i].append(image_frame(obs["rgb"][i]))
            raw = adapter.predict_action(obs, seed=environment_seeds).float().cpu().numpy()
            if not np.isfinite(raw).all():
                raise ValueError("nonfinite rollout action")
            action = np.clip(raw, -1, 1)
            raw_actions.append(raw)
            actions.append(action)
            # Write before sending: preserve the triggering input even on worker failure.
            np.savez_compressed(
                evidence / f"chunk_{step:04d}_input.npz",
                rgb=obs["rgb"],
                state=obs["state"],
                action_raw=raw,
                action_executed=action,
                environment_seeds=environment_seeds,
            )
            record(evidence, "action_sent", chunk=step, completed_chunks=step)
            obs, reward, terminated, truncated = env.step(action)
            np.savez_compressed(
                evidence / f"chunk_{step:04d}_result.npz",
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            record(evidence, "action_returned", chunk=step, completed_chunks=step + 1)
            rewards[step] = reward
            firsts[step + 1] = np.logical_or(terminated, truncated)
            if step % 5 == 0 or step + 1 == steps:
                logger.info(
                    "STATUS phase=rollout completed=%d total=%d envs=%d",
                    step + 1,
                    steps,
                    args.episodes,
                )
        # Final obs can be an autoreset frame; keep it out of the GIF.
        states.append(np.asarray(obs["state"]).copy())
    rows = _summarize_official_square_rollout(
        rewards, firsts, act_steps=4, success_threshold=float(cfg.evaluation.success_threshold)
    )
    if len(rows) != args.episodes:
        raise ValueError(f"Expected {args.episodes} completed episodes, got {len(rows)}")
    for row in rows:
        i = row["env_index"]
        row["environment_seed"] = environment_seeds[i]
        row["manual_stage_review"] = None
        row["video"] = f"env_{environment_seeds[i]}.gif"
        frames[i][0].save(
            output / row["video"], save_all=True, append_images=frames[i][1:], duration=200, loop=0
        )
    np.savez_compressed(
        output / "rollout_trace.npz",
        states=np.stack(states),
        action_raw=np.stack(raw_actions),
        action_executed=np.stack(actions),
        rewards=rewards,
        firsts=firsts,
        environment_seeds=environment_seeds,
    )
    result = {
        "diagnostic_only": True,
        "episodes": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "environment_seeds": environment_seeds,
        "per_episode": rows,
        "video_note": "5 Hz policy-input RGB, not full-rate simulator video; "
        "grasp/lift/insertion stages require manual review.",
    }
    write_json(output / "rollout_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/rl/lightwam_base_robomimic_square.yaml"
    )
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-windows", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 8 or args.max_steps % 4:
        parser.error("episodes must be positive; max-steps must be >=8 and divisible by 4")
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
        force=True,
    )
    cfg = OmegaConf.load(args.config)
    if cfg.lightwam.action_horizon != 4 or cfg.action.dim != 7:
        raise ValueError("This diagnostic requires Square 4x7 actions")
    checkpoints = [Path(p) for p in args.checkpoints.split(",")]
    if len({p.stem for p in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoint stems must be unique")
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    rendering = {
        "requested_backend": os.environ.get("ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND", "egl"),
        "requested_offsamples": os.environ.get("ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES"),
        "start_method": os.environ.get("ROBOT_PRACTICE_ROLLOUT_START_METHOD", "default"),
        "serial_step": os.environ.get("ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP") == "1",
        "verification": "See renderer_verified events and offSamples in per-worker forensics",
    }
    manifest = {
        "rendering": rendering,
        "arguments": vars(args),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "model_config": OmegaConf.to_container(
            OmegaConf.load(cfg.lightwam.model_config), resolve=True
        ),
        "source_sha256": {},
    }
    for path in [
        Path(__file__),
        Path(__file__).with_name("lightwam_base.py"),
        Path(cfg.lightwam.vendor_root) / "src/lightwam/models/wan22/lightwam.py",
    ]:
        manifest["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "manifest.json", manifest)
    dataset = RobomimicLightWAMWindowDataset(
        cfg.data.npz,
        context_path=cfg.lightwam.context,
        prompt=cfg.lightwam.prompt,
        image_size=cfg.lightwam.image_size,
        action_horizon=4,
        video_frames=cfg.lightwam.video_frames,
        normalized=True,
    )
    indices = _sample_window_indices(len(dataset), args.num_windows, seed=0, random_sample=False)
    donors = donor_indices(dataset, indices)
    summary = {
        "diagnostic_only": True,
        "models": {},
        "automatic_training": False,
        "rendering": rendering,
    }
    for position, checkpoint in enumerate(checkpoints, 1):
        destination = output / checkpoint.stem
        destination.mkdir()
        logger.info(
            "STATUS phase=load checkpoint=%s completed=%d total=%d",
            checkpoint,
            position - 1,
            len(checkpoints),
        )
        started = time.monotonic()
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
        result = {"offline": offline(adapter, dataset, indices, donors, destination)}
        if not args.offline_only:
            result["rollout"] = rollout(adapter, cfg, args, destination)
        result["elapsed_seconds"] = time.monotonic() - started
        summary["models"][checkpoint.stem] = result
        write_json(output / "metrics.json", summary)
        del adapter, model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("DONE phase=checkpoint completed=%d total=%d", position, len(checkpoints))
    logger.info("DONE phase=all completed=%d total=%d", len(checkpoints), len(checkpoints))


if __name__ == "__main__":
    main()
