"""Paired one-shot RGB substitution; restore physics using the full executed prefix."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)


def substitute_rgb(obs, rgb):
    # NPZ stores uint8 pixels; the simulator transport casts the same 0..255
    # representation to float32. Do not rescale or reinterpret the pixel values.
    live = obs["rgb"]
    if rgb.shape != live.shape or rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Replacement image shape mismatch: {rgb.shape} vs {live.shape}")
    for image in (live, rgb):
        if (image.dtype not in (np.dtype("uint8"), np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(image).all() or image.min() < 0 or image.max() > 255):
            raise ValueError("Expected finite CHW pixels in 0..255")
    converted = rgb.astype(live.dtype, copy=True)
    if not np.array_equal(converted.astype(np.float64), rgb.astype(np.float64)):
        raise ValueError("Replacement dtype conversion would change pixel values")
    result = {k: v.copy() for k, v in obs.items()}
    result["rgb"] = converted
    return result


def verify_prefix(reference, replay, stop):
    errors = {}
    for key in ("sim_states", "integration_states", "rgb", "proprio"):
        expected = reference[key][:stop + 1]
        got = replay[key]
        if got.shape != expected.shape or not np.isfinite(got).all():
            raise ValueError(f"Invalid prefix {key}")
        errors[key] = float(np.abs(got.astype(float) - expected.astype(float)).max())
    if max(errors.values()) > 1e-8:
        raise ValueError(f"Prefix restoration failed: {errors}")
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trajectory", type=int, default=16)
    p.add_argument("--rollout-step", type=int, default=260)
    p.add_argument("--demo-frame", type=int, default=87)
    p.add_argument("--max-steps", type=int, default=600,
                   help="Original episode budget, including the replayed prefix")
    args = p.parse_args()
    if not 0 <= args.rollout_step < args.max_steps or args.rollout_step % 4 or args.max_steps % 4:
        p.error("Require 4-step boundaries and rollout-step < max-steps")
    import torch
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.fastwam_square_diagnostic import image_from_tensor
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime
    from robot_practice.experiments.rl.lightwam_train_scene_eval import (
        Simulator,
        file_hash,
        policy_action,
        summarize,
        write_json,
    )

    found = []
    for path in args.source.glob("shard_*/metrics.json"):
        meta = json.loads(path.read_text())
        for name, model in meta.get("models", {}).items():
            if str(args.trajectory) in model["episodes"]:
                found.append((path, meta, name, model))
    if len(found) != 1:
        raise ValueError("Expected exactly one source episode")
    path, meta, name, record = found[0]
    episode = record["episodes"][str(args.trajectory)]
    if not episode["prefix_replay_verification"]["passed"]:
        raise ValueError("Source prefix verification failed")
    trace_path = path.parent / name / f"trajectory_{args.trajectory}" / "control_trace.npz"
    with np.load(trace_path) as z:
        reference = {k: z[k] for k in z.files}
    if args.rollout_step + 4 > len(reference["actions"]):
        raise ValueError("Source episode too short")
    cfg = OmegaConf.load(meta["arguments"]["config"])
    cfg.data.npz = meta["arguments"]["npz"]
    for key in ("npz", "hdf5"):
        if file_hash(meta["arguments"][key]) != meta["source_sha256"][key]:
            raise ValueError(f"Source {key} changed")
    with np.load(cfg.data.npz) as z:
        starts = np.r_[0, np.cumsum(z["traj_lengths"])]
        start, end = map(int, starts[args.trajectory:args.trajectory + 2])
        if not 0 <= args.demo_frame < end - start:
            raise ValueError("Invalid demonstration frame")
        replacement = z["images"][start + args.demo_frame].copy()
    worker_args = SimpleNamespace(**meta["arguments"])
    worker_args.trajectory_ids = [args.trajectory]
    worker_args.diagnostic_views = False
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    result = {"status": "running", "trajectory": args.trajectory,
              "rollout_step": args.rollout_step, "demo_frame": args.demo_frame,
              "max_steps": args.max_steps, "cases": {},
              "source_metrics_sha256": file_hash(path), "trace_sha256": file_hash(trace_path),
              "checkpoint_sha256": file_hash(record["checkpoint"]),
              "protocol": "Only first policy input RGB replaced; physics/proprio unchanged; "
                          "subsequent inputs live. Native infer_action drives both branches. "
                          "Predicted video is diagnostic and never used as control input.",
              "limitations": ["RGB substitution can contradict physical state.",
                              "A single paired sample does not prove general causality.",
                              "Step 260 is a candidate, not a verified grasp-stage match."]}
    write_json(args.output / "metrics.json", result)
    LOG.info("START phase=load")
    model, adapter, result["provenance"] = load_fastwam(record["checkpoint"], cfg)
    seed = int(meta["arguments"]["seed"])
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    with Simulator(worker_args, args.output, python, root) as sim, torch.inference_mode():
        for index, variant in enumerate(("original_rgb", f"demo{args.demo_frame}_rgb_once"), 1):
            out = args.output / variant
            prefix = out / "prefix_replay"
            prefix.mkdir(parents=True)
            obs = sim.call("reset", args.trajectory)
            sim.call("capture_start", str(prefix))
            for t in range(0, args.rollout_step, 4):
                obs = sim.call("step", (reference["actions"][t:t + 4], None))["obs"]
                if t % 40 == 0:
                    LOG.info("STATUS phase=prefix variant=%s completed=%d total=%d",
                             variant, t + 4, args.rollout_step)
            sim.call("capture_stop", None)
            with np.load(prefix / "control_trace.npz") as z:
                gate = verify_prefix(reference, z, args.rollout_step)
            actual_input = {k: v.copy() for k, v in obs.items()}
            policy_input = (substitute_rgb(obs, replacement) if index == 2 else actual_input)
            raw, action = policy_action(adapter, policy_input, seed)
            first_action = action.copy()
            np.savez_compressed(out / "first_input_action.npz", actual_rgb=obs["rgb"],
                                input_rgb=policy_input["rgb"], proprio=policy_input["state"],
                                raw_action=raw, executed_action=action)
            rgb, state = adapter._prepare_observation(policy_input)
            generated = model.infer_joint(
                prompt=None, input_image=rgb[0].to(model.device),
                proprio=state[0].to(model.device), context=adapter.context.to(model.device),
                context_mask=adapter.context_mask.to(model.device), num_video_frames=5,
                action_horizon=4, action=None, num_inference_steps=20, seed=seed,
                test_action_with_infer_action=True)
            diagnostic_action = generated["action"].detach().float().cpu().numpy().reshape(4, 7)
            diagnostic_action_error = float(np.abs(np.clip(diagnostic_action, -1, 1)
                                                   - first_action).max())
            image_from_tensor(rgb[0]).save(out / "POLICY_INPUT.png")
            for j, image in enumerate(generated["video"]):
                image.save(out / f"PRED_tplus{j}.png")
            sim.call("capture_start", str(out))
            rewards, successes = [], []
            for t in range(args.rollout_step, args.max_steps, 4):
                if t != args.rollout_step:
                    raw, action = policy_action(adapter, obs, seed)
                reply = sim.call("step", (action, None))
                obs = reply["obs"]
                rewards.extend(reply["rewards"])
                successes.extend(reply["successes"])
                if (t - args.rollout_step) % 40 == 0 or t + 4 == args.max_steps:
                    LOG.info("STATUS phase=continuation variant=%s completed=%d total=%d",
                             variant, t + 4, args.max_steps)
            capture = sim.call("capture_stop", None)
            with np.load(out / "control_trace.npz") as z:
                images = [Image.fromarray(x.transpose(1, 2, 0).astype(np.uint8)) for x in z["rgb"]]
                for j in range(5):
                    images[j].save(out / f"ACTUAL_tplus{j}.png")
                images[0].save(out / "ACTUAL_rollout.gif", save_all=True,
                               append_images=images[1:], duration=50, loop=0)
                first_q = z["proprio"][:5].copy()
            row = dict(summarize(rewards, successes), prefix_errors=gate, capture=capture,
                       first_actions=first_action.tolist(),
                       diagnostic_vs_executed_action_max_error=diagnostic_action_error,
                       first_action_max_error_vs_recorded=float(np.abs(
                           first_action - reference["actions"][args.rollout_step:args.rollout_step + 4]
                       ).max()),
                       first_four_proprio_normalized=first_q.tolist(),
                       time_axis="Control steps elapsed after intervention")
            result["cases"][variant] = row
            write_json(args.output / "metrics.json", result)
            LOG.info("DONE phase=branches completed=%d total=2", index)
    result["status"] = "complete"
    write_json(args.output / "metrics.json", result)
    (args.output / "index.html").write_text(
        '<meta charset="utf-8"><h1>一次RGB替换对照</h1><p>PRED是生成图；ACTUAL是真实执行。'
        '时间从干预点计数；只有B首次输入替换为示教图像。视频不驱动动作。</p>' + ''.join(
            f'<h2>{v}</h2><p>策略输入</p><img src="{v}/POLICY_INPUT.png">'
            '<p>预测未来1..4</p>' + ''.join(f'<img src="{v}/PRED_tplus{j}.png">' for j in range(1, 5))
            + '<p>实际未来1..4</p>' + ''.join(f'<img width="224" src="{v}/ACTUAL_tplus{j}.png">'
                                            for j in range(1, 5))
            + f'<p>实际完整续跑</p><img width="384" src="{v}/ACTUAL_rollout.gif">'
            for v in result["cases"]))
    LOG.info("DONE phase=all completed=2 total=2")


if __name__ == "__main__":
    main()
