"""Intervene on physical demonstration state, not on generated video or fake observations."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)


def checked_frames(text, length):
    frames = sorted(set(map(int, text.split(","))))
    if not frames or any(t < 0 or t >= length - 4 for t in frames):
        raise ValueError("Each start frame must leave at least four recorded actions")
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trajectory", type=int, default=16)
    p.add_argument("--frames", default="69,77,85,93,101,109,117")
    p.add_argument("--max-steps", type=int, default=600)
    args = p.parse_args()
    if args.max_steps <= 0 or args.max_steps % 4:
        p.error("max-steps must be positive and divisible by 4")
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime
    from robot_practice.experiments.rl.lightwam_train_scene_eval import (
        Simulator,
        file_hash,
        policy_action,
        summarize,
        write_json,
    )

    matches = []
    for path in args.source.glob("shard_*/metrics.json"):
        meta = json.loads(path.read_text())
        for model in meta.get("models", {}).values():
            if str(args.trajectory) in model["episodes"]:
                matches.append((path, meta, model))
    if len(matches) != 1:
        raise ValueError("Expected exactly one recorded policy episode")
    source_path, meta, record = matches[0]
    cfg = OmegaConf.load(meta["arguments"]["config"])
    cfg.data.npz = meta["arguments"]["npz"]
    with np.load(cfg.data.npz) as z:
        starts = np.r_[0, np.cumsum(z["traj_lengths"])]
        begin, end = map(int, starts[args.trajectory:args.trajectory + 2])
        actions = z["actions"][begin:end]
        states, images = z["states"][begin:end], z["images"][begin:end]
    frames = checked_frames(args.frames, len(actions))
    if file_hash(cfg.data.npz) != meta["source_sha256"]["npz"]:
        raise ValueError("Training data changed")
    worker_args = SimpleNamespace(**meta["arguments"])
    worker_args.trajectory_ids = [args.trajectory]
    worker_args.diagnostic_views = False
    if file_hash(worker_args.hdf5) != meta["source_sha256"]["hdf5"]:
        raise ValueError("Source demonstration changed")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    result = {"status": "running", "trajectory": args.trajectory, "frames": frames,
              "source_metrics_sha256": file_hash(source_path), "cases": {},
              "intervention": "Restore demonstration XML and physical state, then fresh RGB/proprio",
              "limitations": ["Frames are grasp-stage candidates, not verified grasp labels.",
                              ("Restored demonstration state excludes original controller internals; "
                               "remaining demonstration action continuation is a separate gate."),
                              "This is not video-frame substitution or evidence of video causality.",
                              "Policy budget is additional control steps after intervention."]}
    write_json(args.output / "metrics.json", result)
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    LOG.info("START phase=load")
    model, adapter, result["provenance"] = load_fastwam(record["checkpoint"], cfg)
    with Simulator(worker_args, args.output, python, root) as sim:
        result["source"] = sim.audit

        def run_variant(frame, variant):
            reset = sim.call("reset_at", (args.trajectory, frame))
            obs = reset["obs"]
            gate = {"sim_state_max_abs_error": reset["sim_state_max_abs_error"],
                    "proprio_max_abs_error": float(np.abs(obs["state"] - states[frame]).max()),
                    "image_mae": float(np.abs(obs["rgb"].astype(float) - images[frame]).mean())}
            gate["passed"] = (gate["sim_state_max_abs_error"] <= 1e-8
                              and gate["proprio_max_abs_error"] <= 1e-3 and gate["image_mae"] <= 10)
            if not gate["passed"]:
                return {"restore_gate": gate, "status": "excluded_restore"}
            out = args.output / f"frame_{frame:04d}" / variant
            out.mkdir(parents=True)
            sim.call("capture_start", str(out))
            rewards, successes, raw_actions, pictures = [], [], [], []
            total = len(actions) - frame if variant == "demo_continuation" else args.max_steps
            for step in range(0, total, 4):
                pictures.append(Image.fromarray(obs["rgb"].transpose(1, 2, 0).astype(np.uint8)))
                if variant == "demo_continuation":
                    action = actions[frame + step:frame + min(step + 4, total)]
                    raw = action
                else:
                    raw, action = policy_action(adapter, obs, int(meta["arguments"]["seed"]))
                raw_actions.extend(raw)
                reply = sim.call("step", (action, None))
                obs = reply["obs"]
                rewards.extend(reply["rewards"])
                successes.extend(reply["successes"])
                if step % 40 == 0 or step + len(action) == total:
                    LOG.info("STATUS phase=%s frame=%d completed=%d total=%d", variant, frame,
                             step + len(action), total)
            capture = sim.call("capture_stop", None)
            pictures.append(Image.fromarray(obs["rgb"].transpose(1, 2, 0).astype(np.uint8)))
            pictures[0].save(out / "actual_rollout.gif", save_all=True,
                             append_images=pictures[1:], duration=200, loop=0)
            np.save(out / "actions_raw.npy", raw_actions)
            row = dict(summarize(rewards, successes), restore_gate=gate, capture=capture,
                       artifact_directory=str(out.relative_to(args.output)),
                       gif_note="Real simulator observations sampled at chunk boundaries plus final frame")
            write_json(out / "metrics.json", row)
            return row

        for index, frame in enumerate(frames, 1):
            row = {"demo_continuation": run_variant(frame, "demo_continuation")}
            control = row["demo_continuation"]
            eligible = control["restore_gate"]["passed"] and control.get("task_success_any_step", False)
            row["eligible"] = eligible
            if eligible:
                row["policy_continuation"] = run_variant(frame, "policy_continuation")
            else:
                row["policy_skipped_reason"] = "Restoration or remaining demonstration control failed"
            result["cases"][str(frame)] = row
            write_json(args.output / "metrics.json", result)
            LOG.info("DONE phase=frames completed=%d total=%d frame=%d eligible=%s",
                     index, len(frames), frame, eligible)
    result["status"] = "complete"
    write_json(args.output / "metrics.json", result)
    LOG.info("DONE phase=all completed=%d total=%d", len(frames), len(frames))


if __name__ == "__main__":
    main()
