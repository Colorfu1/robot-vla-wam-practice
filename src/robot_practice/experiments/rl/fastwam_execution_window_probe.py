"""Eight paired EGL jobs: two captured states x execute 4/8/16/32 controls."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from robot_practice.experiments.rl.fastwam_dual_eval_multigpu import (
    interrupt,
    probe_mapping,
    shard_environment,
)
from robot_practice.experiments.rl.fastwam_dual_train_fit import predict
from robot_practice.experiments.rl.lightwam_train_scene_eval import (
    Simulator,
    file_hash,
    summarize,
    write_json,
)

LOG = logging.getLogger(__name__)
KEYS = ("sim_states", "integration_states", "rgb", "proprio")


def read_trace(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (*KEYS, "actions", "rewards", "successes")}


def splice(original, intervened, boundary):
    """States at boundary belong to the continuation; actions before it to original."""
    return {k: np.concatenate((original[k][:boundary], intervened[k])) for k in original}


def compare_interval(reference, captured, start, stop):
    errors = {}
    for key in reference:
        end = stop + 1 if key in KEYS else stop
        a, b = reference[key][start:end], captured[key]
        if a.shape != b.shape:
            raise ValueError(f"Replay shape mismatch: {key}")
        errors[key] = float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0))
    if not all(np.isfinite(v) and v <= 1e-8 for v in errors.values()):
        raise ValueError(f"Replay gate failed: {errors}")
    return errors


def action_slices(raw, execute_steps, remaining):
    if raw.shape != (32, 7) or not np.isfinite(raw).all():
        raise ValueError("Invalid action prediction")
    return np.clip(raw[:min(execute_steps, remaining)], -1, 1)


def worker(args, result):
    import torch
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
    from robot_practice.experiments.rl.fastwam_dual_eval import validate_rendering_info
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime

    found = list(args.source.glob(f"gpu_*/trajectory_{args.trajectory}"))
    if len(found) != 1:
        raise ValueError("Expected one captured trajectory")
    scene = found[0]
    meta = json.loads((scene.parent / "metrics.json").read_text())
    detail = json.loads((scene / "metrics.json").read_text())
    if meta["status"] != "complete" or meta["protocol"] != PROTOCOL:
        raise ValueError("Incomplete/incompatible source")
    checkpoint = Path(meta["checkpoint"])
    if file_hash(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed")
    training = json.loads((checkpoint.parent.parent / "metrics.json").read_text())
    train_args = training["arguments"]
    manifest = json.loads((Path(train_args["dual_data"]) / "manifest.json").read_text())
    for key in ("hdf5", "normalization"):
        if file_hash(getattr(args, key)) != manifest["source_sha256"][key]:
            raise ValueError(f"Source mismatch: {key}")
    original = read_trace(scene / "original" / "control_trace.npz")
    reference = original
    if args.branch == "gt_rgb_once":
        if args.step < detail["rollout_step"]:
            raise ValueError("Probe precedes image intervention")
        reference = splice(original, read_trace(scene / args.branch / "control_trace.npz"),
                           detail["rollout_step"])
    if len(reference["actions"]) < args.max_steps:
        raise ValueError("Source trace shorter than requested budget")
    with np.load(scene / args.branch / f"step_{args.step:04d}" / "prediction.npz") as z:
        source_action = z["native_action"].copy()
    cfg = OmegaConf.create({"data": {"npz": train_args["npz"]},
                            "lightwam": {"context": train_args["context"],
                                         "image_size": [224, 448]},
                            "evaluation": {"normalization_path": args.normalization}})
    LOG.info("START phase=model_load completed=0 total=1")
    _, adapter, result["provenance"] = load_fastwam(checkpoint, cfg)
    LOG.info("DONE phase=model_load completed=1 total=1")
    worker_args = SimpleNamespace(hdf5=args.hdf5, npz=train_args["npz"],
                                  normalization=args.normalization,
                                  trajectory_ids=[args.trajectory], diagnostic_views=False)
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    with torch.inference_mode(), Simulator(worker_args, args.output, python, root,
                                          render_backend="egl",
                                          egl_device_id=args.egl_device_id) as sim:
        sim.call("reset", args.trajectory)
        sim.call("capture_start", str(args.output / "prefix"))
        for step in range(0, args.step, 4):
            sim.call("step", (reference["actions"][step:step + 4], None))
            LOG.info("STATUS phase=prefix completed=%d total=%d", step + 4, args.step)
        sim.call("capture_stop", None)
        result["prefix_gate"] = compare_interval(
            reference, read_trace(args.output / "prefix" / "control_trace.npz"), 0, args.step)
        result["rendering"] = sim.call("rendering_info", None)
        validate_rendering_info(result["rendering"], "egl", args.egl_device_id)
        obs = sim.call("dual_observation", None)
        source_rgb = np.asarray(Image.open(scene / args.branch / "actual" /
                                           f"{args.step:04d}.png")).transpose(2, 0, 1)
        if not np.array_equal(obs["rgb"], source_rgb):
            raise ValueError("Branch dual RGB differs from captured source")
        noise_seed = int(meta["seed"]) + args.trajectory
        first = predict(adapter, obs, noise_seed)
        first_error = float(abs(first - source_action).max())
        if first_error != 0:
            raise ValueError(f"Initial prediction no longer identical: {first_error}")
        result.update(first_prediction_max_error=first_error,
                      checkpoint_sha256=meta["checkpoint_sha256"], noise_seed=noise_seed)
        sim.call("capture_start", str(args.output / "continuation"))
        steps, plans, counts, frames, rewards, successes = [], [], [], [], [], []
        t = args.step
        while t < args.max_steps:
            obs = sim.call("dual_observation", None)
            raw = first if t == args.step else predict(adapter, obs, noise_seed)
            executed = action_slices(raw, args.execute_steps, args.max_steps - t)
            steps.append(t)
            plans.append(raw)
            counts.append(len(executed))
            # Render at a fixed 5 Hz in all variants; do NOT replan inside this chunk.
            for offset in range(0, len(executed), 4):
                rgb = sim.call("dual_observation", None)["rgb"]
                frames.append(Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8)))
                reply = sim.call("step", (executed[offset:offset + 4], None))
                rewards.extend(reply["rewards"])
                successes.extend(reply["successes"])
            t += len(executed)
            result.update(control_steps_completed=t, control_steps_total=args.max_steps)
            write_json(args.output / "metrics.json", result)
            LOG.info("STATUS phase=rollout execute=%d completed=%d total=%d",
                     args.execute_steps, t, args.max_steps)
        sim.call("capture_stop", None)
    frames[0].save(args.output / "actual_5hz.gif", save_all=True,
                   append_images=frames[1:], duration=200, loop=0)
    np.savez_compressed(args.output / "plans.npz", decision_steps=steps,
                        native_actions=plans, executed_counts=counts)
    trace = read_trace(args.output / "continuation" / "control_trace.npz")
    if args.execute_steps == 4:
        result["four_step_baseline_gate"] = compare_interval(
            reference, trace, args.step, args.max_steps)
    close = np.flatnonzero(trace["actions"][:, 6] > 0)
    with np.load(args.normalization) as z:
        q = (trace["proprio"] + 1) * (z["obs_max"] - z["obs_min"] + 1e-6) / 2 + z["obs_min"]
    np.savez_compressed(args.output / "finger_response.npz",
                        control_steps=np.arange(args.step, args.max_steps + 1), finger_q=q[:, 7:9])
    result.update(status="complete", continuation=summarize(rewards, successes),
                  first_close_control_step=int(args.step + close[0]) if len(close) else None,
                  close_command_steps=len(close), finger_q_start=q[0, 7:9].tolist(),
                  finger_q_end=q[-1, 7:9].tolist(),
                  finger_q_max_change=np.max(abs(q[:, 7:9] - q[0, 7:9]), axis=0).tolist())


def launch(args, result):
    gpus = args.gpus.split(",")
    if len(gpus) != 8 or len(set(gpus)) != 8 or not all(g.isdecimal() for g in gpus):
        raise ValueError("This paired probe requires eight distinct physical GPU indices")
    mapping = probe_mapping(args.output)
    jobs = [(b, k) for b in ("original", "gt_rgb_once") for k in (4, 8, 16, 32)]
    processes, streams = [], []
    try:
        for gpu, (branch, k) in zip(gpus, jobs):
            name = f"{branch}_execute{k}"
            command = [sys.executable, "-u", "-m", __spec__.name, "--worker",
                       "--source", str(args.source), "--output", str(args.output / name),
                       "--trajectory", str(args.trajectory), "--step", str(args.step),
                       "--max-steps", str(args.max_steps), "--branch", branch,
                       "--execute-steps", str(k), "--egl-device-id", str(mapping[gpu]["egl_id"]),
                       "--hdf5", args.hdf5, "--normalization", args.normalization]
            stream = (args.output / f"{name}.log").open("w")
            streams.append(stream)
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    env=shard_environment(mapping[gpu]["cuda_ordinal_unmasked"], 2))
            processes.append((name, proc))
            LOG.info("START variant=%s gpu=%s pid=%d", name, gpu, proc.pid)
        while True:
            codes = {name: p.poll() for name, p in processes}
            result["exit_codes"] = codes
            write_json(args.output / "metrics.json", result)
            LOG.info("STATUS phase=variants completed=%d total=8",
                     sum(c is not None for c in codes.values()))
            for name, _ in processes:
                path = args.output / name / "metrics.json"
                if path.exists():
                    d = json.loads(path.read_text())
                    LOG.info("STATUS variant=%s completed=%s total=%d", name,
                             d.get("control_steps_completed", 0), args.max_steps)
            if any(c not in (None, 0) for c in codes.values()):
                raise RuntimeError(f"Worker failed: {codes}")
            if all(c == 0 for c in codes.values()):
                break
            time.sleep(10)
        result["variants"] = {}
        rows = []
        for name, _ in processes:
            data = json.loads((args.output / name / "metrics.json").read_text())
            if data["status"] != "complete":
                raise ValueError(f"Incomplete variant {name}")
            result["variants"][name] = data
            rows.append(f'<h2>{name}</h2><p>First close: {data["first_close_control_step"]}; '
                        f'task success: {data["continuation"]["task_success_any_step"]}</p>'
                        f'<img src="{name}/actual_5hz.gif"><a href="{name}/metrics.json">metrics</a>')
        (args.output / "index.html").write_text(
            '<meta charset="utf-8"><h1>ACTUAL rollouts: execution-window probe</h1>'
            '<p>Two different branch states; compare execution windows within each state. '
            'No new GT image injection. Closure is not proof of grasp.</p>' + ''.join(rows))
        result["status"] = "complete"
    finally:
        for _, proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for _, proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for stream in streams:
            stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory", type=int, default=0)
    p.add_argument("--step", type=int, default=124)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--branch", choices=("original", "gt_rgb_once"))
    p.add_argument("--execute-steps", type=int, choices=(4, 8, 16, 32))
    p.add_argument("--egl-device-id", type=int)
    p.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    p.add_argument("--normalization",
                   default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = p.parse_args()
    if not 0 <= args.step < args.max_steps or args.step % 4 or args.max_steps % 4:
        p.error("Need aligned step < max-steps")
    if args.worker and (args.branch is None or args.execute_steps is None
                        or args.egl_device_id is None):
        p.error("Worker requires branch, execute-steps and mapped EGL id")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")])
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    result = {"status": "running", "trajectory": args.trajectory, "branch_step": args.step,
              "source": str(args.source), "branch": args.branch, "execute_steps": args.execute_steps,
              "limitations": ["Selected training-scene diagnostic, not generalization evaluation.",
                              "No new RGB substitution; restore each captured branch history.",
                              "Grasp/lift/drop require visual review; commands are not grasp labels."]}
    try:
        (worker if args.worker else launch)(args, result)
        LOG.info("DONE phase=all completed=1 total=1")
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
