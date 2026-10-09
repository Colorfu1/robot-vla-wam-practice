"""EGL execution-window probes, including saved execute32 -> execute4 handoffs.

Handoff: --source VISUAL --handoff-source EXECUTE32 --switch-steps 380,412
         --gpus 0,1,2,3 --output NEW_OUTPUT. No training or new RGB substitution.
"""

import argparse
import hashlib
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
from PIL import Image, ImageDraw

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


def handoff_jobs(switch_steps, max_steps):
    """Freeze two paired continuations per preselected boundary, no outcome selection."""
    steps = [int(s) for s in switch_steps.split(",")]
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("Need distinct switch steps")
    if any(s < 32 or s >= max_steps or s % 4 for s in steps):
        raise ValueError("Switch steps must be aligned and inside the budget")
    return [(step, k) for step in steps for k in (4, 32)]


def read_handoff(path, visual_source, trajectory, switch_step, max_steps, checkpoint_hash):
    """Join the saved execute32 prefix and continuation; never reset to a mid-state."""
    meta = json.loads((path / "metrics.json").read_text())
    if (meta["status"] != "complete" or meta["execute_steps"] != 32
            or meta["branch"] != "original" or meta["trajectory"] != trajectory
            or Path(meta["source"]).resolve() != visual_source.resolve()
            or meta["checkpoint_sha256"] != checkpoint_hash
            or meta["first_prediction_max_error"] != 0
            or not meta.get("prefix_gate")
            or any(v != 0 for v in meta["prefix_gate"].values())):
        raise ValueError("Incompatible or unverified execute32 source")
    boundary = int(meta["branch_step"])
    prefix = read_trace(path / "prefix" / "control_trace.npz")
    tail = read_trace(path / "continuation" / "control_trace.npz")
    if len(prefix["actions"]) != boundary:
        raise ValueError("Execute32 prefix length mismatch")
    for key in KEYS:
        if not np.array_equal(prefix[key][-1], tail[key][0]):
            raise ValueError(f"Execute32 splice boundary mismatch: {key}")
    reference = splice(prefix, tail, boundary)
    with np.load(path / "plans.npz", allow_pickle=False) as z:
        steps, plans, counts = (z[k].copy() for k in
                                ("decision_steps", "native_actions", "executed_counts"))
    if (steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer)
            or not np.issubdtype(counts.dtype, np.integer)
            or plans.shape != (len(steps), 32, 7) or counts.shape != steps.shape
            or not np.isfinite(plans).all() or not len(steps)
            or steps[0] != boundary or not np.array_equal(steps[1:], steps[:-1] + counts[:-1])
            or (counts <= 0).any() or (counts > 32).any()
            or (counts[:-1] != 32).any()):
        raise ValueError("Invalid execute32 plan timeline")
    executed = np.concatenate([action_slices(p, int(n), int(n)) for p, n in zip(plans, counts)])
    if not np.array_equal(executed, tail["actions"]):
        raise ValueError("Source plans differ from executed actions")
    matches = np.flatnonzero(steps == switch_step)
    if len(matches) != 1 or len(reference["actions"]) < max_steps:
        raise ValueError("Switch must be a recorded execute32 decision within source budget")
    files = ("metrics.json", "plans.npz", "prefix/control_trace.npz",
             "continuation/control_trace.npz")
    hashes = {name: file_hash(path / name) for name in files}
    return reference, plans[int(matches[0])], hashes, int(meta["noise_seed"])


def stage_arrays(rows):
    arrays = {k: np.asarray([r[k] for r in rows]) for k in
              ("nut_xyz", "eef_xyz", "finger_q", "bilateral_contact")}
    for key, width in (("nut_xyz", 3), ("eef_xyz", 3), ("finger_q", 2)):
        if arrays[key].shape != (len(rows), width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid grasp evidence: {key}")
    if not rows or arrays["bilateral_contact"].shape != (len(rows),):
        raise ValueError("Missing grasp evidence")
    return arrays


def handoff_stage_gate(rows):
    """Diagnostic support proxy, not proof of stable grasp; endpoint contacts only."""
    a = stage_arrays(rows)
    heights = (a["nut_xyz"][:, 2] - a["nut_xyz"][0, 2]) * 1000
    contacts = a["bilateral_contact"][-5:]
    return {
        "passed": bool(len(rows) >= 5 and np.all(heights[-5:] >= 20) and contacts.sum() >= 3),
        "nut_rise_mm": float(heights[-1]),
        "last5_min_rise_mm": float(heights[-5:].min()),
        "last5_bilateral_contact_count": int(contacts.sum()),
        "criterion": "last 5 endpoints >=20mm above reset; >=3 bilateral contacts",
        "limitation": "Post-hoc diagnostic stage gate, not a deployable grasp detector.",
    }


def stage_summary(rows, switch_step, actions):
    a = stage_arrays(rows)
    if len(rows) != switch_step + len(actions) + 1:
        raise ValueError("Stage/action indexing mismatch")
    heights = (a["nut_xyz"][:, 2] - a["nut_xyz"][0, 2]) * 1000
    contacts = a["bilateral_contact"][switch_step:]

    def first_true(values, sustained=1):
        for i in range(len(values) - sustained + 1):
            if np.all(values[i:i + sustained]):
                return switch_step + i
        return None

    return {
        "nut_rise_mm_at_switch": float(heights[switch_step]),
        "nut_rise_mm_final": float(heights[-1]),
        "nut_rise_mm_max_after_switch": float(heights[switch_step:].max()),
        "first_open_command_control_step": first_true(actions[:, 6] < 0),
        "first_5_endpoint_contact_loss_start": first_true(~contacts.astype(bool), 5),
        "first_return_within_10mm_of_reset_height": first_true(heights[switch_step:] <= 10),
        "limitation": "Contact loss/height return may be release or placement, not necessarily drop.",
    }


def actual_frame(rgb, step, label):
    frame = Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8))
    canvas = Image.new("RGB", (frame.width, frame.height + 24), "white")
    canvas.paste(frame, (0, 24))
    ImageDraw.Draw(canvas).text((4, 4), f"ACTUAL t={step} {label}", fill="black")
    return canvas


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
    handoff = getattr(args, "handoff_source", None)
    if handoff:
        reference, source_action, hashes, source_seed = read_handoff(
            handoff, args.source, args.trajectory, args.step, args.max_steps,
            meta["checkpoint_sha256"])
        result["handoff_source_sha256"] = hashes
        if source_seed != int(meta["seed"]) + args.trajectory:
            raise ValueError("Handoff noise seed differs from source")
    elif args.branch == "gt_rgb_once":
        if args.step < detail["rollout_step"]:
            raise ValueError("Probe precedes image intervention")
        reference = splice(original, read_trace(scene / args.branch / "control_trace.npz"),
                           detail["rollout_step"])
    if len(reference["actions"]) < args.max_steps:
        raise ValueError("Source trace shorter than requested budget")
    if not handoff:
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
        stage_rows, prefix_frames = [], []
        if handoff:
            stage_rows.append(sim.call("square_grasp_evidence", None))
        sim.call("capture_start", str(args.output / "prefix"))
        for step in range(0, args.step, 4):
            if handoff and step >= args.step - 32:
                rgb = sim.call("dual_observation", None)["rgb"]
                prefix_frames.append(actual_frame(rgb, step, "saved execute32 history"))
            reply = sim.call("step_grasp" if handoff else "step",
                             (reference["actions"][step:step + 4], None))
            if handoff:
                stage_rows.extend(reply["grasp_evidence"])
            LOG.info("STATUS phase=prefix completed=%d total=%d", step + 4, args.step)
        sim.call("capture_stop", None)
        result["prefix_gate"] = compare_interval(
            reference, read_trace(args.output / "prefix" / "control_trace.npz"), 0, args.step)
        result["rendering"] = sim.call("rendering_info", None)
        validate_rendering_info(result["rendering"], "egl", args.egl_device_id)
        obs = sim.call("dual_observation", None)
        if handoff:
            np.savez_compressed(args.output / "prefix_stage_evidence.npz",
                                control_steps=np.arange(len(stage_rows)), **stage_arrays(stage_rows))
            result["handoff_stage_gate"] = handoff_stage_gate(stage_rows)
            write_json(args.output / "metrics.json", result)
            if not result["handoff_stage_gate"]["passed"]:
                raise ValueError("Handoff stage gate failed; inspect endpoint evidence before rerun")
            result["branch_rgb_sha256"] = hashlib.sha256(obs["rgb"].tobytes()).hexdigest()
            result["branch_state_sha256"] = hashlib.sha256(obs["state"].tobytes()).hexdigest()
            Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(args.output / "branch_actual.png")
        else:
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
                frames.append(actual_frame(rgb, t + offset, f"execute{args.execute_steps}")
                              if handoff else Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8)))
                reply = sim.call("step_grasp" if handoff else "step",
                                 (executed[offset:offset + 4], None))
                if handoff:
                    stage_rows.extend(reply["grasp_evidence"])
                rewards.extend(reply["rewards"])
                successes.extend(reply["successes"])
            t += len(executed)
            result.update(control_steps_completed=t, control_steps_total=args.max_steps)
            write_json(args.output / "metrics.json", result)
            LOG.info("STATUS phase=rollout execute=%d completed=%d total=%d",
                     args.execute_steps, t, args.max_steps)
        sim.call("capture_stop", None)
        if handoff:
            rgb = sim.call("dual_observation", None)["rgb"]
            frames.append(actual_frame(rgb, args.max_steps, "final"))
    frames = prefix_frames + frames
    frames[0].save(args.output / "actual_5hz.gif", save_all=True,
                   append_images=frames[1:], duration=200, loop=0)
    np.savez_compressed(args.output / "plans.npz", decision_steps=steps,
                        native_actions=plans, executed_counts=counts)
    trace = read_trace(args.output / "continuation" / "control_trace.npz")
    if handoff:
        np.savez_compressed(args.output / "stage_evidence.npz",
                            control_steps=np.arange(len(stage_rows)), **stage_arrays(stage_rows))
        result["stage_summary"] = stage_summary(stage_rows, args.step, trace["actions"])
        if args.execute_steps == 32:
            result["execute32_suffix_gate"] = compare_interval(
                reference, trace, args.step, args.max_steps)
    elif args.execute_steps == 4:
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
    first_success = result["continuation"]["first_success_step"]
    result["first_success_absolute_control_step"] = (
        args.step + first_success if first_success is not None else None)


def launch(args, result):
    gpus = args.gpus.split(",")
    handoff = getattr(args, "handoff_source", None)
    if handoff:
        jobs = [(f"handoff_{step:04d}_execute{k}", "original", k, step)
                for step, k in handoff_jobs(args.switch_steps, args.max_steps)]
    else:
        jobs = [(f"{b}_execute{k}", b, k, args.step)
                for b in ("original", "gt_rgb_once") for k in (4, 8, 16, 32)]
    if (len(gpus) != len(jobs) or len(set(gpus)) != len(jobs)
            or not all(g.isdecimal() for g in gpus)):
        raise ValueError(f"Need {len(jobs)} distinct physical GPUs, one per paired variant")
    result["variants_total"] = len(jobs)
    if handoff:
        # Check candidates and source identity before launching expensive model loads.
        LOG.info("START phase=handoff_source_check completed=0 total=%d", len(jobs) // 2)
        source_meta = json.loads((handoff / "metrics.json").read_text())
        for i, (step, _) in enumerate(handoff_jobs(args.switch_steps, args.max_steps)[::2]):
            read_handoff(handoff, args.source, args.trajectory, step, args.max_steps,
                         source_meta["checkpoint_sha256"])
            LOG.info("STATUS phase=handoff_source_check completed=%d total=%d", i + 1, len(jobs) // 2)
    mapping = probe_mapping(args.output)
    processes, streams = [], []
    try:
        for gpu, (name, branch, k, step) in zip(gpus, jobs):
            command = [sys.executable, "-u", "-m", __spec__.name, "--worker",
                       "--source", str(args.source), "--output", str(args.output / name),
                       "--trajectory", str(args.trajectory), "--step", str(step),
                       "--max-steps", str(args.max_steps), "--branch", branch,
                       "--execute-steps", str(k), "--egl-device-id", str(mapping[gpu]["egl_id"]),
                       "--hdf5", args.hdf5, "--normalization", args.normalization]
            if handoff:
                command.extend(["--handoff-source", str(handoff)])
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
            LOG.info("STATUS phase=variants completed=%d total=%d",
                     sum(c is not None for c in codes.values()), len(jobs))
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
                        + (f'<p>Switch at {data["branch_step"]}; '
                           f'stage gate: {data["handoff_stage_gate"]["passed"]}; '
                           f'first open: {data["stage_summary"]["first_open_command_control_step"]}'
                           '</p>' if handoff else '')
                        +
                        f'<img src="{name}/actual_5hz.gif"><a href="{name}/metrics.json">metrics</a>')
        if handoff:
            result["paired_gates"] = validate_handoff_pairs(result["variants"], args.switch_steps)
        note = (
            "Same saved execute32 history within each switch pair. Prefix is recorded policy "
            "actions, not GT actions; no image substitution. All frames are ACTUAL, not PRED. "
            "Post-hoc stage selection is diagnostic only; contact loss is not necessarily a drop."
            if handoff else "Two different branch states; compare execution windows within each "
            "state. No new GT image injection. Closure is not proof of grasp.")
        (args.output / "index.html").write_text(
            '<meta charset="utf-8"><h1>ACTUAL rollouts: execution-window probe</h1>'
            f'<p>{note}</p>' + ''.join(rows))
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


def validate_handoff_pairs(variants, switch_steps):
    gates = {}
    for step in map(int, switch_steps.split(",")):
        a, b = (variants[f"handoff_{step:04d}_execute{k}"] for k in (4, 32))
        for row in (a, b):
            if (row["status"] != "complete" or not row["handoff_stage_gate"]["passed"]
                    or row["first_prediction_max_error"] != 0
                    or not row.get("prefix_gate") or any(row["prefix_gate"].values())):
                raise ValueError("Handoff worker gate failed")
        keys = ("branch_rgb_sha256", "branch_state_sha256", "checkpoint_sha256",
                "noise_seed", "handoff_source_sha256")
        if any(a[k] != b[k] for k in keys):
            raise ValueError("Handoff pair used different source, inputs, weights or noise")
        if not b.get("execute32_suffix_gate") or any(b["execute32_suffix_gate"].values()):
            raise ValueError("Execute32 control did not reproduce source suffix")
        gates[str(step)] = {"passed": True, "execute32_suffix_exact": True}
    return gates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory", type=int, default=0)
    p.add_argument("--step", type=int, default=124)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--handoff-source", type=Path,
                   help="Completed original_execute32 variant; restore its prefix, then replan")
    p.add_argument("--switch-steps", default="380,412",
                   help="Handoff parent only: preselected execute32 decision boundaries")
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
    if args.handoff_source and args.worker and (args.branch != "original"
                                               or args.execute_steps not in (4, 32)):
        p.error("Handoff worker requires original history and execute4/32")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")])
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    result = {"status": "running", "trajectory": args.trajectory, "branch_step": args.step,
              "source": str(args.source), "branch": args.branch, "execute_steps": args.execute_steps,
              "handoff_source": str(args.handoff_source) if args.handoff_source else None,
              "switch_steps": (args.switch_steps if args.handoff_source and not args.worker
                               else None),
              "diagnostic_only": True,
              "limitations": ["Selected training-scene diagnostic, not generalization evaluation.",
                              "No new RGB substitution; restore each captured branch history.",
                              "Grasp/lift/drop require visual review; commands are not grasp labels."]}
    if args.handoff_source:
        if not args.worker:
            result["branch_step"] = None
        result["limitations"].extend([
            "Switch points selected post hoc on existing execute32 history, not a deployable rule.",
            "Physical endpoint evidence is never a policy input or an online switch trigger.",
        ])
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
