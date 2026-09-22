"""Replay recorded actions with diagnostic cameras; never load or rerun a policy."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LOG = logging.getLogger(__name__)


def load_trace(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def prefix_trace(trace, steps):
    return {k: v[:steps if k in ("actions", "rewards", "successes") else steps + 1]
            for k, v in trace.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, help="Original sharded evaluation")
    p.add_argument("--intervention", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime
    from robot_practice.experiments.rl.lightwam_train_scene_eval import (
        Simulator,
        file_hash,
        write_json,
    )
    from robot_practice.experiments.rl.scene_capture import compare_traces

    intervention = json.loads((args.intervention / "metrics.json").read_text())
    if intervention["status"] != "complete":
        raise ValueError("Intervention incomplete")
    matches = [path for path in args.source.glob("shard_*/metrics.json")
               if file_hash(path) == intervention["source_metrics_sha256"]]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous matching source shard")
    source = matches[0]
    meta = json.loads(source.read_text())
    trajectory = intervention["trajectory"]
    model_names = [k for k, v in meta["models"].items() if str(trajectory) in v["episodes"]]
    if len(model_names) != 1:
        raise ValueError("Ambiguous model trace")
    original_path = source.parent / model_names[0] / f"trajectory_{trajectory}" / "control_trace.npz"
    if file_hash(original_path) != intervention["trace_sha256"]:
        raise ValueError("Original trace changed")
    original = load_trace(original_path)
    worker_args = SimpleNamespace(**meta["arguments"])
    worker_args.trajectory_ids = [trajectory]
    worker_args.diagnostic_views = True
    for key in ("npz", "hdf5"):
        if file_hash(meta["arguments"][key]) != meta["source_sha256"][key]:
            raise ValueError(f"Changed {key}")
    jobs = [(name, args.intervention / name / "control_trace.npz", intervention["rollout_step"])
            for name in intervention["cases"]]
    jobs.append(("demonstration", source.parent / "demonstration" /
                 f"trajectory_{trajectory}" / "control_trace.npz", 0))
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    result = {"status": "running", "trajectory": trajectory, "cases": {},
              "protocol": "Recorded action replay only; no model inference or new policy inputs",
              "intervention_metrics_sha256": file_hash(args.intervention / "metrics.json")}
    write_json(args.output / "metrics.json", result)
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    with Simulator(worker_args, args.output, python, root) as sim:
        for index, (name, trace_path, prefix_steps) in enumerate(jobs, 1):
            reference = load_trace(trace_path)
            out = args.output / name
            out.mkdir()
            sim.call("reset", trajectory)
            if prefix_steps:
                prefix_dir = out / "prefix_replay"
                sim.call("capture_start", str(prefix_dir))
                for step in range(0, prefix_steps, 4):
                    sim.call("step", (original["actions"][step:step + 4], None))
                    if step % 40 == 0:
                        LOG.info("STATUS phase=prefix case=%s completed=%d total=%d",
                                 name, step + 4, prefix_steps)
                sim.call("capture_stop", None)
                gate = compare_traces(prefix_trace(original, prefix_steps),
                                      load_trace(prefix_dir / "control_trace.npz"))
                if not gate["passed"]:
                    raise ValueError(f"Prefix mismatch {name}: {gate}")
            sim.call("capture_start", str(out))
            total = len(reference["actions"])
            for step in range(0, total, 4):
                action = reference["actions"][step:step + 4]
                sim.call("step", (action, None))
                if step % 40 == 0 or step + len(action) == total:
                    LOG.info("STATUS phase=cameras case=%s completed=%d total=%d",
                             name, step + len(action), total)
            capture = sim.call("capture_stop", None)
            gate = compare_traces(reference, load_trace(out / "control_trace.npz"))
            result["cases"][name] = {"verification": gate, "capture": capture,
                                     "absolute_start_step": prefix_steps,
                                     "reference_trace_sha256": file_hash(trace_path)}
            write_json(args.output / "metrics.json", result)
            if not gate["passed"]:
                raise ValueError(f"Extra rendering changed replay {name}: {gate}")
            LOG.info("DONE phase=cases completed=%d total=%d", index, len(jobs))
    result["status"] = "complete"
    write_json(args.output / "metrics.json", result)
    (args.output / "index.html").write_text(
        '<meta charset="utf-8"><h1>Same-state camera comparison</h1>'
        f'<p>Actual action replays, NOT predictions. Branch t=0 is episode step {intervention["rollout_step"]}; '
        'demonstration t=0 is demo start. Policy96 is simulator RGB, not injected demo87 RGB. '
        'Additional cameras are diagnostic only.</p>' + ''.join(
            f'<p><a href="{name}/views.html">{name}</a></p>' for name, _, _ in jobs))
    LOG.info("DONE phase=all completed=%d total=%d", len(jobs), len(jobs))


if __name__ == "__main__":
    main()
