"""Restore selected demonstration states and photograph them without stepping."""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory", type=int, default=16)
    p.add_argument("--frames", default="85,100,117")
    p.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    p.add_argument("--npz", default="datasets/raw/dppo/robomimic/square-img/train.npz")
    p.add_argument("--normalization", default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = p.parse_args()
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime
    from robot_practice.experiments.rl.lightwam_train_scene_eval import Simulator, write_json
    from robot_practice.experiments.rl.scene_capture import compare_traces

    frames = sorted(set(map(int, args.frames.split(","))))
    with np.load(args.npz) as z:
        lengths = z["traj_lengths"]
        if not 0 <= args.trajectory < len(lengths):
            p.error("Invalid trajectory")
        start = int(sum(lengths[:args.trajectory]))
        if not frames or any(t < 0 or t >= lengths[args.trajectory] for t in frames):
            p.error("Invalid demonstration frame")
        images = {t: z["images"][start + t].copy() for t in frames}
        states = {t: z["states"][start + t].copy() for t in frames}
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    worker_args = SimpleNamespace(**vars(args), trajectory_ids=[args.trajectory], diagnostic_views=True)
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    result = {"status": "running", "trajectory": args.trajectory, "frames": {},
              "protocol": "Static demo state restoration; zero actions; no policy loaded"}
    write_json(args.output / "metrics.json", result)
    sections = []
    with Simulator(worker_args, args.output, python, root) as sim:
        result["source"] = sim.audit
        for index, t in enumerate(frames, 1):
            out = args.output / f"frame_{t:04d}"
            out.mkdir()
            reset = sim.call("reset_at", (args.trajectory, t))
            gate = {"sim_state_error": reset["sim_state_max_abs_error"],
                    "proprio_error": float(np.abs(reset["obs"]["state"] - states[t]).max())}
            if gate["sim_state_error"] > 1e-8 or gate["proprio_error"] > 1e-3:
                raise ValueError(f"State restoration failed: {gate}")
            # Reserved prefix_replay name disables extra rendering for baseline capture.
            sim.call("capture_start", str(out / "prefix_replay"))
            sim.call("capture_stop", None)
            sim.call("capture_start", str(out))
            capture = sim.call("capture_stop", None)
            # Observe again after rendering, without reset or action.
            sim.call("capture_start", str(out / "after" / "prefix_replay"))
            sim.call("capture_stop", None)
            with np.load(out / "prefix_replay/control_trace.npz") as a, np.load(
                    out / "after/prefix_replay/control_trace.npz") as b:
                unchanged = compare_traces(a, b)
            if not unchanged["passed"]:
                raise ValueError(f"Rendering changed state: {unchanged}")
            rgb = images[t].transpose(1, 2, 0).astype(np.uint8)
            Image.fromarray(rgb).save(out / "TRAIN_original96.png")
            result["frames"][str(t)] = {"restore_gate": gate, "render_noninterference": unchanged,
                                       "views": capture["diagnostic_views"]}
            labels = [("Original training 96", "TRAIN_original96.png"),
                      ("Restored native 96", "views/0000_fresh96.png"),
                      ("Native agentview 224", "views/0000_agent224.png")]
            wrist = capture["diagnostic_views"]["wrist_camera"]
            if wrist:
                labels.append((f"Native wrist 224: {wrist}", "views/0000_wrist224.png"))
            sections.append(f'<h2>Demo frame {t}</h2><div style="display:flex;gap:12px">' + ''.join(
                f'<div><p>{label}</p><img width="224" height="224" '
                f'style="image-rendering:pixelated" src="{out.name}/{file}"></div>'
                for label, file in labels) + ('<p>Wrist unavailable</p>' if not wrist else '') + '</div>')
            write_json(args.output / "metrics.json", result)
            LOG.info("STATUS phase=static completed=%d total=%d frame=%d", index, len(frames), t)
    (args.output / "index.html").write_text(
        '<meta charset="utf-8"><h1>Static same-state camera comparison</h1>'
        '<p>No rollout, actions or policy inference. First column is stored training RGB; '
        'others are restored-state renders. 96px images enlarged with nearest-neighbor, '
        'not additional detail. Compare columns 2/3 for resolution, 3/4 for viewpoint.</p>'
        + ''.join(sections))
    result["status"] = "complete"
    write_json(args.output / "metrics.json", result)
    LOG.info("DONE phase=all completed=%d total=%d", len(frames), len(frames))


if __name__ == "__main__":
    main()
