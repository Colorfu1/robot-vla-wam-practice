"""Clearly separated INPUT/PRED/GT/COPY panels around a grasp-stage candidate."""

import argparse
import html
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LOG = logging.getLogger(__name__)


def select_windows(gt_state, gt_action, actual_state, low, high):
    closing = np.flatnonzero(gt_action[:, 6] > 0)
    if not len(closing):
        raise ValueError("No positive demonstration gripper command; specify frames manually")
    gt_anchor = int(closing[0])
    raw_gt = (gt_state + 1) * (high - low + 1e-6) / 2 + low
    raw_actual = (actual_state + 1) * (high - low + 1e-6) / 2 + low
    candidates = np.arange(0, len(actual_state) - 4, 4)
    distance = np.linalg.norm(raw_actual[candidates, :3] - raw_gt[gt_anchor, :3], axis=1)
    actual_anchor = int(candidates[np.argmin(distance)])

    def around(anchor, length, chunk=False):
        return sorted({max(0, min(length - 5, anchor + delta)) // (4 if chunk else 1)
                       * (4 if chunk else 1) for delta in (-16, -8, 0, 8, 16)})

    return {"demo_anchor": gt_anchor, "rollout_anchor": actual_anchor,
            "eef_distance_m": float(distance.min()),
            "demo_frames": around(gt_anchor, len(gt_state)),
            "rollout_frames": around(actual_anchor, len(actual_state), True),
            "alignment_verified": False,
            "selection": "First positive demo gripper command; closest rollout EEF position "
            "at a policy boundary. Candidate only, not verified pose/task-stage alignment."}


def sheet(source, step, first, predictions, truth, path):
    labels = ["INPUT", "PRED", "GT", "COPY"]
    rows = [[first, None, None, None, None], [None] + predictions[1:],
            [None] + truth[1:], [None] + [first] * 4]
    canvas = Image.new("RGB", (1120, 4 * 254), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (label, frames) in enumerate(zip(labels, rows)):
        for offset, frame in enumerate(frames):
            x, y = offset * 224, r * 254
            draw.text((x + 3, y + 3), f"{label} {source} s={step + offset} t+{offset}",
                      fill=(180, 0, 0) if label == "PRED" else (0, 0, 0))
            if frame is not None:
                canvas.paste(frame, (x, y + 28))
    canvas.save(path)


def write_viewer(output, cases):
    """One input frame at a time; never blend predicted and recorded rows."""
    entries = [{"case": key, "source": row["source"], "step": row["control_step"]}
               for key, row in cases.items()]
    payload = json.dumps(entries).replace("<", "\\u003c")
    (output / "index.html").write_text('''<!doctype html><meta charset="utf-8">
<title>Frame-by-frame future prediction</title>
<h1>INPUT / PRED (red) / GT / COPY</h1>
<p>Every case independently conditions on its real input image AND proprioception.
Future GT is never an input. Each prediction covers only the next 4 controls (0.2 s).
This is NOT a long autoregressive prediction. Video is NOT action-conditioned.</p>
<p>First locate the event in GT. Judge anticipation only when that event lies inside t+1..t+4.
Finger closure alone does not establish secure grasp.</p>
<button id="prev">Previous input</button><button id="next">Next input</button>
<input id="slider" type="range" min="0" value="0" style="width:40%">
<strong id="caption"></strong><br>
<label>GT event frame (manual): <input id="event" type="number" value="100"></label>
<span id="coverage"></span><br>
<img id="comparison" style="max-width:100%"><p><a href="all_frames.html">All input windows</a></p>
<script>
const cases=PAYLOAD;
const slider=document.getElementById('slider');slider.max=cases.length-1;
function show(){const c=cases[Number(slider.value)];
 document.getElementById('caption').textContent=c.source+' input '+c.step+
 ' -> predicted/real future '+(c.step+1)+'..'+(c.step+4);
 document.getElementById('comparison').src=c.case+'/comparison.png';
 const e=Number(document.getElementById('event').value), d=e-c.step;
 document.getElementById('coverage').textContent=d>=1&&d<=4 ?
 ' Event IN forecast window: t+'+d : d<=0 ?
 ' Event at/before input: not a pre-event prediction' : 'Event beyond forecast horizon';}
slider.oninput=show;document.getElementById('event').oninput=show;
function move(d){slider.value=Math.max(0,Math.min(cases.length-1,Number(slider.value)+d));show();}
document.getElementById('prev').onclick=()=>move(-1);
document.getElementById('next').onclick=()=>move(1);
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
 if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);});show();
</script>'''.replace("PAYLOAD", payload))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, help="Completed multigpu run")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trajectory", type=int, default=16)
    p.add_argument("--sources", choices=("both", "demo", "rollout"), default="both")
    p.add_argument("--intervention-root", type=Path,
                   help="Completed midstate run; adds genuine autonomous intervention traces")
    p.add_argument("--intervention-frames", default=",".join(map(str, range(41))))
    p.add_argument("--demo-frames", help="Optional comma-separated source demonstration indices")
    p.add_argument("--rollout-frames", help="Control indices; non-boundaries are offline queries")
    args = p.parse_args()
    if args.intervention_root and args.sources != "both":
        p.error("Three-way comparison requires --sources both")
    import torch
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.fastwam_square_diagnostic import (
        frame_metrics,
        image_from_tensor,
    )
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json

    found = []
    for path in args.source.glob("shard_*/metrics.json"):
        meta = json.loads(path.read_text())
        for name, model in meta.get("models", {}).items():
            if str(args.trajectory) in model["episodes"]:
                found.append((path, meta, name, model))
    if len(found) != 1:
        raise ValueError("Expected exactly one recorded model episode")
    path, meta, name, record = found[0]
    episode = record["episodes"][str(args.trajectory)]
    demo = meta["replay"][str(args.trajectory)]
    if not demo["eligible"] or not all(r["prefix_replay_verification"]["passed"]
                                       for r in (episode, demo["rollout"])):
        raise ValueError("Requires eligible demonstration and verified prefix replays")
    directory = path.parent / name / f"trajectory_{args.trajectory}"
    with np.load(directory / "control_trace.npz") as z:
        actual = {k: z[k] for k in ("rgb", "proprio", "actions")}
    cfg = OmegaConf.load(meta["arguments"]["config"])
    cfg.data.npz = meta["arguments"]["npz"]
    with np.load(cfg.data.npz) as z, np.load(meta["arguments"]["normalization"]) as norm:
        starts = np.r_[0, np.cumsum(z["traj_lengths"])]
        begin, end = map(int, starts[args.trajectory:args.trajectory + 2])
        gt = {"rgb": z["images"][begin:end], "proprio": z["states"][begin:end],
              "actions": z["actions"][begin:end]}
        selected = select_windows(gt["proprio"], gt["actions"], actual["proprio"],
                                  norm["obs_min"], norm["obs_max"])
    if file_hash(cfg.data.npz) != meta["source_sha256"]["npz"]:
        raise ValueError("NPZ differs from recorded scene evaluation")
    for source in ("demo", "rollout"):
        override = getattr(args, source + "_frames")
        if override:
            selected[source + "_frames"] = sorted(set(map(int, override.split(","))))
        if args.sources != "both" and args.sources != source:
            selected[source + "_frames"] = []
    sources = [("demo", gt), ("rollout", actual)]
    details = {"demo": {"kind": "GT input, independently conditioned every window",
                        "truth": "Original recorded demonstration future",
                        "time_axis": "Original demonstration frame"},
               "rollout": {"kind": "Original autonomous policy rollout",
                           "truth": "Actual recorded failed rollout future",
                           "time_axis": "Elapsed steps since original episode start"}}
    excluded = {}
    if args.intervention_root:
        from robot_practice.experiments.rl.fastwam_threeway import intervention_sources

        frames = sorted(set(map(int, args.intervention_frames.split(","))))
        extra, descriptions, excluded = intervention_sources(
            args.intervention_root, file_hash(path), args.trajectory, frames)
        sources.extend(extra)
        details.update(descriptions)
        for source, _ in extra:
            selected[source + "_frames"] = frames
    for source, data in sources:
        for t in selected[source + "_frames"]:
            if t < 0 or t + 4 >= len(data["rgb"]):
                raise ValueError(f"Invalid {source} frame {t}")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    seed = int(meta["arguments"]["seed"])
    result = {"status": "running", "selection": selected, "cases": {}, "seed": seed,
              "source_details": details, "excluded_interventions": excluded,
              "trajectory": args.trajectory, "source_metrics_sha256": file_hash(path),
              "checkpoint_sha256": file_hash(record["checkpoint"]),
              "protocol": {"future_gt_used_for_generation": False, "action_conditioned": False,
                           "sources": args.sources, "independent_real_input_per_window": True,
                           "denoising_steps": 20, "future_frames": 4,
                           "demo_gt": "Original training NPZ consecutive frames",
                           "rollout_gt": "Recorded live simulator consecutive control steps"},
              "limitations": ["Automatic anchors are candidates, not verified alignment.",
                              ("Video is action-unconditioned; recorded actions may differ from "
                               "joint predicted actions, especially for demonstration inputs."),
                              "Global image MAE does not measure grasp correctness."]}
    write_json(args.output / "metrics.json", result)
    LOG.info("START phase=load")
    model, adapter, provenance = load_fastwam(record["checkpoint"], cfg)
    result["provenance"] = provenance
    total = sum(len(selected[s + "_frames"]) for s, _ in sources)
    links = []
    with torch.inference_mode():
        for source, data in sources:
            for t in selected[source + "_frames"]:
                case = f"{source}_step_{t:04d}"
                out = args.output / case
                out.mkdir()
                rgb, state = adapter._prepare_observation(
                    {"rgb": data["rgb"][t], "state": data["proprio"][t]})
                generated = model.infer_joint(
                    prompt=None, input_image=rgb[0].to(model.device),
                    proprio=state[0].to(model.device), context=adapter.context.to(model.device),
                    context_mask=adapter.context_mask.to(model.device), num_video_frames=5,
                    action_horizon=4, action=None, num_inference_steps=20, seed=seed,
                    test_action_with_infer_action=True,
                )
                # Future observations are processed only AFTER inference returns.
                truth = []
                for offset in range(5):
                    image, _ = adapter._prepare_observation(
                        {"rgb": data["rgb"][t + offset], "state": data["proprio"][t + offset]})
                    truth.append(image_from_tensor(image[0]))
                predictions = generated["video"]
                first = image_from_tensor(rgb[0])
                display_source = source.replace("intervention_", "RESET")
                sheet(display_source, t, first, predictions, truth, out / "comparison.png")
                first.save(out / "INPUT.png")
                for offset in range(1, 5):
                    predictions[offset].save(out / f"PRED_tplus{offset}.png")
                    truth[offset].save(out / f"GT_tplus{offset}.png")
                action = generated["action"].detach().float().cpu().numpy().reshape(4, 7)
                np.savez_compressed(out / "predictions.npz",
                                    predicted_rgb=np.stack([np.asarray(x) for x in predictions[1:]]),
                                    gt_rgb=np.stack([np.asarray(x) for x in truth[1:]]),
                                    predicted_action=action, reference_actions=data["actions"][t:t+4])
                result["cases"][case] = {
                    "source": source, "control_step": t,
                    "query_kind": ("independent_GT_input" if source == "demo" else
                                   "policy_boundary_requery" if t % 4 == 0 else
                                   "offline_between_policy_calls"),
                    "future_frames": frame_metrics(predictions, truth, first),
                    "reference_gripper_commands": data["actions"][t:t+4, 6].tolist(),
                    "predicted_gripper_commands": action[:, 6].tolist(),
                    "clipped_action_mae_vs_reference": float(np.abs(
                        np.clip(action, -1, 1) - data["actions"][t:t+4]).mean())}
                write_json(args.output / "metrics.json", result)
                links.append(f'<h2>{html.escape(case)}</h2><img style="max-width:100%" '
                             f'src="{case}/comparison.png">')
                LOG.info("STATUS phase=preview completed=%d total=%d case=%s",
                         len(result["cases"]), total, case)
    (args.output / "all_frames.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>INPUT / PRED / GT / COPY</h1>'
        '<p>PRED is generated. GT is recorded, never model output. '
        'Demo and rollout have separate timelines; anchors are not verified alignment. '
        'All future columns span 50–200 ms. Video is NOT action-conditioned.</p>' + ''.join(links))
    if args.intervention_root:
        from robot_practice.experiments.rl.fastwam_threeway import threeway_viewer

        threeway_viewer(args.output, result["cases"], details, excluded)
    else:
        write_viewer(args.output, result["cases"])
    result["status"] = "complete"
    write_json(args.output / "metrics.json", result)
    LOG.info("DONE phase=all completed=%d total=%d", total, total)


if __name__ == "__main__":
    main()
