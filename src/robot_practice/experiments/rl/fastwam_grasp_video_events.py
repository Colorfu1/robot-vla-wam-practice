"""CPU-only event extraction from recorded state and already generated FastWAM MP4s."""

import argparse
import html
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from robot_practice.experiments.rl.fastwam_rollout_video import case_name, ffmpeg_binary
from robot_practice.experiments.rl.lightwam_train_scene_eval import write_json

log = logging.getLogger(__name__)


def find_events(states, actions, obs_min, obs_max):
    """Select by actual state change only, never by prediction error or appearance."""
    if states.shape != (150, 9) or actions.shape != (150, 4, 7):
        raise ValueError("Expected 150 pre-action observations and 150 action chunks")
    # Invert the installed DPPO wrapper including its epsilon; do not use final reset state.
    raw = (states + 1) * 0.5 * (obs_max - obs_min + 1e-6) + obs_min
    aperture = np.abs(raw[:, 7] - raw[:, 8]) * 1000
    closing = aperture[:-1] - aperture[1:]
    upward = np.diff(raw[:, 2]) * 1000
    gripper = actions.mean(axis=1)[:, -1]
    close = int(np.argmax(closing))
    has_close = bool(closing[close] >= 5)
    candidates = np.flatnonzero((upward >= 5) & (gripper[:-1] > 0.5) & (aperture[:-1] < 50))
    if has_close:
        candidates = candidates[candidates >= close]
    events = []
    if has_close:
        events.append(("strongest_closing", close))
    if len(candidates):
        events.append(("upward_eef_candidate", int(candidates[np.argmax(upward[candidates])])))
    if not events:
        events.append(("no_event_reference", 50))
    rows = []
    for kind, chunk in events:
        rows.append({"kind": kind, "chunk": chunk, "video_seconds": chunk / 5,
                     "control_step_before": chunk * 4, "control_step_after": chunk * 4 + 4,
                     "joint_position_difference_before_mm": float(aperture[chunk]),
                     "joint_position_difference_after_mm": float(aperture[chunk+1]),
                     "closing_mm": float(closing[chunk]),
                     "eef_upward_mm": float(upward[chunk]),
                     "mean_gripper_action": float(gripper[chunk])})
    return {"has_closing_above_5mm": has_close,
            "has_upward_candidate": bool(len(candidates)),
            "max_closing_mm": float(closing.max()), "max_eef_upward_mm": float(upward.max()),
            "events": rows}


def export_event(binary, source, destination, chunk):
    destination.mkdir()
    start, end = max(0, chunk - 2), min(149, chunk + 2)
    for basename in ("comparison_5hz", "forecast_strip_5hz"):
        video = source / f"{basename}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        # Select by frame index, not keyframe seeking. One chunk per second for inspection.
        subprocess.run([
            binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(video),
            "-vf", f"select='between(n,{start},{end})',setpts=N/TB", "-r", "1", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination / f"{basename}_slow.mp4"),
        ], check=True)
        subprocess.run([
            binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(video),
            "-vf", f"select='eq(n,{chunk})',scale=iw*2:ih*2:flags=neighbor",
            "-frames:v", "1", str(destination / f"{basename}_zoom.png"),
        ], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-root", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--normalization", default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = p.parse_args()
    root, eval_root, output = Path(args.video_root), Path(args.eval_root), Path(args.output)
    manifest = json.loads((root / "metrics.json").read_text())
    evaluation = json.loads((eval_root / "metrics.json").read_text())
    if manifest["status"] != "complete" or evaluation["status"] != "complete":
        raise ValueError("Video generation and source evaluation must be complete")
    if manifest["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
        raise ValueError("Checkpoint provenance mismatch")
    with np.load(args.normalization) as data:
        low, high = data["obs_min"].copy(), data["obs_max"].copy()
    binary = ffmpeg_binary()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "run.log"),
    ])
    result = {"status": "running", "cases": {}, "arguments": vars(args),
              "limitations": ["EEF ascent is not proof that the object was lifted.",
                              "Finger joint position difference is not measured object width.",
                              "Thresholds are diagnostic screens, not grasp detection.",
                              "Zoom is nearest-neighbor display enlargement, not new detail."]}
    write_json(output / "metrics.json", result)
    sections = []
    try:
        for index, row in enumerate(manifest["cases"], 1):
            seed, name = row["environment_seed"], case_name(row)
            sources = []
            for trace in eval_root.glob("gpu_*/batch_*/rollout_trace.npz"):
                with np.load(trace) as data:
                    ids = data["environment_seeds"].tolist()
                    if seed in ids:
                        i = ids.index(seed)
                        sources.append((data["states"][:150, i, -1].copy(),
                                        data["action_executed"][:, i].copy()))
            if len(sources) != 1:
                raise ValueError(f"Expected exactly one trace for {seed}")
            case = find_events(*sources[0], low, high)
            result["cases"][name] = case
            source_metrics = json.loads((root / name / "metrics.json").read_text())
            sections.append(f"<h2>{name}</h2><p>"
                            f"Closing &gt;=5mm: {case['has_closing_above_5mm']}; "
                            f"upward candidate: {case['has_upward_candidate']}</p>")
            for event in case["events"]:
                chunk = event["chunk"]
                event["image_metrics"] = source_metrics["chunks"][chunk]
                label = f"{name}_{event['kind']}_chunk_{chunk:04d}"
                export_event(binary, root / name, output / label, chunk)
                write_json(output / label / "event.json", event)
                sections.append(
                    f"<h3>{label}</h3><pre>{html.escape(json.dumps(event, indent=2))}</pre>"
                    f'<a href="{label}/comparison_5hz_zoom.png">Full-size comparison</a><br>'
                    f'<img width="100%" style="max-width:1344px" '
                    f'src="{label}/comparison_5hz_zoom.png"><br>'
                    f'<a href="{label}/forecast_strip_5hz_zoom.png">Full-size five-frame strip</a><br>'
                    f'<img width="100%" src="{label}/forecast_strip_5hz_zoom.png"><br>'
                    f'<video controls preload="metadata" width="672" '
                    f'src="{label}/comparison_5hz_slow.mp4"></video><br>'
                    f'<video controls preload="metadata" width="100%" '
                    f'src="{label}/forecast_strip_5hz_slow.mp4"></video>'
                )
            write_json(output / "metrics.json", result)
            log.info("STATUS phase=extract completed=%d total=%d case=%s events=%d",
                     index, len(manifest["cases"]), name, len(case["events"]))
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Grasp event inspection</title>'
            '<h1>State-selected closing and upward-motion candidates</h1>'
            '<p>Three columns: actual before / predicted after 4 steps / actual after 4 steps. '
            'Slow clips show five neighboring chunks at 1 chunk/second. '
            'Upward EEF motion does not establish object lift. No-event references are not grasps.</p>'
            + "\n".join(sections)
        )
        result["status"] = "complete"
        log.info("DONE phase=extract completed=%d total=%d", len(result["cases"]), len(manifest["cases"]))
    except BaseException:
        result["status"] = "failed"
        log.exception("Extraction failed; partial outputs retained")
        raise
    finally:
        write_json(output / "metrics.json", result)


if __name__ == "__main__":
    main()
