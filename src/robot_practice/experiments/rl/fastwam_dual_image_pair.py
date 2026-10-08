"""Same demonstration state/noise, saved RGB versus freshly rendered RGB."""

import logging

import numpy as np

from robot_practice.experiments.rl.fastwam_dual_train_fit import action_errors, predict
from robot_practice.experiments.rl.lightwam_train_scene_eval import write_json

LOG = logging.getLogger(__name__)


def stage_frames(actions):
    """Sample around close-command onsets, not presumed physical grasp events."""
    close = np.asarray(actions)[:, 6] > 0
    onsets = np.flatnonzero(close & np.r_[True, ~close[:-1]])
    labels = {0: ["initial"]}
    for onset in onsets:
        for offset in (-16, -8, -4, -1, 0, 1, 4, 8):
            frame = int(onset + offset)
            if 0 <= frame < len(actions) - 32:
                labels.setdefault(frame, []).append(f"close_command_{onset}:{offset:+d}")
    return sorted(labels.items())


def paired_predict(adapter, saved_rgb, live_rgb, state, seed):
    if (saved_rgb.shape != (3, 224, 448) or live_rgb.shape != saved_rgb.shape or
            saved_rgb.dtype != np.uint8 or live_rgb.dtype != np.uint8):
        raise ValueError("Expected two uint8 dual-view RGB images")
    saved = predict(adapter, {"rgb": saved_rgb.copy(), "state": state.copy()}, seed)
    live = predict(adapter, {"rgb": live_rgb.copy(), "state": state.copy()}, seed)
    repeat = predict(adapter, {"rgb": saved_rgb.copy(), "state": state.copy()}, seed)
    return saved, live, float(np.max(np.abs(saved - repeat)))


def summary(saved, live, targets):
    return {"saved_rgb": action_errors(saved, targets),
            "rendered_rgb": action_errors(live, targets),
            "paired_action_change_first4_mae": float(np.mean(np.abs(
                np.clip(saved, -1, 1)[:, :4] - np.clip(live, -1, 1)[:, :4])))}


def run_pairs(sim, adapter, dataset, ids, output, seed):
    from PIL import Image, ImageDraw

    starts = np.r_[0, np.cumsum(dataset.traj_lengths)]
    rows, saved_predictions, live_predictions, targets = [], [], [], []
    selections = [(tid, frame, labels) for tid in ids for frame, labels in stage_frames(
        dataset.actions[int(starts[tid]):int(starts[tid + 1])])]
    for number, (tid, frame, labels) in enumerate(selections, 1):
        start = int(starts[tid]) + frame
        restored = sim.call("reset_at", (tid, frame))
        obs = sim.call("dual_observation", None)
        state = dataset.states[start].copy()
        state_error = float(np.max(np.abs(obs["state"] - state)))
        if restored["sim_state_max_abs_error"] > 1e-8 or state_error > 1e-3:
            raise ValueError(f"State restoration mismatch at trajectory={tid} frame={frame}")
        saved_rgb = dataset.arrays["images"][start].copy()
        saved, live, repeat_error = paired_predict(adapter, saved_rgb, obs["rgb"],
                                                  state, seed + tid)
        if repeat_error > 1e-5:
            raise ValueError(f"Same-input repeat is not reproducible: {repeat_error}")
        target = dataset.actions[start:start + 32].copy()
        diff = np.abs(saved_rgb.astype(float) - obs["rgb"].astype(float))
        row = {"trajectory": tid, "frame": frame, "labels": labels,
               "sim_state_max_abs_error": restored["sim_state_max_abs_error"],
               "proprio_max_abs_error": state_error, "same_input_repeat_max_abs": repeat_error,
               "image_mae": {"agentview": float(diff[:, :, :224].mean()),
                             "wrist": float(diff[:, :, 224:].mean())},
               "first4_gripper": {"target": target[:4, 6].tolist(),
                                  "saved_rgb": saved[:4, 6].tolist(),
                                  "rendered_rgb": live[:4, 6].tolist()},
               **summary(saved[None], live[None], target[None])}
        directory = output / f"trajectory_{tid}" / f"frame_{frame:04d}"
        directory.mkdir(parents=True)
        canvas = Image.new("RGB", (448, 496), "white")
        draw = ImageDraw.Draw(canvas)
        for y, title, rgb in ((0, "Saved training RGB", saved_rgb),
                               (248, "EGL RGB at identical demo state", obs["rgb"])):
            draw.text((4, y + 4), title, fill="black")
            canvas.paste(Image.fromarray(rgb.transpose(1, 2, 0)), (0, y + 24))
        canvas.save(directory / "input_comparison.png")
        write_json(directory / "metrics.json", row)
        rows.append(row)
        saved_predictions.append(saved)
        live_predictions.append(live)
        targets.append(target)
        LOG.info("STATUS phase=image_pair completed=%d total=%d trajectory=%d frame=%d",
                 number, len(selections), tid, frame)
    np.savez_compressed(output / "image_pairs.npz", saved=saved_predictions,
                        live=live_predictions, target=targets,
                        trajectory_frame=[(r["trajectory"], r["frame"]) for r in rows])
    return {"pairs": rows, "paired_summary": summary(np.asarray(saved_predictions),
            np.asarray(live_predictions), np.asarray(targets)), "status": "complete",
            "evaluated": len(ids), "requested": len(ids)}
