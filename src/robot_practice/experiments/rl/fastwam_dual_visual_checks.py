"""Live dual-view forecasts and paired one-shot demonstration RGB substitution."""

import html
import logging

import numpy as np
from PIL import Image

from robot_practice.experiments.rl.fastwam_dual_train_fit import predict
from robot_practice.experiments.rl.fastwam_image_intervention import substitute_rgb, verify_prefix
from robot_practice.experiments.rl.lightwam_train_scene_eval import summarize, write_json

LOG = logging.getLogger(__name__)


def preclose_frame(actions):
    """A command-onset candidate, not a physical-grasp annotation."""
    close = np.asarray(actions)[:, 6] > 0
    transitions = np.flatnonzero(close[1:] & ~close[:-1]) + 1
    if not len(transitions):
        raise ValueError("No open-to-close transition; select another demonstration")
    return int(transitions[0] - 1)


def select_branch(states, target):
    """Nearest normalized proprio at a real decision boundary; heuristic only."""
    states, target = np.asarray(states), np.asarray(target)
    if states.ndim != 2 or states.shape[1] != 9 or target.shape != (9,):
        raise ValueError("Expected normalized proprio [N,9] and [9]")
    if not np.isfinite(states).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite branch candidates")
    distances = np.mean((states - target) ** 2, axis=1)
    index = int(np.argmin(distances))
    return index * 4, float(np.sqrt(distances[index]))


def save_rgb(path, rgb):
    Image.fromarray(np.asarray(rgb).transpose(1, 2, 0).astype(np.uint8)).save(path)


def forecast(model, adapter, obs, raw, seed, directory):
    directory.mkdir(parents=True, exist_ok=False)
    save_rgb(directory / "POLICY_INPUT.png", obs["rgb"])
    rgb, state = adapter._prepare_observation(obs)
    generated = model.infer_joint(
        prompt=None, input_image=rgb[0].to(model.device),
        proprio=state[0].to(model.device), context=adapter.context.to(model.device),
        context_mask=adapter.context_mask.to(model.device), num_video_frames=9,
        action_horizon=32, action=None, num_inference_steps=20, seed=seed,
        test_action_with_infer_action=False)
    if len(generated["video"]) != 9:
        raise ValueError("Expected nine decoded frames")
    for i, image in enumerate(generated["video"]):
        image.save(directory / f"PRED_plus_{4 * i:02d}.png")
    joint = generated["action"].detach().float().cpu().numpy().reshape(32, 7)
    np.savez_compressed(directory / "prediction.npz", native_action=raw,
                        joint_action=joint, proprio=obs["state"])
    return float(np.max(np.abs(raw - joint)))


def page(directory, rows, start, max_steps):
    parts = ['<!doctype html><meta charset="utf-8"><title>Dual visual checks</title>',
             ('<style>body{font-family:sans-serif}img{width:448px}table{border-collapse:collapse}'
              'td,th{border:1px solid #aaa;padding:4px}section{overflow-x:auto}</style>'),
             '<h1>ACTUAL = simulator; POLICY_INPUT = model input; PRED = generated</h1>',
             ('<p>Each image: agentview left, wrist right. Offsets are 20 Hz control steps. '
             'PRED +0 is reconstruction. After +4 the real rollout replans. '
              'Videos do not control actions; native action inference does.</p>'),
             '<a href="actual_5hz.gif">Full actual rollout (5 Hz)</a>']
    for step in rows:
        name = f"step_{step:04d}"
        parts.append(f'<section id="step_{step}"><h2>Control step {step}</h2><table>')
        parts.append('<tr><th>offset</th>' + ''.join(f'<th>+{d}</th>' for d in range(0, 33, 4))
                     + '</tr>')
        for label in ("PRED", "ACTUAL"):
            cells = []
            for delta in range(0, 33, 4):
                path = (f"{name}/PRED_plus_{delta:02d}.png" if label == "PRED" else
                        f"actual/{step + delta:04d}.png")
                cells.append(f'<td><img loading="lazy" src="{path}"></td>'
                             if (directory / path).exists() else '<td>outside budget</td>')
            parts.append(f'<tr><th>{label}</th>{"".join(cells)}</tr>')
        parts.append(f'</table><p>POLICY_INPUT at {step}: '
                     f'<img src="{name}/POLICY_INPUT.png"></p></section>')
    parts.append(f'<p>Actual continuation interval: {start}..{max_steps}</p>')
    (directory / "index.html").write_text('\n'.join(parts))


def continuation(sim, model, adapter, directory, start, max_steps, seed, replacement=None):
    """Assumes simulator already at start; capture per-control-step physical evidence."""
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "actual").mkdir()
    sim.call("capture_start", str(directory))
    states, actions, rewards, successes, errors, rows = [], [], [], [], [], []
    for step in range(start, max_steps, 4):
        actual = sim.call("dual_observation", None)
        save_rgb(directory / "actual" / f"{step:04d}.png", actual["rgb"])
        obs = (substitute_rgb(actual, replacement)
               if step == start and replacement is not None else actual)
        raw = predict(adapter, obs, seed)
        states.append(actual["state"].copy())
        executed = np.clip(raw[:4], -1, 1)
        # Forecast every decision, but execute only the independently sampled native action.
        errors.append(forecast(model, adapter, obs, raw, seed, directory / f"step_{step:04d}"))
        rows.append(step)
        reply = sim.call("step", (executed, None))
        actions.extend(executed)
        rewards.extend(reply["rewards"])
        successes.extend(reply["successes"])
        LOG.info("STATUS phase=visual_rollout branch=%s completed=%d total=%d",
                 directory.name, step + 4, max_steps)
    save_rgb(directory / "actual" / f"{max_steps:04d}.png",
             sim.call("dual_observation", None)["rgb"])
    capture = sim.call("capture_stop", None)
    frames = [Image.open(directory / "actual" / f"{t:04d}.png").copy()
              for t in range(start, max_steps + 1, 4)]
    frames[0].save(directory / "actual_5hz.gif", save_all=True, append_images=frames[1:],
                   duration=200, loop=0)
    np.savez_compressed(directory / "decisions.npz", states=states, actions=actions,
                        diagnostic_native_action_max_abs=errors)
    page(directory, rows, start, max_steps)
    result = {"start_control_step": start, "summary_of_continuation": summarize(rewards, successes),
              "diagnostic_native_action_max_abs": max(errors), "capture": capture}
    write_json(directory / "metrics.json", result)
    return np.asarray(states), np.asarray(actions), result


def run_visual_checks(sim, model, adapter, dataset, ids, output, seed, max_steps):
    results = {}
    for number, tid in enumerate(ids, 1):
        directory = output / f"trajectory_{tid}"
        directory.mkdir()
        begin = int(sum(dataset.traj_lengths[:tid]))
        end = begin + int(dataset.traj_lengths[tid])
        frame = preclose_frame(dataset.actions[begin:end])
        replacement = dataset.arrays["images"][begin + frame].copy()
        gt_state = dataset.states[begin + frame].copy()
        reset = sim.call("reset_at", (tid, 0))
        initial = sim.call("dual_observation", None)
        image_error = float(np.abs(initial["rgb"].astype(float)
                                   - dataset.arrays["images"][begin]).mean())
        state_error = float(np.max(np.abs(initial["state"] - dataset.states[begin])))
        if (reset["sim_state_max_abs_error"] > 1e-8 or image_error > 10
                or state_error > 1e-3):
            raise ValueError("Initial training-scene observation restoration failed")
        sim.call("reset", tid)
        states, actions, original = continuation(
            sim, model, adapter, directory / "original", 0, max_steps, seed + tid)
        # Leave at least 32 control steps for the continuation when the budget allows.
        step, distance = select_branch(states[:max(1, len(states) - 7)], gt_state)
        # Restore the exact original prefix, including Python controller evolution.
        prefix = directory / "prefix_replay"
        sim.call("reset", tid)
        sim.call("capture_start", str(prefix))
        for t in range(0, step, 4):
            sim.call("step", (actions[t:t + 4], None))
            if t % 40 == 0:
                LOG.info("STATUS phase=prefix_replay completed=%d total=%d", t + 4, step)
        sim.call("capture_stop", None)
        with np.load(directory / "original" / "control_trace.npz") as reference, \
                np.load(prefix / "control_trace.npz") as replay:
            gate = verify_prefix(reference, replay, step)
        live = sim.call("dual_observation", None)
        original_rgb = np.asarray(Image.open(directory / "original" / "actual" /
                                             f"{step:04d}.png")).transpose(2, 0, 1)
        if not np.array_equal(live["rgb"], original_rgb):
            raise ValueError("Dual RGB prefix restoration mismatch")
        _, changed_actions, intervention = continuation(
            sim, model, adapter, directory / "gt_rgb_once", step, max_steps, seed + tid,
            replacement=replacement)
        gt_obs = {"rgb": replacement, "state": gt_state}
        gt_raw = predict(adapter, gt_obs, seed + tid)
        forecast(model, adapter, gt_obs, gt_raw, seed + tid, directory / "gt_input_offline")
        demo_images = []
        for f in range(max(0, frame - 12), min(end - begin, frame + 33)):
            image = Image.fromarray(dataset.arrays["images"][begin + f].transpose(1, 2, 0))
            demo_images.append(image.copy())
        demo_images[0].save(directory / "demo_grasp_20hz.gif", save_all=True,
                            append_images=demo_images[1:], duration=50, loop=0)
        gt_rows = []
        for delta in range(0, 33, 4):
            if begin + frame + delta < end:
                save_rgb(directory / "gt_input_offline" / f"DEMO_plus_{delta:02d}.png",
                         dataset.arrays["images"][begin + frame + delta])
                actual_tag = f'<img src="DEMO_plus_{delta:02d}.png">'
            else:
                actual_tag = 'Outside demonstration'
            gt_rows.append(f'<tr><th>+{delta}</th><td>{actual_tag}</td>'
                           f'<td><img src="PRED_plus_{delta:02d}.png"></td></tr>')
        (directory / "gt_input_offline" / "index.html").write_text(
            '<meta charset="utf-8"><h1>GT RGB + GT proprio: offline prediction, NOT rollout</h1>'
            '<table><tr><th>Control offset</th><th>DEMO actual</th><th>PRED</th></tr>'
            + ''.join(gt_rows) + '</table>')
        results[str(tid)] = {"demo_frame": frame, "rollout_step": step,
                            "selection": "first close-command preframe; nearest normalized proprio",
                            "selection_is_heuristic": True, "normalized_proprio_rmse": distance,
                            "demo_clip_first_frame": max(0, frame - 12),
                            "demo_clip_stride": 1,
                            "demo_proprio": gt_state.tolist(),
                            "rollout_proprio": states[step // 4].tolist(),
                            "initial_image_mae": image_error,
                            "initial_proprio_max_abs_error": state_error,
                            "original_first4_grip": actions[step:step + 4, 6].tolist(),
                            "intervention_first4_grip": changed_actions[:4, 6].tolist(),
                            "prefix_max_abs_errors": gate, "original": original,
                            "gt_rgb_once": intervention}
        write_json(directory / "metrics.json", results[str(tid)])
        title = html.escape(f"trajectory {tid}: demo frame {frame}, rollout step {step}")
        (directory / "index.html").write_text(
            '<meta charset="utf-8"><h1>' + title + '</h1>'
            '<p>Heuristic stage match, not a verified physical grasp annotation.</p>'
            f'<a href="original/index.html#step_{step}">Original prediction at branch point</a><br>'
            '<a href="original/index.html">Original live rollout + predictions</a><br>'
            '<a href="gt_rgb_once/index.html">One-shot GT RGB intervention + predictions</a><br>'
            '<a href="gt_input_offline/index.html">GT input future predictions (offline)</a><br>'
            '<a href="demo_grasp_20hz.gif">Continuous demonstration grasp clip (20 Hz)</a><br>'
            '<a href="metrics.json">Branch point, restore gate and outcomes</a>')
        LOG.info("DONE phase=visual_cases completed=%d total=%d", number, len(ids))
    return {"status": "complete", "evaluated": len(ids), "visual_cases": results}
