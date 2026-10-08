"""Training-subset action fitting and restored-scene rollouts for dual-view FastWAM."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_practice.experiments.rl.lightwam_train_scene_eval import (
    Simulator,
    file_hash,
    summarize,
    write_json,
)

LOG = logging.getLogger(__name__)


def select_windows(lengths, actions, trajectory, count):
    """Evenly sample valid starts and include gripper-transition windows."""
    begin = int(np.sum(lengths[:trajectory]))
    length = int(lengths[trajectory])
    valid = length - 32
    if valid <= 0 or count < 1:
        raise ValueError("Need a positive sample count and a complete 32-action window")
    frames = set(np.linspace(0, valid - 1, min(count, valid), dtype=int).tolist())
    grip = actions[begin:begin + length, 6] > 0
    transitions = np.flatnonzero(grip[1:] != grip[:-1]) + 1
    for transition in transitions:
        for offset in range(4):
            frames.add(int(np.clip(transition - offset, 0, valid - 1)))
    return [begin + frame for frame in sorted(frames)]


def action_errors(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if (prediction.shape != target.shape or prediction.ndim != 3
            or prediction.shape[1:] != (32, 7) or not len(prediction)
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("Expected finite prediction/target arrays [N,32,7]")
    result = {"windows": len(prediction)}
    for name, horizon in (("executed_first4", 4), ("full32", 32)):
        raw, gt = prediction[:, :horizon], target[:, :horizon]
        clipped = np.clip(raw, -1, 1)
        error = clipped - gt
        close = gt[..., 6] > 0
        result[name] = {
            "mse_raw": float(np.mean((raw - gt) ** 2)),
            "mse_clipped": float(np.mean(error ** 2)),
            "arm_mse_clipped": float(np.mean(error[..., :6] ** 2)),
            "arm_mae_clipped": float(np.mean(np.abs(error[..., :6]))),
            "gripper_mse_clipped": float(np.mean(error[..., 6] ** 2)),
            "gripper_sign_accuracy": float(np.mean((clipped[..., 6] > 0) == close)),
            "gripper_close_recall": (
                float(np.mean(clipped[..., 6][close] > 0)) if close.any() else None
            ),
            "gripper_target_close_count": int(close.sum()),
            "clip_rate": float(np.mean(np.abs(raw) > 1)),
        }
    return result


def predict(adapter, observation, seed):
    cond = {key: np.asarray(value)[None, None] for key, value in observation.items()}
    raw = adapter.predict_action(cond, seed=seed).float().cpu().numpy()[0]
    if raw.shape != (32, 7) or not np.isfinite(raw).all():
        raise ValueError("Expected finite dual-view prediction [32,7]")
    return raw


def rollout(sim, trajectory, directory, actions, *, adapter=None, max_steps=600, seed=0):
    from PIL import Image

    directory.mkdir(parents=True)
    sim.call("reset", trajectory)
    total = len(actions) if adapter is None else max_steps
    rewards, successes, frames, predicted = [], [], [], []
    label = "demonstration" if adapter is None else "policy"
    for step in range(0, total, 4):
        obs = sim.call("dual_observation", None)
        frames.append(Image.fromarray(obs["rgb"].transpose(1, 2, 0).astype(np.uint8)))
        if adapter is None:
            executed = actions[step:step + 4]
        else:
            raw = predict(adapter, obs, seed)
            predicted.append(raw)
            executed = np.clip(raw[:4], -1, 1)
        reply = sim.call("step", (executed, step if adapter is None else None))
        rewards.extend(reply["rewards"])
        successes.extend(reply["successes"])
        if step % 40 == 0 or step + len(executed) == total:
            LOG.info("STATUS phase=%s trajectory=%d completed=%d total=%d",
                     label, trajectory, step + len(executed), total)
    frames[0].save(directory / "rollout.gif", save_all=True,
                   append_images=frames[1:], duration=200, loop=0)
    np.savez_compressed(directory / "rollout.npz", predicted=predicted,
                        rewards=rewards, successes=successes)
    result = summarize(rewards, successes)
    result["within_400_steps"] = summarize(rewards[:400], successes[:400])
    result["within_policy_budget"] = summarize(rewards[:max_steps], successes[:max_steps])
    write_json(directory / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-ids", default="0,10,20,30,40,50,60,70")
    parser.add_argument("--windows-per-trajectory", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-pair-only", action="store_true")
    parser.add_argument("--protocol-parity-only", action="store_true")
    parser.add_argument("--autocast-action-only", action="store_true")
    parser.add_argument("--visual-checks-only", action="store_true")
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--egl-device-id", type=int)
    parser.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    parser.add_argument("--normalization",
                        default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = parser.parse_args()
    if sum((args.image_pair_only, args.protocol_parity_only, args.autocast_action_only,
            args.visual_checks_only)) > 1:
        parser.error("Choose one diagnostic mode")
    if ((args.render_backend == "egl" and
         (args.egl_device_id is None or args.egl_device_id < 0)) or
            (args.render_backend == "osmesa" and args.egl_device_id is not None)):
        parser.error("EGL requires a mapped nonnegative --egl-device-id; OSMesa forbids it")
    ids = [int(value) for value in args.trajectory_ids.split(",")]
    if (not ids or len(set(ids)) != len(ids) or args.windows_per_trajectory < 1
            or args.max_steps <= 0 or args.max_steps % 4):
        parser.error("Unique trajectory IDs, positive window count and steps divisible by 4 required")

    import torch
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL, DualViewDataset
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime

    training = json.loads((args.checkpoint.parent.parent / "metrics.json").read_text())
    if training["protocol"].get("observation_action") != PROTOCOL:
        raise ValueError("Expected a dual-view training checkpoint")
    train_args = training["arguments"]
    dataset = DualViewDataset(train_args["dual_data"], train_args["context"])
    if any(i < 0 or i >= len(dataset.traj_lengths) for i in ids):
        parser.error("Trajectory ID outside the training dataset")
    manifest = json.loads((Path(train_args["dual_data"]) / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.output / "run.log")])
    result = {"status": "running", "training_subset_only": True,
              "image_pair_only": args.image_pair_only,
              "protocol_parity_only": args.protocol_parity_only,
              "autocast_action_only": args.autocast_action_only,
              "visual_checks_only": args.visual_checks_only,
              "checkpoint": str(args.checkpoint), "trajectory_ids": ids,
              "seed": args.seed, "noise_seed_rule": "seed + trajectory on each policy call",
              "windows_per_trajectory": args.windows_per_trajectory,
              "protocol": PROTOCOL, "max_steps": args.max_steps,
              "render_backend": args.render_backend, "egl_device_id": args.egl_device_id,
              "open_loop": {}, "scenes": {},
              "requested": len(ids), "evaluated": 0,
              "limitations": ["Selected training trajectories, not all 100 demonstrations.",
                              "Open-loop sampling includes extra gripper-transition windows.",
                              "GIFs contain real simulator observations at 5 Hz."]}
    write_json(args.output / "metrics.json", result)
    try:
        LOG.info("START phase=source_check completed=0 total=2")
        for key in ("hdf5", "normalization"):
            if file_hash(getattr(args, key)) != manifest["source_sha256"][key]:
                raise ValueError(f"Training source mismatch: {key}")
        LOG.info("DONE phase=source_check completed=2 total=2")
        result["checkpoint_sha256"] = file_hash(args.checkpoint)
        cfg = OmegaConf.create({"data": {"npz": train_args["npz"]},
                                "lightwam": {"context": train_args["context"],
                                             "image_size": [224, 448]},
                                "evaluation": {"normalization_path": args.normalization}})
        LOG.info("START phase=model_load completed=0 total=1")
        _model, adapter, result["provenance"] = load_fastwam(args.checkpoint, cfg)
        LOG.info("DONE phase=model_load completed=1 total=1")
        if args.visual_checks_only:
            from robot_practice.experiments.rl.fastwam_dual_eval import validate_rendering_info
            from robot_practice.experiments.rl.fastwam_dual_visual_checks import run_visual_checks

            worker_args = SimpleNamespace(hdf5=args.hdf5, npz=train_args["npz"],
                                          normalization=args.normalization,
                                          trajectory_ids=ids, diagnostic_views=False)
            python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
            with torch.inference_mode(), Simulator(
                    worker_args, args.output, python, root, render_backend=args.render_backend,
                    egl_device_id=args.egl_device_id) as sim:
                result["source"] = sim.audit
                result.update(run_visual_checks(sim, _model, adapter, dataset, ids,
                                                 args.output, args.seed, args.max_steps))
                result["rendering"] = sim.call("rendering_info", None)
                validate_rendering_info(result["rendering"], args.render_backend,
                                        args.egl_device_id)
            result["limitations"] = [
                "New paired rollouts, not an exact replay of a previous evaluation artifact.",
                "Automatic preclose and proprio matches require visual stage verification.",
                "One-shot RGB substitution may contradict physics; no video-to-action causality.",
                "Generated video never controls actions. Real rollout replans every four steps."]
            LOG.info("DONE phase=all completed=%d total=%d", len(ids), len(ids))
            return
        if args.autocast_action_only:
            from robot_practice.experiments.rl.fastwam_autocast_action_check import (
                run_autocast_check,
            )

            result["render_backend"] = None
            result.update(run_autocast_check(adapter, dataset, ids, args.output, args.seed,
                                             args.windows_per_trajectory))
            LOG.info("DONE phase=all completed=%d total=%d windows=%d", len(ids), len(ids),
                     result["autocast_summary"]["windows"])
            return
        if args.protocol_parity_only:
            from robot_practice.experiments.rl.fastwam_protocol_parity import run_parity

            result["render_backend"] = None
            result.update(run_parity(_model, dataset, ids, args.output, args.seed))
            LOG.info("DONE phase=all completed=%d total=%d cases=%d", len(ids), len(ids),
                     len(result["parity_cases"]))
            return
        if args.image_pair_only:
            from robot_practice.experiments.rl.fastwam_dual_eval import validate_rendering_info
            from robot_practice.experiments.rl.fastwam_dual_image_pair import run_pairs

            worker_args = SimpleNamespace(hdf5=args.hdf5, npz=train_args["npz"],
                                          normalization=args.normalization,
                                          trajectory_ids=ids, diagnostic_views=False)
            python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
            with torch.inference_mode(), Simulator(
                    worker_args, args.output, python, root, render_backend=args.render_backend,
                    egl_device_id=args.egl_device_id) as sim:
                result["source"] = sim.audit
                result.update(run_pairs(sim, adapter, dataset, ids, args.output, args.seed))
                result["rendering"] = sim.call("rendering_info", None)
                validate_rendering_info(result["rendering"], args.render_backend,
                                        args.egl_device_id)
            result["limitations"] = [
                "Static demonstration states; no autonomous rollout or causal success test.",
                "Close-command onset is not proof of physical grasp.",
                "Both branches use identical saved proprio, context and noise seed."]
            LOG.info("DONE phase=all completed=%d total=%d pairs=%d", len(ids), len(ids),
                     len(result["pairs"]))
            return
        selections = [(trajectory, start) for trajectory in ids for start in select_windows(
            dataset.traj_lengths, dataset.actions, trajectory, args.windows_per_trajectory)]
        predicted, truth = [], []
        with torch.inference_mode():
            for number, (trajectory, start) in enumerate(selections, 1):
                obs = {"rgb": dataset.arrays["images"][start].copy(),
                       "state": dataset.states[start].copy()}
                predicted.append(predict(adapter, obs, args.seed + trajectory))
                truth.append(dataset.actions[start:start + 32].copy())
                if number % 16 == 0 or number == len(selections):
                    LOG.info("STATUS phase=open_loop completed=%d total=%d", number, len(selections))
            result["open_loop"] = action_errors(predicted, truth)
            result["open_loop"]["per_trajectory"] = {}
            for trajectory in ids:
                selected = [i for i, (t, _) in enumerate(selections) if t == trajectory]
                result["open_loop"]["per_trajectory"][str(trajectory)] = action_errors(
                    np.asarray(predicted)[selected], np.asarray(truth)[selected])
            np.savez_compressed(args.output / "open_loop.npz", predicted=predicted,
                                target=truth, trajectory_and_start=selections)
            write_json(args.output / "metrics.json", result)

            worker_args = SimpleNamespace(hdf5=args.hdf5, npz=train_args["npz"],
                                          normalization=args.normalization,
                                          trajectory_ids=ids, diagnostic_views=False)
            python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
            starts = np.r_[0, np.cumsum(dataset.traj_lengths)]
            with Simulator(worker_args, args.output, python, root,
                           render_backend=args.render_backend,
                           egl_device_id=args.egl_device_id) as sim:
                result["source"] = sim.audit
                for number, trajectory in enumerate(ids, 1):
                    begin, end = map(int, starts[trajectory:trajectory + 2])
                    reset = sim.call("reset_at", (trajectory, 0))
                    obs = sim.call("dual_observation", None)
                    if "rendering" not in result:
                        from robot_practice.experiments.rl.fastwam_dual_eval import (
                            validate_rendering_info,
                        )
                        result["rendering"] = sim.call("rendering_info", None)
                        validate_rendering_info(result["rendering"], args.render_backend,
                                                args.egl_device_id)
                    if obs["rgb"].shape != (3, 224, 448):
                        raise ValueError("Restored dual RGB has incorrect shape")
                    image_diff = np.abs(obs["rgb"].astype(float)
                                        - dataset.arrays["images"][begin])
                    image_error = float(image_diff.mean())
                    view_errors = {"agentview": float(image_diff[:, :, :224].mean()),
                                   "wrist": float(image_diff[:, :, 224:].mean())}
                    state_error = float(np.max(np.abs(obs["state"] - dataset.states[begin])))
                    audit = {"sim_state_max_abs_error": reset["sim_state_max_abs_error"],
                             "image_mae_0_255": image_error,
                             "per_camera_mae_0_255": view_errors,
                             "proprio_normalized_max_abs_error": state_error}
                    row = result["scenes"][str(trajectory)] = {"audit": audit}
                    if (audit["sim_state_max_abs_error"] > 1e-8 or max(view_errors.values()) > 10
                            or state_error > 1e-3):
                        raise ValueError(f"Training observation restoration failed: {trajectory}")
                    actions = dataset.actions[begin:end]
                    row["demonstration"] = rollout(
                        sim, trajectory, args.output / "demonstration" / str(trajectory), actions,
                        max_steps=args.max_steps)
                    row["eligible"] = row["demonstration"]["task_success_any_step"]
                    if row["eligible"]:
                        row["policy"] = rollout(
                            sim, trajectory, args.output / "policy" / str(trajectory), actions,
                            adapter=adapter, max_steps=args.max_steps, seed=args.seed + trajectory)
                        result["evaluated"] += 1
                    write_json(args.output / "metrics.json", result)
                    LOG.info("DONE phase=scenes completed=%d total=%d eligible=%s",
                             number, len(ids), row["eligible"])
        rows = [r["policy"] for r in result["scenes"].values() if "policy" in r]
        result.update(status="complete" if len(rows) == len(ids) else "partial_excluded_scenes",
                      task_success_rate=(float(np.mean([r["task_success_any_step"] for r in rows]))
                                         if rows else None),
                      success_rate_400=(float(np.mean([
                          r["within_400_steps"]["task_success_any_step"] for r in rows]))
                                        if rows else None))
        LOG.info("DONE phase=all completed=%d total=%d success_rate=%s",
                 len(rows), len(ids), result["task_success_rate"])
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        LOG.exception("Training-subset diagnostic failed")
        raise
    finally:
        write_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
