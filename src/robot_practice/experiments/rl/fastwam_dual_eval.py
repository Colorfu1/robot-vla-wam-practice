"""Dual-view live closed-loop evaluation: predict 32, execute only 4, 600-step episodes."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LOG = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    p.add_argument("--normalization", default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = p.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.max_steps % 4:
        p.error("Positive episodes and max-steps divisible by 4 required")
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL, executed_actions
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_base import _resolve_dppo_runtime
    from robot_practice.experiments.rl.lightwam_train_scene_eval import (
        Simulator,
        file_hash,
        summarize,
        write_json,
    )

    training = json.loads((args.checkpoint.parent.parent / "metrics.json").read_text())
    if training["protocol"].get("observation_action") != PROTOCOL:
        raise ValueError("Not a dual-view 32/4/9/4 checkpoint")
    data = json.loads((Path(training["arguments"]["dual_data"]) / "manifest.json").read_text())
    for key in ("hdf5", "normalization"):
        if file_hash(getattr(args, key)) != data["source_sha256"][key]:
            raise ValueError(f"Changed dataset source {key}")
    cfg = OmegaConf.create({"data": {"npz": training["arguments"]["npz"]},
                           "lightwam": {"context": training["arguments"]["context"],
                                        "image_size": [224, 448]},
                           "evaluation": {"normalization_path": args.normalization}})
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    _model, adapter, provenance = load_fastwam(args.checkpoint, cfg)
    result = {"status": "running", "protocol": PROTOCOL, "provenance": provenance,
              "initialization": "seeded random reset; NOT demonstration state",
              "checkpoint_sha256": file_hash(args.checkpoint), "episodes": []}
    write_json(args.output / "metrics.json", result)
    worker_args = SimpleNamespace(hdf5=args.hdf5, npz=cfg.data.npz, normalization=args.normalization,
                                  trajectory_ids=[0], diagnostic_views=False)
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    with Simulator(worker_args, args.output, python, root) as sim, torch.inference_mode():
        for episode in range(args.episodes):
            seed = args.seed + episode
            sim.call("reset_seed", seed)
            rewards, successes, predictions, pictures = [], [], [], []
            for step in range(0, args.max_steps, 4):
                obs = sim.call("dual_observation", None)
                pictures.append(Image.fromarray(obs["rgb"].transpose(1, 2, 0)))
                cond = {k: v[None, None] for k, v in obs.items()}
                raw = adapter.predict_action(cond, seed=seed).float().cpu().numpy()[0]
                if not np.isfinite(raw).all():
                    raise ValueError("Nonfinite prediction")
                action = np.clip(executed_actions(raw), -1, 1)
                predictions.append(raw)
                reply = sim.call("step", (action, None))
                rewards.extend(reply["rewards"])
                successes.extend(reply["successes"])
                if step % 40 == 0 or step + 4 == args.max_steps:
                    LOG.info("STATUS phase=rollout episode=%d completed=%d total=%d",
                             episode, step + 4, args.max_steps)
            out = args.output / f"seed_{seed}"
            out.mkdir()
            pictures[0].save(out / "actual.gif", save_all=True, append_images=pictures[1:], duration=200, loop=0)
            np.savez_compressed(out / "actions.npz", predicted=predictions, rewards=rewards, successes=successes)
            result["episodes"].append(dict(summarize(rewards, successes), seed=seed,
                                           within_400=summarize(rewards[:400], successes[:400])))
            write_json(args.output / "metrics.json", result)
            LOG.info("DONE phase=episodes completed=%d total=%d", episode + 1, args.episodes)
    result.update(status="complete", task_success_rate=float(np.mean(
        [row["task_success_any_step"] for row in result["episodes"]])))
    write_json(args.output / "metrics.json", result)
    LOG.info("DONE phase=all completed=%d total=%d", args.episodes, args.episodes)


if __name__ == "__main__":
    main()
