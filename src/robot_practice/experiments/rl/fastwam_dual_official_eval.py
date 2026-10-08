"""FastWAM dual-view policy on DPPO's Square vector-reset evaluation protocol.

Each child evaluates a contiguous subset of the official 20 environments. The
environment bridge uses the upstream image and MultiStep wrappers, including
automatic reset after every 400 control steps. Camera channels are rearranged
from the wrapper's 6x224x224 layout to the model's 3x224x448 layout.
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)
OFFICIAL_ENVS = 20
OFFICIAL_SEED = 42
OFFICIAL_POLICY_STEPS = 400
POLICY_STEPS_PER_EPISODE = 100


def pack_dual_observation(observation):
    """Preserve agentview-left/wrist-right order used by the training export."""
    rgb = np.asarray(observation["rgb"])
    state = np.asarray(observation["state"])
    if rgb.ndim != 5 or rgb.shape[1:] != (1, 6, 224, 224):
        raise ValueError(f"Expected vector RGB [N,1,6,224,224], got {rgb.shape}")
    if state.shape != (rgb.shape[0], 1, 9):
        raise ValueError(f"Expected vector proprio [N,1,9], got {state.shape}")
    if not np.isfinite(rgb).all() or not np.isfinite(state).all():
        raise ValueError("Nonfinite dual-view observation")
    return {
        "rgb": np.concatenate((rgb[:, :, :3], rgb[:, :, 3:]), axis=-1),
        "state": state,
    }


def validate_official_shard(seed, env_start, env_count, policy_steps):
    if seed != OFFICIAL_SEED or policy_steps not in (POLICY_STEPS_PER_EPISODE,
                                                    OFFICIAL_POLICY_STEPS):
        raise ValueError("Official Square uses seed=42 and 400 policy steps (100 for smoke)")
    if env_count < 1 or env_start < 0 or env_start + env_count > OFFICIAL_ENVS:
        raise ValueError("Environment shard must lie within official indices 0..19")
    return policy_steps == OFFICIAL_POLICY_STEPS


def summarize_shard(rewards, firsts, env_start, seed, policy_steps):
    from robot_practice.experiments.rl.lightwam_base import _summarize_official_square_rollout

    reward_array = np.asarray(rewards, dtype=np.float64)
    first_array = np.asarray(firsts, dtype=bool)
    expected = list(range(0, policy_steps + 1, POLICY_STEPS_PER_EPISODE))
    if reward_array.shape != (policy_steps, first_array.shape[1]):
        raise ValueError("Reward or firsts shape mismatch")
    for local_index in range(reward_array.shape[1]):
        actual = np.flatnonzero(first_array[:, local_index]).tolist()
        if actual != expected:
            raise ValueError(f"Unexpected reset boundaries env={env_start + local_index}: "
                             f"{actual}, expected {expected}")
    rows = _summarize_official_square_rollout(
        reward_array, first_array, act_steps=4, success_threshold=1.0
    )
    expected_rows = reward_array.shape[1] * policy_steps // POLICY_STEPS_PER_EPISODE
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} complete episodes, got {len(rows)}")
    seen = set()
    for row in rows:
        local_index = int(row["env_index"])
        row["env_index"] = env_start + local_index
        row["environment_seed"] = seed + row["env_index"]
        row["episode_sequence"] = row["start_policy_step"] // POLICY_STEPS_PER_EPISODE
        key = row["env_index"], row["episode_sequence"]
        if key in seen:
            raise ValueError(f"Duplicate official episode {key}")
        seen.add(key)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-start", type=int, required=True)
    parser.add_argument("--env-count", type=int, required=True)
    parser.add_argument("--egl-device-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=OFFICIAL_SEED)
    parser.add_argument("--policy-steps", type=int, default=OFFICIAL_POLICY_STEPS)
    parser.add_argument("--dppo-root", type=Path, default=Path("third_party/dppo_official"))
    parser.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    parser.add_argument("--normalization",
                        default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = parser.parse_args()
    try:
        formal = validate_official_shard(args.seed, args.env_start, args.env_count,
                                         args.policy_steps)
    except ValueError as exc:
        parser.error(str(exc))
    if args.egl_device_id < 0:
        parser.error("Mapped EGL device id must be nonnegative")

    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from robot_practice.experiments.rl.dppo_square_official import REVISION
    from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL, executed_actions
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
    from robot_practice.experiments.rl.lightwam_base import (
        _DPPOSquareEnvClient,
        _resolve_dppo_runtime,
    )
    from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json

    training = json.loads((args.checkpoint.parent.parent / "metrics.json").read_text())
    if training["protocol"].get("observation_action") != PROTOCOL:
        raise ValueError("Expected dual-view FastWAM checkpoint")
    dual_data = Path(training["arguments"]["dual_data"])
    manifest = json.loads((dual_data / "manifest.json").read_text())
    for key in ("hdf5", "normalization"):
        if file_hash(getattr(args, key)) != manifest["source_sha256"][key]:
            raise ValueError(f"Changed dataset source {key}")
    cfg = OmegaConf.create({"data": {"npz": training["arguments"]["npz"]},
                           "lightwam": {"context": training["arguments"]["context"],
                                        "image_size": [224, 448]},
                           "evaluation": {"normalization_path": args.normalization}})
    root = args.dppo_root.resolve()
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != REVISION or dirty:
        raise ValueError("Expected clean pinned official DPPO checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.output / "run.log")], force=True)
    expected_episodes = args.env_count * args.policy_steps // POLICY_STEPS_PER_EPISODE
    result = {
        "status": "running", "evaluation_mode": "official_dppo" if formal else "smoke",
        "protocol": PROTOCOL, "checkpoint_sha256": file_hash(args.checkpoint),
        "dppo_revision": revision,
        "arguments": {"checkpoint": str(args.checkpoint), "env_start": args.env_start,
                      "env_count": args.env_count, "seed": args.seed,
                      "policy_steps": args.policy_steps, "egl_device_id": args.egl_device_id},
        "official_environment": {"seed": OFFICIAL_SEED, "envs": OFFICIAL_ENVS,
                                 "max_episode_steps": 400, "action_steps": 4,
                                 "auto_reset": True, "reward_threshold": 1.0},
        "requested": expected_episodes, "evaluated": 0, "episodes": [],
        "policy_steps_completed": 0,
    }
    write_json(args.output / "metrics.json", result)
    python, root = _resolve_dppo_runtime(
        dppo_python_value=None, dppo_root_value=str(root)
    )
    started = time.monotonic()
    try:
        LOG.info("START phase=model completed=0 total=1")
        _model, adapter, provenance = load_fastwam(args.checkpoint, cfg)
        result["provenance"] = provenance
        write_json(args.output / "metrics.json", result)
        LOG.info("DONE phase=model completed=1 total=1")
        worker_log = args.output / "env_worker.log"
        with _DPPOSquareEnvClient(
            dppo_python=python, dppo_root=root, normalization_path=Path(args.normalization),
            num_envs=args.env_count, max_episode_steps=400, action_horizon=4,
            policy_seed=args.seed + args.env_start, worker_log=worker_log,
            dual_view=True, egl_device_id=args.egl_device_id,
        ) as environment, torch.inference_mode():
            expected_seeds = list(range(args.seed + args.env_start,
                                        args.seed + args.env_start + args.env_count))
            if environment.environment_seeds != expected_seeds:
                raise ValueError("Official environment seed assignment mismatch")
            observation = environment.reset()
            first_image = pack_dual_observation(observation)["rgb"][0, 0]
            Image.fromarray(np.clip(first_image.transpose(1, 2, 0), 0, 255)
                            .astype(np.uint8)).save(args.output / "first_observation.png")
            rewards = np.zeros((args.policy_steps, args.env_count), dtype=np.float64)
            firsts = np.zeros((args.policy_steps + 1, args.env_count), dtype=bool)
            firsts[0] = True
            for step in range(args.policy_steps):
                cond = pack_dual_observation(observation)
                predicted = adapter.predict_action(cond, seed=expected_seeds)
                raw = predicted.float().cpu().numpy()
                if raw.shape != (args.env_count, 32, 7) or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid FastWAM action shape/values: {raw.shape}")
                action = np.clip(executed_actions(raw), -1, 1)
                observation, reward, terminated, truncated = environment.step(action)
                rewards[step] = reward
                firsts[step + 1] = np.logical_or(terminated, truncated)
                if (step + 1) % 10 == 0 or step + 1 == args.policy_steps:
                    result["policy_steps_completed"] = step + 1
                    write_json(args.output / "metrics.json", result)
                    LOG.info("STATUS phase=rollout completed=%d total=%d envs=%d",
                             step + 1, args.policy_steps, args.env_count)
            rows = summarize_shard(rewards, firsts, args.env_start, args.seed,
                                   args.policy_steps)
            np.savez_compressed(args.output / "rollout_trace.npz", rewards=rewards,
                                episode_boundaries=firsts)
            result.update(status="complete", evaluated=len(rows), episodes=rows,
                          success_rate=float(np.mean([row["success"] for row in rows])))
            LOG.info("DONE phase=all completed=%d total=%d success_rate=%.4f",
                     len(rows), expected_episodes, result["success_rate"])
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        LOG.exception("Official-protocol FastWAM evaluation failed")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
