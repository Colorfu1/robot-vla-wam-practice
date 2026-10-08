"""Run FastWAM on the official DPPO Square 20-env/400-call evaluation protocol."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from robot_practice.experiments.rl.dppo_square_official import REVISION
from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
from robot_practice.experiments.rl.fastwam_dual_eval_multigpu import (
    probe_mapping,
    shard_environment,
)
from robot_practice.experiments.rl.fastwam_dual_official_eval import (
    OFFICIAL_ENVS,
    OFFICIAL_POLICY_STEPS,
    OFFICIAL_SEED,
    POLICY_STEPS_PER_EPISODE,
)
from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json

LOG = logging.getLogger(__name__)


def official_assignments(gpus, env_count=OFFICIAL_ENVS):
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError("GPU ids must be distinct nonnegative physical indices")
    if not 1 <= env_count <= OFFICIAL_ENVS:
        raise ValueError("Official environment count must be between 1 and 20")
    base, extra = divmod(env_count, len(gpus))
    start, plan = 0, []
    for offset, gpu in enumerate(gpus):
        count = base + (offset < extra)
        if count:
            plan.append((gpu, start, count))
            start += count
    assert start == env_count
    return plan


def shard_command(args, output, gpu, start, count, egl_id, policy_steps):
    return [
        sys.executable, "-u", "-m", "robot_practice.experiments.rl.fastwam_dual_official_eval",
        "--checkpoint", args.checkpoint, "--output", str(output / f"gpu_{gpu}"),
        "--env-start", str(start), "--env-count", str(count),
        "--egl-device-id", str(egl_id), "--seed", str(OFFICIAL_SEED),
        "--policy-steps", str(policy_steps), "--dppo-root", args.dppo_root,
        "--hdf5", args.hdf5,
        "--normalization", args.normalization,
    ]


def collect(output, plan, mapping, checkpoint_hash, *, policy_steps, final=False):
    rows = []
    policy_updates_completed = 0
    for gpu, start, count in plan:
        path = output / f"gpu_{gpu}" / "metrics.json"
        if not path.is_file():
            if final:
                raise ValueError(f"Missing shard metrics: {path}")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            if final:
                raise
            continue
        selected = data.get("arguments", {})
        expected_count = count * policy_steps // POLICY_STEPS_PER_EPISODE
        if (data.get("checkpoint_sha256") != checkpoint_hash
                or data.get("dppo_revision") != REVISION
                or data.get("protocol") != PROTOCOL
                or data.get("evaluation_mode") != (
                    "smoke" if policy_steps == POLICY_STEPS_PER_EPISODE else "official_dppo"
                )
                or selected.get("env_start") != start
                or selected.get("env_count") != count
                or selected.get("seed") != OFFICIAL_SEED
                or selected.get("policy_steps") != policy_steps
                or selected.get("egl_device_id") != mapping[gpu]["egl_id"]
                or data.get("requested") != expected_count):
            raise ValueError(f"Official shard protocol/provenance mismatch: {path}")
        completed_steps = data.get("policy_steps_completed", 0)
        if not isinstance(completed_steps, int) or not 0 <= completed_steps <= policy_steps:
            raise ValueError(f"Invalid shard policy progress: {path}")
        policy_updates_completed += completed_steps * count
        batch = data.get("episodes", [])
        if len(batch) != data.get("evaluated") or len(batch) > expected_count:
            raise ValueError(f"Invalid shard episode count: {path}")
        for row in batch:
            env_index = row["env_index"]
            sequence = row["episode_sequence"]
            if (not start <= env_index < start + count
                    or not 0 <= sequence < policy_steps // POLICY_STEPS_PER_EPISODE
                    or row["environment_seed"] != OFFICIAL_SEED + env_index
                    or row["start_policy_step"] != sequence * POLICY_STEPS_PER_EPISODE
                    or row["policy_steps"] != POLICY_STEPS_PER_EPISODE):
                raise ValueError(f"Invalid official episode identity: {path}")
        if final and (data.get("status") != "complete" or len(batch) != expected_count):
            raise ValueError(f"Incomplete official shard: {path}")
        rows.extend(batch)
    identities = [(row["env_index"], row["episode_sequence"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate official episodes across shards")
    expected = {(env_index, sequence)
                for _gpu, start, count in plan for env_index in range(start, start + count)
                for sequence in range(policy_steps // POLICY_STEPS_PER_EPISODE)}
    if final and set(identities) != expected:
        raise ValueError("Missing official episodes")
    rows.sort(key=lambda row: (row["env_index"], row["episode_sequence"]))
    return {"requested": len(expected), "evaluated": len(rows), "per_episode": rows,
            "policy_updates_completed": policy_updates_completed,
            "policy_updates_total": policy_steps * sum(count for _, _, count in plan),
            "success_rate": (sum(row["success"] for row in rows) / len(rows)
                             if rows else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true",
                        help="Three environments, first 100 policy calls; not formal")
    parser.add_argument("--dppo-root", default="third_party/dppo_official")
    parser.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    parser.add_argument("--normalization",
                        default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    env_count = 3 if args.smoke else OFFICIAL_ENVS
    policy_steps = POLICY_STEPS_PER_EPISODE if args.smoke else OFFICIAL_POLICY_STEPS
    try:
        plan = official_assignments(args.gpus.split(","), env_count)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
                        force=True)
    result = {"status": "running", "evaluation_mode": "smoke" if args.smoke else "official_dppo",
              "dppo_revision": REVISION, "protocol": PROTOCOL,
              "arguments": vars(args), "official_environment": {"seed": OFFICIAL_SEED,
              "envs": OFFICIAL_ENVS, "policy_steps": OFFICIAL_POLICY_STEPS,
              "episode_control_steps": 400, "action_steps": 4,
              "auto_reset": True}, "shards": [], "device_mapping": {},
              "requested": env_count * policy_steps // POLICY_STEPS_PER_EPISODE,
              "evaluated": 0, "per_episode": [], "policy_updates_completed": 0,
              "policy_updates_total": env_count * policy_steps}
    write_json(output / "metrics.json", result)
    processes, streams = [], []
    started = time.monotonic()
    try:
        LOG.info("START phase=checkpoint_hash completed=0 total=1")
        checkpoint_hash = file_hash(args.checkpoint)
        result["checkpoint_sha256"] = checkpoint_hash
        LOG.info("DONE phase=checkpoint_hash completed=1 total=1")
        LOG.info("START phase=device_mapping completed=0 total=1")
        mapping = probe_mapping(output)
        result["device_mapping"] = {gpu: mapping[gpu] for gpu, _, _ in plan}
        LOG.info("DONE phase=device_mapping completed=1 total=1")
        for gpu, start, count in plan:
            device = mapping[gpu]
            command = shard_command(args, output, gpu, start, count,
                                    device["egl_id"], policy_steps)
            env = shard_environment(str(device["cuda_ordinal_unmasked"]), args.cpu_threads)
            for name in ("ROBOT_PRACTICE_ROLLOUT_FORENSICS",
                         "ROBOT_PRACTICE_READPIXELS_PROBE",
                         "ROBOT_PRACTICE_ROLLOUT_START_METHOD",
                         "ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP"):
                env.pop(name, None)
            stream = (output / f"gpu_{gpu}.launch.log").open("w")
            streams.append(stream)
            process = subprocess.Popen(command, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((gpu, process))
            result["shards"].append({"gpu": gpu, "env_start": start, "env_count": count,
                                     "cuda_ordinal_unmasked": device["cuda_ordinal_unmasked"],
                                     "egl_device_id": device["egl_id"], "pid": process.pid,
                                     "command": command})
            LOG.info("START phase=shard gpu=%s envs=%s..%s egl=%s pid=%d", gpu,
                     start, start + count - 1, device["egl_id"], process.pid)
        write_json(output / "metrics.json", result)
        while True:
            codes = {gpu: process.poll() for gpu, process in processes}
            result.update(collect(output, plan, mapping, checkpoint_hash,
                                  policy_steps=policy_steps))
            result["exit_codes"] = codes
            write_json(output / "metrics.json", result)
            LOG.info("STATUS phase=eval completed=%d total=%d "
                     "policy_updates=%d/%d shards_finished=%d shards_total=%d",
                     result["evaluated"], result["requested"],
                     result["policy_updates_completed"], result["policy_updates_total"],
                     sum(code is not None for code in codes.values()), len(codes))
            if any(code is not None and code != 0 for code in codes.values()):
                raise RuntimeError(f"Official shard failed codes={codes}; inspect gpu_*.launch.log")
            if all(code is not None for code in codes.values()):
                break
            time.sleep(10)
        result.update(collect(output, plan, mapping, checkpoint_hash,
                              policy_steps=policy_steps, final=True))
        result["status"] = "complete"
        LOG.info("DONE phase=all completed=%d total=%d success_rate=%.4f",
                 result["evaluated"], result["requested"], result["success_rate"])
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        LOG.exception("Multi-GPU official-protocol evaluation failed")
        raise
    finally:
        for _gpu, process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _gpu, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in streams:
            stream.close()
        result["exit_codes"] = {gpu: process.returncode for gpu, process in processes}
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "metrics.json", result)


if __name__ == "__main__":
    main()
