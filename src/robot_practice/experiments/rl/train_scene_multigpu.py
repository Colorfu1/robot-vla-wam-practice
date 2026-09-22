"""Independent training-scene rollout shards; no DDP and no protocol changes."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def split_scenes(ids, gpus):
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= 100 for i in ids):
        raise ValueError("Expected unique training trajectory ids in [0, 100)")
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU ids must be nonempty and unique")
    return [(gpu, ids[i::len(gpus)]) for i, gpu in enumerate(gpus) if ids[i::len(gpus)]]


def merge_shards(shards, requested):
    episodes, replay, failed, variants = {}, {}, [], set()
    passed = total = 0
    for directory, data in shards:
        if data["status"] not in ("complete", "complete_with_prefix_replay_mismatch"):
            raise ValueError(f"Incomplete or excluded scenes: {directory}")
        if len(data["models"]) != 1:
            raise ValueError("Expected exactly one model per shard")
        name, model = next(iter(data["models"].items()))
        variants.add((name, model["checkpoint"], model["weight_variant"]))
        for tid, row in model["episodes"].items():
            if tid in episodes:
                raise ValueError(f"Duplicate trajectory {tid}")
            row = dict(row, artifact_directory=f"{directory}/{name}/trajectory_{tid}")
            episodes[tid] = row
            if not row["task_success_any_step"]:
                failed.append({"trajectory": int(tid), "artifact_directory": row["artifact_directory"],
                               "prefix_replay_pass": row["prefix_replay_verification"]["passed"],
                               "failure_type": "unreviewed"})
        replay.update(data["replay"])
        passed += data["capture_verification"]["passed"]
        total += data["capture_verification"]["total"]
    if len(variants) != 1 or set(map(int, episodes)) != set(requested):
        raise ValueError("Model identity or exact-once trajectory coverage mismatch")
    name, checkpoint, variant = variants.pop()
    return {
        "status": "complete" if passed == total else "complete_with_prefix_replay_mismatch",
        "capture_verification": {"passed": passed, "total": total,
                                 "direct_snapshot_resume_verified": False},
        "replay": replay, "failure_candidates": sorted(failed, key=lambda x: x["trajectory"]),
        "models": {name: {"checkpoint": checkpoint, "weight_variant": variant,
                          "evaluated": len(episodes), "requested": len(requested),
                          "episodes": episodes,
                          "task_success_rate": sum(r["task_success_any_step"]
                                                   for r in episodes.values()) / len(episodes),
                          "official_chunk_success_rate": sum(r["official_chunk_success"]
                                                              for r in episodes.values()) / len(episodes)}},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--trajectory-ids", default=",".join(map(str, [
        i for i in range(100) if i not in {0, 10, 20, 30, 40, 50, 60, 70}
    ][:24])))
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()
    ids = [int(x) for x in args.trajectory_ids.split(",")]
    assignments = split_scenes(ids, args.gpus.split(","))
    if args.cpu_threads < 1 or not Path(args.checkpoint).is_file():
        p.error("Positive cpu-threads and existing checkpoint required")
    args.output.mkdir(parents=True, exist_ok=False)
    logs = args.output / "task_logs"
    logs.mkdir()
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    manifest = {"status": "running", "trajectory_ids": ids, "max_steps": 600,
                "checkpoint": args.checkpoint, "assignments": [], "shards": []}

    def save():
        (args.output / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")

    processes, handles, finished = [], [], set()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, (gpu, selected) in enumerate(assignments):
            directory = f"shard_{index:02d}"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "LP_NUM_THREADS"):
                env[key] = str(args.cpu_threads)
            command = [sys.executable, "-u", "-m",
                       "robot_practice.experiments.rl.lightwam_train_scene_eval",
                       "--policy", "fastwam", "--capture-scene", "--max-steps", "600",
                       "--checkpoints", args.checkpoint, "--output", str(args.output / directory),
                       "--trajectory-ids", ",".join(map(str, selected))]
            handle = (logs / f"{directory}.log").open("w")
            handles.append(handle)
            child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            processes.append(child)
            manifest["assignments"].append({"shard": directory, "gpu": gpu,
                                             "trajectory_ids": selected, "pid": child.pid,
                                             "command": command})
            LOG.info("START shard=%s gpu=%s trajectories=%s", directory, gpu, selected)
        save()
        while len(finished) < len(processes):
            for index, process in enumerate(processes):
                if index in finished:
                    continue
                code = process.poll()
                directory = manifest["assignments"][index]["shard"]
                path = logs / f"{directory}.log"
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 8192))
                    lines = stream.read().decode(errors="replace").splitlines()
                progress = [line for line in lines if "STATUS " in line or "DONE " in line]
                LOG.info("STATUS shard=%s latest=%s", directory,
                         progress[-1] if progress else "loading / initialization")
                if code is not None:
                    manifest["assignments"][index]["returncode"] = code
                    save()
                    if code != 0:
                        raise RuntimeError(f"{directory} exit={code}; inspect {path}")
                    finished.add(index)
            LOG.info("STATUS phase=shards completed=%d total=%d", len(finished), len(processes))
            if len(finished) < len(processes):
                time.sleep(10)
        shards = [(a["shard"], json.loads((args.output / a["shard"] / "metrics.json").read_text()))
                  for a in manifest["assignments"]]
        manifest.update(merge_shards(shards, ids))
        manifest["shards"] = [name for name, _ in shards]
        save()
        LOG.info("DONE phase=all completed=%d total=%d status=%s", len(ids), len(ids),
                 manifest["status"])
    except BaseException:
        manifest["status"] = "failed_or_interrupted"
        save()
        raise
    finally:
        # Only terminate process groups created by this launcher, including simulator children.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
