"""Fresh multi-GPU evaluation, preserving global batches and environment seeds."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json
from robot_practice.experiments.rl.square_paired_eval import summarize_batches

logger = logging.getLogger(__name__)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def assignments(gpus, episodes, batch_size):
    if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdecimal() for g in gpus):
        raise ValueError("GPU ids must be distinct nonnegative integers")
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch-size must be positive")
    starts = list(range(0, episodes, batch_size))
    return [(gpu, starts[i :: len(gpus)]) for i, gpu in enumerate(gpus) if i < len(starts)]


def collect(output, plan, args, checkpoint_hash, config_hash, final=False):
    rows = []
    for gpu, starts in plan:
        path = output / f"gpu_{gpu}" / "metrics.json"
        if not path.exists():
            if final:
                raise ValueError(f"Missing shard result: {path}")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            if final:
                raise
            continue  # Child may be between truncate and write; next poll retries.
        saved = data["arguments"]
        if (
            data.get("checkpoint_sha256") != checkpoint_hash
            or data.get("config_sha256") != config_hash
            or data.get("weight_variant") != "model"
            or saved["policy"] != args.policy
            or saved["seed"] != args.seed
            or saved["episodes"] != args.episodes
            or saved["batch_size"] != args.batch_size
            or data.get("batch_starts") != starts
        ):
            raise ValueError(f"Shard protocol/provenance mismatch: {path}")
        expected = {
            args.seed + i for s in starts for i in range(s, min(s + args.batch_size, args.episodes))
        }
        batch = data.get("per_episode", [])
        observed = [r["environment_seed"] for r in batch]
        if len(set(observed)) != len(observed) or not set(observed) <= expected:
            raise ValueError(f"Duplicate or unexpected seeds: {path}")
        if final and (data.get("status") != "complete" or set(observed) != expected):
            raise ValueError(f"Incomplete shard: {path}")
        rows.extend(batch)
    seeds = [r["environment_seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds across shards")
    if final and set(seeds) != set(range(args.seed, args.seed + args.episodes)):
        raise ValueError("Missing evaluation episodes")
    return summarize_batches(sorted(rows, key=lambda r: r["environment_seed"]), args.episodes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=("lightwam", "dp", "fastwam"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpus", required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--config", default="configs/experiments/rl/lightwam_base_robomimic_square.yaml")
    p.add_argument("--dp-vendor", default="third_party/dppo_official")
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    plan = assignments(args.gpus.split(","), args.episodes, args.batch_size)
    if args.cpu_threads < 1:
        p.error("cpu-threads must be positive")
    checkpoint_hash, config_hash = file_hash(args.checkpoint), file_hash(args.config)
    if (
        args.policy == "dp"
        and not (Path(args.dp_vendor) / "model/diffusion/diffusion_eval.py").is_file()
    ):
        raise FileNotFoundError("Missing DPPO vendor source")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
        force=True,
    )
    result = {
        "status": "running",
        "arguments": vars(args),
        "weight_variant": "model",
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "protocol": "random reset, 600 control steps, official chunk reward criterion",
        "shards": [{"gpu": g, "batch_starts": s} for g, s in plan],
        **summarize_batches([], args.episodes),
    }
    write_json(output / "metrics.json", result)
    processes, streams = [], []
    try:
        for gpu, starts in plan:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            for key in (
                "LP_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ):
                env[key] = str(args.cpu_threads)
            command = [
                sys.executable,
                "-u",
                "-m",
                "robot_practice.experiments.rl.square_paired_eval",
                f"--policy={args.policy}",
                f"--checkpoint={args.checkpoint}",
                f"--output={output / f'gpu_{gpu}'}",
                f"--episodes={args.episodes}",
                f"--batch-size={args.batch_size}",
                f"--seed={args.seed}",
                f"--config={args.config}",
                f"--dp-vendor={args.dp_vendor}",
                "--batch-starts=" + ",".join(map(str, starts)),
            ]
            stream = (output / f"gpu_{gpu}.launch.log").open("w")
            streams.append(stream)
            proc = subprocess.Popen(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(proc)
            logger.info("START gpu=%s pid=%d batch_starts=%s", gpu, proc.pid, starts)
        while True:
            codes = [proc.poll() for proc in processes]
            result.update(collect(output, plan, args, checkpoint_hash, config_hash))
            write_json(output / "metrics.json", result)
            logger.info(
                "STATUS phase=eval completed=%d total=%d shards_finished=%d shards_total=%d",
                result["evaluated"],
                args.episodes,
                sum(code is not None for code in codes),
                len(codes),
            )
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Shard failed; return codes={codes}; inspect gpu_*.launch.log")
            if all(code is not None for code in codes):
                break
            time.sleep(10)
        result.update(collect(output, plan, args, checkpoint_hash, config_hash, final=True))
        result["status"] = "complete"
        logger.info(
            "DONE phase=all completed=%d total=%d rate400=%.4f rate600=%.4f",
            result["evaluated"],
            args.episodes,
            result["success_rate_400"],
            result["success_rate_600"],
        )
    except BaseException:
        result["status"] = "failed"
        logger.exception("Multi-GPU evaluation failed; preserving partial results")
        raise
    finally:
        # Only stop process groups created by this invocation, never existing jobs.
        for proc in processes:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for stream in streams:
            stream.close()
        write_json(output / "metrics.json", result)


if __name__ == "__main__":
    main()
