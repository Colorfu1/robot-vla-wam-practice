"""Multi-GPU FastWAM dual-view closed-loop rollout, one policy and simulator per GPU.

Example: python -m robot_practice.experiments.rl.fastwam_dual_eval_multigpu \
  --checkpoint outputs/<train>/checkpoints/<step>.pt --output outputs/<new-eval> \
  --gpus 0,1,2,3,4,5,6,7 --episodes 8 --render-backend egl
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
from robot_practice.experiments.rl.fastwam_dual_eval import validate_rendering_info
from robot_practice.experiments.rl.lightwam_train_scene_eval import (
    NVIDIA_EGL_VENDOR,
    file_hash,
    write_json,
)

LOG = logging.getLogger(__name__)
MAPPING_PROBE = Path("scripts/experiments/rl/probe_egl_gpu_mapping.py")


def interrupt(signum, _frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def assignments(gpus, episodes):
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError("GPU ids must be distinct nvidia-smi nonnegative integer indices")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    return [(gpu, list(range(offset, episodes, len(gpus))))
            for offset, gpu in enumerate(gpus) if offset < episodes]


def parse_mapping(output):
    if not any(line.startswith("DONE enumerated=") and "unresolved=0" in line
               for line in output.splitlines()):
        raise ValueError("EGL enumeration incomplete or has unresolved devices")
    mapping, egl_ids = {}, set()
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 5 or not fields[0].isdecimal():
            continue
        egl, cuda, physical, pci, uuid = fields
        if (not physical.isdecimal() or not cuda.isdecimal() or
                pci == "None" or uuid == "UNRESOLVED" or
                physical in mapping or int(egl) in egl_ids):
            raise ValueError(f"Invalid or ambiguous EGL mapping row: {line}")
        mapping[physical] = {"egl_id": int(egl), "cuda_ordinal_unmasked": int(cuda),
                             "pci_bus": pci, "gpu_uuid": uuid}
        egl_ids.add(int(egl))
    if not mapping:
        raise ValueError("No resolved EGL-to-physical-GPU rows")
    return mapping


def probe_mapping(output):
    if not Path(NVIDIA_EGL_VENDOR).is_file():
        raise FileNotFoundError(NVIDIA_EGL_VENDOR)
    if not MAPPING_PROBE.is_file():
        raise FileNotFoundError(MAPPING_PROBE)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = NVIDIA_EGL_VENDOR
    completed = subprocess.run(
        [sys.executable, "-u", str(MAPPING_PROBE)], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
    )
    (output / "mapping.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"EGL mapping probe code={completed.returncode}; inspect mapping.log")
    return parse_mapping(completed.stdout)


def shard_environment(cuda_ordinal, cpu_threads):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_ordinal)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("MUJOCO_GL", None)
    env.pop("PYOPENGL_PLATFORM", None)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(cpu_threads)
    return env


def build_shard_command(args, output, gpu, indices, egl_id):
    command = [
        sys.executable, "-u", "-m", "robot_practice.experiments.rl.fastwam_dual_eval",
        "--checkpoint", args.checkpoint, "--output", str(output / f"gpu_{gpu}"),
        "--episodes", str(args.episodes), "--episode-indices", ",".join(map(str, indices)),
        "--seed", str(args.seed), "--max-steps", str(args.max_steps),
        "--render-backend", args.render_backend,
        "--hdf5", args.hdf5, "--normalization", args.normalization,
    ]
    if egl_id is not None:
        command += ["--egl-device-id", str(egl_id)]
    return command


def collect(output, plan, mapping, args, checkpoint_hash, final=False):
    rows = []
    for gpu, indices in plan:
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
            continue  # The child can be between truncate and write.
        selected = data.get("arguments", {})
        expected_egl = mapping[gpu]["egl_id"] if args.render_backend == "egl" else None
        if (data.get("checkpoint_sha256") != checkpoint_hash or
                data.get("protocol") != PROTOCOL or
                selected.get("episodes") != args.episodes or
                selected.get("episode_indices") != indices or
                selected.get("seed") != args.seed or
                selected.get("max_steps") != args.max_steps or
                selected.get("render_backend") != args.render_backend or
                selected.get("egl_device_id") != expected_egl):
            raise ValueError(f"Shard protocol/provenance mismatch: {path}")
        batch = data.get("episodes", [])
        if data.get("evaluated") != len(batch) or data.get("requested") != len(indices):
            raise ValueError(f"Shard progress mismatch: {path}")
        observed = [row["episode_index"] for row in batch]
        if (len(set(observed)) != len(observed) or not set(observed) <= set(indices) or
                any(row["seed"] != args.seed + row["episode_index"] for row in batch)):
            raise ValueError(f"Duplicate or unexpected episode indices/seeds: {path}")
        if batch:
            validate_rendering_info(data["rendering"], args.render_backend, expected_egl)
        if final and (data.get("status") != "complete" or set(observed) != set(indices)):
            raise ValueError(f"Incomplete shard: {path}")
        rows.extend(batch)
    episode_indices = [row["episode_index"] for row in rows]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("Duplicate episodes across shards")
    if final and set(episode_indices) != set(range(args.episodes)):
        raise ValueError("Missing evaluation episodes")
    rows.sort(key=lambda row: row["episode_index"])
    n = len(rows)
    return {
        "requested": args.episodes,
        "evaluated": n,
        "success_rate_600": sum(row["task_success_any_step"] for row in rows) / n if n else 0.0,
        "success_rate_400": sum(row["within_400"]["task_success_any_step"] for row in rows)
        / n if n else 0.0,
        "official_chunk_success_rate_600": sum(row["official_chunk_success"] for row in rows)
        / n if n else 0.0,
        "official_chunk_success_rate_400": sum(
            row["within_400"]["official_chunk_success"] for row in rows
        ) / n if n else 0.0,
        "per_episode": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                        help="Distinct physical nvidia-smi indices")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    parser.add_argument("--normalization",
                        default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    args = parser.parse_args()
    try:
        plan = assignments(args.gpus.split(","), args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_steps < 1 or args.max_steps % 4 or args.cpu_threads < 1:
        parser.error("max-steps must be positive/divisible by four; cpu-threads must be positive")
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
                        force=True)
    result = {"status": "running", "arguments": vars(args), "protocol": PROTOCOL,
              "initialization": "seeded random reset; NOT demonstration state",
              "device_mapping": {}, "shards": [], "requested": args.episodes,
              "evaluated": 0, "per_episode": []}
    write_json(output / "metrics.json", result)
    processes, streams = [], []
    started = time.monotonic()
    try:
        LOG.info("START phase=checkpoint_hash completed=0 total=1")
        checkpoint_hash = file_hash(args.checkpoint)
        result["checkpoint_sha256"] = checkpoint_hash
        LOG.info("DONE phase=checkpoint_hash completed=1 total=1")
        LOG.info("START phase=device_mapping completed=0 total=1")
        full_mapping = probe_mapping(output)
        result["device_mapping"] = {gpu: full_mapping[gpu] for gpu, _ in plan}
        LOG.info("DONE phase=device_mapping completed=1 total=1")
        write_json(output / "metrics.json", result)
        for gpu, indices in plan:
            identity = result["device_mapping"][gpu]
            egl_id = identity["egl_id"] if args.render_backend == "egl" else None
            child_env = shard_environment(identity["cuda_ordinal_unmasked"], args.cpu_threads)
            command = build_shard_command(args, output, gpu, indices, egl_id)
            stream = (output / f"gpu_{gpu}.launch.log").open("w")
            streams.append(stream)
            proc = subprocess.Popen(command, env=child_env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((gpu, proc))
            result["shards"].append({"gpu": gpu,
                                     "cuda_ordinal_unmasked": identity["cuda_ordinal_unmasked"],
                                     "egl_device_id": egl_id,
                                     "episode_indices": indices, "pid": proc.pid,
                                     "command": command})
            LOG.info("START phase=shard physical_gpu=%s cuda_ordinal=%s egl=%s pid=%d "
                     "episodes=%s", gpu, identity["cuda_ordinal_unmasked"],
                     egl_id, proc.pid, indices)
        write_json(output / "metrics.json", result)
        while True:
            codes = {gpu: proc.poll() for gpu, proc in processes}
            result.update(collect(output, plan, result["device_mapping"], args, checkpoint_hash))
            result["exit_codes"] = codes
            write_json(output / "metrics.json", result)
            LOG.info("STATUS phase=eval completed=%d total=%d shards_finished=%d shards_total=%d",
                     result["evaluated"], args.episodes,
                     sum(code is not None for code in codes.values()), len(codes))
            if any(code is not None and code != 0 for code in codes.values()):
                raise RuntimeError(f"Shard failed codes={codes}; inspect gpu_*.launch.log")
            if all(code is not None for code in codes.values()):
                break
            time.sleep(10)
        result.update(collect(output, plan, result["device_mapping"], args,
                              checkpoint_hash, final=True))
        result["status"] = "complete"
        LOG.info("DONE phase=all completed=%d total=%d rate400=%.4f rate600=%.4f",
                 result["evaluated"], args.episodes,
                 result["success_rate_400"], result["success_rate_600"])
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        LOG.exception("Multi-GPU dual-view evaluation failed; partial results preserved")
        raise
    finally:
        # Only terminate process groups created by this launcher.
        for _gpu, proc in processes:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _gpu, proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for stream in streams:
            stream.close()
        result["exit_codes"] = {gpu: proc.returncode for gpu, proc in processes}
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "metrics.json", result)


if __name__ == "__main__":
    main()
