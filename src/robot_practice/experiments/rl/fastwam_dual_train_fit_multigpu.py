"""Shard training-subset open/closed-loop diagnostics across physical GPUs."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
from robot_practice.experiments.rl.fastwam_dual_eval_multigpu import (
    assignments,
    interrupt,
    probe_mapping,
    shard_environment,
)
from robot_practice.experiments.rl.fastwam_dual_train_fit import action_errors
from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json

LOG = logging.getLogger(__name__)


def merge(output, plan, checkpoint_hash, max_steps, image_pair_only=False,
          protocol_parity_only=False, autocast_action_only=False, visual_checks_only=False):
    scenes, predictions, targets = {}, [], []
    pairs, live_predictions = [], []
    parity_cases = []
    autocast_cases = []
    visual_cases = {}
    for gpu, ids in plan:
        directory = output / f"gpu_{gpu}"
        data = json.loads((directory / "metrics.json").read_text())
        if (data["status"] not in ("complete", "partial_excluded_scenes") or
                data["trajectory_ids"] != ids or data["checkpoint_sha256"] != checkpoint_hash or
                data["protocol"] != PROTOCOL or data["max_steps"] != max_steps or
                (not image_pair_only and not protocol_parity_only and not autocast_action_only
                 and not visual_checks_only and
                 set(data["scenes"]) != set(map(str, ids))) or
                bool(data.get("image_pair_only", False)) != image_pair_only or
                bool(data.get("protocol_parity_only", False)) != protocol_parity_only or
                bool(data.get("autocast_action_only", False)) != autocast_action_only or
                bool(data.get("visual_checks_only", False)) != visual_checks_only):
            raise ValueError(f"Incomplete or mismatched shard: {directory}")
        if visual_checks_only:
            if (data["status"] != "complete" or
                    set(data.get("visual_cases", {})) != set(map(str, ids)) or
                    visual_cases.keys() & data["visual_cases"].keys()):
                raise ValueError(f"Missing or duplicate visual cases: {directory}")
            visual_cases.update(data["visual_cases"])
            continue
        if autocast_action_only:
            if (data["status"] != "complete" or data.get("inference_steps") != 20 or
                    {r["trajectory"] for r in data["autocast_cases"]} != set(ids)):
                raise ValueError(f"Missing autocast cases: {directory}")
            autocast_cases.extend(data["autocast_cases"])
            with np.load(directory / "autocast_actions.npz", allow_pickle=False) as arrays:
                predictions.append(arrays["normal"])
                live_predictions.append(arrays["autocast"])
                targets.append(arrays["target"])
            continue
        if protocol_parity_only:
            if (data["status"] != "complete" or
                    {r["trajectory"] for r in data["parity_cases"]} != set(ids)):
                raise ValueError(f"Missing parity cases: {directory}")
            parity_cases.extend(data["parity_cases"])
            continue
        if image_pair_only:
            if (data["status"] != "complete" or
                    {r["trajectory"] for r in data["pairs"]} != set(ids)):
                raise ValueError(f"Missing paired trajectories: {directory}")
            pairs.extend(data["pairs"])
            with np.load(directory / "image_pairs.npz", allow_pickle=False) as arrays:
                predictions.append(arrays["saved"])
                live_predictions.append(arrays["live"])
                targets.append(arrays["target"])
            continue
        if scenes.keys() & data["scenes"].keys():
            raise ValueError("Duplicate training trajectories")
        scenes.update(data["scenes"])
        with np.load(directory / "open_loop.npz", allow_pickle=False) as arrays:
            predictions.append(arrays["predicted"])
            targets.append(arrays["target"])
    if visual_checks_only:
        links = [f'<li><a href="gpu_{gpu}/trajectory_{tid}/index.html">trajectory {tid}</a></li>'
                 for gpu, ids in plan for tid in ids]
        (output / "index.html").write_text(
            '<meta charset="utf-8"><h1>Dual-view forecasts and one-shot GT RGB</h1><ul>'
            + ''.join(links) + '</ul>')
        return {"status": "complete", "visual_cases": visual_cases,
                "evaluated": len(visual_cases)}
    if autocast_action_only:
        from robot_practice.experiments.rl.fastwam_autocast_action_check import summary

        return {"status": "complete", "autocast_cases": autocast_cases, "inference_steps": 20,
                "evaluated": sum(len(ids) for _, ids in plan),
                "autocast_summary": summary(np.concatenate(predictions),
                                            np.concatenate(live_predictions),
                                            np.concatenate(targets))}
    if protocol_parity_only:
        return {"status": "complete", "parity_cases": parity_cases,
                "evaluated": sum(len(ids) for _, ids in plan)}
    if image_pair_only:
        from robot_practice.experiments.rl.fastwam_dual_image_pair import summary

        return {"status": "complete", "pairs": pairs,
                "evaluated": sum(len(ids) for _, ids in plan),
                "paired_summary": summary(np.concatenate(predictions),
                                          np.concatenate(live_predictions),
                                          np.concatenate(targets))}
    rows = [row["policy"] for row in scenes.values() if "policy" in row]
    return {
        "status": "complete" if len(rows) == len(scenes) else "partial_excluded_scenes",
        "scenes": scenes, "evaluated": len(rows), "requested": len(scenes),
        "open_loop": action_errors(np.concatenate(predictions), np.concatenate(targets)),
        "task_success_rate": (sum(r["task_success_any_step"] for r in rows) / len(rows)
                              if rows else None),
        "success_rate_400": (sum(r["within_400_steps"]["task_success_any_step"] for r in rows)
                             / len(rows) if rows else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--trajectory-ids", default="0,10,20,30,40,50,60,70")
    parser.add_argument("--windows-per-trajectory", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-pair-only", action="store_true")
    parser.add_argument("--protocol-parity-only", action="store_true")
    parser.add_argument("--autocast-action-only", action="store_true")
    parser.add_argument("--visual-checks-only", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if sum((args.image_pair_only, args.protocol_parity_only, args.autocast_action_only,
            args.visual_checks_only)) > 1:
        parser.error("Choose one diagnostic mode")
    ids = [int(x) for x in args.trajectory_ids.split(",")]
    if (len(set(ids)) != len(ids) or min(ids) < 0 or args.cpu_threads < 1 or
            args.windows_per_trajectory < 1 or args.max_steps < 1 or args.max_steps % 4):
        parser.error("Invalid trajectory IDs, window count, threads or control-step budget")
    plan = [(gpu, [ids[i] for i in indices])
            for gpu, indices in assignments(args.gpus.split(","), len(ids))]
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.output / "run.log")])
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    result = {"status": "running", "training_subset_only": True,
              "image_pair_only": args.image_pair_only,
              "protocol_parity_only": args.protocol_parity_only,
              "autocast_action_only": args.autocast_action_only,
              "visual_checks_only": args.visual_checks_only,
              "render_backend": None if (args.protocol_parity_only or args.autocast_action_only)
              else "egl",
              "protocol": PROTOCOL, "shards": [],
              "requested": len(ids), "evaluated": 0, "max_steps": args.max_steps}
    processes, streams = [], []
    try:
        LOG.info("START phase=checkpoint_hash completed=0 total=1")
        result["checkpoint_sha256"] = file_hash(args.checkpoint)
        LOG.info("DONE phase=checkpoint_hash completed=1 total=1")
        mapping = probe_mapping(args.output)
        result["device_mapping"] = {gpu: mapping[gpu] for gpu, _ in plan}
        for gpu, selected in plan:
            identity = mapping[gpu]
            command = [sys.executable, "-u", "-m",
                       "robot_practice.experiments.rl.fastwam_dual_train_fit",
                       "--checkpoint", args.checkpoint,
                       "--output", str(args.output / f"gpu_{gpu}"),
                       "--trajectory-ids", ",".join(map(str, selected)),
                       "--windows-per-trajectory", str(args.windows_per_trajectory),
                       "--max-steps", str(args.max_steps), "--seed", str(args.seed),
                       "--render-backend", "egl", "--egl-device-id", str(identity["egl_id"])]
            if args.image_pair_only:
                command.append("--image-pair-only")
            if args.protocol_parity_only:
                command.append("--protocol-parity-only")
            if args.autocast_action_only:
                command.append("--autocast-action-only")
            if args.visual_checks_only:
                command.append("--visual-checks-only")
            stream = (args.output / f"gpu_{gpu}.launch.log").open("w")
            streams.append(stream)
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    env=shard_environment(identity["cuda_ordinal_unmasked"],
                                                          args.cpu_threads))
            processes.append((gpu, proc))
            result["shards"].append({"gpu": gpu, "trajectory_ids": selected,
                                     "command": command, "pid": proc.pid})
            LOG.info("START gpu=%s egl=%s trajectories=%s", gpu, identity["egl_id"], selected)
        while True:
            codes = {gpu: proc.poll() for gpu, proc in processes}
            result["exit_codes"] = codes
            write_json(args.output / "metrics.json", result)
            for gpu, _ in processes:
                log = args.output / f"gpu_{gpu}" / "run.log"
                if log.is_file():
                    lines = [s for s in log.read_text().splitlines()
                             if "STATUS" in s or "DONE" in s or "START" in s]
                    if lines:
                        LOG.info("STATUS gpu=%s latest=%s", gpu, lines[-1])
            LOG.info("STATUS phase=shards completed=%d total=%d",
                     sum(code is not None for code in codes.values()), len(codes))
            if any(code is not None and code != 0 for code in codes.values()):
                raise RuntimeError(f"Worker failure: {codes}; inspect gpu_*.launch.log")
            if all(code is not None for code in codes.values()):
                break
            time.sleep(10)
        result.update(merge(args.output, plan, result["checkpoint_sha256"], args.max_steps,
                            args.image_pair_only, args.protocol_parity_only,
                            args.autocast_action_only, args.visual_checks_only))
        LOG.info("DONE phase=all completed=%d total=%d success_rate=%s", result["evaluated"],
                 result["requested"], result.get("task_success_rate", "not_measured"))
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for _, proc in processes:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for stream in streams:
            stream.close()
        result["exit_codes"] = {gpu: proc.returncode for gpu, proc in processes}
        write_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
