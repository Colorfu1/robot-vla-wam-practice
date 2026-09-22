"""Isolated fresh-process EGL export trials plus matched OSMesa reference."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi physical index")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--skip-renderer-query", action="store_true",
                        help="Skip glGetString in both backends; actual renderer remains unverified")
    args = parser.parse_args()
    if args.rounds < 1 or args.trajectories < 1:
        parser.error("rounds and trajectories must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not Path(env["__EGL_VENDOR_LIBRARY_FILENAMES"]).is_file():
        raise FileNotFoundError(env["__EGL_VENDOR_LIBRARY_FILENAMES"])
    probe = Path(__file__).with_name("probe_egl_gpu_mapping.py")
    mapping = subprocess.run([sys.executable, str(probe)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    (args.output / "mapping.log").write_text(mapping.stdout)
    mapping.check_returncode()
    matches = [line.split() for line in mapping.stdout.splitlines()
               if len(line.split()) == 5 and line.split()[0].isdigit()
               and line.split()[2] == str(args.gpu)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected unique physical GPU mapping, got {matches}; see mapping.log")
    egl_id = matches[0][0]
    summary = {"status": "running", "device_mapping": matches[0], "trials": [],
               "skip_renderer_query": args.skip_renderer_query}

    def save():
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))

    save()
    trials = [("osmesa", "osmesa_reference")]
    trials += [("egl", f"egl_{i:02d}") for i in range(1, args.rounds + 1)]
    for index, (backend, name) in enumerate(trials, 1):
        child_env = dict(env, MUJOCO_EGL_DEVICE_ID=egl_id, PYTHONFAULTHANDLER="1")
        target = args.output / name
        command = [sys.executable, "-u", "-m", "robot_practice.experiments.rl.square_export_dual",
                   "--output", str(target), "--render-backend", backend,
                   "--max-trajectories", str(args.trajectories)]
        if args.skip_renderer_query:
            command.append("--skip-renderer-query")
        print(f"START trial={name} gpu={args.gpu} egl={egl_id} log={name}.log", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    lines = (target / "run.log").read_text().splitlines() if (
                        target / "run.log").is_file() else []
                    latest = next((line for line in reversed(lines) if "STATUS" in line), "setup")
                    print(f"STATUS trial={name} completed={index - 1} total={len(trials)} "
                          f"latest={latest}", flush=True)
        record = {"name": name, "returncode": process.returncode, "command": command}
        manifest = target / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            record.update(manifest_status=data["status"], graphics=data.get("graphics"),
                          timing=data.get("timing"))
        if process.returncode == 0 and record.get("manifest_status") == "complete":
            reference = args.output / "osmesa_reference"
            if backend == "egl" and summary["trials"][0].get("returncode") == 0:
                for key in ("states", "actions", "traj_lengths"):
                    if not np.array_equal(np.load(reference / f"{key}.npy"),
                                          np.load(target / f"{key}.npy")):
                        raise ValueError(f"Unmatched {key}")
                a = np.load(reference / "images.npy", mmap_mode="r")
                b = np.load(target / "images.npy", mmap_mode="r")
                if a.shape != b.shape:
                    raise ValueError("Unmatched image shapes")
                per_view = [[], []]
                for first, second in zip(a, b):
                    diff = np.abs(first.astype(np.float32) - second.astype(np.float32))
                    for view in range(2):
                        per_view[view].append(float(diff[:, :, view * 224:(view + 1) * 224].mean()))
                record["image_mae_0_255"] = {
                    key: {"mean": float(np.mean(values)), "max_frame": max(values)}
                    for key, values in zip(("agentview", "wrist"), per_view)
                }
        summary["trials"].append(record)
        save()
        print(f"STATUS completed={index} total={len(trials)} exit_code={process.returncode}",
              flush=True)
    summary["status"] = "complete" if all(
        r["returncode"] == 0 and r.get("manifest_status") == "complete"
        for r in summary["trials"]
    ) else "failed_trials"
    summary["limitation"] = "Static-state export only; not a rollout stability guarantee."
    save()
    print(f"DONE status={summary['status']} metrics={args.output / 'metrics.json'}", flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
