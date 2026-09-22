"""Offline rolling future predictions along six recorded FastWAM evaluation episodes."""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json

log = logging.getLogger(__name__)


def select_cases(metrics):
    if metrics.get("status") != "complete" or metrics.get("evaluated") != 80:
        raise ValueError("Requires complete 80-episode evaluation")
    if metrics["arguments"]["policy"] != "fastwam":
        raise ValueError("Expected FastWAM evaluation")
    rows = sorted(metrics["per_episode"], key=lambda row: row["environment_seed"])
    if len(rows) != 80 or len({r["environment_seed"] for r in rows}) != 80:
        raise ValueError("Invalid episode inventory")
    cases = []
    for success in (True, False):
        group = [r for r in rows if r["success_600"] is success]
        if len(group) < 3:
            raise ValueError("Need at least three cases in each class")
        cases.extend(group[:3])
    return cases


def case_name(row):
    return f"{'success' if row['success_600'] else 'failure'}_{row['environment_seed']}"


def locate_batch(root, seed):
    matches = []
    for first in Path(root).glob("gpu_*/batch_*/forensics/chunk_0000_input.npz"):
        with np.load(first, allow_pickle=False) as data:
            if seed in data["environment_seeds"].tolist():
                matches.append(first.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected one recorded batch for seed {seed}, got {matches}")
    directory = matches[0]
    for chunk in range(150):
        if not (directory / f"chunk_{chunk:04d}_input.npz").is_file():
            raise FileNotFoundError(f"Missing seed={seed} chunk={chunk}")
    return directory


def read_observation(directory, chunk, seed):
    with np.load(directory / f"chunk_{chunk:04d}_input.npz", allow_pickle=False) as data:
        ids = data["environment_seeds"].tolist()
        if ids.count(seed) != 1:
            raise ValueError("Missing or duplicated seed")
        index = ids.index(seed)
        obs = {key: data[key][index, -1].copy() for key in ("rgb", "state")}
        action = data["action_raw"][index].copy()
    if obs["rgb"].ndim != 3 or obs["state"].shape != (9,) or action.shape != (4, 7):
        raise ValueError("Unexpected recorded observation/action shape")
    return obs, action


def ffmpeg_binary():
    binary = shutil.which("ffmpeg")
    if binary is None:
        import imageio_ffmpeg

        binary = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([binary, "-hide_banner", "-encoders"],
                            capture_output=True, text=True, check=True)
    if "libx264" not in result.stdout:
        raise RuntimeError("ffmpeg with libx264 is required for browser-compatible MP4")
    return binary


class VideoWriter:
    def __init__(self, binary, path, size, fps):
        self.size = size
        self.log = path.with_suffix(".ffmpeg.log").open("w")
        self.process = subprocess.Popen([
            binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{size[0]}x{size[1]}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

    def append(self, frame):
        if frame.size != self.size:
            raise ValueError("Video frame size mismatch")
        self.process.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())

    def close(self):
        try:
            self.process.stdin.close()
        finally:
            try:
                code = self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise
            finally:
                self.log.close()
        if code:
            raise RuntimeError(f"ffmpeg failed with code {code}; inspect *.ffmpeg.log")


def panel(frames, labels, title):
    canvas = Image.new("RGB", (224 * len(frames), 272), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), title, fill="black")
    for index, (frame, label) in enumerate(zip(frames, labels)):
        draw.text((224 * index + 4, 25), label, fill="black")
        if frame is not None:
            if frame.size != (224, 224):
                raise ValueError("Expected native 224x224 frame")
            canvas.paste(frame, (224 * index, 48))
        else:
            draw.text((224 * index + 4, 140), "NOT RECORDED", fill="black")
    return canvas


def run_case(args, row, checkpoint_hash):
    import torch
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.fastwam_square_diagnostic import image_from_tensor
    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "run.log"),
    ])
    seed = row["environment_seed"]
    result = {"status": "running", "episode": row, "completed": 0, "total": 150,
              "checkpoint_sha256": checkpoint_hash, "chunks": [],
              "protocol": "Recorded-observation rolling forecast, not autoregressive rollout",
              "denoising_steps": 20, "sample_seed": seed,
              "gt_action_conditioning": False, "future_gt_conditioning": False,
              "conditioning_stride_control_steps": 4, "predicted_future_frames_per_chunk": 4,
              "last_endpoint_reference": "not recorded; no final/autoreset image substituted"}
    write_json(output / "metrics.json", result)
    writers = []
    try:
        binary = ffmpeg_binary()
        evidence = locate_batch(args.eval_root, seed)
        result["source_evidence"] = str(evidence)
        model, adapter, result["provenance"] = load_fastwam(
            args.checkpoint, OmegaConf.load(args.config)
        )
        (output / "predictions").mkdir()
        specs = [("actual_5hz.mp4", (224, 272), 5),
                 ("predicted_20hz.mp4", (224, 272), 20),
                 ("comparison_5hz.mp4", (672, 272), 5),
                 ("forecast_strip_5hz.mp4", (1120, 272), 5)]
        for name, size, fps in specs:
            writers.append(VideoWriter(binary, output / name, size, fps))
        started = time.monotonic()
        for chunk in range(150):
            obs, recorded_action = read_observation(evidence, chunk, seed)
            rgb, proprio = adapter._prepare_observation(obs)
            first = image_from_tensor(rgb[0])
            reference = None
            if chunk < 149:
                next_obs, _ = read_observation(evidence, chunk + 1, seed)
                next_rgb, _ = adapter._prepare_observation(next_obs)
                reference = image_from_tensor(next_rgb[0])
            with torch.inference_mode():
                generated = model.infer_joint(
                    prompt=None, input_image=rgb[0].to(model.device),
                    proprio=proprio[0].to(model.device),
                    context=adapter.context.to(model.device),
                    context_mask=adapter.context_mask.to(model.device),
                    num_video_frames=5, action_horizon=4, action=None,
                    num_inference_steps=20, seed=seed,
                    test_action_with_infer_action=False,
                )
            frames = generated["video"]
            if len(frames) != 5:
                raise ValueError("Expected five video frames")
            action = generated["action"].detach().float().cpu().numpy()
            pixels = np.stack([np.asarray(f.convert("RGB")) for f in frames])
            if pixels.shape != (5, 224, 224, 3) or not np.isfinite(action).all():
                raise ValueError("Invalid video/action output")
            np.savez_compressed(output / "predictions" / f"chunk_{chunk:04d}.npz",
                                rgb=pixels, action_raw=action, recorded_action_raw=recorded_action,
                                chunk=chunk, environment_seed=seed)
            title = f"seed={seed} {'SUCCESS' if row['success_600'] else 'FAILURE'} chunk={chunk:03d}"
            writers[0].append(panel([first], [f"Actual step {chunk * 4}"], title))
            for t in range(1, 5):  # Exclude known frame zero, preserve 600 predicted frames.
                writers[1].append(panel([frames[t]], [f"Pred step {chunk * 4+t}; cond {chunk*4}"],
                                        f"seed={seed} rolling forecast"))
            comparison = panel([first, frames[4], reference],
                               [f"Actual step {chunk*4}", f"Pred step {chunk*4+4}",
                                f"Actual step {chunk*4+4}"], title)
            writers[2].append(comparison)
            writers[3].append(panel([first, *frames[1:]],
                                   ["Actual condition", "Pred +1", "Pred +2", "Pred +3", "Pred +4"],
                                   title))
            if chunk % 10 == 0 or chunk == 149:
                comparison.save(output / f"comparison_chunk_{chunk:04d}.png")
            record = {"chunk": chunk,
                      "joint_vs_recorded_action_max_abs_error": float(np.abs(action-recorded_action).max()),
                      "prediction_endpoint_mae": None, "copy_first_endpoint_mae": None}
            if reference is not None:
                truth = np.asarray(reference, dtype=np.float32)
                record["prediction_endpoint_mae"] = float(np.abs(pixels[4].astype(float)-truth).mean())
                record["copy_first_endpoint_mae"] = float(np.abs(np.asarray(first, dtype=float)-truth).mean())
            result["chunks"].append(record)
            result["completed"] = chunk + 1
            write_json(output / "metrics.json", result)
            if chunk % 5 == 0 or chunk == 149:
                elapsed = time.monotonic() - started
                log.info("STATUS phase=video seed=%d completed=%d total=150 eta_seconds=%.1f",
                         seed, chunk+1, elapsed/(chunk+1)*(149-chunk))
        for writer in writers:
            writer.close()
        writers.clear()
        scored = [r for r in result["chunks"] if r["prediction_endpoint_mae"] is not None]
        result["endpoint_summary"] = {
            "scored_chunks": len(scored),
            "prediction_mae": float(np.mean([r["prediction_endpoint_mae"] for r in scored])),
            "copy_first_mae": float(np.mean([r["copy_first_endpoint_mae"] for r in scored])),
            "joint_vs_recorded_action_max_abs_error": max(
                r["joint_vs_recorded_action_max_abs_error"] for r in result["chunks"]
            ),
        }
        result["status"] = "complete"
        log.info("DONE phase=video seed=%d completed=150 total=150", seed)
    except BaseException:
        result["status"] = "failed"
        log.exception("Video generation failed; partial outputs retained")
        raise
    finally:
        for writer in writers:
            if writer.process.poll() is None:
                try:
                    writer.close()
                except Exception:
                    log.exception("Could not finalize partial video")
        write_json(output / "metrics.json", result)


def interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def write_index(output, cases):
    sections = []
    for row in cases:
        name = case_name(row)
        sections.append(
            f'<h2>{name}</h2><p>First successful chunk ends at control step: '
            f'{row.get("first_success_chunk_end_control_step")}</p>'
            f'<video controls preload="metadata" width="672" '
            f'src="{name}/comparison_5hz.mp4"></video><p>'
            f'<a href="{name}/predicted_20hz.mp4">Predicted future (rolling)</a> | '
            f'<a href="{name}/actual_5hz.mp4">Recorded observations</a> | '
            f'<a href="{name}/forecast_strip_5hz.mp4">All prediction horizons</a> | '
            f'<a href="{name}/metrics.json">Metrics</a></p>'
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>FastWAM rolling forecasts</title>'
        '<h1>3 successful / 3 failed recorded episodes</h1>'
        '<p>Offline predictions: each chunk reconditions on the recorded observation. '
        'Not an autonomous 600-step video rollout. Middle frames have no recorded GT. '
        'The final endpoint is not recorded.</p>' + "\n".join(sections)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--worker-seed", type=int)
    p.add_argument("--config", default="configs/experiments/rl/lightwam_base_robomimic_square.yaml")
    args = p.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    root = Path(args.eval_root)
    metrics = json.loads((root / "metrics.json").read_text())
    cases = select_cases(metrics)
    checkpoint_hash = file_hash(args.checkpoint)
    if checkpoint_hash != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from recorded evaluation")
    if file_hash(args.config) != metrics["config_sha256"]:
        raise ValueError("Configuration differs from recorded evaluation")
    if args.worker_seed is not None:
        row = next(r for r in cases if r["environment_seed"] == args.worker_seed)
        run_case(args, row, checkpoint_hash)
        return
    gpus = args.gpus.split(",")
    if len(gpus) != 6 or len(set(gpus)) != 6 or any(not g.isdecimal() for g in gpus):
        p.error("Provide six distinct GPU ids: one per case")
    ffmpeg_binary()  # Check encoder before starting six large models.
    for row in cases:
        locate_batch(root, row["environment_seed"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "run.log"),
    ])
    result = {"status": "running", "arguments": vars(args), "cases": cases,
              "selection": "first three successful and first three failed seeds, ascending",
              "checkpoint_sha256": checkpoint_hash, "source_metrics_sha256": file_hash(root/"metrics.json"),
              "completed": 0, "total": 900}
    write_json(output / "metrics.json", result)
    processes, streams = [], []
    try:
        for gpu, row in zip(gpus, cases):
            name = case_name(row)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2")
            stream = (output / f"{name}.launch.log").open("w")
            streams.append(stream)
            command = [sys.executable, "-u", "-m", __spec__.name,
                       f"--checkpoint={args.checkpoint}", f"--eval-root={args.eval_root}",
                       f"--config={args.config}", f"--output={output/name}",
                       f"--worker-seed={row['environment_seed']}"]
            processes.append(subprocess.Popen(command, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True))
            log.info("START case=%s gpu=%s", name, gpu)
        while True:
            snapshots = {}
            for row in cases:
                path = output / case_name(row) / "metrics.json"
                if path.exists():
                    try:
                        snapshot = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        continue
                    snapshots[case_name(row)] = {k: snapshot[k] for k in ("status", "completed", "total")}
            codes = [proc.poll() for proc in processes]
            result["progress"] = snapshots
            result["completed"] = sum(d["completed"] for d in snapshots.values())
            write_json(output / "metrics.json", result)
            log.info("STATUS phase=video completed=%d total=900 cases_finished=%d total_cases=6",
                     result["completed"], sum(code == 0 for code in codes))
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Worker failed: {codes}; inspect case launch logs")
            if all(code == 0 for code in codes):
                if len(snapshots) != 6 or any(d["status"] != "complete" or d["completed"] != 150
                                              for d in snapshots.values()):
                    raise RuntimeError("Incomplete case results")
                break
            time.sleep(10)
        write_index(output, cases)
        result["status"] = "complete"
        log.info("DONE phase=all completed=900 total=900")
    except BaseException:
        result["status"] = "failed"
        log.exception("Stopped; partial outputs retained")
        raise
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in processes:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for stream in streams:
            stream.close()
        write_json(output / "metrics.json", result)


if __name__ == "__main__":
    main()
