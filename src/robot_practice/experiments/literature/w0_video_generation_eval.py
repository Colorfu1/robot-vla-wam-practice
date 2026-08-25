"""Reproducible W0 video-generation preservation evaluation.

The evaluator has two deliberately separate tracks:

* ``VBench-I2V-mini-21`` uses the public VBench-I2V image/prompt metadata and
  compares a small, deterministic subset of the official dimensions.
* ``W0-Video-40`` uses one deterministic clip per LIBERO task from the local
  training demonstrations and reports future-frame reconstruction metrics.

The FastWAM vendor checkout remains the source of model implementation.  This
module owns manifests, sampling, provenance, metrics, progress logging, and the
small amount of orchestration needed to call the pinned evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_NAME = "robot_practice.experiments.literature.w0_video_generation_eval"
DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/literature/w0_video_generation_eval.json"
DEFAULT_W0_CONFIG = REPO_ROOT / "configs/experiments/literature/w0_fastwam_reproduction.json"

VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
WAN22_COMMIT = "81e67351b81197330c9533e7f345233e7702b44a"
W0_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
VBENCH_DIMENSIONS = (
    "i2v_subject",
    "subject_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "i2v_background",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "camera_motion",
)
VBENCH_SUBJECT_TYPES = (
    "animal",
    "food",
    "multiple-human",
    "other",
    "plant",
    "single-human",
    "transportation",
)
VBENCH_BACKGROUND_QUOTA = {
    "abstract": 1,
    "architecture": 2,
    "indoor": 2,
    "scenery": 2,
}
VBENCH_CAMERA_INSTRUCTIONS = (
    "camera pans left",
    "camera pans right",
    "camera tilts up",
    "camera tilts down",
    "camera zooms in",
    "camera zooms out",
    "camera static",
)
W0_DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _portable(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _portable(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _rooted(path: str | Path | None, *, default: Path | None = None) -> Path:
    if path is None:
        if default is None:
            raise ValueError("path is required")
        return default
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_info(path: str | Path, label: str = "file") -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"missing {label}: {_portable(target)}")
    return {"path": _portable(target), "bytes": target.stat().st_size, "sha256": _sha256(target)}


def _git_command(path: str | Path, *arguments: str) -> list[str]:
    """Build a Git command that also works for a root-owned bind mount in Docker."""

    target = Path(path).resolve()
    return ["git", "-c", f"safe.directory={target}", "-C", str(target), *arguments]


def _git_head(path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            _git_command(path, "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_status(path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            _git_command(path, "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _require_pinned_checkout(path: str | Path, *, expected: str, label: str) -> str:
    target = Path(path)
    head = _git_head(target)
    if head != expected:
        raise RuntimeError(
            f"{label} must be checked out at {expected}, got {head!r}: {_portable(target)}"
        )
    status = _git_status(target)
    if status:
        raise RuntimeError(f"{label} checkout has local changes: {_portable(target)}")
    return head


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class ProgressLog:
    """Write structured progress to both stdout and a persistent log."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "run.log"

    def write(self, message: str) -> None:
        line = str(message)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def status(self, *, stage: str, completed: int, total: int, detail: str = "") -> None:
        suffix = f" {detail}" if detail else ""
        self.write(f"STATUS stage={stage} completed={completed} total={total}{suffix}")

    def done(self, *, stage: str, total: int) -> None:
        self.write(f"DONE stage={stage} completed={total} total={total}")


def _safe_filename(value: str, *, max_length: int = 180) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _.,-]+", "_", value).strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return (cleaned or "sample")[:max_length]


def _center_crop_resize(array: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized = image.resize(
        (max(width, round(src_w * scale)), max(height, round(src_h * scale))), Image.BILINEAR
    )
    left = max((resized.width - width) // 2, 0)
    top = max((resized.height - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def _read_video_frames_ffmpeg(path: Path, requested: Sequence[int] | None) -> list[np.ndarray]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dimensions = probe.stdout.strip().split("x")
    if len(dimensions) != 2:
        raise RuntimeError(f"ffprobe did not return video dimensions for {_portable(path)}")
    width, height = (int(value) for value in dimensions)
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    frame_size = width * height * 3
    if len(decoded.stdout) % frame_size != 0:
        raise RuntimeError(f"ffmpeg returned a partial frame for {_portable(path)}")
    video = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    if requested is None:
        return [np.asarray(frame) for frame in video]
    if min(requested, default=0) < 0 or max(requested, default=0) >= len(video):
        raise IndexError(f"requested frame outside video: {_portable(path)} indices={requested}")
    return [np.asarray(video[index]) for index in requested]


def _read_video_frames(path: str | Path, indices: Sequence[int] | None = None) -> list[np.ndarray]:
    """Decode RGB frames, preferring quiet system ffmpeg for AV1-containing datasets."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"missing video: {_portable(target)}")
    requested = None if indices is None else [int(index) for index in indices]
    wanted = None if requested is None else set(requested)

    try:
        return _read_video_frames_ffmpeg(target, requested)
    except FileNotFoundError:
        # Minimal environments may not have ffmpeg/ffprobe.  Keep the OpenCV
        # path as a fallback for ordinary H.264 files in that case.
        pass
    except subprocess.CalledProcessError as exc:
        raw_error = exc.stderr or ""
        error = (
            raw_error.decode("utf-8", errors="replace")
            if isinstance(raw_error, bytes)
            else str(raw_error)
        )[-500:]
        raise RuntimeError(f"ffmpeg could not decode {_portable(target)}: {error}") from exc

    frames: dict[int, np.ndarray] = {}
    try:
        import cv2

        capture = cv2.VideoCapture(str(target))
        if capture.isOpened():
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if wanted is None or frame_index in wanted:
                    frames[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_index += 1
                if wanted is not None and len(frames) == len(wanted):
                    break
            capture.release()
        else:
            capture.release()
    except ImportError:
        pass

    if requested is None and frames:
        return [frames[index] for index in sorted(frames)]
    if requested is not None and len(frames) == len(wanted):
        return [frames[index] for index in requested]

    try:
        import imageio.v3 as imageio

        video = np.asarray(imageio.imread(target))
    except Exception as exc:  # pragma: no cover - depends on local codec stack
        raise RuntimeError(f"could not decode video {_portable(target)}") from exc
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"expected RGB video [T,H,W,3], got {video.shape} for {_portable(target)}")
    if requested is None:
        return [np.asarray(frame, dtype=np.uint8) for frame in video]
    if min(requested, default=0) < 0 or max(requested, default=0) >= len(video):
        raise IndexError(f"requested frame outside video: {_portable(target)} indices={requested}")
    return [np.asarray(video[index], dtype=np.uint8) for index in requested]


def _write_video(path: str | Path, frames: Sequence[np.ndarray], *, fps: int = 16) -> None:
    if not frames:
        raise ValueError("cannot write an empty video")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    height, width = normalized[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in normalized):
        raise ValueError("all video frames must have identical dimensions")
    try:
        import cv2

        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if writer.isOpened():
            for frame in normalized:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            if target.is_file() and target.stat().st_size > 0:
                return
        writer.release()
    except ImportError:
        pass
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(target), normalized, fps=fps, codec="libx264")
    except Exception as exc:  # pragma: no cover - depends on local codec stack
        raise RuntimeError(f"could not encode video {_portable(target)}") from exc


def _write_image(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(target)


def _load_config(path: str | Path | None) -> dict[str, Any]:
    target = _rooted(path, default=DEFAULT_CONFIG)
    if not target.is_file():
        raise FileNotFoundError(f"missing video-generation config: {_portable(target)}")
    payload = _read_json(target)
    if not isinstance(payload, dict):
        raise TypeError("video-generation config must be a JSON object")
    return payload


def _load_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise TypeError(f"invalid sample manifest: {_portable(path)}")
    if not payload["samples"]:
        raise ValueError(f"sample manifest is empty: {_portable(path)}")
    return payload


def _visible_cuda_tokens() -> list[str]:
    """Return the physical CUDA selectors corresponding to visible logical GPUs."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is required by the model env
        raise RuntimeError("multi-GPU W0 evaluation requires PyTorch") from exc
    if not torch.cuda.is_available():
        return []
    device_count = int(torch.cuda.device_count())
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        return [str(index) for index in range(device_count)]
    tokens = [token.strip() for token in configured.split(",") if token.strip()]
    if not tokens or tokens == ["-1"]:
        return []
    return tokens[:device_count]


def _require_multi_gpu_devices(num_gpus: int) -> list[str]:
    if num_gpus <= 0:
        raise ValueError("--num-gpus must be a positive integer")
    devices = _visible_cuda_tokens()
    if len(devices) < num_gpus:
        raise RuntimeError(
            f"--num-gpus={num_gpus} requested, but only {len(devices)} CUDA device(s) are visible"
        )
    return devices[:num_gpus]


def _make_manifest_shards(
    *,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    requested_gpus: int,
    work_root: Path,
) -> list[tuple[Path, list[dict[str, Any]]]]:
    samples = [dict(sample) for sample in manifest_payload["samples"]]
    worker_count = min(requested_gpus, len(samples))
    parent_info = {"path": _portable(manifest_path), "sha256": _sha256(manifest_path)}
    shards: list[tuple[Path, list[dict[str, Any]]]] = []
    for worker_index in range(worker_count):
        subset = samples[worker_index::worker_count]
        shard_dir = work_root / f"shard-{worker_index:02d}"
        shard_manifest = shard_dir / "manifest.json"
        shard_payload = dict(manifest_payload)
        shard_payload["samples"] = subset
        shard_payload["multi_gpu_shard"] = {
            "index": worker_index,
            "count": worker_count,
            "requested_gpus": requested_gpus,
            "strategy": "round_robin_manifest_order",
            "parent_manifest": parent_info,
        }
        _write_json(shard_manifest, shard_payload)
        shards.append((shard_dir, subset))
    _write_json(
        work_root / "shards.json",
        {
            "schema_version": 1,
            "requested_gpus": requested_gpus,
            "worker_count": worker_count,
            "parent_manifest": parent_info,
            "shards": [
                {
                    "index": index,
                    "manifest": _portable(shard_dir / "manifest.json"),
                    "sample_count": len(subset),
                    "sample_ids": [str(sample["sample_id"]) for sample in subset],
                }
                for index, (shard_dir, subset) in enumerate(shards)
            ],
        },
    )
    return shards


def _option(name: str, value: Any) -> str:
    return f"--{name}={value}"


def _relay_worker_output(
    *,
    label: str,
    stream: Any,
    log_path: Path,
) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        for line in stream:
            log.write(line)
            log.flush()
            print(f"[{label}] {line.rstrip()}", flush=True)


def _run_multi_gpu_workers(
    *,
    stage: str,
    worker_commands: Sequence[Sequence[str]],
    worker_dirs: Sequence[Path],
    device_tokens: Sequence[str],
    progress_root: Path,
) -> None:
    if not worker_commands:
        raise ValueError("multi-GPU evaluation requires at least one worker")
    if not (
        len(worker_commands) == len(worker_dirs) == len(device_tokens)
    ):
        raise ValueError("worker command, directory, and device counts must match")

    progress = ProgressLog(progress_root)
    worker_count = len(worker_commands)
    progress.status(stage=stage, completed=0, total=worker_count, detail="workers started")
    processes: list[subprocess.Popen[str]] = []
    relay_threads: list[threading.Thread] = []
    try:
        for index, (command, worker_dir, device_token) in enumerate(
            zip(worker_commands, worker_dirs, device_tokens)
        ):
            worker_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device_token
            process = subprocess.Popen(
                list(command),
                cwd=str(REPO_ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append(process)
            assert process.stdout is not None
            relay = threading.Thread(
                target=_relay_worker_output,
                kwargs={
                    "label": f"worker={index} device={device_token}",
                    "stream": process.stdout,
                    "log_path": worker_dir / "worker.stdout.log",
                },
                daemon=True,
            )
            relay.start()
            relay_threads.append(relay)
            progress.status(
                stage=stage,
                completed=index + 1,
                total=worker_count,
                detail=f"worker={index} device={device_token} launched",
            )
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise

    completed: set[int] = set()
    failure: tuple[int, int] | None = None
    while len(completed) < worker_count:
        for index, process in enumerate(processes):
            if index in completed:
                continue
            return_code = process.poll()
            if return_code is None:
                continue
            completed.add(index)
            if return_code != 0:
                failure = (index, return_code)
                break
            progress.status(
                stage=stage,
                completed=len(completed),
                total=worker_count,
                detail=f"worker={index} completed",
            )
        if failure is not None:
            break
        if len(completed) < worker_count:
            time.sleep(0.2)

    if failure is not None:
        failed_index, return_code = failure
        for index, process in enumerate(processes):
            if index != failed_index and process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for relay in relay_threads:
            relay.join()
        raise RuntimeError(
            f"{stage} worker {failed_index} failed with exit code {return_code}; "
            f"see the worker.stdout.log under {progress_root}"
        )

    for relay in relay_threads:
        relay.join()
    progress.done(stage=stage, total=worker_count)


def _protocol_from_file(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid protocol manifest: {_portable(path)}")
    required = {"kind", "width", "height", "frames", "fps", "steps", "action_conditioned"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"protocol is missing keys: {missing}")
    if payload["action_conditioned"] is not False:
        raise ValueError("this evaluator only supports action_conditioned=false")
    return payload


def _import_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object in {_portable(path)}")
                rows.append(value)
    return rows


def _video_path_from_info(
    dataset: Path, info: Mapping[str, Any], key: str, episode_index: int
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = str(info["video_path"])
    relative = template.format(
        episode_chunk=episode_index // chunks_size,
        video_key=key,
        episode_index=episode_index,
    )
    path = dataset / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset video: {_portable(path)}")
    return path


def _load_w0_task_rows(
    dataset: Path,
) -> tuple[dict[int, str], dict[int, dict[str, Any]], dict[str, Any]]:
    info_path = dataset / "meta/info.json"
    task_path = dataset / "meta/tasks.jsonl"
    episode_path = dataset / "meta/episodes.jsonl"
    for path in (info_path, task_path, episode_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing W0 metadata: {_portable(path)}")
    info = _read_json(info_path)
    task_rows = _import_jsonl(task_path)
    episode_rows = _import_jsonl(episode_path)
    task_languages = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    language_to_task = {language: task for task, language in task_languages.items()}
    episodes: dict[int, dict[str, Any]] = {}
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        task_names = row.get("tasks", [])
        if isinstance(task_names, str):
            task_names = [task_names]
        matches = [language_to_task[name] for name in task_names if name in language_to_task]
        if len(matches) != 1:
            raise ValueError(
                f"episode {episode_index} does not map to exactly one task in {_portable(task_path)}"
            )
        episodes[episode_index] = {
            "episode_index": episode_index,
            "task_id": matches[0],
            "length": int(row["length"]),
        }
    return (
        task_languages,
        episodes,
        {
            "info": _file_info(info_path, "dataset info"),
            "tasks": _file_info(task_path, "dataset tasks"),
            "episodes": _file_info(episode_path, "dataset episodes"),
            "total_episodes": int(info.get("total_episodes", -1)),
            "total_frames": int(info.get("total_frames", -1)),
        },
    )


def _motion_candidates(
    dataset: Path,
    info: Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    raw_frames: int,
    video_stride: int,
) -> list[dict[str, Any]]:
    episode_index = int(episode["episode_index"])
    front_path = _video_path_from_info(dataset, info, "observation.images.image", episode_index)
    wrist_path = _video_path_from_info(
        dataset, info, "observation.images.wrist_image", episode_index
    )
    front = _read_video_frames(front_path)
    wrist = _read_video_frames(wrist_path)
    length = min(len(front), len(wrist), int(episode["length"]))
    if length < raw_frames:
        return []
    front_small = np.stack(
        [_center_crop_resize(frame, width=64, height=64) for frame in front[:length]]
    ).astype(np.float32)
    wrist_small = np.stack(
        [_center_crop_resize(frame, width=64, height=64) for frame in wrist[:length]]
    ).astype(np.float32)
    front_diff = np.mean(
        np.abs(front_small[video_stride:] - front_small[:-video_stride]), axis=(1, 2, 3)
    )
    wrist_diff = np.mean(
        np.abs(wrist_small[video_stride:] - wrist_small[:-video_stride]), axis=(1, 2, 3)
    )
    candidates: list[dict[str, Any]] = []
    for start in range(length - raw_frames + 1):
        selected = list(range(start, start + raw_frames, video_stride))
        diff_indices = selected[:-1]
        front_score = float(np.mean(front_diff[diff_indices]))
        wrist_score = float(np.mean(wrist_diff[diff_indices]))
        candidates.append(
            {
                "episode_index": episode_index,
                "start_frame": start,
                "motion_score": (front_score + wrist_score) / 2.0,
                "front_motion_score": front_score,
                "wrist_motion_score": wrist_score,
                "length": length,
            }
        )
    return candidates


def prepare_w0_manifest(
    *,
    dataset_root: str | Path,
    suites: Sequence[str],
    selection: str,
    raw_frames: int,
    video_stride: int,
    base_seed: int,
    output: str | Path,
) -> dict[str, Any]:
    if selection != "one-per-task-motion-p75":
        raise ValueError(f"unsupported W0 selection: {selection}")
    if raw_frames != 33 or video_stride != 4:
        raise ValueError("the frozen W0 protocol requires raw_frames=33 and video_stride=4")
    output_path = _rooted(output)
    progress = ProgressLog(output_path.parent)
    samples: list[dict[str, Any]] = []
    dataset_fingerprint: dict[str, Any] = {}
    for suite in suites:
        dataset = _rooted(dataset_root) / f"{suite}_no_noops_lerobot"
        task_languages, episodes, fingerprint = _load_w0_task_rows(dataset)
        dataset_fingerprint[suite] = fingerprint
        if sorted(task_languages) != list(range(10)):
            raise ValueError(f"{suite} must contain task indices 0..9")
        info = _read_json(dataset / "meta/info.json")
        for task_id in range(10):
            task_episodes = [
                episode for episode in episodes.values() if episode["task_id"] == task_id
            ]
            if not task_episodes:
                raise ValueError(f"{suite} task {task_id} has no episodes")
            candidates: list[dict[str, Any]] = []
            for episode in sorted(task_episodes, key=lambda item: int(item["episode_index"])):
                candidates.extend(
                    _motion_candidates(
                        dataset,
                        info,
                        episode,
                        raw_frames=raw_frames,
                        video_stride=video_stride,
                    )
                )
            if not candidates:
                raise ValueError(f"{suite} task {task_id} has no 33-frame candidate window")
            scores = np.asarray(
                [candidate["motion_score"] for candidate in candidates], dtype=np.float64
            )
            p75 = float(np.percentile(scores, 75))
            chosen = min(
                candidates,
                key=lambda item: (
                    abs(float(item["motion_score"]) - p75),
                    int(item["episode_index"]),
                    int(item["start_frame"]),
                ),
            )
            global_index = len(samples)
            episode_index = int(chosen["episode_index"])
            start_frame = int(chosen["start_frame"])
            frame_indices = list(range(start_frame, start_frame + raw_frames, video_stride))
            front_path = _video_path_from_info(
                dataset, info, "observation.images.image", episode_index
            )
            wrist_path = _video_path_from_info(
                dataset, info, "observation.images.wrist_image", episode_index
            )
            samples.append(
                {
                    "sample_id": f"{suite}_task{task_id:02d}",
                    "suite": suite,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "start_frame": start_frame,
                    "frame_indices": frame_indices,
                    "raw_frames": raw_frames,
                    "video_stride": video_stride,
                    "front_video": _portable(front_path),
                    "wrist_video": _portable(wrist_path),
                    "front_video_info": _file_info(front_path, "selected front video"),
                    "wrist_video_info": _file_info(wrist_path, "selected wrist video"),
                    "language": task_languages[task_id],
                    "model_prompt": W0_DEFAULT_PROMPT.format(task=task_languages[task_id]),
                    "seed": int(base_seed + global_index),
                    "motion_score": float(chosen["motion_score"]),
                    "motion_p75": p75,
                    "candidate_count": len(candidates),
                }
            )
            progress.status(
                stage=f"prepare-w0:{suite}",
                completed=task_id + 1,
                total=10,
                detail=f"task={task_id} episode={episode_index} start={start_frame}",
            )
        progress.done(stage=f"prepare-w0:{suite}", total=10)
    if len(samples) != len(suites) * 10:
        raise AssertionError(f"expected {len(suites) * 10} samples, got {len(samples)}")
    manifest = {
        "schema_version": 1,
        "dataset": "W0-Video-40",
        "selection": selection,
        "dataset_root": _portable(_rooted(dataset_root)),
        "suites": list(suites),
        "protocol": {
            "kind": "fastwam-native",
            "width": 448,
            "height": 224,
            "raw_frames": raw_frames,
            "video_stride": video_stride,
            "video_frames": 9,
            "frames": 9,
            "camera_keys": ["front", "wrist"],
            "camera_resolution": [224, 224],
            "concat": "horizontal",
            "fps": 16,
            "steps": 10,
            "num_inference_steps": 10,
            "action_conditioned": False,
            "action": None,
            "proprio": None,
            "future_frame_indices": list(range(1, 9)),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "samples": samples,
    }
    _write_json(output_path, manifest)
    progress.done(stage="prepare-w0", total=len(samples))
    return manifest


def _vbench_info_path(vbench_root: Path, cache_root: Path) -> Path:
    candidates = [
        cache_root / "vbench2_i2v_full_info.json",
        cache_root / "vbench2_beta_i2v/vbench2_i2v_full_info.json",
        vbench_root / "vbench2_beta_i2v/vbench2_i2v_full_info.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "missing VBench-I2V metadata; expected vbench2_i2v_full_info.json under the pinned "
        f"checkout or {_portable(cache_root)}"
    )


def _vbench_image_roots(vbench_root: Path, cache_root: Path) -> list[Path]:
    candidates = [
        vbench_root / "vbench2_beta_i2v/data/crop",
        cache_root / "crop",
        cache_root / "vbench2_beta_i2v/data/crop",
        cache_root / "data/crop",
    ]
    return [candidate for candidate in candidates if candidate.is_dir()]


def _find_vbench_image(
    image_roots: Sequence[Path], image_name: str, resolution: str = "16-9"
) -> Path:
    for root in image_roots:
        for ratio in (resolution, "16-9", "1-1", "8-5", "7-4"):
            candidate = root / ratio / image_name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"missing VBench image {image_name!r} under {[str(p) for p in image_roots]}"
    )


def _selection_hash(seed: int, entry: Mapping[str, Any]) -> str:
    payload = (
        f"{seed}\n{entry.get('image_type', '')}\n{entry.get('image_name', '')}\n"
        f"{entry.get('prompt_en', '')}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _select_vbench_entries(info: Sequence[Mapping[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    def unique_base(dimension: str, image_type: str | None = None) -> list[dict[str, Any]]:
        candidates = [
            dict(item)
            for item in info
            if dimension in item.get("dimension", [])
            and "image_name" in item
            and (image_type is None or item.get("image_type") == image_type)
        ]
        candidates.sort(key=lambda item: (_selection_hash(seed, item), str(item["image_name"])))
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in candidates:
            name = str(item["image_name"])
            if name not in seen:
                seen.add(name)
                result.append(item)
        return result

    subject: list[dict[str, Any]] = []
    selected_image_names: set[str] = set()
    for image_type in VBENCH_SUBJECT_TYPES:
        candidates = [
            item
            for item in unique_base("i2v_subject", image_type)
            if str(item["image_name"]) not in selected_image_names
        ]
        if not candidates:
            raise ValueError(f"VBench-I2V subject category has no candidate: {image_type}")
        chosen = candidates[0]
        subject.append(chosen)
        selected_image_names.add(str(chosen["image_name"]))

    background: list[dict[str, Any]] = []
    for image_type, quota in VBENCH_BACKGROUND_QUOTA.items():
        candidates = [
            item
            for item in unique_base("i2v_background", image_type)
            if str(item["image_name"]) not in selected_image_names
        ]
        if len(candidates) < quota:
            raise ValueError(f"VBench-I2V background category {image_type} has fewer than {quota}")
        chosen = candidates[:quota]
        background.extend(chosen)
        selected_image_names.update(str(item["image_name"]) for item in chosen)

    background_candidates = [
        item
        for item in unique_base("i2v_background")
        if str(item["image_name"]) not in selected_image_names
    ]
    remaining = [
        item
        for item in background_candidates
        if str(item["image_name"]) not in selected_image_names
    ]
    if not remaining:
        raise ValueError("no unused VBench-I2V background image is available for camera-motion")
    camera_image = remaining[0]
    camera_entries: list[dict[str, Any]] = []
    for instruction in VBENCH_CAMERA_INSTRUCTIONS:
        matches = [
            dict(item)
            for item in info
            if "camera_motion" in item.get("dimension", [])
            and item.get("image_name") == camera_image.get("image_name")
            and instruction in str(item.get("prompt_en", ""))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one camera prompt for {camera_image['image_name']!r} "
                f"and {instruction!r}, got {len(matches)}"
            )
        camera_entries.append(matches[0])
    return subject + background + camera_entries


def prepare_vbench_assets(
    *, vbench_root: str | Path, cache_root: str | Path, output: str | Path
) -> dict[str, Any]:
    vbench_path = _rooted(vbench_root)
    _require_pinned_checkout(vbench_path, expected=VBENCH_COMMIT, label="VBench")
    cache_path = _rooted(cache_root)
    info_path = _vbench_info_path(vbench_path, cache_path)
    image_roots = _vbench_image_roots(vbench_path, cache_path)
    if not image_roots:
        raise FileNotFoundError(
            "VBench-I2V image suite is missing; prepare crop/{1-1,8-5,7-4,16-9} before this phase"
        )
    info = _read_json(info_path)
    if not isinstance(info, list) or len(info) == 0:
        raise ValueError(f"VBench-I2V metadata is empty: {_portable(info_path)}")
    payload = {
        "schema_version": 1,
        "dataset": "VBench-I2V",
        "vbench_root": _portable(vbench_path),
        "vbench_commit": VBENCH_COMMIT,
        "expected_vbench_commit": VBENCH_COMMIT,
        "full_info": _file_info(info_path, "VBench-I2V metadata"),
        "image_roots": [_portable(root) for root in image_roots],
        "image_root_checks": {
            _portable(root): sum(1 for item in root.rglob("*") if item.is_file())
            for root in image_roots
        },
        "evaluator_cache_root": _portable(cache_path),
        "package_versions": _package_versions(["torch", "torchvision", "vbench", "lpips"]),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_path = _rooted(output)
    _write_json(output_path, payload)
    progress = ProgressLog(output_path.parent)
    progress.done(stage="prepare-vbench", total=1)
    return payload


def _load_video_state(checkpoint: Path, *, map_location: str) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=map_location)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("mot"), Mapping):
        raise TypeError("checkpoint has no mot state dict")
    source = payload["mot"]
    prefixes = ("mixtures.video.", "video.")
    video_state: dict[str, Any] = {}
    for key, value in source.items():
        key_text = str(key)
        for prefix in prefixes:
            if key_text.startswith(prefix):
                video_state[key_text[len(prefix) :]] = value
                break
    if not video_state:
        raise ValueError("mot state dict has no video expert keys")
    return video_state


def _wan_strict_video_load(checkpoint: Path, wan_root: Path) -> dict[str, Any]:
    """Try a metadata-only strict load into the pinned Wan2.2 TI2V DiT.

    ``map_location='meta'`` avoids allocating checkpoint tensors on host RAM.  A
    failure is a protocol result, not a reason to mutate either checkpoint.
    """

    result: dict[str, Any] = {"checkpoint": _portable(checkpoint), "strict_load": False}
    model_file = wan_root / "wan/modules/model.py"
    if not model_file.is_file():
        result["reason"] = f"missing Wan2.2 model source: {_portable(model_file)}"
        return result
    try:
        import torch

        if str(wan_root) not in sys.path:
            sys.path.insert(0, str(wan_root))
        from wan.modules.model import WanModel

        video_state = _load_video_state(checkpoint, map_location="meta")
        with torch.device("meta"):
            model = WanModel(
                model_type="ti2v",
                patch_size=(1, 2, 2),
                text_len=128,
                in_dim=48,
                dim=3072,
                ffn_dim=14336,
                freq_dim=256,
                text_dim=4096,
                out_dim=48,
                num_heads=24,
                num_layers=30,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
            )
        model.load_state_dict(video_state, strict=True)
    except Exception as exc:  # noqa: BLE001 - the reason is persisted for protocol audit
        result["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        return result
    result["strict_load"] = True
    result["video_state_keys"] = len(video_state)
    return result


def _cuda_device_index(device: str) -> int:
    value = str(device)
    if not value.startswith("cuda"):
        return 0
    if ":" not in value:
        return 0
    return int(value.rsplit(":", 1)[1])


def _load_wan22_ti2v_model(
    *, checkpoint: Path, wan_root: Path, model_root: Path, device: str
) -> Any:
    import torch

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Wan2.2-compatible smoke/generation requires a CUDA device")
    if not model_root.is_dir():
        raise FileNotFoundError(f"missing official Wan2.2 model directory: {_portable(model_root)}")
    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    try:
        from wan import WanTI2V
        from wan.configs import WAN_CONFIGS
    except ImportError as exc:  # pragma: no cover - available only in the model environment
        raise RuntimeError("the pinned Wan2.2 checkout is not importable") from exc

    model = WanTI2V(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=str(model_root),
        device_id=_cuda_device_index(device),
        init_on_cpu=True,
        convert_model_dtype=True,
    )
    video_state = _load_video_state(checkpoint, map_location="cpu")
    model.model.load_state_dict(video_state, strict=True)
    return model


def _wan_video_to_frames(
    video: Any, *, width: int, height: int, frame_count: int
) -> list[np.ndarray]:
    import torch

    if torch.is_tensor(video):
        array = video.detach().float().cpu().numpy()
    else:
        array = np.asarray(video)
    if array.ndim != 4:
        raise ValueError(f"Wan2.2 returned an unexpected video shape: {array.shape}")
    if array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 3, 0))
    elif array.shape[1] in (1, 3):
        array = np.transpose(array, (0, 2, 3, 1))
    else:
        raise ValueError(f"Wan2.2 returned an unexpected channel layout: {array.shape}")
    if len(array) != frame_count:
        raise ValueError(f"Wan2.2 returned {len(array)} frames, expected {frame_count}")
    if float(array.min()) < 0.0:
        array = (array + 1.0) / 2.0
    if float(array.max()) <= 1.5:
        array = array * 255.0
    frames = []
    for frame in array:
        image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), mode="RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
        frames.append(np.asarray(image, dtype=np.uint8))
    return frames


def _generate_wan22_video(
    *, model: Any, prompt: str, input_image: np.ndarray, protocol: Mapping[str, Any], seed: int
) -> list[np.ndarray]:
    from PIL import Image

    image = Image.fromarray(np.asarray(input_image, dtype=np.uint8), mode="RGB")
    video = model.generate(
        input_prompt=prompt,
        img=image,
        max_area=int(protocol["width"]) * int(protocol["height"]),
        frame_num=int(protocol["frames"]),
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=int(protocol["steps"]),
        guide_scale=5.0,
        seed=int(seed),
        offload_model=True,
    )
    return _wan_video_to_frames(
        video,
        width=int(protocol["width"]),
        height=int(protocol["height"]),
        frame_count=int(protocol["frames"]),
    )


def check_protocol(
    *,
    wan_root: str | Path,
    fw_v: str | Path,
    fw_fz: str | Path,
    model_root: str | Path,
    device: str,
    output: str | Path,
) -> dict[str, Any]:
    wan_path = _rooted(wan_root)
    _require_pinned_checkout(wan_path, expected=WAN22_COMMIT, label="Wan2.2")
    checkpoint_paths = {"fw_v": _rooted(fw_v), "fw_fz": _rooted(fw_fz)}
    missing = [name for name, path in checkpoint_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required checkpoint(s): "
            + ", ".join(f"{name} ({_portable(checkpoint_paths[name])})" for name in missing)
        )
    checks = {
        name: _wan_strict_video_load(path, wan_path) for name, path in checkpoint_paths.items()
    }
    strict_compatible = all(bool(item.get("strict_load")) for item in checks.values())
    protocol = {
        "schema_version": 1,
        "kind": "fastwam-native",
        "generator": "FastWAM.infer_joint",
        "sampler": "fastwam-native-flow-schedule",
        "width": 448,
        "height": 224,
        "frames": 9,
        "fps": 16,
        "steps": 10,
        "action_conditioned": False,
        "action": None,
        "proprio": None,
        "input_crop_ratio": "2-1",
        "wan22_candidate": {
            "strict_video_expert_load": strict_compatible,
            "smoke": "not_run",
            "model_root": _portable(_rooted(model_root)),
        },
        "checks": checks,
        "wan_root": _portable(wan_path),
        "wan_commit": _git_head(wan_path),
        "expected_wan_commit": WAN22_COMMIT,
    }
    output_path = _rooted(output)
    progress = ProgressLog(output_path.parent)
    progress.status(stage="check-protocol", completed=0, total=1)
    if strict_compatible:
        smoke_results: dict[str, Any] = {}
        try:
            for name, checkpoint in checkpoint_paths.items():
                wan_model = _load_wan22_ti2v_model(
                    checkpoint=checkpoint,
                    wan_root=wan_path,
                    model_root=_rooted(model_root),
                    device=device,
                )
                smoke_image = np.full((720, 1280, 3), 127, dtype=np.uint8)
                smoke_frames = _generate_wan22_video(
                    model=wan_model,
                    prompt="a static test scene",
                    input_image=smoke_image,
                    protocol={"width": 1280, "height": 720, "frames": 81, "steps": 1},
                    seed=20260824,
                )
                if len(smoke_frames) != 81:
                    raise ValueError(f"{name} smoke returned {len(smoke_frames)} frames")
                smoke_results[name] = {"status": "passed", "frames": len(smoke_frames)}
                del wan_model
                import torch

                torch.cuda.empty_cache()
            protocol.update(
                {
                    "kind": "wan22-ti2v-compatible",
                    "generator": "WanTI2V.generate",
                    "sampler": "Wan2.2-official-unipc",
                    "width": 1280,
                    "height": 720,
                    "frames": 81,
                    "fps": 16,
                    "steps": 50,
                    "input_crop_ratio": "official-best-output-size",
                }
            )
            protocol["wan22_candidate"].update(
                {
                    "smoke": "passed",
                    "smoke_steps": 1,
                    "smoke_seed": 20260824,
                    "per_checkpoint": smoke_results,
                }
            )
        except Exception as exc:  # noqa: BLE001 - fallback reason is part of the protocol record
            protocol["wan22_candidate"].update(
                {
                    "smoke": "failed",
                    "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            )
    else:
        protocol["wan22_candidate"]["reason"] = "at least one video expert failed strict load"
    _write_json(output_path, protocol)
    progress.status(
        stage="check-protocol",
        completed=1,
        total=1,
        detail=f"selected={protocol['kind']}",
    )
    progress.done(stage="check-protocol", total=1)
    return protocol


def _w0_sample_input(sample: Mapping[str, Any], camera: str = "both") -> np.ndarray:
    """Build the conditioning frame for a W0 sample.

    ``camera="both"`` reproduces the frozen protocol: front and wrist centre-cropped to
    224x224 each and concatenated horizontally into 448x224, which is what training saw.

    ``camera="front"`` / ``"wrist"`` return a single 224x224 view. That is a diagnostic
    deviation, not the protocol: it exists to separate "the video expert cannot predict this
    domain" from "the video expert cannot read a two-camera side-by-side image", which the
    frozen protocol confounds. A frozen video expert carries the Wan2.2 weights unchanged, and
    no natural video contains a side-by-side pair, so the concatenated layout is out of
    distribution for it in a way a single view is not.
    """
    if camera not in {"both", "front", "wrist"}:
        raise ValueError(f"unsupported camera: {camera!r}")
    indices = [int(index) for index in sample["frame_indices"]]
    views = []
    if camera in {"both", "front"}:
        front = _read_video_frames(_rooted(sample["front_video"]), indices=[indices[0]])[0]
        views.append(_center_crop_resize(front, width=224, height=224))
    if camera in {"both", "wrist"}:
        wrist = _read_video_frames(_rooted(sample["wrist_video"]), indices=[indices[0]])[0]
        views.append(_center_crop_resize(wrist, width=224, height=224))
    return views[0] if len(views) == 1 else np.concatenate(views, axis=1)


def _w0_ground_truth(sample: Mapping[str, Any], camera: str = "both") -> list[np.ndarray]:
    if camera not in {"both", "front", "wrist"}:
        raise ValueError(f"unsupported camera: {camera!r}")
    indices = [int(index) for index in sample["frame_indices"]]
    front = (
        _read_video_frames(_rooted(sample["front_video"]), indices=indices)
        if camera in {"both", "front"}
        else None
    )
    wrist = (
        _read_video_frames(_rooted(sample["wrist_video"]), indices=indices)
        if camera in {"both", "wrist"}
        else None
    )
    frames = []
    for index in range(len(indices)):
        views = []
        if front is not None:
            views.append(_center_crop_resize(front[index], width=224, height=224))
        if wrist is not None:
            views.append(_center_crop_resize(wrist[index], width=224, height=224))
        frames.append(views[0] if len(views) == 1 else np.concatenate(views, axis=1))
    return frames


def _prepare_model_environment(vendor_root: Path, cache_root: Path) -> None:
    source_root = vendor_root / "FastWAM/src"
    for path in (source_root, vendor_root / "FastWAM"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    model_base = cache_root / "models"
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(model_base))
    os.environ.setdefault("W0_FASTWAM_MODEL_ROOT", str(model_base))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "1")


def _load_fastwam_model(
    *, checkpoint: Path, vendor_root: Path, cache_root: Path, device: str
) -> Any:
    _prepare_model_environment(vendor_root, cache_root)
    try:
        import hydra
        import torch
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - only available in fastwam-upstream
        raise RuntimeError("FastWAM generation requires the fastwam-upstream environment") from exc
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("FastWAM video generation requires a CUDA device")
    config_dir = vendor_root / "FastWAM/configs"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"missing FastWAM config directory: {_portable(config_dir)}")
    GlobalHydra.instance().clear()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("max", lambda value: max(value), replace=True)
    OmegaConf.register_new_resolver(
        "split", lambda value, index: value.split("/")[int(index)], replace=True
    )
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = hydra.compose(
            config_name="sim_libero",
            overrides=[
                "task=libero_uncond_2cam224_1e-4",
                "model.load_text_encoder=true",
                "model.skip_dit_load_from_pretrain=true",
                f"model.action_dit_pretrained_path={cache_root / 'ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'}",
            ],
        )
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    payload = model.load_checkpoint(str(checkpoint))
    _assert_checkpoint_actually_loaded(model, payload, checkpoint=checkpoint)
    return model.eval()


def _assert_checkpoint_actually_loaded(
    model: Any, payload: Mapping[str, Any] | None, *, checkpoint: Path
) -> dict[str, Any]:
    """Report what the vendor's strict=False load silently skipped.

    ``FastWAM.load_checkpoint`` calls ``self.mot.load_state_dict(payload["mot"], strict=False)``.
    Combined with ``model.skip_dit_load_from_pretrain=true`` -- which this loader sets, so the
    DiT is not seeded from the Wan2.2 weights -- any key-name mismatch leaves that parameter at
    its random initialisation and prints nothing. A randomly initialised video expert still
    produces 9 decodable frames, so the failure would only show up as unexplained garbage in the
    output video. Fail here instead.
    """
    if not isinstance(payload, Mapping) or "mot" not in payload:
        raise RuntimeError(
            f"checkpoint {_portable(checkpoint)} has no 'mot' state dict; cannot verify the load"
        )
    payload_keys = set(payload["mot"].keys())
    model_keys = set(model.mot.state_dict().keys())
    missing = sorted(model_keys - payload_keys)
    unexpected = sorted(payload_keys - model_keys)
    report = {
        "checkpoint_tensors": len(payload_keys),
        "model_tensors": len(model_keys),
        "loaded": len(model_keys & payload_keys),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_sample": missing[:5],
        "unexpected_sample": unexpected[:5],
    }
    _report_generation_env(f"checkpoint load audit: {json.dumps(report)}")
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint did not fully load into the model: "
            f"{len(missing)} model tensors absent from the checkpoint "
            f"(e.g. {missing[:3]}), {len(unexpected)} checkpoint tensors unused "
            f"(e.g. {unexpected[:3]}). The vendor loads with strict=False, so this would "
            "otherwise be silent and leave those parameters randomly initialised."
        )
    return report


def _report_generation_env(message: str) -> None:
    print(f"[W0-VIDEO-EVAL] {message}", file=sys.stderr, flush=True)


def _model_input(array: np.ndarray, *, device: str, dtype: Any) -> Any:
    import torch

    tensor = torch.from_numpy(np.asarray(array, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype) * (2.0 / 255.0) - 1.0


def _dummy_video_frames(
    *, input_image: np.ndarray, protocol: Mapping[str, Any], seed: int
) -> list[np.ndarray]:
    """Create deterministic moving-image videos for pipeline-only validation.

    This deliberately does not emulate FastWAM quality or architecture.  It only exercises
    the same manifest -> frame-shape -> video-writer path when a real checkpoint is unavailable.
    """

    width = int(protocol["width"])
    height = int(protocol["height"])
    frame_count = int(protocol["frames"])
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("dummy protocol dimensions and frame count must be positive")
    base = _center_crop_resize(np.asarray(input_image, dtype=np.uint8), width=width, height=height)
    frames: list[np.ndarray] = []
    for index in range(frame_count):
        shift_x = (index % 9) - 4
        shift_y = ((index * 2) % 7) - 3
        frame = np.roll(base, shift=(shift_y, shift_x), axis=(0, 1)).astype(np.int16)
        noise = np.random.default_rng(int(seed) + index).normal(0.0, 1.5, frame.shape)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def generate_videos(
    *,
    dataset: str,
    variant: str,
    checkpoint: str | Path | None,
    manifest: str | Path,
    output_root: str | Path,
    vendor_root: str | Path,
    cache_root: str | Path,
    wan_root: str | Path,
    wan_model_root: str | Path,
    device: str,
    overwrite: bool,
    dummy_model: bool = False,
    single_camera: str = "both",
) -> dict[str, Any]:
    if dataset not in {"vbench-mini-21", "w0-video-40"}:
        raise ValueError(f"unsupported generation dataset: {dataset}")
    if variant not in {"fw_v", "fw_fz"}:
        raise ValueError(f"unsupported variant: {variant}")
    if single_camera not in {"both", "front", "wrist"}:
        raise ValueError(f"unsupported --single-camera value: {single_camera!r}")
    if single_camera != "both" and dataset != "w0-video-40":
        raise ValueError("--single-camera only applies to the w0-video-40 dataset")
    checkpoint_path = None if checkpoint is None else _rooted(checkpoint)
    if not dummy_model and checkpoint_path is None:
        raise ValueError("a checkpoint is required unless --dummy-model is set")
    manifest_path = _rooted(manifest)
    output_path = _rooted(output_root)
    manifest_payload = _load_manifest(manifest_path)
    protocol = manifest_payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("action_conditioned") is not False:
        raise ValueError("generation manifest must freeze action_conditioned=false")
    if dataset == "vbench-mini-21" and manifest_payload.get("dataset") != "VBench-I2V-mini-21":
        raise ValueError("VBench generation requires a VBench-I2V-mini-21 manifest")
    if dataset == "w0-video-40" and manifest_payload.get("dataset") != "W0-Video-40":
        raise ValueError("W0 generation requires a W0-Video-40 manifest")
    protocol_kind = str(protocol.get("kind", ""))
    if protocol_kind not in {"fastwam-native", "wan22-ti2v-compatible"}:
        raise ValueError(f"unsupported generation protocol: {protocol_kind!r}")
    if dataset == "w0-video-40" and protocol_kind != "fastwam-native":
        raise ValueError("W0-Video-40 is fixed to the FastWAM-native protocol")
    expected_frames = int(protocol["frames"])
    expected_steps = int(protocol["steps"])
    if single_camera != "both":
        # A single view is 224x224, so the protocol's 448 width no longer applies. Mark the
        # deviation in the protocol itself so run_config.json cannot be mistaken for a formal
        # run, and so `score` fails on the shape rather than mixing the two silently.
        protocol = dict(protocol)
        protocol["width"] = 224
        protocol["height"] = 224
        protocol["camera_keys"] = [single_camera]
        protocol["concat"] = "none"
        protocol["kind"] = f"{protocol_kind}-single-camera-diagnostic"
        _report_generation_env(
            f"single-camera diagnostic: camera={single_camera} size=224x224 "
            "(deviates from the frozen 448x224 two-camera protocol)"
        )
    video_dir = output_path / "videos"
    progress = ProgressLog(output_path)
    samples = list(manifest_payload["samples"])
    existing = 0
    output_paths: dict[str, Path] = {}
    for sample in samples:
        filename = str(sample.get("video_filename", f"{sample['sample_id']}.mp4"))
        target = video_dir / filename
        output_paths[str(sample["sample_id"])] = target
        if target.is_file() and not overwrite:
            existing += 1
    progress.status(stage=f"generate:{variant}", completed=existing, total=len(samples))
    model = None
    generated = 0
    for index, sample in enumerate(samples):
        target = output_paths[str(sample["sample_id"])]
        if target.is_file() and not overwrite:
            continue
        if model is None and not dummy_model:
            assert checkpoint_path is not None
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"missing checkpoint: {_portable(checkpoint_path)}")
            if protocol_kind == "wan22-ti2v-compatible":
                _require_pinned_checkout(_rooted(wan_root), expected=WAN22_COMMIT, label="Wan2.2")
                model = _load_wan22_ti2v_model(
                    checkpoint=checkpoint_path,
                    wan_root=_rooted(wan_root),
                    model_root=_rooted(wan_model_root),
                    device=device,
                )
            else:
                model = _load_fastwam_model(
                    checkpoint=checkpoint_path,
                    vendor_root=_rooted(vendor_root),
                    cache_root=_rooted(cache_root),
                    device=device,
                )
        if dataset == "vbench-mini-21":
            input_path = _rooted(sample["conditioning_image"])
            input_image = np.asarray(Image.open(input_path).convert("RGB"), dtype=np.uint8)
        else:
            input_image = _w0_sample_input(sample, camera=single_camera)
        if input_image.shape[:2] != (int(protocol["height"]), int(protocol["width"])):
            raise ValueError(
                f"input size mismatch for {sample['sample_id']}: got {input_image.shape[:2]}, "
                f"expected {(protocol['height'], protocol['width'])}"
            )
        prompt = str(sample["model_prompt"])
        if dummy_model:
            frames = _dummy_video_frames(
                input_image=input_image,
                protocol=protocol,
                seed=int(sample["seed"]),
            )
        elif protocol_kind == "wan22-ti2v-compatible":
            frames = _generate_wan22_video(
                model=model,
                prompt=prompt,
                input_image=input_image,
                protocol=protocol,
                seed=int(sample["seed"]),
            )
        else:
            with __import__("torch").no_grad():
                prediction = model.infer_joint(
                    prompt=prompt,
                    input_image=_model_input(input_image, device=device, dtype=model.torch_dtype),
                    num_video_frames=expected_frames,
                    action_horizon=32,
                    action=None,
                    proprio=None,
                    text_cfg_scale=1.0,
                    num_inference_steps=expected_steps,
                    seed=int(sample["seed"]),
                    rand_device="cpu",
                    tiled=False,
                    test_action_with_infer_action=False,
                )
            frames = [
                np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in prediction["video"]
            ]
        if len(frames) != expected_frames:
            raise ValueError(
                f"model returned {len(frames)} frames for {sample['sample_id']}, expected {expected_frames}"
            )
        _write_video(target, frames, fps=int(protocol["fps"]))
        generated += 1
        progress.status(
            stage=f"generate:{variant}",
            completed=existing + generated,
            total=len(samples),
            detail=f"sample={index + 1}/{len(samples)} id={sample['sample_id']}",
        )
    progress.done(stage=f"generate:{variant}", total=len(samples))
    run_config = {
        "schema_version": 1,
        "dataset": dataset,
        "variant": variant,
        "checkpoint": (
            {
                "path": "dummy://deterministic-input-shift",
                "dummy": True,
            }
            if dummy_model
            else _file_info(checkpoint_path, "checkpoint")
        ),
        "manifest": {"path": _portable(manifest_path), "sha256": _sha256(manifest_path)},
        "protocol": protocol,
        "generation": {
            "model_kind": "dummy" if dummy_model else "fastwam",
            "action_conditioned": False,
            "action": None,
            "proprio": None,
            "seed_source": "manifest.sample.seed",
            "single_camera": single_camera,
            "num_inference_steps": expected_steps,
        },
        "wan_root": _portable(_rooted(wan_root)),
        "wan_model_root": _portable(_rooted(wan_model_root)),
        "generated": len(samples),
        "newly_generated": generated,
        "package_versions": _package_versions(["torch", "torchvision", "hydra-core"]),
        "platform": {"python": platform.python_version(), "system": platform.platform()},
    }
    _write_json(output_path / "run_config.json", run_config)
    return run_config


def generate_videos_multi_gpu(
    *,
    dataset: str,
    variant: str,
    checkpoint: str | Path,
    manifest: str | Path,
    output_root: str | Path,
    vendor_root: str | Path,
    cache_root: str | Path,
    wan_root: str | Path,
    wan_model_root: str | Path,
    device: str,
    overwrite: bool,
    num_gpus: int,
    config: str | Path,
) -> dict[str, Any]:
    """Generate a fixed manifest with one isolated worker process per GPU."""

    if dataset != "w0-video-40":
        raise ValueError("multi-GPU generation currently supports only W0-Video-40")
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("--device must be cuda or cuda:0 when --num-gpus>1")
    manifest_path = _rooted(manifest)
    output_path = _rooted(output_root)
    manifest_payload = _load_manifest(manifest_path)
    samples = list(manifest_payload["samples"])
    device_tokens = _require_multi_gpu_devices(num_gpus)
    work_root = output_path / ".multigpu" / "generate"
    shards = _make_manifest_shards(
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        requested_gpus=num_gpus,
        work_root=work_root,
    )

    worker_commands: list[list[str]] = []
    worker_dirs: list[Path] = []
    for worker_index, (shard_dir, _subset) in enumerate(shards):
        worker_output = shard_dir / "generation"
        worker_dirs.append(worker_output)
        worker_commands.append(
            [
                sys.executable,
                "-m",
                MODULE_NAME,
                _option("mode", "generate"),
                _option("config", config),
                _option("dataset", dataset),
                _option("variant", variant),
                _option("checkpoint", checkpoint),
                _option("manifest", shard_dir / "manifest.json"),
                _option("output-root", worker_output),
                _option("vendor-root", vendor_root),
                _option("model-cache-root", cache_root),
                _option("wan-root", wan_root),
                _option("wan-model-root", wan_model_root),
                _option("device", "cuda:0"),
                _option("num-gpus", 1),
                *(["--overwrite"] if overwrite else []),
            ]
        )
        if worker_index >= len(device_tokens):  # pragma: no cover - guarded by helper
            raise RuntimeError("not enough CUDA devices for the generated worker shards")

    _run_multi_gpu_workers(
        stage=f"generate:{variant}:multi-gpu",
        worker_commands=worker_commands,
        worker_dirs=worker_dirs,
        device_tokens=device_tokens[: len(shards)],
        progress_root=output_path,
    )

    child_configs: list[dict[str, Any]] = []
    video_dir = output_path / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for shard_dir, subset in shards:
        child_config_path = shard_dir / "generation/run_config.json"
        child_config = _read_json(child_config_path)
        if not isinstance(child_config, dict):
            raise TypeError(f"invalid worker run config: {_portable(child_config_path)}")
        if child_config.get("generated") != len(subset):
            raise ValueError(
                f"worker generated count mismatch in {_portable(child_config_path)}: "
                f"expected {len(subset)}, got {child_config.get('generated')}"
            )
        child_configs.append(child_config)
        source_dir = shard_dir / "generation/videos"
        for sample in subset:
            filename = str(sample.get("video_filename", f"{sample['sample_id']}.mp4"))
            source = source_dir / filename
            target = video_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"missing worker video: {_portable(source)}")
            if overwrite or not target.exists():
                shutil.copy2(source, target)

    expected_files = {
        str(sample.get("video_filename", f"{sample['sample_id']}.mp4")) for sample in samples
    }
    missing = sorted(
        filename for filename in expected_files if not (video_dir / filename).is_file()
    )
    if missing:
        raise FileNotFoundError(f"multi-GPU generation missing merged videos: {missing[:5]}")

    run_config = dict(child_configs[0])
    run_config["manifest"] = {
        "path": _portable(manifest_path),
        "sha256": _sha256(manifest_path),
    }
    run_config["generated"] = len(samples)
    run_config["newly_generated"] = sum(
        int(config_payload.get("newly_generated", 0)) for config_payload in child_configs
    )
    run_config["multi_gpu"] = {
        "requested_gpus": num_gpus,
        "worker_count": len(shards),
        "device_tokens": list(device_tokens[: len(shards)]),
        "strategy": "round_robin_manifest_order",
        "shards": [
            {
                "index": index,
                "manifest": _portable(shard_dir / "manifest.json"),
                "output": _portable(shard_dir / "generation"),
                "sample_count": len(subset),
            }
            for index, (shard_dir, subset) in enumerate(shards)
        ],
    }
    _write_json(output_path / "run_config.json", run_config)
    ProgressLog(output_path).done(stage=f"generate:{variant}", total=len(samples))
    return run_config


def _ssim_global(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64) / 255.0
    y = np.asarray(second, dtype=np.float64) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = x.mean(axis=(0, 1))
    mu_y = y.mean(axis=(0, 1))
    var_x = ((x - mu_x) ** 2).mean(axis=(0, 1))
    var_y = ((y - mu_y) ** 2).mean(axis=(0, 1))
    cov = ((x - mu_x) * (y - mu_y)).mean(axis=(0, 1))
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(np.mean(score))


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(
        np.mean((np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)) ** 2)
    )
    return float(10.0 * math.log10((255.0**2) / max(mse, 1e-12)))


def _find_lpips_weights(explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        path = _rooted(explicit)
        return path if path.is_file() else None
    candidates = [
        Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        / "hub/checkpoints/alex.pth",
        Path.home() / ".cache/torch/hub/checkpoints/alex.pth",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _load_lpips(
    explicit_weights: str | Path | None, device: str, allow_missing: bool
) -> tuple[Any, dict[str, Any]]:
    weight_path = _find_lpips_weights(explicit_weights)
    if weight_path is None:
        if allow_missing:
            return None, {"available": False, "reason": "LPIPS alex weights not found"}
        raise FileNotFoundError(
            "LPIPS evaluator weights are missing; prepare alex.pth locally or pass "
            "--allow-missing-lpips for a PSNR/SSIM-only diagnostic"
        )
    try:
        import lpips

        model = lpips.LPIPS(net="alex", model_path=str(weight_path)).to(device).eval()
        return model, {
            "available": True,
            "package_version": importlib.metadata.version("lpips"),
            "weights": _file_info(weight_path, "LPIPS weights"),
        }
    except Exception as exc:  # pragma: no cover - depends on optional evaluator env
        if allow_missing:
            return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        raise RuntimeError("LPIPS evaluator initialization failed") from exc


def _lpips_value(model: Any, first: np.ndarray, second: np.ndarray, device: str) -> float:
    import torch

    x = torch.from_numpy(first).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    y = torch.from_numpy(second).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    with torch.no_grad():
        return float(model(x.to(device), y.to(device)).mean().item())


def _camera_metric_frames(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _height, width = frame.shape[:2]
    if width % 2 != 0:
        raise ValueError(f"concatenated W0 frame width must be even, got {width}")
    return frame[:, : width // 2], frame[:, width // 2 :]


def _w0_sample_metrics(
    gt_frames: Sequence[np.ndarray],
    pred_frames: Sequence[np.ndarray],
    *,
    lpips_model: Any,
    lpips_device: str,
) -> dict[str, float | None]:
    if len(gt_frames) != 9 or len(pred_frames) != 9:
        raise TypeError(
            f"W0 metrics require exactly 9 frames, got gt={len(gt_frames)} pred={len(pred_frames)}"
        )
    values: dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips": []}
    for frame_index in range(1, 9):
        gt_cameras = _camera_metric_frames(gt_frames[frame_index])
        pred_cameras = _camera_metric_frames(pred_frames[frame_index])
        for gt_camera, pred_camera in zip(gt_cameras, pred_cameras):
            if gt_camera.shape != pred_camera.shape:
                raise ValueError(
                    f"GT/pred camera shape mismatch: {gt_camera.shape} vs {pred_camera.shape}"
                )
            values["psnr"].append(_psnr(gt_camera, pred_camera))
            values["ssim"].append(_ssim_global(gt_camera, pred_camera))
            if lpips_model is not None:
                values["lpips"].append(
                    _lpips_value(lpips_model, gt_camera, pred_camera, lpips_device)
                )
    return {
        "psnr": float(np.mean(values["psnr"])),
        "ssim": float(np.mean(values["ssim"])),
        "lpips": None if not values["lpips"] else float(np.mean(values["lpips"])),
    }


def _aggregate_sample_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_suite: dict[str, dict[str, Any]] = {}
    suites = sorted({str(sample["suite"]) for sample in samples})
    for suite in suites:
        rows = [sample for sample in samples if sample["suite"] == suite]
        per_suite[suite] = {
            "count": len(rows),
            "psnr": float(np.mean([row["psnr"] for row in rows])),
            "ssim": float(np.mean([row["ssim"] for row in rows])),
            "lpips": (
                None
                if any(row["lpips"] is None for row in rows)
                else float(np.mean([row["lpips"] for row in rows]))
            ),
        }
    aggregate = {
        "count": len(samples),
        "psnr": float(np.mean([sample["psnr"] for sample in samples])),
        "ssim": float(np.mean([sample["ssim"] for sample in samples])),
        "lpips": (
            None
            if any(sample["lpips"] is None for sample in samples)
            else float(np.mean([sample["lpips"] for sample in samples]))
        ),
    }
    return {"per_suite": per_suite, "macro": aggregate}


def score_w0(
    *,
    variant: str,
    manifest: str | Path,
    videos: str | Path,
    output: str | Path,
    checkpoint: str | Path | None,
    lpips_weights: str | Path | None,
    device: str,
    allow_missing_lpips: bool,
) -> dict[str, Any]:
    manifest_path = _rooted(manifest)
    manifest_payload = _load_manifest(manifest_path)
    if manifest_payload.get("dataset") != "W0-Video-40":
        raise ValueError("W0 score requires a W0-Video-40 manifest")
    output_path = _rooted(output)
    progress = ProgressLog(output_path.parent)
    video_root = _rooted(videos)
    lpips_model, lpips_info = _load_lpips(lpips_weights, device, allow_missing_lpips)
    scored: list[dict[str, Any]] = []
    samples = list(manifest_payload["samples"])
    progress.status(stage=f"score:{variant}", completed=0, total=len(samples))
    for index, sample in enumerate(samples):
        video_path = video_root / f"{sample['sample_id']}.mp4"
        gt_frames = _w0_ground_truth(sample)
        pred_frames = _read_video_frames(video_path)
        metrics = _w0_sample_metrics(
            gt_frames,
            pred_frames,
            lpips_model=lpips_model,
            lpips_device=device,
        )
        scored.append(
            {
                "sample_id": sample["sample_id"],
                "suite": sample["suite"],
                "task_id": sample["task_id"],
                **metrics,
            }
        )
        progress.status(
            stage=f"score:{variant}",
            completed=index + 1,
            total=len(samples),
            detail=f"sample={sample['sample_id']}",
        )
    progress.done(stage=f"score:{variant}", total=len(samples))
    checkpoint_info = None if checkpoint is None else _file_info(_rooted(checkpoint), "checkpoint")
    if checkpoint_info is None:
        run_config_path = video_root.parent / "run_config.json"
        if run_config_path.is_file():
            run_config = _read_json(run_config_path)
            checkpoint_info = run_config.get("checkpoint")
    if not isinstance(checkpoint_info, Mapping):
        raise TypeError(
            "W0 metrics require checkpoint provenance; pass --checkpoint or generate videos "
            "with this evaluator first"
        )
    payload = {
        "schema_version": 1,
        "dataset": "W0-Video-40",
        "variant": variant,
        "manifest": {"path": _portable(manifest_path), "sha256": _sha256(manifest_path)},
        "protocol": manifest_payload["protocol"],
        "checkpoint": checkpoint_info,
        "metric_definition": {
            "future_frame_indices": list(range(1, 9)),
            "camera_aggregation": "equal mean of front and wrist",
            "psnr": "RGB frame PSNR, data range 0..255",
            "ssim": "global channel-wise SSIM with standard C1/C2 constants",
            "lpips": "LPIPS 0.1.4 AlexNet when evaluator weights are available",
        },
        "metric_availability": {"psnr": True, "ssim": True, "lpips": lpips_info["available"]},
        "lpips_evaluator": lpips_info,
        "samples": scored,
        "summary": _aggregate_sample_metrics(scored),
        "package_versions": _package_versions(["numpy", "Pillow", "lpips", "torch"]),
    }
    _write_json(output_path, payload)
    return payload


def score_w0_multi_gpu(
    *,
    variant: str,
    manifest: str | Path,
    videos: str | Path,
    output: str | Path,
    checkpoint: str | Path | None,
    lpips_weights: str | Path | None,
    device: str,
    allow_missing_lpips: bool,
    num_gpus: int,
    config: str | Path,
) -> dict[str, Any]:
    """Score a W0 manifest with one LPIPS worker process per GPU."""

    if device not in {"cuda", "cuda:0"}:
        raise ValueError("--device must be cuda or cuda:0 when --num-gpus>1")
    manifest_path = _rooted(manifest)
    output_path = _rooted(output)
    manifest_payload = _load_manifest(manifest_path)
    if manifest_payload.get("dataset") != "W0-Video-40":
        raise ValueError("multi-GPU score requires a W0-Video-40 manifest")
    samples = list(manifest_payload["samples"])
    device_tokens = _require_multi_gpu_devices(num_gpus)
    work_root = output_path.parent / ".multigpu" / "score"
    shards = _make_manifest_shards(
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        requested_gpus=num_gpus,
        work_root=work_root,
    )

    worker_commands: list[list[str]] = []
    worker_dirs: list[Path] = []
    for shard_dir, _subset in shards:
        worker_output = shard_dir / "metrics.json"
        worker_dirs.append(shard_dir)
        command = [
            sys.executable,
            "-m",
            MODULE_NAME,
            _option("mode", "score"),
            _option("config", config),
            _option("dataset", "w0-video-40"),
            _option("variant", variant),
            _option("manifest", shard_dir / "manifest.json"),
            _option("videos", videos),
            _option("output", worker_output),
            _option("device", "cuda:0"),
            _option("num-gpus", 1),
        ]
        if checkpoint is not None:
            command.append(_option("checkpoint", checkpoint))
        if lpips_weights is not None:
            command.append(_option("lpips-weights", lpips_weights))
        if allow_missing_lpips:
            command.append("--allow-missing-lpips")
        worker_commands.append(command)

    _run_multi_gpu_workers(
        stage=f"score:{variant}:multi-gpu",
        worker_commands=worker_commands,
        worker_dirs=worker_dirs,
        device_tokens=device_tokens[: len(shards)],
        progress_root=output_path.parent,
    )

    expected_ids = [str(sample["sample_id"]) for sample in samples]
    metrics_by_id: dict[str, dict[str, Any]] = {}
    child_metrics: list[dict[str, Any]] = []
    for shard_dir, subset in shards:
        child_path = shard_dir / "metrics.json"
        child_payload = _read_json(child_path)
        if not isinstance(child_payload, dict):
            raise TypeError(f"invalid worker metrics: {_portable(child_path)}")
        if child_payload.get("dataset") != "W0-Video-40" or child_payload.get("variant") != variant:
            raise ValueError(f"worker metrics identity mismatch: {_portable(child_path)}")
        rows = child_payload.get("samples")
        if not isinstance(rows, list) or len(rows) != len(subset):
            raise ValueError(f"worker metrics sample count mismatch: {_portable(child_path)}")
        child_metrics.append(child_payload)
        for row in rows:
            if not isinstance(row, dict) or "sample_id" not in row:
                raise TypeError(f"invalid worker metric row: {_portable(child_path)}")
            sample_id = str(row["sample_id"])
            if sample_id in metrics_by_id:
                raise ValueError(f"duplicate W0 metric sample: {sample_id}")
            metrics_by_id[sample_id] = row

    if set(metrics_by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(metrics_by_id))
        extra = sorted(set(metrics_by_id) - set(expected_ids))
        raise ValueError(
            f"merged W0 metrics sample mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    first = dict(child_metrics[0])
    availability = [payload.get("metric_availability") for payload in child_metrics]
    if any(item != availability[0] for item in availability[1:]):
        raise ValueError("W0 workers reported inconsistent metric availability")
    merged_samples = [metrics_by_id[sample_id] for sample_id in expected_ids]
    first["manifest"] = {
        "path": _portable(manifest_path),
        "sha256": _sha256(manifest_path),
    }
    first["samples"] = merged_samples
    first["summary"] = _aggregate_sample_metrics(merged_samples)
    first["multi_gpu"] = {
        "requested_gpus": num_gpus,
        "worker_count": len(shards),
        "device_tokens": list(device_tokens[: len(shards)]),
        "strategy": "round_robin_manifest_order",
        "shards": [
            {
                "index": index,
                "manifest": _portable(shard_dir / "manifest.json"),
                "metrics": _portable(shard_dir / "metrics.json"),
                "sample_count": len(subset),
            }
            for index, (shard_dir, subset) in enumerate(shards)
        ],
    }
    _write_json(output_path, first)
    ProgressLog(output_path.parent).done(stage=f"score:{variant}", total=len(samples))
    return first


def _vbench_weight_checks(cache_root: Path) -> dict[str, Any]:
    expected = {
        "dino": cache_root / "dino_model/dino_vitbase16_pretrain.pth",
        "clip_b32": cache_root / "clip_model/ViT-B-32.pt",
        "clip_l14": cache_root / "clip_model/ViT-L-14.pt",
        "aesthetic": cache_root / "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
        "musiq": cache_root / "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
        "amt": cache_root / "amt_model/amt-s.pth",
        "raft": cache_root / "raft_model/models/raft-things.pth",
    }
    required_dirs = {
        "dino_repo": cache_root / "dino_model/facebookresearch_dino_main",
    }
    checks: dict[str, Any] = {}
    for key, path in expected.items():
        checks[key] = (
            _file_info(path, f"VBench evaluator weight {key}")
            if path.is_file()
            else {
                "path": _portable(path),
                "missing": True,
            }
        )
    for key, path in required_dirs.items():
        checks[key] = (
            {"path": _portable(path), "directory": True}
            if path.is_dir()
            else {"path": _portable(path), "missing": True, "directory": True}
        )
    missing = [key for key, value in checks.items() if value.get("missing")]
    if missing:
        raise FileNotFoundError(
            "missing VBench evaluator weights: "
            + ", ".join(f"{key} ({checks[key]['path']})" for key in missing)
        )
    return checks


def _vbench_custom_info(manifest: Mapping[str, Any], video_root: Path) -> list[dict[str, Any]]:
    records = []
    for sample in manifest["samples"]:
        video_name = str(sample["video_filename"])
        video_path = video_root / video_name
        records.append(
            {
                "prompt_en": sample["prompt_en"],
                "dimension": list(sample["dimensions"]),
                "video_list": [str(video_path.resolve())],
                "custom_image_path": str(_rooted(sample["conditioning_image"]).resolve()),
                "sample_id": sample["sample_id"],
            }
        )
    return records


def score_vbench(
    *,
    variant: str,
    manifest: str | Path,
    videos: str | Path,
    output: str | Path,
    vbench_root: str | Path,
    cache_root: str | Path,
    device: str,
    allow_vbench_downloads: bool,
) -> dict[str, Any]:
    manifest_path = _rooted(manifest)
    manifest_payload = _load_manifest(manifest_path)
    if manifest_payload.get("dataset") != "VBench-I2V-mini-21":
        raise ValueError("VBench score requires a VBench-I2V-mini-21 manifest")
    video_root = _rooted(videos)
    output_path = _rooted(output)
    progress = ProgressLog(output_path.parent)
    missing_videos = [
        sample["sample_id"]
        for sample in manifest_payload["samples"]
        if not (video_root / str(sample["video_filename"])).is_file()
    ]
    if missing_videos:
        raise FileNotFoundError(f"missing VBench videos: {missing_videos[:5]}")
    cache_path = _rooted(cache_root)
    weight_checks: dict[str, Any]
    if allow_vbench_downloads:
        weight_checks = {"validation": "skipped (--allow-vbench-downloads)"}
    else:
        weight_checks = _vbench_weight_checks(cache_path)
    vbench_path = _rooted(vbench_root)
    if not vbench_path.is_dir():
        raise FileNotFoundError(f"missing pinned VBench checkout: {_portable(vbench_path)}")
    _require_pinned_checkout(vbench_path, expected=VBENCH_COMMIT, label="VBench")
    if str(vbench_path) not in sys.path:
        sys.path.insert(0, str(vbench_path))
    os.environ["VBENCH_CACHE_DIR"] = str(cache_path)
    os.environ.setdefault("TORCH_HOME", str(cache_path / "torch"))
    try:
        import torch
        from vbench2_beta_i2v.utils import init_submodules
    except ImportError as exc:  # pragma: no cover - depends on external evaluator env
        raise RuntimeError("VBench scoring requires the pinned VBench Python dependencies") from exc
    protocol = manifest_payload["protocol"]
    info_path = output_path.parent / "vbench_custom_info.json"
    _write_json(info_path, _vbench_custom_info(manifest_payload, video_root))
    try:
        submodules = init_submodules(
            list(VBENCH_DIMENSIONS),
            local=not allow_vbench_downloads,
            read_frame=False,
            resolution=str(protocol.get("vbench_resolution", "16-9")),
        )
    except Exception as exc:
        raise RuntimeError("VBench evaluator initialization failed") from exc
    dimension_results: dict[str, Any] = {}
    dimension_counts = {
        dimension: sum(dimension in sample["dimensions"] for sample in manifest_payload["samples"])
        for dimension in VBENCH_DIMENSIONS
    }
    for index, dimension in enumerate(VBENCH_DIMENSIONS):
        progress.status(
            stage=f"score:{variant}:{dimension}",
            completed=0,
            total=dimension_counts[dimension],
        )
        if dimension in {"i2v_subject", "i2v_background", "camera_motion"}:
            module_name = f"vbench2_beta_i2v.{dimension}"
        else:
            module_name = f"vbench.{dimension}"
        module = importlib.import_module(module_name)
        compute = getattr(module, f"compute_{dimension}")
        kwargs = {}
        if dimension == "imaging_quality":
            kwargs["imaging_quality_preprocessing_mode"] = "longer"
        result = compute(str(info_path), torch.device(device), submodules[dimension], **kwargs)
        score = result[0] if isinstance(result, (tuple, list)) else result
        dimension_results[dimension] = {
            "score": float(score),
            "sample_count": dimension_counts[dimension],
        }
        progress.status(
            stage=f"score:{variant}:{dimension}",
            completed=dimension_counts[dimension],
            total=dimension_counts[dimension],
            detail=f"score={float(score):.8f}",
        )
        progress.done(stage=f"score:{variant}:{dimension}", total=dimension_counts[dimension])
    payload = {
        "schema_version": 1,
        "dataset": "VBench-I2V-mini-21",
        "variant": variant,
        "manifest": {"path": _portable(manifest_path), "sha256": _sha256(manifest_path)},
        "protocol": protocol,
        "checkpoint": None,
        "vbench": {
            "root": _portable(vbench_path),
            "commit": VBENCH_COMMIT,
            "expected_commit": VBENCH_COMMIT,
            "custom_info": _portable(info_path),
            "dimensions": dimension_results,
            "weights": weight_checks,
        },
        "package_versions": _package_versions(
            ["torch", "torchvision", "vbench", "decord", "dreamsim", "clip", "pyiqa"]
        ),
    }
    run_config = video_root.parent / "run_config.json"
    if run_config.is_file():
        generation_config = _read_json(run_config)
        payload["checkpoint"] = generation_config.get("checkpoint")
    if not isinstance(payload["checkpoint"], Mapping):
        raise TypeError(
            "VBench metrics require checkpoint provenance; generate videos with this evaluator "
            "before scoring"
        )
    _write_json(output_path, payload)
    progress.done(stage=f"score:{variant}", total=len(manifest_payload["samples"]))
    return payload


def _metric_delta(name: str, fw_v: float | None, fw_fz: float | None) -> float | None:
    if fw_v is None or fw_fz is None:
        return None
    return float(fw_v - fw_fz) if name != "lpips" else float(fw_fz - fw_v)


def _load_metrics(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"metrics must be a JSON object: {_portable(path)}")
    return payload


def _make_labelled_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 155, 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return np.asarray(image)


def _make_comparison_video(
    *,
    sample: Mapping[str, Any],
    fw_fz_video: Path,
    fw_v_video: Path,
    output: Path,
) -> None:
    gt = _w0_ground_truth(sample)
    fz = _read_video_frames(fw_fz_video)
    fwv = _read_video_frames(fw_v_video)
    if not (len(gt) == len(fz) == len(fwv) == 9):
        raise ValueError(f"comparison video frame count mismatch for {sample['sample_id']}")
    frames = []
    for index in range(9):
        panels = [
            _make_labelled_frame(gt[0], "conditioning"),
            _make_labelled_frame(gt[index], "GT"),
            _make_labelled_frame(fz[index], "FW-FZ"),
            _make_labelled_frame(fwv[index], "FW-V"),
        ]
        frames.append(np.concatenate(panels, axis=1))
    _write_video(output, frames, fps=16)


def _make_suite_montage(comparison_paths: Sequence[Path], output: Path) -> None:
    if len(comparison_paths) != 10:
        raise ValueError(
            f"suite montage requires 10 comparison videos, got {len(comparison_paths)}"
        )
    all_videos = [_read_video_frames(path) for path in comparison_paths]
    frames = []
    for frame_index in range(9):
        rows = []
        for row_index in range(2):
            row = [all_videos[row_index * 5 + col][frame_index] for col in range(5)]
            rows.append(np.concatenate(row, axis=1))
        frames.append(np.concatenate(rows, axis=0))
    _write_video(output, frames, fps=16)


def compare_metrics(
    *,
    vbench_fw_fz: str | Path,
    vbench_fw_v: str | Path,
    w0_fw_fz: str | Path,
    w0_fw_v: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    vbench_fz = _load_metrics(vbench_fw_fz)
    vbench_v = _load_metrics(vbench_fw_v)
    w0_fz = _load_metrics(w0_fw_fz)
    w0_v = _load_metrics(w0_fw_v)
    for left, right, label in (
        (vbench_fz, vbench_v, "VBench"),
        (w0_fz, w0_v, "W0"),
    ):
        if left.get("manifest", {}).get("sha256") != right.get("manifest", {}).get("sha256"):
            raise ValueError(f"{label} metrics do not use the same manifest")
        if left.get("protocol") != right.get("protocol"):
            raise ValueError(f"{label} metrics do not use the same protocol")
    vbench_delta: dict[str, Any] = {}
    for dimension in VBENCH_DIMENSIONS:
        fz_score = vbench_fz["vbench"]["dimensions"][dimension]["score"]
        fwv_score = vbench_v["vbench"]["dimensions"][dimension]["score"]
        vbench_delta[dimension] = {
            "fw_fz": fz_score,
            "fw_v": fwv_score,
            "delta_fw_v_minus_fw_fz": float(fwv_score - fz_score),
        }
    fz_rows = {str(row["sample_id"]): row for row in w0_fz["samples"]}
    fwv_rows = {str(row["sample_id"]): row for row in w0_v["samples"]}
    if set(fz_rows) != set(fwv_rows):
        raise ValueError("W0 metrics do not contain the same sample IDs")
    w0_per_sample = []
    for sample_id in sorted(fz_rows):
        row_fz = fz_rows[sample_id]
        row_v = fwv_rows[sample_id]
        w0_per_sample.append(
            {
                "sample_id": sample_id,
                "suite": row_fz["suite"],
                "task_id": row_fz["task_id"],
                "psnr_delta_fw_v_minus_fw_fz": _metric_delta("psnr", row_v["psnr"], row_fz["psnr"]),
                "ssim_delta_fw_v_minus_fw_fz": _metric_delta("ssim", row_v["ssim"], row_fz["ssim"]),
                "lpips_delta_fw_fz_minus_fw_v": _metric_delta(
                    "lpips", row_v["lpips"], row_fz["lpips"]
                ),
            }
        )
    output_path = _rooted(output)
    progress = ProgressLog(output_path.parent)
    comparison_video_dir = output_path.parent / "w0_video_40/comparison_videos"
    montage_dir = output_path.parent / "w0_video_40/montages"
    manifest_path = _rooted(w0_v["manifest"]["path"])
    manifest = _load_manifest(manifest_path)
    fz_video_root = _rooted(Path(w0_fw_fz).parent / "videos")
    fwv_video_root = _rooted(Path(w0_fw_v).parent / "videos")
    comparison_paths: dict[str, Path] = {}
    for index, sample in enumerate(manifest["samples"]):
        sample_id = str(sample["sample_id"])
        path = comparison_video_dir / f"{sample_id}.mp4"
        _make_comparison_video(
            sample=sample,
            fw_fz_video=fz_video_root / f"{sample_id}.mp4",
            fw_v_video=fwv_video_root / f"{sample_id}.mp4",
            output=path,
        )
        comparison_paths[sample_id] = path
        progress.status(
            stage="compare:w0-videos", completed=index + 1, total=len(manifest["samples"])
        )
    for suite in W0_SUITES:
        paths = [comparison_paths[f"{suite}_task{task_id:02d}"] for task_id in range(10)]
        _make_suite_montage(paths, montage_dir / f"{suite}.mp4")
    progress.done(stage="compare:w0-videos", total=len(manifest["samples"]))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "vbench": vbench_delta,
        "w0_video_40": {
            "fw_fz": w0_fz["summary"],
            "fw_v": w0_v["summary"],
            "per_sample": w0_per_sample,
            "delta_convention": {
                "psnr": "FW-V - FW-FZ",
                "ssim": "FW-V - FW-FZ",
                "lpips": "FW-FZ - FW-V",
            },
            "comparison_videos": _portable(comparison_video_dir),
            "montages": _portable(montage_dir),
        },
        "inputs": {
            "vbench_fw_fz": _portable(vbench_fw_fz),
            "vbench_fw_v": _portable(vbench_fw_v),
            "w0_fw_fz": _portable(w0_fw_fz),
            "w0_fw_v": _portable(w0_fw_v),
        },
    }
    _write_json(output_path, payload)
    progress.done(stage="compare", total=1)
    return payload


def _parse_suites(value: str) -> list[str]:
    suites = [item.strip() for item in value.split(",") if item.strip()]
    if not suites:
        raise ValueError("--suites must not be empty")
    unknown = sorted(set(suites) - set(W0_SUITES))
    if unknown:
        raise ValueError(f"unknown W0 suites: {unknown}")
    return suites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["prepare-vbench", "prepare-w0", "check-protocol", "generate", "score", "compare"],
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--vbench-root", default="third_party/vbench")
    parser.add_argument("--wan-root", default="third_party/wan22")
    parser.add_argument("--cache-root", default="pretrained/vbench_i2v")
    parser.add_argument(
        "--dataset-root", default="pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2"
    )
    parser.add_argument("--suites", default=",".join(W0_SUITES))
    parser.add_argument("--selection")
    parser.add_argument("--selection-seed", type=int, default=20260824)
    parser.add_argument("--base-seed", type=int, default=52000)
    parser.add_argument("--raw-frames", type=int, default=33)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--protocol")
    parser.add_argument("--fw-v")
    parser.add_argument("--fw-fz")
    parser.add_argument("--checkpoint")
    parser.add_argument("--variant", choices=["fw_v", "fw_fz"])
    parser.add_argument("--dataset", choices=["vbench-mini-21", "w0-video-40"])
    parser.add_argument("--manifest")
    parser.add_argument("--videos")
    parser.add_argument("--output")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--vendor-root", default="third_party/w0_fastwam")
    parser.add_argument("--model-cache-root", default="pretrained/w0_fastwam")
    parser.add_argument(
        "--wan-model-root", default="pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--single-camera",
        choices=("both", "front", "wrist"),
        default="both",
        help=(
            "diagnostic: condition on one 224x224 camera instead of the frozen 448x224 "
            "two-camera concat, to separate domain failure from layout failure"
        ),
    )
    parser.add_argument(
        "--dummy-model",
        action="store_true",
        help=(
            "generate deterministic input-shift videos without loading a checkpoint "
            "(pipeline smoke only)"
        ),
    )
    parser.add_argument("--allow-missing-lpips", action="store_true")
    parser.add_argument("--lpips-weights")
    parser.add_argument("--allow-vbench-downloads", action="store_true")
    parser.add_argument("--vbench-fw-fz")
    parser.add_argument("--vbench-fw-v")
    parser.add_argument("--w0-fw-fz")
    parser.add_argument("--w0-fw-v")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_gpus <= 0:
        raise SystemExit("--num-gpus must be a positive integer")
    if args.num_gpus > 1:
        if args.mode not in {"generate", "score"}:
            raise SystemExit("--num-gpus>1 is only supported for generate and score")
        if args.device not in {"cuda", "cuda:0"}:
            raise SystemExit("--device must be cuda or cuda:0 when --num-gpus>1")
        if args.dummy_model:
            raise SystemExit("--dummy-model currently supports --num-gpus=1 only")
        if args.single_camera != "both":
            raise SystemExit("--single-camera currently supports --num-gpus=1 only")
    # Loading the config is intentional even when the caller only uses the
    # protocol helper: it makes a run fail early if the experiment contract was
    # edited without updating its entry point.
    _load_config(args.config)
    if args.mode == "prepare-vbench":
        if not args.output:
            raise SystemExit("--output is required for prepare-vbench")
        if args.selection:
            assets_path = _rooted(args.output).parent / "assets.json"
            assets = _read_json(assets_path)
            if assets.get("expected_vbench_commit") != VBENCH_COMMIT:
                raise ValueError("assets.json was not prepared from the pinned VBench commit")
            full_info = _read_json(_rooted(assets["full_info"]["path"]))
            protocol_path = args.protocol
            if not protocol_path:
                raise SystemExit("--protocol is required when --selection=mini-21")
            protocol = _protocol_from_file(_rooted(protocol_path))
            entries = _select_vbench_entries(full_info, seed=args.selection_seed)
            if len(entries) != 21:
                raise ValueError(f"expected 21 selected VBench entries, got {len(entries)}")
            image_roots = [_rooted(path) for path in assets["image_roots"]]
            output_path = _rooted(args.output)
            image_output = output_path.parent / "images"
            samples = []
            for index, entry in enumerate(entries):
                image_name = str(entry["image_name"])
                source = _find_vbench_image(image_roots, image_name)
                image = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
                conditioned = _center_crop_resize(
                    image, width=int(protocol["width"]), height=int(protocol["height"])
                )
                group = (
                    "subject"
                    if "i2v_subject" in entry["dimension"]
                    else "background"
                    if "i2v_background" in entry["dimension"]
                    else "camera"
                )
                sample_id = (
                    f"{group}_{index:02d}_{_safe_filename(str(entry['prompt_en']), max_length=70)}"
                )
                conditioning_path = (
                    image_output / f"{index:02d}_{_safe_filename(image_name, max_length=80)}.png"
                )
                _write_image(conditioning_path, conditioned)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "group": group,
                        "image_type": entry.get("image_type"),
                        "image_name": image_name,
                        "source_image": _portable(source),
                        "source_image_info": _file_info(source, "VBench source image"),
                        "conditioning_image": _portable(conditioning_path),
                        "conditioning_image_info": _file_info(
                            conditioning_path, "VBench conditioning image"
                        ),
                        "prompt_en": str(entry["prompt_en"]),
                        "model_prompt": str(entry["prompt_en"]),
                        "dimensions": list(entry["dimension"]),
                        "seed": int(args.base_seed + index),
                        "video_filename": f"{index:02d}__{_safe_filename(str(entry['prompt_en']))}.mp4",
                    }
                )
            if len({sample["image_name"] for sample in samples}) != 15:
                raise ValueError("VBench-I2V-mini-21 must contain exactly 15 unique images")
            manifest = {
                "schema_version": 1,
                "dataset": "VBench-I2V-mini-21",
                "selection": "mini-21",
                "selection_seed": args.selection_seed,
                "base_seed": args.base_seed,
                "protocol": {**protocol, "vbench_resolution": "16-9"},
                "assets": assets,
                "selected_image_count": len({sample["image_name"] for sample in samples}),
                "samples": samples,
            }
            _write_json(output_path, manifest)
            ProgressLog(output_path.parent).done(stage="prepare-vbench:mini-21", total=len(samples))
        else:
            prepare_vbench_assets(
                vbench_root=args.vbench_root,
                cache_root=args.cache_root,
                output=args.output,
            )
    elif args.mode == "prepare-w0":
        prepare_w0_manifest(
            dataset_root=args.dataset_root,
            suites=_parse_suites(args.suites),
            selection=args.selection or "one-per-task-motion-p75",
            raw_frames=args.raw_frames,
            video_stride=args.video_stride,
            base_seed=args.base_seed,
            output=args.output
            or "outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json",
        )
    elif args.mode == "check-protocol":
        if not args.fw_v or not args.fw_fz or not args.output:
            raise SystemExit("check-protocol requires --fw-v, --fw-fz and --output")
        check_protocol(
            wan_root=args.wan_root,
            fw_v=args.fw_v,
            fw_fz=args.fw_fz,
            model_root=args.wan_model_root,
            device=args.device,
            output=args.output,
        )
    elif args.mode == "generate":
        required = {
            "dataset": args.dataset,
            "variant": args.variant,
            "manifest": args.manifest,
            "output_root": args.output_root,
        }
        if not args.dummy_model:
            required["checkpoint"] = args.checkpoint
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise SystemExit(f"generate is missing: {', '.join(missing)}")
        if args.num_gpus > 1:
            generate_videos_multi_gpu(
                dataset=args.dataset,
                variant=args.variant,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                output_root=args.output_root,
                vendor_root=args.vendor_root,
                cache_root=args.model_cache_root,
                wan_root=args.wan_root,
                wan_model_root=args.wan_model_root,
                device=args.device,
                overwrite=args.overwrite,
                num_gpus=args.num_gpus,
                config=args.config,
            )
        else:
            generate_videos(
                dataset=args.dataset,
                variant=args.variant,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                output_root=args.output_root,
                vendor_root=args.vendor_root,
                cache_root=args.model_cache_root,
                wan_root=args.wan_root,
                wan_model_root=args.wan_model_root,
                device=args.device,
                overwrite=args.overwrite,
                dummy_model=args.dummy_model,
                single_camera=args.single_camera,
            )
    elif args.mode == "score":
        required = {
            "dataset": args.dataset,
            "variant": args.variant,
            "manifest": args.manifest,
            "videos": args.videos,
            "output": args.output,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise SystemExit(f"score is missing: {', '.join(missing)}")
        if args.dataset == "w0-video-40":
            if args.num_gpus > 1:
                score_w0_multi_gpu(
                    variant=args.variant,
                    manifest=args.manifest,
                    videos=args.videos,
                    output=args.output,
                    checkpoint=args.checkpoint,
                    lpips_weights=args.lpips_weights,
                    device=args.device,
                    allow_missing_lpips=args.allow_missing_lpips,
                    num_gpus=args.num_gpus,
                    config=args.config,
                )
            else:
                score_w0(
                    variant=args.variant,
                    manifest=args.manifest,
                    videos=args.videos,
                    output=args.output,
                    checkpoint=args.checkpoint,
                    lpips_weights=args.lpips_weights,
                    device=args.device,
                    allow_missing_lpips=args.allow_missing_lpips,
                )
        else:
            if args.num_gpus > 1:
                raise SystemExit("multi-GPU score currently supports only --dataset=w0-video-40")
            score_vbench(
                variant=args.variant,
                manifest=args.manifest,
                videos=args.videos,
                output=args.output,
                vbench_root=args.vbench_root,
                cache_root=args.cache_root,
                device=args.device,
                allow_vbench_downloads=args.allow_vbench_downloads,
            )
    elif args.mode == "compare":
        required = {
            "vbench-fw-fz": args.vbench_fw_fz,
            "vbench-fw-v": args.vbench_fw_v,
            "w0-fw-fz": args.w0_fw_fz,
            "w0-fw-v": args.w0_fw_v,
            "output": args.output,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise SystemExit(f"compare is missing: {', '.join(missing)}")
        compare_metrics(
            vbench_fw_fz=args.vbench_fw_fz,
            vbench_fw_v=args.vbench_fw_v,
            w0_fw_fz=args.w0_fw_fz,
            w0_fw_v=args.w0_fw_v,
            output=args.output,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
