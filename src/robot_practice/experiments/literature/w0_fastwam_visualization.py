"""Build a task-selectable Rerun archive from the W0 Fast-WAM LIBERO videos.

The upstream Fast-WAM evaluator stores rollout videos and per-task result JSON, but
does not emit the simulator state/action trace used by the repository's live LIBERO
visualizer.  This module therefore keeps the semantics honest: one Rerun Recording
per LIBERO task, with the selected MP4 as an embedded ``AssetVideo`` and a markdown
summary containing the task and aggregate result.

The recordings are merged into one archive.  Rerun's Recording selector can then be
used to switch between the 40 task-level scenes from the left sidebar.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from robot_practice.evaluation.common import (
    launch_rerun_viewer,
    practice_root,
    write_json_atomic,
)
from robot_practice.evaluation.rerun_visualization import stage_merged_recordings
from robot_practice.rerun_launcher import OFFLINE_APPLICATION_ID

DEFAULT_EVAL_DIR = Path("outputs/literature/w0_fastwam/eval_fwv_21700_4gpu")
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_DIR / "rerun_scenes"
DEFAULT_OUTPUT_NAME = "w0_fastwam_40_scenes.rrd"
VIDEO_TIMELINE = "video_time"
VIDEO_FPS = 24.0

_VIDEO_RE = re.compile(
    r"^.*--episode=task(?P<task_id>\d+)_trial(?P<trial>\d+)"
    r"--success=(?P<success>True|False)--task=.*\.mp4$"
)


@dataclass(frozen=True)
class W0VideoTrial:
    path: Path
    trial_index: int
    success: bool


@dataclass(frozen=True)
class W0Scene:
    suite: str
    task_id: int
    task_description: str
    successes: int
    total_episodes: int
    result_path: Path
    videos: tuple[W0VideoTrial, ...]

    @property
    def success_rate(self) -> float:
        if self.total_episodes <= 0:
            raise ValueError(f"{self.suite} task {self.task_id} has no episodes")
        return self.successes / self.total_episodes

    @property
    def failure_episodes(self) -> tuple[int, ...]:
        return tuple(video.trial_index for video in self.videos if not video.success)


@dataclass(frozen=True)
class SelectedSceneVideo:
    scene: W0Scene
    video: W0VideoTrial
    selection_policy: str
    fallback_used: bool


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _portable(path: str | Path) -> str:
    value = Path(path).expanduser().resolve()
    root = practice_root().resolve()
    try:
        return value.relative_to(root).as_posix()
    except ValueError:
        return value.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {_portable(path)}")
    return value


def _parse_video(path: Path, *, expected_suite: str, expected_task_id: int) -> W0VideoTrial:
    match = _VIDEO_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected W0 rollout video filename: {_portable(path)}")
    task_id = int(match.group("task_id"))
    if task_id != expected_task_id:
        raise ValueError(
            f"Video task id mismatch for {expected_suite} task {expected_task_id}: "
            f"{_portable(path)} declares task {task_id}"
        )
    return W0VideoTrial(
        path=path,
        trial_index=int(match.group("trial")),
        success=match.group("success") == "True",
    )


def _result_files(eval_dir: Path) -> Iterable[Path]:
    return sorted(eval_dir.glob("workers/*/*/*_results.json"))


def discover_scenes(eval_dir: str | Path) -> tuple[W0Scene, ...]:
    """Discover the 40 task-level scenes from the completed W0 evaluator output."""

    root = _rooted(eval_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"W0 evaluation directory does not exist: {_portable(root)}")

    scenes: list[W0Scene] = []
    seen: set[tuple[str, int]] = set()
    result_paths = tuple(_result_files(root))
    if not result_paths:
        raise FileNotFoundError(f"No W0 task result JSON found under {_portable(root / 'workers')}")

    for result_path in result_paths:
        payload = _read_json(result_path)
        suite = str(payload.get("task_suite", ""))
        task_id_value = payload.get("task_id")
        if not suite or isinstance(task_id_value, bool) or not isinstance(task_id_value, int):
            raise ValueError(f"Malformed task identity in {_portable(result_path)}")
        task_id = int(task_id_value)
        key = (suite, task_id)
        if key in seen:
            raise ValueError(f"Duplicate W0 task result for {suite} task {task_id}")
        seen.add(key)

        videos_dir = result_path.parent / "videos"
        videos = tuple(
            sorted(
                (
                    _parse_video(path, expected_suite=suite, expected_task_id=task_id)
                    for path in videos_dir.glob("*.mp4")
                ),
                key=lambda item: item.trial_index,
            )
        )
        if not videos:
            raise FileNotFoundError(f"No rollout videos found under {_portable(videos_dir)}")
        trial_ids = tuple(video.trial_index for video in videos)
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError(f"Duplicate video trial index for {suite} task {task_id}")

        successes = payload.get("successes")
        total_episodes = payload.get("total_episodes")
        if (
            isinstance(successes, bool)
            or not isinstance(successes, int)
            or isinstance(total_episodes, bool)
            or not isinstance(total_episodes, int)
            or successes < 0
            or total_episodes <= 0
            or successes > total_episodes
        ):
            raise ValueError(f"Malformed task metrics in {_portable(result_path)}")
        if len(videos) != total_episodes:
            raise ValueError(
                f"Video count mismatch for {suite} task {task_id}: "
                f"expected {total_episodes}, got {len(videos)}"
            )

        success_from_videos = sum(video.success for video in videos)
        if success_from_videos != successes:
            raise ValueError(
                f"Video/result success mismatch for {suite} task {task_id}: "
                f"JSON={successes}, videos={success_from_videos}"
            )
        scenes.append(
            W0Scene(
                suite=suite,
                task_id=task_id,
                task_description=str(payload.get("task_description", "")).strip(),
                successes=successes,
                total_episodes=total_episodes,
                result_path=result_path,
                videos=videos,
            )
        )

    scenes.sort(key=lambda scene: (scene.suite, scene.task_id))
    if len(scenes) != 40:
        raise ValueError(f"Expected 40 W0 LIBERO scenes, discovered {len(scenes)}")
    return tuple(scenes)


def select_scene_video(
    scene: W0Scene,
    *,
    policy: str = "first-failure",
    trial_index: int | None = None,
) -> SelectedSceneVideo:
    """Select one representative MP4 while retaining all trials in the manifest."""

    videos = scene.videos
    if policy == "trial":
        if trial_index is None:
            raise ValueError("trial_index is required when policy='trial'")
        selected = next((video for video in videos if video.trial_index == trial_index), None)
        if selected is None:
            available = ", ".join(str(video.trial_index) for video in videos)
            raise ValueError(
                f"{scene.suite} task {scene.task_id} has no trial {trial_index}; "
                f"available trials: {available}"
            )
        return SelectedSceneVideo(scene, selected, policy, False)
    if trial_index is not None:
        raise ValueError("trial_index is only valid with policy='trial'")
    if policy == "first-failure":
        selected = next((video for video in videos if not video.success), None)
        if selected is not None:
            return SelectedSceneVideo(scene, selected, policy, False)
        policy = "first-success"
        selected = next((video for video in videos if video.success), None)
        fallback_used = True
    elif policy == "first-success":
        selected = next((video for video in videos if video.success), None)
        fallback_used = False
    else:
        raise ValueError(f"Unknown scene video selection policy: {policy!r}")

    if selected is None:
        raise ValueError(f"{scene.suite} task {scene.task_id} has no video matching {policy}")
    effective_policy = "first-failure" if fallback_used else policy
    return SelectedSceneVideo(scene, selected, effective_policy, fallback_used)


def _scene_recording_name(selected: SelectedSceneVideo) -> str:
    scene = selected.scene
    outcome = "success" if selected.video.success else "failure"
    return (
        f"{scene.suite}/scene_{scene.task_id:02d} | {scene.task_description} | "
        f"SR={scene.successes}/{scene.total_episodes} "
        f"({scene.success_rate * 100.0:.1f}%) | "
        f"trial={selected.video.trial_index:02d} {outcome}"
    )


def _scene_metrics_markdown(selected: SelectedSceneVideo) -> str:
    scene = selected.scene
    video = selected.video
    failures = ", ".join(str(index) for index in scene.failure_episodes) or "none"
    fallback = "yes (no failed rollout exists for this scene)" if selected.fallback_used else "no"
    return "\n".join(
        (
            f"# {scene.suite} · scene {scene.task_id:02d}",
            "",
            f"- **Task:** {scene.task_description}",
            f"- **Success rate:** {scene.successes}/{scene.total_episodes} "
            f"({scene.success_rate * 100.0:.1f}%)",
            f"- **Selected trial:** {video.trial_index} "
            f"({'success' if video.success else 'failure'})",
            f"- **Selection fallback:** {fallback}",
            f"- **All failed trials:** {failures}",
            f"- **Source video:** `{_portable(video.path)}`",
            "",
            "This Recording contains the selected MP4 only. The W0 evaluator did not persist "
            "simulator state or actions, so no 3D/action reconstruction is claimed here.",
        )
    )


def _scene_semantics_markdown() -> str:
    return "\n".join(
        (
            "# W0 Fast-WAM video visualization",
            "",
            "- The video is the original Fast-WAM evaluator MP4 embedded as a Rerun AssetVideo.",
            "- The left Recording selector switches between the 40 LIBERO task scenes.",
            "- The default representative is the first failed trial; scenes with no failures "
            "use the first successful trial.",
            "- The timeline is the video timeline at 24 FPS.",
            "- This archive contains no inferred robot pose, action, reward, or simulator state.",
        )
    )


def _build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(
                origin="rollout/video",
                contents=["rollout/video"],
                name="Selected rollout video",
            ),
            rrb.Tabs(
                rrb.TextDocumentView(
                    origin="summary",
                    contents=["summary/metrics"],
                    name="Scene metrics",
                ),
                rrb.TextDocumentView(
                    origin="documentation",
                    contents=["documentation/semantics"],
                    name="Visualization semantics",
                ),
                active_tab=0,
                name="Scene information",
            ),
            row_shares=[0.78, 0.22],
        ),
        rrb.BlueprintPanel(expanded=True),
        rrb.TimePanel(timeline=VIDEO_TIMELINE, fps=VIDEO_FPS, expanded=True),
        collapse_panels=False,
    )


def _write_scene_recording(
    selected: SelectedSceneVideo,
    output_path: Path,
) -> str:
    stream = rr.RecordingStream(
        OFFLINE_APPLICATION_ID,
        recording_id=str(uuid.uuid4()),
    )
    try:
        stream.save(output_path)
        stream.send_recording_name(_scene_recording_name(selected))
        stream.log(
            "summary/metrics",
            rr.TextDocument(_scene_metrics_markdown(selected), media_type="text/markdown"),
            static=True,
        )
        stream.log(
            "documentation/semantics",
            rr.TextDocument(_scene_semantics_markdown(), media_type="text/markdown"),
            static=True,
        )
        video_asset = rr.AssetVideo(path=selected.video.path)
        stream.log("rollout/video", video_asset, static=True)
        frame_timestamps_ns = video_asset.read_frame_timestamps_nanos()
        if len(frame_timestamps_ns) == 0:
            raise ValueError(f"Video has no frames: {_portable(selected.video.path)}")
        stream.send_columns(
            "rollout/video",
            indexes=[
                rr.TimeColumn(
                    VIDEO_TIMELINE,
                    duration=1e-9 * np.asarray(frame_timestamps_ns, dtype=np.int64),
                )
            ],
            columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        )
        rr.send_blueprint(
            _build_blueprint(),
            make_active=False,
            make_default=True,
            recording=stream,
        )
        stream.flush()
        return _scene_recording_name(selected)
    finally:
        stream.disconnect()


def _verify_archive(output_path: Path, expected_count: int) -> None:
    verify_cli = shutil.which("rerun")
    if verify_cli is None:
        raise RuntimeError("The rerun CLI is required to verify the merged archive")
    completed = subprocess.run(
        [verify_cli, "rrd", "verify", str(output_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Rerun archive verification failed: {completed.stderr.strip()}")
    import rerun.dataframe as rdf

    archive = rdf.load_archive(output_path)
    if archive.num_recordings() != expected_count:
        raise RuntimeError(
            f"Merged Rerun archive has {archive.num_recordings()} recordings; "
            f"expected {expected_count}"
        )


def build_w0_scene_archive(
    *,
    eval_dir: str | Path = DEFAULT_EVAL_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_name: str = DEFAULT_OUTPUT_NAME,
    selection_policy: str = "first-failure",
    trial_index: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create and verify the 40-Recording W0 scene archive."""

    eval_root = _rooted(eval_dir)
    output_root = _rooted(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / output_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {_portable(output_path)}; pass --overwrite to replace it"
        )

    scenes = discover_scenes(eval_root)
    selected = tuple(
        select_scene_video(scene, policy=selection_policy, trial_index=trial_index)
        for scene in scenes
    )
    with tempfile.TemporaryDirectory(prefix=".w0_scene_recordings-", dir=output_root) as temp_dir:
        temporary_root = Path(temp_dir)
        recording_paths: list[Path] = []
        recording_names: list[str] = []
        for index, item in enumerate(selected):
            temporary_path = temporary_root / f"scene_{index:02d}.rrd"
            recording_names.append(_write_scene_recording(item, temporary_path))
            recording_paths.append(temporary_path)
            print(
                f"STATUS recordings={index + 1}/{len(selected)} "
                f"scene={item.scene.suite}:{item.scene.task_id} "
                f"trial={item.video.trial_index} success={item.video.success}",
                flush=True,
            )
        staging_path = stage_merged_recordings(recording_paths, output_path)
        staging_path.replace(output_path)

    _verify_archive(output_path, len(selected))
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "mode": "w0_fastwam_video",
        "application_id": OFFLINE_APPLICATION_ID,
        "source_eval_dir": _portable(eval_root),
        "output_archive": _portable(output_path),
        "num_scenes": len(selected),
        "num_eval_suites": len({item.scene.suite for item in selected}),
        "selection_policy": selection_policy,
        "trial_index": trial_index,
        "recordings": [
            {
                "name": name,
                "suite": item.scene.suite,
                "task_id": item.scene.task_id,
                "task_description": item.scene.task_description,
                "selected_trial": item.video.trial_index,
                "selected_success": item.video.success,
                "successes": item.scene.successes,
                "total_episodes": item.scene.total_episodes,
                "source_video": _portable(item.video.path),
                "all_trials": [
                    {
                        "trial": video.trial_index,
                        "success": video.success,
                        "source_video": _portable(video.path),
                    }
                    for video in item.scene.videos
                ],
            }
            for name, item in zip(recording_names, selected, strict=True)
        ],
    }
    write_json_atomic(output_root / "visualization_manifest.json", manifest)
    return {
        "status": "complete",
        "output": _portable(output_path),
        "manifest": _portable(output_root / "visualization_manifest.json"),
        "num_scenes": len(selected),
        "num_eval_suites": len({item.scene.suite for item in selected}),
        "selection_policy": selection_policy,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--selection-policy",
        choices=("first-failure", "first-success", "trial"),
        default="first-failure",
        help="Representative video per scene; all trials remain in the manifest.",
    )
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-open-viewer", action="store_false", dest="open_viewer")
    parser.set_defaults(open_viewer=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    result = build_w0_scene_archive(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        selection_policy=args.selection_policy,
        trial_index=args.trial_index,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    launch_rerun_viewer(_rooted(args.output_dir) / args.output_name, enabled=args.open_viewer)


if __name__ == "__main__":
    main()
