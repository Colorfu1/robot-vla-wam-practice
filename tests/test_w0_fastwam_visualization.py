from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_practice.experiments.literature.w0_fastwam_visualization import (
    discover_scenes,
    select_scene_video,
)


SUITES = ("libero_goal", "libero_10", "libero_object", "libero_spatial")


def _make_fake_eval(tmp_path: Path) -> Path:
    eval_dir = tmp_path / "eval"
    for suite in SUITES:
        for task_id in range(10):
            worker = eval_dir / "workers" / f"{suite}_task{task_id}_trial0000_count0050"
            suite_dir = worker / suite
            videos_dir = suite_dir / "videos"
            videos_dir.mkdir(parents=True)
            failed_trial = 3 if task_id == 0 else None
            successes = 49 if failed_trial is not None else 50
            for trial in range(50):
                success = trial != failed_trial
                video_name = (
                    f"2026_01_01-00_00_00--episode=task{task_id}_trial{trial}"
                    f"--success={success}--task=task_{task_id}.mp4"
                )
                (videos_dir / video_name).write_bytes(b"")
            (suite_dir / f"gpu0_task{task_id}_results.json").write_text(
                json.dumps(
                    {
                        "task_suite": suite,
                        "task_id": task_id,
                        "task_description": f"task {task_id}",
                        "successes": successes,
                        "total_episodes": 50,
                    }
                ),
                encoding="utf-8",
            )
    return eval_dir


def test_discover_scenes_validates_and_orders_all_40_tasks(tmp_path: Path) -> None:
    scenes = discover_scenes(_make_fake_eval(tmp_path))

    assert len(scenes) == 40
    assert len({scene.suite for scene in scenes}) == 4
    assert [(scene.suite, scene.task_id) for scene in scenes[:3]] == [
        ("libero_10", 0),
        ("libero_10", 1),
        ("libero_10", 2),
    ]
    assert scenes[0].total_episodes == 50
    assert scenes[0].failure_episodes == (3,)


def test_first_failure_selection_falls_back_to_first_success(tmp_path: Path) -> None:
    scenes = discover_scenes(_make_fake_eval(tmp_path))
    failed_scene = next(scene for scene in scenes if scene.task_id == 0)
    clean_scene = next(
        scene for scene in scenes if scene.suite == "libero_10" and scene.task_id == 1
    )

    failed = select_scene_video(failed_scene, policy="first-failure")
    clean = select_scene_video(clean_scene, policy="first-failure")

    assert (failed.video.trial_index, failed.video.success, failed.fallback_used) == (
        3,
        False,
        False,
    )
    assert (clean.video.trial_index, clean.video.success, clean.fallback_used) == (
        0,
        True,
        True,
    )


def test_trial_selection_requires_an_existing_trial(tmp_path: Path) -> None:
    scene = discover_scenes(_make_fake_eval(tmp_path))[0]

    selected = select_scene_video(scene, policy="trial", trial_index=7)
    assert selected.video.trial_index == 7

    with pytest.raises(ValueError, match="no trial 99"):
        select_scene_video(scene, policy="trial", trial_index=99)

