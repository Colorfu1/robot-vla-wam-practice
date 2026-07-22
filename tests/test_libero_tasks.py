from __future__ import annotations

import logging

import pytest

from robot_practice.evaluation.libero_tasks import (
    SUPPORTED_SUITES,
    LiberoTaskIdentity,
    log_task_identity,
    normalize_task_language,
    resolve_libero_task,
)


def test_resolve_current_a_sanity_task() -> None:
    task = resolve_libero_task("libero_object", 8)
    assert task == LiberoTaskIdentity(
        suite="libero_object",
        task_id=8,
        task_name="pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        language="pick up the chocolate pudding and place it in the basket",
        problem_folder="libero_object",
        bddl_file="pick_up_the_chocolate_pudding_and_place_it_in_the_basket.bddl",
        source_relative_path=(
            "libero_object/"
            "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5"
        ),
        episode_length=280,
    )
    assert task.cache_name == "task_008"


def test_resolve_all_130_supported_tasks() -> None:
    expected_counts = {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    resolved = [
        resolve_libero_task(suite, task_id)
        for suite, count in expected_counts.items()
        for task_id in range(count)
    ]
    assert len(resolved) == 130
    assert tuple(expected_counts) == SUPPORTED_SUITES
    assert len({(task.suite, task.task_id) for task in resolved}) == 130
    assert all(task.source_relative_path.endswith("_demo.hdf5") for task in resolved)


@pytest.mark.parametrize("task_id", [-1, 10])
def test_object_task_id_bounds(task_id: int) -> None:
    with pytest.raises(ValueError, match="valid range 0..9"):
        resolve_libero_task("libero_object", task_id)


def test_libero_90_upper_bound() -> None:
    with pytest.raises(ValueError, match="valid range 0..89"):
        resolve_libero_task("libero_90", 90)


def test_aggregate_and_unknown_suites_are_actionable() -> None:
    with pytest.raises(ValueError, match="libero_10.*libero_90"):
        resolve_libero_task("libero_100", 0)
    with pytest.raises(ValueError, match="choose one of"):
        resolve_libero_task("unknown", 0)


def test_language_normalization_does_not_rewrite_words() -> None:
    assert normalize_task_language("  pick   up\n the bowl ") == "pick up the bowl"


def test_task_log_contains_required_identity(caplog: pytest.LogCaptureFixture) -> None:
    task = resolve_libero_task("libero_goal", 0)
    with caplog.at_level(logging.INFO):
        log_task_identity(task)
    text = caplog.text
    assert "[Task] suite: libero_goal" in text
    assert "[Task] task_id: 0" in text
    assert f"[Task] name: {task.task_name}" in text
    assert f"[Task] language: {task.language}" in text
