from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from robot_practice.experiments.smolvla_libero.data_preparation import (
    RNG_CONTRACT,
    ActionEpisode,
    DemoMapping,
    EpisodeCatalogEntry,
    RawDemo,
    action_fingerprint,
    audit_paraphrases,
    build_background_split,
    build_target_task_split,
    filter_raw_noop_actions,
    load_raw_demos,
    match_action_episodes,
    normalize_instruction,
    scaffold_paraphrases,
    sha256_file,
    verify_paraphrase_binding,
)


def test_raw_action_noop_v1_keeps_motion_and_gripper_transitions() -> None:
    actions = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, -1],
            [0.2, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, -0.2, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    filtered = filter_raw_noop_actions(actions)

    assert filtered.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(filtered, actions[[1, 3, 5]].astype(np.float32))


def test_action_fingerprint_is_float32_and_endian_stable() -> None:
    actions = np.arange(21, dtype=np.float64).reshape(3, 7) / 10

    assert action_fingerprint(actions) == action_fingerprint(actions.astype(">f4"))


def _action_episode(episode_index: int, actions: np.ndarray) -> ActionEpisode:
    return ActionEpisode(
        episode_index=episode_index,
        task_index=10,
        actions=np.asarray(actions, dtype=np.float32),
        states=np.zeros((len(actions), 8), dtype=np.float32),
    )


def _raw_demo(name: str, actions: np.ndarray) -> RawDemo:
    return RawDemo(
        name=name,
        actions=np.asarray(actions, dtype=np.float32),
    )


def test_action_episode_matching_requires_one_unique_raw_demo() -> None:
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    episode = _action_episode(42, actions)
    matching = _raw_demo("demo_3", actions)
    unused = _raw_demo("demo_4", actions + 1)

    matches, unmatched = match_action_episodes([episode], [matching, unused])

    assert matches[42][0].name == "demo_3"
    assert matches[42][1] == 0
    assert unmatched == ["demo_4"]


def test_action_episode_matching_rejects_missing_or_ambiguous_matches() -> None:
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    episode = _action_episode(42, actions)

    with pytest.raises(ValueError, match="0 raw action matches"):
        match_action_episodes([episode], [_raw_demo("demo_0", actions + 1)])

    with pytest.raises(ValueError, match="2 raw action matches"):
        match_action_episodes(
            [episode],
            [_raw_demo("demo_0", actions), _raw_demo("demo_1", actions)],
        )


def test_action_episode_matching_rejects_reused_raw_demo() -> None:
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)

    with pytest.raises(ValueError, match="maps to multiple LeRobot episodes"):
        match_action_episodes(
            [_action_episode(41, actions), _action_episode(42, actions)],
            [_raw_demo("demo_0", actions)],
        )


def test_load_raw_demos_applies_noop_contract(tmp_path: Path) -> None:
    path = tmp_path / "task_demo.hdf5"
    with h5py.File(path, "w") as hdf5:
        demo = hdf5.create_group("data/demo_0")
        demo.create_dataset(
            "actions",
            data=np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, -1],
                    [0.1, 0, 0, 0, 0, 0, -1],
                ],
                dtype=np.float64,
            ),
        )

    demos = load_raw_demos(path)

    assert len(demos) == 1
    assert demos[0].name == "demo_0"
    assert demos[0].actions.shape == (1, 7)


def test_background_split_excludes_targets_and_is_deterministic() -> None:
    catalog = {}
    episode_index = 0
    for task_index in range(40):
        for local_index in range(10):
            catalog[episode_index] = EpisodeCatalogEntry(
                episode_index=episode_index,
                task_index=task_index,
                length=100 + local_index,
            )
            episode_index += 1
    task_names = {task_index: f"task {task_index}" for task_index in range(40)}

    first = build_background_split(catalog, task_names, seed=1000)
    second = build_background_split(catalog, task_names, seed=1000)

    assert first == second
    assert first["rng_contract"] == {
        "name": RNG_CONTRACT,
        "bit_generator": "PCG64",
        "seed": 1000,
        "namespaces": {"background_task": 0},
    }
    assert first["aggregate"]["num_tasks"] == 37
    assert not {"10", "17", "18"} & set(first["tasks"])
    for task in first["tasks"].values():
        assert len(task["train_episode_indices"]) == 9
        assert len(task["validation_episode_indices"]) == 1
        assert not set(task["train_episode_indices"]) & set(task["validation_episode_indices"])
        assert task["train_episode_indices"] == sorted(task["train_episode_indices"])
        assert task["validation_episode_indices"] == sorted(task["validation_episode_indices"])

    task_zero_ids = list(range(10))
    expected = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([1000, 0, 0]))
    ).permutation(task_zero_ids)
    assert first["tasks"]["0"]["validation_episode_indices"] == sorted(
        int(value) for value in expected[:1]
    )


def _mapping(episode_index: int, *, raw_demo: str | None = None) -> DemoMapping:
    return DemoMapping(
        episode_index=episode_index,
        raw_demo=raw_demo or f"demo_{episode_index}",
        action_max_abs_error=0,
    )


def test_target_split_separates_demo_and_closed_loop_namespaces() -> None:
    mappings = [_mapping(episode) for episode in range(48)]

    split = build_target_task_split(
        mappings,
        task_id=1,
        seed=1000,
        official_state_count=50,
        offline_validation_count=10,
        closed_loop_validation_count=20,
        closed_loop_test_count=30,
    )

    train_records = split["demonstration_split"]["train_demonstrations"]
    offline_validation_records = split["demonstration_split"]["offline_validation_demonstrations"]
    train_episodes = {record["episode_index"] for record in train_records}
    offline_validation_episodes = {record["episode_index"] for record in offline_validation_records}
    assert len(train_records) == 38
    assert len(offline_validation_records) == 10
    assert not train_episodes & offline_validation_episodes
    assert train_episodes | offline_validation_episodes == set(range(48))
    assert [record["episode_index"] for record in train_records] == sorted(train_episodes)
    assert [record["episode_index"] for record in offline_validation_records] == sorted(
        offline_validation_episodes
    )

    expected_demo_order = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([1000, 1, 1]))
    ).permutation(np.arange(48))
    assert offline_validation_episodes == {int(value) for value in expected_demo_order[:10]}

    closed_loop_validation = set(split["closed_loop_split"]["validation_init_state_ids"])
    closed_loop_test = set(split["closed_loop_split"]["test_init_state_ids"])
    assert len(closed_loop_validation) == 20
    assert len(closed_loop_test) == 30
    assert not closed_loop_validation & closed_loop_test
    assert closed_loop_validation | closed_loop_test == set(range(50))
    expected_state_order = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([1000, 2, 1]))
    ).permutation(np.arange(50))
    assert closed_loop_validation == {int(value) for value in expected_state_order[:20]}


def test_target_split_rejects_too_few_mapped_demonstrations() -> None:
    with pytest.raises(ValueError, match="requires more than 10"):
        build_target_task_split(
            [_mapping(episode) for episode in range(10)],
            task_id=1,
            seed=1000,
            official_state_count=50,
            offline_validation_count=10,
            closed_loop_validation_count=20,
            closed_loop_test_count=30,
        )


def test_target_split_rejects_duplicate_episode_or_raw_demo_identity() -> None:
    with pytest.raises(ValueError, match="duplicate episode_index"):
        build_target_task_split(
            [_mapping(1), _mapping(1, raw_demo="demo_other")]
            + [_mapping(episode) for episode in range(2, 12)],
            task_id=1,
            seed=1000,
            official_state_count=50,
            offline_validation_count=10,
            closed_loop_validation_count=20,
            closed_loop_test_count=30,
        )

    with pytest.raises(ValueError, match="duplicate raw_demo"):
        build_target_task_split(
            [
                _mapping(episode, raw_demo="demo_shared" if episode < 2 else None)
                for episode in range(12)
            ],
            task_id=1,
            seed=1000,
            official_state_count=50,
            offline_validation_count=10,
            closed_loop_validation_count=20,
            closed_loop_test_count=30,
        )


def _approved_paraphrases() -> dict:
    return {
        "schema_version": 1,
        "suite": "libero_goal",
        "canonical_probability": 0.25,
        "tasks": {
            "1": {
                "canonical": "put the bowl on the stove",
                "train_paraphrases": [
                    f"place the bowl onto stove variant {index}" for index in range(8)
                ],
                "held_out_paraphrases": [
                    f"move the bowl to stove held out {index}" for index in range(5)
                ],
                "semantic_review": {
                    "status": "approved",
                    "reviewed_at": "2026-07-30",
                    "checks": {
                        "object_preserved": True,
                        "target_preserved": True,
                        "completion_condition_preserved": True,
                        "train_test_separated": True,
                    },
                },
            }
        },
    }


def test_paraphrase_audit_requires_review_and_rejects_normalized_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paraphrases.json"
    payload = _approved_paraphrases()
    path.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_paraphrases(path, suite="libero_goal", task_ids=(1,))

    assert audit["status"] == "passed"
    assert audit["tasks"]["1"]["train_paraphrase_count"] == 8

    payload["tasks"]["1"]["semantic_review"]["status"] = "pending"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        audit_paraphrases(path, suite="libero_goal", task_ids=(1,))

    payload = _approved_paraphrases()
    payload["tasks"]["1"]["train_paraphrases"][0] = " PUT THE BOWL ON THE STOVE. "
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate normalized"):
        audit_paraphrases(path, suite="libero_goal", task_ids=(1,))


def test_target_split_binds_exact_reviewed_paraphrases(tmp_path: Path) -> None:
    paraphrases = tmp_path / "paraphrases.json"
    target_split = tmp_path / "target_split.json"
    payload = _approved_paraphrases()
    paraphrases.write_text(json.dumps(payload), encoding="utf-8")
    target_split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "libero_goal",
                "paraphrases_sha256": sha256_file(paraphrases),
                "tasks": {"1": {}},
            }
        ),
        encoding="utf-8",
    )

    audit = verify_paraphrase_binding(
        target_split=target_split,
        paraphrases=paraphrases,
    )

    assert audit["status"] == "passed"

    payload["tasks"]["1"]["train_paraphrases"][0] += " revised"
    paraphrases.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum does not match"):
        verify_paraphrase_binding(
            target_split=target_split,
            paraphrases=paraphrases,
        )


def test_scaffold_paraphrases_is_pending_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "paraphrases.json"

    payload = scaffold_paraphrases(
        suite="libero_goal",
        task_ids=(1, 4, 8),
        output_path=path,
    )

    assert payload["tasks"]["1"]["semantic_review"]["status"] == "pending"
    assert payload["tasks"]["4"]["train_paraphrases"] == []
    with pytest.raises(FileExistsError):
        scaffold_paraphrases(
            suite="libero_goal",
            task_ids=(1, 4, 8),
            output_path=path,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" Put   the bowl on the stove. ", "put the bowl on the stove"),
        ("PUT THE BOWL ON THE STOVE!", "put the bowl on the stove"),
    ],
)
def test_instruction_normalization(value: str, expected: str) -> None:
    assert normalize_instruction(value) == expected
