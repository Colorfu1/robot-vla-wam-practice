from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import rerun.dataframe as rdf

from robot_practice.experiments.pusht_diffusion import (
    development_recordings,
    rerun_visualization,
)
from robot_practice.experiments.pusht_diffusion.development_recordings import (
    CATEGORY_ORDER,
    DEVELOPMENT_ENVIRONMENT_SEEDS,
    DEVELOPMENT_EXPERIMENT_ID,
    MODEL_APPLICATION_IDS,
    TaskRecordingArtifact,
    TaskRecordingMetadata,
    load_comparison_task_metadata,
    task_metadata_from_result,
    validate_paired_recording_directories,
    verify_task_recording,
    write_rollout_task_recording,
)
from robot_practice.experiments.pusht_diffusion.rerun_visualization import (
    DEVELOPMENT_ROLLOUT_ROOT,
)
from robot_practice.experiments.pusht_diffusion.rollout import RolloutResult


def _rollout(model: str, seed: int, success: bool) -> RolloutResult:
    offset = 0.0 if model == "bc_chunk" else 0.25
    agent_path = np.asarray(
        [[seed + offset, 1.0], [seed + offset + 1.0, 2.0]],
        dtype=np.float32,
    )
    block_path = np.asarray(
        [[3.0, seed + offset, 0.0], [4.0, seed + offset + 1.0, 0.1]],
        dtype=np.float32,
    )
    action_targets = np.asarray([[5.0 + offset, 6.0]], dtype=np.float32)
    rewards = np.asarray([1.0 if success else 0.25], dtype=np.float32)
    image_value = (seed + (0 if model == "bc_chunk" else 50)) % 254
    frames = (
        np.full((8, 8, 3), image_value, dtype=np.uint8),
        np.full((8, 8, 3), image_value + 1, dtype=np.uint8),
    )
    return RolloutResult(
        model=model,
        environment_seed=seed,
        policy_seed=0,
        success=success,
        sum_reward=float(rewards.sum()),
        max_reward=float(rewards.max()),
        final_reward=float(rewards[-1]),
        episode_length=1,
        agent_path=agent_path,
        block_path=block_path,
        action_targets=action_targets,
        rewards=rewards,
        frames=frames,
        replans=(),
        condition_id=DEVELOPMENT_EXPERIMENT_ID,
        scheduler="DDPM" if model == "diffusion" else None,
        num_inference_steps=100 if model == "diffusion" else None,
    )


def _metadata(result: RolloutResult) -> TaskRecordingMetadata:
    return task_metadata_from_result(
        result,
        checkpoint_step=150000 if result.model == "bc_chunk" else 175000,
        checkpoint_id=("a" if result.model == "bc_chunk" else "b") * 64,
    )


def _category(*, bc_success: bool, diffusion_success: bool) -> str:
    if bc_success and not diffusion_success:
        return "bc_pass_diffusion_fail"
    if diffusion_success and not bc_success:
        return "diffusion_pass_bc_fail"
    if not bc_success and not diffusion_success:
        return "both_fail"
    return "both_pass"


def _comparison_payload() -> dict:
    categories = {name: [] for name in CATEGORY_ORDER}
    cases = []
    for seed in DEVELOPMENT_ENVIRONMENT_SEEDS:
        bc = _rollout("bc_chunk", seed, success=seed % 2 == 0)
        diffusion = _rollout("diffusion", seed, success=seed % 3 == 0)
        category = _category(
            bc_success=bc.success,
            diffusion_success=diffusion.success,
        )
        categories[category].append(seed)
        cases.append(
            {
                "environment_seed": seed,
                "category": category,
                "initial_state": {
                    "agent_position": bc.agent_path[0].tolist(),
                    "block_pose": bc.block_path[0].tolist(),
                },
                "bc_chunk": {
                    "success": bc.success,
                    "max_reward": bc.max_reward,
                    "rerun_recording_path": f"recordings/bc_chunk/env_{seed:04d}.rrd",
                    "rerun_entity_path": DEVELOPMENT_ROLLOUT_ROOT,
                },
                "diffusion": {
                    "success": diffusion.success,
                    "max_reward": diffusion.max_reward,
                    "rerun_recording_path": f"recordings/diffusion/env_{seed:04d}.rrd",
                    "rerun_entity_path": DEVELOPMENT_ROLLOUT_ROOT,
                },
            }
        )
    return {
        "schema_version": 1,
        "experiment": DEVELOPMENT_EXPERIMENT_ID,
        "environment_seeds": list(DEVELOPMENT_ENVIRONMENT_SEEDS),
        "policy_seed": 0,
        "num_rollouts_per_policy": len(DEVELOPMENT_ENVIRONMENT_SEEDS),
        "num_rollouts": 2 * len(DEVELOPMENT_ENVIRONMENT_SEEDS),
        "summaries": {
            "bc_chunk": {
                "checkpoint_step": 150000,
                "checkpoint_id": "a" * 64,
            },
            "diffusion": {
                "checkpoint_step": 175000,
                "checkpoint_id": "b" * 64,
            },
        },
        "category_counts": {category: len(categories[category]) for category in CATEGORY_ORDER},
        "categories": categories,
        "cases": cases,
    }


def _application_component_paths(path: Path) -> set[str]:
    recording = rdf.load_archive(path).all_recordings()[0]
    return {
        str(descriptor.entity_path)
        for descriptor in recording.schema().component_columns()
        if str(descriptor.entity_path) != "/__properties"
        and not str(descriptor.entity_path).startswith("/task/")
    }


def test_development_blueprint_contains_only_simulator_and_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def constructor(name: str):
        def create(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            return object()

        return create

    for name in (
        "Blueprint",
        "Horizontal",
        "Vertical",
        "Spatial2DView",
        "TimeSeriesView",
        "TextDocumentView",
        "DataframeView",
    ):
        monkeypatch.setattr(
            rerun_visualization.rrb,
            name,
            constructor(name),
        )

    rerun_visualization.build_development_rollout_blueprint()

    assert [name for name, _args, _kwargs in calls] == [
        "Spatial2DView",
        "TimeSeriesView",
        "Horizontal",
        "Blueprint",
    ]
    kwargs_by_name = {name: kwargs for name, _args, kwargs in calls}
    assert kwargs_by_name["Horizontal"]["column_shares"] == [3.0, 1.0]
    assert kwargs_by_name["TimeSeriesView"]["contents"] == ["/rollout/reward"]
    assert all("env_" not in repr(kwargs) for _name, _args, kwargs in calls)


@pytest.mark.parametrize(
    ("sampling_protocol", "expected_title"),
    [
        ("initial_noise_only", "E1a — initial-noise-only"),
        ("full_sampling_seed", "E1b — full-sampling-seed"),
    ],
)
def test_fixed_observation_blueprint_has_synchronized_bc_and_diffusion_panels(
    monkeypatch: pytest.MonkeyPatch,
    sampling_protocol: str,
    expected_title: str,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def constructor(name: str):
        def create(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            return object()

        return create

    for name in (
        "Blueprint",
        "Horizontal",
        "Spatial2DView",
    ):
        monkeypatch.setattr(
            rerun_visualization.rrb,
            name,
            constructor(name),
        )

    rerun_visualization.build_fixed_observation_blueprint(sampling_protocol=sampling_protocol)

    assert [name for name, _args, _kwargs in calls] == [
        "Spatial2DView",
        "Spatial2DView",
        "Horizontal",
        "Blueprint",
    ]
    spatial_views = [kwargs for name, _args, kwargs in calls if name == "Spatial2DView"]
    assert spatial_views[0]["origin"] == "/e1/bc"
    assert spatial_views[0]["contents"] == ["/e1/bc/**"]
    assert spatial_views[1]["origin"] == "/e1/dp"
    assert spatial_views[1]["contents"] == ["/e1/dp/**"]
    assert spatial_views[1]["name"] == expected_title
    horizontal_kwargs = next(kwargs for name, _args, kwargs in calls if name == "Horizontal")
    assert horizontal_kwargs["column_shares"] == [1.0, 1.0]
    blueprint_kwargs = next(kwargs for name, _args, kwargs in calls if name == "Blueprint")
    assert blueprint_kwargs == {"auto_layout": False, "auto_views": False}


def test_fixed_observation_prediction_colors_are_unique() -> None:
    colors = (*rerun_visualization.BC_SAMPLE_COLORS, *rerun_visualization.SAMPLE_COLORS)

    assert len(colors) == 23
    assert len(set(colors)) == 23


def test_rollout_overlay_coordinates_are_scaled_to_rendered_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_path = np.asarray([[128.0, 128.0], [512.0, 512.0]])
    block_path = np.asarray([[256.0, 64.0, 0.0], [384.0, 256.0, 0.1]])
    action_targets = np.asarray([[400.0, 320.0]])
    rollout = RolloutResult(
        model="bc_chunk",
        environment_seed=1000,
        policy_seed=0,
        success=True,
        sum_reward=1.0,
        max_reward=1.0,
        final_reward=1.0,
        episode_length=1,
        agent_path=agent_path,
        block_path=block_path,
        action_targets=action_targets,
        rewards=np.asarray([1.0]),
        frames=(
            np.zeros((128, 256, 3), dtype=np.uint8),
            np.zeros((128, 256, 3), dtype=np.uint8),
        ),
        replans=(),
    )
    logged: dict[str, list[tuple[str, np.ndarray]]] = {}
    log_options: dict[str, list[dict[str, object]]] = {}

    class Recording:
        def set_time(self, _timeline: str, *, sequence: int) -> None:
            assert sequence >= 0

        def log(self, entity_path: str, value: object, **kwargs: object) -> None:
            log_options.setdefault(entity_path, []).append(kwargs)
            if isinstance(value, tuple):
                logged.setdefault(entity_path, []).append(value)

    def line_strips(points: list[np.ndarray], **_kwargs: object) -> tuple[str, np.ndarray]:
        return ("line", np.asarray(points[0]))

    def point_cloud(points: list[np.ndarray], **_kwargs: object) -> tuple[str, np.ndarray]:
        return ("points", np.asarray(points))

    monkeypatch.setattr(rerun_visualization.rr, "LineStrips2D", line_strips)
    monkeypatch.setattr(rerun_visualization.rr, "Points2D", point_cloud)

    next_frame = rerun_visualization.log_rollout_to_stream(
        Recording(),  # type: ignore[arg-type]
        rollout,
        global_frame_start=0,
        rollout_index=0,
        entity_root="/rollout",
    )

    assert next_frame == 2
    np.testing.assert_allclose(
        logged["/rollout/complete_agent_path"][0][1],
        [[64.0, 32.0], [256.0, 128.0]],
    )
    np.testing.assert_allclose(
        logged["/rollout/complete_block_path"][0][1],
        [[128.0, 16.0], [192.0, 64.0]],
    )
    np.testing.assert_allclose(
        logged["/rollout/complete_action_targets"][0][1],
        [[200.0, 80.0]],
    )
    np.testing.assert_allclose(
        [value[1][0] for value in logged["/rollout/current_agent"]],
        [[64.0, 32.0], [256.0, 128.0]],
    )
    np.testing.assert_allclose(
        [value[1][0] for value in logged["/rollout/current_block"]],
        [[128.0, 16.0], [192.0, 64.0]],
    )
    np.testing.assert_allclose(
        logged["/rollout/current_action_target"][0][1],
        [[200.0, 80.0]],
    )
    assert log_options["/rollout/complete_action_targets"] == [{"static": True}]
    assert log_options["/rollout/current_action_target"] == [{}]
    np.testing.assert_array_equal(rollout.agent_path, agent_path)
    np.testing.assert_array_equal(rollout.block_path, block_path)
    np.testing.assert_array_equal(rollout.action_targets, action_targets)


def test_e2_recording_blueprint_is_active_and_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rollout = _rollout("diffusion", 2000, success=True)
    destination = tmp_path / "env_2000.rrd"
    sent_blueprints: list[dict[str, object]] = []

    class Recording:
        def save(self, path: Path) -> None:
            path.write_bytes(b"rrd")

        def send_recording_name(self, _name: str) -> None:
            pass

        def flush(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

    recording = Recording()
    blueprint = object()
    monkeypatch.setattr(
        rerun_visualization.rr,
        "RecordingStream",
        lambda *_args, **_kwargs: recording,
    )
    monkeypatch.setattr(
        rerun_visualization,
        "build_development_rollout_blueprint",
        lambda: blueprint,
    )
    monkeypatch.setattr(
        rerun_visualization.rr,
        "send_blueprint",
        lambda value, **kwargs: sent_blueprints.append({"blueprint": value, **kwargs}),
    )
    monkeypatch.setattr(
        rerun_visualization,
        "log_rollout_to_stream",
        lambda *_args, **_kwargs: rollout.agent_path.shape[0],
    )

    rerun_visualization.record_single_rollout(
        output_path=destination,
        rollout=rollout,
        recording_id="e2b_dp_env_2000_seed_00",
        recording_name="E2b | DP | env 2000 | seed 00 | success",
    )

    assert sent_blueprints == [
        {
            "blueprint": blueprint,
            "make_active": True,
            "make_default": True,
            "recording": recording,
        }
    ]


def test_fixed_observation_overlays_use_the_same_image_coordinate_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logged: dict[str, list[tuple[str, np.ndarray]]] = {}
    candidate_times: list[int] = []
    sent_blueprints: list[tuple[object, bool, bool]] = []
    stream_options: list[dict[str, object]] = []

    class Recording:
        def set_time(self, _timeline: str, *, sequence: int) -> None:
            candidate_times.append(sequence)

        def log(self, entity_path: str, value: object, **_kwargs: object) -> None:
            if isinstance(value, tuple):
                logged.setdefault(entity_path, []).append(value)

        def send_blueprint(
            self,
            blueprint: object,
            *,
            make_active: bool,
            make_default: bool,
        ) -> None:
            sent_blueprints.append((blueprint, make_active, make_default))

        def disconnect(self) -> None:
            pass

    def line_strips(points: list[np.ndarray], **_kwargs: object) -> tuple[str, np.ndarray]:
        return ("line", np.asarray(points[0]))

    def point_cloud(points: object, **_kwargs: object) -> tuple[str, np.ndarray]:
        return ("points", np.asarray(points))

    def make_stream(*_args: object, **kwargs: object) -> Recording:
        stream_options.append(kwargs)
        return Recording()

    monkeypatch.setattr(rerun_visualization, "_stream", make_stream)
    monkeypatch.setattr(rerun_visualization.rr, "LineStrips2D", line_strips)
    monkeypatch.setattr(rerun_visualization.rr, "Points2D", point_cloud)

    candidate = {
        "candidate_id": "case_0",
        "candidate_ordinal": 0,
        "previous_image": np.zeros((128, 256, 3), dtype=np.uint8),
        "current_image": np.zeros((128, 256, 3), dtype=np.uint8),
        "agent_position": np.asarray([128.0, 128.0]),
        "bc_actions": np.asarray(
            [
                [[400.0, 320.0], [512.0, 512.0]],
                [[384.0, 256.0], [448.0, 384.0]],
                [[256.0, 256.0], [320.0, 320.0]],
            ]
        ),
        "diffusion_actions": np.asarray(
            [
                [[128.0, 256.0], [256.0, 384.0]],
                [[256.0, 128.0], [384.0, 256.0]],
            ]
        ),
    }
    rerun_visualization.record_fixed_observation_samples(
        output_path=tmp_path / "recording.rrd",
        candidates=[
            candidate,
            {**candidate, "candidate_id": "case_1", "candidate_ordinal": 1},
        ],
        sampling_protocol="initial_noise_only",
    )

    assert candidate_times == [0, 1]
    assert stream_options == [
        {
            "app_id": "robot_practice_pusht_e1a_initial_noise_only",
            "recording_id": "pusht_e1a_fixed_observation",
        }
    ]
    assert sent_blueprints == [(sent_blueprints[0][0], True, True)]
    assert all("case_0" not in path and "case_1" not in path for path in logged)
    root = "/e1"
    np.testing.assert_allclose(logged[f"{root}/bc/agent"][0][1], [[64.0, 32.0]])
    np.testing.assert_allclose(logged[f"{root}/dp/agent"][0][1], [[64.0, 32.0]])
    assert len(logged[f"{root}/bc/agent"]) == 2
    assert len(logged[f"{root}/dp/agent"]) == 2
    np.testing.assert_allclose(
        logged[f"{root}/bc/pred_00/path"][0][1],
        [[64.0, 32.0], [200.0, 80.0], [256.0, 128.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/bc/pred_00/steps"][0][1],
        [[200.0, 80.0], [256.0, 128.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/bc/pred_00/first_action"][0][1],
        [[200.0, 80.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/bc/pred_00/endpoint"][0][1],
        [[256.0, 128.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/dp/pred_00/path"][0][1],
        [[64.0, 32.0], [64.0, 64.0], [128.0, 96.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/dp/pred_00/steps"][0][1],
        [[64.0, 64.0], [128.0, 96.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/dp/pred_00/first_action"][0][1],
        [[64.0, 64.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/dp/pred_00/endpoint"][0][1],
        [[128.0, 96.0]],
    )
    np.testing.assert_allclose(
        logged[f"{root}/dp/pred_01/path"][0][1],
        [[64.0, 32.0], [128.0, 32.0], [192.0, 64.0]],
    )
    for model, sample_count in (("bc", 3), ("dp", 2)):
        for sample_index in range(sample_count):
            prediction_root = f"{root}/{model}/pred_{sample_index:02d}"
            for child in ("path", "steps", "first_action", "endpoint"):
                assert len(logged[f"{prediction_root}/{child}"]) == 2


@pytest.mark.skipif(shutil.which("rerun") is None, reason="Rerun CLI is unavailable")
def test_task_recording_uses_stable_identity_paths_and_default_blueprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _rollout("bc_chunk", 1000, success=True)
    metadata = _metadata(result)
    calls: list[dict[str, object]] = []
    original = development_recordings.rr.send_blueprint

    def capture_blueprint(blueprint: object, **kwargs: object) -> None:
        calls.append({"blueprint": blueprint, **kwargs})
        original(blueprint, **kwargs)

    monkeypatch.setattr(development_recordings.rr, "send_blueprint", capture_blueprint)
    path = write_rollout_task_recording(
        output_path=tmp_path / metadata.recording_filename,
        result=result,
        metadata=metadata,
        rollout_index=0,
    )

    assert metadata.recording_id == "env_1000"
    assert metadata.recording_name == "Task env=1000 | BCChunk | success"
    assert metadata.recording_relative_path == "recordings/bc_chunk/env_1000.rrd"
    assert metadata.rollout_root == "/rollout"
    assert verify_task_recording(path, metadata) == "env_1000"
    recording = rdf.load_archive(path).all_recordings()[0]
    assert str(recording.application_id()) == MODEL_APPLICATION_IDS["bc_chunk"]
    timeline = (
        recording.view(index="global_frame", contents="/rollout/**")
        .select()
        .read_all()
        .column("global_frame")
        .to_pylist()
    )
    assert timeline == [0, 1]
    descriptors = list(recording.schema().component_columns())
    for entity_path in (
        "/rollout/metadata",
        "/rollout/success",
        "/rollout/rollout_index",
    ):
        matching = [
            descriptor for descriptor in descriptors if str(descriptor.entity_path) == entity_path
        ]
        assert matching
        assert all(descriptor.is_static for descriptor in matching)
    reward_timeline = (
        recording.view(index="global_frame", contents="/rollout/reward")
        .select()
        .read_all()
        .column("global_frame")
        .to_pylist()
    )
    simulator_timeline = (
        recording.view(index="global_frame", contents="/rollout/simulator")
        .select()
        .read_all()
        .column("global_frame")
        .to_pylist()
    )
    assert reward_timeline == [1]
    assert simulator_timeline == [0, 1]
    assert len(calls) == 1
    assert calls[0]["make_active"] is False
    assert calls[0]["make_default"] is True


@pytest.mark.skipif(shutil.which("rerun") is None, reason="Rerun CLI is unavailable")
def test_different_tasks_have_identical_dynamic_entity_paths(tmp_path: Path) -> None:
    artifacts = {}
    for model, seed in (("bc_chunk", 1000), ("diffusion", 1001)):
        result = _rollout(model, seed, success=True)
        metadata = _metadata(result)
        path = write_rollout_task_recording(
            output_path=tmp_path / model / metadata.recording_filename,
            result=result,
            metadata=metadata,
            rollout_index=seed - 1000,
        )
        verify_task_recording(path, metadata)
        artifacts[model] = path

    bc_paths = _application_component_paths(artifacts["bc_chunk"])
    diffusion_paths = _application_component_paths(artifacts["diffusion"])
    assert bc_paths == diffusion_paths
    assert all(path.startswith("/rollout/") for path in bc_paths)
    assert not any("env_" in path for path in bc_paths)


def test_comparison_metadata_rejects_category_disagreement(tmp_path: Path) -> None:
    payload = _comparison_payload()
    payload["cases"][0]["category"] = "both_fail"
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="category is inconsistent"):
        load_comparison_task_metadata(path)


def test_comparison_metadata_rejects_wrong_recording_path(tmp_path: Path) -> None:
    payload = _comparison_payload()
    payload["cases"][0]["bc_chunk"]["rerun_recording_path"] = "recordings/bc_chunk/wrong.rrd"
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rerun_recording_path"):
        load_comparison_task_metadata(path)


@pytest.mark.skipif(shutil.which("rerun") is None, reason="Rerun CLI is unavailable")
def test_paired_directories_require_exact_matching_file_sets(tmp_path: Path) -> None:
    root = tmp_path / "recordings"
    artifacts_by_model: dict[str, list[TaskRecordingArtifact]] = {
        "bc_chunk": [],
        "diffusion": [],
    }
    expected_by_model: dict[str, list[TaskRecordingMetadata]] = {
        "bc_chunk": [],
        "diffusion": [],
    }
    for model in ("bc_chunk", "diffusion"):
        for seed in (1000, 1001):
            result = _rollout(model, seed, success=True)
            metadata = _metadata(result)
            path = root / model / metadata.recording_filename
            write_rollout_task_recording(
                output_path=path,
                result=result,
                metadata=metadata,
                rollout_index=seed - 1000,
            )
            artifacts_by_model[model].append(TaskRecordingArtifact(metadata=metadata, path=path))
            expected_by_model[model].append(metadata)

    paths = validate_paired_recording_directories(
        root,
        artifacts_by_model=artifacts_by_model,
        expected_by_model=expected_by_model,
    )
    assert [path.name for path in paths["bc_chunk"]] == ["env_1000.rrd", "env_1001.rrd"]
    (root / "diffusion" / "env_9999.rrd").write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="unexpected RRD file set"):
        validate_paired_recording_directories(
            root,
            artifacts_by_model=artifacts_by_model,
            expected_by_model=expected_by_model,
        )
