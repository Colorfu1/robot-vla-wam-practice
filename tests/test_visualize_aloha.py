from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import robot_practice.visualize_aloha as viz
from robot_practice.aloha_kinematics import CANONICAL_JOINT_NAMES


class FakeRecorder:
    def __init__(self) -> None:
        self.times: list[tuple[str, int | None, float | None]] = []
        self.logs: list[tuple[str, Any, bool]] = []

    def set_time(
        self,
        timeline: str,
        *,
        sequence: int | None = None,
        duration: float | None = None,
    ) -> None:
        self.times.append((timeline, sequence, duration))

    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None:
        self.logs.append((entity_path, entity, static))


def _features() -> dict[str, dict[str, object]]:
    return {
        key: {"shape": [14], "names": list(CANONICAL_JOINT_NAMES)}
        for key in viz.REQUIRED_NUMERIC_KEYS
    }


def _metadata(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "repo_id": "lerobot/test_aloha",
        "robot_type": "aloha",
        "total_episodes": 2,
        "features": _features(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_index", -1, "non-negative"),
        ("max_frames", 0, "positive"),
        ("batch_size", 0, "positive"),
        ("num_workers", -1, "non-negative"),
        ("save", Path("recording.bin"), "rrd"),
    ],
)
def test_validate_config_rejects_invalid_options(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "repo_id": "lerobot/test_aloha",
        "episode_index": 0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        viz.validate_config(viz.VisualizationConfig(**values))


def test_validate_dataset_layout_accepts_canonical_aloha() -> None:
    viz.validate_dataset_layout(_metadata())


def test_validate_dataset_layout_rejects_missing_feature() -> None:
    features = _features()
    features.pop(viz.EFFORT_KEY)
    with pytest.raises(ValueError, match="missing required feature.*observation.effort"):
        viz.validate_dataset_layout(_metadata(features=features))


def test_validate_dataset_layout_rejects_wrong_shape() -> None:
    features = _features()
    features[viz.ACTION_KEY] = {"shape": [12], "names": None}
    with pytest.raises(ValueError, match=r"action must have shape \[14\]"):
        viz.validate_dataset_layout(_metadata(features=features))


def test_validate_dataset_layout_rejects_wrong_joint_order() -> None:
    features = _features()
    names = list(CANONICAL_JOINT_NAMES)
    names[0], names[1] = names[1], names[0]
    features[viz.STATE_KEY] = {"shape": [14], "names": names}
    with pytest.raises(ValueError, match="motor order"):
        viz.validate_dataset_layout(_metadata(features=features))


def test_validate_episode_index_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        viz.validate_episode_index(_metadata(), 2)


def test_find_cached_root_skips_metadata_only_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "lerobot-cache"
    partial = cache / "lerobot" / "test_aloha"
    (partial / "meta").mkdir(parents=True)
    (partial / "meta" / "info.json").write_text("{}")

    snapshot = tmp_path / "complete-snapshot"
    (snapshot / "meta").mkdir(parents=True)
    (snapshot / "data").mkdir()
    (snapshot / "meta" / "info.json").write_text("{}")
    (snapshot / "data" / "episode.parquet").write_bytes(b"parquet")

    monkeypatch.setattr(viz, "HF_LEROBOT_HOME", cache)
    monkeypatch.setattr(viz, "snapshot_download", lambda *args, **kwargs: str(snapshot))

    assert viz.find_cached_dataset_root("lerobot/test_aloha", None) == snapshot


@pytest.mark.parametrize(("no_images", "download_videos"), [(True, False), (False, True)])
def test_load_dataset_configures_video_download_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_images: bool,
    download_videos: bool,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeMetadata:
        def __init__(self, repo_id: str, *, root: Path | None) -> None:
            self.repo_id = repo_id
            self.root = root
            self.robot_type = "aloha"
            self.total_episodes = 1
            self.features = _features()

    class FakeDataset:
        def __init__(self, repo_id: str, **kwargs: object) -> None:
            self.repo_id = repo_id
            calls.append(kwargs)

    monkeypatch.setattr(viz, "find_cached_dataset_root", lambda *args: tmp_path)
    monkeypatch.setattr(viz, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(viz, "LeRobotDataset", FakeDataset)

    config = viz.VisualizationConfig(
        repo_id="lerobot/test_aloha",
        episode_index=0,
        no_images=no_images,
    )
    viz.load_dataset(config)

    assert calls == [
        {
            "root": tmp_path,
            "episodes": [0],
            "download_videos": download_videos,
        }
    ]


def test_no_images_dataloader_never_calls_dataset_getitem() -> None:
    rows = [
        {
            "frame_index": torch.tensor(index),
            "timestamp": torch.tensor(index / 50),
        }
        for index in range(3)
    ]

    class DatasetWithForbiddenDecoder:
        hf_dataset = rows

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            raise AssertionError(f"video decoder was called for index {index}")

        def __len__(self) -> int:
            return len(rows)

    batches = list(
        viz.make_dataloader(
            DatasetWithForbiddenDecoder(),
            frame_count=3,
            no_images=True,
            batch_size=2,
            num_workers=0,
        )
    )

    assert torch.cat([batch["frame_index"] for batch in batches]).tolist() == [0, 1, 2]


def test_image_dataloader_decodes_each_selected_frame_once() -> None:
    calls: list[int] = []

    class CountingDataset:
        hf_dataset: list[object] = []

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            calls.append(index)
            return {"frame_index": torch.tensor(index)}

        def __len__(self) -> int:
            return 4

    list(
        viz.make_dataloader(
            CountingDataset(),
            frame_count=3,
            no_images=False,
            batch_size=2,
            num_workers=0,
        )
    )

    assert calls == [0, 1, 2]


def test_compute_velocity_uses_actual_timestamp_delta() -> None:
    previous = np.array([1.0, 3.0])
    current = np.array([2.0, 1.0])

    np.testing.assert_array_equal(viz.compute_velocity(current, 10.0, None, None), [0.0, 0.0])
    np.testing.assert_allclose(
        viz.compute_velocity(current, 0.75, previous, 0.25),
        [2.0, -4.0],
    )


@pytest.mark.parametrize("timestamp", [0.5, 0.4, np.nan])
def test_compute_velocity_rejects_non_increasing_or_invalid_timestamp(timestamp: float) -> None:
    state = np.zeros(2)
    with pytest.raises(ValueError, match="strictly increase"):
        viz.compute_velocity(state, timestamp, state, 0.5)


def test_signal_paths_and_action_minus_state(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = FakeRecorder()
    state = np.arange(14, dtype=np.float64)
    action = state + 0.25
    effort = state + 1.0
    velocity = state - 1.0
    monkeypatch.setattr(viz.rr, "Scalars", lambda value: value)

    viz._log_signals(recorder, state, action, effort, velocity)
    values = {path: entity for path, entity, static in recorder.logs if not static}

    assert len(values) == 5 * 14
    assert values["signals/position/left/waist"] == 0.0
    assert values["signals/action/right/gripper"] == 13.25
    assert values["signals/effort/left/elbow"] == 3.0
    assert values["signals/velocity/right/waist"] == 6.0
    assert values["signals/action_minus_state/right/wrist_rotate"] == 0.25


def test_log_frame_assigns_both_timelines_and_exact_entity_paths() -> None:
    recorder = FakeRecorder()
    state = np.zeros(14)

    viz.log_frame(
        recorder,
        frame_index=7,
        elapsed_seconds=0.14,
        state=state,
        action=state,
        effort=state,
        velocity=state,
        images={},
    )

    assert recorder.times == [
        ("frame_index", 7, None),
        ("timestamp", None, 0.14),
    ]
    paths = [path for path, _, static in recorder.logs if not static]
    expected_signals = {
        f"signals/{quantity}/{side}/{joint}"
        for quantity in ("position", "action", "effort", "velocity", "action_minus_state")
        for side in ("left", "right")
        for joint in viz.ARM_JOINT_NAMES
    }
    assert {path for path in paths if path.startswith("signals/")} == expected_signals
    for side in ("left", "right"):
        for entity in ("skeleton", "joints", "gripper", "end_effector"):
            assert paths.count(f"robot/{side}/{entity}") == 1


def test_complete_end_effector_paths_are_static() -> None:
    recorder = FakeRecorder()
    points = [np.zeros(3), np.ones(3)]

    viz.log_complete_paths(recorder, points, points)

    assert [(path, static) for path, _, static in recorder.logs] == [
        ("robot/left/end_effector_path", True),
        ("robot/right/end_effector_path", True),
    ]


def test_save_sink_is_configured_before_blueprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    blueprint = object()
    output = tmp_path / "episode.rrd"
    monkeypatch.setattr(viz, "build_blueprint", lambda *args: blueprint)
    monkeypatch.setattr(viz.rr, "init", lambda *args, **kwargs: events.append(("init", kwargs)))
    monkeypatch.setattr(viz.rr, "save", lambda path: events.append(("save", path)))
    monkeypatch.setattr(
        viz.rr,
        "send_blueprint",
        lambda value: events.append(("blueprint", value)),
    )
    monkeypatch.setattr(
        viz.rr,
        "spawn",
        lambda *args, **kwargs: pytest.fail("save mode must not spawn a viewer"),
    )

    viz._initialize_recording(
        viz.VisualizationConfig(
            repo_id="lerobot/test_aloha",
            episode_index=0,
            save=output,
        ),
        (),
        50.0,
    )

    assert events == [
        ("init", {"spawn": False}),
        ("save", output),
        ("blueprint", blueprint),
    ]


def test_cached_episode_offline_rrd_integration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_id = "lerobot/aloha_mobile_cabinet"
    root = viz.find_cached_dataset_root(repo_id, None)
    if root is None:
        pytest.skip(f"{repo_id} is not available in the local LeRobot cache")

    output = tmp_path / "aloha-three-frames.rrd"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    summary = viz.visualize(
        viz.VisualizationConfig(
            repo_id=repo_id,
            episode_index=0,
            root=root,
            max_frames=3,
            no_images=True,
            save=output,
        )
    )

    assert summary.frames == 3
    assert output.stat().st_size > 0
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        pytest.skip("Rerun CLI is unavailable for .rrd verification")
    result = subprocess.run(
        [rerun_cli, "rrd", "verify", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "1 file verified without error" in result.stdout + result.stderr
