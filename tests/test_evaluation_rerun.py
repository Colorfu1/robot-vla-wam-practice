from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import robot_practice.evaluation.rerun_visualization as viz


class FakeRecorder:
    def __init__(self) -> None:
        self.times: list[tuple[str, int]] = []
        self.logs: list[tuple[str, Any, bool]] = []

    def set_time(self, timeline: str, *, sequence: int) -> None:
        self.times.append((timeline, sequence))

    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None:
        self.logs.append((entity_path, entity, static))


def _patch_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viz.rr, "Scalars", lambda value: value)

    class FakeImage:
        def __init__(self, image: np.ndarray) -> None:
            self.image = image

        def compress(self) -> "FakeImage":
            return self

    monkeypatch.setattr(viz.rr, "Image", FakeImage)


def test_image_conversion_supports_normalized_chw() -> None:
    image = torch.zeros(3, 4, 5)
    image[0] = 1.0
    converted = viz.image_to_hwc_uint8(image)
    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert converted[0, 0].tolist() == [255, 0, 0]


def test_offline_frame_sets_global_timeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    viz.log_offline_frame(
        recorder,
        global_frame=12,
        episode_index=823,
        frame_index=4,
        batch_id=2,
        target_action=np.zeros(7),
        predicted_action=np.ones(7),
        model_query_latency_ms=3.5,
        images={"observation.images.image": torch.zeros(3, 4, 5)},
    )
    assert recorder.times == [("global_frame", 12)]
    paths = [path for path, _, _ in recorder.logs]
    assert "cameras/image" in paths
    assert "context/episode_index" in paths
    assert "actions/target/x" in paths
    assert "actions/predicted/gripper" in paths
    assert "errors/absolute/yaw" in paths
    assert "latency/model_query_ms" in paths


def test_libero_queue_hit_omits_model_query_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entities(monkeypatch)
    recorder = FakeRecorder()
    viz.log_libero_step(
        recorder,
        global_step=8,
        episode_index=0,
        step_index=8,
        action=np.zeros(7),
        reward=0.0,
        success_so_far=False,
        done_after_step=False,
        policy_step_latency_ms=1.0,
        model_query_latency_ms=None,
        images={},
    )
    assert recorder.times == [("global_step", 8)]
    paths = [path for path, _, _ in recorder.logs]
    assert "latency/policy_step_ms" in paths
    assert "latency/model_query_ms" not in paths


def test_recording_configures_file_sink_before_blueprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    blueprint = object()
    monkeypatch.setattr(viz, "build_offline_blueprint", lambda *args: blueprint)
    monkeypatch.setattr(viz.rr, "init", lambda *args, **kwargs: events.append(("init", kwargs)))
    monkeypatch.setattr(viz.rr, "save", lambda path: events.append(("save", path)))
    monkeypatch.setattr(
        viz.rr, "send_blueprint", lambda value: events.append(("blueprint", value))
    )
    output = tmp_path / "prediction.rrd"
    viz.initialize_recording(
        output_path=output,
        mode="offline",
        camera_keys=(),
        fps=10.0,
    )
    assert events == [
        ("init", {"spawn": False}),
        ("save", output),
        ("blueprint", blueprint),
    ]


def test_small_rrd_is_valid(tmp_path: Path) -> None:
    output = tmp_path / "small.rrd"
    viz.initialize_recording(
        output_path=output,
        mode="offline",
        camera_keys=(),
        fps=10.0,
    )
    viz.log_offline_frame(
        viz.rr,
        global_frame=0,
        episode_index=823,
        frame_index=0,
        batch_id=0,
        target_action=np.zeros(7),
        predicted_action=np.zeros(7),
        model_query_latency_ms=1.0,
        images={},
    )
    viz.finish_recording()
    assert output.stat().st_size > 0
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        pytest.skip("Rerun CLI is unavailable")
    result = subprocess.run(
        [rerun_cli, "rrd", "verify", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
