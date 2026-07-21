"""Rerun recording helpers with one global sequence timeline per evaluation mode."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from numpy.typing import NDArray

ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


class Recorder(Protocol):
    def set_time(self, timeline: str, *, sequence: int) -> None: ...

    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None: ...


def _camera_path(camera_key: str) -> str:
    return f"cameras/{camera_key.rsplit('.', maxsplit=1)[-1]}"


def image_to_hwc_uint8(image: torch.Tensor | NDArray[Any]) -> NDArray[np.uint8]:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {array.shape}")
    if array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-first or channel-last image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Image contains NaN or Infinity")
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.asarray(array, dtype=np.uint8)


def _camera_views(camera_keys: Sequence[str]) -> rrb.Container | rrb.View:
    views = [
        rrb.Spatial2DView(
            origin=_camera_path(key),
            contents=[_camera_path(key)],
            name=key.rsplit(".", maxsplit=1)[-1],
        )
        for key in camera_keys
    ]
    if not views:
        return rrb.TextDocumentView(origin="summary", contents=["summary/metrics"])
    if len(views) == 1:
        return views[0]
    return rrb.Horizontal(*views, name="Cameras")


def build_offline_blueprint(camera_keys: Sequence[str], fps: float) -> rrb.Blueprint:
    charts = rrb.Tabs(
        rrb.TimeSeriesView(
            origin="actions",
            contents=["actions/target/**", "actions/predicted/**"],
            name="Target vs prediction",
        ),
        rrb.TimeSeriesView(
            origin="errors",
            contents=["errors/absolute/**"],
            name="Absolute error",
        ),
        rrb.TimeSeriesView(
            origin="latency",
            contents=["latency/**"],
            name="Model-call latency",
        ),
        rrb.TextDocumentView(
            origin="summary",
            contents=["summary/metrics"],
            name="Metrics",
        ),
        name="Offline signals",
    )
    return rrb.Blueprint(
        rrb.Vertical(_camera_views(camera_keys), charts, row_shares=[0.62, 0.38]),
        rrb.TimePanel(timeline="global_frame", fps=fps, expanded=True),
        collapse_panels=False,
    )


def build_libero_blueprint(camera_keys: Sequence[str], fps: float) -> rrb.Blueprint:
    charts = rrb.Tabs(
        rrb.TimeSeriesView(
            origin="actions/executed",
            contents=["actions/executed/**"],
            name="Executed actions",
        ),
        rrb.TimeSeriesView(
            origin="rollout",
            contents=["rollout/reward", "rollout/success_so_far", "rollout/done"],
            name="Reward and terminal state",
        ),
        rrb.TimeSeriesView(
            origin="latency",
            contents=["latency/**"],
            name="Policy latency",
        ),
        rrb.TextDocumentView(
            origin="summary",
            contents=["summary/metrics"],
            name="Metrics",
        ),
        name="Closed-loop signals",
    )
    return rrb.Blueprint(
        rrb.Vertical(_camera_views(camera_keys), charts, row_shares=[0.62, 0.38]),
        rrb.TimePanel(timeline="global_step", fps=fps, expanded=True),
        collapse_panels=False,
    )


def initialize_recording(
    *,
    output_path: Path,
    mode: str,
    camera_keys: Sequence[str],
    fps: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "offline":
        application_id = "robot_practice_act_offline"
        blueprint = build_offline_blueprint(camera_keys, fps)
    elif mode == "libero":
        application_id = "robot_practice_act_libero"
        blueprint = build_libero_blueprint(camera_keys, fps)
    else:
        raise ValueError(f"Unknown recording mode: {mode}")
    rr.init(application_id, spawn=False)
    rr.save(output_path)
    rr.send_blueprint(blueprint)


def finish_recording() -> None:
    rr.disconnect()


def _validate_action(action: Sequence[float], name: str) -> NDArray[np.float64]:
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain seven finite values, got {array}")
    return array


def log_offline_frame(
    recorder: Recorder,
    *,
    global_frame: int,
    episode_index: int,
    frame_index: int,
    batch_id: int,
    target_action: Sequence[float],
    predicted_action: Sequence[float],
    model_query_latency_ms: float | None,
    images: Mapping[str, torch.Tensor | NDArray[Any]],
) -> None:
    """Log one observation after setting the global frame timeline exactly once."""

    target = _validate_action(target_action, "target_action")
    predicted = _validate_action(predicted_action, "predicted_action")
    recorder.set_time("global_frame", sequence=global_frame)

    for key, image in images.items():
        recorder.log(_camera_path(key), rr.Image(image_to_hwc_uint8(image)).compress())
    recorder.log("context/episode_index", rr.Scalars(episode_index))
    recorder.log("context/frame_index", rr.Scalars(frame_index))
    recorder.log("context/batch_id", rr.Scalars(batch_id))
    for index, action_name in enumerate(ACTION_NAMES):
        recorder.log(f"actions/target/{action_name}", rr.Scalars(float(target[index])))
        recorder.log(
            f"actions/predicted/{action_name}", rr.Scalars(float(predicted[index]))
        )
        recorder.log(
            f"errors/absolute/{action_name}",
            rr.Scalars(float(abs(predicted[index] - target[index]))),
        )
    if model_query_latency_ms is not None:
        recorder.log("latency/model_query_ms", rr.Scalars(model_query_latency_ms))


def log_libero_step(
    recorder: Recorder,
    *,
    global_step: int,
    episode_index: int,
    step_index: int,
    action: Sequence[float],
    reward: float,
    success_so_far: bool,
    done_after_step: bool,
    policy_step_latency_ms: float,
    model_query_latency_ms: float | None,
    images: Mapping[str, torch.Tensor | NDArray[Any]],
) -> None:
    """Log one control transition on the single global-step timeline."""

    executed = _validate_action(action, "executed_action")
    recorder.set_time("global_step", sequence=global_step)
    for key, image in images.items():
        recorder.log(_camera_path(key), rr.Image(image_to_hwc_uint8(image)).compress())
    recorder.log("context/episode_index", rr.Scalars(episode_index))
    recorder.log("context/step_index", rr.Scalars(step_index))
    for index, action_name in enumerate(ACTION_NAMES):
        recorder.log(
            f"actions/executed/{action_name}", rr.Scalars(float(executed[index]))
        )
    recorder.log("rollout/reward", rr.Scalars(float(reward)))
    recorder.log("rollout/success_so_far", rr.Scalars(int(success_so_far)))
    recorder.log("rollout/done", rr.Scalars(int(done_after_step)))
    recorder.log("latency/policy_step_ms", rr.Scalars(policy_step_latency_ms))
    if model_query_latency_ms is not None:
        recorder.log("latency/model_query_ms", rr.Scalars(model_query_latency_ms))


def metrics_markdown(metrics: Mapping[str, Any]) -> str:
    aggregate = metrics.get("aggregate", {})
    definitions = metrics.get("metric_definitions", {})
    lines = [f"# {metrics.get('mode', 'evaluation')} metrics", ""]
    for name, value in aggregate.items():
        rendered = json.dumps(value, ensure_ascii=False)
        description = definitions.get(name, "")
        suffix = f" — {description}" if description else ""
        lines.append(f"- **{name}**: `{rendered}`{suffix}")
    return "\n".join(lines)


def log_metric_summary(recorder: Recorder, metrics: Mapping[str, Any]) -> None:
    recorder.log(
        "summary/metrics",
        rr.TextDocument(metrics_markdown(metrics), media_type="text/markdown"),
        static=True,
    )
