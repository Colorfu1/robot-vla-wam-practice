"""Record and visualize S1 Policy Decorator PegInsertionSide rollouts.

The S1 evaluator intentionally runs in the pinned ManiSkill2 environment, which
does not need the Rerun SDK.  Recording and visualization are consequently two
separate layers:

* :class:`PolicyDecoratorPegRolloutRecorder` stores RGB frames, the 50D S1
  state, base actions, residual corrections, and native terminal signals in a
  dependency-free ``npz`` bundle.
* :func:`write_policy_decorator_peg_visualization` converts selected bundles to
  one self-contained Rerun recording (and optionally an MP4) per episode.

The base and residual runs should use the same input manifest.  This makes the
two independently generated recordings share reset seeds and lets them be
opened in two separate Rerun windows for a direct visual comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_FPS = 20.0
TIMELINE = "s1_step"
RENDER_PATH = "s1/rollout/render"
SIGNALS_ROOT = "s1/rollout/signals"
EVENTS_ROOT = "s1/rollout/events"
CONTEXT_PATH = "s1/rollout/context"
STATE_DIM = 50
ACTION_DIM = 7

ACTION_NAMES = (
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
)
STATE_NAMES = tuple(f"state_{index:02d}" for index in range(STATE_DIM))
ACTION_COLORS = (
    (78, 121, 167),
    (242, 142, 43),
    (89, 161, 79),
    (225, 87, 89),
    (176, 122, 161),
    (156, 117, 95),
    (47, 158, 170),
)
RESIDUAL_COLORS = (
    (118, 183, 178),
    (255, 152, 75),
    (100, 190, 120),
    (230, 110, 125),
    (190, 145, 190),
    (180, 145, 120),
    (75, 175, 185),
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return value


def _finite_vector(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


def _rgb_frame(value: Any, *, name: str = "frame") -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"{name} must be an HWC RGB frame, got {frame.shape}")
    if frame.dtype != np.uint8:
        if not np.issubdtype(frame.dtype, np.integer):
            raise ValueError(f"{name} must contain integer RGB values, got {frame.dtype}")
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


@dataclass
class _ActiveEpisode:
    episode_id: int
    reset_seed: int
    metadata: dict[str, Any]
    frames: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    base_action: list[np.ndarray] = field(default_factory=list)
    residual_unit: list[np.ndarray] = field(default_factory=list)
    residual_delta: list[np.ndarray] = field(default_factory=list)
    composed_action: list[np.ndarray] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)
    success_once: list[bool] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    decision_step: list[int] = field(default_factory=list)
    horizon_index: list[int] = field(default_factory=list)


class PolicyDecoratorPegRolloutRecorder:
    """Persist selected S1 closed-loop episodes without importing Rerun."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        max_episodes: int,
        fps: float = DEFAULT_FPS,
        metadata: Mapping[str, Any] | None = None,
        episode_ids: Sequence[int] | None = None,
    ) -> None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")
        output = Path(output_dir).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty rollout directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        self.output_dir = output
        self.max_episodes = int(max_episodes)
        self.fps = float(fps)
        self.metadata = dict(metadata or {})
        self.episode_ids = (
            None if episode_ids is None else frozenset(int(value) for value in episode_ids)
        )
        self._active: _ActiveEpisode | None = None
        self._summaries: list[dict[str, Any]] = []

    @property
    def captured_episode_count(self) -> int:
        return len(self._summaries)

    def should_capture(self, episode_id: int) -> bool:
        if self._active is not None:
            raise RuntimeError("cannot check capture eligibility during an active episode")
        if self.episode_ids is not None and int(episode_id) not in self.episode_ids:
            return False
        captured = {int(summary["episode_id"]) for summary in self._summaries}
        return int(episode_id) not in captured and len(captured) < self.max_episodes

    def start_episode(
        self,
        *,
        episode_id: int,
        reset_seed: int,
        state: Any,
        frame: Any,
        metadata: Mapping[str, Any],
    ) -> None:
        if self._active is not None:
            raise RuntimeError("another S1 rollout episode is already active")
        if not self.should_capture(episode_id):
            raise ValueError(f"episode_id={episode_id} is not eligible for capture")
        self._active = _ActiveEpisode(
            episode_id=int(episode_id),
            reset_seed=int(reset_seed),
            metadata={**self.metadata, **dict(metadata)},
            frames=[_rgb_frame(frame)],
            state=[_finite_vector(state, name="state", shape=(STATE_DIM,))],
            success_once=[False],
        )

    def record_step(
        self,
        *,
        state: Any,
        frame: Any,
        base_action: Any,
        residual_unit: Any,
        residual_delta: Any,
        composed_action: Any,
        reward: float,
        success: bool,
        success_once: bool,
        terminated: bool,
        truncated: bool,
        decision_step: int,
        horizon_index: int,
    ) -> None:
        active = self._active
        if active is None:
            return
        current_frame = _rgb_frame(frame)
        if current_frame.shape != active.frames[0].shape:
            raise ValueError(
                "S1 render shape changed during one rollout episode: "
                f"{active.frames[0].shape} -> {current_frame.shape}"
            )
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("rollout reward must be finite")
        active.frames.append(np.array(current_frame, copy=True))
        active.state.append(_finite_vector(state, name="state", shape=(STATE_DIM,)))
        active.base_action.append(
            _finite_vector(base_action, name="base_action", shape=(ACTION_DIM,))
        )
        active.residual_unit.append(
            _finite_vector(residual_unit, name="residual_unit", shape=(ACTION_DIM,))
        )
        active.residual_delta.append(
            _finite_vector(residual_delta, name="residual_delta", shape=(ACTION_DIM,))
        )
        active.composed_action.append(
            _finite_vector(composed_action, name="composed_action", shape=(ACTION_DIM,))
        )
        active.reward.append(reward_value)
        active.success.append(bool(success))
        active.success_once.append(bool(success_once))
        active.terminated.append(bool(terminated))
        active.truncated.append(bool(truncated))
        active.decision_step.append(int(decision_step))
        active.horizon_index.append(int(horizon_index))

    def finish_episode(
        self,
        *,
        success: bool,
        first_success_step: int | None,
        return_total: float,
        steps: int,
        decision_steps: int,
    ) -> dict[str, Any] | None:
        active = self._active
        if active is None:
            return None
        if not active.base_action:
            raise ValueError(f"captured episode {active.episode_id} contains no actions")
        frame_count = len(active.frames)
        action_count = len(active.base_action)
        if frame_count != action_count + 1 or len(active.state) != frame_count:
            raise RuntimeError(
                f"captured episode {active.episode_id} has inconsistent frames/actions/state: "
                f"{frame_count}/{action_count}/{len(active.state)}"
            )
        stem = f"episode_{active.episode_id:04d}"
        path = self.output_dir / f"{stem}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite rollout bundle: {path}")
        metadata = {
            **active.metadata,
            "episode_id": active.episode_id,
            "reset_seed": active.reset_seed,
            "success": bool(success),
            "first_success_step": first_success_step,
            "return": float(return_total),
            "steps": int(steps),
            "decision_steps": int(decision_steps),
            "frame_count": frame_count,
            "action_count": action_count,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                frames=np.stack(active.frames, axis=0).astype(np.uint8, copy=False),
                state=np.stack(active.state, axis=0).astype(np.float32, copy=False),
                base_action=np.stack(active.base_action, axis=0).astype(np.float32, copy=False),
                residual_unit=np.stack(active.residual_unit, axis=0).astype(
                    np.float32, copy=False
                ),
                residual_delta=np.stack(active.residual_delta, axis=0).astype(
                    np.float32, copy=False
                ),
                composed_action=np.stack(active.composed_action, axis=0).astype(
                    np.float32, copy=False
                ),
                reward=np.asarray(active.reward, dtype=np.float32),
                success=np.asarray(active.success, dtype=bool),
                success_once=np.asarray(active.success_once[1:], dtype=bool),
                frame_success_once=np.asarray(active.success_once, dtype=bool),
                terminated=np.asarray(active.terminated, dtype=bool),
                truncated=np.asarray(active.truncated, dtype=bool),
                decision_step=np.asarray(active.decision_step, dtype=np.int32),
                horizon_index=np.asarray(active.horizon_index, dtype=np.int32),
            )
        temporary.replace(path)
        summary = {
            **metadata,
            "file": path.name,
            "render_shape": list(active.frames[0].shape),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
        }
        self._summaries.append(summary)
        self._active = None
        return summary

    def finalize(self) -> dict[str, Any]:
        if self._active is not None:
            raise RuntimeError("cannot finalize while a rollout episode is active")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": "policy_decorator_s1_peg_closed_loop_rollout",
            "task": "PegInsertionSide-v2",
            "fps": self.fps,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "visualization_semantics": {
                "render": "native RGB output from PegInsertionSide-v2 environment.render()",
                "state": "the exact 50D S1 policy state, names kept positional",
                "base_action": "frozen Policy Decorator base output before residual composition",
                "residual_unit": "SAC residual actor output before residual scale",
                "residual_delta": "residual_unit multiplied by the resolved residual scale",
                "composed_action": "post-composition action after the evaluator action-space clip",
                "success": "native ManiSkill evaluate()['success'] and episode-level latch",
            },
            **self.metadata,
            "episodes": list(self._summaries),
        }
        _write_json_atomic(self.output_dir / "rollout_manifest.json", manifest)
        return manifest


@dataclass(frozen=True)
class PolicyDecoratorPegEpisode:
    episode_id: int
    metadata: Mapping[str, Any]
    frames: np.ndarray
    state: np.ndarray
    base_action: np.ndarray
    residual_unit: np.ndarray
    residual_delta: np.ndarray
    composed_action: np.ndarray
    reward: np.ndarray
    success: np.ndarray
    success_once: np.ndarray
    frame_success_once: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    decision_step: np.ndarray
    horizon_index: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def action_count(self) -> int:
        return int(self.base_action.shape[0])

    @property
    def result_label(self) -> str:
        return "success" if bool(self.metadata.get("success", False)) else "failure"


def _episode_record_map(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    values = manifest.get("episodes")
    if not isinstance(values, list):
        raise TypeError("rollout manifest must contain an episodes list")
    records: dict[int, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("rollout manifest episode records must be objects")
        episode_id = int(value["episode_id"])
        if episode_id in records:
            raise ValueError(f"duplicate rollout episode_id={episode_id}")
        records[episode_id] = value
    return records


def load_policy_decorator_peg_episode(
    rollout_dir: str | Path,
    *,
    episode_id: int,
) -> PolicyDecoratorPegEpisode:
    root = Path(rollout_dir).expanduser().resolve()
    manifest = _read_json(root / "rollout_manifest.json")
    records = _episode_record_map(manifest)
    if int(episode_id) not in records:
        raise KeyError(f"rollout episode_id={episode_id} is not present in {root}")
    record = records[int(episode_id)]
    bundle = root / str(record["file"])
    if not bundle.is_file():
        raise FileNotFoundError(f"rollout bundle is missing: {bundle}")
    with np.load(bundle, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    required = {
        "frames",
        "state",
        "base_action",
        "residual_unit",
        "residual_delta",
        "composed_action",
        "reward",
        "success",
        "success_once",
        "frame_success_once",
        "terminated",
        "truncated",
        "decision_step",
        "horizon_index",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"rollout bundle {bundle} is missing arrays: {missing}")
    frames = arrays["frames"]
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"invalid S1 RGB frames in {bundle}: {frames.shape}/{frames.dtype}")
    frame_count = int(frames.shape[0])
    action_count = frame_count - 1
    if arrays["state"].shape != (frame_count, STATE_DIM):
        raise ValueError(
            f"S1 state must have shape {(frame_count, STATE_DIM)}, got {arrays['state'].shape}"
        )
    for key in ("base_action", "residual_unit", "residual_delta", "composed_action"):
        if arrays[key].shape != (action_count, ACTION_DIM):
            raise ValueError(
                f"S1 {key} must have shape {(action_count, ACTION_DIM)}, got {arrays[key].shape}"
            )
    for key in (
        "reward",
        "success",
        "success_once",
        "terminated",
        "truncated",
        "decision_step",
        "horizon_index",
    ):
        if arrays[key].shape != (action_count,):
            raise ValueError(
                f"S1 {key} must have shape {(action_count,)}, got {arrays[key].shape}"
            )
    if arrays["frame_success_once"].shape != (frame_count,):
        raise ValueError(
            f"S1 frame_success_once must have shape {(frame_count,)}, "
            f"got {arrays['frame_success_once'].shape}"
        )
    return PolicyDecoratorPegEpisode(
        episode_id=int(episode_id),
        metadata=dict(record),
        frames=frames,
        state=arrays["state"].astype(np.float32, copy=False),
        base_action=arrays["base_action"].astype(np.float32, copy=False),
        residual_unit=arrays["residual_unit"].astype(np.float32, copy=False),
        residual_delta=arrays["residual_delta"].astype(np.float32, copy=False),
        composed_action=arrays["composed_action"].astype(np.float32, copy=False),
        reward=arrays["reward"].astype(np.float32, copy=False),
        success=arrays["success"].astype(bool, copy=False),
        success_once=arrays["success_once"].astype(bool, copy=False),
        frame_success_once=arrays["frame_success_once"].astype(bool, copy=False),
        terminated=arrays["terminated"].astype(bool, copy=False),
        truncated=arrays["truncated"].astype(bool, copy=False),
        decision_step=arrays["decision_step"].astype(np.int32, copy=False),
        horizon_index=arrays["horizon_index"].astype(np.int32, copy=False),
    )


def _selected_episode_ids(
    manifest: Mapping[str, Any],
    requested: Sequence[int] | None,
    *,
    max_episodes: int,
) -> tuple[int, ...]:
    records = _episode_record_map(manifest)
    available = tuple(sorted(records))
    if not available:
        raise ValueError("rollout manifest contains no episodes")
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if requested:
        selected = tuple(int(value) for value in requested)
        if len(set(selected)) != len(selected):
            raise ValueError("episode IDs must be unique")
        missing = [value for value in selected if value not in records]
        if missing:
            raise ValueError(f"requested episode IDs are missing: {missing}")
        return selected
    return available[:max_episodes]


def _write_video(frames: np.ndarray, *, output_path: Path, fps: float) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--write-video requires ffmpeg on PATH")
    height, width, channels = frames.shape[1:]
    if channels != 3:
        raise ValueError(f"expected RGB frames, got {frames.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        input=np.ascontiguousarray(frames).tobytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {detail}")
    return output_path


def _build_blueprint(*, fps: float) -> Any:
    import rerun.blueprint as rrb

    signal_tabs = rrb.Tabs(
        rrb.TimeSeriesView(
            origin=f"{SIGNALS_ROOT}/base_action",
            contents=[f"{SIGNALS_ROOT}/base_action/**"],
            name="Frozen base action",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TimeSeriesView(
            origin=f"{SIGNALS_ROOT}/residual_delta",
            contents=[f"{SIGNALS_ROOT}/residual_delta/**"],
            name="Scaled SAC residual",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TimeSeriesView(
            origin=f"{SIGNALS_ROOT}/composed_action",
            contents=[f"{SIGNALS_ROOT}/composed_action/**"],
            name="Composed action",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TimeSeriesView(
            origin=f"{SIGNALS_ROOT}/state",
            contents=[f"{SIGNALS_ROOT}/state/**"],
            name="S1 50D state",
            plot_legend=rrb.PlotLegend(visible=False),
        ),
        rrb.TimeSeriesView(
            origin=EVENTS_ROOT,
            contents=[f"{EVENTS_ROOT}/**"],
            name="Success, reward, and terminal signals",
            plot_legend=rrb.PlotLegend(visible=True),
        ),
        rrb.TextDocumentView(
            origin=CONTEXT_PATH,
            contents=[CONTEXT_PATH],
            name="Rollout metadata",
        ),
        name="S1 rollout signals",
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial2DView(
                origin=RENDER_PATH,
                contents=[RENDER_PATH],
                name="PegInsertionSide environment render",
            ),
            signal_tabs,
            row_shares=[0.58, 0.42],
        ),
        rrb.TimePanel(timeline=TIMELINE, fps=fps, expanded=True),
        collapse_panels=False,
    )


def _log_series_definitions(
    recording: Any,
    *,
    root: str,
    names: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
) -> None:
    import rerun as rr

    for index, name in enumerate(names):
        recording.log(
            f"{root}/{name}",
            rr.SeriesLines(
                colors=[colors[index % len(colors)]],
                widths=[1.5],
                names=[name],
            ),
            static=True,
        )


def _log_static_context(recording: Any, episode: PolicyDecoratorPegEpisode) -> None:
    import rerun as rr

    _log_series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/base_action",
        names=ACTION_NAMES,
        colors=ACTION_COLORS,
    )
    _log_series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/residual_delta",
        names=ACTION_NAMES,
        colors=RESIDUAL_COLORS,
    )
    _log_series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/composed_action",
        names=ACTION_NAMES,
        colors=ACTION_COLORS,
    )
    _log_series_definitions(
        recording,
        root=f"{SIGNALS_ROOT}/state",
        names=STATE_NAMES,
        colors=((31, 119, 180), (255, 127, 14), (44, 160, 44)),
    )
    for name, color in (
        ("reward", (80, 145, 220)),
        ("success", (0, 170, 80)),
        ("success_once", (0, 210, 120)),
        ("terminated", (220, 80, 80)),
        ("truncated", (220, 150, 40)),
    ):
        recording.log(
            f"{EVENTS_ROOT}/{name}",
            rr.SeriesLines(colors=[color], widths=[2.0], names=[name]),
            static=True,
        )
    notes = (
        f"# S1 Policy Decorator Peg rollout | episode {episode.episode_id:04d}\n\n"
        f"- **Task:** {episode.metadata.get('requested_env_id', 'PegInsertionSide-v2')}\n"
        f"- **Policy variant:** {episode.metadata.get('arm', 'unknown')}\n"
        f"- **Reset seed:** {episode.metadata.get('reset_seed', 'unknown')}\n"
        f"- **Policy seed:** {episode.metadata.get('policy_seed', 'unknown')}\n"
        f"- **Frames/actions:** {episode.frame_count}/{episode.action_count}\n"
        f"- **Success:** {episode.metadata.get('success', False)}\n"
        f"- **First success step:** {episode.metadata.get('first_success_step')}\n"
        f"- **Residual checkpoint step:** "
        f"{episode.metadata.get('residual_checkpoint_step', 'not applicable')}\n\n"
        "The image panel is the native environment RGB render. The state panel keeps the "
        "official 50D S1 state positional because the pinned checkout does not publish stable "
        "field names in this local adapter. `composed_action` is the final clipped action.\n"
    )
    recording.log(
        CONTEXT_PATH,
        rr.TextDocument(notes, media_type="text/markdown"),
        static=True,
    )


def _log_frame(recording: Any, episode: PolicyDecoratorPegEpisode, frame_index: int) -> None:
    import rerun as rr

    recording.set_time(TIMELINE, sequence=frame_index)
    recording.log(RENDER_PATH, rr.Image(episode.frames[frame_index]).compress())
    for index, value in enumerate(episode.state[frame_index]):
        recording.log(f"{SIGNALS_ROOT}/state/{STATE_NAMES[index]}", rr.Scalars(float(value)))
    recording.log(
        f"{EVENTS_ROOT}/success_once",
        rr.Scalars(int(episode.frame_success_once[frame_index])),
    )
    if frame_index == 0:
        return
    action_index = frame_index - 1
    for root, values in (
        ("base_action", episode.base_action[action_index]),
        ("residual_delta", episode.residual_delta[action_index]),
        ("composed_action", episode.composed_action[action_index]),
    ):
        for index, value in enumerate(values):
            recording.log(
                f"{SIGNALS_ROOT}/{root}/{ACTION_NAMES[index]}",
                rr.Scalars(float(value)),
            )
    recording.log(f"{EVENTS_ROOT}/reward", rr.Scalars(float(episode.reward[action_index])))
    recording.log(f"{EVENTS_ROOT}/success", rr.Scalars(int(episode.success[action_index])))
    recording.log(
        f"{EVENTS_ROOT}/terminated",
        rr.Scalars(int(episode.terminated[action_index])),
    )
    recording.log(
        f"{EVENTS_ROOT}/truncated",
        rr.Scalars(int(episode.truncated[action_index])),
    )


def write_policy_decorator_peg_episode_recording(
    episode: PolicyDecoratorPegEpisode,
    *,
    output_path: Path,
    fps: float,
) -> Path:
    """Write one independent Rerun recording for one S1 episode."""

    import rerun as rr

    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite Rerun recording: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    variant = str(episode.metadata.get("arm", "unknown"))
    recording = rr.RecordingStream(
        "robot_practice_peg_policy_decorator",
        recording_id=f"peg_policy_decorator_{variant}_{episode.episode_id:04d}",
    )
    try:
        recording.save(temporary)
        recording.send_recording_name(
            f"PegInsertionSide-v2 Policy Decorator {variant} | episode "
            f"{episode.episode_id:04d} | {episode.result_label}"
        )
        rr.send_blueprint(
            _build_blueprint(fps=fps),
            make_active=False,
            make_default=True,
            recording=recording,
        )
        _log_static_context(recording, episode)
        for frame_index in range(episode.frame_count):
            _log_frame(recording, episode, frame_index)
        recording.flush()
    finally:
        recording.disconnect()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Rerun recording was not written: {temporary}")
    temporary.replace(output_path)
    return output_path


def write_policy_decorator_peg_visualization(
    *,
    rollout_dir: str | Path,
    output_dir: str | Path,
    episode_ids: Sequence[int] | None = None,
    max_episodes: int = 1,
    fps: float = DEFAULT_FPS,
    write_video: bool = False,
    open_viewer: bool = False,
) -> dict[str, Any]:
    """Convert selected raw S1 bundles into independent Rerun/MP4 artifacts."""

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    root = Path(rollout_dir).expanduser().resolve()
    manifest = _read_json(root / "rollout_manifest.json")
    selected = _selected_episode_ids(manifest, episode_ids, max_episodes=max_episodes)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty visualization directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for completed, episode_id in enumerate(selected, start=1):
        print(f"STATUS completed={completed - 1} total={len(selected)}", flush=True)
        episode = load_policy_decorator_peg_episode(root, episode_id=episode_id)
        stem = f"episode_{episode.episode_id:04d}_{episode.result_label}"
        recording_path = write_policy_decorator_peg_episode_recording(
            episode,
            output_path=output / f"{stem}.rrd",
            fps=fps,
        )
        video_path = None
        if write_video:
            video_path = _write_video(
                episode.frames,
                output_path=output / f"{stem}.mp4",
                fps=fps,
            )
        viewer_opened = False
        if open_viewer:
            # Keep the normal conversion path independent of the optional
            # evaluation stack (which currently imports pyarrow at module load).
            from robot_practice.evaluation.common import launch_rerun_viewer

            viewer_opened = launch_rerun_viewer(recording_path, enabled=True)
        summaries.append(
            {
                **dict(episode.metadata),
                "episode_id": episode.episode_id,
                "frame_count": episode.frame_count,
                "action_count": episode.action_count,
                "result": episode.result_label,
                "recording": recording_path.as_posix(),
                "video": None if video_path is None else video_path.as_posix(),
                "viewer_opened": viewer_opened,
            }
        )
        print(f"STATUS completed={completed} total={len(selected)}", flush=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "source": "policy_decorator_s1_peg_closed_loop_rollout",
        "task": "PegInsertionSide-v2",
        "rollout_dir": root.as_posix(),
        "output_dir": output.as_posix(),
        "fps": float(fps),
        "write_video": bool(write_video),
        "selected_episode_ids": list(selected),
        "viewer": {
            "kind": "peg_policy_decorator_closed_loop",
            "application_id": "robot_practice_peg_policy_decorator",
            "launcher": "rerun",
        },
        "episodes": summaries,
    }
    _write_json_atomic(output / "visualization_manifest.json", result)
    print(f"DONE completed={len(selected)} total={len(selected)}", flush=True)
    return result


def _parse_episode_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", required=True)
    parser.add_argument("--episode-ids", "--episodes")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--open-viewer", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    write_policy_decorator_peg_visualization(
        rollout_dir=arguments.rollout_dir,
        output_dir=arguments.output_dir,
        episode_ids=_parse_episode_ids(arguments.episode_ids),
        max_episodes=arguments.max_episodes,
        fps=arguments.fps,
        write_video=arguments.write_video,
        open_viewer=arguments.open_viewer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
