"""Compact two-dimensional Rerun recordings for PushT Pilot outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import torch

from .rollout import RolloutResult

AGENT_COLOR = (0, 174, 239)
ACTION_COLOR = (244, 125, 32)
BLOCK_COLOR = (120, 190, 90)
BC_COLOR = (225, 87, 89)
SAMPLE_COLORS = (
    (78, 121, 167),
    (242, 142, 43),
    (89, 161, 79),
    (225, 87, 89),
    (176, 122, 161),
    (156, 117, 95),
    (237, 201, 72),
    (186, 176, 172),
    (255, 157, 167),
    (118, 183, 178),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


def image_to_hwc_uint8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        image = value.detach().cpu().numpy()
    else:
        image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected one image with three dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.asarray(np.clip(image, 0, 255), dtype=np.uint8)


def _stream(output_path: Path, *, app_id: str, recording_id: str) -> rr.RecordingStream:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(app_id, recording_id=recording_id)
    recording.save(output_path)
    return recording


def record_demonstrations(
    *,
    output_path: str | Path,
    frames: Sequence[Mapping[str, Any]],
) -> Path:
    destination = Path(output_path)
    recording = _stream(
        destination,
        app_id="robot_practice_pusht_data_inspection",
        recording_id="pusht_demonstrations",
    )
    try:
        agent_history: dict[int, list[np.ndarray]] = {}
        action_history: dict[int, list[np.ndarray]] = {}
        for global_frame, frame in enumerate(frames):
            episode = int(frame["episode_index"])
            recording.set_time("global_frame", sequence=global_frame)
            agent_history.setdefault(episode, []).append(
                np.asarray(frame["agent_position"], dtype=np.float32)
            )
            action_history.setdefault(episode, []).append(
                np.asarray(frame["action_target"], dtype=np.float32)
            )
            root = f"episodes/episode_{episode:03d}"
            recording.log(f"{root}/image", rr.Image(frame["image"]).compress())
            recording.log(
                f"{root}/agent_path",
                rr.LineStrips2D(
                    [np.stack(agent_history[episode])],
                    colors=[AGENT_COLOR],
                    labels=["agent path"],
                    show_labels=False,
                ),
            )
            recording.log(
                f"{root}/action_target_path",
                rr.LineStrips2D(
                    [np.stack(action_history[episode])],
                    colors=[ACTION_COLOR],
                    labels=["absolute action targets"],
                    show_labels=False,
                ),
            )
            recording.log(f"{root}/frame_index", rr.Scalars(int(frame["frame_index"])))
            recording.log(f"{root}/reward", rr.Scalars(float(frame["reward"])))
    finally:
        recording.disconnect()
    return destination


def record_fixed_observation_samples(
    *,
    output_path: str | Path,
    candidates: Sequence[Mapping[str, Any]],
) -> Path:
    destination = Path(output_path)
    recording = _stream(
        destination,
        app_id="robot_practice_pusht_fixed_observation",
        recording_id="pusht_e1_samples",
    )
    try:
        for candidate_index, candidate in enumerate(candidates):
            recording.set_time("candidate", sequence=candidate_index)
            root = f"candidates/{candidate['candidate_id']}"
            recording.log(
                f"{root}/previous_image",
                rr.Image(candidate["previous_image"]).compress(),
            )
            recording.log(
                f"{root}/current_image",
                rr.Image(candidate["current_image"]).compress(),
            )
            agent = np.asarray(candidate["agent_position"], dtype=np.float32)
            recording.log(
                f"{root}/agent",
                rr.Points2D([agent], colors=[AGENT_COLOR], radii=6.0),
            )
            bc_actions = candidate.get("bc_actions")
            if bc_actions is not None:
                recording.log(
                    f"{root}/bc_chunk",
                    rr.LineStrips2D(
                        [np.asarray(bc_actions)],
                        colors=[BC_COLOR],
                        labels=["BCChunk"],
                        show_labels=False,
                    ),
                )
            for sample_index, actions in enumerate(candidate["diffusion_actions"]):
                recording.log(
                    f"{root}/diffusion/sample_{sample_index:02d}",
                    rr.LineStrips2D(
                        [np.asarray(actions)],
                        colors=[SAMPLE_COLORS[sample_index % len(SAMPLE_COLORS)]],
                        labels=[f"noise sample {sample_index}"],
                        show_labels=False,
                    ),
                )
            endpoints = np.asarray(candidate["diffusion_actions"])[:, -1]
            recording.log(
                f"{root}/diffusion/endpoints",
                rr.Points2D(
                    endpoints,
                    colors=[
                        SAMPLE_COLORS[index % len(SAMPLE_COLORS)] for index in range(len(endpoints))
                    ],
                    radii=4.0,
                    labels=[f"sample {index}" for index in range(len(endpoints))],
                    show_labels=False,
                ),
            )
            if "metrics" in candidate:
                metrics = candidate["metrics"]
                recording.log(
                    f"{root}/metrics",
                    rr.TextDocument(
                        "\n".join(
                            f"- {key}: {value}"
                            for key, value in metrics.items()
                            if key != "input_tensor_hashes_json"
                        )
                    ),
                )
    finally:
        recording.disconnect()
    return destination


def record_rollouts(
    *,
    output_path: str | Path,
    rollouts: Sequence[RolloutResult],
) -> Path:
    destination = Path(output_path)
    recorder = RolloutRecording(destination)
    try:
        for rollout in rollouts:
            recorder.add(rollout)
    finally:
        recorder.close()
    return destination


class RolloutRecording:
    """Incrementally write rollouts so rendered frames do not accumulate in memory."""

    def __init__(self, output_path: str | Path) -> None:
        self.destination = Path(output_path)
        self.recording = _stream(
            self.destination,
            app_id="robot_practice_pusht_rollouts",
            recording_id="pusht_closed_loop_rollouts",
        )
        self.global_frame = 0
        self.rollout_index = 0
        self._closed = False

    def add(self, rollout: RolloutResult) -> None:
        if self._closed:
            raise RuntimeError("Cannot add a rollout after closing the recording")
        root = (
            f"rollouts/{rollout.condition_id}/{rollout.model}"
            f"/env_{rollout.environment_seed:04d}"
            f"/policy_{rollout.policy_seed:02d}"
        )
        for frame_index in range(rollout.agent_path.shape[0]):
            self.recording.set_time("global_frame", sequence=self.global_frame)
            self.global_frame += 1
            if frame_index == 0:
                self.recording.log(
                    f"{root}/complete_agent_path",
                    rr.LineStrips2D(
                        [rollout.agent_path],
                        colors=[AGENT_COLOR],
                        labels=["complete measured agent path"],
                        show_labels=False,
                    ),
                )
                self.recording.log(
                    f"{root}/complete_block_path",
                    rr.LineStrips2D(
                        [rollout.block_path[:, :2]],
                        colors=[BLOCK_COLOR],
                        labels=["complete measured block path"],
                        show_labels=False,
                    ),
                )
                if rollout.action_targets.size:
                    self.recording.log(
                        f"{root}/complete_action_targets",
                        rr.LineStrips2D(
                            [rollout.action_targets],
                            colors=[ACTION_COLOR],
                            labels=["complete absolute target path"],
                            show_labels=False,
                        ),
                    )
            if frame_index < len(rollout.frames):
                self.recording.log(
                    f"{root}/simulator",
                    rr.Image(rollout.frames[frame_index]).compress(),
                )
            self.recording.log(
                f"{root}/current_agent",
                rr.Points2D(
                    [rollout.agent_path[frame_index]],
                    colors=[AGENT_COLOR],
                    radii=6.0,
                    labels=["current measured agent"],
                    show_labels=False,
                ),
            )
            self.recording.log(
                f"{root}/current_block",
                rr.Points2D(
                    [rollout.block_path[frame_index, :2]],
                    colors=[BLOCK_COLOR],
                    radii=6.0,
                    labels=["current measured block"],
                    show_labels=False,
                ),
            )
            if frame_index > 0:
                self.recording.log(
                    f"{root}/current_action_target",
                    rr.Points2D(
                        [rollout.action_targets[frame_index - 1]],
                        colors=[ACTION_COLOR],
                        radii=4.0,
                        labels=["current absolute action target"],
                        show_labels=False,
                    ),
                )
                self.recording.log(
                    f"{root}/reward",
                    rr.Scalars(float(rollout.rewards[frame_index - 1])),
                )
        self.recording.log(f"{root}/success", rr.Scalars(int(rollout.success)))
        self.recording.log(f"{root}/rollout_index", rr.Scalars(self.rollout_index))
        self.recording.log(
            f"{root}/metadata",
            rr.TextDocument(
                "\n".join(
                    [
                        f"condition: {rollout.condition_id}",
                        f"scheduler: {rollout.scheduler}",
                        f"num_inference_steps: {rollout.num_inference_steps}",
                        f"success: {rollout.success}",
                        f"max_reward: {rollout.max_reward:.6f}",
                    ]
                )
            ),
        )
        self.rollout_index += 1

    def close(self) -> None:
        if not self._closed:
            self.recording.disconnect()
            self._closed = True
