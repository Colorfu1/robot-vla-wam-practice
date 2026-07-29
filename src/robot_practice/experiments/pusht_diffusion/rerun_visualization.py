"""Compact two-dimensional Rerun recordings for PushT Pilot outputs."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from .e0_geometry import WORKSPACE_SIZE
from .rollout import RolloutResult

AGENT_COLOR = (0, 174, 239)
ACTION_COLOR = (244, 125, 32)
BLOCK_COLOR = (120, 190, 90)
BC_SAMPLE_COLORS = (
    (255, 0, 0),
    (255, 20, 147),
    (128, 0, 255),
)
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
DEVELOPMENT_ROLLOUT_ROOT = "/rollout"
FIXED_OBSERVATION_ROOT = "/e1"
E3_COMPARISON_ROOT = "/conditions"
FIXED_OBSERVATION_PROTOCOLS = {
    "initial_noise_only": {
        "title": "E1a — initial-noise-only",
        "app_id": "robot_practice_pusht_e1a_initial_noise_only",
        "recording_id": "pusht_e1a_fixed_observation",
    },
    "full_sampling_seed": {
        "title": "E1b — full-sampling-seed",
        "app_id": "robot_practice_pusht_e1b_full_sampling_seed",
        "recording_id": "pusht_e1b_fixed_observation",
    },
}


def build_development_rollout_blueprint() -> rrb.Blueprint:
    """Build the seed-independent layout shared by development task Recordings."""

    root = DEVELOPMENT_ROLLOUT_ROOT
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                name="PushT rollout",
                origin=root,
                contents=[
                    f"{root}/simulator",
                    f"{root}/complete_agent_path",
                    f"{root}/complete_block_path",
                    f"{root}/complete_action_targets",
                    f"{root}/current_agent",
                    f"{root}/current_block",
                    f"{root}/current_action_target",
                ],
            ),
            rrb.TimeSeriesView(
                name="Reward",
                origin=f"{root}/reward",
                contents=[f"{root}/reward"],
            ),
            column_shares=[3.0, 1.0],
        ),
        auto_layout=False,
        auto_views=False,
    )


def _e3_condition_title(condition_id: str) -> str:
    scheduler, separator, raw_steps = condition_id.rpartition("_")
    if separator and scheduler and raw_steps.isdigit():
        return f"{scheduler.upper()}-{int(raw_steps)}"
    return condition_id.replace("_", "-").upper()


def build_e3_sampling_steps_blueprint(
    condition_ids: Sequence[str],
) -> rrb.Blueprint:
    """Build synchronized rollout panels for one E3 environment."""

    conditions = tuple(str(value) for value in condition_ids)
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("E3 condition_ids must be non-empty and unique")
    views = [
        rrb.Spatial2DView(
            name=_e3_condition_title(condition_id),
            origin=f"{E3_COMPARISON_ROOT}/{condition_id}",
            contents=[
                f"{E3_COMPARISON_ROOT}/{condition_id}/simulator",
                f"{E3_COMPARISON_ROOT}/{condition_id}/complete_agent_path",
                f"{E3_COMPARISON_ROOT}/{condition_id}/complete_block_path",
                f"{E3_COMPARISON_ROOT}/{condition_id}/complete_action_targets",
                f"{E3_COMPARISON_ROOT}/{condition_id}/current_agent",
                f"{E3_COMPARISON_ROOT}/{condition_id}/current_block",
                f"{E3_COMPARISON_ROOT}/{condition_id}/current_action_target",
            ],
        )
        for condition_id in conditions
    ]
    rows = [
        rrb.Horizontal(
            *views[start : start + 3],
            column_shares=[1.0] * len(views[start : start + 3]),
        )
        for start in range(0, len(views), 3)
    ]
    reward = rrb.TimeSeriesView(
        name="Reward by sampling condition",
        origin=E3_COMPARISON_ROOT,
        contents=[f"{E3_COMPARISON_ROOT}/{condition_id}/reward" for condition_id in conditions],
    )
    return rrb.Blueprint(
        rrb.Vertical(
            *rows,
            reward,
            row_shares=[3.0] * len(rows) + [1.0],
        ),
        auto_layout=False,
        auto_views=False,
    )


def build_fixed_observation_blueprint(
    *,
    sampling_protocol: str = "initial_noise_only",
) -> rrb.Blueprint:
    """Build the synchronized BCChunk and Diffusion E1 comparison panels."""

    try:
        protocol = FIXED_OBSERVATION_PROTOCOLS[sampling_protocol]
    except KeyError as exc:
        raise ValueError(f"Unknown E1 sampling protocol: {sampling_protocol}") from exc
    root = FIXED_OBSERVATION_ROOT
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                name="BCChunk — 3 repeated predictions",
                origin=f"{root}/bc",
                contents=[f"{root}/bc/**"],
            ),
            rrb.Spatial2DView(
                name=str(protocol["title"]),
                origin=f"{root}/dp",
                contents=[f"{root}/dp/**"],
            ),
            column_shares=[1.0, 1.0],
        ),
        auto_layout=False,
        auto_views=False,
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


def workspace_points_to_image(
    points: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    *,
    image: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Map PushT workspace coordinates onto the pixel coordinates of ``image``."""

    frame = image_to_hwc_uint8(image)
    height, width = frame.shape[:2]
    if isinstance(points, torch.Tensor):
        values = points.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        values = np.asarray(points, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError(f"Expected points with a final dimension of 2, got {values.shape}")
    scale = np.asarray(
        [width / WORKSPACE_SIZE, height / WORKSPACE_SIZE],
        dtype=np.float32,
    )
    return values * scale


def record_demonstration_episode(
    *,
    output_path: str | Path,
    frames: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Path:
    """Write one full E0 demonstration as one selector-visible Recording."""

    if not frames:
        raise ValueError("Cannot record an empty demonstration")
    episode = int(summary["episode_index"])
    if any(int(frame["episode_index"]) != episode for frame in frames):
        raise ValueError("All demonstration frames must belong to the summary episode")
    destination = Path(output_path)
    recording = _stream(
        destination,
        app_id="robot_practice_pusht_data_inspection",
        recording_id=f"episode_{episode:03d}",
    )
    recording.send_recording_name(
        f"Episode {episode:03d} | {summary['split']} | {summary['strategy_label']}"
    )
    try:
        recording.log("/task/episode_index", rr.Scalars(episode), static=True)
        recording.log("/task/split", rr.TextDocument(str(summary["split"])), static=True)
        recording.log(
            "/task/strategy",
            rr.TextDocument(str(summary["strategy_label"])),
            static=True,
        )
        recording.log(
            "/task/max_reward",
            rr.Scalars(float(summary["max_reward"])),
            static=True,
        )
        recording.log(
            "/task/outcome",
            rr.TextDocument(str(summary["outcome_label"])),
            static=True,
        )
        first_image = np.asarray(frames[0]["image"])
        agent_path = workspace_points_to_image(
            [frame["agent_position"] for frame in frames],
            image=first_image,
        )
        action_path = workspace_points_to_image(
            [frame["action_target"] for frame in frames],
            image=first_image,
        )
        valid_block_frames = [frame for frame in frames if frame["block_pose_valid"]]
        block_path = np.asarray(
            [frame["block_position_image"] for frame in valid_block_frames],
            dtype=np.float32,
        )
        for local_frame, frame in enumerate(frames):
            recording.set_time("frame", sequence=local_frame)
            if local_frame == 0:
                recording.log(
                    "/demonstration/complete_agent_path",
                    rr.LineStrips2D(
                        [agent_path],
                        colors=[AGENT_COLOR],
                        labels=["complete measured agent path"],
                        show_labels=False,
                    ),
                )
                recording.log(
                    "/demonstration/complete_action_target_path",
                    rr.LineStrips2D(
                        [action_path],
                        colors=[ACTION_COLOR],
                        labels=["complete absolute action-target path"],
                        show_labels=False,
                    ),
                )
                if len(block_path) >= 2:
                    recording.log(
                        "/demonstration/complete_block_path",
                        rr.LineStrips2D(
                            [block_path],
                            colors=[BLOCK_COLOR],
                            labels=["image-derived moving block path"],
                            show_labels=False,
                        ),
                    )
                recording.log(
                    "/demonstration/metadata",
                    rr.TextDocument(
                        "\n".join(
                            [
                                f"episode: {episode}",
                                f"split: {summary['split']}",
                                f"strategy: {summary['strategy_label']}",
                                f"approach: {summary['approach_side']}",
                                f"block motion: {summary['primary_block_motion']}",
                                f"rotation first: {summary['rotation_first']}",
                                f"recontact: {summary['recontact']}",
                                f"geometry coverage: {summary['geometry_coverage']:.3f}",
                                f"outcome: {summary['outcome_label']}",
                                f"max reward: {summary['max_reward']:.6f}",
                            ]
                        )
                    ),
                )
            image = np.asarray(frame["image"])
            recording.log("/demonstration/image", rr.Image(image).compress())
            recording.log(
                "/demonstration/current_agent",
                rr.Points2D(
                    [agent_path[local_frame]],
                    colors=[AGENT_COLOR],
                    radii=3.0,
                    labels=["current measured agent"],
                    show_labels=False,
                ),
            )
            recording.log(
                "/demonstration/current_action_target",
                rr.Points2D(
                    [action_path[local_frame]],
                    colors=[ACTION_COLOR],
                    radii=2.5,
                    labels=["current absolute action target"],
                    show_labels=False,
                ),
            )
            if frame["block_pose_valid"]:
                block = np.asarray(frame["block_position_image"], dtype=np.float32)
                angle = float(frame["block_angle_rad"])
                direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
                recording.log(
                    "/demonstration/current_block",
                    rr.Points2D(
                        [block],
                        colors=[BLOCK_COLOR],
                        radii=3.0,
                        labels=["image-derived moving block center"],
                        show_labels=False,
                    ),
                )
                recording.log(
                    "/demonstration/current_block_orientation",
                    rr.LineStrips2D(
                        [np.stack((block, block + direction * 10.0))],
                        colors=[BLOCK_COLOR],
                        labels=["T-head direction"],
                        show_labels=False,
                    ),
                )
            recording.log(
                "/demonstration/frame_index",
                rr.Scalars(int(frame["frame_index"])),
            )
            recording.log("/demonstration/reward", rr.Scalars(float(frame["reward"])))
            surface_distance = float(frame["agent_block_surface_distance"])
            if np.isfinite(surface_distance):
                recording.log(
                    "/demonstration/agent_block_surface_distance",
                    rr.Scalars(surface_distance),
                )
    finally:
        recording.flush()
        recording.disconnect()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Demonstration Recording was not written: {destination}")
    return destination


def merge_rerun_recordings(
    *,
    input_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Path:
    """Merge independent task Recordings into one Rerun selector archive."""

    sources = [Path(path) for path in input_paths]
    if not sources:
        raise ValueError("At least one Rerun Recording is required")
    missing = [path for path in sources if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Rerun Recording is missing or empty: {missing[0]}")
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        raise RuntimeError("The rerun CLI is required to merge E0 demonstration Recordings")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Rerun archive already exists: {destination}")
    merged = destination.with_name(f".{destination.stem}.merged.rrd")
    merged.unlink(missing_ok=True)

    def run(arguments: Sequence[str], *, operation: str) -> None:
        completed = subprocess.run(
            [rerun_cli, "rrd", *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Rerun {operation} failed: {detail}")

    try:
        run(
            ["merge", "-o", str(merged), *(str(path) for path in sources)],
            operation="merge",
        )
        run(["compact", "-o", str(destination), str(merged)], operation="compaction")
        run(["verify", str(destination)], operation="verification")
    finally:
        merged.unlink(missing_ok=True)
    return destination


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
            image = np.asarray(frame["image"])
            agent_history.setdefault(episode, []).append(
                workspace_points_to_image(
                    [frame["agent_position"]],
                    image=image,
                )[0]
            )
            action_history.setdefault(episode, []).append(
                workspace_points_to_image(
                    [frame["action_target"]],
                    image=image,
                )[0]
            )
            root = f"episodes/episode_{episode:03d}"
            recording.log(f"{root}/image", rr.Image(image).compress())
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
    sampling_protocol: str,
    recording_id: str | None = None,
    recording_name: str | None = None,
) -> Path:
    if not candidates:
        raise ValueError("Cannot record an empty fixed-observation candidate set")
    try:
        protocol = FIXED_OBSERVATION_PROTOCOLS[sampling_protocol]
    except KeyError as exc:
        raise ValueError(f"Unknown E1 sampling protocol: {sampling_protocol}") from exc
    ordinals = [int(candidate["candidate_ordinal"]) for candidate in candidates]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError(f"candidate_ordinal values must be unique, got {ordinals}")
    expected_ordinals = list(range(len(candidates)))
    if sorted(ordinals) != expected_ordinals:
        raise ValueError(
            "candidate_ordinal values must form a zero-based contiguous sequence; "
            f"expected {expected_ordinals}, got {sorted(ordinals)}"
        )
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: int(candidate["candidate_ordinal"]),
    )
    destination = Path(output_path)
    recording = _stream(
        destination,
        app_id=str(protocol["app_id"]),
        recording_id=(str(protocol["recording_id"]) if recording_id is None else recording_id),
    )
    try:
        if recording_name is not None:
            recording.send_recording_name(recording_name)
        recording.send_blueprint(
            build_fixed_observation_blueprint(sampling_protocol=sampling_protocol),
            make_active=True,
            make_default=True,
        )
        for candidate in ordered_candidates:
            candidate_ordinal = int(candidate["candidate_ordinal"])
            recording.set_time("candidate", sequence=candidate_ordinal)
            root = FIXED_OBSERVATION_ROOT
            current_image = np.asarray(candidate["current_image"])
            bc_root = f"{root}/bc"
            diffusion_root = f"{root}/dp"
            recording.log(
                f"{bc_root}/image",
                rr.Image(current_image).compress(),
            )
            recording.log(
                f"{diffusion_root}/image",
                rr.Image(current_image).compress(),
            )
            agent = workspace_points_to_image(
                [candidate["agent_position"]],
                image=current_image,
            )[0]
            raw_bc_actions = np.asarray(candidate["bc_actions"], dtype=np.float32)
            raw_diffusion_actions = np.asarray(candidate["diffusion_actions"], dtype=np.float32)
            bc_actions = workspace_points_to_image(
                raw_bc_actions,
                image=current_image,
            )
            diffusion_actions = workspace_points_to_image(
                raw_diffusion_actions,
                image=current_image,
            )
            if bc_actions.ndim != 3 or bc_actions.shape[0] != 3 or bc_actions.shape[-1] != 2:
                raise ValueError(
                    f"bc_actions must have shape [3, num_steps, 2], got {bc_actions.shape}"
                )
            if diffusion_actions.ndim != 3 or diffusion_actions.shape[-1] != 2:
                raise ValueError(
                    "diffusion_actions must have shape [num_samples, num_steps, 2], "
                    f"got {diffusion_actions.shape}"
                )
            if bc_actions.shape[1:] != diffusion_actions.shape[1:]:
                raise ValueError(
                    "BCChunk and Diffusion predictions must use the same action shape; "
                    f"got {bc_actions.shape[1:]} and {diffusion_actions.shape[1:]}"
                )
            if not np.isfinite(bc_actions).all() or not np.isfinite(diffusion_actions).all():
                raise ValueError(
                    f"Candidate {candidate['candidate_id']} contains non-finite action targets"
                )
            for panel_root in (bc_root, diffusion_root):
                recording.log(
                    f"{panel_root}/agent",
                    rr.Points2D(
                        [agent],
                        colors=[AGENT_COLOR],
                        radii=rr.Radius.ui_points(6.0),
                        draw_order=20.0,
                    ),
                )
            for sample_index, actions in enumerate(bc_actions):
                _log_predicted_action_plan(
                    recording,
                    entity_root=f"{bc_root}/pred_{sample_index:02d}",
                    agent=agent,
                    actions=actions,
                    color=BC_SAMPLE_COLORS[sample_index],
                    label=f"BCChunk repeat {sample_index}",
                    line_radius_ui_points=2.5,
                )
            for sample_index, actions in enumerate(diffusion_actions):
                _log_predicted_action_plan(
                    recording,
                    entity_root=f"{diffusion_root}/pred_{sample_index:02d}",
                    agent=agent,
                    actions=actions,
                    color=SAMPLE_COLORS[sample_index % len(SAMPLE_COLORS)],
                    label=f"{protocol['title']} sample {sample_index}",
                    line_radius_ui_points=1.5,
                )
    finally:
        recording.disconnect()
    return destination


def _log_predicted_action_plan(
    recording: rr.RecordingStream,
    *,
    entity_root: str,
    agent: np.ndarray,
    actions: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    line_radius_ui_points: float,
) -> None:
    """Log one selectable E1 action plan without implying that it was executed."""

    action_points = np.asarray(actions)
    if action_points.ndim != 2 or action_points.shape[-1] != 2 or len(action_points) == 0:
        raise ValueError(
            f"actions must have non-empty shape [num_steps, 2], got {action_points.shape}"
        )
    agent_point = np.asarray(agent)
    if agent_point.shape != (2,):
        raise ValueError(f"agent must have shape [2], got {agent_point.shape}")

    path = np.concatenate([agent_point[None], action_points], axis=0)
    step_labels = [f"{label} · step {step_index}" for step_index in range(len(action_points))]
    recording.log(
        f"{entity_root}/path",
        rr.LineStrips2D(
            [path],
            radii=rr.Radius.ui_points(line_radius_ui_points),
            colors=[color],
            labels=[f"{label} · predicted plan (not executed)"],
            show_labels=False,
            draw_order=10.0,
        ),
    )
    recording.log(
        f"{entity_root}/steps",
        rr.Points2D(
            action_points,
            colors=[color],
            radii=rr.Radius.ui_points(2.5),
            labels=step_labels,
            show_labels=False,
            draw_order=11.0,
        ),
    )
    recording.log(
        f"{entity_root}/first_action",
        rr.Points2D(
            [action_points[0]],
            colors=[color],
            radii=rr.Radius.ui_points(5.0),
            labels=[f"{label} · step 0 (first action target)"],
            show_labels=False,
            draw_order=12.0,
        ),
    )
    recording.log(
        f"{entity_root}/endpoint",
        rr.Points2D(
            [action_points[-1]],
            colors=[color],
            radii=rr.Radius.ui_points(4.0),
            labels=[f"{label} · step {len(action_points) - 1} (endpoint)"],
            show_labels=False,
            draw_order=12.0,
        ),
    )


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


def record_single_rollout(
    *,
    output_path: str | Path,
    rollout: RolloutResult,
    recording_id: str,
    recording_name: str,
) -> Path:
    """Write one independently selectable complete closed-loop rollout."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Rollout Recording already exists: {destination}")
    recording = rr.RecordingStream(
        "robot_practice_pusht_e2_rollouts",
        recording_id=recording_id,
    )
    try:
        recording.save(destination)
        recording.send_recording_name(recording_name)
        rr.send_blueprint(
            build_development_rollout_blueprint(),
            make_active=True,
            make_default=True,
            recording=recording,
        )
        next_frame = log_rollout_to_stream(
            recording,
            rollout,
            global_frame_start=0,
            rollout_index=0,
            entity_root=DEVELOPMENT_ROLLOUT_ROOT,
        )
        if next_frame != rollout.agent_path.shape[0]:
            raise RuntimeError("Rollout Recording timeline length is inconsistent")
        recording.flush()
    finally:
        recording.disconnect()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Rollout Recording was not written: {destination}")
    return destination


def rollout_entity_path(rollout: RolloutResult) -> str:
    """Return the canonical entity root for one PushT rollout."""

    return (
        f"rollouts/{rollout.condition_id}/{rollout.model}"
        f"/env_{rollout.environment_seed:04d}"
        f"/policy_{rollout.policy_seed:02d}"
    )


def log_rollout_to_stream(
    recording: rr.RecordingStream,
    rollout: RolloutResult,
    *,
    global_frame_start: int,
    rollout_index: int,
    entity_root: str | None = None,
) -> int:
    """Log one rollout and return the next unused ``global_frame`` value."""

    if global_frame_start < 0:
        raise ValueError("global_frame_start must be non-negative")
    if rollout_index < 0:
        raise ValueError("rollout_index must be non-negative")
    root = rollout_entity_path(rollout) if entity_root is None else entity_root.rstrip("/")
    if not root:
        raise ValueError("entity_root must not be empty")
    agent_path = np.asarray(rollout.agent_path)
    block_path = np.asarray(rollout.block_path[:, :2])
    action_targets = np.asarray(rollout.action_targets)
    if rollout.frames:
        first_frame = rollout.frames[0]
        agent_path = workspace_points_to_image(agent_path, image=first_frame)
        block_path = workspace_points_to_image(block_path, image=first_frame)
        action_targets = workspace_points_to_image(
            action_targets,
            image=first_frame,
        )
    global_frame = global_frame_start
    for frame_index in range(rollout.agent_path.shape[0]):
        recording.set_time("global_frame", sequence=global_frame)
        global_frame += 1
        if frame_index == 0:
            recording.log(
                f"{root}/complete_agent_path",
                rr.LineStrips2D(
                    [agent_path],
                    colors=[AGENT_COLOR],
                    labels=["complete measured agent path"],
                    show_labels=False,
                ),
            )
            recording.log(
                f"{root}/complete_block_path",
                rr.LineStrips2D(
                    [block_path],
                    colors=[BLOCK_COLOR],
                    labels=["complete measured block path"],
                    show_labels=False,
                ),
            )
            if action_targets.size:
                recording.log(
                    f"{root}/complete_action_targets",
                    rr.LineStrips2D(
                        [action_targets],
                        colors=[ACTION_COLOR],
                        labels=["complete absolute target path"],
                        show_labels=False,
                    ),
                    static=True,
                )
        if frame_index < len(rollout.frames):
            recording.log(
                f"{root}/simulator",
                rr.Image(rollout.frames[frame_index]).compress(),
            )
        recording.log(
            f"{root}/current_agent",
            rr.Points2D(
                [agent_path[frame_index]],
                colors=[AGENT_COLOR],
                radii=6.0,
                labels=["current measured agent"],
                show_labels=False,
            ),
        )
        recording.log(
            f"{root}/current_block",
            rr.Points2D(
                [block_path[frame_index]],
                colors=[BLOCK_COLOR],
                radii=6.0,
                labels=["current measured block"],
                show_labels=False,
            ),
        )
        if frame_index > 0:
            recording.log(
                f"{root}/current_action_target",
                rr.Points2D(
                    [action_targets[frame_index - 1]],
                    colors=[ACTION_COLOR],
                    radii=4.0,
                    labels=["current absolute action target"],
                    show_labels=False,
                ),
            )
            recording.log(
                f"{root}/reward",
                rr.Scalars(float(rollout.rewards[frame_index - 1])),
            )
    recording.log(
        f"{root}/success",
        rr.Scalars(int(rollout.success)),
        static=True,
    )
    recording.log(
        f"{root}/rollout_index",
        rr.Scalars(rollout_index),
        static=True,
    )
    recording.log(
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
        static=True,
    )
    return global_frame


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
        self.global_frame = log_rollout_to_stream(
            self.recording,
            rollout,
            global_frame_start=self.global_frame,
            rollout_index=self.rollout_index,
        )
        self.rollout_index += 1

    def close(self) -> None:
        if not self._closed:
            self.recording.disconnect()
            self._closed = True


class E3SceneRecording:
    """Incrementally write all sampling conditions for one E3 environment."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        environment_seed: int,
        condition_ids: Sequence[str],
    ) -> None:
        self.destination = Path(output_path)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise FileExistsError(f"E3 scene Recording already exists: {self.destination}")
        self.environment_seed = int(environment_seed)
        self.condition_ids = tuple(str(value) for value in condition_ids)
        if not self.condition_ids or len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("E3 condition_ids must be non-empty and unique")
        self._seen: set[str] = set()
        self._closed = False
        self.recording = rr.RecordingStream(
            "robot_practice_pusht_e3_sampling_steps",
            recording_id=f"e3_env_{self.environment_seed:04d}",
        )
        self.recording.save(self.destination)
        self.recording.send_recording_name(
            f"E3 | env {self.environment_seed} | {len(self.condition_ids)} sampling conditions"
        )
        self.recording.log(
            "/task/environment_seed",
            rr.Scalars(self.environment_seed),
            static=True,
        )
        rr.send_blueprint(
            build_e3_sampling_steps_blueprint(self.condition_ids),
            make_active=True,
            make_default=True,
            recording=self.recording,
        )

    def add(self, rollout: RolloutResult) -> None:
        if self._closed:
            raise RuntimeError("Cannot add an E3 condition after closing the recording")
        if rollout.environment_seed != self.environment_seed:
            raise ValueError(
                "E3 scene Recording environment mismatch: "
                f"{rollout.environment_seed} != {self.environment_seed}"
            )
        if rollout.condition_id not in self.condition_ids:
            raise ValueError(f"Unexpected E3 condition: {rollout.condition_id}")
        if rollout.condition_id in self._seen:
            raise ValueError(f"Duplicate E3 condition: {rollout.condition_id}")
        next_frame = log_rollout_to_stream(
            self.recording,
            rollout,
            global_frame_start=0,
            rollout_index=self.condition_ids.index(rollout.condition_id),
            entity_root=f"{E3_COMPARISON_ROOT}/{rollout.condition_id}",
        )
        if next_frame != rollout.agent_path.shape[0]:
            raise RuntimeError("E3 condition timeline length is inconsistent")
        self._seen.add(rollout.condition_id)

    def close(self, *, validate_complete: bool = True) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        if validate_complete and self._seen != set(self.condition_ids):
            missing = sorted(set(self.condition_ids) - self._seen)
            error = RuntimeError(f"E3 scene Recording is missing conditions: {missing}")
        try:
            self.recording.flush()
        finally:
            self.recording.disconnect()
            self._closed = True
        if error is not None:
            raise error
        if not self.destination.is_file() or self.destination.stat().st_size == 0:
            raise RuntimeError(f"E3 scene Recording was not written: {self.destination}")
