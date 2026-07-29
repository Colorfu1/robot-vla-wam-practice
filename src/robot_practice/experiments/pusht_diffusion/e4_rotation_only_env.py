"""Deterministic, center-aligned PushT scene for the E4 rotation-only smoke check."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely

from .config import default_output_root
from .output import AtomicOutputDirectory
from .provenance import write_json_atomic

E4_ROTATION_CENTER = np.array([256.0, 256.0], dtype=np.float64)
E4_TEE_CENTROID_LOCAL = np.array([0.0, 285.0 / 7.0], dtype=np.float64)
E4_START_ANGLE = 0.0
E4_GOAL_ANGLE = float(np.pi)
E4_AGENT_RADIUS = 15.0
E4_AGENT_CLEARANCE = 1.0
E4_EXPECTED_INITIAL_COVERAGE = 19.0 / 49.0
E4_SAFE_AGENT_POSITION = np.array([50.0, 50.0], dtype=np.float64)


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return rotation @ np.asarray(vector, dtype=np.float64)


def _body_origin_for_center(angle: float) -> np.ndarray:
    return E4_ROTATION_CENTER - _rotate(E4_TEE_CENTROID_LOCAL, angle)


E4_START_BODY_ORIGIN = _body_origin_for_center(E4_START_ANGLE)
E4_GOAL_BODY_ORIGIN = _body_origin_for_center(E4_GOAL_ANGLE)
E4_AGENT_START = E4_START_BODY_ORIGIN + _rotate(
    np.array([0.0, -(E4_AGENT_RADIUS + E4_AGENT_CLEARANCE)], dtype=np.float64),
    E4_START_ANGLE,
)
E4_RESET_STATE = np.array(
    [
        *E4_AGENT_START,
        *E4_START_BODY_ORIGIN,
        E4_START_ANGLE,
    ],
    dtype=np.float64,
)
E4_GOAL_POSE = np.array(
    [
        *E4_GOAL_BODY_ORIGIN,
        E4_GOAL_ANGLE,
    ],
    dtype=np.float64,
)
E4_LOWER_LEFT_START_CENTER = np.array([128.0, 384.0], dtype=np.float64)
E4_LOWER_LEFT_GOAL_CENTER = E4_ROTATION_CENTER.copy()
E4_LOWER_LEFT_START_ANGLE = float(-3.0 * np.pi / 4.0)
E4_LOWER_LEFT_GOAL_ANGLE = float(np.pi / 4.0)
E4_LOWER_LEFT_CONTACT_TANGENT = _rotate(
    np.array([-1.0, 0.0], dtype=np.float64),
    E4_LOWER_LEFT_START_ANGLE,
)
E4_LOWER_LEFT_MAX_CONTACT_OFFSET_PX = 40.0
E4_LOWER_LEFT_START_BODY_ORIGIN = E4_LOWER_LEFT_START_CENTER - _rotate(
    E4_TEE_CENTROID_LOCAL, E4_LOWER_LEFT_START_ANGLE
)
E4_LOWER_LEFT_GOAL_BODY_ORIGIN = E4_LOWER_LEFT_GOAL_CENTER - _rotate(
    E4_TEE_CENTROID_LOCAL, E4_LOWER_LEFT_GOAL_ANGLE
)
E4_LOWER_LEFT_AGENT_START = E4_LOWER_LEFT_START_BODY_ORIGIN + _rotate(
    np.array([0.0, -(E4_AGENT_RADIUS + E4_AGENT_CLEARANCE)], dtype=np.float64),
    E4_LOWER_LEFT_START_ANGLE,
)
E4_LOWER_LEFT_RESET_STATE = np.array(
    [
        *E4_LOWER_LEFT_AGENT_START,
        *E4_LOWER_LEFT_START_BODY_ORIGIN,
        E4_LOWER_LEFT_START_ANGLE,
    ],
    dtype=np.float64,
)
E4_LOWER_LEFT_GOAL_POSE = np.array(
    [
        *E4_LOWER_LEFT_GOAL_BODY_ORIGIN,
        E4_LOWER_LEFT_GOAL_ANGLE,
    ],
    dtype=np.float64,
)


def e4_lower_left_agent_start(contact_offset_px: float = 0.0) -> np.ndarray:
    """Return the agent start shifted along the crossbar toward rendered lower-right."""

    offset = float(contact_offset_px)
    if not np.isfinite(offset):
        raise ValueError("contact_offset_px must be finite")
    if abs(offset) > E4_LOWER_LEFT_MAX_CONTACT_OFFSET_PX:
        raise ValueError(
            "contact_offset_px must stay within "
            f"[-{E4_LOWER_LEFT_MAX_CONTACT_OFFSET_PX:g}, "
            f"{E4_LOWER_LEFT_MAX_CONTACT_OFFSET_PX:g}]"
        )
    return E4_LOWER_LEFT_AGENT_START + offset * E4_LOWER_LEFT_CONTACT_TANGENT


def e4_lower_left_reset_state(contact_offset_px: float = 0.0) -> np.ndarray:
    """Return the fixed block pose with a configurable crossbar contact point."""

    state = E4_LOWER_LEFT_RESET_STATE.copy()
    state[:2] = e4_lower_left_agent_start(contact_offset_px)
    return state


class E4RotationOnlyPushTEnv(PushTEnv):
    """PushT with a deterministic, center-aligned 180-degree target.

    The block remains a free rigid body. "Rotation-only" describes the relation
    between the initial and target poses; it does not add a pivot constraint.
    """

    reset_state = E4_RESET_STATE

    def __init__(
        self,
        *,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 96,
        observation_height: int = 96,
        visualization_width: int = 384,
        visualization_height: int = 384,
        damping: float | None = None,
    ) -> None:
        super().__init__(
            obs_type=obs_type,
            render_mode=render_mode,
            block_cog=tuple(E4_TEE_CENTROID_LOCAL),
            damping=damping,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
        )

    def _setup(self) -> None:
        super()._setup()
        self.goal_pose = E4_GOAL_POSE.copy()

    def _set_state(self, state: np.ndarray) -> None:
        """Apply a pose without Pymunk's legacy position-then-angle shift."""

        requested = np.asarray(state, dtype=np.float64)
        if requested.shape != (5,) or not np.isfinite(requested).all():
            raise ValueError(f"E4 state must contain five finite values, got {requested!r}")

        self.agent.position = requested[:2].tolist()
        self.agent.velocity = (0.0, 0.0)
        self.block.angle = float(requested[4])
        self.block.position = requested[2:4].tolist()
        self.block.velocity = (0.0, 0.0)
        self.block.angular_velocity = 0.0
        self.space.reindex_shapes_for_body(self.block)
        self.space.step(self.dt)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        requested_options = dict(options or {})
        requested_state = requested_options.get("reset_to_state")
        if requested_state is not None:
            requested = np.asarray(requested_state, dtype=np.float64)
            if requested.shape != self.reset_state.shape or not np.allclose(
                requested,
                self.reset_state,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "E4 rotation-only scene rejects reset_to_state values that change "
                    "its deterministic geometry"
                )
        requested_options["reset_to_state"] = self.reset_state.copy()
        return super().reset(seed=seed, options=requested_options)


class E4LowerLeftPushTEnv(E4RotationOnlyPushTEnv):
    """Displaced 180-degree task with the initial T in the rendered lower-left."""

    reset_state = E4_LOWER_LEFT_RESET_STATE

    def __init__(
        self,
        *,
        contact_offset_px: float = 0.0,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 96,
        observation_height: int = 96,
        visualization_width: int = 384,
        visualization_height: int = 384,
        damping: float | None = None,
    ) -> None:
        self.contact_offset_px = float(contact_offset_px)
        self.reset_state = e4_lower_left_reset_state(self.contact_offset_px)
        super().__init__(
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            damping=damping,
        )

    def _setup(self) -> None:
        super()._setup()
        self.goal_pose = E4_LOWER_LEFT_GOAL_POSE.copy()


def make_e4_rotation_only_env(
    *,
    episode_length: int = 300,
    render: bool = True,
    observation_width: int = 96,
    observation_height: int = 96,
    visualization_width: int = 384,
    visualization_height: int = 384,
) -> gym.Env:
    """Construct the repository-local E4 environment."""

    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    env = E4RotationOnlyPushTEnv(
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=observation_width,
        observation_height=observation_height,
        visualization_width=visualization_width,
        visualization_height=visualization_height,
    )
    if not render:
        env.visualization_width = observation_width
        env.visualization_height = observation_height
    return gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)


def make_e4_lower_left_env(
    *,
    episode_length: int = 300,
    render: bool = True,
    contact_offset_px: float = 0.0,
    observation_width: int = 96,
    observation_height: int = 96,
    visualization_width: int = 384,
    visualization_height: int = 384,
) -> gym.Env:
    """Construct the lower-left-to-center 180-degree scene."""

    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    env = E4LowerLeftPushTEnv(
        contact_offset_px=contact_offset_px,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=observation_width,
        observation_height=observation_height,
        visualization_width=visualization_width,
        visualization_height=visualization_height,
    )
    if not render:
        env.visualization_width = observation_width
        env.visualization_height = observation_height
    return gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)


def _xy(value: Any) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    return [float(array[0]), float(array[1])]


def _set_exact_goal_state(env: gym.Env) -> None:
    base = env.unwrapped
    base.agent.position = E4_SAFE_AGENT_POSITION.tolist()
    base.agent.velocity = (0.0, 0.0)
    base.block.angle = E4_GOAL_ANGLE
    base.block.position = E4_GOAL_BODY_ORIGIN.tolist()
    base.block.velocity = (0.0, 0.0)
    base.block.angular_velocity = 0.0
    base.space.reindex_shapes_for_body(base.block)


def measure_e4_scene() -> tuple[np.ndarray, dict[str, Any]]:
    """Measure the initial scene and inherited exact-goal success path."""

    initial_env = make_e4_rotation_only_env(render=True)
    try:
        _, reset_info = initial_env.reset(seed=0)
        base = initial_env.unwrapped
        frame = np.asarray(initial_env.render()).copy()
        block_geom = pymunk_to_shapely(base.block, base.block.shapes)
        goal_body = base.get_goal_pose_body(base.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, base.block.shapes)
        initial_block_centroid = np.array(
            [block_geom.centroid.x, block_geom.centroid.y],
            dtype=np.float64,
        )
        goal_centroid = np.array(
            [goal_geom.centroid.x, goal_geom.centroid.y],
            dtype=np.float64,
        )
        initial_cog_world = np.asarray(
            base.block.local_to_world(base.block.center_of_gravity),
            dtype=np.float64,
        )
        goal_cog_world = np.asarray(
            goal_body.local_to_world(tuple(E4_TEE_CENTROID_LOCAL)),
            dtype=np.float64,
        )
        _, initial_reward, initial_terminated, _, initial_info = initial_env.step(
            E4_AGENT_START.astype(np.float32)
        )
    finally:
        initial_env.close()

    final_env = make_e4_rotation_only_env(render=False)
    try:
        final_env.reset(seed=0)
        _set_exact_goal_state(final_env)
        final_base = final_env.unwrapped
        final_cog_world = np.asarray(
            final_base.block.local_to_world(final_base.block.center_of_gravity),
            dtype=np.float64,
        )
        _, final_reward, final_terminated, _, final_info = final_env.step(
            E4_SAFE_AGENT_POSITION.astype(np.float32)
        )
    finally:
        final_env.close()

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E4",
        "validation_kind": "Smoke / scene geometry",
        "rotation_center": _xy(E4_ROTATION_CENTER),
        "tee_center_of_gravity_local": _xy(E4_TEE_CENTROID_LOCAL),
        "start_body_origin": _xy(E4_START_BODY_ORIGIN),
        "goal_body_origin": _xy(E4_GOAL_BODY_ORIGIN),
        "start_angle_rad": E4_START_ANGLE,
        "goal_angle_rad": E4_GOAL_ANGLE,
        "agent_start": _xy(E4_AGENT_START),
        "agent_clearance_px": E4_AGENT_CLEARANCE,
        "initial_block_centroid": _xy(initial_block_centroid),
        "goal_centroid": _xy(goal_centroid),
        "initial_cog_world": _xy(initial_cog_world),
        "goal_cog_world": _xy(goal_cog_world),
        "final_cog_world": _xy(final_cog_world),
        "success_threshold": float(base.success_threshold),
        "initial_coverage": float(initial_info["coverage"]),
        "initial_reward": float(initial_reward),
        "initial_success": bool(initial_info["is_success"]),
        "initial_terminated": bool(initial_terminated),
        "initial_reset_contacts": int(reset_info["n_contacts"]),
        "exact_goal_coverage": float(final_info["coverage"]),
        "exact_goal_reward": float(final_reward),
        "exact_goal_success": bool(final_info["is_success"]),
        "exact_goal_terminated": bool(final_terminated),
    }
    _validate_e4_measurements(metrics)
    return frame, metrics


def _validate_e4_measurements(metrics: dict[str, Any]) -> None:
    for key in (
        "initial_block_centroid",
        "goal_centroid",
        "initial_cog_world",
        "goal_cog_world",
        "final_cog_world",
    ):
        if not np.allclose(metrics[key], E4_ROTATION_CENTER, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"E4 {key} is not center-aligned: {metrics[key]}")
    if not np.isclose(
        metrics["initial_coverage"],
        E4_EXPECTED_INITIAL_COVERAGE,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(f"Unexpected E4 initial coverage: {metrics['initial_coverage']}")
    if metrics["initial_success"] or metrics["initial_terminated"]:
        raise RuntimeError("E4 initial state must not be successful")
    if metrics["initial_reset_contacts"] != 0:
        raise RuntimeError("E4 initial state must not contain a block-agent contact")
    if not np.isclose(metrics["exact_goal_coverage"], 1.0, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"Unexpected E4 exact-goal coverage: {metrics['exact_goal_coverage']}")
    if metrics["exact_goal_reward"] != 1.0:
        raise RuntimeError(f"Unexpected E4 exact-goal reward: {metrics['exact_goal_reward']}")
    if not metrics["exact_goal_success"] or not metrics["exact_goal_terminated"]:
        raise RuntimeError("E4 exact-goal state must terminate successfully")


def write_e4_scene_preview(
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the ignored E4 scene preview and machine-readable smoke metrics."""

    destination = Path(output_dir).expanduser().resolve()
    frame, metrics = measure_e4_scene()
    with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
        preview_path = temporary / "scene_preview.png"
        converted = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(preview_path), converted):
            raise OSError(f"Could not write E4 scene preview to {preview_path}")
        write_json_atomic(temporary / "scene_metrics.json", metrics)
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and render the PushT E4 rotation-only scene."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e4_rotation_only_scene",
        help="Generated directory for scene_preview.png and scene_metrics.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = write_e4_scene_preview(args.output_dir, overwrite=args.overwrite)
    print(f"E4 rotation-only scene validated: {result}")


if __name__ == "__main__":
    main()
