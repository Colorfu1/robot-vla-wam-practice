from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymunk
import pytest
import torch
from gym_pusht.envs.pusht import pymunk_to_shapely
from shapely.geometry import Point

from robot_practice.experiments.pusht_diffusion import rollout as rollout_module
from robot_practice.experiments.pusht_diffusion.e4_diffusion_ood_rollout import (
    RESULT_LABEL,
    _aggregate_metrics,
    parse_policy_seeds,
)
from robot_practice.experiments.pusht_diffusion.e4_rotation_only_env import (
    E4_AGENT_RADIUS,
    E4_AGENT_START,
    E4_EXPECTED_INITIAL_COVERAGE,
    E4_GOAL_ANGLE,
    E4_GOAL_BODY_ORIGIN,
    E4_LOWER_LEFT_AGENT_START,
    E4_LOWER_LEFT_CONTACT_TANGENT,
    E4_LOWER_LEFT_GOAL_ANGLE,
    E4_LOWER_LEFT_GOAL_CENTER,
    E4_LOWER_LEFT_GOAL_POSE,
    E4_LOWER_LEFT_RESET_STATE,
    E4_LOWER_LEFT_START_ANGLE,
    E4_LOWER_LEFT_START_CENTER,
    E4_RESET_STATE,
    E4_ROTATION_CENTER,
    E4_SAFE_AGENT_POSITION,
    E4_TEE_CENTROID_LOCAL,
    e4_lower_left_agent_start,
    e4_lower_left_reset_state,
    make_e4_lower_left_env,
    make_e4_rotation_only_env,
    measure_e4_scene,
    write_e4_scene_preview,
)


def _centroid(geometry) -> np.ndarray:
    return np.array([geometry.centroid.x, geometry.centroid.y], dtype=np.float64)


def test_e4_tee_centroid_and_world_centers_are_exact() -> None:
    expected_centroid_y = (3600.0 * 15.0 + 2700.0 * 75.0) / (3600.0 + 2700.0)
    assert E4_TEE_CENTROID_LOCAL == pytest.approx([0.0, expected_centroid_y])
    assert expected_centroid_y == pytest.approx(285.0 / 7.0)

    env = make_e4_rotation_only_env()
    try:
        env.reset(seed=0)
        base = env.unwrapped
        assert tuple(base.block.center_of_gravity) == pytest.approx(tuple(E4_TEE_CENTROID_LOCAL))

        block_geometry = pymunk_to_shapely(base.block, base.block.shapes)
        goal_body = base.get_goal_pose_body(base.goal_pose)
        goal_geometry = pymunk_to_shapely(goal_body, base.block.shapes)

        assert _centroid(block_geometry) == pytest.approx(E4_ROTATION_CENTER)
        assert _centroid(goal_geometry) == pytest.approx(E4_ROTATION_CENTER)
        assert tuple(base.block.local_to_world(base.block.center_of_gravity)) == pytest.approx(
            E4_ROTATION_CENTER
        )
        assert tuple(goal_body.local_to_world(tuple(E4_TEE_CENTROID_LOCAL))) == pytest.approx(
            E4_ROTATION_CENTER
        )
    finally:
        env.close()


def test_e4_reset_is_seed_invariant_and_rejects_conflicting_state() -> None:
    env = make_e4_rotation_only_env()
    try:
        first_observation, first_info = env.reset(seed=1)
        second_observation, second_info = env.reset(seed=999)

        assert first_observation["agent_pos"] == pytest.approx(second_observation["agent_pos"])
        assert np.array_equal(first_observation["pixels"], second_observation["pixels"])
        assert first_info["block_pose"] == pytest.approx(second_info["block_pose"])
        assert first_info["goal_pose"] == pytest.approx(second_info["goal_pose"])

        matching, _ = env.reset(options={"reset_to_state": E4_RESET_STATE.copy()})
        assert matching["agent_pos"] == pytest.approx(E4_AGENT_START)

        conflicting = E4_RESET_STATE.copy()
        conflicting[0] += 1.0
        with pytest.raises(ValueError, match="rejects reset_to_state"):
            env.reset(options={"reset_to_state": conflicting})
    finally:
        env.close()


def test_e4_block_is_free_and_agent_starts_with_one_pixel_clearance() -> None:
    env = make_e4_rotation_only_env()
    try:
        _, info = env.reset(seed=0)
        base = env.unwrapped
        block_geometry = pymunk_to_shapely(base.block, base.block.shapes)
        center_to_block = block_geometry.distance(Point(*base.agent.position))

        assert base.block.body_type == pymunk.Body.DYNAMIC
        assert len(base.space.constraints) == 0
        assert center_to_block - E4_AGENT_RADIUS == pytest.approx(1.0)
        assert info["n_contacts"] == 0
    finally:
        env.close()


def test_e4_lower_left_scene_matches_displaced_180_degree_layout() -> None:
    env = make_e4_lower_left_env()
    try:
        observation, info = env.reset(seed=0)
        base = env.unwrapped
        block_geometry = pymunk_to_shapely(base.block, base.block.shapes)
        goal_body = base.get_goal_pose_body(base.goal_pose)
        goal_geometry = pymunk_to_shapely(goal_body, base.block.shapes)
        center_to_block = block_geometry.distance(Point(*base.agent.position))

        assert _centroid(block_geometry) == pytest.approx(E4_LOWER_LEFT_START_CENTER)
        assert _centroid(goal_geometry) == pytest.approx(E4_LOWER_LEFT_GOAL_CENTER)
        assert observation["agent_pos"] == pytest.approx(E4_LOWER_LEFT_AGENT_START)
        assert info["block_pose"] == pytest.approx(E4_LOWER_LEFT_RESET_STATE[2:])
        assert center_to_block - E4_AGENT_RADIUS == pytest.approx(1.0)
        assert base.block.center_of_gravity == pytest.approx(E4_TEE_CENTROID_LOCAL)
        assert base.goal_pose == pytest.approx(E4_LOWER_LEFT_GOAL_POSE)
        assert base.block.angle == pytest.approx(E4_LOWER_LEFT_START_ANGLE)
        assert base.goal_pose[2] == pytest.approx(E4_LOWER_LEFT_GOAL_ANGLE)
        assert (E4_LOWER_LEFT_GOAL_ANGLE - E4_LOWER_LEFT_START_ANGLE) % (
            2.0 * np.pi
        ) == pytest.approx(np.pi)
        assert block_geometry.intersection(goal_geometry).area == pytest.approx(0.0)
        assert block_geometry.distance(goal_geometry) > 0.0
        assert len(base.space.constraints) == 0
    finally:
        env.close()


@pytest.mark.parametrize("contact_offset_px", [-20.0, 0.0, 20.0])
def test_e4_lower_left_contact_offset_moves_only_agent_along_crossbar(
    contact_offset_px: float,
) -> None:
    env = make_e4_lower_left_env(contact_offset_px=contact_offset_px)
    try:
        observation, info = env.reset(seed=0)
        base = env.unwrapped
        block_geometry = pymunk_to_shapely(base.block, base.block.shapes)
        expected_agent = (
            E4_LOWER_LEFT_AGENT_START
            + contact_offset_px * E4_LOWER_LEFT_CONTACT_TANGENT
        )

        assert observation["agent_pos"] == pytest.approx(expected_agent)
        assert observation["agent_pos"] == pytest.approx(
            e4_lower_left_agent_start(contact_offset_px)
        )
        assert info["block_pose"] == pytest.approx(E4_LOWER_LEFT_RESET_STATE[2:])
        assert base.reset_state == pytest.approx(e4_lower_left_reset_state(contact_offset_px))
        assert block_geometry.distance(Point(*base.agent.position)) - E4_AGENT_RADIUS == (
            pytest.approx(1.0)
        )
        assert info["n_contacts"] == 0
    finally:
        env.close()


def test_e4_lower_left_contact_offset_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        make_e4_lower_left_env(contact_offset_px=float("nan"))
    with pytest.raises(ValueError, match="within"):
        make_e4_lower_left_env(contact_offset_px=41.0)


def test_e4_initial_and_exact_goal_use_inherited_success_logic() -> None:
    env = make_e4_rotation_only_env()
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(E4_AGENT_START.astype(np.float32))

        assert info["coverage"] == pytest.approx(E4_EXPECTED_INITIAL_COVERAGE)
        assert reward == pytest.approx(E4_EXPECTED_INITIAL_COVERAGE / 0.95)
        assert info["is_success"] is False
        assert terminated is False
        assert truncated is False

        base = env.unwrapped
        base.agent.position = E4_SAFE_AGENT_POSITION.tolist()
        base.agent.velocity = (0.0, 0.0)
        base.block.angle = E4_GOAL_ANGLE
        base.block.position = E4_GOAL_BODY_ORIGIN.tolist()
        base.block.velocity = (0.0, 0.0)
        base.block.angular_velocity = 0.0
        base.space.reindex_shapes_for_body(base.block)

        _, reward, terminated, truncated, info = env.step(E4_SAFE_AGENT_POSITION.astype(np.float32))

        assert tuple(base.block.local_to_world(base.block.center_of_gravity)) == pytest.approx(
            E4_ROTATION_CENTER
        )
        assert info["coverage"] == pytest.approx(1.0)
        assert reward == pytest.approx(1.0)
        assert info["is_success"] is True
        assert terminated is True
        assert truncated is False
    finally:
        env.close()


def test_e4_observation_action_and_render_interfaces_match_pusht() -> None:
    env = make_e4_rotation_only_env(
        observation_width=96,
        observation_height=96,
        visualization_width=384,
        visualization_height=384,
    )
    try:
        observation, _ = env.reset(seed=0)
        frame = env.render()

        assert set(observation) == {"pixels", "agent_pos"}
        assert observation["pixels"].shape == (96, 96, 3)
        assert observation["agent_pos"].shape == (2,)
        assert env.action_space.shape == (2,)
        assert frame.shape == (384, 384, 3)
        assert frame.dtype == np.uint8
        assert np.ptp(frame) > 0
    finally:
        env.close()


def test_e4_measurement_and_preview_outputs(tmp_path: Path) -> None:
    frame, metrics = measure_e4_scene()

    assert frame.shape == (384, 384, 3)
    assert metrics["validation_kind"] == "Smoke / scene geometry"
    assert metrics["initial_coverage"] == pytest.approx(E4_EXPECTED_INITIAL_COVERAGE)
    assert metrics["initial_success"] is False
    assert metrics["exact_goal_coverage"] == pytest.approx(1.0)
    assert metrics["exact_goal_success"] is True

    destination = write_e4_scene_preview(tmp_path / "preview")
    assert destination == (tmp_path / "preview").resolve()
    assert (destination / "scene_preview.png").stat().st_size > 0
    payload = json.loads((destination / "scene_metrics.json").read_text(encoding="utf-8"))
    assert payload == metrics


def test_run_rollout_accepts_an_injected_environment_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakePolicy:
        config = type("Config", (), {"type": "diffusion"})()

    class FakeExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            self.replans = []

        def select_action(self, _observation):
            return torch.tensor([[20.0, 30.0]])

    class FakeEnvironment:
        def __init__(self) -> None:
            self.closed = False

        @staticmethod
        def _observation() -> dict[str, np.ndarray]:
            return {
                "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
                "agent_pos": np.array([10.0, 10.0]),
            }

        def reset(self, *, seed: int):
            calls["reset_seed"] = seed
            return self._observation(), {
                "pos_agent": np.array([10.0, 10.0]),
                "block_pose": np.array([20.0, 20.0, 0.0]),
            }

        def step(self, action: np.ndarray):
            calls["action"] = action.copy()
            return (
                self._observation(),
                0.5,
                True,
                False,
                {
                    "pos_agent": np.array([11.0, 11.0]),
                    "block_pose": np.array([20.0, 20.0, 0.1]),
                    "is_success": True,
                },
            )

        def close(self) -> None:
            self.closed = True
            calls["closed"] = True

    def factory(*, episode_length: int, render: bool):
        calls["factory"] = (episode_length, render)
        return FakeEnvironment()

    monkeypatch.setattr(rollout_module, "ExplicitChunkExecutor", FakeExecutor)
    monkeypatch.setattr(rollout_module, "preprocess_observation", lambda value: value)

    result = rollout_module.run_rollout(
        policy=FakePolicy(),
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        environment_seed=0,
        policy_seed=4,
        record_frames=False,
        episode_length=7,
        environment_factory=factory,
    )

    assert calls["factory"] == (7, False)
    assert calls["reset_seed"] == 0
    assert calls["action"] == pytest.approx([20.0, 30.0])
    assert calls["closed"] is True
    assert result.success is True
    assert result.episode_length == 1


def test_e4_ood_policy_seed_validation_and_diagnostic_summary() -> None:
    assert parse_policy_seeds("0,1,2,3,4") == (0, 1, 2, 3, 4)
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        parse_policy_seeds("0,0")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        parse_policy_seeds("0,-1")

    metrics = _aggregate_metrics(
        [
            {"policy_seed": 0, "success": False, "max_reward": 0.4, "final_reward": 0.2},
            {"policy_seed": 1, "success": True, "max_reward": 1.0, "final_reward": 1.0},
        ],
        contact_offset_px=20.0,
    )
    assert metrics["label"] == RESULT_LABEL
    assert metrics["contact_offset_px"] == pytest.approx(20.0)
    assert metrics["num_rollouts"] == 2
    assert metrics["diagnostic_success_rate"] == pytest.approx(0.5)
