"""Simulator-grounded goal classification for the LIBERO bowl prompt-swap test."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym

GOAL_CLASSIFIER_NAME = "libero_bowl_three_goal_predicate_v1"
GOAL_LABELS = ("stove", "cabinet", "plate")
FAILED_GOAL = "failed"
MULTI_GOAL_CONFLICT = "multi_goal_conflict"

# These predicates are present together in each of the three LIBERO-Goal bowl
# tasks. Evaluating all of them against one simulator state avoids using the
# reference task's single-goal check_success() as the prompt-swap classifier.
GOAL_PREDICATES: Mapping[str, tuple[str, str, str]] = {
    "stove": ("on", "akita_black_bowl_1", "flat_stove_1_cook_region"),
    "cabinet": ("on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"),
    "plate": ("on", "akita_black_bowl_1", "plate_1"),
}


def _raw_libero_environment(env: gym.vector.VectorEnv) -> Any:
    if getattr(env, "num_envs", None) != 1 or not hasattr(env, "envs"):
        raise RuntimeError("Prompt Swap requires one synchronous LIBERO sub-environment")
    wrapper = env.envs[0]
    bridge = getattr(wrapper, "_env", None)
    candidates = (bridge, getattr(bridge, "env", None))
    raw = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None
            and callable(getattr(candidate, "_eval_predicate", None))
        ),
        None,
    )
    if raw is None:
        raise RuntimeError("LIBERO environment does not expose predicate evaluation")
    object_states = getattr(raw, "object_states_dict", None)
    if not isinstance(object_states, dict):
        raise TypeError("LIBERO environment does not expose object_states_dict")
    required_objects = {
        object_name for predicate in GOAL_PREDICATES.values() for object_name in predicate[1:]
    }
    missing = sorted(required_objects - set(object_states))
    if missing:
        raise RuntimeError(
            "Prompt Swap reference scene is missing classifier objects: " + ", ".join(missing)
        )
    return raw


def disable_reference_goal_termination(env: gym.vector.VectorEnv) -> None:
    """Prevent the reference task goal from resetting a Prompt Swap rollout."""

    _raw_libero_environment(env)
    bridge = getattr(env.envs[0], "_env", None)
    if not callable(getattr(bridge, "check_success", None)):
        raise TypeError("LIBERO environment does not expose check_success")
    # LeRobot's LIBERO wrapper calls this public method after every raw step and
    # immediately resets on True. The classifier below calls _eval_predicate
    # directly, so disabling this one method does not disable terminal scoring.
    bridge.check_success = lambda: False


def classify_bowl_goal(env: gym.vector.VectorEnv) -> dict[str, Any]:
    """Classify one terminal simulator state into exactly one target or Failed."""

    raw = _raw_libero_environment(env)
    predicates = {
        goal: bool(raw._eval_predicate(list(predicate)))
        for goal, predicate in GOAL_PREDICATES.items()
    }
    matches = [goal for goal in GOAL_LABELS if predicates[goal]]
    if len(matches) == 1:
        predicted_goal = matches[0]
        failure_reason = None
    elif not matches:
        predicted_goal = FAILED_GOAL
        failure_reason = "no_goal_reached"
    else:
        predicted_goal = FAILED_GOAL
        failure_reason = MULTI_GOAL_CONFLICT
    return {
        "classifier": GOAL_CLASSIFIER_NAME,
        "predicted_goal": predicted_goal,
        "failure_reason": failure_reason,
        "predicate_matches": predicates,
    }


__all__ = [
    "FAILED_GOAL",
    "GOAL_CLASSIFIER_NAME",
    "GOAL_LABELS",
    "GOAL_PREDICATES",
    "classify_bowl_goal",
    "disable_reference_goal_termination",
]
