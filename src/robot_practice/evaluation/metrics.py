"""Pure NumPy metrics shared by offline and closed-loop ACT evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ACTION_DIMENSION = 7
ARM_DIMENSIONS = 6

OFFLINE_METRIC_DEFINITIONS = {
    "num_episodes": "参与离线评测的 demonstration episode 数量",
    "num_frames": "参与离线预测的数据帧数量",
    "num_model_queries": "实际调用 predict_action_chunk 的批次数量",
    "batch_size": "每次离线模型调用的配置 batch size",
    "first_action_mae": "当前步预测动作与 demonstration 动作的平均绝对误差",
    "per_dimension_mae": "当前步各动作维度的平均绝对误差",
    "chunk_mae_by_horizon": "每个未来 action horizon 上的平均绝对误差",
    "delta_action_mae": "预测动作变化与 demonstration 动作变化的平均绝对误差",
    "teacher_forced_action_tv": "teacher-forced 预测中相邻机械臂动作的平均变化量",
    "teacher_forced_action_jerk": "teacher-forced 预测动作的二阶差分抖动代理",
    "inference_latency_p50_ms": "一半离线模型批调用不超过的推理耗时",
    "inference_latency_p95_ms": "95% 离线模型批调用不超过的推理耗时",
}

CLOSED_LOOP_METRIC_DEFINITIONS = {
    "is_formal": (
        "是否使用所选 LIBERO suite 的完整步数上限，可作为正式闭环评测"
    ),
    "num_episodes": "参与闭环评测的 rollout 数量",
    "num_successes": "由 LIBERO final_info 判定成功的 rollout 数量",
    "success_rate": "成功完成任务的 rollout 比例",
    "avg_sum_reward": "每个 rollout 累计 reward 的平均值",
    "avg_max_reward": "每个 rollout 最大单步 reward 的平均值",
    "action_tv": "相邻执行动作的平均变化量，越低通常越平滑",
    "action_jerk": "动作二阶差分，反映动作突变和来回抖动",
    "mean_gripper_toggles": "每个 rollout 夹爪开合状态切换次数的平均值",
    "gripper_toggles": "夹爪开合状态发生切换的次数",
    "policy_step_latency_p50_ms": "一半控制步骤不超过的策略调用耗时",
    "policy_step_latency_p95_ms": "95% 控制步骤不超过的策略调用耗时",
    "model_query_latency_p50_ms": "一半神经网络实际推理不超过的耗时",
    "model_query_latency_p95_ms": "95% 神经网络实际推理不超过的耗时",
    "num_policy_steps": "实际发送给模拟器的控制步骤数量",
    "num_model_queries": "ACT action queue 为空时实际调用模型的次数",
}


def _float_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Infinity")
    return array


def _validate_chunks(
    predicted: ArrayLike,
    target: ArrayLike,
    valid_horizon: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    predicted_array = _float_array(predicted, name="predicted chunks")
    target_array = _float_array(target, name="target chunks")
    valid_array = np.asarray(valid_horizon, dtype=np.bool_)
    if predicted_array.shape != target_array.shape:
        raise ValueError(
            "predicted and target chunks must have identical shapes; "
            f"got {predicted_array.shape} and {target_array.shape}"
        )
    if predicted_array.ndim != 3 or predicted_array.shape[-1] != ACTION_DIMENSION:
        raise ValueError(
            "action chunks must have shape [frames, horizon, 7]; "
            f"got {predicted_array.shape}"
        )
    if valid_array.shape != predicted_array.shape[:2]:
        raise ValueError(
            "valid_horizon must have shape [frames, horizon]; "
            f"got {valid_array.shape} for chunks {predicted_array.shape}"
        )
    return predicted_array, target_array, valid_array


def _mean_or_none(values: ArrayLike) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if not np.isfinite(array).all():
        raise ValueError("metric inputs contain NaN or Infinity")
    return float(array.mean())


def macro_mean(values: Sequence[float | None]) -> float | None:
    """Average defined episode metrics without weighting longer episodes more heavily."""

    defined = [value for value in values if value is not None]
    return _mean_or_none(defined)


def percentile(values: ArrayLike, quantile: float) -> float | None:
    """Return NumPy's linear percentile, or ``None`` for an empty sample."""

    if not 0 <= quantile <= 100:
        raise ValueError("quantile must be between 0 and 100")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if not np.isfinite(array).all():
        raise ValueError("latency samples contain NaN or Infinity")
    return float(np.percentile(array, quantile))


def first_action_mae(
    predicted: ArrayLike,
    target: ArrayLike,
    valid_horizon: ArrayLike,
) -> float | None:
    predicted_array, target_array, valid_array = _validate_chunks(
        predicted, target, valid_horizon
    )
    selected = valid_array[:, 0]
    return _mean_or_none(np.abs(predicted_array[selected, 0] - target_array[selected, 0]))


def per_dimension_mae(
    predicted: ArrayLike,
    target: ArrayLike,
    valid_horizon: ArrayLike,
) -> list[float | None]:
    predicted_array, target_array, valid_array = _validate_chunks(
        predicted, target, valid_horizon
    )
    selected = valid_array[:, 0]
    if not selected.any():
        return [None] * ACTION_DIMENSION
    errors = np.abs(predicted_array[selected, 0] - target_array[selected, 0])
    return [float(value) for value in errors.mean(axis=0)]


def chunk_mae_by_horizon(
    predicted: ArrayLike,
    target: ArrayLike,
    valid_horizon: ArrayLike,
) -> list[float | None]:
    predicted_array, target_array, valid_array = _validate_chunks(
        predicted, target, valid_horizon
    )
    output: list[float | None] = []
    for horizon in range(predicted_array.shape[1]):
        selected = valid_array[:, horizon]
        output.append(
            _mean_or_none(
                np.abs(
                    predicted_array[selected, horizon] - target_array[selected, horizon]
                )
            )
        )
    return output


def delta_action_mae(
    predicted: ArrayLike,
    target: ArrayLike,
    valid_horizon: ArrayLike,
) -> float | None:
    """MAE of first action differences where both adjacent horizons are valid."""

    predicted_array, target_array, valid_array = _validate_chunks(
        predicted, target, valid_horizon
    )
    if predicted_array.shape[1] < 2:
        return None
    delta_valid = valid_array[:, 1:] & valid_array[:, :-1]
    delta_error = np.abs(
        np.diff(predicted_array, axis=1) - np.diff(target_array, axis=1)
    )
    return _mean_or_none(delta_error[delta_valid])


def _validate_action_sequence(actions: ArrayLike) -> NDArray[np.float64]:
    array = _float_array(actions, name="actions")
    if array.ndim != 2 or array.shape[1] != ACTION_DIMENSION:
        raise ValueError(f"actions must have shape [steps, 7]; got {array.shape}")
    return array


def action_tv(actions: ArrayLike) -> float | None:
    """Mean absolute first difference over time and the six arm dimensions."""

    array = _validate_action_sequence(actions)
    if len(array) < 2:
        return None
    return float(np.abs(np.diff(array[:, :ARM_DIMENSIONS], axis=0)).mean())


def action_jerk(actions: ArrayLike) -> float | None:
    """Mean absolute second difference; a per-step proxy, not physical jerk."""

    array = _validate_action_sequence(actions)
    if len(array) < 3:
        return None
    return float(np.abs(np.diff(array[:, :ARM_DIMENSIONS], n=2, axis=0)).mean())


def gripper_toggles(actions: ArrayLike) -> int:
    """Count transitions between negative and non-negative gripper commands."""

    array = _validate_action_sequence(actions)
    if len(array) < 2:
        return 0
    states = array[:, -1] >= 0
    return int(np.count_nonzero(states[1:] != states[:-1]))


def compute_offline_metrics(
    *,
    predicted_chunks: ArrayLike,
    target_chunks: ArrayLike,
    valid_horizon: ArrayLike,
    episode_indices: ArrayLike,
    batch_size: int,
    model_query_latencies_ms: Sequence[float],
    status: str,
) -> dict[str, Any]:
    """Build the exact offline ``metrics.json`` payload from frame-level arrays."""

    predicted, target, valid = _validate_chunks(
        predicted_chunks, target_chunks, valid_horizon
    )
    episodes = np.asarray(episode_indices, dtype=np.int64)
    if episodes.shape != (predicted.shape[0],):
        raise ValueError("episode_indices must contain one value per predicted frame")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if status not in {"complete", "smoke"}:
        raise ValueError("offline status must be 'complete' or 'smoke'")

    per_episode: list[dict[str, Any]] = []
    for episode_index in sorted(np.unique(episodes).tolist()):
        selected = episodes == episode_index
        first_predictions = predicted[selected, 0]
        per_episode.append(
            {
                "episode_index": int(episode_index),
                "num_frames": int(selected.sum()),
                "first_action_mae": first_action_mae(
                    predicted[selected], target[selected], valid[selected]
                ),
                "per_dimension_mae": per_dimension_mae(
                    predicted[selected], target[selected], valid[selected]
                ),
                "chunk_mae_by_horizon": chunk_mae_by_horizon(
                    predicted[selected], target[selected], valid[selected]
                ),
                "delta_action_mae": delta_action_mae(
                    predicted[selected], target[selected], valid[selected]
                ),
                "teacher_forced_action_tv": action_tv(first_predictions),
                "teacher_forced_action_jerk": action_jerk(first_predictions),
            }
        )

    aggregate = {
        "num_episodes": len(per_episode),
        "num_frames": int(predicted.shape[0]),
        "num_model_queries": len(model_query_latencies_ms),
        "batch_size": int(batch_size),
        "first_action_mae": first_action_mae(predicted, target, valid),
        "per_dimension_mae": per_dimension_mae(predicted, target, valid),
        "chunk_mae_by_horizon": chunk_mae_by_horizon(predicted, target, valid),
        "delta_action_mae": delta_action_mae(predicted, target, valid),
        "teacher_forced_action_tv": macro_mean(
            [item["teacher_forced_action_tv"] for item in per_episode]
        ),
        "teacher_forced_action_jerk": macro_mean(
            [item["teacher_forced_action_jerk"] for item in per_episode]
        ),
        "inference_latency_p50_ms": percentile(model_query_latencies_ms, 50),
        "inference_latency_p95_ms": percentile(model_query_latencies_ms, 95),
    }
    return {
        "schema_version": 1,
        "mode": "offline",
        "status": status,
        "aggregate": aggregate,
        "per_episode": per_episode,
        "metric_definitions": dict(OFFLINE_METRIC_DEFINITIONS),
    }


def compute_closed_loop_metrics(
    episodes: Sequence[Mapping[str, Any]],
    *,
    formal: bool,
    status: str,
) -> dict[str, Any]:
    """Build the exact closed-loop ``metrics.json`` payload."""

    if status not in {"complete", "smoke"}:
        raise ValueError("closed-loop status must be 'complete' or 'smoke'")
    if not episodes:
        raise ValueError("at least one rollout episode is required")

    per_episode: list[dict[str, Any]] = []
    all_policy_latency: list[float] = []
    all_model_latency: list[float] = []
    for episode in episodes:
        actions = _validate_action_sequence(episode["actions"])
        rewards = _float_array(episode["rewards"], name="rewards")
        policy_latency = _float_array(
            episode["policy_step_latencies_ms"], name="policy step latency"
        )
        model_latency = _float_array(
            episode["model_query_latencies_ms"], name="model query latency"
        )
        if rewards.shape != (len(actions),) or policy_latency.shape != (len(actions),):
            raise ValueError("each action must have one reward and one policy-step latency")
        all_policy_latency.extend(policy_latency.tolist())
        all_model_latency.extend(model_latency.tolist())
        per_episode.append(
            {
                "episode_index": int(episode["episode_index"]),
                "seed": int(episode["seed"]),
                "init_state_id": int(episode["init_state_id"]),
                "num_steps": len(actions),
                "success": bool(episode["success"]),
                "sum_reward": float(rewards.sum()),
                "max_reward": float(rewards.max()) if rewards.size else None,
                "termination_reason": str(episode["termination_reason"]),
                "action_tv": action_tv(actions),
                "action_jerk": action_jerk(actions),
                "gripper_toggles": gripper_toggles(actions),
                "policy_step_latency_p50_ms": percentile(policy_latency, 50),
                "policy_step_latency_p95_ms": percentile(policy_latency, 95),
                "model_query_latency_p50_ms": percentile(model_latency, 50),
                "model_query_latency_p95_ms": percentile(model_latency, 95),
                "num_model_queries": len(model_latency),
            }
        )

    num_successes = sum(bool(item["success"]) for item in per_episode)
    aggregate = {
        "is_formal": bool(formal),
        "num_episodes": len(per_episode),
        "num_successes": int(num_successes),
        "success_rate": num_successes / len(per_episode) if formal else None,
        "avg_sum_reward": _mean_or_none(
            [item["sum_reward"] for item in per_episode]
        ),
        "avg_max_reward": _mean_or_none(
            [item["max_reward"] for item in per_episode if item["max_reward"] is not None]
        ),
        "action_tv": macro_mean([item["action_tv"] for item in per_episode]),
        "action_jerk": macro_mean([item["action_jerk"] for item in per_episode]),
        "mean_gripper_toggles": _mean_or_none(
            [item["gripper_toggles"] for item in per_episode]
        ),
        "policy_step_latency_p50_ms": percentile(all_policy_latency, 50),
        "policy_step_latency_p95_ms": percentile(all_policy_latency, 95),
        "model_query_latency_p50_ms": percentile(all_model_latency, 50),
        "model_query_latency_p95_ms": percentile(all_model_latency, 95),
        "num_policy_steps": len(all_policy_latency),
        "num_model_queries": len(all_model_latency),
    }
    return {
        "schema_version": 1,
        "mode": "libero",
        "status": status,
        "aggregate": aggregate,
        "per_episode": per_episode,
        "metric_definitions": dict(CLOSED_LOOP_METRIC_DEFINITIONS),
    }
