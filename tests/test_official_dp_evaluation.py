from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OFFICIAL_DP_ROOT = (
    Path(__file__).resolve().parents[1]
    / "third_party/maniskill_official/examples/baselines/diffusion_policy"
)
sys.path.insert(0, str(OFFICIAL_DP_ROOT))

_append_episode_metrics = importlib.import_module(
    "diffusion_policy.evaluate"
)._append_episode_metrics
_train_rgbd = importlib.import_module("train_rgbd")


def test_final_info_vector_metrics_are_one_value_per_episode_and_respect_limit() -> None:
    metrics = defaultdict(list)
    info = {
        "final_info": {
            "episode": {
                "success_once": torch.tensor([True, False, True, True]),
                "success_at_end": torch.tensor([False, False, True, True]),
                "episode_len": torch.tensor([500, 500, 500, 500]),
            }
        },
        "_final_info": torch.tensor([True, True, True, True]),
    }

    added = _append_episode_metrics(
        metrics,
        info,
        torch.ones(4, dtype=torch.bool),
        limit=3,
    )

    assert added == 3
    assert len(metrics["success_once"]) == 3
    assert [bool(value) for value in metrics["success_once"]] == [True, False, True]
    assert [int(value) for value in metrics["episode_len"]] == [500, 500, 500]


def test_gymnasium_episode_metrics_use_presence_masks() -> None:
    metrics = defaultdict(list)
    info = {
        "episode": {
            "success_once": np.asarray([True, False, True]),
            "_success_once": np.asarray([True, False, True]),
            "return": np.asarray([1.0, 0.0, 2.0]),
            "_return": np.asarray([True, False, True]),
        },
        "_episode": np.asarray([True, False, True]),
    }

    added = _append_episode_metrics(
        metrics,
        info,
        np.asarray([True, False, True]),
    )

    assert added == 2
    assert [bool(value) for value in metrics["success_once"]] == [True, True]
    assert [float(value) for value in metrics["return"]] == [1.0, 2.0]


def test_cuda_truncated_tensor_is_safe_at_metric_boundary() -> None:
    if not torch.cuda.is_available():
        return

    metrics = defaultdict(list)
    info = {
        "episode": {"success": torch.ones(2, device="cuda")},
        "_episode": torch.ones(2, dtype=torch.bool, device="cuda"),
    }

    added = _append_episode_metrics(
        metrics,
        info,
        torch.ones(2, dtype=torch.bool, device="cuda"),
    )

    assert added == 2
    assert [float(value) for value in metrics["success"]] == [1.0, 1.0]


def test_distributed_evaluation_shards_and_merges_episode_metrics() -> None:
    counts = [_train_rgbd._partition_count(10, 4, rank) for rank in range(4)]
    assert counts == [3, 3, 2, 2]
    offsets = [_train_rgbd._partition_offset(10, 4, rank) for rank in range(4)]
    assert offsets == [0, 3, 6, 8]
    merged = _train_rgbd._merge_eval_metrics(
        [
            {"success_once": np.asarray([1.0, 0.0])},
            {"success_once": np.asarray([1.0])},
        ]
    )
    np.testing.assert_array_equal(merged["success_once"], np.asarray([1.0, 0.0, 1.0]))


class _FakeEvalEnv:
    num_envs = 4

    def reset(self, seed=None):
        del seed
        return {"state": torch.zeros((self.num_envs, 1, 1))}, {}

    def step(self, action):
        assert action.shape == (self.num_envs, 1)
        info = {
            "final_info": {
                "episode": {
                    "success_once": torch.tensor([True, False, True, False]),
                    "success_at_end": torch.tensor([True, False, False, False]),
                }
            },
            "_final_info": torch.ones(self.num_envs, dtype=torch.bool),
        }
        return (
            {"state": torch.zeros((self.num_envs, 1, 1))},
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.ones(self.num_envs, dtype=torch.bool),
            info,
        )


class _FakeAgent:
    def eval(self):
        return self

    def train(self):
        return self

    def get_action(self, observation):
        return torch.zeros((observation["state"].shape[0], 1, 1))


class _CudaFakeAgent(_FakeAgent):
    def get_action(self, observation):
        if not torch.cuda.is_available():
            return super().get_action(observation)
        return torch.zeros(
            (observation["state"].shape[0], 1, 1),
            device="cuda",
        )


class _RecordingEvalEnv(_FakeEvalEnv):
    def step(self, action):
        if torch.cuda.is_available():
            assert isinstance(action, np.ndarray)
        return super().step(action)


def test_evaluate_caps_non_divisible_vector_batch() -> None:
    evaluate = importlib.import_module("diffusion_policy.evaluate").evaluate

    metrics = evaluate(
        n=3,
        agent=_FakeAgent(),
        eval_envs=_FakeEvalEnv(),
        device=torch.device("cpu"),
        sim_backend="gpu",
        progress_bar=False,
    )

    assert metrics["success_once"].shape == (3,)
    assert metrics["success_at_end"].shape == (3,)


def test_cpu_rollout_receives_host_actions_from_cuda_policy() -> None:
    if not torch.cuda.is_available():
        return
    evaluate = importlib.import_module("diffusion_policy.evaluate").evaluate

    metrics = evaluate(
        n=1,
        agent=_CudaFakeAgent(),
        eval_envs=_RecordingEvalEnv(),
        device=torch.device("cuda"),
        sim_backend="physx_cpu",
        progress_bar=False,
    )

    assert metrics["success_once"].shape == (1,)
