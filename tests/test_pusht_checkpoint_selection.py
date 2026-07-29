from __future__ import annotations

from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest

from robot_practice.experiments.pusht_diffusion import checkpoint_selection


class _FakeProgress:
    def __init__(self) -> None:
        self.updates = 0
        self.postfixes: list[dict[str, object]] = []
        self.messages: list[str] = []

    def set_postfix(self, **values: object) -> None:
        values.pop("refresh", None)
        self.postfixes.append(values)

    def update(self, amount: int) -> None:
        self.updates += amount

    def write(self, message: str) -> None:
        self.messages.append(message)


def test_checkpoint_evaluation_reports_rollout_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeBCPolicy:
        pass

    policy = FakeBCPolicy()
    loaded = SimpleNamespace(
        policy=policy,
        preprocessor=object(),
        postprocessor=object(),
    )
    rollout_results = iter(
        [
            SimpleNamespace(success=True, max_reward=0.8),
            SimpleNamespace(success=False, max_reward=0.4),
        ]
    )
    monkeypatch.setattr(checkpoint_selection, "BCChunkPolicy", FakeBCPolicy)
    monkeypatch.setattr(
        checkpoint_selection,
        "load_policy_with_processors",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        checkpoint_selection,
        "_offline_error",
        lambda *_args, **_kwargs: (0.01, 0.02),
    )
    monkeypatch.setattr(
        checkpoint_selection,
        "run_rollout",
        lambda **_kwargs: next(rollout_results),
    )
    monkeypatch.setattr(checkpoint_selection, "sha256_file", lambda _path: "checkpoint-id")
    monkeypatch.setattr(checkpoint_selection.torch.cuda, "is_available", lambda: False)
    progress = _FakeProgress()

    rows = checkpoint_selection._evaluate_run(
        model_name="bc_chunk",
        checkpoints=((25_000, tmp_path),),
        dataset_root=tmp_path,
        development_episodes=(1, 2),
        environment_seeds=(1000, 1001),
        train_seed=1000,
        code_commit="commit",
        progress=progress,
    )

    assert progress.updates == 2
    assert [entry["phase"] for entry in progress.postfixes] == [
        "loading",
        "offline",
        "rollout",
        "rollout",
    ]
    assert progress.postfixes[-1]["environment_seed"] == 1001
    assert progress.messages == [
        ("bc_chunk checkpoint 025000: success=0.500, mean_max_reward=0.600, offline_mse=0.020000")
    ]
    assert rows[0]["closed_loop_success_rate"] == pytest.approx(0.5)
    assert rows[0]["mean_max_reward"] == pytest.approx(0.6)


def test_checkpoint_tasks_prioritize_diffusion_for_load_balancing(
    tmp_path: Path,
) -> None:
    tasks = checkpoint_selection._checkpoint_tasks(
        ((25_000, tmp_path / "bc-25"), (50_000, tmp_path / "bc-50")),
        ((25_000, tmp_path / "diff-25"), (50_000, tmp_path / "diff-50")),
    )

    assert [(task.model_name, task.step) for task in tasks] == [
        ("diffusion", 25_000),
        ("diffusion", 50_000),
        ("bc_chunk", 25_000),
        ("bc_chunk", 50_000),
    ]


def test_multi_gpu_validation_uses_visible_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoint_selection.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(checkpoint_selection.torch.cuda, "device_count", lambda: 4)

    checkpoint_selection._validate_num_gpus(1)
    checkpoint_selection._validate_num_gpus(4)
    with pytest.raises(RuntimeError, match="only 4 CUDA device"):
        checkpoint_selection._validate_num_gpus(5)
    with pytest.raises(ValueError, match="at least 1"):
        checkpoint_selection._validate_num_gpus(0)


def test_checkpoint_worker_binds_checkpoint_to_requested_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_devices: list[int] = []
    evaluated_devices: list[str | None] = []
    monkeypatch.setattr(
        checkpoint_selection.torch.cuda,
        "set_device",
        selected_devices.append,
    )
    monkeypatch.setattr(checkpoint_selection.torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(
        checkpoint_selection.torch,
        "set_num_interop_threads",
        lambda _value: None,
    )

    def fake_evaluate_checkpoint(**kwargs: object) -> dict[str, object]:
        device = kwargs["device"]
        assert isinstance(device, str)
        evaluated_devices.append(device)
        callback = kwargs["event_callback"]
        assert callable(callback)
        callback("loading", None)
        callback("rollout", 1000)
        callback("rollout_complete", 1000)
        return {
            "model": kwargs["model_name"],
            "step": kwargs["step"],
        }

    monkeypatch.setattr(
        checkpoint_selection,
        "_evaluate_checkpoint",
        fake_evaluate_checkpoint,
    )
    task_queue: Queue[object] = Queue()
    event_queue: Queue[object] = Queue()
    task_queue.put(
        checkpoint_selection.CheckpointTask(
            model_name="diffusion",
            step=25_000,
            path=tmp_path,
        )
    )
    task_queue.put(None)

    checkpoint_selection._checkpoint_worker(
        3,
        task_queue,
        event_queue,
        checkpoint_selection.WorkerConfig(
            dataset_root=tmp_path,
            development_episodes=(1,),
            environment_seeds=(1000,),
            train_seed=1000,
            code_commit="commit",
        ),
    )

    events = [event_queue.get_nowait() for _ in range(event_queue.qsize())]
    assert selected_devices == [3]
    assert evaluated_devices == ["cuda:3"]
    assert [event.phase for event in events[:-1]] == [
        "loading",
        "rollout",
        "rollout_complete",
    ]
    assert events[-1].kind == "result"
    assert events[-1].row == {"model": "diffusion", "step": 25_000}
