"""Shared target-task split and sampling contracts for SmolVLA LIBERO."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .b0_training import (
    DATASET_REPO_ID,
    EXPECTED_ACTION_DIM,
    EXPECTED_FPS,
    EXPECTED_STATE_DIM,
    EXPECTED_TOTAL_EPISODES,
    EXPECTED_TOTAL_FRAMES,
    DatasetAudit,
    load_json_object,
    resolve_repository_path,
)
from .data_preparation import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_SUITE,
    DEFAULT_TASK_IDS,
    MAPPING_CONTRACT,
    SCHEMA_VERSION,
    TARGET_TASKS,
    load_episode_catalog,
    sha256_file,
)

SAMPLING_STRATEGY = "uniform_task_then_uniform_episode"
EXPECTED_TRAIN_EPISODES_BY_TASK = {1: 38, 4: 36, 8: 39}
EXPECTED_VALIDATION_EPISODES_BY_TASK = {1: 10, 4: 10, 8: 10}
EXPECTED_TRAIN_FRAMES_BY_TASK = {1: 3809, 4: 3594, 8: 3619}
EXPECTED_VALIDATION_FRAMES_BY_TASK = {1: 1045, 4: 980, 8: 944}
EXPECTED_TRAIN_EPISODES = sum(EXPECTED_TRAIN_EPISODES_BY_TASK.values())
EXPECTED_VALIDATION_EPISODES = sum(EXPECTED_VALIDATION_EPISODES_BY_TASK.values())
EXPECTED_TRAIN_FRAMES = sum(EXPECTED_TRAIN_FRAMES_BY_TASK.values())
EXPECTED_VALIDATION_FRAMES = sum(EXPECTED_VALIDATION_FRAMES_BY_TASK.values())


@dataclass(frozen=True)
class TargetSplit:
    source: Path
    sha256: str
    task_ids: tuple[int, ...]
    combined_task_indices: tuple[int, ...]
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    train_frames: int
    validation_frames: int
    train_episodes_by_task: Mapping[int, tuple[int, ...]]
    validation_episodes_by_task: Mapping[int, tuple[int, ...]]
    payload: dict[str, Any]


def _episode_records(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    episode_indices: list[int] = []
    raw_demos: list[str] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise TypeError(f"{field}[{index}] must be an object")
        episode_index = record.get("episode_index")
        raw_demo = record.get("raw_demo")
        error = record.get("action_max_abs_error")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError(f"{field}[{index}].episode_index must be an integer")
        if not isinstance(raw_demo, str) or not raw_demo:
            raise TypeError(f"{field}[{index}].raw_demo must be a non-empty string")
        if isinstance(error, bool) or not isinstance(error, (int, float)):
            raise TypeError(f"{field}[{index}].action_max_abs_error must be numeric")
        if not math.isfinite(float(error)) or float(error) > 1e-6:
            raise ValueError(f"{field}[{index}] violates the raw-action mapping contract")
        episode_indices.append(episode_index)
        raw_demos.append(raw_demo)
    if episode_indices != sorted(episode_indices):
        raise ValueError(f"{field} episode indices must be sorted")
    if len(episode_indices) != len(set(episode_indices)):
        raise ValueError(f"{field} contains duplicate episode indices")
    if len(raw_demos) != len(set(raw_demos)):
        raise ValueError(f"{field} contains duplicate raw demonstrations")
    return tuple(episode_indices)


def load_target_split(path: str | Path) -> TargetSplit:
    source = resolve_repository_path(path)
    payload = load_json_object(source)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported target split schema")
    if payload.get("dataset_repo_id") != DATASET_REPO_ID:
        raise ValueError("Target split dataset repo ID mismatch")
    if payload.get("suite") != DEFAULT_SUITE:
        raise ValueError("Target split suite mismatch")
    if payload.get("split_seed") != DEFAULT_SPLIT_SEED:
        raise ValueError("Target split seed mismatch")
    if payload.get("mapping_contract") != MAPPING_CONTRACT:
        raise ValueError("Target split action mapping contract mismatch")
    if payload.get("training_sampling_strategy") != SAMPLING_STRATEGY:
        raise ValueError("Target split training sampling strategy mismatch")

    tasks_payload = payload.get("tasks")
    if not isinstance(tasks_payload, dict):
        raise TypeError("Target split tasks must be an object")
    expected_task_ids = tuple(sorted(DEFAULT_TASK_IDS))
    try:
        observed_task_ids = tuple(sorted(int(value) for value in tasks_payload))
    except ValueError as error:
        raise TypeError("Target split task keys must be integers") from error
    if observed_task_ids != expected_task_ids:
        raise ValueError("Target split task IDs do not match the frozen C1/C2/C3 targets")

    train_by_task: dict[int, tuple[int, ...]] = {}
    validation_by_task: dict[int, tuple[int, ...]] = {}
    combined_task_indices: list[int] = []
    train_frames = 0
    validation_frames = 0
    for task_id in expected_task_ids:
        task = tasks_payload[str(task_id)]
        if not isinstance(task, dict):
            raise TypeError(f"Target task {task_id} must be an object")
        definition = TARGET_TASKS[task_id]
        if task.get("task_id") != task_id:
            raise ValueError(f"Target task {task_id} task_id mismatch")
        if task.get("combined_task_index") != definition["combined_task_index"]:
            raise ValueError(f"Target task {task_id} combined task index mismatch")
        if task.get("canonical") != definition["canonical"]:
            raise ValueError(f"Target task {task_id} canonical instruction mismatch")
        demonstration_split = task.get("demonstration_split")
        if not isinstance(demonstration_split, dict):
            raise TypeError(f"Target task {task_id} demonstration split is missing")
        if demonstration_split.get("sampling_strategy") != SAMPLING_STRATEGY:
            raise ValueError(f"Target task {task_id} sampling strategy mismatch")
        train = _episode_records(
            demonstration_split.get("train_demonstrations"),
            field=f"tasks.{task_id}.train_demonstrations",
        )
        validation = _episode_records(
            demonstration_split.get("offline_validation_demonstrations"),
            field=f"tasks.{task_id}.offline_validation_demonstrations",
        )
        if set(train) & set(validation):
            raise ValueError(f"Target task {task_id} train/validation episodes overlap")
        if len(train) != EXPECTED_TRAIN_EPISODES_BY_TASK[task_id]:
            raise ValueError(f"Target task {task_id} train episode count mismatch")
        if len(validation) != EXPECTED_VALIDATION_EPISODES_BY_TASK[task_id]:
            raise ValueError(f"Target task {task_id} validation episode count mismatch")
        task_train_frames = demonstration_split.get("train_frames")
        task_validation_frames = demonstration_split.get("offline_validation_frames")
        if task_train_frames != EXPECTED_TRAIN_FRAMES_BY_TASK[task_id]:
            raise ValueError(f"Target task {task_id} train frame count mismatch")
        if task_validation_frames != EXPECTED_VALIDATION_FRAMES_BY_TASK[task_id]:
            raise ValueError(f"Target task {task_id} validation frame count mismatch")
        train_by_task[task_id] = train
        validation_by_task[task_id] = validation
        combined_task_indices.append(int(definition["combined_task_index"]))
        train_frames += int(task_train_frames)
        validation_frames += int(task_validation_frames)

    train_episodes = tuple(sorted(ep for values in train_by_task.values() for ep in values))
    validation_episodes = tuple(
        sorted(ep for values in validation_by_task.values() for ep in values)
    )
    if len(train_episodes) != len(set(train_episodes)):
        raise ValueError("An episode appears in multiple target training tasks")
    if len(validation_episodes) != len(set(validation_episodes)):
        raise ValueError("An episode appears in multiple target validation tasks")
    if set(train_episodes) & set(validation_episodes):
        raise ValueError("Target train and validation episodes overlap globally")
    if len(train_episodes) != EXPECTED_TRAIN_EPISODES:
        raise ValueError("Target train episode aggregate mismatch")
    if len(validation_episodes) != EXPECTED_VALIDATION_EPISODES:
        raise ValueError("Target validation episode aggregate mismatch")
    if train_frames != EXPECTED_TRAIN_FRAMES:
        raise ValueError("Target train frame aggregate mismatch")
    if validation_frames != EXPECTED_VALIDATION_FRAMES:
        raise ValueError("Target validation frame aggregate mismatch")
    return TargetSplit(
        source=source,
        sha256=sha256_file(source),
        task_ids=expected_task_ids,
        combined_task_indices=tuple(combined_task_indices),
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        train_frames=train_frames,
        validation_frames=validation_frames,
        train_episodes_by_task=train_by_task,
        validation_episodes_by_task=validation_by_task,
        payload=payload,
    )


def validate_target_dataset(dataset_root: str | Path, split: TargetSplit) -> DatasetAudit:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = resolve_repository_path(dataset_root)
    catalog, _ = load_episode_catalog(root)
    observed_train_frames = 0
    observed_validation_frames = 0
    expected_combined_indices = {
        task_id: int(TARGET_TASKS[task_id]["combined_task_index"]) for task_id in split.task_ids
    }
    for task_id in split.task_ids:
        combined_index = expected_combined_indices[task_id]
        for episode_index in split.train_episodes_by_task[task_id]:
            entry = catalog.get(episode_index)
            if entry is None:
                raise ValueError(f"Target train episode {episode_index} is absent from dataset")
            if entry.task_index != combined_index:
                raise ValueError(
                    f"Target train episode {episode_index} belongs to task "
                    f"{entry.task_index}, expected {combined_index}"
                )
            observed_train_frames += entry.length
        for episode_index in split.validation_episodes_by_task[task_id]:
            entry = catalog.get(episode_index)
            if entry is None:
                raise ValueError(
                    f"Target validation episode {episode_index} is absent from dataset"
                )
            if entry.task_index != combined_index:
                raise ValueError(
                    f"Target validation episode {episode_index} belongs to task "
                    f"{entry.task_index}, expected {combined_index}"
                )
            observed_validation_frames += entry.length
    if observed_train_frames != split.train_frames:
        raise ValueError("Dataset target train frames disagree with target split")
    if observed_validation_frames != split.validation_frames:
        raise ValueError("Dataset target validation frames disagree with target split")

    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=root)
    if int(metadata.total_episodes) != EXPECTED_TOTAL_EPISODES:
        raise ValueError("Target dataset total episode count mismatch")
    if int(metadata.total_frames) != EXPECTED_TOTAL_FRAMES:
        raise ValueError("Target dataset total frame count mismatch")
    required_features = {
        "observation.images.image": ("video", (256, 256, 3)),
        "observation.images.image2": ("video", (256, 256, 3)),
        "observation.state": ("float32", (EXPECTED_STATE_DIM,)),
        "action": ("float32", (EXPECTED_ACTION_DIM,)),
    }
    for key, (expected_dtype, expected_shape) in required_features.items():
        feature = metadata.features.get(key)
        if not isinstance(feature, dict):
            raise TypeError(f"Dataset feature is missing or malformed: {key}")
        if feature.get("dtype") != expected_dtype:
            raise ValueError(f"Dataset feature {key} dtype mismatch")
        if tuple(feature.get("shape", ())) != expected_shape:
            raise ValueError(f"Dataset feature {key} shape mismatch")
    if int(metadata.fps) != EXPECTED_FPS:
        raise ValueError(f"Expected dataset fps={EXPECTED_FPS}, got {metadata.fps}")
    return DatasetAudit(
        dataset_root=root,
        total_episodes=int(metadata.total_episodes),
        total_frames=int(metadata.total_frames),
        train_episodes=len(split.train_episodes),
        train_frames=split.train_frames,
        fps=int(metadata.fps),
        features={
            key: {
                "dtype": metadata.features[key]["dtype"],
                "shape": list(metadata.features[key]["shape"]),
            }
            for key in required_features
        },
    )


class UniformTaskEpisodeSampler:
    """Sample task, episode within task, then frame within episode uniformly."""

    def __init__(
        self,
        dataset: Any,
        split: TargetSplit,
        *,
        seed: int,
        num_samples: int | None = None,
    ) -> None:
        episode_column: Sequence[Any] = dataset.hf_dataset["episode_index"]
        relative_indices: dict[int, list[int]] = defaultdict(list)
        for relative_index, episode_index in enumerate(episode_column):
            relative_indices[int(episode_index)].append(relative_index)
        observed = set(relative_indices)
        expected = set(split.train_episodes)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                "Balanced sampler dataset episode mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if any(not indices for indices in relative_indices.values()):
            raise ValueError("Balanced sampler found an empty episode")
        observed_frames = sum(len(indices) for indices in relative_indices.values())
        if observed_frames != split.train_frames:
            raise ValueError(
                f"Balanced sampler expected {split.train_frames} frames, found {observed_frames}"
            )
        self._task_ids = split.task_ids
        self._episodes_by_task = {
            task_id: split.train_episodes_by_task[task_id] for task_id in split.task_ids
        }
        self._indices_by_episode = dict(relative_indices)
        self._seed = int(seed)
        self._epoch = 0
        self._num_samples = observed_frames if num_samples is None else int(num_samples)
        if self._num_samples <= 0:
            raise ValueError("Balanced sampler num_samples must be positive")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = random.Random(self._seed + self._epoch)
        self._epoch += 1
        for _ in range(self._num_samples):
            task_id = generator.choice(self._task_ids)
            episode_index = generator.choice(self._episodes_by_task[task_id])
            yield generator.choice(self._indices_by_episode[episode_index])

    def __len__(self) -> int:
        return self._num_samples


@contextmanager
def override_target_training_sampler(
    lerobot_train_module: Any,
    split: TargetSplit,
    *,
    seed: int,
) -> Iterator[None]:
    """Replace LeRobot's frame shuffle with the frozen target-task sampler."""

    original = lerobot_train_module.torch.utils.data.DataLoader

    class BalancedDataLoader(original):
        def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
            # Accelerate performs isinstance checks against torch's current
            # DataLoader class. Restore it as soon as the one target loader is
            # constructed so all subsequent prepare/shard logic sees the
            # original class while this instance remains a valid subclass.
            lerobot_train_module.torch.utils.data.DataLoader = original
            if kwargs.get("sampler") is not None:
                raise ValueError("C1 target sampler cannot replace an existing sampler")
            kwargs["shuffle"] = False
            kwargs["sampler"] = UniformTaskEpisodeSampler(dataset, split, seed=seed)
            super().__init__(dataset, *args, **kwargs)

    lerobot_train_module.torch.utils.data.DataLoader = BalancedDataLoader
    try:
        yield
    finally:
        lerobot_train_module.torch.utils.data.DataLoader = original
