from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_practice.evaluation.libero_dataset_cache import (
    ACTION_DIM,
    CACHE_SCHEMA_VERSION,
    CONVERTER_VERSION,
    DatasetSelection,
    DatasetTaskResolutionError,
    IncompatibleDatasetCacheError,
    RawDatasetSource,
    download_raw_task,
    inspect_lerobot_dataset,
    is_noop_action,
    prepare_task_dataset,
    resolve_agentview_image_convention,
    resolve_offline_dataset,
    select_episodes,
    validate_generated_cache,
    validate_raw_hdf5,
)
from robot_practice.evaluation.libero_tasks import resolve_libero_task


def _dataset_root(tmp_path: Path, *, languages: list[str]) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 10,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
            "observation.images.image": {"dtype": "video", "shape": [256, 256, 3]},
            "observation.images.image2": {"dtype": "video", "shape": [256, 256, 3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    tasks = pa.table(
        {
            "task_index": list(range(len(languages))),
            "task": languages,
        }
    )
    pq.write_table(tasks, root / "meta" / "tasks.parquet")
    frames = pa.table(
        {
            "episode_index": [10, 10, 11, 12],
            "task_index": [0, 0, 0, min(1, len(languages) - 1)],
        }
    )
    pq.write_table(frames, root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_inspect_and_select_combined_dataset(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_object", 8)
    root = _dataset_root(tmp_path, languages=[task.language, "another task"])
    selection = inspect_lerobot_dataset(
        root=root,
        repo_id="lerobot/libero",
        task=task,
        source="existing_combined_dataset",
        episode_namespace="global",
    )
    assert selection.task_index == 0
    assert selection.episodes == (10, 11)
    assert select_episodes(selection, (11,)).episodes == (11,)
    with pytest.raises(ValueError, match="do not belong"):
        select_episodes(selection, (12,))


def test_dataset_language_mapping_must_be_unique(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_object", 8)
    root = _dataset_root(tmp_path, languages=[task.language, task.language])
    with pytest.raises(DatasetTaskResolutionError, match="2 exact matches"):
        inspect_lerobot_dataset(
            root=root,
            repo_id="lerobot/libero",
            task=task,
            source="test",
            episode_namespace="test",
        )


def test_episode_selection_rejects_duplicates() -> None:
    selection = DatasetSelection(
        repo_id="test/data",
        root=Path("/tmp/data"),
        task_index=0,
        episodes=(1, 2),
        source="test",
        episode_namespace="test",
    )
    with pytest.raises(ValueError, match="duplicate"):
        select_episodes(selection, (1, 1))


def test_agentview_convention_requires_verified_provenance_for_auto(
    tmp_path: Path,
) -> None:
    selection = DatasetSelection(
        repo_id="lerobot/libero",
        root=tmp_path,
        task_index=0,
        episodes=(1,),
        source="explicit_dataset_root",
        episode_namespace="global",
    )
    with pytest.raises(ValueError, match="Cannot verify"):
        resolve_agentview_image_convention(
            selection,
            "auto",
            visualization_enabled=True,
        )
    assert (
        resolve_agentview_image_convention(
            selection,
            "lerobot_180",
            visualization_enabled=True,
        )
        == "lerobot_180"
    )
    assert (
        resolve_agentview_image_convention(
            selection,
            "auto",
            visualization_enabled=False,
        )
        is None
    )


def test_agentview_convention_auto_reads_generated_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "output_contract": {
                    "agentview_image_convention": "physical_opencv",
                }
            }
        ),
        encoding="utf-8",
    )
    selection = DatasetSelection(
        repo_id="robot-practice/task",
        root=tmp_path,
        task_index=0,
        episodes=(0,),
        source="generated_task_cache",
        episode_namespace="local",
        source_manifest=manifest,
    )
    assert (
        resolve_agentview_image_convention(
            selection,
            "auto",
            visualization_enabled=True,
        )
        == "physical_opencv"
    )


def test_noop_filter_preserves_gripper_changes() -> None:
    still_open = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    still_closed = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    moving = np.array([0.01, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    assert is_noop_action(still_open, None)
    assert is_noop_action(still_open, still_open)
    assert not is_noop_action(still_closed, still_open)
    assert not is_noop_action(moving, still_open)


def _raw_hdf5(path: Path, *, language: str) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": language})
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        data.attrs["num_demos"] = 1
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 2
        demo.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float64))
        demo.create_dataset("states", data=np.zeros((2, 12), dtype=np.float64))
        demo.create_dataset("robot_states", data=np.zeros((2, 9), dtype=np.float64))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((2, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((2, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("ee_states", data=np.zeros((2, 6), dtype=np.float64))
        obs.create_dataset("gripper_states", data=np.zeros((2, 2), dtype=np.float64))


def test_validate_raw_hdf5_contract(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_goal", 0)
    path = tmp_path / "task.hdf5"
    _raw_hdf5(path, language=task.language)
    summary = validate_raw_hdf5(path, task)
    assert summary.demo_names == ("demo_0",)
    assert summary.total_frames == 2
    assert summary.image_shape == (128, 128, 3)
    assert summary.source_control_frequency_hz == 20


def test_validate_raw_hdf5_rejects_wrong_language(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_goal", 0)
    path = tmp_path / "task.hdf5"
    _raw_hdf5(path, language="different task")
    with pytest.raises(ValueError, match="language mismatch"):
        validate_raw_hdf5(path, task)


def test_download_is_single_file_and_writes_provenance(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_90", 0)
    payload = b"hdf5-placeholder"
    sibling = SimpleNamespace(
        rfilename=task.source_relative_path,
        size=len(payload),
        lfs={"sha256": "abc", "size": len(payload)},
    )
    api = SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(
            sha="revision-123", siblings=[sibling]
        )
    )
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload)
        return str(destination)

    source = download_raw_task(
        task,
        raw_root=tmp_path / "raw",
        api=api,  # type: ignore[arg-type]
        downloader=downloader,
        free_bytes=10 * 1024**3,
    )
    assert source.path.read_bytes() == payload
    assert source.revision == "revision-123"
    assert calls[0]["filename"] == task.source_relative_path
    assert calls[0]["revision"] == "revision-123"
    assert source.path.with_name(f"{source.path.name}.source.json").is_file()


def test_download_rejects_insufficient_space_before_writing(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_90", 0)
    sibling = SimpleNamespace(
        rfilename=task.source_relative_path,
        size=100,
        lfs={"sha256": "abc", "size": 100},
    )
    api = SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="revision", siblings=[sibling])
    )
    with pytest.raises(OSError, match="Insufficient disk space"):
        download_raw_task(
            task,
            raw_root=tmp_path / "raw",
            api=api,  # type: ignore[arg-type]
            downloader=lambda **kwargs: pytest.fail("download must not start"),
            free_bytes=1,
        )


def test_explicit_dataset_root_cannot_be_rebuilt(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_goal", 0)
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_offline_dataset(
            task,
            episodes=None,
            dataset_root=tmp_path,
            rebuild_dataset_cache=True,
        )


def test_incompatible_cache_version_is_actionable(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_goal", 0)
    root = tmp_path / "cache"
    root.mkdir()
    (root / "source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION + 1,
                "converter_version": CONVERTER_VERSION,
                "task": task.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IncompatibleDatasetCacheError, match="--rebuild-dataset-cache"):
        validate_generated_cache(root, task)


def test_resolver_reuses_combined_data_without_preparing(tmp_path: Path) -> None:
    task = resolve_libero_task("libero_object", 8)
    combined = _dataset_root(tmp_path, languages=[task.language, "another task"])
    selection = resolve_offline_dataset(
        task,
        episodes=(10,),
        combined_root=combined,
        cache_root=tmp_path / "cache",
        prepare=lambda *args, **kwargs: pytest.fail("prepare must not run"),
    )
    assert selection.source == "existing_combined_dataset"
    assert selection.episodes == (10,)


class _FakeReplayEnvironment:
    def __init__(self) -> None:
        self.robots = [SimpleNamespace(controller=SimpleNamespace(use_delta=False))]
        self.steps = 0
        self.closed = False

    @staticmethod
    def _observation() -> dict[str, np.ndarray]:
        return {
            "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.01, -0.01]),
        }

    def reset(self) -> dict[str, np.ndarray]:
        self.steps = 0
        return self._observation()

    def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        assert state.shape == (12,)
        return self._observation()

    def step(self, action: list[float]) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        assert len(action) == ACTION_DIM
        self.steps += 1
        return self._observation(), 0.0, False, {}

    def check_success(self) -> bool:
        return self.steps > 10

    def close(self) -> None:
        self.closed = True


def test_prepare_task_dataset_replays_and_reopens_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = resolve_libero_task("libero_goal", 0)
    raw_path = tmp_path / "raw.hdf5"
    _raw_hdf5(raw_path, language=task.language)
    with h5py.File(raw_path, "r+") as handle:
        handle["data/demo_0/actions"][0] = np.array([0.1, 0, 0, 0, 0, 0, -1])
    source = RawDatasetSource(
        path=raw_path,
        repository="test/libero",
        revision="revision",
        relative_path=task.source_relative_path,
        size_bytes=raw_path.stat().st_size,
        lfs_sha256="test",
        license="CC BY 4.0",
        attribution_url="https://example.test/libero",
    )
    monkeypatch.setattr(
        "robot_practice.evaluation.libero_dataset_cache.download_raw_task",
        lambda *args, **kwargs: source,
    )
    environment = _FakeReplayEnvironment()
    selection = prepare_task_dataset(
        task,
        cache_root=tmp_path / "converted",
        environment_factory=lambda selected: environment,
    )
    assert environment.closed
    assert selection.episodes == (0,)
    assert selection.task_index == 0
    assert (selection.root / "source_manifest.json").is_file()
    manifest = json.loads(
        (selection.root / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["output_contract"]["agentview_image_convention"] == "lerobot_180"
    assert (selection.root / "replay_report.json").is_file()
    assert not (selection.root / ".incomplete").exists()
