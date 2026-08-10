from __future__ import annotations

import errno
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from robot_practice.experiments.maniskill.audit_demos import (
    _state_fingerprint_from_observation,
    audit_raw_data,
    compare_fingerprints,
)
from robot_practice.experiments.maniskill.convert_demos import (
    ACTION_CHANNEL_NAMES,
    ACTION_KEY,
    CAMERA_KEY,
    EXCLUDED_STATE_FIELDS,
    STATE_KEY,
    _publish_directory,
    action_channel_stats,
    convert_replay,
    dataset_features,
    iter_episode_frames,
    trim_frame_count,
)
from robot_practice.experiments.maniskill.split import (
    collect_demonstration_seeds,
    evaluation_seed_plan,
    load_seed_sets,
    reset_batches_for_set,
    seed_isolation_report,
)


def test_publish_directory_copies_when_rename_crosses_filesystems(
    tmp_path: Path, monkeypatch
) -> None:
    temporary = tmp_path / "temporary"
    destination = tmp_path / "published"
    (temporary / "nested").mkdir(parents=True)
    (temporary / "nested" / "payload.bin").write_bytes(b"payload")

    original_rename = Path.rename

    def raise_cross_device(self: Path, target: str | Path) -> Path:
        if self == temporary and Path(target) == destination:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", raise_cross_device)

    _publish_directory(temporary, destination)

    assert (destination / "nested" / "payload.bin").read_bytes() == b"payload"
    assert not temporary.exists()


def _make_replay(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    replay_h5 = tmp_path / "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
    metadata = {
        "env_info": {
            "env_id": "PickCube-v1",
            "env_kwargs": {
                "obs_mode": "rgb",
                "control_mode": "pd_ee_delta_pose",
                "sim_backend": "physx_cpu",
            },
        },
        "episodes": [
            {
                "episode_id": 0,
                "reset_kwargs": {"seed": 100},
                "elapsed_steps": 2,
                "success": True,
            },
            {
                "episode_id": 1,
                "reset_kwargs": {"seed": 101},
                "elapsed_steps": 2,
                "success": False,
            },
        ],
    }
    replay_h5.with_suffix(".json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    with h5py.File(replay_h5, "w") as handle:
        for episode_id in (0, 1):
            episode = handle.create_group(f"traj_{episode_id}")
            episode.create_dataset(
                "actions",
                data=np.asarray(
                    [
                        [1.0, 0.5, 0.25, 0.2, 0.1, 0.05, -1.0],
                        [0.5, 0.25, 0.125, 0.1, 0.05, 0.025, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
            episode.create_dataset("success", data=np.asarray([False, episode_id == 0]))
            obs = episode.create_group("obs")
            sensor_data = obs.create_group("sensor_data")
            camera = sensor_data.create_group("base_camera")
            camera.create_dataset(
                "rgb",
                data=np.zeros((3, 128, 128, 3), dtype=np.uint8),
            )
            agent = obs.create_group("agent")
            agent.create_dataset("qpos", data=np.zeros((3, 9), dtype=np.float32))
            agent.create_dataset("qvel", data=np.ones((3, 9), dtype=np.float32))
            extra = obs.create_group("extra")
            extra.create_dataset(
                "goal_pos",
                data=np.full((3, 3), 0.25, dtype=np.float32),
            )
            extra.create_dataset("is_grasped", data=np.zeros(3, dtype=bool))
    return replay_h5, metadata


def test_dataset_contract_and_frame_timeline(tmp_path: Path) -> None:
    replay_h5, _ = _make_replay(tmp_path)
    features = dataset_features(7)
    assert features[CAMERA_KEY]["shape"] == (128, 128, 3)
    assert features[STATE_KEY]["shape"] == (21,)
    assert features[ACTION_KEY]["shape"] == (7,)
    assert dataset_features(4)[ACTION_KEY]["shape"] == (4,)

    with h5py.File(replay_h5, "r") as handle:
        frames = list(iter_episode_frames(handle["traj_0"], action_dim=7))

    assert len(frames) == 2
    assert frames[0][CAMERA_KEY].shape == (128, 128, 3)
    assert frames[0][STATE_KEY].shape == (21,)
    assert frames[0][ACTION_KEY].shape == (7,)
    assert "is_grasped" not in frames[0]
    assert "extra.is_grasped" in EXCLUDED_STATE_FIELDS
    assert {
        "extra.obj_pose",
        "extra.tcp_to_obj_pos",
        "extra.obj_to_goal_pos",
        "extra.tcp_pose",
    }.issubset(EXCLUDED_STATE_FIELDS)

    with h5py.File(replay_h5, "r") as handle:
        assert trim_frame_count({"success": True}, handle["traj_0"], success_tail_frames=0) == 2


def test_action_channel_stats_records_rotation_gate(tmp_path: Path) -> None:
    replay_h5, metadata = _make_replay(tmp_path)
    records = {int(item["episode_id"]): item for item in metadata["episodes"]}
    stats = action_channel_stats(
        replay_h5=replay_h5,
        metadata=metadata,
        records=records,
    )

    assert stats["channel_order"] == list(ACTION_CHANNEL_NAMES)
    assert stats["action_dim"] == 7
    assert stats["num_action_steps"] == 4
    assert stats["rotation_to_translation_abs_median_ratio"] == pytest.approx(0.2)
    assert stats["rotation_channels_effective"] is True
    assert set(stats["channels"]) == set(ACTION_CHANNEL_NAMES)


def test_raw_audit_and_initial_fingerprint_duplicate_report(tmp_path: Path) -> None:
    replay_h5, _ = _make_replay(tmp_path)
    raw, tail = audit_raw_data(replay_h5)
    assert raw["summary"]["num_episodes"] == 2
    assert raw["summary"]["total_frames"] == 4
    assert tail["aligned_by_tau"][0]["tau"] == 0

    training = [
        {
            "episode_id": 0,
            "robot_qpos": [0.0] * 9,
            "cube_position": [0.0] * 3,
            "cube_orientation": [1.0, 0.0, 0.0, 0.0],
            "goal_position": [0.1, 0.1, 0.2],
        }
    ]
    evaluation = [
        {
            "episode_id": 0,
            "state_set": "development",
            "robot_qpos": [0.0] * 9,
            "cube_position": [0.0, 0.0, 1e-5],
            "cube_orientation": [1.0, 0.0, 0.0, 0.0],
            "goal_position": [0.1, 0.1, 0.2],
        }
    ]
    details, report = compare_fingerprints(training, evaluation, tolerance=1e-4)
    assert details[0]["exact_duplicate"] is True
    assert report["num_exact_duplicates"] == 1
    assert report["status"] == "duplicate_present"


def test_state_dict_observation_fingerprint_extracts_named_fields() -> None:
    observation = {
        "agent": {
            "qpos": np.arange(18, dtype=np.float32).reshape(2, 9),
            "qvel": np.arange(18, dtype=np.float32).reshape(2, 9) + 100,
        },
        "extra": {
            "obj_pose": np.arange(14, dtype=np.float32).reshape(2, 7) + 200,
            "goal_pos": np.arange(6, dtype=np.float32).reshape(2, 3) + 300,
        },
    }

    fields = _state_fingerprint_from_observation(observation)

    assert set(fields) == {"robot_qpos", "cube_position", "cube_orientation", "goal_position"}
    assert fields["robot_qpos"].shape == (2, 9)
    np.testing.assert_array_equal(fields["cube_position"][0], [200, 201, 202])
    np.testing.assert_array_equal(fields["cube_orientation"][1], [210, 211, 212, 213])
    np.testing.assert_array_equal(fields["goal_position"][1], [303, 304, 305])


def test_convert_replay_publishes_only_native_successes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_h5, _ = _make_replay(tmp_path)
    from lerobot.datasets import lerobot_dataset

    class FakeDataset:
        def __init__(self, root: Path, features: dict[str, object]) -> None:
            self.root = root
            self.features = features
            self.frames: list[dict[str, object]] = []
            self.episodes = 0
            root.mkdir(parents=True)

        @classmethod
        def create(cls, **kwargs: object) -> FakeDataset:
            return cls(Path(kwargs["root"]), kwargs["features"])  # type: ignore[arg-type]

        def add_frame(self, frame: dict[str, object]) -> None:
            self.frames.append(frame)

        def save_episode(self, *, parallel_encoding: bool) -> None:
            assert parallel_encoding is False
            self.episodes += 1

        def finalize(self) -> None:
            meta = self.root / "meta"
            meta.mkdir(exist_ok=True)
            (meta / "info.json").write_text(
                json.dumps({"features": self.features}), encoding="utf-8"
            )

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", FakeDataset)
    output_dir = tmp_path / "dataset"
    survival_report = tmp_path / "replay_survival.json"
    action_stats_output = tmp_path / "action_channel_stats.json"
    result = convert_replay(
        replay_h5=replay_h5,
        source_h5=replay_h5,
        output_dir=output_dir,
        repo_id="local/test_pickcube",
        survival_report=survival_report,
        action_stats_output=action_stats_output,
    )

    assert result["episodes"] == 1
    assert result["frames"] == 2
    assert output_dir.is_dir()
    manifest = json.loads((output_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["output"]["episodes"] == 1
    assert manifest["output"]["failed_episodes"] == 0
    assert json.loads(action_stats_output.read_text(encoding="utf-8"))["action_dim"] == 7
    report = json.loads(survival_report.read_text(encoding="utf-8"))
    assert report["replays"]["pd_ee_delta_pose"]["converted_dataset"]["episodes"] == 1

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        convert_replay(
            replay_h5=replay_h5,
            output_dir=output_dir,
            repo_id="local/test_pickcube",
        )


def test_seed_split_and_demo_seed_isolation(tmp_path: Path) -> None:
    split_path = Path("configs/experiments/maniskill/split.json")
    split = load_seed_sets(split_path)
    sets = split["sets"]
    assert len(sets["train"]) == 2000
    assert len(sets["development"]) == 200
    assert len(sets["test"]) == 400
    assert split["sim_backend"] == "physx_cuda"
    assert split["state_match_tolerance"] == pytest.approx(1e-4)
    assert split["batches"]["development"][0]["head_seed"] == 2000
    assert split["batches"]["test"][0]["count"] == 400
    assert len(reset_batches_for_set(split, "development")[0]) == 200
    assert len(reset_batches_for_set(split, "test")[0]) == 400
    assert not (set(sets["train"]) & set(sets["development"]))
    assert not (set(sets["train"]) & set(sets["test"]))
    assert not (set(sets["development"]) & set(sets["test"]))

    metadata_path = tmp_path / "trajectory.json"
    metadata_path.write_text(
        json.dumps(
            {
                "env_info": {"env_id": "PickCube-v1"},
                "episodes": [
                    {"episode_id": 0, "reset_kwargs": {"seed": 17}},
                    {"episode_id": 1, "reset_kwargs": {"seed": 18}},
                ],
            }
        ),
        encoding="utf-8",
    )
    demo_seeds = collect_demonstration_seeds((metadata_path,))
    report = seed_isolation_report(split, demo_seeds)
    assert report["status"] == "disjoint"
    assert report["intersections"] == {"development": [], "test": []}

    overlapping = tmp_path / "overlapping.json"
    overlapping.write_text(
        json.dumps(
            {
                "env_info": {"env_id": "PickCube-v1"},
                "episodes": [{"reset_kwargs": {"seed": 2000}}],
            }
        ),
        encoding="utf-8",
    )
    overlapping_report = seed_isolation_report(
        split,
        collect_demonstration_seeds((overlapping,)),
    )
    assert overlapping_report["status"] == "overlap_present"
    effective, plan = evaluation_seed_plan(
        split,
        "development",
        collect_demonstration_seeds((overlapping,)),
    )
    assert len(effective) == 200
    assert effective[0] == 2000
    assert plan["status"] == "state_fingerprint_required"
