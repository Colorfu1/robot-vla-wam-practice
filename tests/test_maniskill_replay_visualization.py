from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import rerun.dataframe as rdf

from robot_practice.experiments.maniskill.panda_kinematics import PandaKinematics
from robot_practice.experiments.maniskill.replay_visualization import (
    load_replay_episode,
    select_episode_ids,
    write_replay_visualization,
)
from robot_practice.rerun_launcher import (
    OFFLINE_APPLICATION_ID,
    RecordingKind,
    detect_rrd_kind,
)


def _make_replay(tmp_path: Path) -> Path:
    replay_h5 = tmp_path / "trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5"
    metadata = {
        "env_info": {
            "env_id": "PickCube-v1",
            "env_kwargs": {
                "control_mode": "pd_ee_delta_pose",
                "sim_backend": "physx_cpu",
            },
        },
        "episodes": [
            {
                "episode_id": 0,
                "reset_kwargs": {"seed": 200},
                "success": True,
            }
        ],
    }
    replay_h5.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    with h5py.File(replay_h5, "w") as handle:
        episode = handle.create_group("traj_0")
        action_count = 3
        frame_count = action_count + 1
        episode.create_dataset(
            "actions",
            data=np.arange(action_count * 7, dtype=np.float32).reshape(action_count, 7),
        )
        episode.create_dataset("success", data=np.asarray([False, True, True]))
        episode.create_dataset("terminated", data=np.asarray([False, False, True]))
        episode.create_dataset("truncated", data=np.asarray([False, False, False]))
        obs = episode.create_group("obs")
        sensor_data = obs.create_group("sensor_data")
        camera = sensor_data.create_group("base_camera")
        camera.create_dataset(
            "rgb",
            data=np.arange(frame_count * 128 * 128 * 3, dtype=np.uint8).reshape(
                frame_count, 128, 128, 3
            ),
        )
        agent = obs.create_group("agent")
        agent.create_dataset(
            "qpos", data=np.arange(frame_count * 9, dtype=np.float32).reshape(frame_count, 9)
        )
        agent.create_dataset(
            "qvel",
            data=(100 + np.arange(frame_count * 9, dtype=np.float32)).reshape(frame_count, 9),
        )
        extra = obs.create_group("extra")
        extra.create_dataset(
            "goal_pos",
            data=(200 + np.arange(frame_count * 3, dtype=np.float32)).reshape(frame_count, 3),
        )
        extra.create_dataset("is_grasped", data=np.asarray([False, False, True, True]))
        extra.create_dataset(
            "tcp_pose",
            data=np.concatenate(
                [
                    np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3),
                    np.tile([1.0, 0.0, 0.0, 0.0], (frame_count, 1)),
                ],
                axis=1,
            ),
        )
        env_states = episode.create_group("env_states")
        actors = env_states.create_group("actors")
        cube = np.zeros((frame_count, 13), dtype=np.float32)
        cube[:, :3] = np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3)
        actors.create_dataset("cube", data=cube)
        goal = np.zeros((frame_count, 13), dtype=np.float32)
        goal[:, :3] = 10 + np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3)
        actors.create_dataset("goal_site", data=goal)
    return replay_h5


def test_select_episode_ids_supports_prefix_and_explicit_values() -> None:
    assert select_episode_ids([3, 1, 2], None, max_episodes=2) == (1, 2)
    assert select_episode_ids([3, 1, 2], [3, 1], max_episodes=1) == (3, 1)


def test_load_replay_episode_preserves_observation_and_action_alignment(tmp_path: Path) -> None:
    replay_h5 = _make_replay(tmp_path)
    with h5py.File(replay_h5, "r") as handle:
        episode = load_replay_episode(
            handle["traj_0"],
            episode_id=0,
            metadata_record={"reset_kwargs": {"seed": 200}},
        )

    assert episode.seed == 200
    assert episode.images.shape == (4, 128, 128, 3)
    assert episode.state.shape == (4, 21)
    assert episode.actions.shape == (3, 7)
    assert episode.first_success_step == 2
    assert episode.success_once is True
    assert episode.success_at_end is True
    assert episode.is_grasped is not None
    assert episode.tcp_pose is not None
    assert episode.cube_position is not None
    assert episode.goal_position is not None
    np.testing.assert_array_equal(episode.state[0, :9], np.arange(9, dtype=np.float32))
    np.testing.assert_array_equal(episode.state[0, 18:], [200.0, 201.0, 202.0])


def test_write_replay_visualization_creates_rrd_and_manifest(tmp_path: Path) -> None:
    replay_h5 = _make_replay(tmp_path)
    output_dir = tmp_path / "visualization"
    manifest = write_replay_visualization(
        replay_h5=replay_h5,
        output_dir=output_dir,
        episode_ids=[0],
        include_privileged_overlay=True,
    )

    recording = output_dir / "episode_0000_success.rrd"
    assert recording.is_file()
    assert recording.stat().st_size > 0
    assert manifest["selected_episode_ids"] == [0]
    assert manifest["viewer"]["kind"] == "offline"
    assert manifest["viewer"]["application_id"] == OFFLINE_APPLICATION_ID
    assert manifest["episodes"][0]["has_privileged_scene_state"] is True
    assert manifest["episodes"][0]["result"] == "success"
    assert manifest["episodes"][0]["viewer_kind"] == "offline"
    assert manifest["episodes"][0]["application_id"] == OFFLINE_APPLICATION_ID
    saved_manifest = json.loads(
        (output_dir / "visualization_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["episodes"][0]["recording"].endswith("episode_0000_success.rrd")
    assert detect_rrd_kind(recording) is RecordingKind.OFFLINE
    recording_ids = {
        str(item.application_id()) for item in rdf.load_archive(recording).all_recordings()
    }
    assert recording_ids == {OFFLINE_APPLICATION_ID}


def test_panda_ik_reconstructs_a_recorded_pickcube_pose() -> None:
    kinematics = PandaKinematics()
    arm_qpos = np.asarray(
        [-0.02142199, 0.39203027, -0.00224465, -1.96494925, -0.00391862, 2.37847877, 0.77302724],
        dtype=np.float64,
    )
    target_tcp_pose = np.asarray(
        [
            0.00399399,
            -0.01500511,
            0.17159641,
            0.00199049,
            0.999932,
            -0.00415629,
            0.01071513,
        ],
        dtype=np.float64,
    )

    pose = kinematics.solve(
        target_tcp_pose,
        seed_qpos=arm_qpos,
        gripper_qpos=[0.04, 0.04],
    )

    assert pose.ik_converged is True
    assert pose.used_qpos_fallback is False
    assert pose.skeleton_points.shape == (10, 3)
    assert pose.left_finger_points.shape == (3, 3)
    assert pose.right_finger_points.shape == (3, 3)
    assert pose.position_residual_m < 1e-4
    assert pose.rotation_residual_rad < 1e-3
