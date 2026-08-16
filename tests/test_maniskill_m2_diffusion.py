from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robot_practice.experiments.maniskill import m2_diffusion

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PRACTICE_ROOT / "configs/experiments/maniskill/m2_diffusion_policy.json"
M2_CONFIG = PRACTICE_ROOT / "configs/experiments/maniskill/m2_w_action_only_flow.json"
LAUNCHER = PRACTICE_ROOT / "scripts/experiments/maniskill/train_m2_diffusion.sh"


def test_adapter_config_is_explicitly_lazy_rgb_video_storage() -> None:
    config = m2_diffusion.validate_adapter_config(CONFIG)

    assert config["dataset"]["storage"] == "video"
    assert config["dataset"]["image_dtype"] == "uint8"
    assert config["dataset"]["image_layout"] == "HWC"
    assert config["dataset"]["image_color"] == "RGB"
    assert config["dataset"]["image_color_stage"] == "decoded_model_input"
    assert config["dataset"]["video_codec"] == "h264"
    assert config["policy"]["normalization_mapping"] == {
        "VISUAL": "MEAN_STD",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }


def test_split_selection_contains_train_and_development_but_no_test() -> None:
    m2_config = m2_diffusion.load_config(M2_CONFIG)
    selection = m2_diffusion.build_split_selection(m2_config)

    assert len(selection["train_source_episode_ids"]) == 787
    assert len(selection["development_source_episode_ids"]) == 100
    assert len(selection["train_dataset_episode_indices"]) == 787
    assert len(selection["development_dataset_episode_indices"]) == 100
    assert selection["test_dataset_episode_indices"] == []
    assert selection["test_access_allowed"] is False
    assert set(selection["train_dataset_episode_indices"]).isdisjoint(
        selection["development_dataset_episode_indices"]
    )


def test_native_config_round_trips_through_installed_lerobot(tmp_path: Path) -> None:
    adapter_config = m2_diffusion.validate_adapter_config(CONFIG)
    native = m2_diffusion._build_native_train_config(
        adapter_config=adapter_config,
        dataset_root=tmp_path / "rgb_dataset",
        repo_id="local/test_m2_diffusion_rgb",
        train_episode_indices=[0, 1, 2],
        output_dir=tmp_path / "run",
        job_name="test_m2_diffusion",
        batch_size=1,
        steps=1,
        save_freq=1,
        log_freq=1,
        seed=1000,
        num_workers=0,
    )
    path = tmp_path / "train_config.json"
    path.write_text(json.dumps(native), encoding="utf-8")

    parsed = TrainPipelineConfig.from_pretrained(path)
    assert parsed.policy is not None
    assert parsed.policy.type == "diffusion"
    assert parsed.dataset.repo_id == "local/test_m2_diffusion_rgb"
    assert parsed.dataset.episodes == [0, 1, 2]
    assert parsed.dataset.use_imagenet_stats is False
    assert parsed.policy.normalization_mapping["VISUAL"].value == "MEAN_STD"
    assert parsed.policy.normalization_mapping["STATE"].value == "MEAN_STD"
    assert parsed.policy.normalization_mapping["ACTION"].value == "MEAN_STD"
    assert parsed.eval_freq == 0
    assert parsed.wandb.enable is False
    assert parsed.policy.push_to_hub is False


def test_video_backed_dataset_decodes_rgb_model_input(tmp_path: Path) -> None:
    root = tmp_path / "video_dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/test_m2_video",
        root=root,
        robot_type="panda",
        fps=20,
        features=m2_diffusion._video_features(),
        use_videos=True,
        vcodec="h264",
        encoder_threads=2,
    )
    for frame_index in range(3):
        image = np.full((128, 128, 3), frame_index * 20, dtype=np.uint8)
        dataset.add_frame(
            {
                "observation.images.base_camera": image,
                "observation.images.hand_camera": image,
                "observation.state": np.zeros(18, dtype=np.float32),
                "action": np.zeros(7, dtype=np.float32),
                "task": "StackCube-v1",
            }
        )
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()

    m2_diffusion._verify_rgb_dataset(
        root,
        repo_id="local/test_m2_video",
        train_indices=[0],
        development_indices=[],
        expected_episode_count=1,
    )
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["video_path"]
    assert all(info["features"][key]["dtype"] == "video" for key in m2_diffusion.IMAGE_KEYS)


def _run_launcher(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=PRACTICE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launcher_prints_prepare_and_native_pipeline() -> None:
    result = _run_launcher("--num-gpus=8", "--print-command")

    assert result.returncode == 0, result.stderr
    assert "m2_diffusion prepare" in result.stdout
    assert "accelerate launch" in result.stdout
    assert "--multi_gpu" in result.stdout
    assert "--num_processes=8" in result.stdout
    assert "-m lerobot.scripts.lerobot_train" in result.stdout
    assert "--config_path=outputs/maniskill/m2_diffusion/configs/" in result.stdout
    assert "--mixed_precision=bf16" in result.stdout


def test_launcher_requires_one_gpu_count_and_rejects_duplicates() -> None:
    missing = _run_launcher("--print-command")
    duplicate = _run_launcher("--num-gpus=8", "--num-gpu=8", "--print-command")

    assert missing.returncode == 64
    assert "--num-gpus=N is required exactly once" in missing.stderr
    assert duplicate.returncode == 64
    assert "duplicate --num-gpus/--num-gpu" in duplicate.stderr
