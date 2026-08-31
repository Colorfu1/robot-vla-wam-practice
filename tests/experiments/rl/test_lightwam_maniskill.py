from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from robot_practice.experiments.rl.lightwam_maniskill import (
    LightWAMBasePolicy,
    PEG_PROPRIO_DIM,
    PegLightWAMContract,
    build_peg_lightwam_observation,
    extract_peg_proprio,
    preprocess_composite_rgb,
    resolve_text_embedding_cache,
)
from robot_practice.experiments.rl.lightwam_train import _dataset_config
from robot_practice.experiments.rl.maniskill_lightwam_residual import (
    compose_peg_residual_action,
)


def _observation() -> dict[str, object]:
    return {
        "agent": {
            "qpos": np.arange(9, dtype=np.float32),
            "qvel": np.arange(9, 18, dtype=np.float32),
            "base_pose": np.arange(7, dtype=np.float32),
        },
        "extra": {"tcp_pose": np.arange(7, dtype=np.float32) + 10},
        "sensor_data": {
            "base_camera": {"Color": np.zeros((128, 128, 3), dtype=np.uint8)},
            "hand_camera": {"rgb": np.full((128, 128, 3), 255, dtype=np.uint8)},
        },
    }


def test_peg_observation_uses_only_the_32d_non_privileged_contract() -> None:
    proprio = extract_peg_proprio(_observation())
    assert proprio.shape == (PEG_PROPRIO_DIM,)
    np.testing.assert_array_equal(proprio[:9], np.arange(9, dtype=np.float32))
    np.testing.assert_array_equal(proprio[9:18], np.arange(9, 18, dtype=np.float32))
    np.testing.assert_array_equal(proprio[18:25], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(proprio[25:], np.arange(7, dtype=np.float32) + 10)


def test_peg_observation_matches_official_two_camera_image_contract() -> None:
    processed = build_peg_lightwam_observation(_observation())
    assert tuple(processed.image.shape) == (3, 224, 448)
    assert processed.image.dtype == torch.float32
    # Bicubic interpolation has the same small overshoot as the official
    # torchvision preprocessing path; the important contract is finite
    # float32 [C,H,W], not a post-resize clamp that would change that path.
    assert bool(torch.isfinite(processed.image).all())
    assert processed.proprio.shape == (PEG_PROPRIO_DIM,)


def test_peg_observation_accepts_upstream_image_alias() -> None:
    observation = _observation()
    observation["image"] = observation.pop("sensor_data")
    processed = build_peg_lightwam_observation(observation)
    assert tuple(processed.image.shape) == (3, 224, 448)


def test_composite_preprocessing_resizes_and_center_crops() -> None:
    image = np.zeros((64, 128, 3), dtype=np.uint8)
    output = preprocess_composite_rgb(image, video_size=(32, 64))
    assert tuple(output.shape) == (3, 32, 64)
    torch.testing.assert_close(output, torch.full((3, 32, 64), -1.0))


def test_zero_peg_residual_is_exact_base_action_and_all_channels_are_explicit() -> None:
    base = np.linspace(-0.5, 0.5, 16 * 7, dtype=np.float32).reshape(16, 7)
    residual = np.zeros_like(base)
    composed = compose_peg_residual_action(
        base,
        residual,
        action_std=np.ones(7, dtype=np.float32),
        res_scale=0.1,
        action_low=np.full(7, -1.0, dtype=np.float32),
        action_high=np.full(7, 1.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(composed, base)


def test_peg_residual_is_scaled_in_physical_action_units_and_clipped() -> None:
    base = np.zeros((2, 7), dtype=np.float32)
    residual = np.ones((2, 7), dtype=np.float32)
    composed = compose_peg_residual_action(
        base,
        residual,
        action_std=np.full(7, 2.0, dtype=np.float32),
        res_scale=0.5,
        action_low=np.full(7, -0.25, dtype=np.float32),
        action_high=np.full(7, 0.25, dtype=np.float32),
    )
    np.testing.assert_allclose(composed, 0.25 * np.ones((2, 7), dtype=np.float32))


def test_short_residual_only_changes_current_replan_window() -> None:
    base = np.zeros((16, 7), dtype=np.float32)
    residual = np.ones((4, 7), dtype=np.float32)
    composed = compose_peg_residual_action(
        base,
        residual,
        action_std=np.ones(7, dtype=np.float32),
        res_scale=0.1,
        action_low=np.full(7, -1.0, dtype=np.float32),
        action_high=np.full(7, 1.0, dtype=np.float32),
    )
    np.testing.assert_allclose(composed[:4], 0.1 * np.ones((4, 7), dtype=np.float32))
    np.testing.assert_array_equal(composed[4:], base[4:])


def test_contract_defaults_match_peg_insertion_side() -> None:
    contract = PegLightWAMContract()
    assert contract.env_id == "PegInsertionSide-v2"
    assert contract.proprio_dim == 32
    assert contract.action_dim == 7
    assert contract.camera_names == ("base_camera", "hand_camera")


def test_official_peg_training_config_uses_sixteen_action_horizon(tmp_path: Path) -> None:
    config = _dataset_config(
        data_root=tmp_path,
        video_height=224,
        video_width=448,
        text_cache="text_embeds",
        stats_path=None,
        seed=1,
        val_set_proportion=0.0,
        is_training_set=True,
        override_instruction="insert the peg into the hole",
    )
    assert config["num_frames"] == 17
    assert config["action_video_freq_ratio"] == 4
    assert config["num_frames"] - 1 == 16


def test_official_text_cache_resolution_matches_lightwam_filename(tmp_path) -> None:
    prompt = "insert the peg into the hole"
    official_prompt = (
        "A video recorded from a robot's point of view executing the following instruction: "
        + prompt
    )
    expected = tmp_path / (
        f"{hashlib.sha256(official_prompt.encode('utf-8')).hexdigest()}"
        ".t5_len128.wan21t2v13b.pt"
    )
    expected.touch()
    resolved = resolve_text_embedding_cache(tmp_path, prompt_or_task=prompt)
    assert resolved.name.endswith(".t5_len128.wan21t2v13b.pt")
    assert resolve_text_embedding_cache(tmp_path, prompt_or_task=prompt) == resolved


def test_lightwam_base_forwards_seed_for_paired_diffusion_evaluation() -> None:
    class FakeNormalizer:
        def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
            return state

        def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
            return action[0] if action.ndim == 3 else action

    class FakeModel:
        torch_dtype = torch.float32

        def __init__(self) -> None:
            self.seen_seed = None

        def infer_action(self, **kwargs):
            self.seen_seed = kwargs["seed"]
            return {"action": torch.zeros(1, 16, 7)}

    policy = object.__new__(LightWAMBasePolicy)
    policy.contract = PegLightWAMContract()
    policy.device = torch.device("cpu")
    policy.model = FakeModel()
    policy.normalizer = FakeNormalizer()
    policy.context = torch.zeros(128, 4)
    policy.context_mask = torch.ones(128, dtype=torch.bool)
    policy.prompt = None
    policy.num_inference_steps = 4

    action = policy.predict(_observation(), seed=123000007)

    assert action.shape == (16, 7)
    assert policy.model.seen_seed == 123000007
