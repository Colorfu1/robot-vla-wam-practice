import numpy as np
import pytest
from PIL import Image

from robot_practice.experiments.rl.lightwam_future_preview import find_eval_observation, save_frames


def test_observation_uses_only_selected_seed_and_current_frame(tmp_path):
    directory = tmp_path / "gpu_0/batch_000/forensics"
    directory.mkdir(parents=True)
    np.savez(
        directory / "chunk_0010_input.npz",
        environment_seeds=[10001, 10002],
        rgb=np.zeros((2, 1, 3, 96, 96)),
        state=np.ones((2, 1, 9)),
    )
    _, obs = find_eval_observation(tmp_path, 10001, 10)
    assert set(obs) == {"rgb", "state"}
    assert obs["rgb"].shape == (3, 96, 96)
    assert obs["state"].shape == (9,)
    with pytest.raises(ValueError):
        find_eval_observation(tmp_path, 999, 10)


def test_save_five_frame_preview(tmp_path):
    save_frames([Image.new("RGB", (16, 16), (i * 30, 0, 0)) for i in range(5)], tmp_path)
    assert Image.open(tmp_path / "prediction_strip.png").size == (80, 16)
    assert Image.open(tmp_path / "prediction_slow.gif").n_frames == 5
    with pytest.raises(ValueError):
        save_frames([], tmp_path)


@pytest.mark.parametrize("factor", [1, 2])
def test_sampling_stays_in_training_latent_space(factor):
    from types import SimpleNamespace

    import torch

    from robot_practice.experiments.rl.lightwam_future_preview import predict_future_video

    calls = []
    full = torch.ones(1, 2, 1, 8, 8)
    low = torch.ones(1, 2, 1, 8 // factor, 8 // factor)

    def predict(**kw):
        x = kw["latents_video"]
        assert x.shape == (1, 2, 2, 8 // factor, 8 // factor)
        assert torch.equal(x[:, :, :1], low)
        assert kw["apply_spatial_downsample"] is False
        assert kw["restore_spatial_resolution"] is False
        assert kw["action"] is None
        calls.append(x.clone())
        return torch.ones_like(x)

    def restore(x, meta):
        assert len(calls) == 3  # Restore only after all denoising steps.
        return torch.nn.functional.interpolate(x, size=(2, 8, 8), mode="nearest")

    model = SimpleNamespace(
        device="cpu", torch_dtype=torch.float32,
        vae=SimpleNamespace(temporal_downsample_factor=4),
        video_expert=SimpleNamespace(fuse_vae_embedding_in_latents=False),
        _encode_input_image_latents_tensor=lambda **kw: full.clone(),
        _maybe_downsample_video_latents_for_backbone=lambda x: (low.clone(), {}),
        _append_proprio_to_context=lambda **kw: (kw["context"], kw["context_mask"]),
        _predict_video_only=predict,
        _restore_video_prediction_spatial_resolution=restore,
        _decode_latents=lambda x: x,
        infer_video_scheduler=SimpleNamespace(
            build_inference_schedule=lambda **kw: (torch.ones(3), torch.ones(3) / 3),
            step=lambda pred, delta, x: x - delta * pred,
        ),
    )
    result = predict_future_video(
        model=model, image=torch.zeros(3, 64, 64), proprio=torch.zeros(9),
        context=torch.zeros(1, 2, 3), context_mask=torch.ones(1, 2, dtype=torch.bool),
        steps=3, seed=0,
    )
    assert result["video"].shape == (1, 2, 2, 8, 8)
    assert torch.equal(result["video"][:, :, :1], full)
    assert result["sampling"]["sampling_latent_shape"][-2:] == [8 // factor] * 2
