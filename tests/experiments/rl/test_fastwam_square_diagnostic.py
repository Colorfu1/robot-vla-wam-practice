import numpy as np
import torch
from PIL import Image
from torch import nn

from robot_practice.experiments.rl.fastwam_square_diagnostic import frame_metrics, preview
from robot_practice.experiments.rl.lightwam_base import LightWAMBaseAdapter


def test_future_metrics_exclude_first_and_missing():
    dark = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    bright = Image.fromarray(np.full((224, 224, 3), 100, dtype=np.uint8))
    metrics = frame_metrics([dark] * 5, [bright, None, None, None, bright], dark)
    assert set(metrics) == {"4"}
    assert metrics["4"]["prediction_mae_0_255"] == 100
    assert metrics["4"]["copy_first_mae_0_255"] == 100


def test_preview_gt_is_only_used_after_generation(tmp_path):
    calls = []
    frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    class FakeFastWAM(nn.Module):
        device = "cpu"
        torch_dtype = torch.float32

        def infer_joint(self, **kwargs):
            calls.append("generate")
            assert kwargs["action"] is None
            assert kwargs["num_video_frames"] == 5
            assert kwargs["num_inference_steps"] == 20
            assert kwargs["test_action_with_infer_action"] is True
            assert kwargs["input_image"].shape == (3, 224, 224)
            assert kwargs["proprio"].shape == (9,)
            assert not any("gt" in k or "future" in k for k in kwargs)
            return {"video": [frame] * 5, "action": torch.zeros(4, 7)}

        def _encode_video_latents(self, video):
            calls.append("oracle_encode")
            assert video.shape == (1, 3, 5, 224, 224)
            return torch.zeros(1, 48, 2, 14, 14)

        def _decode_latents(self, latent):
            return [frame] * 5

    model = FakeFastWAM()
    adapter = LightWAMBaseAdapter(
        model, context=torch.zeros(2, 4096), context_mask=torch.ones(2),
        num_inference_steps=20,
    )
    output = tmp_path / "case"
    metrics = preview(
        model, adapter, {"rgb": torch.zeros(3, 96, 96), "state": torch.zeros(9)},
        [frame] * 5, output, seed=0, gt_video=torch.zeros(3, 5, 224, 224),
    )
    assert calls == ["generate", "oracle_encode"]
    assert set(metrics["future_frames"]) == {"1", "2", "3", "4"}
    assert metrics["vae_latent_shape"] == [1, 48, 2, 14, 14]
    assert (output / "comparison.png").is_file()
    assert (output / "prediction_strip.png").is_file()
    assert (output / "prediction_slow.gif").is_file()
