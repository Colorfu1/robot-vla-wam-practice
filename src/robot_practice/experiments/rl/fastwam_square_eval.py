"""Load a Square FastWAM training snapshot into the shared rollout protocol."""

import copy
import json
import logging
import os
import sys
from pathlib import Path

from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash

logger = logging.getLogger(__name__)


def load_weights_strict(model, payload):
    """Never silently retain pretrained video/action/proprio weights during evaluation."""
    if not {"mot", "proprio_encoder", "step"} <= payload.keys():
        raise ValueError("Expected complete FastWAM mot/proprio_encoder/step snapshot")
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    logger.info("Checkpoint loaded strict=true components=mot,proprio_encoder step=%s",
                payload["step"])


def load_fastwam(checkpoint, cfg):
    import torch

    from robot_practice.experiments.rl.lightwam_base import (
        LightWAMBaseAdapter,
        RobomimicLightWAMWindowDataset,
    )

    checkpoint = Path(checkpoint)
    metrics_path = checkpoint.parent.parent / "metrics.json"
    training = json.loads(metrics_path.read_text())
    if training["status"] != "complete" or training["mode"] != "train":
        raise ValueError("Requires completed training provenance")
    args, protocol = training["arguments"], training["protocol"]
    vendor, assets = Path(args["vendor"]).resolve(), Path(args["assets"]).resolve()
    source = vendor / "src/fastwam/models/wan22/fastwam.py"
    if file_hash(source) != protocol["source_hash"]:
        raise ValueError("FastWAM vendor source differs from training")
    for name, path in (("npz", cfg.data.npz), ("context", cfg.lightwam.context)):
        if file_hash(path) != protocol["data_sha256"][name]:
            raise ValueError(f"Training/evaluation {name} identity mismatch")
    dual = args.get("dual_data")
    image_size = (224, 448) if dual else (224, 224)
    horizon = 32 if dual else 4
    if list(cfg.lightwam.image_size) != list(image_size):
        raise ValueError(f"Checkpoint requires image size {image_size}; use matching evaluator")
    os.environ.update(DIFFSYNTH_MODEL_BASE_PATH=str(assets / "models"),
                      DIFFSYNTH_SKIP_DOWNLOAD="true", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(vendor / "src"))
    from fastwam.runtime import create_fastwam

    kwargs = copy.deepcopy(protocol["model"])
    kwargs["action_dit_pretrained_path"] = str(
        assets / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    )
    model = create_fastwam(**kwargs, device="cuda:0", model_dtype=torch.bfloat16)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    load_weights_strict(model, payload)
    step = int(payload["step"])
    del payload
    model.eval().requires_grad_(False)
    if dual:
        from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL, DualViewDataset

        if protocol.get("observation_action") != PROTOCOL:
            raise ValueError("Dual-view checkpoint protocol mismatch")
        if file_hash(Path(dual) / "manifest.json") != protocol["dual_manifest_sha256"]:
            raise ValueError("Dual-view dataset manifest changed")
        dataset = DualViewDataset(dual, cfg.lightwam.context)
    else:
        dataset = RobomimicLightWAMWindowDataset(
            cfg.data.npz, context_path=cfg.lightwam.context,
            action_horizon=4, video_frames=5, image_size=(224, 224), normalized=True,
        )
    # Both vendor APIs have the same infer_action signature and observation contract.
    # Reuse preprocessing only; dispatches to FastWAM, NOT the LightWAM action head.
    adapter = LightWAMBaseAdapter(
        model, action_horizon=horizon, action_dim=7, image_size=image_size,
        context=dataset.context, context_mask=dataset.context_mask,
        num_inference_steps=20, clip_actions=False,
    ).eval()
    provenance = {
        "training_metrics_sha256": file_hash(metrics_path),
        "vendor_sha256": file_hash(source),
        "normalization_sha256": file_hash(cfg.evaluation.normalization_path),
        "checkpoint_step": step,
        "strict_checkpoint_load": True,
        "inference_api": "FastWAM.infer_action",
        "inference_steps": 20,
        "noise_seed": "fixed per environment seed on every policy call",
        "video_inference": False,
        "model_config": kwargs,
        "action_horizon": horizon,
        "image_size": list(image_size),
    }
    return model, adapter, provenance
