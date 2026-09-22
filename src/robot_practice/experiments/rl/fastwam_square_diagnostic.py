"""FastWAM sampled-action error and native future-video previews (no training)."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam
from robot_practice.experiments.rl.lightwam_base import RobomimicLightWAMWindowDataset
from robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic import (
    _evaluate_model,
    trajectory_windows,
)
from robot_practice.experiments.rl.lightwam_future_preview import find_eval_observation, save_frames
from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash, write_json

logger = logging.getLogger(__name__)


def image_from_tensor(image):
    """CHW [-1,1] -> RGB, without display-only scaling or sharpening."""
    array = ((image.detach().float().cpu().clamp(-1, 1) + 1) * 127.5)
    return Image.fromarray(array.round().to(torch.uint8).permute(1, 2, 0).numpy())


def frame_metrics(prediction, truth, first):
    """Only score available future GT; exclude the conditioning frame."""
    rows = {}
    for t, frame in enumerate(truth):
        if t == 0 or frame is None:
            continue
        gt = np.asarray(frame, dtype=np.float32)
        pred = np.asarray(prediction[t], dtype=np.float32)
        baseline = np.asarray(first, dtype=np.float32)
        if pred.shape != gt.shape or baseline.shape != gt.shape:
            raise ValueError("Image shapes differ")
        rows[str(t)] = {
            "prediction_mae_0_255": float(np.abs(pred - gt).mean()),
            "prediction_mse_0_255": float(np.square(pred - gt).mean()),
            "copy_first_mae_0_255": float(np.abs(baseline - gt).mean()),
        }
    return rows


def comparison_sheet(rows, path):
    width = height = 224
    sheet = Image.new("RGB", (160 + width * 5, 26 + len(rows) * (height + 8)), "white")
    draw = ImageDraw.Draw(sheet)
    for t in range(5):
        draw.text((160 + t * width, 5), f"t+{t} ({t * 50} ms)", fill="black")
    for r, (label, frames) in enumerate(rows):
        y = 26 + r * (height + 8)
        draw.text((4, y + 8), label, fill="black")
        for t, frame in enumerate(frames):
            if frame is None:
                draw.text((170 + t * width, y + 100), "NOT RECORDED", fill="black")
            else:
                if frame.size != (224, 224):
                    raise ValueError("Expected native 224x224 previews")
                sheet.paste(frame, (160 + t * width, y))
    sheet.save(path)


@torch.inference_mode()
def preview(model, adapter, cond, truth, output, seed, gt_video=None):
    output.mkdir()
    rgb, state = adapter._prepare_observation(cond)
    logger.info("START phase=video case=%s seed=%d", output.name, seed)
    generated = model.infer_joint(
        prompt=None, input_image=rgb[0].to(model.device),
        proprio=state[0].to(model.device),
        context=adapter.context.to(model.device),
        context_mask=adapter.context_mask.to(model.device),
        num_video_frames=5, action_horizon=4, action=None,
        num_inference_steps=20, seed=seed,
        test_action_with_infer_action=True,
    )
    frames = generated["video"]
    save_frames(frames, output)
    np.save(output / "joint_action.npy", generated["action"].cpu().numpy())
    first = image_from_tensor(rgb[0])
    metrics = {"seed": seed, "future_frames": frame_metrics(frames, truth, first)}
    rows = [("Reference", truth), ("Prediction", frames), ("Copy first", [first] * 5)]
    if gt_video is not None:
        # Oracle reconstruction control ONLY: GT is encoded after free generation.
        latent = model._encode_video_latents(
            gt_video.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        )
        if isinstance(latent, list):
            latent = torch.stack(latent)
        reconstructed = model._decode_latents(latent)
        if len(reconstructed) != 5:
            raise ValueError("VAE reconstruction must contain five frames")
        rows.append(("GT VAE oracle", reconstructed))
        metrics["vae_latent_shape"] = list(latent.shape)
        metrics["vae_reconstruction"] = frame_metrics(reconstructed, truth, first)
    for label, row in rows:
        prefix = label.lower().replace(" ", "_")
        for t, frame in enumerate(row):
            if frame is not None:
                frame.save(output / f"{prefix}_{t}.png")
    comparison_sheet(rows, output / "comparison.png")
    metrics["protocol"] = {
        "api": "FastWAM.infer_joint", "denoising_steps": 20,
        "future_gt_used_for_prediction": False, "gt_action_conditioning": False,
        "frame0_is_conditioning_not_future": True,
        "reference_future_indices": [int(t) for t in metrics["future_frames"]],
        "note": "GT VAE oracle uses future GT only for reconstruction, not generation. "
                "Image MAE is descriptive, not a task success metric.",
    }
    write_json(output / "metrics.json", metrics)
    logger.info("DONE phase=video case=%s", output.name)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/experiments/rl/lightwam_base_robomimic_square.yaml")
    parser.add_argument("--trajectory-ids", default="0,10,20,30,40,50,60,70")
    parser.add_argument("--video-frames", default="0,40,80", help="Start frames in NPZ trajectory 0")
    parser.add_argument("--eval-seed", type=int, default=10001)
    parser.add_argument("--eval-chunk", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()
    if args.eval_chunk < 0:
        parser.error("eval-chunk must be nonnegative")
    cfg = OmegaConf.load(args.config)
    dataset = RobomimicLightWAMWindowDataset(
        cfg.data.npz, context_path=cfg.lightwam.context,
        action_horizon=4, video_frames=5, image_size=(224, 224), normalized=True,
    )
    indices, records = trajectory_windows(dataset, [int(t) for t in args.trajectory_ids.split(",")])
    _, train_records = trajectory_windows(dataset, [0])
    by_frame = {r["frame"]: r["window_index"] for r in train_records}
    frame_ids = [int(f) for f in args.video_frames.split(",")]
    if len(set(frame_ids)) != len(frame_ids) or any(f not in by_frame for f in frame_ids):
        parser.error("video-frames must be unique valid trajectory-0 window start frames")
    eval_path, eval_obs = find_eval_observation(args.eval_root, args.eval_seed, args.eval_chunk)
    next_path, next_obs = find_eval_observation(args.eval_root, args.eval_seed, args.eval_chunk + 1)
    eval_metrics = json.loads((Path(args.eval_root) / "metrics.json").read_text())
    checkpoint_hash = file_hash(args.checkpoint)
    if eval_metrics.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Recorded evaluation used a different checkpoint")
    if eval_metrics.get("arguments", {}).get("policy") != "fastwam":
        raise ValueError("Expected FastWAM rollout reference")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "run.log"),
    ])
    result = {
        "status": "running", "arguments": vars(args), "checkpoint_sha256": checkpoint_hash,
        "training_set_only": True,
        "limitations": ["Open-loop action error is not closed-loop success.",
                        "Video spans only 0.2 seconds; first frame is conditioned.",
                        ("Training trajectory previews are GT demo observations, not successful "
                         "FastWAM closed-loop cases."),
                        "Eval GT intermediate frames are absent; only endpoint is scored.",
                        "Action-unconditioned video need not match the recorded action future."],
    }
    write_json(output / "trajectory_selection.json", {"windows": len(indices), "records": records})
    write_json(output / "metrics.json", result)
    try:
        logger.info("START phase=load completed=0 total=1")
        model, adapter, provenance = load_fastwam(args.checkpoint, cfg)
        result["provenance"] = provenance
        result["video"] = {}
        for f in frame_ids:
            item = dataset[by_frame[f]]
            video = item["video"]
            name = f"train_trajectory_0_frame_{f:04d}"
            result["video"][name] = preview(
                model, adapter, {"rgb": video[:, 0], "state": item["proprio"][0]},
                [image_from_tensor(video[:, t]) for t in range(5)],
                output / name, args.sample_seed, gt_video=video,
            )
            write_json(output / "metrics.json", result)
        current_rgb, _ = adapter._prepare_observation(eval_obs)
        next_rgb, _ = adapter._prepare_observation(next_obs)
        name = f"eval_{args.eval_seed}_chunk_{args.eval_chunk:04d}"
        result["video"][name] = preview(
            model, adapter, eval_obs,
            [image_from_tensor(current_rgb[0]), None, None, None, image_from_tensor(next_rgb[0])],
            output / name, args.eval_seed,
        )
        result["video"][name]["reference_files"] = [str(eval_path), str(next_path)]
        write_json(output / "metrics.json", result)
        logger.info("START phase=open_loop completed=0 total=%d", len(indices))
        offline = output / "open_loop"
        offline.mkdir()
        with (offline / "run.log").open("w") as log_file:
            result["open_loop"] = _evaluate_model(
                model=model, dataset=dataset, indices=indices,
                context=dataset.context, context_mask=dataset.context_mask,
                action_horizon=4, action_dim=7, image_size=(224, 224),
                num_inference_steps=20, seed=args.sample_seed, label=Path(args.checkpoint).stem,
                output_dir=offline, log_file=log_file,
            )
        write_json(offline / "metrics.json", result["open_loop"])
        result["status"] = "complete"
        logger.info("DONE phase=all completed=2 total=2 windows=%d", len(indices))
    except BaseException:
        result["status"] = "failed"
        logger.exception("Diagnostic failed; completed previews retained")
        raise
    finally:
        write_json(output / "metrics.json", result)


if __name__ == "__main__":
    main()
