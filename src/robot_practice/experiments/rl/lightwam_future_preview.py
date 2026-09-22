"""Free-running future-frame prediction from two recorded rollout cases, no future GT input."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_practice.experiments.rl.lightwam_train_scene_eval import Simulator, write_json

logger = logging.getLogger(__name__)


def predict_future_video(model, image, proprio, context, context_mask, steps=20, seed=0):
    """Preview-only sampler matching the low-resolution video training objective.

    No future observations/actions enter this function. Upsampling happens once,
    after denoising, not on each predicted flow. The action inference path is untouched.
    Caller must use inference_mode and an eval-mode model.
    """
    import torch

    image = image.to(device=model.device, dtype=model.torch_dtype)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    first_full = model._encode_input_image_latents_tensor(input_image=image)
    first_low, meta = model._maybe_downsample_video_latents_for_backbone(first_full)
    shape = list(first_low.shape)
    shape[2] = 4 // model.vae.temporal_downsample_factor + 1
    # Independent unit-variance noise on the TRAINING grid, not pooled full-grid noise.
    latents = torch.randn(
        shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32
    ).to(device=model.device, dtype=model.torch_dtype)
    latents[:, :, :1] = first_low
    context = context.to(device=model.device, dtype=model.torch_dtype)
    context_mask = context_mask.to(device=model.device, dtype=torch.bool)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    context, context_mask = model._append_proprio_to_context(
        context=context, context_mask=context_mask,
        proprio=proprio.reshape(1, -1).to(device=model.device, dtype=model.torch_dtype),
    )
    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=steps, device=model.device, dtype=latents.dtype,
        shift_override=None,
    )
    for i, (timestep, delta) in enumerate(zip(timesteps, deltas), 1):
        prediction = model._predict_video_only(
            latents_video=latents, timestep_video=timestep.unsqueeze(0),
            context=context, context_mask=context_mask, action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
            apply_spatial_downsample=False, restore_spatial_resolution=False,
        )
        if prediction.shape != latents.shape:
            raise ValueError("Predicted flow must match the training-space sampling grid")
        latents = model.infer_video_scheduler.step(prediction, delta, latents)
        latents[:, :, :1] = first_low
        logger.info("STATUS phase=denoise completed=%d total=%d", i, len(timesteps))
    decoded_latents = model._restore_video_prediction_spatial_resolution(latents, meta)
    # Preserve the known conditioning frame; this is NOT a predicted future frame.
    decoded_latents[:, :, :1] = first_full
    return {
        "video": model._decode_latents(decoded_latents),
        "sampling": {
            "protocol": "training_grid_noise_and_denoising_v1",
            "sampling_latent_shape": shape,
            "decode_latent_shape": list(decoded_latents.shape),
            "conditioning_frame": "fixed; restored from full-resolution input latent",
        },
    }


def find_eval_observation(root, seed, chunk):
    matches = []
    for path in Path(root).glob(f"gpu_*/batch_*/forensics/chunk_{chunk:04d}_input.npz"):
        with np.load(path) as data:
            ids = data["environment_seeds"].tolist()
            if seed in ids:
                i = ids.index(seed)
                matches.append((path, {k: data[k][i, -1].copy() for k in ("rgb", "state")}))
    if len(matches) != 1:
        raise ValueError(f"Expected one recorded eval observation, found {len(matches)}")
    return matches[0]


def save_frames(frames, output):
    from PIL import Image

    if len(frames) != 5:
        raise ValueError(f"Expected five generated frames, got {len(frames)}")
    frames = [frame.convert("RGB") for frame in frames]
    for i, frame in enumerate(frames):
        frame.save(output / f"pred_{i}.png")
    frames[0].save(
        output / "prediction_slow.gif",
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
    )
    sheet = Image.new("RGB", (frames[0].width * 5, frames[0].height))
    for i, frame in enumerate(frames):
        sheet.paste(frame, (i * frame.width, 0))
    sheet.save(output / "prediction_strip.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite preview images, metrics and logs in the output directory",
    )
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=10001)
    p.add_argument("--trajectory", type=int, default=0)
    p.add_argument("--inference-steps", type=int, default=20)
    p.add_argument("--config", default="configs/experiments/rl/lightwam_base_robomimic_square.yaml")
    p.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    args = p.parse_args()
    if args.chunk < 0 or args.inference_steps < 1:
        p.error("Invalid chunk or inference-steps")
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from robot_practice.experiments.rl.lightwam_base import (
        LightWAMBaseAdapter,
        RobomimicLightWAMWindowDataset,
        _resolve_dppo_runtime,
        load_official_lightwam,
    )
    from robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic import MODEL_DTYPES

    cfg = OmegaConf.load(args.config)
    checkpoint = Path(args.checkpoint)
    train_root = Path(args.train_root)
    metrics = json.loads((train_root / "metrics.json").read_text())
    model_result = metrics["models"][checkpoint.stem]
    if not model_result["episodes"][str(args.trajectory)]["task_success_any_step"]:
        raise ValueError("Selected training rollout is not a successful case")
    trace_path = train_root / checkpoint.stem / f"trajectory_{args.trajectory}" / "rollout.npz"
    with np.load(trace_path) as trace:
        actions = trace["actions"].copy()
        saved_state = trace["proprio"][args.chunk].copy()
    start = args.chunk * 4
    if start + 4 > len(actions):
        raise ValueError("Chunk outside recorded training rollout")
    eval_path, eval_obs = find_eval_observation(args.eval_root, args.eval_seed, args.chunk)
    _, eval_next = find_eval_observation(args.eval_root, args.eval_seed, args.chunk + 1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.overwrite)
    write_json(output / "metrics.json", {"status": "running", "arguments": vars(args)})
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log", mode="w")],
        force=True,
    )
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    simulator_args = SimpleNamespace(
        hdf5=args.hdf5,
        npz=cfg.data.npz,
        normalization=cfg.evaluation.normalization_path,
        trajectory_ids=[args.trajectory],
    )
    logger.info("START phase=restore completed=0 total=%d", start + 4)
    with Simulator(simulator_args, output, python, root) as sim:
        obs = sim.call("reset", args.trajectory)
        for i in range(0, start, 4):
            obs = sim.call("step", (actions[i : i + 4], None))["obs"]
            if i % 40 == 0:
                logger.info("STATUS phase=restore completed=%d total=%d", i + 4, start + 4)
        train_obs = {k: v.copy() for k, v in obs.items()}
        state_error = float(np.abs(train_obs["state"] - saved_state).max())
        train_next = sim.call("step", (actions[start : start + 4], None))["obs"]
    if state_error > 1e-3:
        raise ValueError(
            f"Training policy-action replay differs from saved proprio by {state_error}; "
            "use --chunk=0 for an initial-state preview or investigate replay drift"
        )
    dataset = RobomimicLightWAMWindowDataset(
        cfg.data.npz,
        context_path=cfg.lightwam.context,
        prompt=cfg.lightwam.prompt,
        image_size=cfg.lightwam.image_size,
        action_horizon=4,
        video_frames=cfg.lightwam.video_frames,
        normalized=True,
    )
    model = load_official_lightwam(
        vendor_root=Path(cfg.lightwam.vendor_root),
        model_config=Path(cfg.lightwam.model_config),
        local_model_root=Path(cfg.lightwam.local_model_root),
        device="cuda:0",
        model_dtype=MODEL_DTYPES[cfg.lightwam.model_dtype],
        checkpoint=checkpoint,
        allow_download=False,
    ).eval()
    adapter = LightWAMBaseAdapter(
        model,
        image_size=cfg.lightwam.image_size,
        context=dataset.context,
        context_mask=dataset.context_mask,
    )
    result = {
        "status": "running",
        "arguments": vars(args),
        "train_replay_proprio_max_abs_error": state_error,
        "limitations": [
            "No future RGB/state/action supplied to prediction.",
            "Five frames cover current time through +0.2 seconds at 20Hz.",
            "Training input is regenerated by replaying recorded policy actions, not a saved original RGB.",
            "Endpoint reference is not supplied to the model; only t+4 is compared.",
            "Denoising uses the training latent grid; upsampling precedes VAE decoding only.",
            "Video quality is not action quality; first frame is fixed conditioning.",
        ],
        "eval_input_source": str(eval_path),
        "cases": {},
    }
    write_json(output / "metrics.json", result)
    for number, (name, current, following) in enumerate(
        [
            (f"eval_{args.eval_seed}", eval_obs, eval_next),
            (f"train_{args.trajectory}", train_obs, train_next),
        ],
        1,
    ):
        destination = output / name
        destination.mkdir(exist_ok=args.overwrite)
        rgb, state = adapter._prepare_observation(current)
        logger.info(
            "START phase=generate case=%s completed=%d total=2 denoise_steps=%d",
            name,
            number - 1,
            args.inference_steps,
        )
        with torch.inference_mode():
            prediction = predict_future_video(
                model, image=rgb[0], proprio=state[0],
                context=dataset.context, context_mask=dataset.context_mask,
                steps=args.inference_steps, seed=0,
            )
        save_frames(prediction["video"], destination)
        original = Image.fromarray(
            np.clip(current["rgb"].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        )
        reference = Image.fromarray(
            np.clip(following["rgb"].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        )
        original.save(destination / "input.png")
        reference.save(destination / "reference_t_plus_4.png")
        generated = prediction["video"][-1].convert("RGB")
        comparison = Image.new("RGB", (generated.width * 3, generated.height))
        for i, image in enumerate([original, generated, reference]):
            comparison.paste(image.resize(generated.size), (i * generated.width, 0))
        comparison.save(destination / "input_prediction_reference.png")
        result["cases"][name] = {
            "status": "complete",
            "sampling": prediction["sampling"],
            "frames": 5,
            "comparison_columns": [
                "input_t",
                "prediction_t_plus_4",
                "simulator_reference_t_plus_4",
            ],
        }
        write_json(output / "metrics.json", result)
        logger.info("DONE phase=generate completed=%d total=2 case=%s", number, name)
    result["status"] = "complete"
    write_json(output / "metrics.json", result)
    logger.info("DONE phase=all completed=2 total=2")


if __name__ == "__main__":
    main()
