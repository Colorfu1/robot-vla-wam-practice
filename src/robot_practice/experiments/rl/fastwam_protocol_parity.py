"""Numerical protocol checks on real weights; no simulator, optimization or rollout."""

import logging

import numpy as np
import torch

from robot_practice.experiments.rl.lightwam_train_scene_eval import write_json

LOG = logging.getLogger(__name__)


def differences(a, b):
    a, b = a.detach().float(), b.detach().float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Comparison requires finite tensors with identical shapes")
    d = a - b
    return {"shape": list(a.shape), "exact_equal": bool(torch.equal(a, b)),
            "mae": float(d.abs().mean()), "max_abs": float(d.abs().max()),
            "reference_rms": float(a.square().mean().sqrt()),
            "relative_l2": float(d.norm() / a.norm().clamp_min(1e-12))}


def action_differences(a, b):
    return {"full32": differences(a, b), "first4": differences(a[:, :4], b[:, :4]),
            "first4_gripper": differences(a[:, :4, 6], b[:, :4, 6])}


def selected_frames(actions):
    valid = len(actions) - 32
    if valid <= 0:
        raise ValueError("No valid 32-action window")
    close = actions[:, 6] > 0
    onsets = np.flatnonzero(close & np.r_[True, ~close[:-1]])
    frames = {0}
    for onset in onsets[:1]:
        frames.update(int(np.clip(onset + offset, 0, valid - 1)) for offset in (-1, 0, 4))
    return sorted(frames)


def prefill(model, first, context, context_mask, horizon):
    pre = model.video_expert.pre_dit(
        x=first, timestep=torch.zeros(first.shape[0], device=first.device, dtype=first.dtype),
        context=context, context_mask=context_mask, action=None,
        fuse_vae_embedding_in_latents=True)
    size = pre["tokens"].shape[1]
    mask = model._build_mot_attention_mask(size, horizon,
                                          int(pre["meta"]["tokens_per_frame"]), first.device)
    cache = model.mot.prefill_video_cache(
        video_tokens=pre["tokens"], video_freqs=pre["freqs"], video_t_mod=pre["t_mod"],
        video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
        video_attention_mask=mask[:size, :size])
    return {"video_kv_cache": cache, "attention_mask": mask, "video_seq_len": size}


def compare_case(model, sample, seed):
    device, dtype = model.device, model.torch_dtype
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inputs = model.build_inputs(sample, tiled=False)
    first = inputs["first_frame_latents"]
    if first is None:
        raise ValueError("Expected clean first-frame conditioning")
    single = model._encode_input_image_latents_tensor(
        sample["video"][:, :, 0].to(device=device, dtype=dtype), tiled=False)
    context = sample["context"].to(device=device, dtype=dtype)
    mask = sample["context_mask"].to(device=device, dtype=torch.bool)
    context, mask = model._append_proprio_to_context(
        context, mask, sample["proprio"][:, 0].to(device=device, dtype=dtype))
    row = {"vae_full_video_first_vs_single_image": differences(first, single),
           "training_vs_inference_context": differences(inputs["context"], context),
           "context_mask_equal": bool(torch.equal(inputs["context_mask"], mask)),
           "forward_comparison_context": "Fixed to training build_inputs context in all branches",
           "timesteps": []}
    # Use exactly the training context for all path comparisons, isolating cache/sequence effects.
    context, mask = inputs["context"], inputs["context_mask"]
    common = {"context": context, "context_mask": mask}
    cached_full = prefill(model, first, context, mask, 32)
    cached_single = prefill(model, single, context, mask, 32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def noise_like(value):
        return torch.randn(value.shape, generator=generator).to(device=device, dtype=dtype)
    action_noise = noise_like(inputs["action"])
    video_noise = noise_like(inputs["input_latents"])
    tv = torch.tensor([500.0], device=device, dtype=dtype)
    video = model.train_video_scheduler.add_noise(inputs["input_latents"], video_noise, tv)
    video[:, :, :1] = first
    for value in (1000.0, 500.0, 50.0):
        ta = torch.tensor([value], device=device, dtype=dtype)
        noisy = model.train_action_scheduler.add_noise(inputs["action"], action_noise, ta)
        joint_args = {"latents_video": video, "latents_action": noisy,
                      "timestep_video": tv, "timestep_action": ta, **common,
                      "fuse_vae_embedding_in_latents": True, "gt_action": inputs["action"]}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, joint_amp = model._predict_joint_noise(**joint_args)
        _, joint = model._predict_joint_noise(**joint_args)
        first_only = model._predict_action_noise(
            first_frame_latents=first, latents_action=noisy, timestep_action=ta,
            **common, fuse_vae_embedding_in_latents=True)
        cached = model._predict_action_noise_with_cache(
            latents_action=noisy, timestep_action=ta, **common, **cached_full)
        repeated = model._predict_action_noise_with_cache(
            latents_action=noisy, timestep_action=ta, **common, **cached_full)
        single_cached = model._predict_action_noise_with_cache(
            latents_action=noisy, timestep_action=ta, **common, **cached_single)
        row["timesteps"].append({
            "action_timestep": value, "video_timestep": 500,
            "joint_autocast_vs_joint": action_differences(joint_amp, joint),
            "joint_vs_first_frame_only": action_differences(joint, first_only),
            "first_frame_only_vs_cache": action_differences(first_only, cached),
            "cache_full_first_vs_single_first": action_differences(cached, single_cached),
            "training_path_vs_inference_path": action_differences(joint_amp, single_cached),
            "cache_repeat_control": action_differences(cached, repeated),
        })
        LOG.info("STATUS phase=parity_timestep completed=%d total=3 t=%g",
                 len(row["timesteps"]), value)
    return row


def run_parity(model, dataset, ids, output, seed):
    if model.video_expert.action_conditioned:
        raise ValueError("This audit assumes action_conditioned=False")
    starts = np.r_[0, np.cumsum(dataset.traj_lengths)]
    selected = [(tid, f) for tid in ids for f in selected_frames(
        dataset.actions[int(starts[tid]):int(starts[tid + 1])])]
    rows = []
    with torch.inference_mode():
        for number, (tid, frame) in enumerate(selected, 1):
            start = int(starts[tid]) + frame
            index = int(np.searchsorted(dataset.windows, start))
            if index >= len(dataset.windows) or int(dataset.windows[index]) != start:
                raise ValueError("Selected start not in training windows")
            sample = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                      for k, v in dataset[index].items()}
            row = {"trajectory": tid, "frame": frame,
                   **compare_case(model, sample, seed + tid)}
            rows.append(row)
            write_json(output / "parity_cases.json", {"cases": rows})
            LOG.info("STATUS phase=protocol_parity completed=%d total=%d trajectory=%d frame=%d",
                     number, len(selected), tid, frame)
    return {"status": "complete", "evaluated": len(ids), "requested": len(ids),
            "parity_cases": rows,
            "limitations": ["BF16 forward-path comparison, not closed-loop evaluation.",
                            "Outputs compared are flow velocities, not final sampled actions.",
                            "Batch size 1 in eval mode; no backward/optimizer equivalence test.",
                            "Differences require interpretation; completion is not parity proof."]}
