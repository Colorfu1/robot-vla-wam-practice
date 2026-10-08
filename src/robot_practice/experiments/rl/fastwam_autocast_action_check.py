"""Paired full 20-step action sampling with autocast off/on, fixed inputs and seed."""

import logging

import numpy as np
import torch

from robot_practice.experiments.rl.fastwam_dual_train_fit import (
    action_errors,
    predict,
    select_windows,
)
from robot_practice.experiments.rl.fastwam_protocol_parity import action_differences
from robot_practice.experiments.rl.lightwam_train_scene_eval import write_json

LOG = logging.getLogger(__name__)


def sample_pair(adapter, observation, seed):
    if adapter.num_inference_steps != 20:
        raise ValueError("Both branches must use exactly 20 sampling steps")
    predictions = []
    # ABAB control: both complete samplers are repeated, with the same explicit seed.
    # No teacher-forced actions, video latents or future states enter either branch.
    with torch.inference_mode():
        for enabled in (False, True, False, True):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                predictions.append(predict(adapter, {k: v.copy() for k, v in observation.items()},
                                           seed))
    off, on, repeat_off, repeat_on = predictions
    controls = {"off_repeat_max_abs": float(np.max(np.abs(off - repeat_off))),
                "on_repeat_max_abs": float(np.max(np.abs(on - repeat_on)))}
    if max(controls.values()) > 1e-5:
        raise ValueError(f"Same-input sampling not repeatable: {controls}")
    return off, on, controls


def summary(off, on, target):
    off, on, target = map(np.asarray, (off, on, target))
    baseline, autocast = action_errors(off, target), action_errors(on, target)
    a, b = np.clip(off, -1, 1), np.clip(on, -1, 1)
    off_error = np.mean((a[:, :4, :6] - target[:, :4, :6]) ** 2, axis=(1, 2))
    on_error = np.mean((b[:, :4, :6] - target[:, :4, :6]) ** 2, axis=(1, 2))
    return {
        "windows": len(off), "normal_inference": baseline, "bf16_autocast": autocast,
        "raw_action_difference": action_differences(torch.as_tensor(off), torch.as_tensor(on)),
        "clipped_action_difference": action_differences(torch.as_tensor(a), torch.as_tensor(b)),
        "first4_gripper_sign_disagreements": int(((a[:, :4, 6] > 0) !=
                                                  (b[:, :4, 6] > 0)).sum()),
        "first4_arm_mse_improved_windows": int((on_error < off_error).sum()),
        "first4_arm_mse_worsened_windows": int((on_error > off_error).sum()),
        "first4_arm_mse_equal_windows": int((on_error == off_error).sum()),
    }


def run_autocast_check(adapter, dataset, ids, output, seed, windows_per_trajectory):
    starts = np.r_[0, np.cumsum(dataset.traj_lengths)]
    selections = [(tid, start) for tid in ids for start in select_windows(
        dataset.traj_lengths, dataset.actions, tid, windows_per_trajectory)]
    baseline, amp, targets, cases = [], [], [], []
    for number, (tid, start) in enumerate(selections, 1):
        observation = {"rgb": dataset.arrays["images"][start].copy(),
                       "state": dataset.states[start].copy()}
        off, on, controls = sample_pair(adapter, observation, seed + tid)
        target = dataset.actions[start:start + 32].copy()
        row = {"trajectory": tid, "frame": int(start - starts[tid]), "noise_seed": seed + tid,
               "repeat_controls": controls, **summary(off[None], on[None], target[None])}
        cases.append(row)
        baseline.append(off)
        amp.append(on)
        targets.append(target)
        write_json(output / "autocast_progress.json", {"completed": number,
                                                       "total": len(selections), "latest": row})
        LOG.info("STATUS phase=autocast_action completed=%d total=%d trajectory=%d frame=%d",
                 number, len(selections), tid, row["frame"])
    np.savez_compressed(output / "autocast_actions.npz", normal=baseline, autocast=amp,
                        target=targets, trajectory_frame=[(r["trajectory"], r["frame"])
                                                        for r in cases])
    return {"status": "complete", "evaluated": len(ids), "requested": len(ids),
            "inference_steps": 20, "autocast_summary": summary(baseline, amp, targets),
            "autocast_cases": cases,
            "limitations": ["Saved training RGB only; no renderer difference in this test.",
                             "Model weights remain BF16 in both branches; toggle autocast only.",
                             "Final sampled actions, not flow velocities; no closed-loop test.",
                             "One fixed noise seed per trajectory, not a multi-seed estimate."]}
