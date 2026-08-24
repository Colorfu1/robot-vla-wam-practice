# Vendor patches

The runtime image checks out FastWAM, LIBERO and lerobot at the commits pinned in
`Dockerfile.w0-fastwam`. Only `fastwam-trainer-freeze-video-expert.patch` is intended to be
applied; the other two are kept for reference and are **not** applied. Verify what the image
actually has with:

```bash
docker run --rm --entrypoint bash <image> -lc \
  'for d in FastWAM LIBERO lerobot; do git -C /opt/robot-practice/w0-vendor/$d status --porcelain; done'
```

Empty output means the tree is clean. These files exist so the work is not lost, not because
anything needs them.

## `fastwam-eval-trial-start-and-render-diagnostics.patch`

Two independent additions to `experiments/libero/eval_libero_single.py`:

- **`EVALUATION.trial_start` support** — lets one task's trials be split across child
  processes (`trial_idx = trial_start + local_trial_idx`). Without it, `+EVALUATION.trial_start=N`
  is a Hydra addition the script never reads, so each shard would re-run the first trials and
  the rest would never execute. `_assert_vendor_supports_trial_start` refuses that layout
  before launching any child, so the hazard cannot reach a result.
  The recorded UP-REL run is unaffected: it used `trials_per_worker == trials_per_task == 50`,
  one child per task, `trial_start` always 0 — its summary shows one source per task, 50 unique
  trial indices and `missing_trial_count = 0`.
- **Per-trial environment recreation plus opt-in render diagnostics** (`_w0_diag_*`,
  `_step_with_render_diagnostics`) — instrumentation and a mitigation from the investigation
  into a native EGL/MuJoCo `SIGABRT`. The actual root cause turned out to be
  `MUJOCO_EGL_DEVICE_ID` being used as a physical GPU index; once the device map was
  calibrated, 2,000 episodes ran clean with one shared environment per task, so the mitigation
  is no longer needed. Keep it in case EGL aborts reappear.

## `fastwam-trainer-freeze-video-expert.patch`

**This one is meant to be applied**, unlike the two above. It enables the `FW-FZ` attribution
arm of the W0 causal study: video expert frozen, `lambda_video=0`, only the action expert and
`proprio_encoder` trainable (about 1 B of the 6.02 B).

Two edits to `src/fastwam/trainer.py`, both required:

1. `_apply_dit_only_train_mode` freezes `model.dit.mixtures["video"]` when
   `W0_FREEZE_VIDEO_EXPERT` is set to anything other than empty or `0`.
2. `trainable_params` is filtered by `requires_grad`. Without this the freeze is useless:
   `_apply_dit_only_train_mode` runs at line 84 and the optimizer is built at line 89 from
   `list(self.model.dit.parameters())`, which ignores `requires_grad`, so ZeRO-1 would still
   allocate Adam state for the 5 B frozen video expert.

Unset or `0` reproduces upstream behaviour exactly, so a single image serves `FW-V`, `FW-NV`
and `FW-FZ`. The output parent is derived from the same variable by
`launch_w0_causal.sh` (`causal_10ep_frozen/`), because `FW-FZ` is also a `no_video` run and a
separately-passed parent could be forgotten -- it would then overwrite `FW-NV` and the sibling
budget check would compare against the wrong group. The manifest records
`freeze_video_expert` for the same reason.

Keep `mot_checkpoint_mixed_attn=false`: gradient checkpointing on a frozen branch warns and
does nothing useful when neither inputs nor parameters require grad.

## `lerobot-local-wan-loader.patch`

Adds `_load_local_wan_text_encoder` and friends to
`src/lerobot/policies/fastwam/wan/components.py`, driven by `W0_FASTWAM_SOURCE_ROOT`. The
image loads the released policy without it — LR-REL's `1885/2000` was produced by the clean
checkout — so it is unused.

## If a patch ever becomes necessary

Add a `RUN git -C /opt/robot-practice/w0-vendor/<repo> apply <patch>` step to a Dockerfile
layer and verify the result in the same layer. Do not edit the vendor tree in a live
container: the edit disappears with the container, and three tests once passed only because a
local untracked checkout was dirty while the image was clean.
