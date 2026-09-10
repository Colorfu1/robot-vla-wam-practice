# Vendor patches

The runtime image checks out FastWAM, LIBERO and lerobot at the commits pinned in
`Dockerfile.w0-fastwam`. `fastwam-trainer-freeze-video-expert.patch` and
`fastwam-memory-efficient-loading.patch` are intended to be applied; the other two are kept
for reference and are **not** applied. Verify what the image actually has with:

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

## `fastwam-memory-efficient-loading.patch`

This patch constructs the large Wan2.2 video DiT, VAE, and ActionDiT directly on the requested
CUDA device in the requested dtype, then streams safetensors into model parameters one tensor at
a time. It avoids the roughly 18.6 GiB fp32 CPU construction of the 5B Video DiT, the old merged
CPU state dict, and a full GPU shard staging allocation. The optional text encoder remains on the
legacy CPU staging path and is disabled in W0 FastWAM.

The FastWAM and ActionDiT `.pt` checkpoint readers use CPU `mmap` with `weights_only=True`, so
checkpoint storage is not eagerly paged into a second large anonymous CPU allocation. The final
parameters are still copied into the existing target-device model, and the model/checkpoint
format is unchanged.

## `lerobot-local-wan-loader.patch`

Adds `_load_local_wan_text_encoder` and friends to
`src/lerobot/policies/fastwam/wan/components.py`, driven by `W0_FASTWAM_SOURCE_ROOT`. The
image loads the released policy without it — LR-REL's `1885/2000` was produced by the clean
checkout — so it is unused.

## Local training and evaluation patches

`lightwam-trainer-staged-stop.patch` records the local trainer changes against
Light-WAM commit `b2785f66e13fd9987e94ae1ecc1c441d5059c9ae`. It separates the
absolute stopping step from the learning-rate schedule horizon and saves weights
and full training state at the stopping step.

`dppo-eval-target.patch` records the evaluation config target correction against
DPPO commit `cc7234ad7ff39a8f32de3af903606723a16f0648`. The DPPO Dockerfile already
performs this correction; the patch also makes the local checkout changes portable.

From the practice repository root, apply only to a matching clean vendor revision:

```bash
git -C third_party/lightwam/Light-WAM apply --check ../../../docker/training/vendor-patches/lightwam-trainer-staged-stop.patch
git -C third_party/lightwam/Light-WAM apply ../../../docker/training/vendor-patches/lightwam-trainer-staged-stop.patch
git -C third_party/dppo/dppo apply --check ../../../docker/training/vendor-patches/dppo-eval-target.patch
git -C third_party/dppo/dppo apply ../../../docker/training/vendor-patches/dppo-eval-target.patch
```

For an already patched checkout, `git apply --reverse --check` verifies that the
patch is present. These commands do not download vendor sources or model assets.

## If a patch ever becomes necessary

Add a `RUN git -C /opt/robot-practice/w0-vendor/<repo> apply <patch>` step to a Dockerfile
layer and verify the result in the same layer. Do not edit the vendor tree in a live
container: the edit disappears with the container, and three tests once passed only because a
local untracked checkout was dirty while the image was clean.
