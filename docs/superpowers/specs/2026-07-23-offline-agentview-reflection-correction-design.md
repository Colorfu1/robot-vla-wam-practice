# Offline Agentview Reflection Correction Design

## Status

Approved in conversation on 2026-07-23. This document corrects the stored-image orientation
assumption in `2026-07-22-offline-agentview-calibration-design.md`. The task-specific MuJoCo
camera pose, intrinsics, display frame, robot geometry, and action coordinates remain valid.

## Problem and Reproduction

The LeRobot LIBERO image convention rotates raw simulator observations by 180 degrees to
match the HuggingFaceVLA/OpenVLA training data. In the local replay converter this is:

```python
image[::-1, ::-1]
```

On the installed robosuite stack, the raw OpenGL renderer requires the height reversal to
become upright. The additional width reversal leaves the stored, human-upright image
horizontally reflected relative to the physical MuJoCo pinhole camera.

Episode 823 provides a deterministic asymmetric reproduction:

- frame 0 is nearly center-symmetric and hides the error;
- by frame 46, the standard projection places the recorded EEF on the left while the RGB
  gripper is on the right;
- reflecting only the projected X coordinate aligns the EEF and the visible gripper;
- rendering the reconstructed frame directly confirms that `raw[::-1]` aligns with the
  standard projection, while the stored dataset image aligns after one additional width
  reversal.

This is not an action-coordinate error. Training, offline prediction, and closed-loop
rollout all use the same LeRobot image convention with unchanged robot-base-frame state and
action values. The defect is limited to projecting physically calibrated 3D geometry onto a
horizontally reflected visualization image.

## Goals

- Preserve the exact image tensor consumed by existing ACT checkpoints.
- Preserve the content and orientation of the two original recorded-image panes through
  the existing tensor-to-HWC-display conversion.
- Show a physically calibrated agentview whose image and 3D projection agree.
- Keep the physical MuJoCo camera pose, positive focal lengths, and free-3D frustum valid.
- Document the LIBERO image convention and its training implications in the A-Sanity guide.

## Non-Goals

- Rewriting the downloaded `lerobot/libero` videos or generated task caches.
- Changing ACT training, normalization statistics, policy preprocessing, state, or action.
- Making old checkpoints consume a new image orientation.
- Calibrating the dynamic eye-in-hand camera.
- Claiming the IK-reconstructed intermediate joints are recorded joint truth.

## Considered Approaches

### A. Separate reflection-corrected visualization entity — selected

Keep the recorded agentview at `demonstration/cameras/image`. For visualized frames only,
create a contiguous `image_hwc[:, ::-1]` copy and log it at a distinct calibrated entity.
Attach the physical `Transform3D` and `Pinhole` to the corrected entity.

This keeps training and recorded evidence immutable, uses a proper right-handed camera, and
makes the visualization-only correction explicit.

### B. Negative focal length or reflected camera axes — rejected

Project onto the original reflected image with a negative horizontal focal length or a
left-handed camera basis. This avoids a second image entity but encodes a reflection as a
non-physical camera, makes the free-3D frustum misleading, and depends on unsupported or
unclear Rerun behavior.

### C. Rewrite the source image convention — rejected

Horizontally correct dataset videos or the original image entity. Existing checkpoints were
trained on the reflected convention, and closed-loop LeRobot preprocessing reproduces it.
Changing only this workspace would create a train/eval mismatch and invalidate comparisons.

## Entity and Blueprint Contract

Use distinct paths:

```text
demonstration/cameras/image       # original checkpoint input / recorded evidence
calibration/agentview_corrected   # visualization-only horizontal correction
```

The original entity receives only the unchanged dynamic `rr.Image`. The corrected entity
receives:

- the horizontally corrected dynamic `rr.Image`;
- the static task-specific `rr.Transform3D` using parent-from-child;
- the static positive-focal-length `rr.Pinhole` using RUB camera coordinates.

The left-side camera panes continue to include only the two original image entity paths.
The upper-right tabs are:

1. `Agentview calibrated (horizontal reflection corrected)`, active by default, with origin
   `calibration/agentview_corrected` and contents containing that image plus the existing
   robot, trajectory, GT command, prediction command, and chunk-preview geometry;
2. `Free 3D - recorded geometry and commands`, including the corrected camera entity so its
   physical cyan frustum remains visible.

No corrected eye-in-hand entity is created because no camera calibration exists for it.

## Dataset Image-Convention Contract

The evaluator must not infer horizontal reflection from the word `LIBERO` alone. Add an
offline-evaluation image convention with these values:

```text
auto             # accept only provenance that the evaluator can verify
lerobot_180      # HuggingFaceVLA/OpenVLA/LeRobot fixed 180-degree convention
physical_opencv  # stored image already matches the physical pinhole projection
```

`auto` is the CLI default. It resolves `lerobot_180` only for a known repository identity
or a generated cache manifest written by this workspace and carrying an explicit convention
field. It resolves `physical_opencv` only from equally explicit provenance. An explicit
`--dataset-root` without convention metadata is unknown and a visualized run must fail with
an actionable message requiring `--agentview-image-convention`. The A-Sanity commands use
`--agentview-image-convention=lerobot_180` because their explicit combined-dataset root has
already been verified in the reproduction.

For `lerobot_180`, the corrected entity is `image_hwc[:, ::-1]`. For
`physical_opencv`, the corrected entity is a contiguous copy with no geometric transform.
The resolved convention is recorded in `run_config.json` and the Rerun semantics document.
Metrics-only runs do not need to resolve a visualization convention.

## Evaluation Data Flow

For each visualized offline frame:

1. Convert the dataset tensor to HWC uint8 once.
2. Validate the agentview shape and dtype against the calibrated HWC metadata.
3. Construct the convention-specific contiguous corrected copy without mutating the source.
4. Only after both arrays are valid, log the original array at the recorded camera entity
   and the corrected copy at the calibrated entity.
5. Log the existing world geometry without changing coordinates.

Policy inference occurs before this logging branch and receives the original dataset tensor.
Offline metrics and predictions are therefore unchanged. With
`--num-visualized-episodes=0`, no calibration environment or corrected image is created.

## Error Handling

- A calibrated run fails before recording frames if image convention is unknown.
- A calibrated frame validates image metadata, dtype, and corrected copy before logging
  either original or corrected image.
- The correction helper rejects non-HWC images and returns a contiguous array with identical
  dtype and shape.
- The evaluator does not silently attach physical calibration to the reflected original
  image.
- Existing IK stale/omitted behavior remains independent of camera calibration.

## Documentation Contract

The A-Sanity guide must state:

- LeRobot/OpenVLA rotates LIBERO images 180 degrees as a fixed dataset convention;
- robot-base state and action values remain unchanged;
- training and closed-loop preprocessing match, so existing checkpoints remain valid;
- changing the training-image convention requires rebuilding data, matching rollout
  preprocessing, recomputing statistics, and retraining;
- the left panes show the exact model input, while the calibrated tab is a visualization-only
  horizontal reflection correction;
- geometry-aware work must treat this convention explicitly.

The README and evaluator script comments should use the same concise distinction.

## Verification

Automated tests must prove:

- the original agentview content and orientation logged to Rerun are unchanged after the
  existing CHW/float-to-HWC/uint8 display conversion;
- the corrected entity equals only `original[:, ::-1]`, with no vertical flip, transpose, or
  channel change;
- the corrected result is contiguous and does not share writable storage with the original;
- original and corrected images use different entity paths;
- only the corrected entity receives `Transform3D` and `Pinhole`;
- the calibrated overlay origin, tab name, tab order, and active tab are exact;
- the free-3D view contains the corrected calibrated camera entity;
- `auto` accepts verified convention metadata and rejects an unknown explicit dataset root;
- explicit `lerobot_180` and `physical_opencv` select exactly their documented transforms;
- a synthetic asymmetric discrete-pixel marker fixture uses
  `x_original = width - 1 - x_physical`, fails under the old orientation, aligns after the
  single width reversal, and covers both X boundaries to lock down off-by-one behavior;
- the predecessor's four floating-point pinhole fixtures are labeled as corrected/physical
  projection coordinates rather than stored-image coordinates;
- a deterministic evaluator harness captures the policy input tensor with visualization
  disabled and enabled and proves bitwise equality of the image tensors plus equality of
  predictions and offline metrics;
- metrics-only evaluation still skips calibration work;
- a real RRD passes `rerun rrd verify` and contains both image entities plus calibration only
  on the corrected entity.

After tests pass, regenerate the existing five-episode `offline_5` result with `--overwrite`.
On asymmetric episode 823 frames, the corrected overlay EEF must lie on the visible gripper.
The original left pane must continue to show the exact image consumed by ACT.
