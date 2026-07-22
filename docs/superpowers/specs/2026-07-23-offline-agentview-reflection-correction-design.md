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
- Preserve the two original recorded-image panes byte-for-byte before Rerun compression.
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

## Evaluation Data Flow

For each visualized offline frame:

1. Convert the dataset tensor to HWC uint8 once.
2. Log that unchanged array at the original camera entity.
3. Validate the agentview array against the calibrated HWC metadata.
4. Create `np.ascontiguousarray(image_hwc[:, ::-1])` only inside the Rerun logging path.
5. Log the corrected array at the calibrated entity.
6. Log the existing world geometry without changing coordinates.

Policy inference occurs before this logging branch and receives the original dataset tensor.
Offline metrics and predictions are therefore unchanged. With
`--num-visualized-episodes=0`, no calibration environment or corrected image is created.

## Error Handling

- A calibrated run fails if agentview metadata or decoded HWC shape does not match.
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

- the original agentview array logged to Rerun is byte-for-byte unchanged;
- the corrected entity equals only `original[:, ::-1]`, with no vertical flip, transpose, or
  channel change;
- the corrected result is contiguous and does not share writable storage with the original;
- original and corrected images use different entity paths;
- only the corrected entity receives `Transform3D` and `Pinhole`;
- the calibrated overlay origin, tab name, tab order, and active tab are exact;
- the free-3D view contains the corrected calibrated camera entity;
- a synthetic asymmetric marker fixture fails under the old orientation and aligns after
  the single width reversal;
- metrics-only evaluation still skips calibration work;
- a real RRD passes `rerun rrd verify` and contains both image entities plus calibration only
  on the corrected entity.

After tests pass, regenerate the existing five-episode `offline_5` result with `--overwrite`.
On asymmetric episode 823 frames, the corrected overlay EEF must lie on the visible gripper.
The original left pane must continue to show the exact image consumed by ACT.
