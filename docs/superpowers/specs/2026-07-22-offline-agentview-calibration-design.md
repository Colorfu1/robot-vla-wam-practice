# Offline Agentview Calibration Design

## Status

Approved in conversation on 2026-07-22. This design corrects the misleading visual
comparison between LIBERO's recorded `agentview` image and Rerun's independently chosen
free-orbit 3D viewpoint. It extends the existing offline Panda skeleton and command-segment
designs without changing any robot, trajectory, or action coordinates.

## Problem

The recorded RGB view and current free 3D view observe the same geometry from different
sides. For LIBERO Object task 8:

```text
robot base, native:       [-0.6, 0.0, 0.0]
agentview, display:       [0.0, 1.496577, 0.65]
agentview forward:        [0.0, -0.848765, -0.528770]
first recorded EEF:       [0.004565, 0.461708, 0.269752] in display coordinates
```

The fixed `agentview` camera therefore stands in front of the workspace and looks back
toward the robot base. Rerun's current Spatial3D view has no `Transform3D` or `Pinhole`
camera model and chooses an unrelated orbit viewpoint. Screen-space front/back in that
view cannot be compared directly with the RGB image.

The simulator geometry is not reversed. Across all 859 selected A-Sanity frames, recorded
EEF native X lies `0.448576 m` to `0.670866 m` in front of the base. On the first frame,
the IK skeleton endpoint differs from the recorded EEF by only `0.072 mm`.

## Goal

Add an explicitly calibrated view in which the recorded RGB image and the existing 3D
robot/command geometry share the real MuJoCo `agentview` projection. Keep both original
camera images unchanged and retain the current free 3D view for orbit inspection.

## Non-Goals

This change does not:

- rotate, reflect, or translate the existing display frame;
- change the base, recorded EEF, IK skeleton, GT commands, or predicted commands;
- claim that the IK-reconstructed elbow is recorded joint truth;
- calibrate `robot0_eye_in_hand`, whose pose is dynamic and cannot be recovered exactly
  without recorded arm joint positions;
- reconstruct objects, fixtures, depth, occlusion, or contacts;
- force Rerun's free-orbit camera to a fixed initial pose, which Rerun 0.26 does not expose
  through `EyeControls3D`.

## Camera Calibration Contract

Add an immutable LIBERO camera calibration value owned by the kinematics layer. It contains:

- camera name (`agentview`);
- image width and height supplied from the evaluated dataset metadata;
- camera position in the normalized display frame;
- a right-handed camera-to-display rotation;
- vertical field of view in radians;
- the derived focal length and principal point.

The calibration is extracted from the actual task-specific MuJoCo model rather than from
hard-coded LIBERO constants:

```text
p_camera_native = sim.data.cam_xpos[agentview]
R_native_from_camera = reshape(sim.data.cam_xmat[agentview], 3, 3)
fov_y_degrees = sim.model.cam_fovy[agentview]
fov_y_radians = deg2rad(fov_y_degrees)

p_camera_display = DisplayFrame.transform_points(p_camera_native)
R_display_from_camera =
    DisplayFrame.rotation_display_from_native @ R_native_from_camera

focal_y = 0.5 * image_height / tan(0.5 * fov_y_radians)
focal_x = focal_y
principal = [0.5 * image_width, 0.5 * image_height]
```

MuJoCo camera axes are treated as `RUB`: X right, Y up, Z back, with projection along
camera `-Z`. Rerun receives `camera_xyz=rr.ViewCoordinates.RUB`.

`sim.model.cam_fovy` is explicitly treated as degrees. Only `fov_y_radians` is validated
against `(0, pi)`, used in trigonometry, stored in the calibration value, or passed to an API
that expects radians.

The final stored MP4 must not receive another image rotation. The local replay converter's
`image[::-1, ::-1]` corrects the renderer buffer before storage. A real episode 823 frame
confirms that the final image follows the standard camera projection. Four frozen native
points and their independently calculated pixels provide an asymmetric regression fixture:

| Native point | Expected pixel in the stored 256-by-256 frame |
|---|---|
| `[-0.6, 0.2, 0.1]` | `[167.591233, 63.759209]` |
| `[-0.1, -0.25, 0.3]` | `[53.063350, 59.090222]` |
| `[-0.1, -0.25, 0.0]` | `[63.056351, 134.426155]` |
| `[-0.1, 0.25, 0.0]` | `[192.944364, 134.426260]` |

These points span both image halves and upper/lower regions, so horizontal mirror, vertical
mirror, and 180-degree rotation errors cannot pass by preserving a centerline point.

Dataset resolution is parsed from the metadata image shape, which is HWC for LeRobot video
features. The parser requires shape `(height, width, channels)` with channels in `{1, 3, 4}`
and emits Rerun resolution `[width, height]`. Policy tensors are CHW and are not used to infer
calibration. Every decoded or tensor-converted frame must become HWC
`(height, width, channels)` before logging and match the calibrated dimensions exactly.

Calibration validation rejects:

- a missing or duplicated `agentview` camera;
- non-finite position, rotation, FOV, or resolution;
- a non-orthonormal or left-handed camera rotation;
- a FOV outside `(0, pi)`;
- non-positive image dimensions or focal length;
- an `agentview` camera not attached to MuJoCo's world body or not using fixed camera mode;
- a camera world pose that changes after a model `forward()` validation step.

The current calibrated overlay supports only a fixed world camera. A moving or body-attached
camera is a configuration error rather than a reason to log a stale static transform.

## Rerun Entity Contract

Keep the recorded agentview image at:

```text
demonstration/cameras/image
```

Log the following static components at that same entity:

- `rr.Transform3D` containing the camera-to-display pose, explicitly using the
  parent-from-child relation;
- `rr.Pinhole` containing the metadata-derived projection, `RUB` coordinates, frustum color
  `(0, 220, 255, 210)`, image-plane distance `0.25 m`, and line width `0.003`.

The dynamic `rr.Image` remains on the same path. The original agentview Spatial2D view
continues to include only that image entity, so it remains an unmodified ground-truth RGB
reference.

The free 3D contents explicitly include the calibrated agentview entity. This renders the
camera origin, direction, frustum, and image plane in their real display-frame pose. The
stable entity path remains inspectable without introducing an always-visible 3D label.

## Calibrated Overlay

Add a `Spatial2DView` with origin `demonstration/cameras/image`. Its contents include:

- the recorded agentview image;
- `demonstration/robot/**`;
- `demonstration/trajectory/**`;
- `comparison/commands/**`;
- `prediction/not_executed/chunk_preview`;
- `prediction/not_executed/chunk_endpoint`.

Rerun uses the entity's `Transform3D` and `Pinhole` to project the existing world-space
geometry into image pixels. No second set of skeleton, trajectory, or command coordinates
is created.

The overlay retains existing colors and semantics:

| Geometry | Meaning |
|---|---|
| deep blue | recorded EEF and IK-reconstructed Panda skeleton |
| light blue | expert first-action OSC translation command |
| deep orange | teacher-forced predicted first-action command, not executed |
| translucent orange | current predicted future command preview, not executed |

The overlay does not include display-axis labels. The source image remains a recorded frame;
only the vector geometry is projected over it. The current EEF projection should align with
the visible gripper. The reconstructed elbow can differ from the RGB robot because the
dataset does not contain the seven recorded arm joint positions.

## Blueprint Layout

The upper left retains the existing two original camera views side by side:

```text
Demonstration (recorded) - image
Demonstration (recorded) - image2
```

The upper right becomes a tab container:

1. `Agentview calibrated overlay` (active by default)
2. `Free 3D - camera frustum and commands`

This allows the untouched agentview image and calibrated overlay to remain visible next to
each other without compressing the interface into four narrow horizontal panes. The action,
error, latency, semantics, and metrics tabs below remain unchanged.

## Evaluation Data Flow

For offline evaluation with at least one visualized episode:

1. Construct the existing task-specific `PandaSkeletonReconstructor`.
2. Read `agentview` calibration from its already initialized MuJoCo model.
3. Parse the stored agentview HWC resolution from LeRobot dataset metadata.
4. Validate and convert the camera pose through the existing `DisplayFrame`.
5. Initialize Rerun and log static world axes, action-series styles, camera transform, and
   pinhole model before sending the blueprint.
6. Log images and the existing per-frame geometry without altering their coordinates.

When `--num-visualized-episodes=0`, the evaluator continues to skip the MuJoCo environment.
It does not require camera calibration and does not expose the calibrated overlay tab.

## Error Handling

A visualized run fails before per-frame logging if the task-specific model cannot supply a
valid fixed agentview calibration. It must not silently fall back to an uncalibrated overlay.
The output-directory incomplete marker and existing cleanup behavior remain authoritative.

An IK failure retains the existing stale/omitted skeleton behavior. It does not invalidate
camera calibration, recorded EEF projection, command projection, or offline metrics.

## Verification

Tests cover:

- exact extraction and validation of camera position, rotation, and FOV from a fake MuJoCo
  model, including the degree-to-radian conversion;
- exact camera-to-display pose composition;
- independent pinhole projection of asymmetric, non-collinear points using RUB coordinates;
- all four frozen episode 823 projection fixtures above, detecting horizontal mirror,
  vertical mirror, axis swap, and 180-degree rotation;
- exact HWC metadata parsing, `[width, height]` Rerun ordering, and per-frame resolution
  validation;
- byte-for-byte equality between the decoded dataset frame and the HWC array passed to
  `rr.Image` before compression, plus a bounded decoded-RRD tolerance for lossy compression;
- rejection of a moving, body-attached, or non-fixed agentview camera;
- exact static `Transform3D` and `Pinhole` entity paths and components;
- exact frustum RGBA, image-plane distance, and line width;
- original RGB views containing no projected geometry;
- the calibrated overlay containing the image plus the intended 3D geometry paths;
- the upper-right tab order and calibrated overlay being active by default;
- the free 3D view including the agentview camera/frustum entity;
- metrics-only evaluation avoiding camera-model construction;
- a bounded real three-frame RRD that passes `rerun rrd verify` and contains camera,
  projection, image, skeleton, EEF, and command components.

After automated verification, regenerate the existing `offline_5` output with `--overwrite`
and manually compare the untouched agentview image with the calibrated overlay. The base,
EEF, and command direction must agree in image space; any elbow discrepancy must remain
explicitly attributable to IK reconstruction rather than camera calibration.

Existing RRD recordings remain immutable and require evaluator rerun before this view
appears.
