# ALOHA Dataset Visualizer Design

## Goal

Build a repository-owned Rerun visualizer for LeRobot ALOHA episodes. It must keep camera
frames, named robot signals, a nominal articulated 3D skeleton, and end-effector paths on
the same frame timeline so an episode can be played, paused, scrubbed, and advanced one
frame at a time.

## Scope

The first version supports the standard 14-value ALOHA layout used by
`lerobot/aloha_mobile_cabinet`:

- six revolute joints and one gripper value for the left arm;
- six revolute joints and one gripper value for the right arm;
- `observation.state`, `action`, and `observation.effort` signals;
- any camera streams declared by the LeRobot dataset metadata;
- one episode per viewer session.

The viewer derives per-joint velocity and `action - state`. It displays a nominal
kinematic skeleton, joint points, gripper opening, current Trossen end-effector frames,
and the complete left/right end-effector paths.

This version does not reconstruct the cabinet, pot, cameras, collisions, dynamics, or
contact forces. The nominal 3D scene is not registered to the RGB images.

## Approach

Use Rerun for all synchronized playback and implement forward kinematics with NumPy.
This keeps the existing frame-by-frame viewer interaction and avoids adding a MuJoCo or
URDF runtime dependency merely to draw a skeleton.

The nominal link transforms and base poses come from the Apache-2.0/MIT licensed
`huggingface/gym-aloha` ViperX 300S MJCF model. The end-effector geometry is consistent
with the ViperX 300S product-of-exponentials parameters published by Trossen Robotics.

Sources:

- https://github.com/huggingface/gym-aloha
- https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html

## Components

### `robot_practice.aloha_kinematics`

This module owns only the nominal robot model and forward kinematics. It exposes:

- the canonical 14 dataset joint names;
- immutable nominal link offsets and left/right base transforms;
- one-arm and bimanual forward-kinematics functions;
- a normalized-gripper-to-finger-width conversion;
- structured results containing joint points, end-effector translation, end-effector
  rotation, and finger points.

For each arm, the six revolute coordinates are applied in this order:

1. waist about Z;
2. shoulder about Y;
3. elbow about Y;
4. forearm roll about X;
5. wrist angle about Y;
6. wrist rotate about X.

The implementation follows the MJCF convention that each child-body translation occurs
before the joint rotation owned by that body. The exact recurrence is:

```text
T = T_base
  · Trans(0,0,0.079)       · Rz(q_waist)
  · Trans(0,0,0.04805)     · Ry(q_shoulder)
  · Trans(0.05955,0,0.3)   · Ry(q_elbow)
  · Trans(0.2,0,0)         · Rx(q_forearm_roll)
  · Trans(0.1,0,0)         · Ry(q_wrist_angle)
  · Trans(0.069744,0,0)    · Rx(q_wrist_rotate)
  · Trans(0.1072,0,0)
```

The nominal link offsets, in metres, are:

```text
shoulder_link       (0,       0, 0.079)
upper_arm_link      (0,       0, 0.04805)
upper_forearm_link  (0.05955, 0, 0.3)
lower_forearm_link  (0.2,     0, 0)
wrist_link          (0.1,     0, 0)
gripper_link        (0.069744,0, 0)
end_effector        (0.1072,  0, 0)
```

The left nominal base is at `(-0.469, 0.5, 0)`. The right nominal base is at
`(0.469, 0.5, 0)` with a pi-radian rotation about Z. Raw gripper values remain unchanged
in signal plots. Only the 3D finger geometry clips the normalized value to `[0, 1]` and
maps it to the real puppet calibration interval `[0.01844, 0.05800]` metres. This mapped
value is the half-opening `q`; the two finger points are placed at local `+q` and `-q`, so
the full displayed separation is `2q`. The separate MJCF slide-joint limits
`[0.021, 0.057]` are not used for this real-robot dataset.

The tool offset is `0.1072` metres, not the `0.15` metre camera-focus/calibration site in
gym-aloha. The preceding transforms reach local X `0.429294` metres and this offset gives
the Trossen home end-effector X position `0.536494` metres.

### `robot_practice.visualize_aloha`

This module owns CLI parsing, dataset validation, derived signals, batched frame loading,
Rerun logging, and the viewer blueprint. It must not duplicate the kinematic equations.

The CLI is:

```bash
python -m robot_practice.visualize_aloha \
  --repo-id lerobot/aloha_mobile_cabinet \
  --episode-index 0
```

Options include:

- `--root PATH` to select a local LeRobot dataset root;
- `--max-frames N` to cap an inspection or test run;
- `--no-images` to skip video decoding and inspect signals/3D quickly;
- `--save PATH.rrd` to save without spawning a viewer;
- `--batch-size N` and `--num-workers N` for decoding control.

If `--root` is omitted, `LeRobotDataset` uses its normal Hugging Face cache/download
behavior. The current cache can be forced offline externally with `HF_HUB_OFFLINE=1`.

## Data Flow

1. Construct `LeRobotDataset` for the requested episode and validate its metadata. An
   image-enabled run uses normal video behavior. A `--no-images` run must pass
   `download_videos=False` during construction; deciding to skip images only after
   construction is too late.
2. Read numeric rows directly from `dataset.hf_dataset` when `--no-images` is used, so
   the fast path never calls video decoding.
3. Otherwise, iterate the episode once through a deterministic DataLoader and decode
   cameras in batches. Do not perform a second video-decoding pass.
4. For every frame:
   - set both `frame_index` and elapsed `timestamp` Rerun timelines;
   - log cameras;
   - log named state, action, effort, velocity, and action-minus-state scalars;
   - compute and log the two current skeletons, joint points, gripper fingers, and
     end-effector coordinate axes;
   - retain only the two end-effector XYZ values for path construction.
5. After iteration, log the complete left/right paths exactly as
   `rr.log(path, rr.LineStrips3D(...), static=True)`, making the full paths intentionally
   visible at every time point.

Velocity uses the actual adjacent timestamps. The first frame has zero velocity. A
non-positive timestamp delta is treated as invalid data rather than silently producing an
infinite value.

## Rerun Entity Layout

```text
episode/task
cameras/<camera_name>
signals/position/<side>/<joint_name>
signals/action/<side>/<joint_name>
signals/effort/<side>/<joint_name>
signals/velocity/<side>/<joint_name>
signals/action_minus_state/<side>/<joint_name>
robot/<side>/skeleton
robot/<side>/joints
robot/<side>/gripper
robot/<side>/end_effector
robot/<side>/end_effector_path
```

The blueprint presents the three camera views and nominal 3D skeleton prominently, with
separate named time-series views for position/action, effort, velocity, and
action-minus-state. Left and right arms use consistent distinct colors. The primary
timeline is configured with `rrb.TimePanel(timeline="frame_index", fps=dataset.fps)`;
`timestamp` is an elapsed-duration timeline rather than a Unix timestamp. Every selected
frame calls exactly:

```python
rr.set_time("frame_index", sequence=frame_index)
rr.set_time("timestamp", duration=elapsed_seconds)
```

For `--save`, the `.rrd` sink is configured before sending the blueprint or logging any
entities.

## Validation And Errors

Fail with an actionable message when:

- the episode index is outside the dataset range;
- any required numeric feature is missing;
- state, action, or effort is not 14-dimensional;
- metadata joint names are present but conflict with the canonical ALOHA order;
- camera files for an image-enabled run are incomplete;
- timestamps are non-finite or fail to increase;
- state, action, or effort contains non-finite values;
- `--max-frames`, `--batch-size`, or `--num-workers` is outside its allowed range
  (`max-frames > 0`, `batch-size > 0`, `num-workers >= 0`).

Slightly out-of-range normalized gripper values are preserved in plots and clipped only
for 3D finger geometry.

The startup log explicitly labels the 3D model as nominal and unregistered to cameras.

## Tests

Unit tests cover:

- zero-pose left/right forward kinematics and expected final transforms;
- a nonzero-angle golden pose that detects translation/rotation order mistakes;
- independent expected left and right world transforms, including a nonzero pose, rather
  than assuming generic mirror symmetry;
- finite, orthonormal end-effector rotations for representative joint values;
- gripper clipping and width conversion;
- velocity and action-minus-state calculations;
- required-feature, dimension, joint-order, episode, and timestamp validation.

Mocked/log-capture tests assert exact entity paths, both time assignments for each frame,
one log per selected frame, static flags on complete paths, and no duplicate decoding. The
`--no-images` test also asserts that dataset `__getitem__` and video querying are never
called.

An offline integration test loads three frames from the cached episode, disables images,
saves a temporary `.rrd`, and verifies successful ingestion in addition to a non-empty
file. A manual acceptance run loads episode 0 with all cameras and confirms synchronized
frame stepping in the Rerun viewer.

## Acceptance Criteria

- Episode 0 opens from the existing cache without redownloading completed files.
- All three cameras stay synchronized with the `frame_index` timeline.
- All 14 state, action, effort, velocity, and action-minus-state signals use readable
  joint names.
- The two skeletons and end-effector poses update on every frame.
- Left/right end-effector trajectories are visible in the 3D view.
- Arrow-key frame stepping updates images, curves, and 3D geometry together.
- `--no-images --max-frames 3 --save <temporary.rrd>` completes offline in a test.
