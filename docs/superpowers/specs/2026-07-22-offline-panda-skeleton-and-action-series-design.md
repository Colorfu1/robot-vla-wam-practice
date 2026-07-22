# Offline Panda Skeleton and Action-Series Interaction Design

## Status

Approved in conversation on 2026-07-22. This design extends
`2026-07-22-act-evaluation-overwrite-and-offline-3d-design.md` and supersedes its
point-and-path-only offline 3D presentation. The output overwrite behavior from that design
is unchanged.

## Goal

Make the offline ACT recording answer two questions visually:

1. Where is the robot, and how is its arm posed while the demonstration frame is shown?
2. For any action dimension, how do ground truth and prediction differ over time?

The recording will render an IK-reconstructed Panda skeleton in a normalized display frame
whose positive Y direction points from the robot base toward the workspace. The action chart
will expose fourteen independently clickable Rerun legend entries, ordered as adjacent
ground-truth/prediction pairs for seven action dimensions.

## Source-Data Limitation

The current LeRobot dataset contains two RGB streams, an 8D state, and a 7D action. The state
is:

```text
eef_position[3] + eef_axis_angle[3] + gripper_qpos[2]
```

It does not contain the Panda's seven joint positions. The upstream LIBERO `robot_states`
field is also a gripper/EEF state rather than a joint vector. Therefore the evaluator cannot
claim to display recorded joint poses.

The arm is reconstructed with the installed LIBERO/robosuite Panda kinematic model. Its
base, link lengths, joint limits, and end-effector target are model-grounded, while its
redundant elbow configuration is an IK result. Every visible label and explanatory note must
call it `IK reconstructed`.

No raw HDF5 download or demonstration replay is introduced by this feature.

## Architecture

Add a focused module:

```text
src/robot_practice/evaluation/libero_kinematics.py
```

It contains two independent units.

### DisplayFrame

`DisplayFrame` owns the mapping from a robot/environment native world frame into the Rerun
display frame. It exposes point and vector transforms and contains user-facing axis labels.
This keeps robot-specific coordinate conventions out of the Rerun logger.

For the current LIBERO Panda, the native base position is the display origin and native +X
is mapped to display +Y:

```text
R_display_from_native = [[ 0, -1, 0],
                         [ 1,  0, 0],
                         [ 0,  0, 1]]

p_display = R_display_from_native @ (p_native - p_base_native)
v_display = R_display_from_native @ v_native
```

This is a right-handed frame:

- display `+X` is native `-Y`;
- display `+Y` is native `+X` and is labelled `Y Forward`;
- display `+Z` is native `+Z` and is labelled `Z Up`.

The same transform is mandatory for the skeleton, recorded EEF path, current EEF point,
prediction chunk preview, prediction endpoint, base geometry, and every orientation marker.

### PandaSkeletonReconstructor

`PandaSkeletonReconstructor` owns one LIBERO Panda simulation model for the selected task. It
is created only when at least one episode is being visualized and is closed on both success
and exceptions. Metrics-only evaluation with `--num-visualized-episodes=0` has no MuJoCo IK
overhead.

The component reads from the model:

- the seven arm qpos indexes and joint limits;
- the Panda neutral/initial joint configuration;
- the gripper end-effector site;
- base, `link0` through `link7`, hand, and gripper geometry identifiers.

For each visualized dataset frame it receives the raw, pre-policy-preprocessor 8D state. It
converts the axis-angle target into a native end-effector orientation and solves a bounded,
damped least-squares 6D IK problem using the site position and rotation Jacobians. Joint
values are clamped to the model limits after every update.

The solver contract is fixed rather than left to implementation defaults:

```text
position_error = target_position - current_position
orientation_error = robosuite axis-angle orientation_error(target_R, current_R)
error = concatenate(position_error, orientation_error)
J = concatenate(site_position_J, site_rotation_J, axis=0)

dq_raw = J.T @ inverse(J @ J.T + damping^2 * I6) @ error
dq = gain * dq_raw
q_next = clip(q + limit_norm(dq, max_step), joint_low + margin, joint_high - margin)
```

| Parameter | Value |
|---|---:|
| maximum iterations | 80 |
| damping | `1e-3` |
| gain | `0.8` |
| maximum per-iteration `L2(dq)` | `0.15 rad` |
| joint-limit margin | `1e-5 rad` |
| position convergence tolerance | `1e-3 m` |
| rotation convergence tolerance | `1 degree` (`pi/180 rad`) |

Position residual is the Euclidean norm of target minus current site position. Rotation
residual is the SO(3) geodesic angle:

```text
acos(clip((trace(target_R @ current_R.T) - 1) / 2, -1, 1))
```

Both residuals must pass in the same iteration. Reaching 80 iterations, a non-finite value,
or a failed linear solve is non-convergence.

The first frame of an episode uses the model's neutral configuration. Later frames use the
previous valid solution as their seed, which selects a continuous solution for the Panda's
redundant degree of freedom. The seed resets at every episode boundary, including when a
dataloader batch spans two episodes.

The returned immutable pose contains native-frame joint/link points, hand/finger geometry,
the solved qpos, position residual, rotation residual, and convergence status. Rerun logging
never reaches into the simulator directly.

After the arm IK solve, `state[6]` and `state[7]` are copied into Panda finger joint 1 and
finger joint 2 respectively, clipped to their model joint ranges. The simulator performs one
forward-kinematics update before finger bodies and fingertip positions are extracted. Thus
the displayed gripper opening follows the recorded 8D state instead of remaining at the
model's default opening.

## IK Failure Contract

A converged pose must satisfy explicit position and rotation tolerances recorded in code and
tests. A non-converged solve must never be silently displayed as a valid current pose.

- At every episode boundary, first log a recursive Rerun clear on the complete
  `demonstration/robot` skeleton subtree and reset the IK seed/history. This prevents
  latest-at data from the previous episode leaking into the new episode.
- If a later frame fails and the current episode has a previous valid pose, log a warning
  with episode, frame, position residual, and rotation residual, then redraw that previous
  pose with label `IK reconstructed (stale)` and log a dynamic stale flag.
- If the first frame of an episode fails, the cleared skeleton subtree remains empty while
  cameras, EEF point, paths, and action plots continue. Repeated failures remain empty until
  the first successful solve in that episode.
- IK failure does not invalidate offline action metrics or prediction Parquet output.
- Invalid input shapes, NaN/Infinity, a missing Panda model entity, or an invalid display
  rotation are configuration/data errors and fail before misleading geometry is written.

The semantics document in Rerun states that the arm is IK-reconstructed and that an orange
prediction path is not a simulator rollout.

## 3D Rerun Presentation

The top area keeps the two recorded camera views next to one `Spatial3DView`. The 3D view
uses a line grid and contains:

- a labelled robot-base marker at display `[0, 0, 0]`;
- red `X`, green `Y Forward`, and blue `Z Up` arrows with visible text labels;
- a blue Panda line skeleton from base through the seven links and hand;
- blue joint points and gripper/finger geometry;
- a blue current EEF point and growing recorded demonstration path;
- the orange prediction action-chunk preview and endpoint;
- no orange prediction skeleton.

The skeleton label is `Demonstration arm - IK reconstructed`. The prediction entities retain
`Prediction (not executed)` labels. Camera images remain recorded dataset images. Object and
fixture meshes are not reconstructed; the adjacent RGB views provide scene/object context.

The display axes are logged as real `Arrows3D` entities rather than relying only on Rerun's
corner orientation widget. Their conventional colors are:

| Axis | Meaning | Color |
|---|---|---|
| X | lateral | red |
| Y | forward toward workspace | green |
| Z | up | blue |

## Action-Series Interaction

Use one Rerun `TimeSeriesView` for action comparison. Its plot legend is visible and contains
fourteen independently clickable series. All fourteen are visible by default. Clicking a
legend item toggles only that series, so the user can retain one dimension pair, one ground
truth or prediction line, all ground-truth lines, or any arbitrary subset.

Entity paths contain stable numeric prefixes:

```text
comparison/actions/00_x/00_gt
comparison/actions/00_x/01_pred
comparison/actions/01_y/00_gt
comparison/actions/01_y/01_pred
...
comparison/actions/06_gripper/00_gt
comparison/actions/06_gripper/01_pred
```

Static `SeriesLines` configuration replaces the internal paths with clean visible names:

```text
x · GT, x · Pred, y · GT, y · Pred, ..., gripper · GT, gripper · Pred
```

The `TimeSeriesView.contents` list explicitly enumerates these fourteen paths in the approved
order; ordering never relies on a recursive query, dictionary iteration, or implicit viewer
sorting. It sets `plot_legend=PlotLegend(visible=True)`. Each path is a separate scalar entity
with one static `SeriesLines(visible_series=[True])`, so all fourteen legend entries start
visible and each can be toggled independently in Rerun 0.26.2.

This guarantees that the GT and prediction for one dimension are adjacent in the legend and
overlaid on the same axes. The action comparison is explicitly the first child and active
tab of the lower `Tabs` container. Absolute Error, Latency, semantics, and Metrics remain
separate later tabs.

### Fourteen-Color Contract

Every series has a unique color. Each dimension uses one hue family, with a darker GT and a
lighter prediction color:

| Dimension | GT | Prediction |
|---|---|---|
| x | `#4E79A7` | `#A0CBE8` |
| y | `#F28E2B` | `#FFBE7D` |
| z | `#59A14F` | `#8CD17D` |
| roll | `#E15759` | `#FF9D9A` |
| pitch | `#B07AA1` | `#D4A6C8` |
| yaw | `#9C755F` | `#D7B5A6` |
| gripper | `#2F9EAA` | `#76C7C9` |

Names, ordering, and colors are explicit static data; they must not depend on dictionary
iteration or Rerun's automatic color assignment.

## Offline Evaluation Data Flow

For each visualized row, `offline_act.py` already has all required policy-comparison data. It
will additionally ask the reconstructor for one skeleton pose:

```text
raw batch observation.state
    ├── policy preprocessor -> ACT -> policy postprocessor -> predicted action chunk
    └── PandaSkeletonReconstructor -> native skeleton pose

native state/path/skeleton + predicted chunk
    -> DisplayFrame
    -> Rerun 3D entities

target first action + predicted first action
    -> fourteen ordered scalar entities
```

IK always consumes the raw dataset state before policy preprocessing. Prediction preview and
action plots always consume the checkpoint-postprocessed action space. These stages are
named explicitly in variables and covered by tests.

## Resource and Lifecycle Behavior

Creating the LIBERO Panda model adds a one-time startup cost to visualized offline runs. The
same model and solver are reused for all displayed frames. IK is executed only for selected
visualization episodes; nonvisualized frames still contribute metrics and Parquet rows but
do not pay IK cost.

The simulator/model is closed in a `finally` or context-manager path. Rerun finalization and
the existing evaluator output-directory lock retain their current cleanup guarantees.
Construction is transactional: if the environment is created but model-entity discovery or
solver initialization fails, the constructor closes the environment before re-raising.

## Verification

Unit and bounded integration tests cover:

- `DisplayFrame` maps the native base to zero, native +X to display +Y, native -Y to display
  +X, and preserves +Z;
- display rotation orthonormality and right-handed determinant validation;
- all 3D entities, including skeleton and prediction preview, use the same transform;
- the base marker and the three arrows have exact X, `Y Forward`, and `Z Up` labels and
  red/green/blue colors;
- IK output has the expected Panda link order and finite joint values within limits;
- FK-generated reachable Panda targets round-trip through IK within `1 mm` and `1 degree`
  under the exact 80-iteration/damping/gain/step-limit contract;
- a five-target FK sequence generated with per-joint source increments no larger than
  `0.02 rad` uses the previous solution, remains finite and within limits, and reconstructs
  with consecutive per-joint changes no larger than `0.10 rad`; a new episode resets to the
  neutral seed;
- changing `state[6:8]` changes the extracted finger and fingertip geometry after FK;
- the episode-boundary recursive clear prevents a previous episode's skeleton from appearing
  when the next episode's first solve fails, and later failures redraw stale geometry only
  when a current-episode previous-valid pose exists;
- metrics-only evaluation does not construct the IK component;
- normal completion, evaluation exceptions, and partial constructor failure all close the
  simulator/model exactly once;
- exactly fourteen action scalar paths and fourteen static series styles are logged;
- all fourteen colors are unique, all series default to visible, and the visible names/order
  are exactly the approved paired sequence;
- the action blueprint explicitly enumerates all fourteen paths, is the default first lower
  panel, includes a visible interactive legend, and retains the other tabs;
- a small Rerun recording verifies successfully and contains axes, skeleton, recorded path,
  prediction preview, and the fourteen scalar series;
- existing closed-loop Rerun entity paths and behavior remain unchanged.

## Non-Goals

This change does not recover recorded Panda joint angles, download or replay raw LIBERO
demonstrations, render object/fixture meshes, register 3D geometry into either RGB camera,
execute the predicted chunk, render an orange predicted robot, or change ACT training,
metrics, Parquet schemas, and closed-loop evaluation.
