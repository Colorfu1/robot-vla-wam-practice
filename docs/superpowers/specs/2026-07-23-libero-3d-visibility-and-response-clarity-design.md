# LIBERO 3D Visibility and Response Clarity Design

**Date:** 2026-07-23
**Status:** Reviewed and implemented

## 1. Goal

Make the LIBERO closed-loop Free 3D view easier to inspect without changing policy
inference, simulator rollout, saved metrics, or action semantics.

The change has two outcomes:

1. the measured cyan end-effector response remains visible when it overlaps the orange
   controller command;
2. users can independently show or hide each meaningful 3D visual through Rerun's native
   Blueprint Panel.

## 2. Interaction Model

Use the stock Rerun Blueprint Panel instead of a custom in-viewport control.

- The Blueprint Panel is expanded when the closed-loop recording opens.
- Its entity hierarchy is the visibility control: the eye icon beside a group or entity
  shows or hides that content in the selected view.
- The Free 3D view initially shows every visual.
- Users may freely combine visible groups rather than selecting a fixed visualization
  preset.
- Rerun's existing state persistence or saved-blueprint mechanisms remain responsible for
  remembering manual viewer layout and visibility changes.

No custom Rerun viewer plugin or overlay checkbox is introduced.

## 3. Selectable 3D Groups

The existing entity paths remain the source of truth and are exposed as these logical
groups in the Free 3D view:

| Entity root | User-facing meaning |
|---|---|
| `display_frame/**` | Coordinate frame, XYZ axes, and spatial reference |
| `rollout/calibration/**` | Fixed agentview camera calibration and frustum |
| `rollout/robot/**` | Robot skeleton from measured simulator joints |
| `rollout/trajectory/**` | Measured end-effector history and current EEF |
| `rollout/command/**` | Current controller command sent to LIBERO |
| `rollout/response/**` | Measured EEF response after that command |
| `rollout/plan/queued/**` | Remaining standard ACT actions that can execute |
| `rollout/plan/discarded/**` | Predicted standard ACT tail that will be discarded |
| `rollout/plan/new_prediction_chunk/**` | Temporal Ensemble's newly predicted chunk |

The calibrated 2D overlay continues to use the same geometry. Visibility selection is
primarily a Free 3D inspection feature because Rerun applies entity visibility per view.

## 4. Command and Response Geometry

The current orange command and measured cyan response begin at the same measured EEF at
time `t`. In successful motion their directions are often almost identical, so equal-width
line strips can occlude one another.

Change their visual encoding:

- **Current command:** retain the orange command segment but reduce its radius.
- **Actual response:** replace the cyan line strip with a thicker cyan arrow whose origin
  is the EEF at `t` and whose vector reaches the measured EEF at `t+1`.
- **Response endpoint:** add a cyan point at the measured EEF at `t+1`.

The arrow and endpoint are separate entities under `rollout/response/**`, so the whole
response can be toggled as one group or its parts can be selected individually.

This is visualization only. No scaling is applied to the measured response, and neither
the command nor response changes rollout data or metric calculations.

## 5. Blueprint Behavior

`build_libero_blueprint()` explicitly adds an expanded `BlueprintPanel` while retaining:

- the expanded `global_step` Time Panel;
- the existing calibrated rollout and Free 3D tabs;
- all existing action, state, reward, latency, semantics, and metrics views.

The Free 3D view lists the selectable roots explicitly rather than relying on a broad
catch-all query. This preserves meaningful eye-toggle groups and avoids pulling unrelated
entities into the view.

The documentation tab is updated to state:

- orange is the controller command;
- cyan arrow and point are the measured simulator response;
- 3D groups can be toggled with eye icons in the Blueprint Panel.

## 6. Compatibility and Failure Handling

- Existing closed-loop RRD files remain readable; they simply lack the new response
  endpoint entity and retain their old response archetype.
- Newly generated RRD files use the new geometry.
- No output schema version change is required because RRD visualization entities are not
  part of `rollouts.parquet`.
- A zero-length response still logs an arrow with a zero vector and an endpoint at the
  measured position; the endpoint preserves visibility.
- Offline ACT visualization is unchanged.

## 7. Verification

Automated tests must verify:

- the Free 3D content query contains every selectable group;
- the fixed agentview calibration/frustum remains available as a selectable group;
- the LIBERO blueprint requests an expanded Blueprint Panel;
- the orange command uses the reduced line radius;
- the cyan response arrow has the correct origin and vector after display-frame
  transformation;
- the cyan response endpoint is logged at the transformed `t+1` EEF;
- response arrow and endpoint share the cyan color and remain under
  `rollout/response/**`;
- existing queue, discarded-tail, Temporal Ensemble, image, chart, and metrics tests still
  pass.

A real closed-loop RRD smoke check must then confirm:

- the Blueprint Panel opens with independently toggleable entity groups;
- the cyan response is visible beside or over the thinner orange command;
- hiding and restoring groups does not affect timeline playback.

## 8. Files in Scope

- `src/robot_practice/evaluation/rerun_visualization.py`
- `tests/test_evaluation_rerun.py`
- the closed-loop visualization usage/semantics documentation if its wording duplicates
  the in-view semantics

Generated recordings and experiment outputs remain untracked.
