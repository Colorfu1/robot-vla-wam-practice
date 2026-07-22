# Offline Teacher-Forced Command Segments Design

## Status

Approved in conversation on 2026-07-22. This design corrects the misleading comparison
between recorded EEF history and a future predicted action chunk in the existing offline
ACT 3D view.

## Goal

At every demonstration frame, show what the expert and ACT commanded from the same recorded
EEF state while keeping the robot's actually recorded EEF history visually separate from
unexecuted controller commands.

## Data Semantics

For dataset frame `t`, the evaluator receives:

```text
observation[t]       recorded RGB + recorded 8D EEF/gripper state
target_action[t]     expert OSC command applied after observation[t]
predicted_chunk[t]   20 ACT commands predicted from observation[t]
```

`predicted_chunk[t, 0]` is the predicted command corresponding to `target_action[t]`.
Neither command is an absolute next EEF pose. LIBERO's relative OSC controller maps the
normalized translation command to a position-goal delta of at most `0.05 m`; the actual
next recorded EEF can move less because of controller response, constraints, and contacts.

## 3D Presentation Contract

The 3D view contains four distinct geometries:

| Geometry | Meaning | Lifetime |
|---|---|---|
| deep-blue connected line | recorded GT EEF positions through frame `t` | growing episode history |
| light-blue independent segments | expert first-action translation commands | retained episode history |
| deep-orange independent segments | predicted first-action translation commands | retained episode history |
| translucent light-orange line | current frame's complete predicted chunk command preview | current frame only |

For each frame, the target and prediction command segments have the same recorded EEF
origin:

```text
origin[t] = observation.state[t, :3]
gt_command_end[t] = origin[t] + 0.05 * clip(target_action[t, :3], -1, 1)
pred_command_end[t] = origin[t] + 0.05 * clip(predicted_chunk[t, 0, :3], -1, 1)
```

Each pair is logged as two separate two-point line strips. Segments from different frames
must never be connected into a synthetic trajectory. At frame `t+1`, the recorded blue
history extends to `observation.state[t+1, :3]`, and a new command pair begins from that
new recorded point. All earlier command pairs remain visible until the episode changes.

The current chunk preview begins at the recorded EEF that produced the model input and
integrates all valid predicted translation commands in sequence:

```text
preview[t, 0] = origin[t]
preview[t, k+1] = preview[t, k] + 0.05 * clip(predicted_chunk[t, k, :3], -1, 1)
```

The preview is overwritten at every Rerun time point rather than accumulated across frames.
Its first segment overlaps the current deep-orange first-action segment; the opaque segment
makes the one-step comparison visible while the translucent line provides future context.

All recorded points, command segments, chunk preview points, skeleton geometry, and the
robot base use the same existing `DisplayFrame` transform. Moving the fixed native Panda
base from `[-0.6, 0, 0]` to display `[0, 0, 0]` therefore preserves all relative geometry.

## Entity and Color Contract

Use these paths and colors:

| Entity path | Color |
|---|---|
| `demonstration/trajectory/eef_recorded` | existing deep blue `(0, 114, 178)` |
| `comparison/commands/gt_first_action_segments` | light blue `(86, 180, 233, 190)` |
| `comparison/commands/pred_first_action_segments` | deep orange `(230, 159, 0)` |
| `prediction/not_executed/chunk_preview` | translucent orange `(255, 190, 80, 96)` |
| `prediction/not_executed/chunk_endpoint` | translucent orange `(255, 190, 80, 128)` |

The two command-history entities retain explicit labels but keep `show_labels=False`,
matching the click-to-inspect label contract. User-facing labels must include `command` and
the prediction label must include `teacher-forced, not executed`. The current preview label
must include `future command preview, not executed`.

The Spatial3D blueprint explicitly includes `comparison/commands/**`. Existing camera,
skeleton, axis, action-series, error, latency, and metrics views remain unchanged.

## Episode and Padding Behavior

- Recorded points and both command-segment histories reset together at every episode
  boundary.
- The first frame starts the blue history with one point and adds one GT/Pred command pair.
- The current action must be valid; malformed or padded current actions are a data-contract
  error.
- Future preview integration stops at the first padded chunk element and rejects a
  non-contiguous valid mask.
- An episode with a one-step valid horizon still shows its first-action pair and a two-point
  current preview.

## Interpretation

The deep-blue line is an actual recorded state trajectory. The blue/orange small segments
and light-orange preview are OSC position-goal command visualizations, not measured motion.
Their endpoints must not be interpreted as the next observed EEF position.

This view directly answers whether ACT predicts the expert translation command at each
teacher-forced observation. Rotation commands and gripper commands remain in the paired
action chart; a 3D position line does not represent them.

## Non-Goals

This change does not create a cumulative orange Pred trajectory, execute actions, simulate
controller response, restore raw MuJoCo states, recover exact joint truth, or change offline
metrics and prediction Parquet output.

## Verification

Tests cover:

- exact one-step endpoints for clipped GT and Pred translations;
- each frame's two segments sharing the recorded EEF origin;
- historical command pairs remaining separate rather than becoming one connected strip;
- synchronized reset of recorded and command histories at episode boundaries;
- exact DisplayFrame transformation of recorded history, both command histories, and the
  current chunk preview;
- exact entity paths, colors, alpha values, hidden labels, and semantic label text;
- current preview replacement semantics and padding behavior;
- unchanged closed-loop Rerun paths;
- a valid bounded RRD containing recorded history, retained command pairs, and one current
  translucent future preview.

Existing RRD files remain immutable. The offline evaluator must be rerun, with `--overwrite`
when reusing an output directory, before this presentation appears.
