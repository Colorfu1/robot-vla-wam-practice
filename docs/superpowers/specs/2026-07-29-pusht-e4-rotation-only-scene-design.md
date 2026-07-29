# PushT E4 Rotation-Only Scene Design

**Date:** 2026-07-29
**Status:** Approved; Phase 0 implemented and smoke validated

## Purpose

E4 introduces a deliberately simplified PushT scene for a later study of two opposite
rotation strategies. The initial T and target T share one geometric center, and the target
orientation differs from the initial orientation by 180 degrees. Therefore the target pose
can be reached without any net translation.

This phase defines and validates only the scene. It does not generate demonstrations, train a
policy, execute a CW or CCW controller, or claim that either strategy is reachable through
closed-loop agent actions.

The previously planned Mode-Conditioned experiment moves from E4 to E5. It remains planned and
is not implemented as part of this work.

## Scope

### Included

- A repository-local PushT environment implementation under
  `src/robot_practice/experiments/pusht_diffusion/`.
- A T whose physical center of gravity equals its geometric area centroid.
- A deterministic initial block pose, target pose, and agent pose.
- The existing PushT overlap-based reward and success definition.
- Geometry, reset, reward, success, and rendering tests.
- One generated PNG preview and a small machine-readable scene summary under `outputs/`.
- Guide and index updates that make the new E4/E5 numbering explicit.

### Excluded

- CW or CCW controllers.
- Feasibility or rollout runners.
- Pymunk pivot constraints; the T remains a free rigid body.
- LeRobotDataset creation or demonstration collection.
- BCChunk, Diffusion, or Mode-Conditioned training.
- Offline or closed-loop model evaluation.
- Rerun recordings, videos, checkpoints, or formal experiment results.

## Geometry Contract

The PushT block is the union of two rectangles in block-local coordinates:

```text
crossbar: [-60, 60] x [0, 30],   area=3600, centroid=(0, 15)
stem:     [-15, 15] x [30, 120], area=2700, centroid=(0, 75)
```

The geometric area centroid is:

\[
c_y =
\frac{3600 \times 15 + 2700 \times 75}{3600 + 2700}
= \frac{285}{7}
\approx 40.7142857
\]

The environment sets:

```text
block.center_of_gravity = (0, 285/7)
rotation_center = (256, 256)
start_angle = 0
goal_angle = pi
```

Changing the raw block and goal `(x, y)` fields to the same values would not align the T
centers because the local origin is not the T centroid. For a desired world-space center
\(C\), local center \(c\), and orientation \(\theta\), the body-origin pose is:

\[
p(\theta) = C - R(\theta)c
\]

This gives:

```text
start body origin = (256, 256 - 285/7)
goal body origin  = (256, 256 + 285/7)
```

Both rendered T area centroids must equal `(256, 256)`. The actual Pymunk
`block.center_of_gravity` must equal `(0, 285/7)`, and its world-space position must remain
`(256, 256)` at the start and goal orientations. The block angle must be assigned before the
body position so Pymunk rotation around a nonzero center of gravity does not shift the requested
world-space center.

## Initial Agent Pose

The agent is placed on the T symmetry axis, outside the midpoint of the crossbar's outer edge.
The circle has the standard PushT radius of 15 pixels and starts with a 1-pixel clearance:

```text
agent_start_local = (0, -(15 + 1))
agent_start_world = start_body_origin + R(start_angle) @ agent_start_local
```

The agent therefore does not collide with the block during reset. If it later approaches along
the symmetry axis, its first nominal contact point is the center of the crossbar. The target
region is non-collidable, so visual overlap between the initial agent and the target overlay
does not alter physics.

## Environment Boundary

The implementation is repository-local and must not edit the installed `gym_pusht` package.
A small local subclass or equivalent environment factory may reuse the upstream physics,
observation, action, rendering, overlap, reward, and termination logic.

The public construction boundary is one importable function:

```python
make_e4_rotation_only_env(...)
```

The factory returns the E4 environment with the deterministic scene contract applied. This
phase does not register a new global Gymnasium environment ID because no training integration
is in scope.

Reset seeds may affect future stochastic extensions but must not change the Phase-0 initial
block, goal, or agent poses. A conflicting caller-supplied reset state must be rejected rather
than silently violating the E4 geometry contract.

## Reward And Success Contract

E4 keeps the installed PushT definition:

\[
coverage =
\frac{Area(T_{block} \cap T_{goal})}{Area(T_{goal})}
\]

```text
reward = clip(coverage / 0.95, 0, 1)
success = coverage > 0.95
terminated = success
```

Expected scene invariants are:

```text
initial coverage ~= 0.3877551020
initial reward   ~= 0.4081632653
initial success  = false

exact-goal coverage = 1.0
exact-goal reward   = 1.0
exact-goal success  = true
```

The exact-goal check moves the agent to a collision-free location before stepping the
environment. This ensures it tests the inherited reward and termination path rather than
injecting an artificial collision at the final pose.

These checks prove that the scene's target is geometrically correct and that the environment
can evaluate completion. They do not prove that an agent policy can reach the target.

## Phase-0 Validation Output

The environment module may expose a minimal command-line preview mode. It is not a rollout or
controller entry point. It performs a deterministic reset, verifies the scene invariants, and
writes:

```text
outputs/pusht_diffusion/pilot/e4_rotation_only_scene/
├── scene_preview.png
└── scene_metrics.json
```

`scene_metrics.json` contains only scene constants and measured geometry:

- rotation center;
- start, goal, and agent poses;
- block and goal geometric centroids;
- initial coverage, reward, and success;
- exact-goal coverage, reward, and success;
- success threshold.

The output directory is generated and ignored. Neither artifact is committed.

## Tests

Tests must cover:

1. The area-centroid calculation equals `285/7`.
2. The actual Pymunk `block.center_of_gravity` equals `(0, 285/7)`.
3. Start and goal T geometric centroids both equal `(256, 256)`, and the Pymunk center of
   gravity maps to that same world-space point at both orientations.
4. Reset is deterministic across different seeds.
5. A conflicting caller-supplied `reset_to_state` is rejected.
6. The block is a free rigid body and no pivot constraint is added.
7. The initial agent has a 1-pixel block clearance and no reset contact.
8. Initial coverage, reward, and success match the registered invariants.
9. Placing the block at the exact target pose makes the inherited `step()` path return reward
   `1.0`, success metadata, and `terminated=True`.
10. The policy observation and two-dimensional absolute-position action interface remain
   compatible with the existing PushT environment.
11. Rendering produces a nonempty image at the configured dimensions.

Targeted verification runs inside the project environment:

```bash
conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_pusht_e4_rotation_only_env.py
```

## Documentation Changes

Implementation updates the guide structure without recording planned work as completed:

```text
reports/guides/pusht_diffusion/e4_rotation_only.md
reports/guides/pusht_diffusion/e5_mode_conditioned.md
```

The existing Mode-Conditioned guide is moved to E5, and its title, identifiers, planned output
paths, and incoming links are updated. The E4 guide documents only the Phase-0 scene command,
generated output paths, overlap threshold, and the explicit limitation that no demonstrations,
controllers, training, or model results exist.

The PushT guide index and any repository links to the old E4 Mode-Conditioned path are updated
in the same change. No experiment-log result entry is added because Phase 0 is a scene smoke
validation, not a model experiment result.

## Acceptance Criteria

The implementation is complete when:

- the installed `gym_pusht` source remains unmodified;
- the local E4 scene satisfies every geometry and success invariant;
- the targeted tests pass;
- the preview command writes the two declared ignored artifacts;
- the preview visibly shows two center-aligned T shapes separated only by 180-degree
  orientation;
- E4 and E5 guide numbering is consistent across repository documentation;
- no controller, dataset, RRD, training entry point, checkpoint, or model-result claim is
  introduced.
