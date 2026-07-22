# Daily Progress Report Design

## Goal

Create a concise, durable record of the work completed in the LeRobot practice repository
on 2026-07-16. The report is intended for the repository owner returning to the work later,
not as a formal experiment result or an upstream-facing changelog.

The final report is now maintained at
`reports/2026-07-16-LeRobot-ACT调试与ALOHA数据可视化.md`.

## Organization

Use an outcome-oriented structure rather than a chronological transcript:

1. goals and current conclusion;
2. LeRobot environment and training command confirmation;
3. VS Code/Python Debug configuration;
4. ALOHA dataset contents and cache locations;
5. synchronized signal, image, and nominal 3D visualization;
6. tests and manual validation;
7. current running state and incomplete work;
8. commands and next actions.

## Source Of Truth

The report must derive claims from repository files and locally verified runtime state. It
should link to reusable source files, specifications, tests, the generated Rerun recording,
and relevant configuration where useful.

Commands should be copyable from the repository root. Long terminal transcripts, raw cache
blob hashes, implementation internals that are already documented in a design spec, and
temporary debugging artifacts are excluded.

## Accuracy Boundaries

- Clearly separate completed work from work still in progress.
- Do not describe the active three-step ACT Debug run as a completed training experiment.
- Record that the current training process is downloading into LeRobot's default dataset
  root even though a complete Hub snapshot cache also exists.
- Do not invent required experiment metrics such as train loss, closed-loop success rate,
  or latency when the run has not produced them.
- Describe the 3D arms as nominal ViperX 300S forward kinematics, not camera-registered scene
  reconstruction or physics simulation.
- Treat generated `.rrd`, screenshots, datasets, and videos as ignored local artifacts.
- Note that this workspace is not a Git checkout, so no commit can be created here.

## Expected Detail

Include exact versions and verification totals only when already checked today. Include the
two meaningful ALOHA cache roots and explain why the second download occurred, but avoid a
full listing of content-addressed blobs. Preserve the most useful launch and visualization
commands so the work can be resumed without reconstructing the conversation.

## Acceptance Criteria

- A reader can identify what was completed and what remains active.
- A reader can launch the ACT Debug configuration and open the Rerun visualization.
- A reader can locate both the complete Hub snapshot and the default LeRobot dataset root.
- The report points to the custom kinematics, visualization, tests, and design documents.
- All reported test counts, paths, versions, and process states are evidence-backed.
- The Markdown is concise and does not duplicate the full visualization design spec.
