# PushT E0 Full-Demonstration Audit Design

## Goal

E0 inspects all 206 PushT demonstrations from beginning to end, identifies global
trajectory strategies, and selects representative observation histories for E1. It
does not need to prove strict conditional multimodality and does not block E1, E2,
training, or latency evaluation.

## Data

For every frame E0 records the dataset-provided agent position, absolute action
target, reward, terminal fields, and rendered image. A codec-tolerant color/connected
component extractor obtains the moving gray-blue T-block center, orientation, and
agent-to-block surface distance. Extraction failures remain explicit and contribute
to a measured coverage value.

For each complete episode E0 reports approach side, block translation direction,
rotation-first behavior, contact segments, re-contact behavior, maximum reward, and
geometry coverage. `next.done` is treated only as a terminal marker. It is not
silently converted into task success.

## Candidate Selection

Global strategy labels are qualitative navigation aids rather than conditional-mode
claims. E1 candidates are selected deterministically across strategy groups from the
committed audit split. The anchor is the frame immediately before first contact when
available, with enough remaining frames for the eight-step future-action window.

Strict matching of similar two-frame conditions with divergent future chunks is an
optional E0+ analysis. Its absence does not invalidate E1's fixed-input sampling test.

## Visualization and Outputs

Each demonstration is written as a task-local Rerun Recording with a local frame
timeline. The final E0 archive contains one selectable Recording per episode and
shows the rendered frame, complete and current agent/block/action trajectories,
block orientation, reward, split, strategy label, and terminal/outcome metadata.

The output also contains frame geometry Parquet, episode features Parquet, strategy
groups, candidate tensors/images, metrics, provenance, and a measured Markdown
summary. Documentation must distinguish global trajectory diversity from strict
conditional multimodality.

## Acceptance

- all 206 complete episodes are processed;
- moving-block extraction coverage and failures are reported;
- every episode has a strategy label and a selector-visible Rerun Recording;
- 5–10 representative audit candidates contain two raw frames, both states, future
  eight-step actions, future frames, source split, and selection reason;
- no result interprets `next.done` alone as task success;
- tests cover block extraction, episode classification, representative selection,
  launcher behavior, and measured output structure.
