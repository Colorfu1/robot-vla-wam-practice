# PushT Compact Rerun Layout Design

**Date:** 2026-07-28
**Status:** Approved for implementation

## Problem

The stable-path PushT Blueprint creates four independent views: simulator, reward,
rollout metadata, and task context. A real Viewer run also restored the previous
seed-specific persisted Blueprint, producing many empty legacy views. After Reset
Blueprint, the generated four-view layout would still spend most of its area on
secondary information.

Real `env_1004.rrd` inspection confirmed:

- `/rollout/simulator` exists at every frame `0..186`;
- `/rollout/reward` exists at frames `1..186`;
- `/rollout/metadata` exists only at frame `186`;
- `/task/*` is one static row.

The primary comparison workflow needs a large simulator view and a readable reward
curve. Metadata remains useful for inspection but does not need permanent screen space.

## Chosen Layout

The embedded default Blueprint contains exactly two views:

1. a `Spatial2DView` named `PushT rollout`, bound to the canonical simulator, complete
   paths, current agent/block, and current action target entities;
2. a `TimeSeriesView` named `Reward`, bound only to `/rollout/reward`.

Use one horizontal container with approximate column shares `3:1`. Do not create
`TextDocumentView`, `DataframeView`, tab containers, or auto-generated views.

The Blueprint remains:

```python
make_active=False
make_default=True
```

This preserves user-edited layouts. A machine carrying the old persisted PushT layout
must still perform one Reset Blueprint to select the regenerated compact default.

## Metadata

Keep `/rollout/metadata` and all `/task/*` entities in every RRD so they remain available
through the data tree and Selection panel. Log `/rollout/metadata`,
`/rollout/success`, and `/rollout/rollout_index` as static data rather than attaching
them only to the final timeline frame.

This changes only their visualization time semantics. It does not change rollout data,
reward, success classification, filenames, Recording IDs, application IDs, entity
paths, comparison JSON, Parquet, videos, or metrics.

## Verification

Automated tests must verify:

- the Blueprint contains exactly one `Spatial2DView`, one `TimeSeriesView`, and one
  horizontal container;
- the Blueprint contains no text, dataframe, tab, or auto-generated view;
- every Blueprint query uses `/rollout/*` and contains no seed-specific path;
- Blueprint activation remains `make_active=False`, `make_default=True`;
- metadata, success, and rollout index are static and retain their expected values;
- simulator and reward timelines remain unchanged;
- the complete PushT test suite passes.

Manual acceptance uses newly generated RRDs or a bounded fixture, performs Reset
Blueprint once, and confirms that only the simulator and reward panes remain and both
update correctly when switching between at least two Recordings.

## Alternatives Rejected

- **Simulator only:** maximizes image area but removes the reward curve used for
  comparison.
- **Keep four views and make metadata static:** removes one empty-state symptom but
  retains unnecessary permanent panes.
- **Force the Blueprint active:** hides stale layouts but breaks the repository's
  persisted-layout contract by overwriting user customization on every open.
