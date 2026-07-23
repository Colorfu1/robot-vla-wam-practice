# Rerun Metric Font Compatibility Design

**Date:** 2026-07-23
**Status:** Draft for specification review

## 1. Goal

Remove missing-glyph boxes from the Rerun metrics panel without changing metric values,
metric computation, saved JSON, or terminal annotations.

Rerun 0.26.2 embeds an Inter font that does not cover Chinese glyphs. The stored UTF-8
metric definitions are valid, but Chinese descriptions become boxes in the native viewer.

## 2. Chosen Scope

Apply an English compatibility layer only when producing Rerun Markdown:

- `metrics.json` keeps the existing Chinese `metric_definitions`.
- terminal metric output keeps the existing Chinese annotations.
- Rerun's `summary/metrics` document uses English descriptions.
- offline and closed-loop metric values, ordering, formatting, and calculations remain
  unchanged.

Rebuilding Rerun with a CJK font and globally translating metric definitions are outside
scope.

## 3. Display Mapping

Add a Rerun-specific English description mapping in
`rerun_visualization.py`. It covers every metric currently emitted by offline and
closed-loop evaluation.

Description selection follows this order:

1. use the Rerun-specific English description for a known metric key;
2. for an unknown key, retain the supplied description only when it is ASCII;
3. for an unknown key with a non-ASCII description, omit the description while retaining
   the metric key and rendered value.

This fallback prevents future unknown Chinese annotations from reintroducing missing-glyph
boxes without hiding the measurement itself.

The generated Rerun Markdown must be ASCII-only for all current evaluator payloads.

## 4. Data Flow

```text
metric calculation
  └─ metrics payload with Chinese metric_definitions
       ├─ metrics.json: unchanged Chinese definitions
       ├─ terminal logger: unchanged Chinese definitions
       └─ Rerun metrics_markdown():
            known key -> English Rerun description
            unknown ASCII description -> original description
            unknown non-ASCII description -> no description
```

The RRD schema and application ID do not change.

## 5. Compatibility

- Existing RRD files retain their embedded Chinese Markdown and are not modified in place.
- Newly generated RRD files use the English display descriptions.
- The generic `metrics_markdown()` behavior remains useful for callers with custom ASCII
  descriptions.
- Unknown aggregate values continue to appear even when their incompatible description is
  omitted.
- No Rerun installation, font package, dataset, checkpoint, or simulator setting changes.

## 6. Verification

Tests must verify:

- the English mapping has a nonempty entry for every key in both
  `OFFLINE_METRIC_DEFINITIONS` and `CLOSED_LOOP_METRIC_DEFINITIONS`;
- all known closed-loop metrics render with the expected English descriptions;
- known offline metrics render with their expected English descriptions instead of merely
  omitting incompatible definitions;
- an unknown metric with an ASCII description retains that description;
- an unknown metric with a Chinese description retains its key/value but omits the
  description;
- current offline and closed-loop Rerun Markdown is ASCII-only;
- the source metrics payload is not mutated;
- the Chinese definitions in `metrics.py` and serialized `metrics.json` remain unchanged;
- the complete existing test suite still passes.

A real RRD smoke check must verify that `/summary/metrics` contains the English descriptions
and no non-ASCII text.

## 7. Files in Scope

- `src/robot_practice/evaluation/rerun_visualization.py`
- `tests/test_evaluation_rerun.py`
- usage documentation only if it explicitly promises Chinese text inside Rerun

Generated RRD, JSON, video, and rollout artifacts remain untracked.
