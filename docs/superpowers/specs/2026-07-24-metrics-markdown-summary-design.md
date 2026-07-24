# Evaluation Metrics Markdown Summary Design

## Goal

Make evaluation results readable without opening the full `metrics.json`.
Every Offline ACT and LIBERO closed-loop output directory will contain a
human-readable `metrics.md` beside `metrics.json`.

The JSON file remains the complete machine-readable source of truth. The
Markdown file is a compact presentation derived only from that JSON payload.

## Output

`metrics.md` contains:

1. Run context: evaluation mode, status, task identity when available, and the
   number of evaluated episodes.
2. An aggregate table with metric name, formatted value, and a short Chinese
   explanation.
3. For Offline ACT, a seven-row per-action-dimension MAE table.
4. A compact per-episode table.

The context block is ordered as mode, status, task suite, task ID, task name,
and episode count. Task fields are omitted when the optional top-level `task`
mapping is absent.

### Offline Tables

The Offline aggregate table has this fixed row order:

| Label | Source key | Format | Chinese explanation |
|---|---|---|---|
| Episodes | `num_episodes` | integer | 离线评测的 demonstration 数量 |
| Frames | `num_frames` | integer | 参与预测的数据帧数量 |
| Model queries | `num_model_queries` | integer | 模型批量推理调用次数 |
| Batch size | `batch_size` | integer | 每次模型调用的样本数 |
| First-action MAE | `first_action_mae` | 4 decimals | 当前动作预测与真值的平均绝对误差 |
| Delta-action MAE | `delta_action_mae` | 5 decimals | 预测动作变化与真值动作变化的误差 |
| Action TV | `teacher_forced_action_tv` | 5 decimals | 相邻预测动作的平均变化量 |
| Action Jerk | `teacher_forced_action_jerk` | 5 decimals | 预测动作二阶差分的抖动代理 |
| Inference latency p50 | `inference_latency_p50_ms` | 2 decimals + `ms` | 一半模型批调用不超过的耗时 |
| Inference latency p95 | `inference_latency_p95_ms` | 2 decimals + `ms` | 95% 模型批调用不超过的耗时 |

The per-action-dimension table reads `aggregate.per_dimension_mae`. It must
contain exactly seven values, displayed in controller action order:
`x`, `y`, `z`, `roll`, `pitch`, `yaw`, `gripper`. Each value uses four decimal
places. A different list length is invalid rather than silently mislabeled.

Offline per-episode columns have this fixed order:

- episode index
- frame count
- first-action MAE
- delta-action MAE
- teacher-forced action TV
- teacher-forced action jerk

### LIBERO Tables

The LIBERO aggregate table has this fixed row order:

| Label | Source key | Format | Chinese explanation |
|---|---|---|---|
| Formal evaluation | `is_formal` | 是/否 | 是否使用完整任务步数上限 |
| Episodes | `num_episodes` | integer | 闭环 rollout 数量 |
| Successes | `num_successes` | integer | LIBERO 判定成功的 rollout 数量 |
| Success rate | `success_rate` | 1 decimal percent | 成功完成任务的 rollout 比例 |
| Average sum reward | `avg_sum_reward` | 4 decimals | 每个 rollout 累计 reward 的平均值 |
| Average max reward | `avg_max_reward` | 4 decimals | 每个 rollout 最大单步 reward 的平均值 |
| Action TV | `action_tv` | 5 decimals | 相邻执行动作的平均变化量 |
| Action Jerk | `action_jerk` | 5 decimals | 执行动作二阶差分的抖动代理 |
| Mean gripper toggles | `mean_gripper_toggles` | 2 decimals | 平均每次 rollout 的夹爪切换次数 |
| Policy latency p50 | `policy_step_latency_p50_ms` | 2 decimals + `ms` | 一半控制步骤不超过的策略耗时 |
| Policy latency p95 | `policy_step_latency_p95_ms` | 2 decimals + `ms` | 95% 控制步骤不超过的策略耗时 |
| Model latency p50 | `model_query_latency_p50_ms` | 2 decimals + `ms` | 一半模型调用不超过的推理耗时 |
| Model latency p95 | `model_query_latency_p95_ms` | 2 decimals + `ms` | 95% 模型调用不超过的推理耗时 |
| Policy steps | `num_policy_steps` | integer | 发送给模拟器的控制动作数 |
| Model queries | `num_model_queries` | integer | 实际模型推理次数 |

LIBERO per-episode columns have this fixed order:

- episode index
- seed
- initial-state ID
- success
- step count
- summed reward
- action TV
- action jerk
- policy p95 latency
- termination reason

### Formatting

Numbers use Python fixed-point formatting and retain trailing zeros:

- counts are integers
- only `success_rate` is multiplied by 100 and rendered with one decimal plus
  `%`
- all latency source values are milliseconds and rendered with `ms`
- reward and per-episode first-action MAE use four decimal places
- TV, jerk, and delta-action MAE use five decimal places
- booleans render as `是` or `否`
- explicit JSON `null` in a declared nullable field renders as `N/A`

Formatting accepts only JSON numeric values represented in Python as `int` or
`float`; booleans are not numbers. Numeric strings, `NaN`, and infinities are
invalid even in nullable fields. Missing keys and incorrectly typed values
always raise validation errors. Only explicitly nullable metrics may be `null`:
Offline delta-action MAE, TV, jerk, and latency fields; LIBERO success rate,
reward, TV, jerk, and latency fields; and the corresponding per-episode derived
metrics.

Task names and termination reasons are escaped before entering Markdown tables:
backslashes and pipe characters are escaped, and newlines become spaces.

## Accepted Metrics Contract

The renderer accepts `schema_version == 2` and `mode` equal to `offline` or
`libero`. Unknown versions and modes raise a validation error.

Both modes require:

- integer `schema_version` exactly equal to `2`
- string `mode` equal to `offline` or `libero`
- top-level `status` equal to `complete` or `smoke`
- an `aggregate` mapping
- a non-empty `per_episode` JSON array whose elements are mappings
- all aggregate keys listed in the corresponding aggregate table

The existing evaluator schema stores `per_episode` as an array, never as a JSON
object keyed by episode. Each row obtains its identity only from its required
`episode_index` field.

`metric_definitions`, `task`, `checkpoint_task_assessment`,
`visualized_episodes`, and other existing provenance fields are optional and do
not change report layout.

Offline additionally requires `aggregate.per_dimension_mae` to be the
seven-value list described above. `aggregate.chunk_mae_by_horizon` remains in
JSON but is intentionally omitted from the compact report.

Offline episode rows require integer, non-negative `episode_index` and
`num_frames`; they also require `first_action_mae`,
`delta_action_mae`, `teacher_forced_action_tv`, and
`teacher_forced_action_jerk` with the nullability rules above.

LIBERO episode rows require non-negative integer `episode_index`, `seed`,
`init_state_id`, and `num_model_queries`; positive integer `num_steps`; boolean `success`;
string `termination_reason`; and `sum_reward`, `max_reward`, `action_tv`,
`action_jerk`, `gripper_toggles`, and the four policy/model latency fields.
Only the derived TV, jerk, and latency fields are nullable.

Episode rows are numerically sorted by `episode_index`; duplicate indices are
invalid. `aggregate.num_episodes` must equal the number of episode rows, so an
empty result is invalid. Other count fields must be non-negative integers.
`success_rate` must be `null` or fall in `[0, 1]`. When the optional `task`
mapping exists, its displayed `suite` and `task_name` are strings and `task_id`
is a non-negative integer.

## Components and Data Flow

Add a focused metrics-report module under
`src/robot_practice/evaluation/`. It accepts an already-loaded metrics mapping,
renders deterministic Markdown, and writes `metrics.md` atomically.

Both evaluators render and validate Markdown before publishing either summary
file. They atomically write `metrics.md` first and `metrics.json` last:

```text
compute metrics
    -> validate and render the in-memory payload
    -> atomically write metrics.md
    -> atomically write metrics.json as the completion marker
```

The renderer supports the existing `offline` and `libero` schemas without
changing metric computation or Rerun visualization.

The module also exposes a small CLI. Given a directory, it recursively finds
`metrics.json` files and writes a sibling `metrics.md`. This is used once to
backfill existing ignored output artifacts and remains available for future
repair or regeneration.

## Validation and Error Handling

For evaluator-generated results, an invalid required metrics structure is a
programming error and must not silently produce a misleading report. The writer
raises a clear exception before the evaluator reports successful completion.
An evaluator result is complete only when both sibling files exist;
`metrics.json` is written last as the completion marker.
`create_output_directory` guarantees a fresh result directory: an explicit
overwrite validates and removes the old evaluator directory before recreating
it. Therefore an evaluator never replaces these two files inside a directory
that still contains a stale `metrics.json`.

For recursive backfill, one invalid or unreadable JSON file emits a warning and
is skipped. Read, validation, permission, and write failures are all counted as
skips, and traversal continues. The CLI reports generated and skipped counts
and returns a non-zero exit status when any file is skipped.

Unknown extra metrics remain in `metrics.json`; they do not automatically widen
the compact tables. This keeps the report stable as the machine-readable schema
grows.

An existing `metrics.md` is atomically overwritten. Rendering is deterministic,
so repeated backfill runs with unchanged JSON produce byte-identical Markdown.

## Existing Results

After implementation, run the backfill CLI against `outputs/`. It creates only
ignored `metrics.md` artifacts beside existing `metrics.json` files. It does not
modify checkpoints, recordings, Parquet files, videos, JSON results, or dataset
artifacts.

## Tests

Unit tests cover:

- exact golden Markdown output for Offline and LIBERO reports
- aggregate, action-dimension, and per-episode ordering
- percentage, latency, boolean, and missing-value formatting
- Markdown escaping
- schema-version and unknown-mode rejection
- empty, partial, duplicate, and malformed episode cases
- non-finite and incorrectly typed numeric values
- repeat-render byte equality
- atomic overwrite and write-failure behavior
- recursive backfill with mixed valid, malformed, and unwritable results,
  including exit status and generated/skipped counts

Integration tests assert that both evaluators create `metrics.md` beside
`metrics.json`.
