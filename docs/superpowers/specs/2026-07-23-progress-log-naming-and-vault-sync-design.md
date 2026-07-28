# Progress Log Naming and Vault Sync Design

## Status

Approved in conversation on 2026-07-23. The user selected repository-canonical sanitized
logs mirrored into the existing Obsidian `BestPractice` directory, with Chinese topic names.

## Scope

This change applies only to the three dated progress reports currently under `reports/`.
`reports/guides/act/act_a_sanity.md` remains an experiment guide, and `reports/experiment_log.md`
remains the append-only structured experiment record required by repository instructions.
An unrelated LIBERO setup note already in the Vault remains untouched.

## Repository Rename Contract

Rename the three canonical reports exactly as follows:

| Existing file | New file |
|---|---|
| `reports/2026-07-16-progress.md` | `reports/daily/2026-07-16-LeRobot-ACT调试与ALOHA数据可视化.md` |
| `reports/2026-07-20-progress.md` | `reports/daily/2026-07-20-ACT-A-Sanity实验与模型源码调试.md` |
| `reports/2026-07-23-progress.md` | `reports/daily/2026-07-23-ACT离线评测与标定可视化.md` |

Each file's H1 uses the same date and Chinese topic wording as its filename, with normal
spaces for readability. Repository-relative Markdown links that reference an old filename
must be updated to the new target.

## Top Summary Contract

Immediately after each H1, add a `## 重要进展` section containing four to six concise
bullets. The bullets state the most important completed outcomes and any crucial limitation:

- 2026-07-16: training/debug entry discovery, VS Code debugging, ALOHA dataset inspection,
  synchronized image/joint visualization, and the absence of formal training metrics;
- 2026-07-20: A-Sanity launcher, offline/closed-loop evaluation design, dataset statistics,
  BF16/Accelerate findings, ACT internals, and the fact that the formal 20k run had not yet
  started on that date;
- 2026-07-23: 100-step versus 20k offline comparison, calibrated agentview correction,
  Panda/command/action visualization semantics, compact metrics tables, persistent
  publication rules, and the absence of a new formal closed-loop result.

Existing detailed sections remain unless they duplicate or contradict the summary. The
summary is not allowed to turn a planned, smoke, teacher-forced, or unmeasured result into a
formal experimental claim.

## Obsidian Replacement Contract

The repository versions are canonical. After rename and sanitization, copy the three exact
files to `<Obsidian Vault>/BestPractice/` using the same Chinese filenames.

Replace the two existing dated notes that cover the same 2026-07-16 and 2026-07-20 work:

- `LeRobot ACT调试与ALOHA数据可视化实践.md`;
- `lerobot-act-experiment-debug.md`.

Delete those old Vault filenames only after all three new files have been written and
verified byte-for-byte against the repository sources. Do not modify
`lerobot-libero-setup-and-smoke-test.md` or any other Vault note. The actual local Vault path
is operational input only and must not be written into repository reports, `AGENTS.md`, or
committed design documentation.

## Persistent Agent Rules

Extend the root `AGENTS.md` with a `Progress Log Writing And Vault Sync` section requiring:

1. dated progress filenames in `YYYY-MM-DD-<clear Chinese topic>.md` form;
2. an H1 containing the same date and clear Chinese topic;
3. a top `重要进展` section with four to six concrete bullets;
4. separation of completed work, measured results, limitations, and next steps;
5. repository-relative links and the existing publication-safety rules;
6. repository reports as canonical sources;
7. Vault synchronization only when explicitly requested, to the user-specified directory;
8. byte comparison before deleting or replacing an older Vault note;
9. no persistence of a personal Vault absolute path in tracked files.

## Privacy And Verification

Before Vault replacement:

- scan all three canonical files for absolute home paths, user or machine names, session or
  conversation IDs, personal Vault names, credentials, private endpoints, raw logs, and
  generated artifact contents;
- allow public repository and benchmark URLs plus selected aggregate metrics;
- verify every repository-relative Markdown link resolves, excluding intentional public
  URLs and placeholder paths;
- confirm the new Vault files are byte-identical to the canonical repository files;
- confirm the two superseded Vault files are gone and the unrelated setup note remains;
- inspect `git status` and ensure no Vault content or generated ML artifact is staged.

This task does not commit the renamed reports, upload, push, or otherwise modify a remote.
