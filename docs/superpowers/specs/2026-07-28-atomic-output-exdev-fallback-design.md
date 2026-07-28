# Atomic Output EXDEV Fallback Design

**Date:** 2026-07-28
**Status:** Approved for implementation

## Goal

Allow PushT experiment results to publish successfully on filesystems that reject the
final directory `rename()` with `errno.EXDEV`, without changing normal-filesystem
behavior or exposing an output as complete before all of its files have been copied.

This change applies to every caller of `AtomicOutputDirectory`, including checkpoint
selection, rollout evaluation, fixed-observation evaluation, reports, and interface
checks. It does not change experiment computation, metrics, seeds, checkpoint ranking,
or output file formats.

## Observed Failure

The eight-GPU checkpoint selection completed all 320 rollouts and wrote every required
result beneath:

```text
outputs/pusht_diffusion/pilot/.checkpoint_selection.incomplete/
```

The final `Path.rename()` returned `OSError: [Errno 18] Invalid cross-device link` on
the HPVS/vePFS-backed output path. Source and parent reported the same device number,
and a subsequent recursive diff after manual `rsync` publication was empty. The
failure is therefore in directory publication, not evaluation or result generation.

## Selected Approach

Keep `rename()` as the primary publication path. Only when it raises `errno.EXDEV`,
fall back to a recursive copy:

1. retain the temporary directory's `.incomplete` marker;
2. create the final directory and its own `.incomplete` marker before copying any
   payload;
3. copy the complete temporary tree into the already marked final directory;
4. if copying succeeds, remove the temporary tree;
5. remove `<final>/.incomplete` last;
6. mark the context as committed.

The marker is the completion boundary. During a fallback copy, the final path may be
visible, but it remains explicitly incomplete until every copy and cleanup step has
succeeded. The implementation must not depend on recursive traversal order to publish
the marker before payload files.

## Alternatives Rejected

- **Always copy:** works on HPVS but adds unnecessary I/O and removes atomic directory
  publication on filesystems where rename works.
- **Write directly to the final directory:** avoids a second copy but changes the
  existing staging-directory contract for every run.
- **Symlink publication:** does not address the failing operation, introduces
  path-lifetime coupling, and was not used by the existing implementation.

## Failure Semantics

The temporary tree keeps `.incomplete` until publication moves or copies it. A
non-`EXDEV` rename failure is re-raised and leaves the temporary tree marked
incomplete.

If creating the destination marker fails, the implementation removes the still-empty
final directory when possible and propagates the error. If fallback copying fails, both
the source staging tree and any partially copied final tree retain `.incomplete`. The
code must not delete either tree or report success. If temporary-tree cleanup or
final-marker removal fails, the final tree retains `.incomplete` and the exception
propagates.

Existing overwrite behavior remains unchanged: a later run requires `--overwrite` to
replace an existing final directory.

## Verification

Automated tests cover:

1. the existing successful rename path;
2. an injected `errno.EXDEV` from `Path.rename()`, followed by successful recursive
   copy, nested-file preservation, staging cleanup, and final marker removal;
3. a non-`EXDEV` rename error that propagates while retaining the staging marker;
4. an injected copy failure that propagates while retaining incomplete markers and
   recoverable files;
5. an injected staging-cleanup failure that leaves both trees marked incomplete;
6. an injected final-marker-removal failure that leaves the copied final tree marked
   incomplete;
7. existing PushT output and launcher tests.

The development-machine verification uses a small throwaway output directory on the
same HPVS mount. It must publish through the fallback without running another model
evaluation.
