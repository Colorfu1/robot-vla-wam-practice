# PushT Fixed-observation Action-mode Analysis Design

**Date:** 2026-07-29
**Status:** Spec review approved; awaiting user review

## Purpose

Add an exploratory offline analysis that asks whether normal Diffusion Policy DDPM sampling
produces multiple action-chunk modes under an identical observation history, and whether the
deterministic BCChunk prediction shows a signature consistent with regression averaging.

The analysis reuses the completed E1 artifacts. It does not load a policy, resample an action,
or execute an environment. Its conclusions complement E2:

- E1 action-mode analysis tests fixed-observation action-distribution multimodality;
- E2b tests whether stochastic sampling produces multiple successful closed-loop behaviors.

Neither experiment alone proves that BC failure is caused by MSE mode averaging.

## Scope and Claim Boundary

The input contains 10 frozen two-frame observation histories. Each candidate has:

- 20 E1a `initial_noise_only` action chunks;
- 20 E1b `full_sampling_seed` action chunks;
- 3 identical deterministic BCChunk predictions.

Every action chunk contains 8 two-dimensional normalized action targets. E1b is the primary
distribution, E1a is a negative-control/noise-floor distribution, and the three identical BC
calls collapse to one BC marker.

The strongest allowed positive result is:

> Under fixed observation histories, normal DDPM sampling produced exploratory evidence of
> multiple separated action-chunk modes. For some candidates, the deterministic BCChunk output
> also occupied a low-density, mean-like location consistent with a mode-averaging signature.

The analysis must not claim that:

- a cluster corresponds to a semantic strategy such as “left” or “right” without manual or
  geometric semantic labeling;
- BC is mathematically equal to the mean of the Diffusion distribution;
- a low-density BC action is invalid in closed loop;
- Diffusion Policy has universally or formally solved MSE mode averaging.

## Inputs and Validation

The default input root is:

```text
outputs/pusht_diffusion/pilot/e1_fixed_observation/
```

Required inputs are `action_chunks.parquet`, `diversity_metrics.parquet`, `metrics.json`, and
`run_manifest.json`. Before analysis, validate:

- E1 schema version is 2;
- there are exactly 10 unique candidates;
- each candidate has exactly 20 E1a and 20 E1b samples;
- every sample has steps `0..7` exactly once and finite normalized action coordinates;
- each candidate has exactly three BC calls and their maximum difference is at most `1e-6`;
- E1a and E1b sample ordinals are `0..19`;
- checkpoint, dataset revision, candidate key, and sampling protocol metadata are internally
  consistent.

Any violation fails before writing a final output directory.

## Action Representation and Distance

For candidate \(c\) and sample \(i\), reconstruct the normalized action chunk
\(A_{c,i}\in\mathbb{R}^{8\times2}\). The primary distance is the existing full-chunk RMS:

```text
d(A, B) = sqrt(mean((A - B) ** 2))
```

No per-candidate standardization is applied before computing this distance. This preserves the
meaning of the policy's normalized action coordinates and matches the existing E1 diversity
metric.

For model-based cross-checking only, flatten each chunk to 16 dimensions and fit PCA independently
within one candidate. Retain at most three components and never more components than needed to
explain 95% of sample variance. The selected dimension is also capped by sample count minus one
and numerical matrix rank.

Define total variance as the sum of the 16 population variances. If it is at most `1e-12`, mark
the sample `degenerate_unimodal=true`, select \(K=1\), and skip PCA and multi-component GMM
fitting. This deterministic path applies equally to E1a and E1b and cannot pass the multimodal
decision.

## Hybrid Exploratory Mode Decision

Analyze E1a and E1b independently for every candidate. E1b is labeled an exploratory multimodal
candidate only when all gates below pass.

### GMM model-selection gate

Fit diagonal-covariance Gaussian mixture models with \(K=1,2,3,4\) in the PCA representation.
Use 20 initializations, covariance regularization `1e-6`, and analysis seed `20260729`.

Require:

- the minimum-BIC model has \(K\ge2\);
- `BIC(K=1) - BIC(best K) >= 10`;
- every component contains at least 3 of the 20 samples by maximum-posterior assignment.

Run 200 deterministic 80% subsamples without replacement. At least 70% must select \(K\ge2\)
with BIC improvement at least 10. The subsample selection is candidate-keyed so row order does
not change results.

Canonicalize the 20 rows by sample ordinal before subsampling. Derive candidate and subsample
seeds through `numpy.random.SeedSequence` from analysis seed `20260729`, fixed stream IDs,
subsample ordinal, and uint32 words from the SHA-256 of the UTF-8 candidate ID. Python `hash()`
is forbidden. Every 80% subsample refits PCA from its own 16 samples using the same variance,
rank, and component-retention rules; it does not reuse the full-sample PCA basis. GMM
initialization receives a separately derived candidate/subsample seed.

### Distance-clustering agreement gate

Run complete-linkage agglomerative clustering from the precomputed full-chunk RMS distance
matrix, using the GMM-selected \(K\). Require:

- every distance cluster contains at least 3 samples;
- precomputed-distance silhouette is at least 0.25;
- adjusted Rand index between GMM and distance-cluster assignments is at least 0.5.

Also write the complete-linkage cluster-count curve at the sorted unique positive merge
distances from that candidate's complete-linkage hierarchy, plus zero and a deterministic
endpoint immediately above the maximum merge distance. This exact curve is sensitivity
evidence and is not used to select a favorable threshold.

### Negative-control validity gate

Apply the identical pipeline to E1a. Report every E1a candidate that passes the same consensus
criteria. If more than 20% of E1a candidates pass, mark the aggregate analysis
`negative_control_failed=true` and prohibit an aggregate positive multimodality conclusion.
Individual E1b measurements remain available as diagnostics.

These gates are pre-registered exploratory heuristics. With only 20 samples per observation,
they do not constitute a formal distribution-level hypothesis test.

## BC Mode-averaging Signature

For every E1b candidate, compute:

- the DP global mean chunk;
- one medoid per distance-cluster mode;
- BC distance to the DP mean;
- BC distance to every mode medoid;
- BC nearest-neighbor distance to the 20 DP samples;
- the percentile of that distance relative to DP leave-one-out nearest-neighbor distances.

An `averaging_consistent_signature` is true only when:

1. E1b passes the hybrid multimodal decision;
2. BC is closer to the DP global mean than to every mode medoid;
3. BC nearest-neighbor distance exceeds the 95th percentile of DP leave-one-out
   nearest-neighbor distances.

This flag means “mean-like and low-density relative to sampled DP chunks.” It does not mean the
BC action is invalid or that averaging caused a rollout failure.

## Outputs

Implement a read-only analyzer with an atomic output directory:

```text
outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis/
├── candidate_summary.parquet
├── mode_assignments.parquet
├── metrics.json
├── metrics.md
├── analysis_manifest.json
└── figures/
    └── <candidate_id>.png
```

`candidate_summary.parquet` stores every gate, cluster count, component size, BIC value,
subsample stability, silhouette, adjusted Rand index, E1a-control result, and BC signature
metric. `mode_assignments.parquet` stores candidate, protocol, sample ordinal, GMM label,
distance-cluster label, and PCA coordinates.

Each figure shows the E1b PCA samples colored by consensus distance cluster, mode medoids, DP
mean, and BC marker. A second panel shows the complete-linkage threshold sensitivity curve.
Figures are diagnostic projections; the mode decision uses the full-chunk distance plus the
specified GMM cross-check.

`analysis_manifest.json` records input file hashes, input run identity, analyzer schema,
analysis seed, all thresholds, dependency versions, and repository provenance. The analyzer
must never modify the parent E1 artifacts.

## Interface

Provide a repository launcher:

```bash
bash scripts/experiments/pusht/analyze_fixed_observation_action_modes.sh \
  --input-dir=outputs/pusht_diffusion/pilot/e1_fixed_observation \
  --output-dir=outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis \
  --overwrite
```

The Python entry point accepts the same input/output/overwrite contract. `--overwrite` replaces
only the exact analysis output directory through the repository's atomic-output mechanism.

## Aggregate Reporting

Lead with conclusions, then provide evidence. Aggregate metrics include:

- number and fraction of E1b candidates passing the hybrid multimodal decision;
- number and fraction of E1a candidates passing the same decision;
- `negative_control_failed`;
- number and fraction of E1b candidates with an averaging-consistent BC signature;
- per-candidate selected \(K\), mode sizes, stability, silhouette, and agreement;
- explicit `exploratory_only=true`.

The guide must combine E1 and E2 conclusions conservatively:

- a positive E1 result supports exploratory fixed-observation action-mode evidence;
- the current E2b result supports multiple successful closed-loop behavior modes;
- their combination is consistent with Diffusion Policy avoiding deterministic behavioral
  collapse;
- it still does not causally prove that BC failures are caused by MSE averaging.

## Error Handling and Tests

Unit tests cover:

- exact reconstruction of 20 chunks from row-wise Parquet records;
- rejection of missing, duplicate, non-finite, or inconsistent rows;
- deterministic PCA/GMM/subsample results under row reordering;
- synthetic unimodal data rejected by the hybrid gate;
- synthetic separated multimodal data accepted;
- clusters smaller than three rejected;
- E1a negative-control aggregate guard;
- BC mean-like/low-density signature true and false cases;
- atomic overwrite behavior and launcher argument forwarding.

An integration test runs the analyzer on a small synthetic E1 artifact tree. After
implementation, run the analyzer on the current completed E1 output, inspect all ten figures,
and report measured results as exploratory rather than formal.
