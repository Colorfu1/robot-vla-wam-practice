# PushT E2 Dual Closed-loop Sampling Protocol Design

**Date:** 2026-07-29
**Status:** Spec review approved; awaiting user review

## Purpose

E1 is complete as an offline sampler-mechanism experiment. It established that changing
the full DDPM reverse sampling stream produces reproducible action-chunk variation, while
changing only the initial noise produces nearly identical fixed-observation chunks. E1 does
not establish closed-loop validity or successful multimodality.

E2 now contains two separate closed-loop protocols:

1. **E2a — Initial noise sensitivity rollout**, a low-priority diagnostic that tests whether
   small differences caused by the initial diffusion sample \(x_T\) are amplified by closed-loop
   interaction.
2. **E2b — Full DDPM sampling rollout**, the primary experiment that measures outcome
   variance and successful trajectory diversity when the complete DDPM sampling stream varies.

The protocols run independently and write separate output directories. E2b does not wait for
E2a and is the only E2 input required by the final Pilot report.

## Non-negotiable Closed-loop Contract

Every E2 sample is one complete environment rollout:

```text
create a fresh environment
env.reset(seed=environment_seed)
policy.reset()
set the protocol and policy seed
repeatedly observe -> infer/reuse action chunk -> env.step(action)
stop only on success, termination, truncation, or the configured episode limit
close the environment
```

No offline action-chunk sample counts as an E2 sample. Replan-level action chunks may be saved
as diagnostic data only because they were generated during a complete closed-loop rollout.
Teacher-forced inputs, fixed observations, and unexecuted candidate chunks remain E1 concepts.

Each rollout records `num_action_samples=1` and `num_rollouts=1` at row level. Aggregate
manifests report the exact number of independently reset and executed rollouts.

`policy.reset()` resets only policy queues. Selecting a protocol or policy seed must not call
`numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed`, or otherwise mutate global
RNG state. Closed-loop sampling uses only the explicit per-replan generators defined below, so
rollout results cannot depend on execution order.

## Protocols

### E2a: Initial Noise Sensitivity Rollout

Purpose: determine whether the small \(x_T\)-only action difference observed in E1a grows into
measurable closed-loop trajectory or outcome differences.

Formal scale:

```text
environment seeds: 2000, 2001
policy seeds: 0, 1, 2, 3, 4
2 environments x 5 policy seeds = 10 complete Diffusion rollouts
```

All five policy seeds for one environment use unique initial-noise tensors but the same reverse
DDPM seed at a given replan index. E2a does not run a separate BCChunk baseline because its
question is sensitivity within one Diffusion policy.

E2a is diagnostic evidence. Its ten samples are not pooled with E2b for success estimation or
mode counting.

### E2b: Full DDPM Sampling Rollout

Purpose: measure the behavior distribution produced by normal stochastic DDPM sampling.

Formal scale:

```text
environment seeds: 2000..2019
policy seeds: 0..19
20 environments x 20 policy seeds = 400 complete Diffusion rollouts
```

E2b also runs one deterministic BCChunk closed-loop baseline for each of the twenty environment
seeds:

```text
20 BCChunk rollouts
+ 400 Diffusion rollouts
= 420 complete E2b rollouts
```

The BCChunk rollouts are a paired task-difficulty and deterministic-policy reference. They do
not participate in Diffusion success variance, trajectory diversity, or successful mode
clustering.

Environment seed is the independent statistical unit. The twenty policy seeds are repeated
stochastic samples within one initial environment state.

## Paired RNG Contract

Both protocols use the same stable initial-noise derivation:

```text
initial_noise_seed = SeedSequence([
    PROTOCOL_NAMESPACE,
    environment_seed,
    policy_seed,
    replan_index,
    INITIAL_NOISE_STREAM,
]).generate_state(1, uint32)[0]
```

E2a deliberately omits `policy_seed` from the reverse stream:

```text
e2a_reverse_seed = SeedSequence([
    PROTOCOL_NAMESPACE,
    environment_seed,
    replan_index,
    E2A_REVERSE_STREAM,
]).generate_state(1, uint32)[0]
```

E2b includes `policy_seed`:

```text
e2b_reverse_seed = SeedSequence([
    PROTOCOL_NAMESPACE,
    environment_seed,
    policy_seed,
    replan_index,
    E2B_REVERSE_STREAM,
]).generate_state(1, uint32)[0]
```

Use distinct, committed stream identifiers. The same `(environment_seed, policy_seed,
replan_index)` pair receives the same initial-noise seed in E2a and E2b. The protocols differ
only in reverse-stream derivation.

Only a real replan increments `replan_index`. Consuming queued actions does not consume a new
seed. A fresh initial-noise `torch.Generator` is created from the recorded initial seed for
every replan and is used only to construct that replan's \(x_T\) tensor. A separate fresh
reverse `torch.Generator` is created from the recorded reverse seed and passed to the DDPM
scheduler. Neither generator is reused, and neither operation mutates global RNG state.

Required invariants:

- identical protocol, environment seed, policy seed, replan index, checkpoint, and runtime
  reproduce the same initial and reverse seeds;
- E2a reverse seeds are identical across its five policy seeds for the same environment and
  replan index;
- E2b reverse seeds are unique across its twenty policy seeds for the same environment and
  replan index;
- paired E2a/E2b replans use identical initial-noise seeds;
- seed derivation does not include checkpoint identity, candidate ordering, scheduler name,
  output path, process ID, or Python `hash()`.

Every replan row records the initial-noise seed and tensor SHA-256, the reverse seed, and the
reverse generator's pre-sampling state SHA-256. These fields are required evidence for the RNG
pairing contract.

## Metrics

### E2a diagnostics

Report:

- complete-rollout success for every sample;
- per-environment success count out of five;
- all-rollout joint agent/block trajectory diversity;
- successful and failed trajectory diversity separately when at least two samples exist;
- first-replan normalized action-chunk diversity, computed only from chunks generated inside
  the closed-loop runs;
- whether success outcomes disagree across the five policy seeds.

The first-replan action metric is diagnostic context, not a replacement for closed-loop
trajectory metrics. E2a supports the statement that initial-noise sensitivity was or was not
amplified; it does not estimate the normal DDPM sampling distribution.

### E2b primary metrics

For environment \(e\), let \(s_{e,i}\) be the binary success of policy seed \(i\), and:

```text
p_e = mean_i(s_e,i)
within_success_variance_e = mean_i((s_e,i - p_e)^2) = p_e * (1 - p_e)
```

Report success variance in both forms needed to interpret stochastic outcomes:

- the twenty per-environment success rates \(p_e\);
- the twenty within-environment variances \(p_e(1-p_e)\);
- mean within-environment success variance;
- between-environment population variance of \(p_e\);
- fixed-policy-seed-0 single-sample success with a Wilson interval;
- all-sample success with an environment-cluster bootstrap interval;
- success-any@20 with a Wilson interval, explicitly labeled oracle coverage rather than
  deployment success.

Trajectory diversity uses the existing joint agent/block representation. Stop each path at
success or episode termination and resample it to 100 normalized-progress points. For two
resampled trajectories \(a\) and \(b\), construct pointwise deltas:

```text
agent_x / 512
agent_y / 512
block_x / 512
block_y / 512
wrapped_block_angle_difference / pi
```

Their pairwise distance is the root mean square over all 100 points and all five dimensions:

```text
distance(a, b) = sqrt(mean(delta(a, b) ** 2))
```

For each environment and outcome subset, compute the arithmetic mean of the upper-triangle
pairwise distances. Environments with fewer than two trajectories in that subset are excluded
and the eligible-environment count is reported. The primary cross-environment diversity
estimate is the equal-weighted arithmetic mean of those per-environment means. Its confidence
interval resamples eligible environment seeds with replacement and recomputes that mean.
Pairs are never pooled across environments.

Report environment-clustered pairwise trajectory diversity for:

- all E2b Diffusion rollouts;
- successful E2b Diffusion rollouts;
- failed E2b Diffusion rollouts.

Successful mode count is computed independently within each environment seed. Apply
complete-linkage clustering only to successful trajectories, using primary distance threshold
`0.05`. An environment with no success has zero successful modes. Also report sensitivity at
`0.025` and `0.075`.

Primary successful-mode outputs are:

- the twenty per-environment successful mode counts;
- mean and maximum successful mode count;
- number and fraction of environments with at least two successful modes;
- number and fraction of environments with at least one success.

Different failed trajectories never count as successful modes.

Wilson intervals and environment-cluster bootstrap intervals use 95% confidence. The bootstrap
uses 10,000 iterations and seed `20260727`. Confidence, iteration count, and bootstrap seed are
committed in `e2_rollout.json` and copied into each run manifest.

## Artifacts and Recording Policy

E2a and E2b write separate atomic output roots:

```text
outputs/pusht_diffusion/pilot/e2a_initial_noise_sensitivity/
outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling/
```

Each root contains:

```text
rollouts.parquet
replans.parquet
per_environment_metrics.parquet
metrics.json
metrics.md
mode_analysis.json
run_manifest.json
recording.rrd
videos/
```

Every rollout stores complete numeric agent, block, action-target, reward, success, and episode
length data. The Rerun recording contains paths for every rollout.

To bound E2b storage without changing execution semantics, embedded frames and MP4 video are
pre-registered for:

- every BCChunk baseline rollout;
- `policy_seed=0` for every E2b environment;
- `policy_seed=0` for both E2a environments.

All other policy seeds still execute full closed-loop rollouts and remain fully represented by
their numeric paths and outcomes. Video selection is seed-based and fixed before results are
known; it is not success-based cherry-picking.

The run manifest records the frame/video selection rule and the exact count of rollouts that
were actually reset and executed.

## CLI and Configuration

One evaluator supports an explicit protocol:

```text
--protocol=initial-noise-sensitivity
--protocol=full-ddpm-sampling
```

Formal defaults come from `configs/experiments/pusht_diffusion/e2_rollout.json`. A
`Closed-loop` run must use the exact registered environment and policy seed lists. Reduced
counts are allowed only with `--label=Smoke`; smoke outputs cannot be promoted to formal
evidence.

The launcher prints the resolved protocol, rollout counts, seed lists, output directory, and
video rule before execution. E2b is the default protocol because it is the primary experiment.

The final Pilot report reads E2b. E2a is linked as a diagnostic result but is never substituted
for missing E2b output.

## E1 Closure

Documentation marks E1 as `Offline / Formal / Stage complete` and records only its measured
conclusion:

- full reverse sampling affects action chunks reproducibly;
- observed action variation is small in absolute workspace units;
- E1 does not establish action validity, successful modes, or closed-loop benefit;
- the next evidence source is E2b.

No new E1 run is required by this redesign.

## Failure Handling

The evaluator fails before launching rollouts when:

- protocol names or formal seed lists disagree with committed configuration;
- E2b is not DDPM with 100 inference steps;
- loaded BCChunk and Diffusion checkpoints disagree on observation or action horizon;
- an environment/policy seed pair is duplicated;
- a formal output would contain a count other than 10 E2a Diffusion rollouts or
  400 E2b Diffusion plus 20 BCChunk rollouts.

Outputs remain atomic. An interrupted formal run is incomplete and cannot be consumed by the
report.

## Testing and Acceptance

Unit and integration tests must verify:

- stable E2a and E2b seed derivation;
- E2a reverse-seed sharing and E2b reverse-seed uniqueness;
- E2a/E2b initial-noise pairing;
- replan indices advance only after all eight queued actions are consumed;
- protocol names and condition IDs are present in rollout and replan rows;
- success variance formulas on synthetic grouped outcomes;
- successful mode count excludes failed trajectories;
- formal count validation rejects altered 2x5 and 20x20 matrices;
- smoke count overrides are accepted only with `label=Smoke`;
- launcher `--print-command` exposes the resolved protocol without creating outputs;
- report validation rejects E2a where E2b is required.

Closed-loop smoke acceptance requires at least one independently reset rollout for each
protocol and verifies that every saved sample has an environment trajectory, executed action
targets, and a terminal reason or configured timeout.

Formal acceptance requires:

```text
E2a: 10 complete Diffusion rollouts
E2b: 400 complete Diffusion rollouts + 20 complete BCChunk rollouts
```

The result may claim successful multimodality only when at least one E2b environment contains
two or more successful modes at the primary `0.05` trajectory threshold. Success-any@20 alone
does not support that claim.

## Considered Alternatives

1. **One combined 430-rollout output:** simpler aggregation, but E2a would block the primary
   experiment and a late failure would invalidate both protocols.
2. **Only E2b:** cheapest main answer, but loses the controlled closed-loop test of whether
   \(x_T\)-only differences amplify.
3. **Separate protocol runs:** selected. It preserves the requested low priority of E2a,
   lets E2b run independently, and keeps every metric tied to one unambiguous RNG contract.
