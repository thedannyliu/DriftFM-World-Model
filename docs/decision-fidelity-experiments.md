# Mandatory experiments for decision-calibrated compute

Status: staged, falsifiable experiment specification, 2026-07-29. No job is
authorized or submitted by this document.

This plan tests the conjectures and conditional assumptions in
[`decision-fidelity-theory.md`](decision-fidelity-theory.md). Proof failure,
negative results, and a static strategy winning are all valid outcomes. Runs
must stop at a failed gate; downstream experiments are not run merely to rescue
the desired narrative.

## Global protocol

- Default cluster: PACE-Phoenix.
- GPU jobs: account `gts-agarg35`, QOS `embers`; never use `inferno` without
  explicit approval.
- Canonical environment:
  `/storage/scratch1/9/eliu354/driftflowworld/envs/pace-cu128-py312-v1`.
- Proposed W&B project: `danny010324/driftworld-decision-fidelity`, private.
- Resume must reuse the same W&B run ID and preserve optimizer/RNG state where
  training is involved.
- Every formal row records task, simulator-state IDs, policy and seed, candidate
  RNG, world-model checkpoint and seed, manifest, code commit, GPU, output path,
  latency, and W&B run ID.
- Environment decision state is the statistical unit. Candidates, frames,
  depths, and continuation repeats are paired repeated measurements.
- Primary intervals use a paired hierarchical bootstrap over
  task → policy/model block → state. Report every preregistered contrast and
  multiplicity-adjust secondary analyses.
- Match measured end-to-end latency, not only \(B\times\mathrm{NFE}\).

## Sequential gate overview

| Stage | Ledger IDs | Question | Run only if | Stop condition |
| --- | --- | --- | --- | --- |
| 1 | N0 | Is there meaningful headroom over the best static breadth/depth frontier? | Existing held-out fixed-depth rows are complete. | The state-wise oracle switch cannot improve the best static arm by a practically meaningful amount. |
| 2 | N1 | Can true continuation utility provide a non-degenerate supervision target? | N0 leaves allocator headroom. | Candidate utility is mostly constant/noisy, or refinement almost never changes regret. |
| 3 | N2 + N5a | Is untrusted depth real beyond one Push-T conversion? | N1 produces reliable labels. | One static allocation dominates across tasks/models, or the effect is DFM-only. |
| 4 | N3 + N4 + N5b | Can calibrated evidence choose compute and beat static/adaptive baselines? | N2/N5a establish heterogeneous value. | Transfer calibration fails or DCCA does not improve the measured frontier. |

The first two stages are the immediate must-runs. Stages 3–4 are mandatory for a
general top-conference claim but should not start after an earlier kill decision.

## N0 — Untouched breadth-versus-depth confirmation

### Question

At approximately 64 model evaluations, is compute better spent on 64 one-step
proposals or on refining the first 32 proposals to NFE2, and is there enough
state-wise variation for an allocator to improve either static arm?

### Design

- Task: Push-T untouched indices 20–99.
- Blocks: four frozen world-model families × policies ep100/ep300.
- New arm: `64x1`, generated as a nested extension of the existing first
  32 proposals using the same candidate RNG.
- Reused arm: the complete held-out `32x2` rows; do not rerun them unless a
  provenance/hash check fails.
- Preserve the first 32 action hashes so coverage from candidates 33–64 is
  identifiable.
- Measure end-to-end H100 latency for both arms and include policy sampling,
  scoring, synchronization, and environment overhead.

### Primary outputs

1. Paired episode IoU and success difference, `64x1 - 32x2`.
2. State-wise oracle-switch upper bound:

   \[
   G_{\mathrm{switch}}
   =\mathbb E[\max(Y_{64x1},Y_{32x2})]
    -\max(\mathbb E[Y_{64x1}],\mathbb E[Y_{32x2}]).
   \]

3. Pareto frontier under measured latency.
4. Proposal diversity and the incremental value of candidates 33–64.

### Decision

- **Continue:** the upper confidence bound on \(G_{\mathrm{switch}}\) exceeds
  `0.01` normalized IoU and neither static arm is uniformly dominant across
  preregistered state strata.
- **Stop allocator work:** even the hindsight state-wise switch has less than
  `0.01` plausible improvement over the best static arm. A learned scheduler
  cannot create material headroom that its oracle lacks.
- If `64x1` wins consistently, retain it as the strongest baseline; do not call
  this evidence for DCCA.

This experiment tests whether an adaptive paper is necessary, not whether the
current method is good.

## N1 — Later-state counterfactual candidate-utility audit

### Question

Do cloned simulator continuations provide a stable, non-degenerate label for
coverage regret, selection regret, and the value of expansion versus refinement?

### Pilot design

- Start with Push-T and both ep100/ep300 proposal policies.
- Sample 100 unique decision states per policy, stratified equally across:
  early free motion, pre-contact, active contact, and recovery/failure.
- From each cloned state, draw one master pool of 64 candidate action chunks.
- Evaluate every candidate with the same paired noise at NFE `1/2/4/8`.
- Execute every candidate from the cloned state under one frozen continuation
  policy and horizon.
- Use common random numbers across candidate/depth comparisons. If the
  continuation is stochastic, run a small repeatability preflight and increase
  repeats until the normalized utility standard error is at most `0.02`, or
  declare the target too noisy.
- Freeze the continuation horizon and utility definition before inspecting
  depth effects. Immediate first-action reward is prohibited because it was
  constant in 11/20 prior states.

### Required measurements

For every state, policy, candidate count \(B\in\{8,16,32,64\}\), and depth:

- true continuation utility for each candidate;
- pool-relative coverage regret \(C_{64}(B)\);
- world-model selection regret \(S(B,d)\);
- score MSE, rank correlation, top-\(k\) recall, and best-candidate retention;
- depth-induced winner changes and whether they improve true utility;
- oracle marginal regret reduction per millisecond for Expand and Refine;
- empirical interval coverage required by Theorems 4–5.

### Hypotheses and falsifiers

| Hypothesis | Evidence for | Falsifier |
| --- | --- | --- |
| Candidate utility is learnable | At least 90% of states have utility range larger than twice the largest per-candidate 95% repeatability half-width in that state. | Mostly constant or unstable continuation values. |
| Depth is not a trusted order | Selection-regret changes have both signs across preregistered blocks/states, or the empirically valid error radii are non-monotone. | One depth weakly dominates and has nested valid error bounds. |
| Allocation is state-dependent | Oracle Expand and Refine are each optimal on at least 10% of usable states. | One compute action is optimal on more than 95% of states, or oracle switching gain is negligible. |
| Prediction quality is insufficient | Reconstruction/score MSE fails to order selection regret near candidate decision boundaries. | A simple validated error metric completely orders the decisions; use that simpler method. |

Failure is publishable evidence about the limits of the proposed mechanism, but
it kills training an allocator from this dataset.

## N2 + N5a — Cross-task and cross-model falsification

### Question

Is evaluator-depth value genuinely task-, policy-, and candidate-dependent, or
is the Push-T result an artifact of one DFM conversion?

### Design

- Tasks: Push-T, official Robomimic Lift, and official Robomimic Can.
- Proposal distributions: at least two preregistered policy checkpoints or
  sampling temperatures per task.
- World-model replication: at least three independent model seeds for the
  selected DFM configuration.
- Model-family replication: one separately trained diffusion world model whose
  compute axis is denoising depth. Multiple checkpoints from one training run do
  not count as independent seeds.
- Use the N1 master-pool and continuation protocol, scaled to at least 100 usable
  states per task-policy-model block after the pilot establishes variance.
- Lock task-normalized utility before pooled analysis.

### Primary tests

1. Paired selection-regret contrasts for NFE/steps `2/4/8` versus `1`.
2. Hierarchical depth × task, depth × policy, and depth × model-family
   interactions.
3. Fraction of states on which each static allocation is oracle-best.
4. Best-static versus state-wise-oracle allocation gap at matched latency.

### Decision

- **Continue general allocator paper:** no single static allocation dominates,
  the oracle allocation gap is materially positive on at least two tasks, and
  heterogeneous depth value replicates in the second model family.
- **Narrow to DFM:** heterogeneous value holds for DFM but not the diffusion
  world model.
- **Stop:** one static allocation is sufficient across blocks or the oracle
  allocation gap is negligible.

Merely obtaining a significant interaction is not enough; it must create
actionable regret headroom.

`N5a` is the second-model counterfactual audit required before training a
general allocator. `N5b` is the later replication of the final N4 allocator in
that model family. Passing one does not imply passing the other.

## N3 — Shift-held-out calibration

### Question

Can information available before execution predict the marginal true-regret
reduction of Expand and Refine?

### Design

- Targets come only from N1/N2 counterfactual continuation returns.
- Candidate evidence is restricted to:
  candidate-count-normalized score statistics, action diversity, paired
  shallow-depth disagreement, model/checkpoint disagreement, predicted
  motion/contact, and action-future or geometry consistency.
- Baselines: constant action prior, raw top-two margin, state-only
  SANTS-style depth choice, MF-BAI with empirically supplied bias bounds,
  GeoBoN/action-consistency gates, and an oracle label.
- Splits are leave-one-policy and leave-one-task blocks; random state splits
  alone are insufficient.
- Predict expected marginal regret reduction per measured millisecond and a
  lower quantile used by Theorem 5's safe switch.

### Primary metrics

- calibration coverage of the lower bound at target error `alpha=0.05`;
- Brier score and ECE for positive-value decisions;
- realized regret reduction of actions with positive lower bound;
- false-safe-switch rate;
- AUROC only as a secondary ranking metric.

### Decision

- **Continue:** the lower bound meets its held-out coverage target after
  multiplicity accounting and selects a positive-value compute action often
  enough to improve expected regret.
- **Stop:** calibration or realized value fails on a held-out task/policy.

A high in-distribution AUROC does not pass this gate.

## N4 — Fixed-latency allocator benchmark

### Question

Does a sequential Expand/Refine/Stop allocator improve embodied decisions, not
just its offline labels?

### Design

Evaluate at preregistered measured latency budgets spanning the observed
approximately `0.8/1.5/2.9/5.6 s` Push-T range. Include:

- base policy without world-model reranking;
- one-step Best-of-N;
- static `64x1`, `32x2`, uniform depth, and the best task-specific static point;
- the existing fixed-fraction racing rule;
- MF-BAI with favorable empirical bias information;
- progressive seed pruning;
- SANTS-style state-only scheduling;
- GeoBoN/action-consistency and pessimistic selectors where compatible;
- DCCA with and without Theorem 5's safe switch;
- oracle coverage, oracle selection, and oracle allocator upper bounds.

Primary evaluation uses untouched episodes, at least 200 per task after a power
analysis based on N2. No DCCA feature or threshold may be changed from final test
episodes.

### Primary claim gate

DCCA must:

1. improve paired task utility over the best static and strongest adaptive
   baseline with a hierarchical 95% interval above zero on at least two tasks;
2. cause no task to lose more than a preregistered one percentage point in
   success;
3. satisfy the measured latency budget including allocator overhead; and
4. retain calibration close to the N3 target.

If it fails, report that calibrated allocation did not beat the static frontier.
Do not respond by adding a more complex router without a new hypothesis.

After a successful primary N4 result, N5b repeats the locked winning allocator
and its strongest baselines with the diffusion world model. Failure narrows the
claim to DFM even if the three DFM tasks pass.

## Theorem-to-experiment traceability

| Mathematical object | Required evidence |
| --- | --- |
| Theorem 1 decomposition | N1 directly records \(C_{64}(B)\) and \(S(B,d)\). |
| Proposition 1.2 maximization failure | N1 measures whether added candidates lower coverage yet increase total regret. |
| Proposition 2 MSE/ranking counterexample | N1 tests whether this theoretical possibility occurs empirically near decision boundaries. |
| Theorem 3 no-free-elimination boundary | N1/N2 estimate violations of ordered bias; N4 compares methods given and not given favorable bias information. |
| Theorem 4 uniform-error condition | N1/N2 report empirical simultaneous coverage and whether radii shrink with depth. |
| Theorem 5 safe switch | N3 tests shift-held-out coverage; N4 reports realized no-harm violations. |
| Proposition 6 one-step value of computation | N1 supplies oracle marginal values; N4 tests the greedy approximation rather than claiming global optimality. |
| Conjectures C1–C5 | N0–N5 may confirm or kill them; no conjecture is promoted from Push-T point curves alone. |

## Result-recording rule

Each completed stage receives a dated immutable file under `docs/results/` with:

- the frozen question and gate;
- exact code and manifest commits;
- task/state/candidate provenance;
- GPU and measured latency;
- checkpoint and W&B IDs;
- all preregistered primary contrasts;
- proof assumptions supported or violated;
- a `continue`, `narrow`, or `stop` decision.

Negative and proof-invalidating results remain in the ledger. They are not
silently replaced by a revised hypothesis.
