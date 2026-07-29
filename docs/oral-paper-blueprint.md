# Oral-level paper blueprint

Status: research direction, not a completed paper claim, 2026-07-29.

Working title:

> **More Simulation Is Not More Fidelity: Decision-Calibrated Test-Time
> Compute for Generative World Models**

The current evidence is a negative result with a useful consequence. It does not
yet constitute an oral-level paper. The paper becomes competitive only if the
phenomenon generalizes, its decision mechanism can be measured, and a simple
allocator beats the best static strategy at matched latency.

## Submission abstract skeleton

Do not fill the bracketed result claims until the corresponding gates pass:

> Generative world models improve embodied planning by scoring many candidate
> actions, but their test-time compute is usually scaled either by drawing more
> candidates or by simulating each candidate more deeply. We show that simulation
> depth is not automatically a fidelity order: it can reduce prediction error
> without improving the action selected, and its decision value changes with the
> state, proposal distribution, and candidate set. We decompose planning regret
> into proposal-coverage and evaluator-selection terms and introduce DCCA, a
> calibrated allocator that chooses whether to expand the candidate set, refine a
> candidate, or stop. On [tasks] and [world-model families], DCCA improves
> [success/IoU] by [effect] at matched latency over [strongest static/adaptive
> baseline], while using [compute reduction] relative to full refinement. These
> results recast test-time scaling for world models as decision-calibrated
> allocation rather than a scalar race toward more simulation.

## One-sentence perspective

A generative planner's test-time compute is not a scalar “fidelity” knob: compute
can either expand **proposal coverage** or reduce **selection error**, and a deeper
simulation is valuable only when its reduction in selection regret exceeds the
coverage lost by evaluating fewer actions.

This is the perspective the paper should try to establish. “Drift Flow Matching
for world models,” “adaptive NFE,” and “decision quality rather than pixels” are
already too incremental or already established.

## Evidence ledger

### What is established in this repository

1. The original residual DFM conversion was wrong. Endpoint normalization removes
   the catastrophic NFE2/4 failure while leaving NFE1 algebraically unchanged.
2. Corrected 100k training gives a reproducible finite NFE2 perceptual optimum for
   k1/grid25, but no family gives monotone NFE scaling through NFE8.
3. Increasing positive particles, source replay, warmup, learning rate, and
   300k--400k training do not restore stable monotone scaling. Long training often
   regresses after a 30k--60k rollout-best checkpoint.
4. The preregistered composition-amplification mechanism is rejected: median
   defect/degradation correlations are `-.4554/-.3004` for 2→4/4→8.
5. In the 20-state planning discovery, pure breadth `64×1` is descriptively best
   in all four selected families. Static top-fraction racing is slower and does
   not improve IoU under either ep100 or ep300.
6. The untouched 80-state confirmation completes 32/32 rows. It does not reproduce
   the discovery k1 NFE4 gain and supplies no reliable universal positive return
   to depth. Exact results are in the
   [confirmation snapshot](results/2026-07-29-unordered-fidelity-confirmation.md).
7. The immediate first-action reward audit is not a valid allocator target:
   candidate rewards are constant in 11/20 discovery states.

### What is not established

- Higher NFE is not proven to be universally harmful.
- The eight selected family-policy curves are not eight independent tasks.
- ep100 and ep300 should be called different proposal distributions until base
  policy coverage and no-world-model returns are measured; do not casually label
  one “stronger.”
- There is no validated predictor of beneficial refinement.
- There is no allocator that beats the best static frontier.
- There is no cross-task or second-world-model confirmation.

## Formal problem

At decision state `x`, draw `B` candidate action chunks
`a_1,...,a_B ~ pi(.|x)`. Let `U_x(a)` be the true continuation utility, and let
`U_hat_{x,d}(a)` be the utility estimated from a generative world model at
inference depth `d`. The planner selects

`i_hat = argmax_i U_hat_{x,d_i}(a_i)`.

Let `U*` be the best achievable utility in the action space and
`U*_B = max_i U_x(a_i)` the oracle utility within the sampled candidate set. The
decision regret decomposes exactly as

`R = U* - U_x(a_i_hat)`

`  = [U* - U*_B] + [U*_B - U_x(a_i_hat)]`

`  = coverage regret C(B|x) + selection regret S(B,{d_i}|x).`

Increasing breadth can reduce `C` while increasing depth can reduce `S` only if
that evaluator is decision-improving for this state, proposal distribution, and
candidate set. Under a wall-clock budget `tau`, the planner should solve

`min E[C + S] subject to measured_latency <= tau`,

not maximize NFE or minimize frame LPIPS.

### Why lower prediction error is insufficient

Even average utility-prediction error does not order decision fidelity. For two
candidates with true utilities `(1,0)`, scores `(0.9,-1.0)` choose correctly with
MSE `.505`, whereas scores `(.49,.51)` have lower MSE `.2601` but choose the
wrong candidate. Ranking is governed by error near the decision boundary, not
global reconstruction error.

This gives two theory targets:

1. an exact coverage/selection regret decomposition and a counterexample showing
   that improved reconstruction or score MSE need not improve selection;
2. an impossibility statement: without a bias/order condition linking a cheap
   evaluator to true utility, no cheap-score-only elimination rule can guarantee
   retention of the optimal candidate. This identifies precisely why standard
   multi-fidelity best-arm guarantees do not transfer automatically.

Do not promise a stronger theorem until its assumptions and proof are complete.

## Proposed method: Decision-Calibrated Compute Allocation

Working name: **DCCA**. The method is proposed; no current result validates it.

At each step, DCCA chooses one of three compute actions:

1. **Expand:** sample a new policy proposal and evaluate it at the cheapest depth.
2. **Refine:** re-evaluate one ambiguous candidate at another depth.
3. **Stop:** execute the current pessimistically selected candidate.

The allocator estimates two quantities per measured millisecond:

- marginal reduction in coverage regret from expansion;
- marginal reduction in selection regret from refinement.

It takes the action with the largest calibrated positive lower bound and stops
when neither bound is positive. Depth is treated as another biased evaluator,
not as ground truth; the final depth is never privileged by definition.

### Training target

Build a counterfactual decision dataset from cloned simulator states:

1. sample a common candidate set;
2. evaluate each candidate at every available depth with paired noise;
3. execute each candidate or a controlled continuation in the simulator;
4. record true continuation utility, coverage regret, selection regret, winner
   changes, and measured latency;
5. repeat at later contact-rich and recovery states, not only the first action.

Train the allocator directly on the marginal change in true decision regret.
Visual losses may be auxiliary features but never labels.

### Cheap evidence

Keep the signal set small and ablate each group:

- candidate-set-normalized score order statistics;
- action diversity and policy entropy;
- paired shallow-depth disagreement on a small probe subset;
- world-model ensemble or checkpoint disagreement;
- predicted motion/contact and action-future consistency.

Raw top-two margin is not a valid cross-allocation feature by itself: in the
discovery rows it rises while IoU falls, with descriptive correlation `-.799`.
Any score statistic must therefore be normalized for candidate count and proposal
distribution.

### Calibration and selection

- Fit expected marginal regret reduction and a lower quantile on a development
  split.
- Calibrate the lower bound on a disjoint split; report coverage and Brier/ECE,
  not only AUROC.
- Use a pessimistic final score when evaluator uncertainty is high.
- Evaluate leave-one-policy and leave-one-task transfer before claiming a general
  scheduler.

The method is deliberately simpler than a full reinforcement-learning router.
If a calibrated one-step allocator cannot beat static strategies, a more complex
controller is not justified.

## Closest-work boundary

| Prior line | What it already owns | Required distinction |
| --- | --- | --- |
| Value Equivalence, NeurIPS 2020 | Models should preserve planning values, not every transition detail. | The contribution is compute allocation and regret measurement, not the slogan “decisions over pixels.” |
| Value of Computation, UAI 2020 | Choose simulations by their expected impact on the eventual action. | DCCA must instantiate and validate this principle for biased generative evaluators; generic VOC is not novel. |
| Multi-Fidelity BAI, NeurIPS 2022/2024 | Cost-aware allocation among biased fidelities, with known bias structure and arm-specific optimal fidelity. | We test and remove the ordered/known-bias assumption, calibrating against environment return. |
| Consistency Models, ICML 2023; ParaDiGMS, NeurIPS 2023; AdaptiveDiffusion, NeurIPS 2024 | One/few-step generation and adaptive/parallel diffusion acceleration. | Their target is sample preservation or generative quality, not proposal coverage versus decision-selection regret. |
| Best-of-N pessimism, ICML 2025 | Coverage and reward-model error can make naive scaling non-monotone. | The decomposition is analogous but tested in closed-loop embodied planning with two compute axes. |
| Diffusion Policy, RSS 2023; RoboMonkey, CoRL 2025; MG-Select, ICLR 2026; TapSampling, ICML 2026 | Action diffusion, test-time action sampling, internal-confidence selection, and task-progress verification. | Action sampling/verification is a baseline; the question is when to buy a new action versus a deeper world-model evaluation. |
| SANTS, 2026 preprint | State-adaptive stopping along a WAM denoising trajectory. | Compare directly; DCCA is candidate-set-conditioned and allocates between breadth and depth using counterfactual regret. |
| GeoBoN, ACID, and compatibility verifiers, 2026 preprints | Geometry or action-consistency signals for selecting/gating rollouts. | These are candidate evidence that DCCA may use or compare against, not the method's novelty. |
| Progressive Seed Pruning, 2026 preprint | Start broad and prune diffusion seeds using intermediate scores. | Generic progressive pruning is occupied; the new claim requires untrusted depth and environment-calibrated regret. |

The full citation and venue audit is in
[`literature-review.md`](literature-review.md).

## Required experiments

### E0 — Close the remaining breadth question

**Question:** On the untouched 80 indices, does `64×NFE1` still beat or match
`32×NFE2`?

Run eight rows: four world-model families × ep100/ep300. Reuse the exact held-out
states and proposal RNG; add candidates 33--64 without changing candidates 1--32.
Report paired IoU/success, latency, proposal diversity, oracle candidate utility,
and coverage regret. This is the highest-information next experiment.

**Gate G0:** if held-out breadth is not competitive with depth, revise the
coverage-first premise before building DCCA.

### E1 — Counterfactual decision-fidelity dataset

**Question:** Does useful depth vary predictably across decisions?

| Axis | Minimum design |
| --- | --- |
| Tasks | Push-T plus released Robomimic Lift and Can |
| Decision states | first state plus stratified later contact, recovery, and near-success states |
| Policies | at least three proposal checkpoints/distributions per task |
| World models | at least three seeds/checkpoints; one variable-NFE DriftFlowWorld and one diffusion-world-model family |
| Candidate set | common 64 proposals, paired across all evaluator depths |
| Depth | NFE or denoising-depth grid including one-step and full/reference settings |
| Ground truth | cloned-state continuation return for every candidate, with common continuation protocol |
| Primary labels | coverage regret, selection regret, beneficial-refinement indicator |
| Secondary | rank correlation, winner flip, LPIPS/pose, latency, memory |

The official DriftWorld repository now exposes Push-T and Robomimic code, with
Robomimic `can_1view`, `lift_1view`, and `lift_2view` training configs. Verify
policy/planning assets before claiming executable cross-task support; the README
still lists Bridge-V2, RT-1, and Language Table code as forthcoming.

**Gate G1:** continue only if depth value changes sign or preferred depth changes
materially across at least two tasks and policy distributions, with candidate
utility measured directly. Otherwise the result is DFM-specific engineering.

### E2 — Can cheap evidence predict beneficial compute?

Train on task-policy blocks, evaluate on held-out blocks, and compare:

- constant-depth prior;
- raw score margin;
- state-only scheduler;
- candidate-set-normalized DCCA features;
- oracle marginal-regret label.

**Gate G2:** require useful calibration and transfer, not merely in-distribution
AUROC. Predeclare a minimum effect after a pilot; do not choose it post hoc.
If no cheap signal transfers, stop the allocator work.

### E3 — Fixed-latency planning benchmark

Compare at multiple measured latency budgets:

- base policy with no world-model reranking;
- DriftWorld/one-step Best-of-N;
- uniform breadth and uniform depth;
- the existing static coarse-to-fine racing rule;
- multi-fidelity successive elimination with empirically supplied bias bounds;
- progressive seed pruning;
- SANTS-style state-only depth scheduling;
- geometry/action-consistency selectors where compatible;
- inference-time pessimistic selection;
- DCCA;
- oracle coverage, oracle selection, and oracle allocator upper bounds.

Use task success or Push-T IoU as primary. Report regret decomposition, latency,
memory, calibration, and visual metrics as secondary.

**Gate G3:** DCCA must improve over the best static and strongest adaptive baseline
at matched measured latency, with a paired hierarchical 95% interval above zero
on at least two tasks and no material harm on the third.

### E4 — Model-family generality

Repeat the core fixed-latency comparison with a diffusion world model whose
fidelity axis is denoising depth. DriftWorld one-pass remains the breadth-only
reference.

**Gate G4:** if the result holds only for the current DFM conversion, narrow the
title and contribution to DriftFlowWorld. Do not claim a property of generative
world models.

## Statistical protocol

- Freeze hypotheses, splits, policies, checkpoints, and primary contrasts before
  new test runs.
- Use environment decision state as the statistical unit; frames and candidates
  are repeated measures, not independent samples.
- Use paired hierarchical bootstrap over task → policy/checkpoint → state.
- Report all family-policy-depth contrasts and multiplicity-adjusted secondary
  inference; avoid a pooled selected-checkpoint p-value.
- Match measured end-to-end latency, not only `B × NFE`. Include policy sampling,
  world-model generation, scoring, allocator overhead, and synchronization.
- Separate model-seed replication from multiple checkpoints of one training run.
- Report negative results and every preregistered row through immutable result
  snapshots and W&B.

## Oral-level contribution set

Only claim this set if all corresponding gates pass:

1. **Perspective:** generative planning compute has separable coverage and
   selection value; inference depth is not automatically decision fidelity.
2. **Theory:** regret decomposition plus a no-order counterexample/impossibility
   boundary for cheap elimination.
3. **Benchmark:** counterfactual candidate-utility data showing when additional
   world-model compute helps or hurts decisions across tasks, policies, seeds,
   and model families.
4. **Method:** a calibrated allocator that chooses breadth, refinement, or
   stopping and improves the matched-latency frontier.

The current repository has evidence for the motivation, not items 3--4.

## Paper narrative

1. Fast generative world models make many imagined actions possible, but current
   scaling discussions collapse compute into “more samples” or “more steps.”
2. Our controlled Push-T audit shows that latency scales predictably with depth
   while decision quality does not, and a discovery depth gain fails held-out
   replication.
3. The reason is structural: breadth controls coverage regret; depth can only
   affect selection regret, and learned depth need not be an ordered evaluator.
4. We measure these regrets directly with counterfactual continuations and show
   their state-, policy-, and candidate-set dependence.
5. DCCA spends the next millisecond where calibrated marginal decision value is
   positive.
6. Across tasks and world-model families, it improves the success-latency frontier
   and exposes when the correct action is to stop computing.

## Reviewer-facing failure modes

| Likely objection | Required answer |
| --- | --- |
| “This is just adaptive NFE.” | Direct SANTS comparison and breadth-versus-depth allocation conditioned on the candidate set. |
| “This is multi-fidelity bandits.” | State the missing fidelity-bias assumption, prove the boundary, and compare to MF-BAI with fair bias information. |
| “This is just Best-of-N.” | Measure separate coverage and selection regret and show the allocator decides between new proposals and evaluator refinement. |
| “Push-T is too small.” | Lift and Can plus a second world-model family are mandatory, not optional appendix experiments. |
| “The phenomenon is noise.” | Untouched splits, multiple model seeds, hierarchical statistics, and cross-policy/task replication. |
| “The gate learns simulator quirks.” | Leave-one-task/policy transfer, calibration, signal ablations, and no-harm results. |
| “Higher depth barely changes decisions.” | Then G1/G2 fail and the allocator paper stops; do not manufacture a method. |

## Immediate decision

Do not spend the next large compute block on more 300k/400k Push-T training or a
router trained from the sparse first-decision audit. Spend it first on E0 and the
smallest E1 data pilot. Those two experiments decide whether this direction has a
paper-level phenomenon before substantial method engineering.
