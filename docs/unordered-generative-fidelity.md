# Unordered generative fidelity for decision-time planning

Status: Push-T discovery and held-out confirmation complete, 2026-07-29. The
evidence rejects using NFE as a guaranteed decision-fidelity ladder, but does not
yet establish a cross-task phenomenon or a useful allocator.

## One-sentence research question

When a learned generative world model offers several inference depths, should those
depths be treated as a conventional low-to-high fidelity ladder, or as
instance-dependent and potentially unordered estimators whose compute must be
allocated using decision evidence?

NFE means **number of function evaluations** of the learned transport model. It is an
inference-compute setting, not a quality metric. MSE, LPIPS, block-pose error, action
ranking, Push-T IoU, success rate, and latency are metrics.

## Evidence that motivated the question

The original residual parameterization had a concrete implementation failure:
NFE=2/4 diverged while NFE=1 remained useful. Endpoint normalization fixed most of
that failure without changing NFE=1. The corrected 100k experiments then supplied a
more interesting signal:

- several families improve prediction MSE or block pose from NFE=1 to 2/4/8;
- the best depth is not consistently the largest depth;
- locked action-risk audits found a large off-manifold penalty at NFE=4/8;
- local composition-defect correlation with downstream degradation was negative,
  falsifying the proposed simple composition-amplification mechanism;
- long training often degraded rollout quality after an early 30k--60k optimum.

These results reject both naive statements: “more NFE always helps” and “more NFE
always hurts.” They instead motivate a decision-level audit with real environment
outcomes.

## Prior work and novelty boundary

| Line of work | What is already established | Consequence here |
| --- | --- | --- |
| [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) | Drifting moves iterative distribution improvement into training and admits one-step inference. | A one-step drifting generator is not novel. |
| [Drift Flow Matching](https://arxiv.org/abs/2605.17244) | Learns arbitrary-time transport and reports improved image generation and robotic-policy success with additional NFE. Its robotics table explicitly evaluates a “Drift Flow Matching Policy.” | DFM, DFM policy, and a generic “more NFE helps” claim are prior art. Our result must explain when NFE does *not* form a reliable ladder in conditional decision models. |
| [Implicit Drifting Policy](https://arxiv.org/abs/2606.01098) and [Drifting Field Policy](https://arxiv.org/abs/2605.07727) | One-step drifting-style action policies using conditional expert geometry or Wasserstein policy improvement. | “Drift policy” is crowded and is not the current contribution. |
| [DriftWorld](https://arxiv.org/abs/2607.15065) | GPC-RANK uses a fast action-conditioned video model to rank policy proposals. | Candidate ranking with a video world model is the baseline, not the claim. |
| [Value Equivalence](https://arxiv.org/abs/2011.03506) and [Policy-Aware Model Learning](https://arxiv.org/abs/2003.00030) | A model need not reconstruct every observation correctly; it should preserve the planner’s values or gradients. | “Decision quality over pixels” is an old principle. We use it as evaluation discipline, not novelty. |
| [Multi-Fidelity Best-Arm Identification](https://proceedings.neurips.cc/paper_files/paper/2022/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html) and its [optimal extension](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html) | Allocate queries among simulators with different costs and bounded, known fidelity bias. | Cheap-to-expensive candidate elimination is not new by itself. The open issue is that learned NFE may violate the assumed fidelity ordering and bias bounds. |
| [SANTS](https://arxiv.org/abs/2605.27947) | Selects a state-dependent stopping point along a WAM video denoising trajectory using downstream action quality. | State-adaptive depth is also prior art. We must test candidate-specific ranking and calibrated fixed-budget allocation, not merely learn another global stopping head. |
| [GeoBoN / gated WAM test-time scaling](https://arxiv.org/abs/2607.17454) | Ranks independent WAM rollouts and gates extra sampling using geometry/action-future consistency. | Adaptive test-time compute broadly is not novel. We compare against breadth and state-level gating concepts. |

The remaining defensible object is therefore narrow:

> Learned inference depth is not automatically a fidelity order. For action
> selection, fidelity may be candidate-, state-, and checkpoint-dependent; treating
> NFE as ordered can waste compute or select a worse action. A useful allocator must
> first estimate whether refinement is trustworthy for the current candidate set.

This can become a strong paper only if it generalizes beyond Push-T and produces a
simple predictive law or calibrated allocator. A Push-T-only breadth/depth benchmark
is useful engineering evidence, not a top-venue contribution.

## Falsifiable hypotheses

### H1 — NFE is not an ordered decision fidelity

For the same observation, policy proposals, injected noise, and checkpoint, increasing
NFE frequently changes action ranking non-monotonically. Higher-NFE scores do not
consistently approach ground-truth candidate outcomes.

**Pass:** at least two model families and both policies show material winner flips or
non-monotonic rank correlation, with bootstrap confidence excluding a negligible
effect. **Fail:** ranking agreement and task outcome improve monotonically with NFE
for nearly every state and family.

### H2 — Useful depth is instance-specific and predictable

Cheap first-pass signals such as top-two score margin, cross-NFE disagreement,
off-manifold distance, or candidate dispersion predict whether refinement changes the
winning action beneficially.

**Pass:** a predictor evaluated on held-out seeds improves calibration/AUROC over
constant-depth and state-only baselines, and transfers to a second policy or task.
**Fail:** no cheap signal predicts beneficial refinement outside its training split.

### H3 — Calibrated racing beats both breadth and depth

At equal measured latency and nominal model evaluations, refine-only-when-useful
candidate allocation improves environment IoU/success over fixed NFE, pure proposal
breadth, and standard coarse-to-fine elimination.

**Pass:** paired held-out improvement over the best fixed strategy with a 95% bootstrap
interval above zero, reproduced across policies and at least two model families.
**Fail:** the best static breadth/depth point matches or beats the allocator.

### H4 — The failure is specific to conditional world-model ranking

DFM policy generation may exhibit monotonic NFE scaling while an action-conditioned
DFM world model does not. This would isolate the problem to conditional future
ranking/off-manifold composition rather than DFM universally.

**Pass:** a matched policy-versus-world-model contrast reproduces the divergence across
at least two tasks. **Blocked in the current company wave:** the repository contains
the Push-T Diffusion Policy and world-model assets, not a compatible official DFM
policy training/evaluation release.

## Current four-node experiment

Manifest: [`manifests/unordered-fidelity.yaml`](manifests/unordered-fidelity.yaml).
Executable row plan:
[`../company/unordered_fidelity_plan.tsv`](../company/unordered_fidelity_plan.tsv).

All rows use frozen checkpoints and 20 fixed Push-T test seeds by default. Each row is
sharded across four H100s. The environment seed, policy proposal RNG, and candidate
count are fixed within each comparison. Results write an atomic shared marker under
`/group-volume`; interrupted rows retain completed shards and reuse them on restart.
Only the final compact summary is logged to W&B.

The policy comparison uses the monolithic
`diffusion_policy_v1/ckpt_save/ckpt-ep{100,300}.pth` checkpoints with the
non-official loader (`model` plus `ema` state) and a shared proposal RNG base seed of
5. This isolates policy training maturity from checkpoint format and sampling-seed
changes. The separate `diffusion_policy_gpc/` component directory is not used in this
comparison.

| Node | Question | Comparisons | Why it is identifiable |
| --- | --- | --- | --- |
| A | Does more compute help with candidate count fixed? | 32 candidates at NFE 1/2/4/8, plus real-simulator first-decision reward for every candidate | Changes only inference depth and directly measures ranking regret. |
| B | Is depth better than breadth? | 64×1, 32×2, 16×4, 8×8 | Matches nominal model evaluations at 64. |
| C | Does candidate racing help? | fixed 16×2 versus 16 coarse candidates with top 1/2, 1/4, or 1/8 refined | Each method uses 32 nominal evaluations but allocates them differently. |
| D | Does the conclusion survive policy shift? | ep300 repeats representative breadth/depth/racing settings | Reuses the same world checkpoints and changes the proposal policy. |

The four representative world models are k1-grid25 validation-best seed 2, k32
latest seed 2, joint-k16 latest seed 2, and deep-base rollout-best seed 1. This is a
deliberate discovery set spanning different observed NFE behavior. It does not replace
later multi-seed and cross-task confirmation.

Primary metric: maximum Push-T IoU over an episode, paired by environment test seed.
Node A additionally measures first-decision candidate-chunk ground-truth reward,
predicted-versus-true rank correlation, oracle-selection fraction, and selection
regret. Secondary metrics are success fraction, wall-clock planning latency, top-two
margin, coarse-to-final winner flips, and per-trial outcomes. LPIPS alone never
decides a planning claim.

## Interim result: Node A fixed candidates

Node A has completed 15/16 fixed-32-candidate rows. Exact W&B IDs, paired intervals,
raw audit findings, and provenance are in the
[2026-07-29 Node A snapshot](results/2026-07-29-unordered-fidelity-node-a-partial.md).
The k1-grid25 NFE8 row is missing.

Full-episode NFE effects are family-dependent:

- k1 improves by `+.07473/+.09600` IoU at NFE2/4 versus NFE1, and both paired
  20-trial intervals exclude zero;
- k32 trends from `-.01564` at NFE2 to `+.02309` at NFE8, with all intervals
  crossing zero;
- joint peaks descriptively at NFE2 and falls below NFE1 at NFE8;
- deep peaks descriptively at NFE4 and rebounds downward at NFE8.

This is compatible with unordered, checkpoint-dependent useful depth, but 20 trials
are underpowered outside k1. It also does not yet provide the mechanism originally
assigned to Node A. The first-decision simulator target has zero candidate variance
in 11/20 states, so rank correlation is defined on only nine states. Joint/deep
selection regret is identical across all NFE even though full-episode IoU changes.
The immediate first-action-chunk audit therefore cannot explain the episode result
and must not be used to train a gate.

The fixed-candidate task result and Node B agree on the practical compute decision:
high NFE can help a selected family, but 64x1 breadth matches or exceeds the observed
high-depth IoU at much lower latency. The next wave confirms full-episode NFE effects
on untouched indices 20--99 and across ep100/ep300; a later mechanism study must use
common later-episode states or longer-horizon counterfactual return.

## Interim result: Node C candidate racing

Node C has completed 15/16 equal-nominal-budget rows. The immutable result, exact W&B
IDs, paired intervals, and provenance are recorded in the
[2026-07-29 Node C snapshot](results/2026-07-29-unordered-fidelity-node-c-partial.md).
The joint-k16 top-eighth/NFE8 row is missing, so this remains a partial result.

The tested static coarse-to-fine racing baseline does not pass the H3 gate:

- refining half of 16 candidates to NFE 2 changes mean IoU by
  `0.00000/+0.00289/-0.00065/+0.00007` for
  k1/k32/joint/deep, but costs 6.2--6.6% more latency than fixed 16x2;
- top-quarter/NFE4 and top-eighth/NFE8 usually reduce paired IoU and cost about
  14% and 40--42% more latency; k32 is significantly worse for both;
- the coarse-to-final winner changes in zero of 20 trials for most rows and one of 20
  at most, so refinement rarely changes the selected action.

The bounded conclusion is that fixed-initial-breadth, deeper-on-fewer-candidates
racing is not useful here. This does not establish that adaptive compute is useless:
Node B must determine whether NFE1 is better used to evaluate more candidates, while
Node A must measure whether depth preserves ground-truth candidate ranking.

## Interim result: Node B breadth-depth frontier

Node B has completed 15/16 equal-nominal-budget rows. Exact per-row W&B IDs, paired
intervals, latency, margins, and provenance are in the
[2026-07-29 Node B snapshot](results/2026-07-29-unordered-fidelity-node-b-partial.md).
The joint-k16 16-candidate/NFE4 row is missing.

At 64 nominal model evaluations, proposal breadth is the descriptive winner:

- 64x1 has the highest mean IoU in all four selected model families; its
  family-average IoU is `.77586`, versus `.75077` for 32x2 and `.70148` for 8x8;
- 64x1-minus-32x2 is positive in all four families
  (`+.00819/+.02394/+.02888/+.03934`), but every family-level 20-trial interval
  still crosses zero;
- only 64x1 and 32x2 remain on the latency/IoU Pareto frontier. The quality gain of
  64x1 costs about 5.8--6.3% measured latency, while 16x4 and 8x8 are dominated.

The result also invalidates raw top-two score margin as a cross-allocation confidence
signal. Mean margin rises from `.1091` at 64x1 to `.2580` at 8x8 while mean IoU
falls by `.0744`; across all 15 rows the descriptive correlation is `-.799`. This is
consistent with candidate-count-dependent order statistics rather than trustworthy
confidence.

Together with Nodes C/D, Node B rejects the tested static “refine fewer candidates
more deeply” strategy. The remaining scientific question is narrower: whether Node A
shows candidate-conditional ranking errors that explain why breadth wins and support
a calibrated, allocation-aware signal. Without that mechanism and cross-task
replication, the Node B result is a strong systems baseline rather than a novel
paper claim.

## Interim result: Node D policy shift

Node D has completed 15/16 ep300 rows. Exact per-row W&B IDs, paired intervals, and
policy-interaction estimates are in the
[2026-07-29 Node D snapshot](results/2026-07-29-unordered-fidelity-node-d-partial.md).
The deep-base 32-candidate/NFE1 row is missing.

The ep300 policy does not rescue static racing:

- racing-minus-fixed paired ΔIoU is
  `+.00048/-.01395/-.01538/-.00828` for k1/k32/joint/deep; every interval includes
  zero while latency increases by 13.2--15.3%;
- 8-candidate/NFE4 is worse on average in all families and conclusively worse for
  k32 (`-.06891`, 95% CI `[-.14413,-.00293]`);
- all seven available ep100-to-ep300 allocation-interaction intervals include zero.

Winner flips rise to 5--15% under ep300 but do not improve mean IoU. The current
evidence therefore rejects this static racing rule across both tested policies; it
does not yet reject an allocator driven by a separately validated decision signal.

## Held-out result: fixed-candidate confirmation

The preregistered confirmation completed all 32 rows: four selected world-model
families × two proposal policies × NFE `1/2/4/8`, with 80 untouched Push-T
indices per row and fixed 32-candidate sets. Exact values and provenance are in the
[2026-07-29 confirmation snapshot](results/2026-07-29-unordered-fidelity-confirmation.md).

The result changes the interpretation of the discovery:

- the k1 ep100 NFE4 discovery gain of `+.09600` does not replicate; the held-out
  effect is `-.0089 [-.0484,+.0301]`;
- all eight point curves are descriptively non-monotone, but 23/24 paired
  NFE-versus-NFE1 intervals include zero;
- ep100 family-average IoU decreases from `.717775` at NFE1 to `.708625` at
  NFE8, whereas ep300 increases from `.571975` to `.592325`;
- latency rises approximately `1.00x/1.88x/3.65x/7.21x` for NFE
  `1/2/4/8`.

The defensible claim is not that higher NFE generally hurts. It is that additional
depth has no reliable universal positive decision return and therefore cannot be
treated as an a priori fidelity order in this planner. The single unadjusted
k1/ep300 NFE4 interval above zero is insufficient for a universal positive claim.

## Decision after confirmation

1. Replace **unordered fidelity** with the more conservative term **untrusted
   decision fidelity** until true candidate utilities show sign-changing depth
   value across tasks.
2. Treat `64×1` and `32×2` as mandatory static baselines, but confirm their
   breadth comparison on the untouched 80 indices before building a method.
3. Do not train a router from raw top-two margin or the sparse first-decision
   reward audit.
4. Build a later-state, longer-horizon counterfactual candidate dataset that
   separates proposal coverage regret from world-model selection regret.
5. Require released Robomimic Lift/Can and a second generative world-model family
   before making a general claim.
6. Stop more 300k/400k Push-T post-training and static top-fraction racing. The
   next work is bounded by the gates in
   [`oral-paper-blueprint.md`](oral-paper-blueprint.md).
