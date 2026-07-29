# Held-out fixed-candidate NFE confirmation

Status: complete, 32/32 rows, 2026-07-29.

This snapshot records the preregistered Push-T confirmation run before any new
method is designed. It supersedes the 20-trial Node A result for claims about
fixed-candidate NFE. It does not supersede the separate breadth-depth or racing
experiments.

## Provenance

- Code commit: `41938d7e481453ccc2199d90c7171a5b97fb327f`.
- Manifest:
  [`../manifests/unordered-fidelity-confirmation.yaml`](../manifests/unordered-fidelity-confirmation.yaml).
- Queue plan: `company/unordered_confirmation_plan.tsv`.
- Task: Push-T, untouched environment indices 20--99, 80 paired trials per row.
- Candidate set: 32 policy proposals; candidate-action hashes match across
  NFE=1/2/4/8 within every family-policy block.
- World models: k1-grid25 seed 2 best, k32 seed 2 latest, joint-k16 seed 2
  latest, and deep-base-k16 seed 1 rollout best.
- Policies: epoch 100 is primary; epoch 300 is the preregistered proposal-policy
  replication.
- Hardware: four independent nodes, four H100 GPUs per node.
- W&B project: `driftfm-unordered-fidelity-confirmation-company`.
- Shared atomic markers:
  `/group-volume/danny-dataset/driftworld/checkpoints/experiments/`
  `gpc-unordered-a-confirm80-*.json`.
- Checkpoint retention: frozen inputs; no new model checkpoint was produced.

NFE is an inference-compute setting, not a metric. The primary metric is paired
episode maximum IoU; success is the fraction above the released 0.95 threshold.

## Exact compact result

IoU, success, and latency columns are ordered by NFE `1/2/4/8`.

| Family | Policy | Mean IoU | Success | Mean planning latency (s) | Descriptive best |
| --- | --- | --- | --- | --- | --- |
| deep-base-k16 | ep100 | `.7153/.7085/.7135/.7080` | `.087/.138/.150/.175` | `.78/1.48/2.86/5.61` | NFE1 |
| joint-k16 | ep100 | `.7265/.7155/.7010/.7089` | `.163/.113/.113/.138` | `.78/1.47/2.85/5.62` | NFE1 |
| k1-grid25 | ep100 | `.7096/.7022/.7008/.7017` | `.163/.113/.113/.125` | `.78/1.47/2.85/5.62` | NFE1 |
| k32 | ep100 | `.7197/.7376/.7373/.7159` | `.138/.237/.175/.113` | `.78/1.47/2.85/5.61` | NFE2 |
| deep-base-k16 | ep300 | `.5710/.5865/.5821/.5980` | `.013/.025/.013/.037` | `.78/1.47/2.85/5.61` | NFE8 |
| joint-k16 | ep300 | `.5644/.5590/.5851/.5863` | `.013/.013/.025/.000` | `.78/1.47/2.85/5.62` | NFE8 |
| k1-grid25 | ep300 | `.5607/.5853/.5915/.5882` | `.000/.037/.000/.025` | `.78/1.47/2.85/5.62` | NFE4 |
| k32 | ep300 | `.5918/.5832/.5913/.5968` | `.000/.025/.013/.013` | `.78/1.47/2.85/5.61` | NFE8 |

## Paired effects relative to NFE1

Intervals are paired bootstrap 95% intervals over the same 80 environment
indices and proposal sets.

| Family | Policy | NFE2 minus NFE1 | NFE4 minus NFE1 | NFE8 minus NFE1 |
| --- | --- | --- | --- | --- |
| deep-base-k16 | ep100 | `-.0068 [-.0343,+.0191]` | `-.0018 [-.0351,+.0314]` | `-.0073 [-.0441,+.0286]` |
| joint-k16 | ep100 | `-.0110 [-.0428,+.0213]` | `-.0255 [-.0631,+.0109]` | `-.0176 [-.0577,+.0218]` |
| k1-grid25 | ep100 | `-.0074 [-.0453,+.0302]` | `-.0089 [-.0484,+.0301]` | `-.0079 [-.0468,+.0298]` |
| k32 | ep100 | `+.0179 [-.0153,+.0498]` | `+.0176 [-.0199,+.0554]` | `-.0038 [-.0413,+.0335]` |
| deep-base-k16 | ep300 | `+.0155 [-.0072,+.0403]` | `+.0111 [-.0136,+.0378]` | `+.0270 [-.0010,+.0591]` |
| joint-k16 | ep300 | `-.0054 [-.0307,+.0175]` | `+.0208 [-.0125,+.0558]` | `+.0219 [-.0085,+.0536]` |
| k1-grid25 | ep300 | `+.0246 [-.0030,+.0540]` | `+.0308 [+.0034,+.0609]` | `+.0275 [-.0041,+.0634]` |
| k32 | ep300 | `-.0086 [-.0326,+.0137]` | `-.0005 [-.0284,+.0256]` | `+.0050 [-.0248,+.0358]` |

The family-average descriptive IoU curves are
`.717775/.715950/.713150/.708625` for ep100 and
`.571975/.578500/.587500/.592325` for ep300. Thus the unadjusted NFE8-minus-NFE1
policy interaction is `+.02950`. This is descriptive: a hierarchical policy
interaction was not preregistered and no multiplicity-adjusted inference was
performed.

## What is and is not supported

1. **A universal positive return to depth is not supported.** All eight point
   curves are non-monotone, the preferred depth changes with family and proposal
   policy, and 23 of 24 paired NFE-versus-NFE1 intervals include zero.
2. **The discovery k1 gain did not replicate.** On the first 20 indices, k1
   NFE4-minus-NFE1 was `+.09600`; on untouched indices 20--99 it is
   `-.0089 [-.0484,+.0301]`. The original result must not anchor the paper.
3. **Higher NFE is not proven harmful.** Most intervals include practically
   meaningful effects in both directions. The valid statement is that depth is
   not a trustworthy *a priori* fidelity ladder for this planner.
4. **One unadjusted contrast is insufficient for a positive depth claim.**
   k1/ep300 NFE4 is the only interval excluding zero. It is one of 24 reported
   contrasts and is not consistent across families or policies.
5. **Latency scales much more reliably than decision quality.** Relative to
   NFE1, measured latency is approximately `1.00x/1.88x/3.65x/7.21x`, while the
   largest supported IoU gain is about three points.
6. **The immediate first-decision audit remains unusable as a mechanism
   target.** Candidate reward was constant in 11/20 discovery states and cannot
   justify training a router.

## Research decision

Treat NFE as an **untrusted, policy- and candidate-set-conditioned evaluator**,
not as a guaranteed low-to-high fidelity hierarchy. Stop further long Push-T
post-training and static top-fraction racing. Before learning an allocator:

1. complete the held-out `64 candidates × NFE1` breadth comparison against the
   existing `32 × NFE2` rows;
2. measure proposal coverage and selection regret with longer-horizon
   counterfactual returns from cloned later-episode states;
3. reproduce the phenomenon on released Robomimic Lift and Can tasks and on a
   second generative world-model family.

The paper-level interpretation and falsification gates are specified in
[`../oral-paper-blueprint.md`](../oral-paper-blueprint.md).
