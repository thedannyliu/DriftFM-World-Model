# Advantage-aligned transport: hypotheses and diagnostic protocol

## Research decision

The original DriftFlowWorld proof of concept did not establish monotonic test-time
scaling. The strongest completed result is a reproducible two-step optimum: K=1 with
exact-grid replay improves locked100 mean LPIPS from `.015329` at NFE1 to `.009854`
at NFE2, then regresses to `.011447/.013297` at NFE4/8. Larger particle counts,
deeper grid exposure, model-generated source replay, and longer training have not
restored monotonic scaling.

The next phase is diagnostic. It does not assume that adding a symmetric consistency
loss is the solution. An exact flow map obeys

```text
Phi(s,u,z;c) = Phi(t,u,Phi(s,t,z;c);c),
```

so all partitions produce the same endpoint. Exact self-consistency alone therefore
cannot explain a quality gain from additional NFE. Useful test-time scaling requires
short learned maps to approximate the underlying conditional flow more accurately
than a single long map, while errors on model-generated intermediate states remain
controlled.

The working perspective is:

> Additional NFE is useful only when the local-map approximation benefit exceeds
> model-induced composition and amplification error.

Here `s`, `t`, and `u` are **generative transport times**, not physical robot time.
Physical action/state consistency is evaluated as a downstream consequence through
block-pose and planning metrics; it is not interchangeable with transport-map
composition.

## Falsifiable hypotheses

### H1 — finite-NFE optimum from composition amplification

For a fixed condition `c` and initial noise `z`, define adjacent-route defect

```text
D(N,2N) = d(y_N, y_2N),
```

where `y_N` is the endpoint obtained with N equal transport intervals. Define
degradation

```text
G(N,2N) = R(y_2N, y*) - R(y_N, y*).
```

The mechanism predicts that checkpoints/examples with larger route defect have larger
positive degradation, especially for NFE2->4 and NFE4->8. A large defect by itself
is not evidence: it is useful when the deeper route is more accurate and harmful when
the deeper route is less accurate.

**Falsifier:** within-checkpoint defect does not predict degradation, or the sign is
inconsistent across seeds and model families.

### H2 — model-induced intermediate states cause the failure

At each transport interval, compare the same local map on:

1. a clean paired marginal source `(1-s) noise + s target`; and
2. the source produced by preceding learned maps.

The off-manifold penalty is the free-source local error minus clean-source local
error. The mechanism predicts a growing positive penalty in the later steps of
NFE4/8.

**Falsifier:** clean and model-source errors are similar in failing intervals, or
model sources are better even when deeper endpoints regress.

### H3 — the defect is decision-relevant

Composition failure should concentrate in examples with more object motion, action
path length, or block-pose change. High-motion examples should exhibit a larger
NFE2->4 or NFE4->8 degradation than low-motion examples.

**Falsifier:** the effect is confined to pixel error, does not appear in block-pose
error, and is unrelated to motion difficulty.

### H4 — advantage-aligned transport can restore an anytime frontier

Only after H1-H3 pass should a new training objective be implemented. Its target is
not equality of all NFE outputs. It must combine:

- a correct local-flow anchor;
- a bounded composition defect on model-generated sources; and
- a one-sided, coupling-aware objective that makes expected action-relevant risk
  non-increasing with additional NFE.

Naively pairing every noise sample with one target video can collapse multimodal
conditional futures. The training objective therefore needs a valid conditional
coupling, distributional scoring rule, or high-quality conditional-flow teacher.

**Falsifier:** the method reduces route defect but flattens the NFE2 benefit, degrades
NFE1 by more than 5%, or fails to improve block-pose and planning outcomes.

## Stage-1 diagnostic design

The first audit reuses completed checkpoints and does not train new models. Each
checkpoint is evaluated on 64 fixed validation chunks with four fixed noise particles.
The audit records per-example values before aggregation.

| Question | Metrics |
| --- | --- |
| Does more compute help? | target MSE and final block-vertex error at NFE1/2/4/8 |
| Are routes inconsistent? | adjacent output MSE for 1->2, 2->4, and 4->8 |
| Is inconsistency harmful? | within-checkpoint correlation of adjacent defect with adjacent degradation |
| Are predicted sources off path? | source MSE to paired marginal; free-minus-clean local error and relative penalty |
| Is the failure dynamics-sensitive? | motion-pixel, action-path, and block-pose-motion terciles |
| Is the effect robust? | three seeds and multiple K/grid/source/training-duration families |

The four-node allocation is:

| Node | Four concurrent checkpoints | Primary contrast |
| --- | --- | --- |
| A | K=1 grid=.25 latest seeds 1/2/3; weekend grid-only rollout-best | reproducible NFE2-positive family |
| B | K=32 latest seeds 1/2/3; weekend K=16 base rollout-best | particle scaling versus base |
| C | K=16 grid+source latest seeds 1/2/3; source-only rollout-best | source replay and coupling |
| D | weekend K=16 base latest/best for seeds 1/2 | 300k regression versus rollout-best |

Every GPU handles one checkpoint. Results and full logs are written below
`/user-volume/driftworld/`; only compact summaries are printed. Each checkpoint is
logged as an online W&B `hypothesis-audit` run.

## Preregistered diagnostic gates

These gates select the next experiment; they are not paper claims.

1. **Composition signal:** for both NFE2->4 and NFE4->8, the median
   within-checkpoint Pearson correlation between route defect and degradation is at
   least `.30`, with positive sign in at least 75% of available checkpoints.
2. **Off-manifold signal:** mean free-source local error exceeds clean-source local
   error by at least 20% in later NFE4/8 intervals in at least 75% of checkpoints
   that exhibit endpoint degradation.
3. **Dynamics signal:** high block-motion tercile degradation is at least 25% larger
   than the low-motion tercile in at least two independent checkpoint families, and
   the direction agrees for pixel and block-pose risk.

Decision:

- All three pass: implement a minimal advantage-aligned training experiment.
- H1/H2 pass but H3 fails: treat this as a generative-quality issue, not a
  world-model/planning paper.
- H1 fails: reject composition amplification as the principal mechanism and audit
  time conditioning, coupling construction, and implementation fidelity instead.
- Only NFE2 remains useful: narrow the project to an adaptive two-step model and do
  not claim arbitrary test-time scaling.

## Top-venue bar

A symmetric composition loss is already covered in spirit by flow-map,
shortcut-model, and consistency-model literature. A top AI venue submission needs:

1. a general error-controlled view explaining when inference depth helps or hurts;
2. a coupling-aware method distinct from applying an existing consistency loss;
3. monotonic or calibrated anytime behavior across multiple action-conditioned
   world-model tasks and seeds;
4. paper-matched DriftWorld evaluation; and
5. fixed-latency depth-versus-proposal-breadth planning results.

Push-T LPIPS alone is insufficient. The project should not make a DriftWorld
superiority claim until the released 1000-video protocol and decision-level
evaluation are complete.

## Company commands

Update and verify the required commit printed in the handoff, then run one command
on each independent 4xH100 node:

```bash
cd /user-volume/repo/DriftFM-World-Model
git pull --ff-only origin main
DRIFTFLOWWORLD_SKIP_ASSETS=1 bash company/setup.sh
wandb login --relogin
```

```bash
# Node A
AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-a
```

```bash
# Node B
AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-b
```

```bash
# Node C
AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-c
```

```bash
# Node D
AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-d
```

Inspect the allocation without GPUs or credentials:

```bash
for node in node-a node-b node-c node-d; do
  bash company/run_hypothesis_audit.sh "$node" --print-plan
done
```

The final terminal JSON is designed to be pasted back into the research log. Do not
start H4 training until the diagnostic summary has been reviewed against the gates
above.

After all four nodes finish, print one combined 16-checkpoint report from any node:

```bash
cd /user-volume/repo/DriftFM-World-Model
python3 company/status_hypothesis_audit.py
```

An individual four-checkpoint node summary is labeled partial coverage. Only the
combined report is allowed to select the next research stage.

## 2026-07-28 observation and Stage 2

Node A, B, and C have completed 12/16 audits. Mean MSE improves by
`14.20%/16.60%/16.90%` at NFE2/4/8 versus NFE1, while final block-vertex error
improves by `4.07%/3.56%/2.72%`. Composition correlations are negative:
median `-.4554/-.3004` at NFE2→4/NFE4→8, with a positive sign in only `0/12`
and `1/12` checkpoints. The preregistered composition-amplification mechanism is
therefore rejected: even if all four missing node-D audits were positive, neither
transition could reach the required 75% positive fraction. Model-source penalties
are large and the off-manifold gate passes, but the negative correlations show that
shift magnitude is not the proposed cause of degradation. The motion gate remains a
provisional lead. Full values and limitations are recorded in
[`results/2026-07-28-1827-all-experiments-status.md`](results/2026-07-28-1827-all-experiments-status.md).

The immediate follow-up is not H4 training. Complete node D for protocol closure and
finish the 1000-video paired advantage frontier on frozen checkpoints, measuring
LPIPS, MSE, block-pose risk, and pre-inference action-path difficulty at
NFE1/2/4/8. Node A's first eight checkpoints already show a reproducible finite NFE2
optimum for K=1/grid=.25: across three latest seeds, NFE2 changes mean LPIPS, MSE,
and block error by `-31.96%`, `-3.61%`, and `-1.96%`. NFE4/8 then worsen MSE and
block dynamics.

The original routing diagnostic used the first half of an ordered evaluation set for
the threshold and the second half for evaluation. Every completed checkpoint reports
exactly zero adaptive-versus-random change, which is consistent with a single-NFE
decision; the old status reporter omitted the selected fraction. Those routing
fields are uninterpretable. Re-summarization exposes the fraction and interleaves
deterministic even-index development and odd-index test examples. Because this
correction follows inspection of the outcome, routing remains exploratory until an
independently preregistered replication.

This test distinguishes three perspectives:

1. generic composition failure;
2. a distortion/perception tradeoff with action-relevant depth benefit; and
3. a non-action-relevant MSE artifact.

Only the second outcome motivates risk-conditional inference depth. The stronger
current framing is **finite-depth transport regularization** rather than generic
test-time scalability: exact-grid supervision can create a useful two-step solver,
while repeated application accumulates distortion and dynamics error. It still
requires a same-code official baseline, paired confidence intervals, latency,
task-level planning evidence, and another environment before meeting the top-venue
bar.
