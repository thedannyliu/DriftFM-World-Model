# Aggressive corrected Drift Flow weekend plan

## Objective

Use four independent 4xH100 nodes to determine whether the current lead result is a
real transport principle:

> Endpoint-normalized Drift Flow becomes composable when exact dyadic inference
> intervals anchor the transport field and a controlled amount of EMA-composed
> source replay exposes the model to its own source distribution.

The plan is deliberately deeper than a 72-hour allocation. An unfinished queue is a
valid outcome because every stage is resumable and every completed milestone is
evaluated online.

## Shared protocol

- Task and parent: Push-T, released DriftWorld step 1,180,500.
- Hardware: one independent 4xH100 node per queue.
- Development evaluation: first 25 videos, full-rollout LPIPS and final block-vertex
  error at NFE 1/2/4/8.
- Screen milestones: 10k and 30k updates.
- Winner milestones: 60k/100k/150k/200k/250k/300k/400k.
- Runner-up milestones: 60k/100k/150k/200k.
- Final evaluation: 100 videos at NFE 1/2/4/8 for both `latest` and rollout-selected
  `best`.
- Logging: online W&B for every training and evaluation run.
- Retention: exactly `ckpt-latest.pth` and `ckpt-best.pth` per run. `latest` is the
  full resumable state. `best` is promoted only when the fixed 25-video rollout score
  improves. Adaptation-validation loss remains logged but no longer controls
  checkpoint retention.

The plan creates at most 72 seed-specific output directories. At the observed
129.6 MiB per checkpoint, two checkpoints per directory require approximately
18.2 GiB of new checkpoint storage. Small metric JSON and logs use the shared
`/user-volume` runtime root.

The rollout selector gives NFE1 twice the base weight, adds all higher-NFE LPIPS and
vertex errors, and penalizes degradation between adjacent NFE values. This prevents a
candidate with a good NFE1 image but collapsing NFE8 dynamics from winning.

## Node A — Is grid/source replay a real interaction?

The 30k screen contains three controls and a 3x3 response surface:

- controls: endpoint-normalized base, grid=.25 only, source-replay=.25 only;
- grid replay probability: .125/.25/.50;
- composed-source replay probability: .10/.25/.50;
- fixed K=16, grid depth NFE4, two-step EMA source composition.

The top candidate continues across five seeds to 400k; the runner-up continues across
three seeds to 200k.

Supporting result: a joint setting beats both single-factor controls on LPIPS and
vertex error, and the direction replicates on at least three of five winner seeds.
If only a narrow seed-1 cell wins, the interaction hypothesis fails.

Planned depth: approximately 2.9M updates, at least 96 hours at an optimistic 30k
updates/hour before evaluation overhead.

## Node B — Must training-grid depth match inference depth?

The screen crosses:

- maximum dyadic training-grid NFE: 2/4/8/16;
- grid replay probability: .125/.25/.50;
- fixed K=16, source replay=.25, two-step EMA composition.

Grid depth `d` samples every adjacent dyadic interval from NFE2 through `d`. The top
candidate continues across five seeds to 400k; the runner-up continues across three
seeds to 200k.

Supporting result: adding NFE8 intervals improves NFE8 rollout without materially
degrading NFE1/2, with a reproducible saturation point. A monotonic compute-only gain
without matching NFE behavior would not support the grid-depth hypothesis.

Planned depth: approximately 2.9M updates plus evaluation.

## Node C — How on-policy must the composed source be?

The screen crosses:

- EMA composition steps: 1/2/4/8;
- composed-source replay probability: .10/.25/.50;
- fixed K=16, grid=.25, grid depth NFE4.

The top candidate continues across five seeds to 400k; the runner-up continues across
three seeds to 200k.

Supporting result: composition depth has a reproducible optimum and improves
higher-NFE dynamics relative to the one-step and source-free controls. If all depths
are equivalent, source-distribution matching is not the mechanism.

Planned depth: approximately 2.9M updates plus evaluation.

## Node D — Does the interaction survive matched causal controls?

Six fixed arms run on seeds 1/2/3 through 300k:

1. K=16 corrected base;
2. K=16 grid-only;
3. K=16 source-only;
4. joint grid/source with K=1;
5. joint grid/source with K=16;
6. joint grid/source with K=32.

Every arm evaluates at 10k/30k/60k/100k/150k/200k/250k/300k and receives a locked
100-video latest/best evaluation.

Supporting result: the joint K=16 arm beats matched base and both single-factor
controls across seeds. K=1/16/32 separates whether positive-particle count is a
necessary part of the interaction or only a variance/computation tradeoff.

Planned depth: 5.4M updates, at least 180 hours at 30k updates/hour before evaluation.

## Decision order

1. Use 30k screens to select; do not wait for every deep run before reading the
   response surfaces.
2. Treat cross-seed direction at 30k as the first replication gate.
3. Use rollout-selected best to detect whether 60k–400k training regresses.
4. Advance to the paper-matched 1000-video evaluation only after one joint family
   beats its causal controls and replicates.
5. Do not claim superiority to DriftWorld from the 25- or 100-video development
   sets.

## Launch and status

Run one command on each node:

```bash
bash company/run_weekend_research_queue.sh node-a
bash company/run_weekend_research_queue.sh node-b
bash company/run_weekend_research_queue.sh node-c
bash company/run_weekend_research_queue.sh node-d
```

All paths are shared. From any node:

```bash
python3 company/status_weekend_research.py all
```
