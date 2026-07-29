# DriftFlowWorld research log

This directory is the source of truth for the Push-T research cycle. Read in this
order:

1. [`research-protocol.md`](research-protocol.md) locks comparisons and decision gates.
2. [`advantage-aligned-transport-hypotheses.md`](advantage-aligned-transport-hypotheses.md)
   defines the current post-PoC mechanism hypotheses and diagnostic gates.
3. [`unordered-generative-fidelity.md`](unordered-generative-fidelity.md) defines the
   current decision-level hypothesis, prior-work boundary, and four-node experiment.
4. [`literature-review.md`](literature-review.md) records the novelty boundary.
5. [`experiments.md`](experiments.md) records hypotheses, jobs, metrics, and decisions.
6. [`results/`](results/) contains dated, immutable experiment snapshots and analysis.
7. [`weekend-research-plan.md`](weekend-research-plan.md) defines the earlier
   four-node aggressive follow-up.
8. [`research-idea.md`](research-idea.md) contains the broader project proposal.

Each experiment table answers one question. Every run records the task, seed,
manifest, GPU, code commit, parent/output checkpoint, W&B project and run ID, metrics,
and conclusion. Tables may be updated with results, but completed rows are not silently
rewritten.

The local paper PDF is intentionally ignored by Git. Large inputs and generated
artifacts live at `/storage/scratch1/9/eliu354/driftflowworld/`; only manifests and
small textual summaries are versioned.

Current complete shared-status snapshot:
[`results/2026-07-28-1827-all-experiments-status.md`](results/2026-07-28-1827-all-experiments-status.md).
The corrected non-degenerate routing addendum is
[`results/2026-07-28-action-routing-refresh.md`](results/2026-07-28-action-routing-refresh.md).
The decision to stop the obsolete deep weekend queues is
[`results/2026-07-28-weekend-pruning-decision.md`](results/2026-07-28-weekend-pruning-decision.md).
The earlier 8/16 diagnostic is retained as
[`results/2026-07-28-hypothesis-audit-partial.md`](results/2026-07-28-hypothesis-audit-partial.md).
The current 15/16 equal-budget candidate-racing result is
[`results/2026-07-29-unordered-fidelity-node-c-partial.md`](results/2026-07-29-unordered-fidelity-node-c-partial.md).
The current 15/16 fixed-candidate NFE result is
[`results/2026-07-29-unordered-fidelity-node-a-partial.md`](results/2026-07-29-unordered-fidelity-node-a-partial.md).
The current 15/16 equal-budget breadth-depth result is
[`results/2026-07-29-unordered-fidelity-node-b-partial.md`](results/2026-07-29-unordered-fidelity-node-b-partial.md).
The current 15/16 ep300 policy-shift result is
[`results/2026-07-29-unordered-fidelity-node-d-partial.md`](results/2026-07-29-unordered-fidelity-node-d-partial.md).
