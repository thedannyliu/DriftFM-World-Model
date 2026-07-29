# DriftFlowWorld research log

This directory is the source of truth for the DriftFlowWorld research cycle.
Push-T is the completed discovery task; Robomimic Lift/Can and a second
world-model family are mandatory gates for the proposed general paper. Read in this
order:

1. [`oral-paper-blueprint.md`](oral-paper-blueprint.md) is the current paper thesis,
   evidence ledger, method boundary, required experiments, and kill criteria.
2. [`decision-fidelity-theory.md`](decision-fidelity-theory.md) is the proof audit:
   proved statements, conditional guarantees, counterexamples, open claims, and
   failed proof attempts.
3. [`decision-fidelity-experiments.md`](decision-fidelity-experiments.md) maps those
   claims to staged mandatory experiments and explicit continue/narrow/stop gates.
4. [`unordered-generative-fidelity.md`](unordered-generative-fidelity.md) records the
   completed Push-T decision-level discovery and held-out conclusion.
5. [`literature-review.md`](literature-review.md) separates established top-venue
   results from current preprints and locks the novelty boundary.
6. [`research-protocol.md`](research-protocol.md) preserves the earlier
   DriftFlowWorld comparison and its historical gates.
7. [`advantage-aligned-transport-hypotheses.md`](advantage-aligned-transport-hypotheses.md)
   preserves the post-PoC mechanism hypotheses and diagnostic gates.
8. [`experiments.md`](experiments.md) records hypotheses, jobs, metrics, and decisions.
9. [`results/`](results/) contains dated, immutable experiment snapshots and analysis.
10. [`weekend-research-plan.md`](weekend-research-plan.md) defines the earlier
   four-node aggressive follow-up.
11. [`research-idea.md`](research-idea.md) preserves the original broader proposal.

Each experiment table answers one question. Every run records the task, seed,
manifest, GPU, code commit, parent/output checkpoint, W&B project and run ID, metrics,
and conclusion. Tables may be updated with results, but completed rows are not silently
rewritten.

The local paper PDF is intentionally ignored by Git. Large inputs and generated
artifacts live at `/storage/scratch1/9/eliu354/driftflowworld/`; only manifests and
small textual summaries are versioned.

The current complete 32/32 held-out fixed-candidate result is
[`results/2026-07-29-unordered-fidelity-confirmation.md`](results/2026-07-29-unordered-fidelity-confirmation.md).
Current complete shared-status snapshot:
[`results/2026-07-28-1827-all-experiments-status.md`](results/2026-07-28-1827-all-experiments-status.md).
The corrected non-degenerate routing addendum is
[`results/2026-07-28-action-routing-refresh.md`](results/2026-07-28-action-routing-refresh.md).
The decision to stop the obsolete deep weekend queues is
[`results/2026-07-28-weekend-pruning-decision.md`](results/2026-07-28-weekend-pruning-decision.md).
The earlier 8/16 diagnostic is retained as
[`results/2026-07-28-hypothesis-audit-partial.md`](results/2026-07-28-hypothesis-audit-partial.md).
The historical 15/16 equal-budget candidate-racing discovery is
[`results/2026-07-29-unordered-fidelity-node-c-partial.md`](results/2026-07-29-unordered-fidelity-node-c-partial.md).
The historical 15/16 fixed-candidate NFE discovery is
[`results/2026-07-29-unordered-fidelity-node-a-partial.md`](results/2026-07-29-unordered-fidelity-node-a-partial.md).
The historical 15/16 equal-budget breadth-depth discovery is
[`results/2026-07-29-unordered-fidelity-node-b-partial.md`](results/2026-07-29-unordered-fidelity-node-b-partial.md).
The historical 15/16 ep300 policy-shift discovery is
[`results/2026-07-29-unordered-fidelity-node-d-partial.md`](results/2026-07-29-unordered-fidelity-node-d-partial.md).
