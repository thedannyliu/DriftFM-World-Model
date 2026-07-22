# DriftFlowWorld research log

This directory is the source of truth for the Push-T research cycle. Read in this
order:

1. [`research-protocol.md`](research-protocol.md) locks comparisons and decision gates.
2. [`literature-review.md`](literature-review.md) records the novelty boundary.
3. [`experiments.md`](experiments.md) records hypotheses, jobs, metrics, and decisions.
4. [`research-idea.md`](research-idea.md) contains the broader project proposal.

Each experiment table answers one question. Every run records the task, seed,
manifest, GPU, code commit, parent/output checkpoint, W&B project and run ID, metrics,
and conclusion. Tables may be updated with results, but completed rows are not silently
rewritten.

The local paper PDF is intentionally ignored by Git. Large inputs and generated
artifacts live at `/storage/scratch1/9/eliu354/driftflowworld/`; only manifests and
small textual summaries are versioned.
