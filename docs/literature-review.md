# Literature and novelty audit

## Closest work

| Work | Relevant contribution | Boundary for this project |
| --- | --- | --- |
| DriftWorld (local paper, 2026) | One-forward-pass action-conditioned video world model and GPC-RANK planning | Reproduce its released Push-T setup and treat it as the endpoint baseline. |
| Drift Flow Matching (2026) | Learns transport between arbitrary marginal times; inference quality can scale with NFE | It does not establish variable-NFE action-conditioned video planning under fixed latency. |
| Gated GeoBoN (2026) | Adaptively spends additional world-model samples based on action/future consistency | Avoid claiming adaptive test-time compute broadly; our variable is transport depth for the same proposal and a measured breadth-depth frontier. |
| Flow matching / rectified flow | Continuous transport and few-step sampling | The novelty cannot be merely adding a time embedding or multi-step integration. |

## Defensible novelty

The central object is a planning budget frontier, not a hybrid loss: for the same action
proposal, endpoint-compatible transport exposes a depth axis that DriftWorld lacks.
This allows direct measurement of when compute should buy more independent proposals
and when it should refine uncertain dynamics. Coarse-to-fine planning is meaningful
only if multi-NFE predictions change proposal ranking in a way that improves paired
environment IoU at equal wall-clock cost.

## Falsifiers

- Multi-NFE improves perceptual metrics but not block pose or proposal ranking.
- More one-step proposals dominate refinement at every matched latency.
- Gains arise from post-training compute, parameter count, or a changed task setup
  rather than arbitrary-time transport.
- The NFE=1 endpoint is degraded enough that one model no longer spans the frontier.

The literature review must be refreshed before writing a paper submission; dates and
claims above reflect the July 2026 project start and are not a substitute for a final
systematic search.
