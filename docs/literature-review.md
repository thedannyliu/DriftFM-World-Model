# Literature and novelty audit

## Closest work

| Work | Relevant contribution | Boundary for this project |
| --- | --- | --- |
| [DriftWorld](https://arxiv.org/abs/2607.15065) (Lu et al., 2026) | One-forward-pass action-conditioned video world model and GPC-RANK planning; reports Push-T IoU 0.781/0.734 with 50 proposals | Reproduce its released Push-T setup and treat it as the endpoint baseline. The released Push-T code, unlike the broader paper, does not expose DINO, no-action negatives, or self-forcing; comparisons therefore use the executable release symmetrically. |
| [Drift Flow Matching](https://arxiv.org/abs/2605.17244) (Ma et al., 2026) | Learns transport between arbitrary marginal times; direct maps retain NFE=1 while composition enables test-time scaling | It does not establish variable-NFE action-conditioned video planning under fixed latency. No official code link was listed on arXiv when this audit was refreshed, so conditional-video choices are explicitly ablated rather than presented as exact reproduction. |
| [Test-Time Scaling for World Action Models](https://arxiv.org/abs/2607.17454) (Zhao et al., 2026) | GeoBoN ranks independent WAM rollouts by frozen cross-view geometry; its gate triggers more samples on inconsistent action/future pairs | Avoid claiming adaptive test-time compute broadly. Our controlled variable is transport depth for the *same action proposal and initial noise*, compared against proposal breadth on a measured fixed-latency frontier. |
| Flow matching / rectified flow | Continuous transport and few-step sampling | The novelty cannot be merely adding a time embedding or multi-step integration. |

## Defensible novelty

The central object is a planning budget frontier, not a hybrid loss: for the same action
proposal, endpoint-compatible transport exposes a depth axis that DriftWorld lacks.
This allows direct measurement of when compute should buy more independent proposals
and when it should refine uncertain dynamics. Coarse-to-fine planning is meaningful
only if multi-NFE predictions change proposal ranking in a way that improves paired
environment IoU at equal wall-clock cost.

The result would change the usual “fast world model means more samples” perspective only if
the experiment identifies a regime where refining dynamics for a small selected subset is
more valuable than purchasing additional independent actions. Merely improving LPIPS at
NFE=4 is supporting evidence, not the paper-level contribution.

## Falsifiers

- Multi-NFE improves perceptual metrics but not block pose or proposal ranking.
- More one-step proposals dominate refinement at every matched latency.
- Gains arise from post-training compute, parameter count, or a changed task setup
  rather than arbitrary-time transport.
- The NFE=1 endpoint is degraded enough that one model no longer spans the frontier.

The literature review must be refreshed before writing a paper submission; dates and
claims above reflect the July 2026 project start and are not a substitute for a final
systematic search.
