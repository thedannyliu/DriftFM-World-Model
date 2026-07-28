# Literature and novelty audit

## Closest work

| Work | Relevant contribution | Boundary for this project |
| --- | --- | --- |
| [DriftWorld](https://arxiv.org/abs/2607.15065) (Lu et al., 2026) | One-forward-pass action-conditioned video world model and GPC-RANK planning; reports Push-T IoU 0.781/0.734 with 50 proposals | Reproduce its released Push-T setup and treat it as the endpoint baseline. The released Push-T code, unlike the broader paper, does not expose DINO, no-action negatives, or self-forcing; comparisons therefore use the executable release symmetrically. |
| [Drift Flow Matching](https://arxiv.org/abs/2605.17244) (Ma et al., 2026) | Learns transport between arbitrary marginal times; direct maps retain NFE=1 while composition enables test-time scaling | It does not establish variable-NFE action-conditioned video planning under fixed latency. No official code link was listed on arXiv when this audit was refreshed, so conditional-video choices are explicitly ablated rather than presented as exact reproduction. |
| [Flow Map Matching](https://arxiv.org/abs/2406.07507) (Boffi et al., 2025 revision) | Formalizes two-time flow maps, their composition property, and direct/Lagrangian/Eulerian training objectives | A generic semigroup or composition loss is not novel. This project must explain and control why approximate conditional maps have a finite useful NFE rather than restating exact flow-map consistency. |
| [One Step Diffusion via Shortcut Models](https://arxiv.org/abs/2410.12557) (Frans et al., 2024) | Conditions one network on current noise level and desired step size for variable one/few-step generation | Step-size conditioning and a single variable-budget network are prior art. The contribution must be action-relevant error control and planning consequences. |
| [Consistency Flow Matching](https://arxiv.org/abs/2407.02398) (Yang et al., 2024) | Enforces velocity-field self-consistency and uses multi-segment optimization for few-step generation | Applying velocity consistency to DriftWorld is incremental; moreover exact consistency makes different partitions agree rather than guaranteeing deeper predictions are better. |
| [MWM](https://arxiv.org/abs/2603.07799) (Yan et al., 2026) | Uses action-conditioned consistency and inference-consistent state distillation for efficient navigation-world-model rollout | Avoid generic claims about first action-conditioned consistency. Our controlled object is composition across generative transport time for a fixed observation/action condition and noise, not only consistency across physical rollout time. |
| [Is the Future Compatible?](https://arxiv.org/abs/2605.07514) (Ruan et al., 2026) | Diagnoses action-state consistency in world-action-model rollouts and uses consensus for test-time selection | Physical action/future compatibility is an important downstream metric but is distinct from the generative-time route defect audited here. |
| [Test-Time Scaling for World Action Models](https://arxiv.org/abs/2607.17454) (Zhao et al., 2026) | GeoBoN ranks independent WAM rollouts by frozen cross-view geometry; its gate triggers more samples on inconsistent action/future pairs | Avoid claiming adaptive test-time compute broadly. Our controlled variable is transport depth for the *same action proposal and initial noise*, compared against proposal breadth on a measured fixed-latency frontier. |
| Flow matching / rectified flow | Continuous transport and few-step sampling | The novelty cannot be merely adding a time embedding or multi-step integration. |

## Defensible novelty

The current central object is an advantage-aligned approximation hierarchy, not a
generic consistency loss. Exact flow-map consistency makes every partition return the
same endpoint, while useful test-time scaling requires shorter approximate maps to
reduce conditional future risk without compounding model-source error. This project
must establish a measurable relation among local-map accuracy, route defect,
action-relevant degradation, and the finite optimal NFE.

If that mechanism is established, the planning-level object remains a fixed-latency
depth-versus-breadth frontier: for the same action proposal and initial noise,
endpoint-compatible transport exposes a depth axis that DriftWorld lacks. The result
would change the usual “fast world model means more samples” perspective only if
refining dynamics for a selected subset is more valuable than purchasing additional
independent actions. Merely improving LPIPS is supporting evidence, not the
paper-level contribution.

## Falsifiers

- Multi-NFE improves perceptual metrics but not block pose or proposal ranking.
- More one-step proposals dominate refinement at every matched latency.
- Gains arise from post-training compute, parameter count, or a changed task setup
  rather than arbitrary-time transport.
- The NFE=1 endpoint is degraded enough that one model no longer spans the frontier.

The literature review must be refreshed before writing a paper submission; dates and
claims above reflect the July 2026 project start and are not a substitute for a final
systematic search.
