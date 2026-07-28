# Advantage-aligned transport audit — partial snapshot, 2026-07-28

## Scope

Source: `company/status_hypothesis_audit.py` at commit `93197b0`, captured
2026-07-28 03:39 UTC on `run336369-sam4-8`.

- Completed allocations: node A and node C, 8 of 16 preregistered checkpoints.
- Missing allocations: node B (K=32/base) and node D (deep latest/best).
- Each checkpoint: 64 fixed validation chunks, four fixed noise particles,
  NFE `1/2/4/8`, endpoint-normalized transport.
- W&B project: `driftfm-world-model-company`, job type `hypothesis-audit`.
- The validation chunks are not the released 1000-video DriftWorld test protocol.

This is a partial diagnostic snapshot. It cannot select H4 training or support a
paper claim.

## Aggregate signal

Across the eight available checkpoints:

| Risk | Mean NFE 1/2/4/8 | Change at NFE2/4/8 vs NFE1 | Adjacent improvement frequency |
| --- | --- | --- | --- |
| Pixel MSE | `.00054176/.00044536/.00042646/.00042184` | `-17.80%/-21.28%/-22.14%` | `8/8`, `8/8`, `6/8` |
| Final block-vertex error | `.026760/.024952/.024769/.024879` | `-6.76%/-7.44%/-7.03%` | `7/8`, `6/8`, `2/8` |

The result differs from the locked100 LPIPS curves, where K=1 grid=.25 improves at
NFE2 and then rebounds at NFE4/8. The available evidence therefore does **not**
support a metric-independent statement that deeper transport universally degrades.
It instead suggests a distortion/perception/action-risk disagreement that must be
measured on aligned examples.

## Preregistered gates

| Gate | Partial result | Interpretation |
| --- | --- | --- |
| Composition | **Fail**: median defect/degradation correlation `-.4715` at 2->4 and `-.2651` at 4->8; positive sign in `0/8` and `1/8` checkpoints | Route disagreement is not a positive predictor of MSE degradation in the observed families. The proposed composition-amplification mechanism is unsupported so far. |
| Off-manifold | Incomplete: NFE4 has no degrading-MSE checkpoints; among the two checkpoints that degrade from NFE4->8, NFE8 median relative penalty is `3.591` and `2/2` exceed 20% | Large model-source shift exists in all reported runs, but endpoints still usually improve in MSE. Off-manifold distance alone is not demonstrated to be harmful. |
| Dynamics | Pass under the preregistered relative-tercile statistic, with grid-only and joint K=16 as passing families | This is a useful lead, not a decision. The statistic can pass while absolute risk still improves, so paper-scale paired risk and deployable difficulty routing are required. |

The partial gate output correctly selects `collect_remaining_audits`. No
advantage-aligned training loss is authorized by this snapshot.

## Operational snapshot

- Original overnight and corrected 100k studies: complete.
- Weekend selected training remains active: grid-only seed 1 at 157,501 updates,
  shallow-grid winner at 207,001, composed-source winner at 277,501, and base K=16
  seed 3 at 187,501.
- Group volume: 3.0 TiB total, 43 GiB available (99% used).
- User volume: 50 GiB total, 26 GiB available.
- Experiment checkpoints: 78 latest + 78 best, 19.74 GiB total.

The next stage uses frozen checkpoints and produces only logs/JSON metrics. Starting
another large training sweep before the complete audit would be both scientifically
premature and unnecessary pressure on the nearly full group volume.

## Next experiment: locked advantage frontier

After node B and node D complete the missing audit, evaluate the same mechanism
families with 1000 Push-T videos at NFE `1/2/4/8`. The evaluation records aligned
per-video LPIPS, MSE, final block-vertex error, action-path length, ground-truth pixel
motion, and ground-truth block motion.

The questions are:

1. Does the MSE/pose improvement survive the paper-sized 1000-video protocol?
2. Does LPIPS rank inference depth differently from MSE and block dynamics?
3. Does action path, which is known before generation, predict the paired NFE2
   advantage? Full-episode routing uses mean action displacement per step so episode
   length is not silently used as the difficulty signal.
4. At the same mean compute budget, does sending high-action-path examples to NFE2
   beat random NFE allocation?

The action-routing threshold is the median normalized action-path difficulty on the
first half of the
ordered evaluation set and is applied without refitting to the second half. This is
a diagnostic split, not a final held-out policy claim. A target-derived oracle may be
reported only as an upper bound and never as a deployable method.

Decision:

- If MSE and pose improve robustly, LPIPS worsens, and action-only routing predicts
  the benefit, pursue **risk-conditional inference depth** and decision-level
  planning evaluation.
- If extra NFE improves all risks, focus on calibrating an anytime stopping rule.
- If only MSE improves while pose and planning do not, treat it as a distortion
  artifact rather than a world-model contribution.
- If no paper-scale benefit survives, reject depth scaling on Push-T and move to a
  different task only with a preregistered reason.
