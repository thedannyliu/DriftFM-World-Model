# Node C equal-budget candidate-racing result

Status: **partial, 15/16 rows complete**. This is a Push-T discovery result, not a
cross-task claim.

## Provenance

| Field | Value |
| --- | --- |
| Report time | `2026-07-29T00:23:56+00:00` |
| Host | `run330248-sam4-4` |
| Code commit | `791482410b03a1eda4c86b8abe99708da91174c4` |
| Task / policy | Push-T / Diffusion Policy `ep100` |
| Policy checkpoint format | monolithic `model` + `ema`, non-official loader |
| Proposal RNG base seed | 5 |
| Hardware | one company node, 4xH100 |
| Trials per row | 20 fixed environment test seeds |
| W&B project | `driftfm-unordered-fidelity-company` |
| Shared markers | `/group-volume/danny-dataset/driftworld/checkpoints/experiments/gpc-unordered-{name}.json` |
| Terminal report | `/user-volume/driftworld/logs/unordered-fidelity-node-c-report-20260729-002356.txt` |

The comparison holds the initial candidate set and nominal world-model evaluation
budget at 32:

- fixed: 16 candidates evaluated at NFE 2;
- half racing: 16 candidates at NFE 1, then 8 candidates refined at NFE 2;
- quarter racing: 16 candidates at NFE 1, then 4 candidates refined at NFE 4;
- eighth racing: 16 candidates at NFE 1, then 2 candidates refined at NFE 8.

NFE is an inference-compute setting. Episode maximum IoU is the primary metric.
Success is the fraction reaching IoU 0.95. Confidence intervals in the first table
are unpaired 95% bootstrap intervals for each row; allocator decisions use the paired
deltas below.

## Per-row results

| Family | Allocation | Mean IoU | 95% CI | Success | Mean latency (s) | Winner flip | W&B |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| k1-grid25 | fixed 16x2 | 0.747918 | [0.659932, 0.832520] | 0.20 | 0.763782 | NA | `bqmymhi6` |
| k1-grid25 | race half, 2 | 0.747918 | [0.659932, 0.832520] | 0.20 | 0.812542 | 0.00 | `ze9kerva` |
| k1-grid25 | race quarter, 4 | 0.732044 | [0.651432, 0.812583] | 0.20 | 0.871453 | 0.00 | `xiodjpyd` |
| k1-grid25 | race eighth, 8 | 0.716297 | [0.628080, 0.807049] | 0.25 | 1.072864 | 0.00 | `596wqxb2` |
| k32 | fixed 16x2 | 0.729613 | [0.646991, 0.811082] | 0.10 | 0.764206 | NA | `lmjosrru` |
| k32 | race half, 2 | 0.732504 | [0.648751, 0.815805] | 0.15 | 0.812230 | 0.00 | `xh10xig8` |
| k32 | race quarter, 4 | 0.690665 | [0.602491, 0.780810] | 0.10 | 0.871752 | 0.05 | `3yv3ui8n` |
| k32 | race eighth, 8 | 0.688566 | [0.605139, 0.774168] | 0.20 | 1.079255 | 0.05 | `iebq43u9` |
| joint-k16 | fixed 16x2 | 0.714234 | [0.632865, 0.798067] | 0.20 | 0.763936 | NA | `uhvv9opz` |
| joint-k16 | race half, 2 | 0.713587 | [0.632220, 0.797445] | 0.20 | 0.811640 | 0.00 | `bdfpiczk` |
| joint-k16 | race quarter, 4 | 0.708977 | [0.615710, 0.805378] | 0.30 | 0.869627 | 0.00 | `dlf7nvpy` |
| joint-k16 | race eighth, 8 | **missing** | — | — | — | — | — |
| deep-base-k16 | fixed 16x2 | 0.756721 | [0.676698, 0.833811] | 0.15 | 0.762827 | NA | `6wcqtmza` |
| deep-base-k16 | race half, 2 | 0.756791 | [0.676903, 0.833891] | 0.15 | 0.813302 | 0.00 | `4n7tyalo` |
| deep-base-k16 | race quarter, 4 | 0.725253 | [0.650760, 0.798397] | 0.15 | 0.870546 | 0.00 | `qbeglwn1` |
| deep-base-k16 | race eighth, 8 | 0.734865 | [0.659660, 0.807650] | 0.15 | 1.083215 | 0.05 | `9r2sel37` |

## Paired racing-minus-fixed results

All available comparisons report `candidate_hashes=ready`: fixed and racing rows used
the same candidate proposals and environment seeds. Wins/ties/losses count per-seed
IoU changes relative to fixed 16x2.

| Family | Racing allocation | Mean paired ΔIoU | Paired 95% CI | W/T/L | Δ success | Latency ratio |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| k1-grid25 | half, 2 | 0.000000 | [0.000000, 0.000000] | 0/20/0 | 0.00 | 1.0638 |
| k1-grid25 | quarter, 4 | -0.015874 | [-0.070747, 0.048336] | 2/9/9 | 0.00 | 1.1410 |
| k1-grid25 | eighth, 8 | -0.031620 | [-0.100587, 0.020926] | 4/8/8 | +0.05 | 1.4047 |
| k32 | half, 2 | +0.002891 | [0.000000, 0.008515] | 2/18/0 | +0.05 | 1.0628 |
| k32 | quarter, 4 | -0.038948 | [-0.079869, -0.004638] | 3/11/6 | 0.00 | 1.1407 |
| k32 | eighth, 8 | -0.041047 | [-0.083976, -0.000488] | 4/9/7 | +0.10 | 1.4123 |
| joint-k16 | half, 2 | -0.000646 | [-0.001939, 0.000000] | 0/19/1 | 0.00 | 1.0624 |
| joint-k16 | quarter, 4 | -0.005256 | [-0.047598, 0.037688] | 5/11/4 | +0.10 | 1.1383 |
| joint-k16 | eighth, 8 | **missing** | — | — | — | — |
| deep-base-k16 | half, 2 | +0.000069 | [0.000000, 0.000207] | 1/19/0 | 0.00 | 1.0662 |
| deep-base-k16 | quarter, 4 | -0.031469 | [-0.077961, 0.003410] | 2/15/3 | 0.00 | 1.1412 |
| deep-base-k16 | eighth, 8 | -0.021856 | [-0.051342, 0.003591] | 2/12/6 | 0.00 | 1.4200 |

## What this result says

The current static coarse-to-fine racing baseline does **not** pass the H3 gate. No
available racing arm
shows a material paired improvement over fixed 16x2 with a confidence interval
strictly above zero:

1. Refining half the candidates to NFE 2 is effectively the fixed strategy. It gives
   exactly the same k1 outcomes and changes only 1--2 of 20 trials in the other
   families, while increasing latency by 6.2--6.6%.
2. Concentrating the same nominal budget into NFE 4 or 8 usually lowers mean IoU and
   costs more wall time. For k32, both reductions are statistically separated from
   zero in the paired bootstrap.
3. Coarse-to-final winner flips are zero in most rows and only 0.05 elsewhere.
   Refinement therefore rarely changes the selected action. Under this candidate set,
   added denoising depth has little decision value.
4. Success counts are too coarse at 20 trials to override paired IoU. Several arms
   gain one or two successes while losing mean IoU; these are descriptive signals,
   not evidence of allocator improvement.

This rejects the tested static racing rule: fixed initial breadth of 16, score-based
pruning, and deeper evaluation of a fixed fraction. It does not test a learned or
calibrated gate and does not show that all adaptive allocation is impossible. A
sharper remaining question is
whether cheap NFE 1 should buy **more initial candidates** instead of deeper
refinement. Node B supplies the static breadth/depth evidence needed before designing
that follow-up.

## Limits and next gate

- One row, `c-joint-race-eighth8`, is missing and Node C has no live process. Rerun
  Node C to close the protocol; completed atomic markers should be skipped.
- This is a 20-seed, one-policy, four-checkpoint Push-T discovery experiment.
- Equal nominal function evaluations did not yield equal wall time. Any final
  compute claim must match measured latency as well as model evaluations.
- Node C did not audit ground-truth reward for every candidate. Node A is required to
  decide whether NFE changes ranking correctly, rather than merely whether racing
  changes episode IoU.
- Do not train a learned gate from these rows. First complete Nodes A/B/D and test
  whether static proposal breadth dominates static depth. If it does, preregister a
  breadth-first allocator and evaluate it on held-out seeds and a second task.
