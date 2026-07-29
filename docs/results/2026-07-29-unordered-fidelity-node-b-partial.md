# Node B equal-budget breadth-depth result

Status: **partial, 15/16 rows complete**. This is a Push-T discovery result, not a
cross-task claim.

## Provenance

| Field | Value |
| --- | --- |
| Report time | `2026-07-29T01:06:13+00:00` |
| Host | `run332013-sam4-3` |
| Reporting checkout | `d473ec435e906fe6de7544e7a386c29754cc88d0` |
| Last code-changing ancestor | `791482410b03a1eda4c86b8abe99708da91174c4` |
| Task / policy | Push-T / Diffusion Policy `ep100` |
| Policy checkpoint format | monolithic `model` + `ema`, non-official loader |
| Proposal RNG base seed | 5 |
| Hardware | one company node, 4xH100 |
| Trials per row | 20 fixed environment test seeds |
| W&B project | `driftfm-unordered-fidelity-company` |
| Shared markers | `/group-volume/danny-dataset/driftworld/checkpoints/experiments/gpc-unordered-{name}.json` |
| Terminal report | `/user-volume/driftworld/logs/unordered-fidelity-node-b-report-20260729-010613.txt` |

All arms have nominal budget 64 world-model evaluations:

- breadth: 64 candidates at NFE 1;
- shallow depth: 32 candidates at NFE 2;
- medium depth: 16 candidates at NFE 4;
- deep: 8 candidates at NFE 8.

NFE is an inference-compute setting. Episode maximum IoU is the primary metric.
Success is the fraction reaching IoU 0.95. Row-level intervals are unpaired bootstrap
intervals; allocation comparisons use the paired environment-seed differences below.
Candidate sets necessarily differ across allocations because proposal count changes.

## Per-row results

| Family | Allocation | Mean IoU | 95% CI | Success | Mean / p95 latency (s) | Top-2 margin | W&B |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| k1-grid25 | 64x1 | 0.787247 | [0.700778, 0.869061] | 0.30 | 1.561445 / 1.573017 | 0.098528 | `azw5q4cj` |
| k1-grid25 | 32x2 | 0.758368 | [0.682425, 0.831903] | 0.10 | 1.475170 / 1.479269 | 0.121247 | `qibcg609` |
| k1-grid25 | 16x4 | 0.728289 | [0.647849, 0.809205] | 0.20 | 1.489418 / 1.493463 | 0.144335 | `ggjnwwky` |
| k1-grid25 | 8x8 | 0.681673 | [0.615323, 0.751930] | 0.10 | 1.580711 / 1.585390 | 0.238754 | `upifhipt` |
| k32 | 64x1 | 0.791451 | [0.712286, 0.866454] | 0.25 | 1.561354 / 1.574519 | 0.099390 | `xurcq75i` |
| k32 | 32x2 | 0.752107 | [0.685585, 0.816868] | 0.10 | 1.475810 / 1.481115 | 0.108896 | `ulpao3kj` |
| k32 | 16x4 | 0.704175 | [0.611027, 0.797729] | 0.20 | 1.490004 / 1.493015 | 0.132804 | `wpg8ctup` |
| k32 | 8x8 | 0.720287 | [0.637440, 0.803745] | 0.20 | 1.578769 / 1.585165 | 0.243592 | `kvtlj67t` |
| joint-k16 | 64x1 | 0.758601 | [0.669478, 0.841171] | 0.20 | 1.564972 / 1.571971 | 0.108836 | `qxwg2a7r` |
| joint-k16 | 32x2 | 0.734656 | [0.661017, 0.808506] | 0.15 | 1.473160 / 1.479039 | 0.142100 | `p456ui7g` |
| joint-k16 | 16x4 | **missing** | — | — | — | — | — |
| joint-k16 | 8x8 | 0.694712 | [0.624478, 0.766806] | 0.05 | 1.582308 / 1.585816 | 0.274079 | `55bxdxa3` |
| deep-base-k16 | 64x1 | 0.766125 | [0.684837, 0.839786] | 0.20 | 1.565986 / 1.576701 | 0.129624 | `ycp9g2a9` |
| deep-base-k16 | 32x2 | 0.757932 | [0.685275, 0.826163] | 0.10 | 1.473671 / 1.481576 | 0.138666 | `afgdyi3e` |
| deep-base-k16 | 16x4 | 0.730352 | [0.656615, 0.802092] | 0.15 | 1.488010 / 1.491610 | 0.171435 | `eg3rvzx8` |
| deep-base-k16 | 8x8 | 0.709240 | [0.627968, 0.790497] | 0.10 | 1.578507 / 1.586046 | 0.275615 | `v96vibjx` |

## Paired arm-minus-32x2 results

All comparisons use the same 20 environment test seeds. They are paired by
environment, but not by candidate identity because the allocation changes proposal
count.

| Family | Arm | Mean paired ΔIoU | Paired 95% CI | W/T/L | Δ success | Latency ratio |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| k1-grid25 | 64x1 | +0.028879 | [-0.046525, 0.101423] | 14/0/6 | +0.20 | 1.0585 |
| k1-grid25 | 16x4 | -0.030079 | [-0.140293, 0.079684] | 9/0/11 | +0.10 | 1.0097 |
| k1-grid25 | 8x8 | -0.076694 | [-0.162837, 0.016056] | 5/0/15 | 0.00 | 1.0715 |
| k32 | 64x1 | +0.039344 | [-0.034410, 0.111663] | 14/0/6 | +0.15 | 1.0580 |
| k32 | 16x4 | -0.047932 | [-0.130713, 0.025083] | 9/0/11 | +0.10 | 1.0096 |
| k32 | 8x8 | -0.031820 | [-0.108905, 0.050531] | 7/0/13 | +0.10 | 1.0698 |
| joint-k16 | 64x1 | +0.023945 | [-0.061517, 0.105274] | 11/0/9 | +0.05 | 1.0623 |
| joint-k16 | 16x4 | **missing** | — | — | — | — |
| joint-k16 | 8x8 | -0.039944 | [-0.131969, 0.059290] | 7/0/13 | -0.10 | 1.0741 |
| deep-base-k16 | 64x1 | +0.008193 | [-0.079049, 0.096767] | 9/0/11 | +0.10 | 1.0626 |
| deep-base-k16 | 16x4 | -0.027580 | [-0.113506, 0.063822] | 6/0/14 | +0.05 | 1.0097 |
| deep-base-k16 | 8x8 | -0.048692 | [-0.133275, 0.039774] | 6/0/14 | 0.00 | 1.0711 |

The only individual pairwise comparison whose paired interval excludes zero is
k1-grid25 64x1 versus 8x8: `+0.105574`, 95% CI
`[0.008580,0.191172]`, with wins/ties/losses `15/1/4`.

## Aggregate static frontier

| Allocation | Families available | Mean family IoU | Mean success | Mean latency (s) | Descriptive best count |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64x1 | 4 | 0.775856 | 0.2375 | 1.563439 | 4 |
| 32x2 | 4 | 0.750766 | 0.1125 | 1.474452 | 0 |
| 16x4 | 3 | 0.720939 | 0.1833 | 1.489144 | 0 |
| 8x8 | 4 | 0.701478 | 0.1125 | 1.580074 | 0 |

Across the four selected model families, the descriptive 64x1-minus-32x2 effects
are `+0.00819/+0.02394/+0.02888/+0.03934` IoU. Their mean is `+0.02509`.
The direction repeats in all four families, but these checkpoints are a selected
Push-T discovery set rather than four independent tasks. This consistency is a
follow-up signal, not a confirmatory cross-family significance test.

Only 64x1 and 32x2 remain on every available family-level latency/IoU Pareto frontier.
The 16x4 and 8x8 arms are dominated.

## What this result says

1. **Proposal breadth is the descriptive winner at equal nominal evaluations.**
   The 64x1 arm has the highest mean IoU in all four families. Trading proposals for
   NFE 4 or 8 lowers mean IoU in every available 32x2 comparison.
2. **The usable static frontier is a latency-quality choice.** Relative to 32x2,
   64x1 gains a mean `0.02509` IoU across the four selected families while taking
   5.8--6.3% longer. The 16x4 arm is only about 1% slower but worse; 8x8 is about 7%
   slower and worse.
3. **Nominal evaluations are not a wall-time match.** The p95/mean latency ratios are
   only 1.002--1.008, so the allocation-dependent overhead is stable rather than
   caused by a few tail-latency outliers.
4. **Raw top-two margin is allocation-dependent and anti-calibrated here.** Mean
   margin rises from `0.1091` at 64x1 to `0.1277/0.1495/0.2580` at
   32x2/16x4/8x8 while mean IoU falls. Across the 15 rows, the descriptive Pearson
   correlation between margin and IoU is `-0.799`. Fewer candidates mechanically
   change score order statistics, so an unnormalized margin cannot serve as a
   cross-allocation confidence gate.
5. **This weakens the case for the current refinement story.** Together with Nodes C
   and D, the result says that spending test-time compute on more proposals is a
   stronger baseline than static deeper refinement. Any learned allocator must beat
   both 64x1 and 32x2 at matched measured latency, not only fixed 16x2 at matched
   nominal evaluations.

## W&B audit and limits

The 15 listed W&B IDs were read on 2026-07-28 EDT using the recorded PACE Python
3.12 environment and existing W&B credentials. All 15 are `finished`, and their
summary IoU, success, latency, and margin values agree with the shared markers.
Each run contains one aggregate history row and only sparse configuration metadata;
W&B does not contain per-candidate simulator outcomes for this node.

- `b-joint-budget-16x4` is missing and Node B has no live process. Complete it for
  protocol closure.
- Four selected checkpoints are not independent task replications, and 20 trials
  leave every 64x1-versus-32x2 family-level interval crossing zero.
- Node B changes candidate count together with NFE by design. It identifies a
  planning allocation frontier, not the isolated causal effect of inference depth.
- Node A remains necessary to hold candidates fixed and measure ground-truth
  candidate ranking, oracle selection, and regret.
- A paper-level breadth-first claim requires a held-out protocol, latency-matched
  budgets, stronger statistics, and at least one additional task. Best-of-N proposal
  breadth alone is not novel.
