# Node D ep300 policy-shift result

Status: **partial, 15/16 rows complete**. This is a Push-T policy-shift discovery
result, not a cross-task claim.

## Provenance

| Field | Value |
| --- | --- |
| Report time | `2026-07-29T00:34:58+00:00` |
| Host | `run330253-sam4-1` |
| Reporting checkout | `2ba2dc5f4f4f33cf7cf9b3033ddab61c78dca03e` |
| Last code-changing ancestor | `791482410b03a1eda4c86b8abe99708da91174c4` |
| Task / policy | Push-T / Diffusion Policy `ep300` |
| Policy checkpoint format | monolithic `model` + `ema`, non-official loader |
| Proposal RNG base seed | 5 |
| Hardware | one company node, 4xH100 |
| Trials per row | 20 fixed environment test seeds |
| W&B project | `driftfm-unordered-fidelity-company` |
| Shared markers | `/group-volume/danny-dataset/driftworld/checkpoints/experiments/gpc-unordered-{name}.json` |
| Terminal report | `/user-volume/driftworld/logs/unordered-fidelity-node-d-report-20260729-003458.txt` |

All arms have nominal budget 32:

- breadth: 32 candidates at NFE 1;
- fixed reference: 16 candidates at NFE 2;
- depth: 8 candidates at NFE 4;
- static racing: 16 candidates at NFE 1, then the top quarter at NFE 4.

NFE is an inference-compute setting. Episode maximum IoU is the primary metric.
Success is the fraction reaching IoU 0.95. Row-level intervals are unpaired bootstrap
intervals; decisions use the paired environment-seed differences below.

## Per-row results

| Family | Allocation | Mean IoU | 95% CI | Success | Mean latency (s) | Winner flip | W&B |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| k1-grid25 | 32x1 breadth | 0.577634 | [0.506351, 0.650342] | 0.00 | 0.778714 | NA | `e78hc2uk` |
| k1-grid25 | 16x2 fixed | 0.586257 | [0.519442, 0.654728] | 0.00 | 0.766502 | NA | `yeoh59g2` |
| k1-grid25 | 8x4 depth | 0.555575 | [0.491270, 0.621050] | 0.00 | 0.796130 | NA | `wjcfm0hw` |
| k1-grid25 | race quarter, 4 | 0.586740 | [0.520332, 0.655671] | 0.00 | 0.871279 | 0.05 | `4bzy8egs` |
| k32 | 32x1 breadth | 0.581510 | [0.509957, 0.654020] | 0.00 | 0.783043 | NA | `qoewjec5` |
| k32 | 16x2 fixed | 0.592603 | [0.522411, 0.664567] | 0.05 | 0.765446 | NA | `7npd0lfg` |
| k32 | 8x4 depth | 0.523690 | [0.457639, 0.591597] | 0.00 | 0.796807 | NA | `7avjsa4a` |
| k32 | race quarter, 4 | 0.578658 | [0.517083, 0.639733] | 0.00 | 0.866658 | 0.10 | `ynbegbid` |
| joint-k16 | 32x1 breadth | 0.601076 | [0.530005, 0.669303] | 0.00 | 0.777885 | NA | `7fzf23lc` |
| joint-k16 | 16x2 fixed | 0.593712 | [0.520037, 0.668164] | 0.00 | 0.763729 | NA | `4h2ode97` |
| joint-k16 | 8x4 depth | 0.565124 | [0.504267, 0.629707] | 0.05 | 0.801953 | NA | `9ji11grj` |
| joint-k16 | race quarter, 4 | 0.578333 | [0.493692, 0.665752] | 0.05 | 0.880662 | 0.10 | `d7z43okl` |
| deep-base-k16 | 32x1 breadth | **missing** | — | — | — | — | — |
| deep-base-k16 | 16x2 fixed | 0.582549 | [0.521972, 0.640028] | 0.00 | 0.764485 | NA | `ulxpurz8` |
| deep-base-k16 | 8x4 depth | 0.569232 | [0.506010, 0.634071] | 0.00 | 0.801332 | NA | `hwoabqrv` |
| deep-base-k16 | race quarter, 4 | 0.574270 | [0.514528, 0.633871] | 0.00 | 0.874387 | 0.15 | `93wvfkgi` |

## Paired ep300 arm-minus-16x2 results

All comparisons are paired by the same 20 environment test seeds. The racing rows
also report `candidate_pairing=ready`, so their initial policy proposals exactly
match the corresponding fixed 16x2 row. Static breadth/depth rows deliberately use
different proposal counts and report `candidate_pairing=different`.

| Family | Arm | Mean paired ΔIoU | Paired 95% CI | W/T/L | Δ success | Latency ratio |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| k1-grid25 | 32x1 breadth | -0.008622 | [-0.063404, 0.045236] | 7/1/12 | 0.00 | 1.0159 |
| k1-grid25 | 8x4 depth | -0.030682 | [-0.093137, 0.028460] | 9/1/10 | 0.00 | 1.0387 |
| k1-grid25 | race quarter, 4 | +0.000484 | [-0.034246, 0.031430] | 6/7/7 | 0.00 | 1.1367 |
| k32 | 32x1 breadth | -0.011092 | [-0.061221, 0.036000] | 8/1/11 | -0.05 | 1.0230 |
| k32 | 8x4 depth | -0.068913 | [-0.144128, -0.002933] | 7/1/12 | -0.05 | 1.0410 |
| k32 | race quarter, 4 | -0.013945 | [-0.047737, 0.007383] | 6/7/7 | -0.05 | 1.1322 |
| joint-k16 | 32x1 breadth | +0.007364 | [-0.054171, 0.070579] | 9/1/10 | 0.00 | 1.0185 |
| joint-k16 | 8x4 depth | -0.028588 | [-0.107710, 0.048824] | 10/1/9 | +0.05 | 1.0500 |
| joint-k16 | race quarter, 4 | -0.015379 | [-0.090133, 0.054898] | 6/6/8 | +0.05 | 1.1531 |
| deep-base-k16 | 32x1 breadth | **missing** | — | — | — | — |
| deep-base-k16 | 8x4 depth | -0.013318 | [-0.067637, 0.036558] | 10/1/9 | 0.00 | 1.0482 |
| deep-base-k16 | race quarter, 4 | -0.008280 | [-0.036283, 0.014386] | 3/10/7 | 0.00 | 1.1438 |

## Ep100-to-ep300 policy interactions

The interaction is `(ep300 arm - ep300 fixed) - (ep100 arm - ep100 fixed)`,
paired by environment index. It tests whether the relative allocation effect changes
with the proposal policy; it is not a comparison of identical candidate sets across
policies.

| Family | Comparison | ep100 ΔIoU | ep300 ΔIoU | Interaction | Interaction 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| k1-grid25 | breadth 32x1 vs fixed 16x2 | -0.064276 | -0.008622 | +0.055654 | [-0.062586, 0.169623] |
| k1-grid25 | race quarter vs fixed | -0.015874 | +0.000484 | +0.016357 | [-0.053500, 0.081723] |
| k32 | breadth 32x1 vs fixed 16x2 | +0.038139 | -0.011092 | -0.049232 | [-0.184733, 0.069510] |
| k32 | race quarter vs fixed | -0.038948 | -0.013945 | +0.025003 | [-0.014857, 0.067587] |
| joint-k16 | breadth 32x1 vs fixed 16x2 | -0.004087 | +0.007364 | +0.011451 | [-0.094518, 0.128711] |
| joint-k16 | race quarter vs fixed | -0.005256 | -0.015379 | -0.010123 | [-0.086132, 0.057452] |
| deep-base-k16 | breadth 32x1 vs fixed 16x2 | **missing** | **missing** | — | — |
| deep-base-k16 | race quarter vs fixed | -0.031469 | -0.008280 | +0.023189 | [-0.022105, 0.075490] |

## What this result says

Node D reproduces the lack of value from static deeper-on-fewer-candidates racing
under the ep300 proposal policy:

1. Racing never materially beats fixed 16x2. Its paired ΔIoU is
   `+0.00048/-0.01395/-0.01538/-0.00828` for k1/k32/joint/deep, every interval
   includes zero, and it costs 13.2--15.3% more latency.
2. Trading breadth for NFE 4 lowers mean IoU in all four families. K32 is the only
   individually conclusive row at 20 trials (`-0.06891`, 95% CI
   `[-0.14413,-0.00293]`); the direction is nevertheless consistent.
3. Breadth 32x1 versus fixed 16x2 is mixed and small for the three completed
   families. No interval excludes zero. The missing deep breadth row prevents a
   complete four-family decision.
4. Racing changes the coarse winner in 5--15% of ep300 trials, more often than the
   0--5% observed with ep100, but these changes do not improve average IoU.
5. Every available ep100-to-ep300 interaction interval includes zero. There is no
   evidence that proposal-policy maturity reverses the allocation conclusion,
   although 20 trials provide weak power for interaction tests.

The absolute fixed-16x2 means are also lower with ep300 than ep100 by
`0.162/0.137/0.121/0.174` IoU for k1/k32/joint/deep. This is a descriptive
distribution-shift signal: it does not by itself establish that the ep300 policy is
intrinsically worse, because the proposal sets differ.

## Limits and next gate

- `d-deep-ep300-32x1` is missing and Node D has no live process. Complete it for
  protocol closure.
- This wave uses 20 Push-T seeds, one ep300 policy checkpoint, and four selected
  world-model checkpoints.
- Node D changes candidate count together with NFE in its static arms. It evaluates
  allocation, not isolated NFE monotonicity; Node A remains the H1 experiment.
- Node D did not audit every candidate against ground truth. Node A is required for
  rank-correlation and selection-regret claims.
- The tested static racing rule now fails under both ep100 and ep300. Do not train a
  learned allocator unless Nodes A/B reveal a repeatable, measurable decision signal
  that a held-out gate could exploit.
