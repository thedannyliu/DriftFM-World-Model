# All-experiments result snapshot — 2026-07-28 18:27 UTC

## Scope and evidence status

Source: the output of the shared status reporters at commit `d797a0f`, captured
2026-07-28 18:27:59 UTC on `run330253-sam4-1`.

- Task: Push-T, initialized from the released DriftWorld step-1,180,500 parent.
- Shared artifacts: `/group-volume/danny-dataset/driftworld`.
- Shared runtime logs and raw evaluation metrics: `/user-volume/driftworld`.
- Corrected long-study `LOCKED` evaluations use 100 videos.
- Weekend milestone evaluations use 25 videos.
- Advantage-aligned mechanism audits use 64 fixed validation chunks and four fixed
  particles. These are diagnostic risks, not paper-test metrics.
- Locked advantage-frontier evaluations use 1000 videos and NFE `1/2/4/8`.
- All metric tuples below follow NFE order `1/2/4/8`; lower is better.

The original overnight study and all four corrected 100k studies are complete. The
aggressive weekend queues and the locked advantage frontier are still running. This
snapshot therefore separates completed evidence, provisional evidence, and invalid
diagnostics explicitly.

## Result-level decisions

1. **The original composition-amplification mechanism is rejected by its
   preregistered gate.** Across 12 audited checkpoints, route-defect/degradation
   correlations are negative, not positive. Even if all four missing node-D audits
   were positive, the positive fractions could reach only `4/16=25%` and
   `5/16=31.25%`, below the required 75%.
2. **Model-composed sources are off manifold, but that shift is not the observed
   cause of degradation.** The off-manifold gate passes while the defect correlation
   gate fails. Existence of a shift is not evidence that it causes rollout error.
3. **A finite two-step operating point is the strongest current lead.** On the
   completed 1000-video K=1/grid=.25 checkpoints, NFE2 improves LPIPS for all three
   latest seeds and modestly improves mean MSE and block error. NFE4/8 then lose
   distortion and dynamics quality. This is a reproducible finite-depth optimum, not
   monotonic anytime scaling.
4. **Very deep post-training is consistently harmful.** Weekend winner checkpoints
   at 300k–400k are much worse than their rollout-selected 30k–60k checkpoints.
   Training duration and rollout-aware selection are part of the method.
5. **Action-routing results are not yet interpretable.** All eight completed frontier
   markers report exactly zero routing change. This is consistent with the original
   ordered first-half/second-half split selecting only one NFE on the test half, but
   the old reporter omitted the selected fraction. The 1000-video model metrics
   remain valid; the routing diagnostic must expose its fraction and be recomputed
   from existing raw metrics.

## Aggressive weekend queues

The complete 30k response surfaces are retained in
[`2026-07-27-all-experiments-status.md`](2026-07-27-all-experiments-status.md).
This table records every arm whose state changed in the 18:27 snapshot.

| Queue / arm | State at capture | Latest evaluated result | Rollout-selected best | Decision |
| --- | --- | --- | --- | --- |
| A: grid-only `.25`, seed 1 | 340,501/400k updates; evaluation at 300k | LPIPS `.018165/.024600/.031385/.031805`; vertex `1.02069/.655874/.785541/.836457` | 30k, score `.0300652` | Deep training collapsed; seeds 2–5 had not started. |
| A: grid `.50` + source `.10`, seed 1 | 30k/200k; selected runner-up waiting behind winner | LPIPS `.004727/.004251/.004860/.005409`; vertex `.119427/.112492/.112404/.112673` | 30k, score `.0320604` | Promising early control, but no deep or cross-seed evidence yet. |
| B: grid-max NFE2, probability `.125`, seed 1 | 342,501/400k; evaluation at 300k | LPIPS `.014924/.014618/.020920/.028952`; vertex `.601098/.177629/.197517/.497634` | 60k, score `.0338872` | The early shallow-grid advantage does not survive 300k; seeds 2–5 had not started. |
| B: grid-max NFE8, probability `.50`, seed 1 | 30k/200k; selected runner-up waiting | LPIPS `.007454/.004427/.004346/.004410`; vertex `.419118/.113210/.114162/.114480` | 30k, score `.0400455` | Useful early depth control; replication remains absent. |
| C: four-step composed source, replay `.10`, seed 1 | **complete**, 400k | LPIPS `.017741/.025803/.032588/.028633`; vertex `1.16684/.646810/1.04781/.744748` | 60k, score `.0344468` | Deep endpoint has collapsed; seeds 2–5 had not started. |
| C: four-step composed source, replay `.50`, seed 1 | 30k/200k; selected runner-up waiting | LPIPS `.007287/.006972/.005033/.005153`; vertex `.559295/.423847/.117215/.116797` | 30k, score `.0497364` | No evidence yet that the runner-up survives depth or seeds. |
| D: corrected base K=16, seed 1 | complete, 300k | LPIPS `.007522/.007506/.012112/.014134`; vertex `.145167/.159705/.403151/.601513` | 30k, score `.034830` | Deep training worsens NFE4/8. |
| D: corrected base K=16, seed 2 | complete, 300k | LPIPS `.006029/.006943/.011440/.012895`; vertex `.136848/.111026/.393817/.556936` | 30k, score `.035044` | Independent confirmation of deep NFE4/8 regression. |
| D: corrected base K=16, seed 3 | complete, 300k | LPIPS `.008803/.010407/.011256/.013872`; vertex `.552102/.537862/.547326/.855108` | 30k, score `.0363439` | Third seed confirms that more post-training is not reliably better. |
| D: grid-only K=16, seed 1 | 60,501/300k; evaluation at 60k | LPIPS `.008472/.008129/.009095/.009439`; vertex `.535990/.524377/.534351/.532717` | 30k, score `.0300652` | Already worse than the selected 30k point; two seeds and all remaining causal arms were waiting. |

At capture time, queues A and B were still training their first selected seed, queue
C had just moved beyond its completed 400k winner, and queue D had started the
grid-only control. The waiting rows are planned work, not negative results.

## Advantage-aligned mechanism audit

### Per-checkpoint results

Node A, B, and C completed 12 of 16 preregistered audits. Node D remained missing.

| Checkpoint | Family | Step | MSE 1/2/4/8 | Block-vertex 1/2/4/8 | Defect corr. 2→4 / 4→8 | Off-manifold penalty 4/8 | W&B |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| A K1-grid25 s1 latest | K1 grid=.25 | 99,999 | `.000700/.000520/.000482/.000468` | `.030520/.027397/.027261/.027739` | `-.5562/-.1169` | `1.705/5.426` | `eaot843a` |
| A K1-grid25 s2 latest | K1 grid=.25 | 99,999 | `.000710/.000530/.000495/.000484` | `.029502/.026061/.025340/.025492` | `-.4582/+.0589` | `1.755/5.547` | `nao453jn` |
| A K1-grid25 s3 latest | K1 grid=.25 | 99,999 | `.000712/.000523/.000483/.000470` | `.030832/.027047/.025747/.025342` | `-.5183/-.0495` | `1.770/5.591` | `m5g0d6r5` |
| A weekend grid-only s1 best | Grid-only K16 | 29,999 | `.000423/.000391/.000387/.000388` | `.023535/.022837/.023379/.023692` | `-.5059/-.3557` | `.996/3.679` | `o0ppltrm` |
| B K32 s1 latest | K32 | 99,999 | `.000418/.000394/.000398/.000406` | `.023648/.024262/.025528/.026106` | `-.1776/-.1163` | `.988/3.643` | `mdy6rp68` |
| B K32 s2 latest | K32 | 99,999 | `.000412/.000393/.000392/.000394` | `.023805/.023994/.024473/.024933` | `-.4788/-.4296` | `.978/3.624` | `h09g7bel` |
| B K32 s3 latest | K32 | 99,999 | `.000422/.000398/.000397/.000401` | `.023680/.023693/.024037/.024296` | `-.4143/-.3104` | `.996/3.671` | `eznzwdli` |
| B weekend base K16 s1 best | Base K16 | 29,999 | `.000412/.000399/.000403/.000409` | `.023519/.024589/.025550/.025975` | `-.4216/-.2905` | `.929/3.491` | `lj65waut` |
| C joint K16 s1 latest | Grid/source K16 | 99,999 | `.000453/.000398/.000388/.000387` | `.025085/.024061/.023973/.023770` | `-.4526/-.4100` | `1.149/4.060` | `qp2bzkpa` |
| C joint K16 s2 latest | Grid/source K16 | 99,999 | `.000458/.000401/.000391/.000391` | `.025432/.024356/.024261/.024508` | `-.4848/-.4273` | `1.153/4.070` | `2xky5d1g` |
| C joint K16 s3 latest | Grid/source K16 | 99,999 | `.000459/.000399/.000388/.000388` | `.023546/.021916/.021747/.021808` | `-.4388/-.3332` | `1.157/4.084` | `19bkdgjs` |
| C weekend source-only s1 best | Source-only K16 | 29,999 | `.000420/.000400/.000398/.000400` | `.025631/.025943/.026442/.026681` | `-.4341/-.1971` | `.928/3.504` | `liuxfgx6` |

### Aggregate diagnostic signal

| Risk | Mean NFE 1/2/4/8 | Change at NFE2/4/8 vs NFE1 | Adjacent improvement count |
| --- | --- | --- | --- |
| Pixel MSE | `.00049983/.00042885/.00041686/.00041536` | `-14.20%/-16.60%/-16.90%` | `12/12`, `10/12`, `6/12` |
| Block-vertex error | `.025728/.024680/.024811/.025029` | `-4.07%/-3.56%/-2.72%` | `7/12`, `6/12`, `2/12` |

Preregistered gate state:

| Gate | 12-run observation | Decision |
| --- | --- | --- |
| Composition amplification | Median defect/degradation correlation `-.4554` at 2→4 and `-.3004` at 4→8; positive sign in `0/12` and `1/12` checkpoints | **Rejected.** The four missing audits cannot reach the required 75% positive fraction. |
| Model-source shift | Among checkpoints whose later transition degrades, median relative penalties are `.958` at NFE4 and `3.633` at NFE8; every included case exceeds 20% | Shift-existence gate passes provisionally, but it does not explain degradation because defect correlation has the opposite sign. |
| Motion concentration | The preregistered relative-tercile gate passes in grid-only K16, joint K16, and K32 | A viable lead, not yet a method or causal result. Absolute risk and paper-scale paired outcomes still govern. |
| Protocol coverage | 12/16; node D missing | Partial. Complete node D for a closed protocol record, even though the composition decision is already mathematically fixed. |

The status JSON incorrectly labeled 12 runs as complete because the summarizer used
12 rather than the preregistered 16 as its coverage threshold. That reporting bug
does not change any metric or gate calculation and is corrected after this snapshot.

## Locked 1000-video advantage frontier

### Completed checkpoint results

Node A completed eight of its eight frozen checkpoints. Node B had not started its
eight checkpoints. These are the first paper-sized paired measurements in this
research cycle.

| Checkpoint | Family | LPIPS 1/2/4/8 | MSE 1/2/4/8 | Block-vertex 1/2/4/8 | W&B |
| --- | --- | --- | --- | --- | --- |
| K1 grid=.25 s1 latest | K1 grid=.25 | `.017655/.011922/.013129/.014693` | `.006635/.006355/.007106/.007939` | `.792254/.739268/.751226/.781697` | `jnwjr70t` |
| K1 grid=.25 s2 latest | K1 grid=.25 | `.019844/.014110/.015616/.017828` | `.007296/.007202/.007964/.009029` | `.883054/.904366/.875726/.921729` | `syxu74v4` |
| K1 grid=.25 s3 latest | K1 grid=.25 | `.019586/.012810/.014425/.016179` | `.007165/.006779/.007605/.008486` | `.841978/.824272/.846778/.844312` | `rysbem5u` |
| K1 grid=.25 s1 validation-best | K1 validation-best | `.016974/.011652/.012817/.014318` | `.006450/.006214/.006830/.007646` | `.772039/.725317/.730966/.828820` | `4j9ancnf` |
| K1 grid=.25 s2 validation-best | K1 validation-best | `.011448/.011562/.012090/.012275` | `.005615/.005772/.006011/.006157` | `.680179/.709716/.774272/.786230` | `andiirbi` |
| K1 grid=.25 s3 validation-best | K1 validation-best | `.011818/.011248/.011803/.012109` | `.005748/.005621/.005870/.005997` | `.718701/.681202/.743111/.746368` | `fr0plzni` |
| K32 s1 latest | K32 | `.011188/.010805/.011401/.011886` | `.005594/.005447/.005774/.006082` | `.671438/.669440/.705747/.702446` | `spjgkt9f` |
| K32 s2 latest | K32 | `.011959/.012232/.012879/.013401` | `.005805/.005883/.006201/.006491` | `.759421/.781121/.789236/.792589` | `a9zro2z0` |

### Cross-seed frontier signal

| Family / checkpoint rule | Coverage | Mean LPIPS 1/2/4/8 | Mean MSE 1/2/4/8 | Mean vertex 1/2/4/8 | Interpretation |
| --- | --- | --- | --- | --- | --- |
| K1 grid=.25 latest | 3/3 | `.019028/.012947/.014390/.016233` | `.007032/.006779/.007558/.008485` | `.839095/.822635/.824577/.849246` | NFE2 changes are `-31.96%` LPIPS, `-3.61%` MSE, and `-1.96%` vertex. All three seeds improve LPIPS and MSE; two of three improve vertex. NFE4/8 are not monotonic. |
| K1 grid=.25 validation-best | 3/3 | `.013413/.011487/.012236/.012901` | `.005938/.005869/.006237/.006600` | `.723640/.705412/.749450/.787139` | NFE2 changes are `-14.36%`, `-1.16%`, and `-2.52%`; two of three seeds improve each risk. Validation-best still does not select an anytime-monotonic model. |
| K32 latest | 2/3 | `.011573/.011518/.012140/.012643` | `.005700/.005665/.005988/.006287` | `.715430/.725281/.747492/.747518` | NFE2 is nearly flat (`-.48%` LPIPS, `-.61%` MSE) and worsens vertex `+1.38%`; only one of two seeds improves at NFE2. This row is partial until seed 3 finishes. |

The repository records the DriftWorld paper's full-rollout LPIPS as `.0146`. K1
latest mean NFE2 is 11.3% lower, K1 validation-best mean NFE2 is 21.3% lower, and
the partial K32 mean NFE2 is 21.1% lower. These are **not yet superiority claims**:
the official checkpoint must be re-evaluated in the same current code path, with
paired examples, identical preprocessing, bootstrap confidence intervals, latency,
and downstream planning metrics.

### Invalid routing field

Every completed marker reported `route_mse=0.0` and `route_vertex=0.0`. An all-NFE1
or all-NFE2 decision would make adaptive and random allocation algebraically
identical, and the ordered first-half/second-half split makes that failure plausible.
The old status output did not include `nfe2_fraction`, so this snapshot cannot prove
which degeneracy occurred. In either case, an exactly zero result on all eight
checkpoints is not usable evidence that action difficulty is useless or that routing
succeeds.

The corrective analysis uses deterministic even-index development examples and
odd-index test examples to interleave the ordered evaluation sequence. It reuses
the raw 1000-video JSON and the same W&B run ID; no GPU evaluation or checkpoint
mutation is required. Because this split was introduced after observing the
degeneracy, routing remains exploratory and requires a later preregistered
replication.

## Operational state

- Live on the capture node: queue A plus one four-process DDP training job.
- The additional low-CPU Python entries are consistent with worker processes; the
  four rank processes were each using one GPU.
- Recent shared logs showed active training and milestone evaluation on queues A–D;
  no failure marker was reported.
- Checkpoints: 79 latest + 79 best, 158 total, 19.99 GiB.
- Completed markers: 255 training and 382/382 evaluation markers.
- Group volume: 3.0 TiB total, 529 GiB free (83% used).
- User volume: 50 GiB total, 23 GiB free (55% used).

## Next decision sequence

1. Let the currently running weekend jobs finish their in-progress milestone; do not
   interpret 300k/400k latest checkpoints as winners over their 30k/60k rollout best.
2. Complete the missing node-D mechanism audit for protocol closure.
3. Complete node-B's eight locked1000 checkpoints and recompute node-A routing from
   existing raw metrics with the interleaved split.
4. Re-evaluate the official DriftWorld checkpoint on the identical locked1000
   examples and code path.
5. Bootstrap paired NFE1→2 effects for LPIPS, MSE, and block error; report latency
   and the fraction of examples improved.
6. If the K1 NFE2 result survives, frame the next hypothesis as **finite-depth
   transport regularization**: exact-grid supervision creates a useful two-step
   coarse solver, but repeated application accumulates model and perceptual error.
   Test a minimal method that explicitly targets the two-step operating point.
7. Do not claim an anytime world model or start advantage-gated training from the
   rejected composition mechanism. A top-venue result still needs task-level
   planning/policy gains and at least one additional environment.
