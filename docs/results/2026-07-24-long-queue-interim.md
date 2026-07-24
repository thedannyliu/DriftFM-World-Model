# Corrected Drift Flow long-queue interim report — 2026-07-24

## Scope and comparison rules

This is an interim snapshot from `company/status_long_research.py all`. All four
queues are still training. The screen and milestone evaluations use the first 25
videos; the completed `LOCKED` evaluations use 100 videos. All entries below are
full-rollout metrics and lower is better. Slash-separated values are NFE
`1/2/4`, or `1/2/4/8` where stated.

The task is Push-T. Runs start from the released DriftWorld checkpoint and use four
independent 4xH100 company nodes. Checkpoints are under
`/group-volume/danny-dataset/driftworld/checkpoints/experiments`; the captured
terminal report is `/user-volume/driftworld/logs/long-research-status.txt`.
Training metrics are logged online to the W&B IDs recorded below.

The released DriftWorld paper reports full-rollout LPIPS `.0146` over 1000 seeds.
That number, the 10-video company smoke, the 25-video development evaluations, and
the 100-video locked evaluations are not treated as directly interchangeable.
Claims against DriftWorld remain gated on the same locked evaluation set.

`latest` is the checkpoint at the stated training step. `best` is selected by the
adaptation-validation loss, not by rollout quality. The results show that this
selection criterion is often misaligned with the research objective.

## Result summary

The endpoint-normalized parameterization has converted the earlier catastrophic
NFE2/4 failure into a real quality-versus-compute question: several runs now improve
as NFE increases. The improvement is not reliably monotonic in training duration.
Exact-grid supervision is the most repeatable useful ingredient, while positive
particle count and source replay help only in particular interactions.

The strongest current candidate is K=16 with 25% exact-grid replay and 25% composed
source replay. Its seed-1 30k checkpoint improves full LPIPS from `.00751657` at
NFE1 to `.00508455` at NFE4 (-32.36%) and block-vertex error from `.560028` to
`.115207` (-79.43%). Seed 2 supports the direction at 10k; seed 3 is approximately
flat in LPIPS but improves block-state error. Seeds 2 and 3 reaching 30k are the
next decisive test.

### A — Does K=1 time-pair coverage expose useful NFE scaling?

All rows are seed-1 10k screens. The selected score combines rollout quality and
block-state behavior; it is not LPIPS alone.

| Variant | Score | Latest full LPIPS 1/2/4 | Latest vertex 1/2/4 | Best full LPIPS 1/2/4 |
| --- | ---: | --- | --- | --- |
| Base endpoint replay .25 | .0579782 | .00682948/.00822882/.00904144 | .423030/.552487/.570005 | .00525905/.00788924/.00845592 |
| No endpoint replay | .0536666 | .00653491/.00698792/.00842077 | .182527/.380405/.587078 | .00889087/.00728811/.00751394 |
| Endpoint replay .50 | .0533048 | .00622166/.00695194/.00788355 | .133745/.556654/.571191 | .00956130/.00818384/.00854202 |
| Exact-grid replay .25 | **.0376506** | .00639854/.00652596/.00505630 | .357236/.396820/.111185 | .00831935/.00800479/.00815957 |
| Uniform time sampling | .0382513 | .00788111/.00462676/.00505835 | .484944/.115725/.108467 | .00836677/.00836005/.00880519 |

Exact-grid replay improves NFE4 LPIPS by 20.98% and vertex error by 68.88%
relative to NFE1. Uniform sampling is also strong at NFE2/4, but its NFE1 quality is
worse. The preregistered selector therefore chose exact-grid replay.

### B — Does increasing positive-particle count systematically help?

All rows are otherwise matched seed-1 10k screens.

| Positive particles | Score | Latest full LPIPS 1/2/4 | Latest vertex 1/2/4 | Best full LPIPS 1/2/4 |
| ---: | ---: | --- | --- | --- |
| 2 | .0560905 | .00682241/.00757780/.00895209 | .414683/.358080/.564785 | .00641672/.00608587/.00546967 |
| 4 | .0476519 | .00696535/.00751177/.00820137 | .405322/.361535/.368163 | .00610094/.00866387/.00886866 |
| 8 | .0484318 | .00433441/.00767057/.00818773 | .119157/.405499/.403310 | .00609399/.00861653/.00880794 |
| 16 | .0545669 | .00830731/.00760927/.00780049 | .362959/.550362/.555214 | .00934825/.00740186/.00801958 |
| 32 | **.0323269** | .00692703/.00547343/.00556589 | .200561/.148489/.157407 | .00692703/.00547343/.00556589 |

K=32 improves NFE2/4 LPIPS by 20.98%/19.65% relative to NFE1 in this screen.
The K sweep is not monotonic: K=8 has the best NFE1 but poor composition, and K=16
does not lie between K=8 and K=32. This is evidence for a useful high-K regime, not
yet evidence for a smooth particle-scaling law.

### C — Is composed-source mismatch the remaining error?

All rows are seed-1 10k screens. `sr` is the fraction of detached EMA-composed
source-replay batches.

| Variant | Score | Latest full LPIPS 1/2/4 | Latest vertex 1/2/4 | Best full LPIPS 1/2/4 |
| --- | ---: | --- | --- | --- |
| K=1, sr=.10 | .0557736 | .00890267/.00765039/.00782028 | .551552/.554565/.552088 | .00877905/.00756347/.00773304 |
| K=1, sr=.25 | .0558610 | .00891588/.00770154/.00784262 | .551706/.555739/.553074 | .00651503/.00728500/.00741295 |
| K=16, sr=.10 | .0579082 | .00890941/.00759799/.00842629 | .549266/.560152/.560233 | .00890941/.00759799/.00842629 |
| K=16, sr=.25 | .0565061 | .00797380/.00769620/.00854847 | .355432/.565223/.565235 | .00797380/.00769620/.00854847 |
| K=16, sr=.50 | .0575355 | .00887124/.00760293/.00830536 | .549180/.562186/.561424 | .00887124/.00760293/.00830536 |
| K=16, grid=.25, sr=.25 | **.0444091** | .00743297/.00667098/.00656359 | .371558/.407444/.407805 | .00593166/.00508356/.00559258 |

Source replay alone does not improve the selection score materially. Its useful
effect appears only with exact-grid replay. The isolated “analytic source versus
composed source” hypothesis is therefore not supported; the grid/source interaction
is the promising mechanism.

### D — Does warmup or learning rate improve specialization?

All rows are K=16 seed-1 10k screens.

| Variant | Score | Latest full LPIPS 1/2/4 | Latest vertex 1/2/4 | Best full LPIPS 1/2/4 |
| --- | ---: | --- | --- | --- |
| Endpoint warmup 1k | .0548926 | .00830258/.00825130/.00785266 | .426429/.551069/.555758 | .00806820/.00787409/.00810181 |
| Endpoint warmup 3k | .0518610 | .00596145/.00783632/.00775945 | .116258/.357790/.567156 | .00450769/.00784698/.00844258 |
| Learning rate ×.5 | .0535339 | .00597088/.00725600/.00804186 | .115867/.549310/.555309 | .00945762/.00748995/.00817092 |
| Learning rate ×2 | **.0275566** | .00565647/.00521859/.00553692 | .129531/.114689/.111399 | .00603984/.00756818/.00793487 |
| Exact-grid replay .50 | .0370930 | .00827988/.00587899/.00484901 | .371036/.120690/.117761 | .00940231/.00721576/.00733464 |

Warmup and the lower learning rate do not restore NFE scaling. The doubled learning
rate wins the balanced selector, but exact-grid .50 gives a much larger seed-1 NFE4
gain: 41.44% lower LPIPS and 68.26% lower vertex error than NFE1. This also shows
that a single aggregate selector can hide a strong NFE4 research candidate.

## Do the selected variants replicate and survive deep training?

Milestone rows use 25 videos. Only the two completed seed-1 `locked100` rows use
100 videos. `updates` records the live queue position, while `step` is the latest
evaluated milestone.

| Family | Seed | Updates / evaluated step | Latest LPIPS | Latest vertex | Best-through-step LPIPS | Training W&B |
| --- | ---: | --- | --- | --- | --- | --- |
| K=1 grid=.25 | 1 | 100k / 100k | .0112448/.00711976/.00958620 | .136236/.122076/.112332 | .0133870/.00677605/.0100834 | `r5nja4dd` |
| K=1 grid=.25 | 2 | 44,001 / 30k | .00503560/.00480775/.00548853 | .115577/.111881/.118408 | .00593216/.00471237/.00542217 | `6e8qnd85` |
| K=1 grid=.25 | 3 | 10k / 10k | .00581609/.00479006/.00514953 | .139770/.128785/.133505 | .00529077/.00578658/.00603952 | `u96kjr2v` |
| K=32 | 1 | 100k / 100k | .00522931/.00496418/.00867124 | .103990/.0974418/.360730 | .00818985/.00887950/.00948479 | `67ui7zkw` |
| K=32 | 2 | 38,501 / 30k | .00679229/.00705017/.00678211 | .543652/.545218/.356955 | .00824209/.00851104/.00895950 | `r6nx6y51` |
| K=32 | 3 | 10k / 10k | .00750815/.00746239/.00769052 | .567474/.550882/.574557 | .00890945/.00806368/.00831328 | `xsapcf5h` |
| K=16 grid=.25 sr=.25 | 1 | 37,001 / 30k | .00751657/.00715043/.00508455 | .560028/.416014/.115207 | .00593166/.00508356/.00559258 | `1qg4eh1w` |
| K=16 grid=.25 sr=.25 | 2 | 10k / 10k | .00648412/.00541094/.00468722 | .160012/.133184/.135221 | .00647842/.00669005/.00643957 | `a7imdyyl` |
| K=16 grid=.25 sr=.25 | 3 | 10k / 10k | .00593080/.00559096/.00595855 | .151514/.127438/.121139 | .00569855/.00565838/.00576517 | `7hzr20pt` |
| K=16 lr×2 | 1 | 85,001 / 60k | .00818066/.00897592/.0100306 | .536998/.532596/.544665 | .00603984/.00756818/.00793487 | `pol71ad4` |
| K=16 lr×2 | 2 | 10k / 10k | .00628364/.00494081/.00579955 | .133289/.129416/.128932 | .00577522/.00625334/.00654947 | `2brllxla` |
| K=16 lr×2 | 3 | 10k / 10k | .00483298/.00545284/.00624874 | .119292/.121537/.246076 | .00772882/.00811595/.00825803 | `u6km38gw` |

Locked seed-1 evaluations:

| Family | Checkpoint | Full LPIPS 1/2/4/8 | Vertex 1/2/4/8 | Evaluation W&B |
| --- | --- | --- | --- | --- |
| K=1 grid=.25 | latest 100k | .0142740/.00959324/.0111189/.0128337 | .366203/.381802/.392317/.405033 | `cxte188a` |
| K=1 grid=.25 | validation-best 100k | .0143258/.00884105/.0106428/.0122470 | not reported in status snapshot | `mp327fqt` |
| K=32 | latest 100k | .00771021/.00771912/.00931270/.0103027 | .279273/.282591/.357600/.403780 | `hbzgz33a` |
| K=32 | validation-best 100k | .00849698/.00891598/.0103231/.0109385 | not reported in status snapshot | `0debz4j0` |

### Replication and duration conclusions

- K=1 grid=.25 has useful NFE2 behavior in all three visible seed curves, but NFE4
  is not consistently better than NFE2. On locked100 seed 1, NFE2 improves LPIPS
  32.79% over NFE1, while NFE4 and NFE8 give back part of that gain.
- K=32's seed-1 10k screen does not replicate cleanly. Seed 3 is nearly flat at
  NFE2 and 2.43% worse at NFE4; locked100 seed 1 is 20.78% worse at NFE4. More
  particles alone are insufficient.
- K=16 grid=.25 sr=.25 has the most consistent dynamics signal. At 10k, seed 2
  improves NFE4 LPIPS 27.71% and seed 3 is nearly flat (+0.47%) while improving
  vertex error 20.05%. Seed 1 becomes much stronger by 30k.
- K=16 lr×2 is transient and seed-sensitive. Seed 2 improves at 10k, seed 3
  regresses, and seed 1 at 60k is 22.61% worse at NFE4 than NFE1.
- Longer training is not a monotonic improvement. The 60k/100k K=32 and lr×2
  curves regress at higher NFE; K=1 retains an NFE2 gain but rebounds at NFE4/8.
  The composed-source/grid family is the promising exception at 30k.
- Validation-best frequently has worse rollout behavior than latest. The next
  training cycle should keep only `latest` and `best`, but define `best` using a
  rollout-aware development score rather than adaptation-validation loss alone.

## Research decision

The current hypothesis is narrower than “more particles” or “on-policy replay”
alone:

> Endpoint-normalized Drift Flow needs transport supervision anchored on the exact
> inference grid; once those intervals are identifiable, a controlled amount of
> model-composed source replay can make the local maps composable.

This is an interaction hypothesis. It predicts that removing either exact-grid
supervision or composed-source replay will weaken NFE4 behavior, and that the joint
variant should reproduce its 30k NFE4 gain across seeds.

The queues should continue. The next gate is:

1. Evaluate K=16/grid=.25/sr=.25 seeds 2 and 3 at 30k.
2. Accept the interaction as the lead idea only if at least two of three seeds
   improve both NFE4 LPIPS and vertex error over NFE1, with no material NFE1
   quality collapse.
3. Compare the surviving checkpoint with official DriftWorld on the same locked
   1000-video protocol and report latency at NFE1/2/4.
4. If the interaction fails replication, return to exact-grid supervision and
   test a rollout-consistency objective directly; do not increase K or replay
   fractions blindly.

No final superiority claim is made from this snapshot.
