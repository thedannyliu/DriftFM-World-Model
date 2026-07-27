# All-experiments result snapshot — 2026-07-27

## Scope

Source: `company/status_all_experiments.py` at commit `99f8211`, captured
2026-07-27 19:09 UTC on `run330254-sam4-2`.

- Task: Push-T from the released DriftWorld step-1,180,500 parent.
- Long-study development metrics: first 25 videos.
- Long-study `LOCKED` metrics: 100 videos.
- Weekend milestone metrics: first 25 videos.
- Metrics below are full-rollout LPIPS and final block-vertex error; lower is better.
- NFE values are ordered `1/2/4/8`.

The paper reports full LPIPS `.0146` on 1000 seeds. None of the 25- or 100-video
results below are treated as a paper-matched superiority result.

## Executive result

The original residual parameterization failed, and endpoint normalization repaired
the catastrophic numerical behavior. The completed corrected long study now shows a
narrower result:

1. K=1 with exact-grid replay produces a reproducible two-step sweet spot. Across
   three locked100 seeds, mean LPIPS is
   `.015329/.009854/.011447/.013297`; NFE2 improves 35.72% over NFE1, but NFE4/8
   give back the gain.
2. K=32, composed-source replay, and doubled learning rate do not yield monotonic
   NFE scaling after 100k. Their mean NFE4 changes versus NFE1 are respectively
   `+15.37%`, `+3.84%`, and `+43.82%`.
3. Weekend Node A selects K=16 grid-only, not source replay or the joint treatment.
   This weakens the proposed grid/source interaction.
4. Weekend Node B selects the shallow NFE2 grid, not a grid matching NFE8. The
   seed-1 screen does not support the grid-depth-matching hypothesis.
5. Weekend Node C finds a four-step EMA source useful early, but its winner degrades
   sharply by 150k. Its rollout-selected best remains at 60k.
6. Deep post-training is repeatedly harmful. Rollout-aware checkpoint selection is
   necessary, and training duration is part of the method rather than a neutral
   compute scale.

The evidence currently supports “exact-grid regularization creates a stable coarse
transport step,” not yet “more NFE gives a progressively better flow.”

## Completed original and corrected studies

The original overnight infrastructure is complete: all 12 training targets and all
paired 10k/20k/30k evaluations are present. Its residual DFM failure remains the
negative control: at 30k, control/DFM-NFE1/NFE4 full LPIPS is
`.003893/.010529/.684529`.

All four corrected long queues are complete for seeds 1/2/3 through 100k, including
100-video latest and validation-best evaluations.

### Locked100 latest checkpoints

| Family | Seed | LPIPS 1/2/4/8 | Vertex 1/2/4/8 | Eval W&B |
| --- | ---: | --- | --- | --- |
| K=1 grid=.25 | 1 | .014274/.009593/.011119/.012834 | .366203/.381802/.392317/.405033 | `cxte188a` |
| K=1 grid=.25 | 2 | .015366/.009563/.011633/.013908 | .349092/.317551/.335118/.342588 | `57s3v9vn` |
| K=1 grid=.25 | 3 | .016347/.010405/.011590/.013148 | .380861/.395062/.385395/.436179 | `ila848a8` |
| K=32 | 1 | .007710/.007719/.009313/.010303 | .279273/.282591/.357600/.403780 | `hbzgz33a` |
| K=32 | 2 | .008212/.007911/.008841/.009616 | .355051/.299505/.305067/.310800 | `odkukymj` |
| K=32 | 3 | .008338/.008542/.009836/.011578 | .356812/.370323/.375317/.428579 | `2slwfyji` |
| K=16 grid=.25 source=.25 | 1 | .008500/.008287/.008570/.009350 | .293679/.361360/.367846/.383780 | `v6wxyxea` |
| K=16 grid=.25 source=.25 | 2 | .008012/.007412/.008013/.009221 | .272969/.275892/.280624/.334015 | `rsuuo286` |
| K=16 grid=.25 source=.25 | 3 | .009479/.009017/.010405/.010875 | .481035/.465407/.532171/.529545 | `45mx9klg` |
| K=16 lr×2 | 1 | .008524/.009911/.012220/.014450 | .301384/.369250/.431897/.579522 | `lfhb66t9` |
| K=16 lr×2 | 2 | .008891/.009901/.012833/.016176 | .328176/.373962/.467336/.574520 | `lewbgs4v` |
| K=16 lr×2 | 3 | .008569/.009931/.012318/.014049 | .293321/.419401/.453262/.515068 | `x86j4s18` |

### Cross-seed locked100 means

| Family | Mean LPIPS 1/2/4/8 | NFE2 vs 1 | NFE4 vs 1 | NFE8 vs 1 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| K=1 grid=.25 | .015329/.009854/.011447/.013297 | -35.72% | -25.32% | -13.26% | Robust NFE2 optimum; not monotonic |
| K=32 | .008087/.008057/.009330/.010499 | -0.36% | +15.37% | +29.83% | Particle-scaling hypothesis rejected |
| K=16 grid=.25 source=.25 | .008663/.008239/.008996/.009816 | -4.90% | +3.84% | +13.30% | Interaction does not survive 100k robustly |
| K=16 lr×2 | .008661/.009914/.012457/.014892 | +14.47% | +43.82% | +71.93% | Optimization shortcut rejected |

Validation-best is not a reliable rollout selector. For example, K=1 grid=.25
validation-best is nearly flat or degrading with NFE on seeds 2/3, while their latest
checkpoints retain a strong NFE2 gain.

## Aggressive weekend study

### Node A — grid/source interaction surface

All 12 seed-1 screens reached 30k. The selector chose grid-only; the runner-up uses
grid=.50/source=.10.

| Variant | Latest evaluated step | LPIPS 1/2/4/8 | Vertex 1/2/4/8 | Rollout best | Train W&B |
| --- | ---: | --- | --- | --- | --- |
| Base | 30k | .004786/.004898/.005585/.006132 | .122601/.112910/.115837/.115296 | 30k@.034830 | `tkaya7ej` |
| Grid .25 only | 30k | .004465/.004526/.004880/.005197 | .100387/.097819/.099679/.101438 | **30k@.030065** | `13ss43ko` |
| Source .25 only | 30k | .007285/.007531/.008268/.008592 | .580630/.580187/.586187/.585091 | 30k@.070821 | `93yt27gj` |
| Grid .125/source .10 | 30k | .006738/.004836/.005299/.005572 | .380069/.119475/.117263/.118931 | 30k@.041848 | `bvvnesmb` |
| Grid .125/source .25 | 30k | .006733/.004849/.005292/.005543 | .381691/.121203/.120343/.120032 | 30k@.041787 | `su81lal2` |
| Grid .125/source .50 | 30k | .006705/.004387/.005127/.005372 | .417773/.118621/.122052/.123339 | 30k@.042356 | `07ostdkl` |
| Grid .25/source .10 | 30k | .007470/.006812/.005267/.005659 | .559596/.397503/.112958/.114950 | 10k@.046426 | `goiihj6x` |
| Grid .25/source .25 | 30k | .007406/.007106/.005132/.005496 | .558647/.417258/.115229/.115270 | 10k@.040816 | `1zshw33o` |
| Grid .25/source .50 | 30k | .007189/.006467/.007346/.007456 | .557885/.367614/.423278/.427352 | 10k@.040496 | `kuoeymke` |
| Grid .50/source .10 | 30k | .004727/.004251/.004860/.005409 | .119427/.112492/.112404/.112673 | **30k@.032060** | `n3olnxpv` |
| Grid .50/source .25 | 30k | .006798/.004156/.004512/.004877 | .418757/.107796/.113753/.114920 | 30k@.040463 | `mov33mbs` |
| Grid .50/source .50 | 30k | .006255/.004049/.004141/.004424 | .364730/.106168/.106719/.106812 | 30k@.036379 | `u3ao0ft3` |

Status: grid-only winner seed 1 is at 55,501 updates; winner seeds 2–5 and runner-up
seeds 2–3 are waiting. Source-only is the worst arm. A joint arm may improve higher
NFE conditional on a poor NFE1 state, but it does not beat grid-only on the balanced
quality/dynamics score. The preregistered interaction hypothesis is not supported by
the seed-1 screen.

### Node B — dyadic grid depth

All 12 seed-1 screens reached 30k. Grid-depth NFE2/probability .125 won; grid-depth
NFE8/probability .50 was runner-up.

| Grid max/probability | Latest evaluated step | LPIPS 1/2/4/8 | Vertex 1/2/4/8 | Rollout best | Train W&B |
| --- | ---: | --- | --- | --- | --- |
| 2/.125 | 100k | .005409/.005689/.005704/.006126 | .124963/.118202/.121862/.126437 | **60k@.033887** | `mapw63z0` |
| 2/.25 | 30k | .007264/.004483/.004823/.005441 | .566300/.121210/.123820/.125337 | 30k@.046303 | `tfs5bpu5` |
| 2/.50 | 30k | .008125/.004355/.004633/.005213 | .562073/.113017/.117557/.118731 | 30k@.047016 | `hus6p1p4` |
| 4/.125 | 30k | .006733/.004849/.005292/.005543 | .381691/.121203/.120343/.120032 | 30k@.041787 | `qzc6sdsj` |
| 4/.25 | 30k | .007406/.007106/.005132/.005496 | .558647/.417258/.115229/.115270 | 10k@.040816 | `d00ues6e` |
| 4/.50 | 30k | .006798/.004156/.004512/.004877 | .418757/.107796/.113753/.114920 | 30k@.040463 | `2f2y3k8d` |
| 8/.125 | 30k | .007606/.006885/.007187/.007472 | .561570/.370219/.394750/.398949 | 10k@.056860 | `6e0agim8` |
| 8/.25 | 30k | .007215/.006990/.007148/.007345 | .558359/.562365/.561952/.560630 | 10k@.061853 | `unu771tu` |
| 8/.50 | 30k | .007454/.004427/.004346/.004410 | .419118/.113210/.114162/.114480 | **30k@.040046** | `ctu7b3y6` |
| 16/.125 | 30k | .007055/.004556/.004765/.005288 | .566446/.125685/.127058/.126485 | 30k@.045332 | `4m3wvj6j` |
| 16/.25 | 30k | .007014/.004912/.005005/.005172 | .405126/.130478/.129482/.130559 | 30k@.041667 | `73usdm5w` |
| 16/.50 | 30k | .007060/.004836/.005024/.005040 | .408258/.118111/.117298/.116450 | 30k@.041111 | `6vz75hbf` |

Status: winner seed 1 is at 130,501 updates; all additional winner/runner seeds are
waiting. The shallowest grid wins the seed-1 score, so “training depth must match
inference NFE” is not supported. The higher-depth grids sometimes improve NFE4/8
relative to their poor NFE1 states, which is not the same as improving the overall
quality/latency frontier.

### Node C — composed-source depth

All 12 seed-1 screens reached 30k. Four EMA composition steps won at source replay
.10; four steps/source .50 was runner-up.

| Steps/source probability | Latest evaluated step | LPIPS 1/2/4/8 | Vertex 1/2/4/8 | Rollout best | Train W&B |
| --- | ---: | --- | --- | --- | --- |
| 1/.10 | 30k | .007548/.006706/.005267/.005754 | .560543/.366804/.113783/.115199 | 30k@.050995 | `0sjz5dt8` |
| 1/.25 | 30k | .007380/.006734/.007315/.007663 | .558631/.366570/.419813/.418945 | 10k@.062188 | `celb5wx9` |
| 1/.50 | 30k | .007281/.006739/.007381/.007462 | .558098/.368388/.426015/.423073 | 10k@.055291 | `0tzicpqg` |
| 2/.10 | 30k | .007470/.006812/.005267/.005659 | .559596/.397503/.112958/.114950 | 10k@.046426 | `tzuf1scv` |
| 2/.25 | 30k | .007406/.007106/.005132/.005496 | .558647/.417258/.115229/.115270 | 10k@.040816 | `o8wgsjlr` |
| 2/.50 | 30k | .007189/.006467/.007346/.007456 | .557885/.367614/.423278/.427352 | 10k@.040496 | `ft4f9u2d` |
| 4/.10 | 150k | .019959/.011569/.017689/.027754 | .650277/.194569/.310998/.570924 | **60k@.034447** | `yylajgpe` |
| 4/.25 | 30k | .007404/.007089/.005093/.005401 | .558667/.415908/.114925/.115152 | 10k@.045964 | `2lm8f9fy` |
| 4/.50 | 30k | .007287/.006972/.005033/.005153 | .559295/.423847/.117215/.116797 | **30k@.049736** | `ls4yojc4` |
| 8/.10 | 30k | .007420/.006754/.007355/.007838 | .558750/.367578/.418689/.419201 | 10k@.046095 | `9lffp4vm` |
| 8/.25 | 30k | .007426/.007101/.005083/.005490 | .560588/.418174/.114870/.115180 | 30k@.051040 | `bjx0y0yy` |
| 8/.50 | 30k | .007254/.007019/.005103/.005223 | .558121/.422633/.117445/.117632 | 30k@.049838 | `8fbxwqrq` |

Status: winner seed 1 is at 199,001 updates, but the latest evaluated 150k model has
collapsed relative to its rollout-best 60k checkpoint. Additional winner and runner
seeds are waiting. Four-step composition affects training, but the current evidence
supports an early optimization effect rather than a stable source-matching mechanism.

### Node D — matched causal controls

Only the K=16 corrected base arm has started.

| Arm/seed | Status | Latest evaluated step | LPIPS 1/2/4/8 | Rollout best | Train W&B |
| --- | --- | ---: | --- | --- | --- |
| Base K=16 / 1 | Complete 300k + locked100 | 300k | .007522/.007506/.012112/.014134 | 30k@.034830 | `uosak4f0` |
| Base K=16 / 2 | Complete 300k + locked100 | 300k | .006029/.006943/.011440/.012895 | 30k@.035044 | `orsdop2v` |
| Base K=16 / 3 | Running, 83,501 updates | 60k | .006720/.004816/.005869/.006435 | 30k@.036344 | `4losyorv` |

Locked100 latest/best LPIPS:

| Seed | Latest 1/2/4/8 | Rollout-best 1/2/4/8 | Eval W&B latest/best |
| ---: | --- | --- | --- |
| 1 | .008924/.009492/.012527/.015438 | .007710/.008345/.008884/.009364 | `focp2eve`/`6vxs9g4c` |
| 2 | .008136/.008709/.011682/.013756 | .008370/.008612/.009242/.009643 | `onzva702`/`etgigagc` |

Grid-only, source-only, and joint K=1/16/32 are all waiting for all three seeds.
Node D therefore cannot yet provide the causal confirmation. The two completed base
seeds independently show severe deep-training regression and validate the
rollout-best retention policy.

## Reproducibility note

Several weekend tags intentionally instantiate identical configurations with the
same seed. They produce identical metrics, for example:

- Node A grid=.125/source=.25 and Node B grid-depth4/probability=.125;
- Node A grid=.25/source=.25, Node B grid-depth4/probability=.25, and Node C
  compose2/source=.25;
- Node A grid=.25/source=.10 and Node C compose2/source=.10.

This validates deterministic execution and configuration plumbing. These duplicate
runs are not independent replications and must not be counted as separate evidence.

## Operational status

- Original overnight queues: complete, no missing results.
- Corrected long queues: 4/4 complete.
- Weekend Nodes A/B/C: screens complete, selected deep training in progress.
- Weekend Node D: base arm in progress; five arms entirely waiting.
- Checkpoints: 78 latest + 78 best, 19.74 GiB.
- Markers: 233 training completions; 348/348 evaluation markers complete.
- Rollout-best records: 39.
- Temporary checkpoint files: 0.
- Dataset and official checkpoint: ready.

## Research decision

Do not stop the running queues, because cross-seed and matched-control evidence is
still missing. However, the next analysis should not assume the original
grid/source-interaction hypothesis is correct.

The current lead statement is:

> Exact-grid exposure stabilizes a coarse, approximately two-step transport update,
> while additional particles, deeper grids, composed-source replay, and longer
> training have not yet produced monotonic refinement.

The decisive remaining tests are Node A grid-only cross-seed behavior and Node D's
grid-only/source-only/joint matched controls. If grid-only wins across those controls,
the research direction should move from “on-policy flow composition” to understanding
why exact intermediate-time regularization improves a small fixed number of updates
but fails under repeated composition.

No DriftWorld superiority claim is made from this snapshot.
