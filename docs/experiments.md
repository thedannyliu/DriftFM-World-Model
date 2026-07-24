# Experiment ledger

Status values are `planned`, `queued`, `running`, `complete`, or `failed`. Metric cells
remain `TBD` until produced by the referenced artifact; absence of a result is never
written as a zero.

Environment: `/storage/scratch1/9/eliu354/driftflowworld/envs/pace-cu128-py312-v1`
(PyTorch 2.11.0+cu128; exact freeze beside the environment). Data and official
checkpoints were prepared under the same scratch root. W&B project:
`danny010324/driftflowworld-pusht` (personal-project default visibility: private).
Company runs use the direct NGC 24.06 Python 3.10 / PyTorch 2.4 / CUDA 12.5 container,
4xH100 per node, shared assets under `/group-volume/danny-dataset/driftworld`, and
runtime logs under `/user-volume/driftworld`.

## Q0 — Does the implementation preserve the DriftWorld endpoint?

| ID | Status | Task / seed | Manifest | GPU | Parent -> output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | complete (`11362493`) | synthetic endpoint / 1 | `docs/manifests/smoke.yaml` | L40S, `embers` | commit `5cf2855` -> `/storage/scratch1/9/eliu354/driftflowworld/runs/smoke/model-11362493.json` | disabled | Endpoint max diff 0; NFE1/4 `(1,4,3,96,96)`; loss 8.99987; peak 2.98 GB; sampled endpoint fraction 1.0. Formal env: 5 tests passed. Official ckpt: step 1180500, 8 expected missing time keys, 0 unexpected. | Endpoint gate passes; rerun fixed non-endpoint pair for GPU arbitrary-time coverage. |
| S0b | complete (`11362958`) | synthetic non-endpoint backward / 1 | `docs/manifests/smoke.yaml` | L40S, `embers` | commit `2092617` -> `/storage/scratch1/9/eliu354/driftflowworld/runs/smoke/model-11362958.json` | disabled | endpoint fraction 0; loss 8.99309; peak 2.98 GB; endpoint diff 0; NFE1/4 shapes correct | Arbitrary-time GPU gate passes. |
| S1 | complete (`11362747`; load `11362969`) | Push-T two-step train then no-op load / 1 | `docs/manifests/smoke.yaml` | 2x L40S, `embers` | official -> `/storage/scratch1/9/eliu354/driftflowworld/checkpoints/smoke-ddp` | offline run `ghri4aqo`, same ID on load | losses 8.73815/8.98508; latest step 1; load restored step 2 in 19 s | DDP, checkpoint load, and W&B ID reuse pass; no-op load alone does not establish next-step equivalence. |
| S2 | queued (`11364127`) | uninterrupted 3 steps vs 2 + resume to 3 / 1 | `docs/manifests/smoke.yaml` | 2x L40S, `embers` | official -> job-scoped continuous/resumed outputs | offline, resumed arm reuses one ID | exact model/optimizer/scheduler/RNG checkpoint equality | Diagnostic for per-rank RNG and DataLoader isolation; not a hard gate for fresh matched post-training. |
| S2c | complete (`20260723-001107`) | company uninterrupted 3 steps vs 2 + resume to 3 / 1 | `company/smoke_resume.sh` | 4xH100, NGC 24.06 | official -> `checkpoints/smoke/resume-equivalence-20260723-001107` | offline `0brqeyfd` / `wpbte6r4` | Losses exactly match: 8.740981, 8.982100, 8.975924. Step, scheduler, and all rank RNG states exact. 28 model tensors differ (max abs 7.45e-9); 547 optimizer tensors differ (max abs 2.05e-8). | Resume is numerically stable within atol 3e-8 / rtol 1e-5; bitwise equality is non-blocking because matched pilots start fresh from the same official checkpoint. |
| S3 | complete (`11364620_2`) | one-video NFE=4 pose-metric smoke / 1 | `docs/manifests/smoke.yaml` | L40S, `embers` | S1 step-1 checkpoint -> `/storage/scratch1/9/eliu354/driftflowworld/runs/metrics/pose-smoke` | eval-only | 64/full vertex proxy 9.780/10.663; xy 98.806/113.949; angle 1.540/1.526 rad; all per-video fields present. | Metric path passes in 6:08 (including about 3 minutes shared-filesystem wait). The intentionally two-step model's quality is not research evidence. |

## Q1 — Is the released DriftWorld result reproduced?

| ID | Status | Task / seed(s) | Manifest | GPU | Checkpoint / output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | queued smoke (`11362770`) | Push-T 64-frame + full / first 10, then locked 1000 videos | `docs/manifests/baseline.yaml` | H100, `embers` | official step 1180500 / `/storage/scratch1/9/eliu354/driftflowworld/runs/metrics/driftworld-official-smoke` | eval-only | MSE, SSIM, PSNR, LPIPS, latency | Validate pipeline on 10 videos before locked run. |
| B0a | complete (`11363556`) | Push-T 64-frame + full / first 2 | `docs/manifests/baseline.yaml` | L40S, `embers` | official step 1180500 / `/storage/scratch1/9/eliu354/driftflowworld/runs/metrics/driftworld-official-l40s-smoke` | eval-only | 64: .000786 MSE, .99669 SSIM, 37.860 PSNR, .002106 LPIPS. Full: .001550, .99462, 35.521, .002640. | End-to-end pipeline passes in 3:04. Two videos are not a reproduction claim; cold-start timing motivated RNG-preserving warm-up before B0. |
| B0b | complete (`11363923`) | RNG-preserving warm-up repeat / first 2 | `docs/manifests/baseline.yaml` | L40S, `embers` | official step 1180500 / `/storage/scratch1/9/eliu354/driftflowworld/runs/metrics/driftworld-official-l40s-warmup` | eval-only | Per-video 64/full metrics exactly match B0a; .00273/.00556 seconds per frame after warm-up. | RNG preservation and steady-state timer pass on GPU in 0:29; L40S timing remains diagnostic only. |
| B0c | complete (`20260723-002303`) | company baseline smoke / first 10 | `company/run_baseline.sh` | H100, NGC 24.06 | official step 1180500 -> `/user-volume/driftworld/results/baseline` | eval-only | 64: .000983 MSE, .001998 LPIPS, .004893 s/frame. Full: .001809 MSE, .003664 LPIPS, .006158 s/frame. | End-to-end company pipeline passes. Ten videos are not a reproduction claim; retain the locked 1000-video evaluation for confirmation. |
| B1 | planned | GPC timing / dev seeds 0:25 | `docs/manifests/baseline.yaml` | H100 | official + policies ep100/ep300 | eval-only | IoU, 50/100/200 proposal latency, peak memory | Defines matched budgets. |

Locked paper references (1000 seeds, one H100): 64-frame MSE 0.0007, SSIM 0.9925,
PSNR 33.7753, LPIPS 0.0050, 0.0037 seconds/frame; full-episode MSE 0.0018,
SSIM 0.9846, PSNR 31.6612, LPIPS 0.0146, 0.0045 seconds/frame. GPC-RANK with 50
proposals reports IoU 0.781/0.734 for policies 1/2 and 0.912 seconds planning time.

## Q2 — Does arbitrary-time post-training create useful NFE scaling?

| ID | Status | Task / seed | Manifest | GPU | Parent -> output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0p/T1p | queued (`11364799_[0-1]`, gated) | matched control/DriftFlowWorld pilot, 10k updates / 1 | `docs/manifests/posttrain.yaml` | 2x H100 each, `embers` | official -> control/DFM | `driftflowworld-pusht` | training loss and time-pair statistics | Starts after B0; both arms start fresh from the same official checkpoint, while S2 remains diagnostic. |
| CT0p/CT1p | complete (`20260723`) | company matched control/DriftFlowWorld: 30k updates seed 1, 10k updates seeds 2,3 | `company/run_overnight.sh` | two independent 4xH100 nodes, NGC 24.06 | official step 1180500 -> seed-specific group-volume checkpoints | control `zq0wd3pt`, `8s3aey6q`, `6iiy232e`; DFM `xr56x64w`, `kaev6xy4`, `13bpabzk` | Seed-1 best adaptation loss: control 7.34217 at 13k; DFM 8.76085 at 1k. Seeds 2/3 completed. | Training infrastructure, resume, online W&B, validation, and compact retention all pass. Loss values are not comparable across objectives. Early DFM best and endpoint degradation argue against extending the unchanged objective to 100k. |
| CE0p/CE1p | complete (`20260723`) | paired seed-1 rollout eval at 10k/20k/30k latest and 30k best; first 25 videos | `company/run_pilot_eval.sh` | 4xH100, NGC 24.06 | CT0p/CT1p -> milestone-tagged user-volume metrics | eval-only | Full LPIPS `(control, DFM NFE1/2/4)`: 10k `(.00523,.02358,.58708,.75669)`; 20k `(.00493,.01738,.53260,.71012)`; 30k `(.00389,.01053,.50014,.68453)`; independently selected best `(.00793,.00606,.71194,.86712)`. DFM full vertex NFE1/4: `.465/10.702`, `.748/10.827`, `.306/10.792`, best `.141/11.137`. | Gate fails decisively: DFM NFE1 improves with updates but is still 170% worse than 30k control; NFE4 is 6402% worse than NFE1. Best DFM preserves a useful NFE1 signal but composition remains catastrophic. Do not run planning or MNAD with this NFE4 teacher. |
| CA1/CA2 | complete (`20260723`) | Drift Flow time-sampling ablation: uniform; endpoint-replay ablation: 0.50; seed 1, 10k each | `company/run_overnight.sh` | one independent 4xH100 node each after matched runs | official step 1180500 -> `driftflow-uniform_seed1` / `driftflow-replay50_seed1` | uniform `ralk43lj`; replay .50 `oirs7ece` | Uniform full LPIPS latest NFE1/2/4 `(.03032,.52813,.69085)` and best `(.00631,.71120,.86651)`; replay .50 latest `(.00432,.71536,.87005)` and best `(.00546,.71132,.86967)`. | Uniform sampling does not restore scaling. Replay .50 recovers strong NFE1, isolating endpoint retention from local transport, but NFE2/4 remain catastrophic. |
| E0p | queued (`11364806`, gated) | 25-video control pilot eval / 1 | `docs/manifests/posttrain.yaml` | H100, `embers` | T0p -> `runs/metrics/pilot10k-control` | eval-only | 64/full pixel metrics | Stable 10k checkpoint comparison. |
| E1p | queued (`11364808_[0-2]`, gated) | 25-video Drift Flow pilot eval, NFE 1/2/4 / 1 | `docs/manifests/posttrain.yaml` | H100 each, `embers` | T1p -> `runs/metrics/pilot10k-driftflow` | eval-only | pixel and frozen-predictor block-pose metrics | Early direction check; not the locked transport claim. |
| T0/T1 | queued (`11362985_[0-1]`, gated) | matched control/DriftFlowWorld resume to 100k / 1 | `docs/manifests/posttrain.yaml` | 2x H100 each, `embers` | T0p/T1p -> final control/DFM | same pilot W&B runs | training loss, then locked rollout metrics | Starts only after all 10k pilot evaluations finish, preserving stable evaluated checkpoints. |

Company adaptation-validation split: for the eight 500-episode datasets, episodes
0–489 post-train and 490–499 validate. The released parent may already have trained on
all 500, so this split monitors post-training overfit but is not an unseen-test claim.
The eight single-episode long-trajectory datasets remain train-only; holding them out
would remove those domains entirely. Validation uses the first 16 deterministic
batches with fixed stochastic-loss RNG. It never advances training RNG.

### Q2 failure audit — Why does composition collapse?

| ID | Status | Hypothesis | Experiment | Falsifying / supporting signal |
| --- | --- | --- | --- | --- |
| D2a | complete (`20260723`) | Local maps fail because continuous sigmoid sampling never trains the exact boundary intervals used by NFE2/4: `(0,.5)`, `(0,.25)`, and `(.75,1)`. | `company/run_transport_repair.sh node-b`: 10k with 25% inference-grid replay, then latest/best NFE1/2/4 evaluation. | Grid latest full LPIPS NFE1/2/4 `(.02454,.52410,.68517)` and best `(.00890,.71132,.86585)`. Exact-grid replay improves latest NFE2/4 only 10.7%/9.5% versus original 10k DFM and does not reverse scaling; coverage alone is rejected. |
| D2b | complete (`20260723`) | Intermediate target marginal is under-sampled: implementation uses 16 source particles but one noisy positive although the design specifies K positives. | `company/run_transport_repair.sh node-a`: 10k with 16 intermediate positives while endpoint replay retains one positive, then latest/best NFE1/2/4 evaluation. | Full LPIPS latest NFE1/2/4 `(.02361,.55958,.73302)` and best `(.00884,.71252,.87299)`. Versus original 10k DFM latest, NFE2/4 improve only 4.7%/3.1%; best composition is unchanged and NFE1 is 46% worse. Positive count alone is rejected. |
| D2c | complete (`20260723`) | Local maps work on interpolant sources but composed predictions leave the training support. | `company/run_transport_audit.sh`: compare paired teacher-forced transitions and free composition on identical NFE1/2/4 grids using the same source particles. | Best/replay checkpoints already fail teacher-forced late intervals: progress is approximately `.50` for `.5→1` and `.25` for `.75→1`; free failure is therefore not primarily an off-support composition problem. Self-forcing is not the first repair. |
| D2d | complete (`20260723`) | The post-trained network remains an endpoint predictor and does not use the time pair as a mean-velocity model. | Eight validation batches × 16 paired particles per checkpoint; compare correct-interval residual with `(0,1)` residual and measure displacement progress. W&B audit IDs: main `i6mlzjhs`/`raul7287`, uniform `o13qe9q7`, K-positive `bf1ba0dl`, replay `eojdcf39`/`9u01bcr3`, grid `wsrssqka`/`38lw1usb`. | Supported strongly: every cosine is above `.99999`. Early/best/replay checkpoints give progress `1.00/.50/.25` on `0→.5/.5→1/.75→1`, exactly matching an endpoint predictor multiplied by raw interval length. Long main/grid training partially changes magnitude but not direction and remains far from correct. |
| D2e | complete (`20260724`) | Endpoint compatibility used the wrong conversion from endpoint prediction to mean velocity: `G(x_t)-x_t` must be divided by remaining path time `1-t`. | Evaluate existing checkpoints without retraining using `x_r=x_t+(r-t)/(1-t)(G(x_t)-x_t)` via `company/run_endpoint_normalized_eval.sh`; NFE1 is algebraically unchanged. | Causal mechanism supported: catastrophic full-LPIPS composition falls from roughly `.71–.87` to `.007–.009` on early/best checkpoints. K-positive best improves from NFE1 `.00884` to NFE2 `.00800` (-9.5%); grid best improves `.00890→.00814` (-8.6%). NFE4 is flat or slightly worse, so corrected parameterization is necessary but not yet sufficient for monotonic scaling. |
| D2f | ready | Training with the same endpoint-normalized transport used at inference should improve intermediate maps; multiple positive target particles may then reduce their marginal-estimation variance. | `company/run_corrected_training.sh`: matched seed-1 training from the official checkpoint, Node A K=1 and Node B K=16; endpoint replay .25, no grid replay; evaluate latest at 1k/3k/10k and best through 10k, each at NFE1/2/4. | Advance K=16 only if its replicated learning curve improves NFE2 by at least 5% without degrading NFE1 by more than 5%; require NFE4 to stop worsening before planning experiments. Otherwise inspect training-target consistency rather than adding replay mixtures. |

## Q3 — Where should a fixed planning budget be spent?

| ID | Status | Task / seeds | Manifest | GPU | Checkpoint / output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | planned | breadth-depth dev / 0:25 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | IoU, latency, peak memory, top-K recall | Select one K/M without test peeking. |
| P1 | planned | locked ep100 + ep300 / 0:100 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | paired IoU delta + bootstrap CI | Primary claim. |
