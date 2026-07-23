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
| CT0p/CT1p | planned overnight | company matched control/DriftFlowWorld pilots, 10k updates / seeds 1,2,3 | `company/run_overnight.sh` | two independent 4xH100 nodes, NGC 24.06 | official step 1180500 -> seed-specific group-volume checkpoints | online `driftfm-world-model-company`, one resumable run per arm/seed | training loss, time-pair statistics, runtime, stable completion marker | Priority is seed-1 matched signal, then replication seeds 2 and 3; no smoke checkpoint is reused. |
| CE0p/CE1p | planned overnight | company paired pilot eval / seed 1, first 25 videos | `company/run_pilot_eval.sh` | 4xH100, NGC 24.06 | CT0p/CT1p seed 1 -> seed-specific user-volume metrics | eval-only | control NFE1 vs Drift Flow NFE1/2/4 pixel and frozen-predictor pose metrics | Node A starts automatically after both seed-1 completion markers; Node B proceeds to seed 2 concurrently. |
| E0p | queued (`11364806`, gated) | 25-video control pilot eval / 1 | `docs/manifests/posttrain.yaml` | H100, `embers` | T0p -> `runs/metrics/pilot10k-control` | eval-only | 64/full pixel metrics | Stable 10k checkpoint comparison. |
| E1p | queued (`11364808_[0-2]`, gated) | 25-video Drift Flow pilot eval, NFE 1/2/4 / 1 | `docs/manifests/posttrain.yaml` | H100 each, `embers` | T1p -> `runs/metrics/pilot10k-driftflow` | eval-only | pixel and frozen-predictor block-pose metrics | Early direction check; not the locked transport claim. |
| T0/T1 | queued (`11362985_[0-1]`, gated) | matched control/DriftFlowWorld resume to 100k / 1 | `docs/manifests/posttrain.yaml` | 2x H100 each, `embers` | T0p/T1p -> final control/DFM | same pilot W&B runs | training loss, then locked rollout metrics | Starts only after all 10k pilot evaluations finish, preserving stable evaluated checkpoints. |

## Q3 — Where should a fixed planning budget be spent?

| ID | Status | Task / seeds | Manifest | GPU | Checkpoint / output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | planned | breadth-depth dev / 0:25 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | IoU, latency, peak memory, top-K recall | Select one K/M without test peeking. |
| P1 | planned | locked ep100 + ep300 / 0:100 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | paired IoU delta + bootstrap CI | Primary claim. |
