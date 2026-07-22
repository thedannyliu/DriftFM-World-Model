# Experiment ledger

Status values are `planned`, `queued`, `running`, `complete`, or `failed`. Metric cells
remain `TBD` until produced by the referenced artifact; absence of a result is never
written as a zero.

Environment: `/storage/scratch1/9/eliu354/driftflowworld/envs/pace-cu128-py312-v1`
(PyTorch 2.11.0+cu128; exact freeze beside the environment). Data and official
checkpoints were prepared under the same scratch root. W&B project:
`danny010324/driftflowworld-pusht` (personal-project default visibility: private).

## Q0 — Does the implementation preserve the DriftWorld endpoint?

| ID | Status | Task / seed | Manifest | GPU | Parent -> output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | complete (`11362493`) | synthetic endpoint / 1 | `docs/manifests/smoke.yaml` | L40S, `embers` | commit `5cf2855` -> `/storage/scratch1/9/eliu354/driftflowworld/runs/smoke/model-11362493.json` | disabled | Endpoint max diff 0; NFE1/4 `(1,4,3,96,96)`; loss 8.99987; peak 2.98 GB; sampled endpoint fraction 1.0. Formal env: 5 tests passed. Official ckpt: step 1180500, 8 expected missing time keys, 0 unexpected. | Endpoint gate passes; rerun fixed non-endpoint pair for GPU arbitrary-time coverage. |
| S0b | queued (`11362958`) | synthetic non-endpoint backward / 1 | `docs/manifests/smoke.yaml` | L40S, `embers` | commit `2092617` -> `/storage/scratch1/9/eliu354/driftflowworld/runs/smoke/model-11362958.json` | disabled | endpoint fraction fixed to 0; loss finite; peak memory | Required before H100 training. |
| S1 | queued (`11362747`, afterok `11362493`; resume `11362969`, afterok first pass) | Push-T two-step train then no-op resume / 1 | `docs/manifests/smoke.yaml` | 2x L40S, `embers` | official -> `/storage/scratch1/9/eliu354/driftflowworld/checkpoints/smoke-ddp` | offline smoke, run ID persisted | loss finite, DDP sync, checkpoint/RNG/run ID; second job must restore step 2 and same run | Must pass before H100 training. |

## Q1 — Is the released DriftWorld result reproduced?

| ID | Status | Task / seed(s) | Manifest | GPU | Checkpoint / output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | queued smoke (`11362770`) | Push-T 64-frame + full / first 10, then locked 1000 videos | `docs/manifests/baseline.yaml` | H100, `embers` | official step 1180500 / `/storage/scratch1/9/eliu354/driftflowworld/runs/metrics/driftworld-official-smoke` | eval-only | MSE, SSIM, PSNR, LPIPS, latency | Validate pipeline on 10 videos before locked run. |
| B1 | planned | GPC timing / dev seeds 0:25 | `docs/manifests/baseline.yaml` | H100 | official + policies ep100/ep300 | eval-only | IoU, 50/100/200 proposal latency, peak memory | Defines matched budgets. |

## Q2 — Does arbitrary-time post-training create useful NFE scaling?

| ID | Status | Task / seed | Manifest | GPU | Parent -> output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | queued (`11362985_0`, gated) | continued DriftWorld, 100k updates / 1 | `docs/manifests/posttrain.yaml` | 2x H100, `embers` | official -> control | `driftflowworld-pusht` | val loss and rollout metrics | Starts only after S0b, S1 resume, and B0 smoke pass. |
| T1 | queued (`11362985_1`, gated) | DriftFlowWorld, 100k updates / 1 | `docs/manifests/posttrain.yaml` | 2x H100, `embers` | official -> DFM | `driftflowworld-pusht` | NFE 1/2/4 metrics and block-pose error | Apply transport gate after matched training. |

## Q3 — Where should a fixed planning budget be spent?

| ID | Status | Task / seeds | Manifest | GPU | Checkpoint / output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | planned | breadth-depth dev / 0:25 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | IoU, latency, peak memory, top-K recall | Select one K/M without test peeking. |
| P1 | planned | locked ep100 + ep300 / 0:100 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | paired IoU delta + bootstrap CI | Primary claim. |
