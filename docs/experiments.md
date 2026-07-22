# Experiment ledger

Status values are `planned`, `queued`, `running`, `complete`, or `failed`. Metric cells
remain `TBD` until produced by the referenced artifact; absence of a result is never
written as a zero.

## Q0 — Does the implementation preserve the DriftWorld endpoint?

| ID | Status | Task / seed | Manifest | GPU | Parent -> output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | planned | synthetic forward / 1 | `docs/manifests/smoke.yaml` | CPU then L40S | released architecture -> none | disabled | load keys, max abs NFE=1 diff, NFE output shapes | Must pass before data jobs. |
| S1 | planned | Push-T one-row train / 1 | `docs/manifests/smoke.yaml` | 2x L40S | official -> smoke checkpoint | `driftflowworld-pusht` | loss finite, DDP sync, resume step/run ID | Must pass before training. |

## Q1 — Is the released DriftWorld result reproduced?

| ID | Status | Task / seed(s) | Manifest | GPU | Checkpoint / output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | planned | Push-T 64-frame + full / locked 1000 videos | `docs/manifests/baseline.yaml` | H100 | official step 1180500 / scratch metrics | eval-only | MSE, SSIM, PSNR, LPIPS, latency | Apply protocol tolerances. |
| B1 | planned | GPC timing / dev seeds 0:25 | `docs/manifests/baseline.yaml` | H100 | official + policies ep100/ep300 | eval-only | IoU, 50/100/200 proposal latency, peak memory | Defines matched budgets. |

## Q2 — Does arbitrary-time post-training create useful NFE scaling?

| ID | Status | Task / seed | Manifest | GPU | Parent -> output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | planned | continued DriftWorld, 100k updates / 1 | `docs/manifests/posttrain.yaml` | 2x H100 | official -> control | `driftflowworld-pusht` | val loss and rollout metrics | Matched-update control. |
| T1 | planned | DriftFlowWorld, 100k updates / 1 | `docs/manifests/posttrain.yaml` | 2x H100 | official -> DFM | `driftflowworld-pusht` | NFE 1/2/4 metrics and block-pose error | Apply transport gate. |

## Q3 — Where should a fixed planning budget be spent?

| ID | Status | Task / seeds | Manifest | GPU | Checkpoint / output | W&B | Metrics | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | planned | breadth-depth dev / 0:25 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | IoU, latency, peak memory, top-K recall | Select one K/M without test peeking. |
| P1 | planned | locked ep100 + ep300 / 0:100 | `docs/manifests/planning.yaml` | up to 8x H100 | selected T1 / per-seed shards | eval-only | paired IoU delta + bootstrap CI | Primary claim. |
