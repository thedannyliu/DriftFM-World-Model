# Company H100 runner

This directory is the only place where company paths and execution assumptions live.
It does not modify the core `driftworld/` configs. Commands override all artifact paths
at runtime.

## Fixed paths and network assumptions

- Clone the repository under `/user-volume/repo/`.
- Store datasets and checkpoints under `/group-volume/danny-dataset/driftworld/`.
- Store package caches, full logs, W&B files, and result JSON under
  `/user-volume/driftworld/`.
- Company machines may pull GitHub and Hugging Face and may log to W&B. They must not
  push to GitHub.

## Setup and asset download

Clone and enter the repository:

```bash
git clone https://github.com/thedannyliu/DriftFM-World-Model.git \
    /user-volume/repo/DriftFM-World-Model
cd /user-volume/repo/DriftFM-World-Model
```

Then run:

```bash
hf auth login
bash company/setup.sh
```

Run this inside the company image
`ngc24.06-ub22-py3.10-cu12.5-cudnn9.1-pytorch2.4-deepspeed0.14-8packing`.
The command installs `company/requirements.txt` directly into the active container;
the company workflow does not use venv or Conda. It preserves the image's PyTorch 2.4
and torchvision packages and downloads the required Push-T data and weights from
Hugging Face. Downloads use a 120-second read timeout, retry transient failures, and
resume files already present. To download or repair assets without reinstalling
packages:

```bash
python3 company/prepare_assets.py
```

Setup stages, package installation, download progress, and retries are shown in the
terminal. The same output is retained in `/user-volume/driftworld/logs/setup.log`.
Downloads use eight workers by default. Set `DRIFTFLOWWORLD_DOWNLOAD_WORKERS=16` for
more throughput; values above 16 are capped to avoid Hugging Face rate limits. The
32,061-file Push-T dataset is fetched as one approximately 0.30 GiB Git pack, then the
two Git LFS objects are downloaded through the authenticated Hub API. The token stays
in the user's Hugging Face login store and is never copied into the shared asset root.

Dependency installation respects `PIP_INDEX_URL` and other company pip-mirror
settings. Do not install a separate PyTorch or CUDA wheel into the container. Package
pins are selected to remain compatible with the NGC image's Transformers, cuDF, and
MLflow dependencies used elsewhere in the image.

On additional nodes sharing already-prepared assets, skip asset work with:

```bash
DRIFTFLOWWORLD_SKIP_ASSETS=1 bash company/setup.sh
```

Set `WANDB_API_KEY` through the company secret manager. Optionally set
`WANDB_ENTITY` and `WANDB_PROJECT`; credentials are never written by these scripts.

## Two independent 4xH100 nodes

Diagnose four-GPU checkpoint/resume numerical equivalence on one node. A bitwise
mismatch is reported quantitatively but does not block a fresh matched pilot:

```bash
bash company/smoke_resume.sh
```

First run the 10-video official baseline on either node:

```bash
bash company/run_baseline.sh
```

Then run the matched 10k pilot concurrently. On node A:

```bash
bash company/run_pilot.sh control
```

On node B:

```bash
bash company/run_pilot.sh driftflow
```

Both use four local H100s with batch size one per GPU, preserving global batch size
four. Re-running the same command resumes the same checkpoint and W&B run. Keep
`GPUS_PER_NODE` unchanged for a resumed run.

After both commands report step 9999, use one four-GPU node for the paired pilot
evaluation:

```bash
bash company/run_pilot_eval.sh
```

Terminal output is intentionally short JSON suitable for pasting back. Full logs stay
under `/user-volume/driftworld/logs/`. Useful overrides include `MAX_STEPS`,
`EVAL_NUM_VIDEOS`, `SEED`, `CUDA_VISIBLE_DEVICES`, `WANDB_ENTITY`, and
`WANDB_PROJECT`.

Training launchers print the resolved GPU, checkpoint, output, log, and W&B settings.
Smoke runs print every loss/checkpoint event; pilot runs print every 100th loss plus
checkpoint events. Any failed DDP subprocess prints the first underlying traceback
context rather than only the final TorchElastic wrapper summary.
Both launchers run a single-process dependency preflight before starting four DDP
workers.
