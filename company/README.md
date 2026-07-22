# Company H100 runner

This directory is the only place where company paths and execution assumptions live.
It does not modify the core `driftworld/` configs. Commands override all artifact paths
at runtime.

## Fixed paths and network assumptions

- Clone the repository under `/user-volume/repo/`.
- Store datasets and checkpoints under `/group-volume/danny-dataset/driftworld/`.
- Store environments, full logs, W&B files, and result JSON under
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
bash company/setup.sh
```

The command is idempotent. It creates a Python environment, installs
`company/requirements.txt`, and downloads only the required Push-T data and weights
from Hugging Face. To download or repair assets without reinstalling the environment:

```bash
/user-volume/driftworld/envs/driftfm-py312/bin/python company/prepare_assets.py
```

If the company image already manages Python packages, set
`DRIFTFLOWWORLD_ENV_PREFIX` to that environment. Dependency installation respects
`PIP_INDEX_URL` and other company pip-mirror settings.

Set `WANDB_API_KEY` through the company secret manager. Optionally set
`WANDB_ENTITY` and `WANDB_PROJECT`; credentials are never written by these scripts.

## Two independent 4xH100 nodes

Validate four-GPU checkpoint/resume equivalence on one node:

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
