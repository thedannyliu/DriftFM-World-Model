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
resume files already present.

The requirements install W&B 0.22.3, which accepts current long-form API keys. After
updating an existing checkout, reinstall without touching shared assets before login:

```bash
DRIFTFLOWWORLD_SKIP_ASSETS=1 bash company/setup.sh
wandb login --relogin
```

To download or repair assets without reinstalling packages:

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

For an unattended priority queue, log in to W&B on both nodes, then start Node A and
Node B respectively:

```bash
wandb login
bash company/start_overnight.sh node-a

wandb login
bash company/start_overnight.sh node-b
```

The queue forces online W&B logging. Seed 1 runs for 30k matched steps, with paired
25-video rollout evaluation at 10k, 20k, and 30k. The final milestone evaluates both
`latest` and validation-selected `best`. Seeds 2 and 3 then run for 10k matched steps.
Finally, spare queue time runs two 10k Drift Flow ablations: uniform time sampling on
Node A and endpoint replay probability 0.50 on Node B. Override targets with
`OVERNIGHT_PRIMARY_STEPS`, `OVERNIGHT_REPLICATION_STEPS`,
`OVERNIGHT_ABLATION_STEPS`, and `OVERNIGHT_MILESTONES`. Queue stdout is detached under
`/user-volume/driftworld/logs/overnight/`; each training and evaluation retains its
own full log in the existing task-specific directory.

`start_overnight.sh` follows the detached queue log in the terminal by default.
Training prints the actual step and loss every 20 updates, plus validation and
checkpoint events. Pressing Ctrl-C stops only the viewer; the printed queue PID keeps
running in the background. Set `OVERNIGHT_FOLLOW_LOGS=0` for an immediate return or
change the interval with `PILOT_PRINT_EVERY`.

Inspect overnight progress, missing tasks, W&B run IDs, retained checkpoint size, and
rollout metrics with:

```bash
python3 company/status_overnight.py all
```

Use `node-a` or `node-b` instead of `all` to restrict the expected training tasks.
Run the command once on each node when `/user-volume` is node-local; completion
markers and evaluation results under `/group-volume` remain shared.

After the initial NFE2/4 failure, run the two isolated transport-repair hypotheses on
the two nodes:

```bash
bash company/run_transport_repair.sh node-a  # 16 intermediate positives
bash company/run_transport_repair.sh node-b  # 25% inference-grid replay
```

Each starts a separate resumable W&B run and output directory, then evaluates
NFE1/2/4 for both `latest` and `best`. Existing trained variants can be evaluated
without retraining, for example:

```bash
bash company/run_variant_eval.sh driftflow-uniform latest
bash company/run_variant_eval.sh driftflow-replay50 latest
```

After both repair runs and their rollout evaluations finish, localize the failure on
the two nodes concurrently:

```bash
# Node A: main DFM latest/best, uniform best, and 16-positive best
bash company/run_transport_audit.sh node-a

# Node B: endpoint-replay .50 latest/best and exact-grid latest/best
bash company/run_transport_audit.sh node-b
```

Each GPU audits one checkpoint using the first eight fixed validation batches and 16
paired source particles. It measures one-chunk teacher-forced local transport, free
NFE1/2/4 composition, displacement progress, orthogonal error, and sensitivity to the
time pair. Every checkpoint gets an online W&B `transport-audit` run. The terminal
prints batch progress and one compact combined JSON; full per-interval metrics and
logs remain under `/user-volume/driftworld/results/transport-audit/` and
`/user-volume/driftworld/logs/transport-audit/`. Increase evidence with
`AUDIT_NUM_BATCHES` or `AUDIT_PARTICLES` only after the default audit identifies a
stable mechanism.

The audit identifies an endpoint-prediction failure mode when late teacher-forced
progress follows the raw interval length and time-conditioned residuals remain nearly
collinear with the endpoint residual. Test the corresponding endpoint-normalized
transport without retraining:

```bash
bash company/run_endpoint_normalized_eval.sh node-a
bash company/run_endpoint_normalized_eval.sh node-b
```

Run one command on each node. The sampler uses
`x_r = x_t + (r-t)/(1-t) * (G(x_t)-x_t)`, which leaves NFE1 exactly unchanged.
Each node evaluates four existing latest/best checkpoints sequentially, logs every
rollout summary online to W&B, keeps parameterization-specific metrics separate from
the original evaluations, and prints one compact final JSON.

Train the corrected parameterization from the official checkpoint on two matched
4xH100 nodes:

```bash
# Node A: one positive target particle
bash company/run_corrected_training.sh node-a

# Node B: 16 positive target particles
bash company/run_corrected_training.sh node-b
```

Both arms use seed 1, logit-normal time sampling, endpoint replay 0.25, no grid
replay, and online W&B. They stop at 1k, 3k, and 10k updates and evaluate the
corresponding `latest` checkpoint at NFE 1/2/4 on 25 videos. At 10k they also
evaluate the minimum-validation-loss checkpoint selected over the full run. Each
milestone has a separate result marker, so rerunning the command skips completed
stages and resumes an incomplete training stage from `ckpt-latest.pth` using the
same W&B run. Only `ckpt-latest.pth` and `ckpt-best.pth` are retained.

For a four-node allocation with more than 20 hours available per 4xH100 node, launch
one independent queue on each node:

```bash
bash company/run_long_research_queue.sh node-a
bash company/run_long_research_queue.sh node-b
bash company/run_long_research_queue.sh node-c
bash company/run_long_research_queue.sh node-d
```

Node A screens K=1 time-pair curricula and includes the basic corrected Node A run.
Node B screens K=2/4/8/16/32 and includes the basic corrected Node B run. Node C
tests two-step EMA composed-source replay at probabilities 0.10/0.25/0.50, including
its interaction with K and grid replay. Node D tests 1k/3k endpoint warmup,
half/double learning rates, and stronger grid replay.

Every candidate is trained and evaluated at 1k/3k/10k. Each node then selects its
own seed-1 candidate using a preregistered score over full-rollout LPIPS, final block
vertex error, and NFE monotonicity. The selected configuration is screened on seeds
2 and 3, and all three seeds continue to 30k/60k/100k. A final 100-video evaluation
runs NFE 1/2/4/8 on both `latest` and `best`. Nodes A/B/D schedule 340k total
updates and Node C 350k; even the fastest observed company throughput implies more
than 20 hours before evaluation overhead. Completed stages are marker-gated, so the
same command safely resumes after a node timeout. A failed candidate, replication,
or final evaluation is recorded and skipped without stopping the remaining queue;
its resumable `latest` state is retained for a later retry.

From any node, summarize all shared queue progress and rollout results in a compact
terminal report suitable for pasting back:

```bash
python3 company/status_long_research.py all
```

The report reads checkpoints, evaluation markers, selections, and W&B run IDs from
`/group-volume/danny-dataset/driftworld/checkpoints/experiments`; it does not depend
on node-local `/user-volume` logs. Use `node-a`, `node-b`, `node-c`, or `node-d`
instead of `all` for a shorter report.

### Aggressive four-node weekend queues

The follow-up in `docs/weekend-research-plan.md` schedules substantially more than
72 hours per node:

```bash
# Run exactly one line on each independent 4xH100 node.
bash company/run_weekend_research_queue.sh node-a
bash company/run_weekend_research_queue.sh node-b
bash company/run_weekend_research_queue.sh node-c
bash company/run_weekend_research_queue.sh node-d
```

Nodes A/B/C each schedule about 2.9M updates: twelve 30k screens, then the winner on
five seeds through 400k and runner-up on three seeds through 200k. Node D schedules
5.4M updates from six matched causal arms × three seeds through 300k. At an
optimistic 30k updates/hour this is about 97 hours on Nodes A/B/C and 180 hours on
Node D, before NFE1/2/4/8 evaluation overhead.

The commands are marker-gated and resumable. Every training process reuses its W&B
run ID. Validation metrics remain online, but weekend runs retain rollout-selected
`best` rather than validation-loss-selected `best`; only `latest` and `best` model
files exist. A 25-video rollout is evaluated at every milestone, and final 100-video
latest/best evaluations are logged online.
The 72 seed-specific output directories use approximately 18.2 GiB for checkpoints
at the observed 129.6 MiB per file.

Print a pasteable shared status report from any node:

```bash
python3 company/status_weekend_research.py all
```

To print one exhaustive terminal report covering the original overnight runs, all
corrected long queues, every completed or unstarted weekend variant, W&B IDs,
rollout metrics, missing milestones, and shared checkpoint inventory:

```bash
python3 company/status_all_experiments.py
```

The command is read-only. Redirect or `tee` its output when it needs to be pasted
back for documentation.

### Advantage-aligned transport diagnosis

The original monotonic-NFE proof of concept is rejected. Before adding another
training loss, use the completed checkpoints to test whether route defect,
model-generated source shift, and motion difficulty explain the NFE2->4/8
regression. Run one command on each independent 4xH100 node:

```bash
AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-a

AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-b

AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-c

AUDIT_NUM_BATCHES=64 AUDIT_PARTICLES=4 \
bash company/run_hypothesis_audit.sh node-d
```

Each node assigns one existing checkpoint to each GPU, forces the corrected
endpoint-normalized transport, logs four online W&B `hypothesis-audit` runs, and
prints per-checkpoint NFE1/2/4/8 pixel/pose risk, adjacent route-defect correlations,
and clean-versus-model-source penalties. It does not modify checkpoints.

After all four nodes finish, print the combined 16-checkpoint gate from any node:

```bash
python3 company/status_hypothesis_audit.py
```

Use `bash company/run_hypothesis_audit.sh node-a --print-plan` to inspect a node
without credentials or GPUs. The hypotheses, falsifiers, preregistered gates, and
checkpoint allocation are in
`docs/advantage-aligned-transport-hypotheses.md`. Do not start the proposed training
stage from an individual node's partial report.

### Two-node locked advantage frontier

The 2026-07-28 partial audit contains node A/C only. With two available 4xH100 nodes,
complete node B/D and then run the paired paper-sized risk frontier. The launcher
requires code commit `bad6e1b` or later:

```bash
# First 4xH100 node: missing K=32/base audit, then eight locked1000 checkpoints.
bash company/run_advantage_followup.sh node-a

# Second 4xH100 node: missing deep-training audit, then eight locked1000 checkpoints.
bash company/run_advantage_followup.sh node-b
```

Each queue first skips or completes its missing four-checkpoint hypothesis audit.
It then evaluates eight frozen checkpoints sequentially; within each checkpoint,
NFE1/2/4/8 occupy GPUs 0/1/2/3 concurrently. The first-25-video benchmark projects
more than 12 hours per queue, although exact duration depends on shared I/O. No model
checkpoint is created. A completed checkpoint is resumable through its
`advantage-locked1000` marker, and every evaluation logs online to the existing W&B
project.

The evaluation records aligned per-video normalized action path and ground-truth motion in
addition to LPIPS, MSE, and final block-pose error. It reports whether an action-path
threshold fixed on the first half improves the second-half NFE1/NFE2 allocation
relative to random routing at the same mean NFE. Inspect both queues with:

```bash
python3 company/status_advantage_frontier.py all
```

Print the allocations without GPUs or credentials:

```bash
bash company/run_advantage_followup.sh node-a --print-plan
bash company/run_advantage_followup.sh node-b --print-plan
```

Post-training holds out episodes 490–499 from each 500-episode domain and evaluates 16
fixed adaptation-validation batches every 500 updates. The released parent may have
already seen these episodes, so this detects post-training overfit but is not an unseen
test-set claim. The single-episode long-trajectory domains remain train-only. Each run
stores exactly two model checkpoints:
`ckpt-latest.pth` is a full resumable state (model, optimizer, scheduler, step, per-rank
RNG, and best-validation metadata), and `ckpt-best.pth` is the full state with minimum
held-out loss. Resume always loads `latest` and reuses the same W&B run ID. Older
`ckpt-2nd-latest.pth` files in a resumed output are removed automatically. W&B logs
metrics only and does not upload checkpoint artifacts.

Terminal output is intentionally short JSON suitable for pasting back. Full logs stay
under `/user-volume/driftworld/logs/`. Useful overrides include `MAX_STEPS`,
`EVAL_NUM_VIDEOS`, `SEED`, `CUDA_VISIBLE_DEVICES`, `WANDB_ENTITY`, and
`WANDB_PROJECT`. Single-run Drift Flow ablations can also set `EXPERIMENT_TAG`,
`DRIFTFLOW_TIME_SAMPLING`, or `DRIFTFLOW_ENDPOINT_REPLAY`.

Training launchers print the resolved GPU, checkpoint, output, log, and W&B settings.
Smoke runs print every loss/checkpoint event; pilot runs print every 20th loss plus
checkpoint events. Any failed DDP subprocess prints the first underlying traceback
context rather than only the final TorchElastic wrapper summary.
Both launchers run a single-process dependency preflight before starting four DDP
workers.
