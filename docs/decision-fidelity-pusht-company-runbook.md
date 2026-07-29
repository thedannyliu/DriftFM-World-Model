# Push-T decision-fidelity company runbook

Status: frozen two-node launch protocol, 2026-07-29. This file allocates the
first two sequential gates in
[`decision-fidelity-experiments.md`](decision-fidelity-experiments.md). The
mathematical claims may fail, the experiment gates may fail, and either outcome
must remain in the result ledger.

## Scope and environment

This run uses two independent company nodes, each with four H100 GPUs. The nodes
share `/group-volume` and `/user-volume`, but each fresh container must install
and verify its own Python packages.

- Required image:
  `ngc24.06-ub22-py3.10-cu12.5-cudnn9.1-pytorch2.4-deepspeed0.14-8packing`.
- Checkout: `/user-volume/repo/DriftFM-World-Model`.
- Shared assets and atomic completion markers:
  `/group-volume/danny-dataset/driftworld`.
- Logs, result arrays, package cache, and W&B staging:
  `/user-volume/driftworld`.
- W&B project: `driftfm-decision-fidelity-pusht-company`.
- No venv or Conda environment is created. The active image's Python 3.10,
  PyTorch 2.4, torchvision, and CUDA packages are retained.

`company/setup_decision_fidelity_node.sh` runs `company/setup.sh` separately on
each node. It removes GUI OpenCV wheels, installs the pinned headless wheel,
requires Pymunk 7's `Space.on_collision`, and verifies exactly four visible
GPUs. If the shared Push-T assets are already complete it skips downloads. If
they are incomplete, only Node A is allowed to repair them; Node B exits rather
than racing a second download.

Each setup keeps a node-specific log:

```text
/user-volume/driftworld/logs/decision-fidelity/setup-<role>-<hostname>.log
```

## Allocation

| Company node | Ledger stage | Frozen question | Rows | Role of the result |
| --- | --- | --- | ---: | --- |
| Node A | N0 | On held-out Push-T episodes, does `64x1` beat the existing `32x2`, and is there state-wise oracle switching headroom? | 8 | Research evidence and the allocator continue/stop gate. |
| Node B | N1 infrastructure smoke | Can common later states and longer counterfactual action returns be reproduced across NFE before paying for the full 200-state N1 pilot? | 4 | Mechanism validation only; never paper evidence. |

The queues are sequential within a node and shard every row over its four local
GPUs. Shared marker locks make rerunning the same command resumable and prevent
duplicate rows.

## Node A — N0 held-out breadth confirmation

### Frozen design

- Test indices: `20:100`, the same untouched 80 episodes as U5.
- Frozen world-model families: K1/grid25 best seed 2, K32 latest seed 2,
  joint-K16 latest seed 2, and deep-base-K16 best seed 1.
- Proposal policies: ep100 primary and ep300 replication.
- New arm: 64 proposals at NFE1.
- Comparator: the already complete matching U5 arm with 32 proposals at NFE2.
- Eight new rows total; no training and no new checkpoint.

The original action-diversity transform is candidate-pool dependent. Therefore
the new arm anchors that transform to candidates 1--32 and uses a 32-candidate
warm-up draw before the first ranked decision. For every episode, the SHA-256
hash of the new arm's first 32 candidates at the first comparable decision must
equal the old 32x2 full-candidate hash. A mismatch invalidates the paired row.
This is a hard provenance gate, not a warning.

### Primary outputs and gate

For each family-policy block, report paired `64x1 - 32x2` IoU, success, and
measured planning latency. Across all preregistered blocks report the
state-wise hindsight switch headroom

\[
G_{\mathrm{switch}}
=\mathbb E[\max(Y_{64x1},Y_{32x2})]
-\max(\mathbb E[Y_{64x1}],\mathbb E[Y_{32x2}]).
\]

- Continue to the formal N1 pilot only if the 95% upper confidence bound on
  \(G_{\mathrm{switch}}\) exceeds `0.01` IoU and all prefix-pairing checks pass.
- Stop allocator work if even the oracle switch lacks that headroom.
- If one static arm wins, retain it as the baseline; that is not evidence for
  an adaptive method.

## Node B — N1 later-state label smoke

### Why this is a smoke

The previous immediate audit was constant in 11/20 states. Before running the
full stratified N1 dataset, this smoke tests whether the simulator clone,
candidate pairing, and longer utility target work at all. It does not satisfy
N1's task, state-count, continuation, or replication requirements.

### Frozen design

- Test indices: `100:116`, disjoint from N0.
- One representative, fully evaluated frozen model: K32 latest seed 2.
- Policies: ep100 and ep300.
- Candidate pool: 16, with the first 8 forming the nested breadth subset.
- Depths: NFE1 and NFE4, giving four rows.
- Execution trajectory: always execute policy candidate 0. World-model ranking
  cannot change the environment state, so the NFE1/NFE4 rows must visit common
  states.
- Audit at most four later MPC decisions per episode.
- Candidate utility: maximum Push-T IoU from all 15 available candidate actions,
  followed by 16 repeated hold-target steps in a cloned simulator.
- Common candidate and state hashes are required across NFE1/NFE4.

This hold-target continuation is deliberately simple and frozen before results.
It is only a non-degeneracy probe. If it passes, the formal N1 pilot replaces it
with the preregistered continuation policy and stratified state dataset.

### Smoke gates

The mechanism passes only when:

1. all paired state and candidate hashes match;
2. clone replay is deterministic;
3. at least 90% of audited states have non-zero candidate-utility variance;
4. both candidates 9--16 (Expand) and an NFE4-selected candidate (Refine) can
   improve oracle regret in at least 10% of usable states; and
5. no row has missing or non-finite utility.

Failure stops the formal N1 launch. It triggers a target/protocol diagnosis, not
more model training.

## Launch sequence

Do not run a queue from an unverified commit. On each node, pull the same commit,
run its local setup, log in to W&B if necessary, print the plan, and then launch.
The exact pasteable commands are also printed in the implementation handoff.

Node A:

```bash
cd /user-volume/repo/DriftFM-World-Model
git pull --ff-only origin main
bash company/setup_decision_fidelity_node.sh node-a
wandb login --relogin
bash company/run_decision_fidelity_pusht.sh node-a --print-plan
bash company/run_decision_fidelity_pusht.sh node-a
```

Node B:

```bash
cd /user-volume/repo/DriftFM-World-Model
git pull --ff-only origin main
bash company/setup_decision_fidelity_node.sh node-b
wandb login --relogin
bash company/run_decision_fidelity_pusht.sh node-b --print-plan
bash company/run_decision_fidelity_pusht.sh node-b
```

Compact shared progress:

```bash
python3 company/status_decision_fidelity_pusht.py all
```

## Result recording

Completion markers live under the shared experiment root and are written
atomically. Raw arrays and full logs remain under the runtime root. After both
queues stop, create one immutable dated file under `docs/results/` containing:

- code commit and both hostnames;
- environment-preflight JSON and setup-log paths;
- every W&B run ID and marker path;
- N0 prefix-pairing status and paired primary contrasts;
- N0 `continue` or `stop` decision;
- N1 smoke hash, non-degeneracy, and Expand/Refine gate results;
- any traceback or failed row.

Do not merge the smoke with formal N1 data, and do not silently rerun a failed
protocol with changed thresholds.
