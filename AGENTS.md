Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Testing Guidelines

There is no formal unit-test suite. Validate with the smallest workflow that exercises your change: import/package checks for library edits, manifest generation for experiment tooling, and one smoke row or Slurm array for training/eval changes. Record task, seed, manifest path, GPU type, checkpoint/output directory, and any W&B project in PR notes.

## Commit & Pull Request Guidelines

Recent history uses imperative subjects: `Add ...` for new utilities or experiment flows, `Document ...` and `Record ...` for status/results, `Fix ...` for behavior corrections, and `Update ...` for refreshed job records. Keep commits narrow and separate code from large generated artifacts. PRs should state the research or pipeline impact, list validation commands or smoke runs, link relevant docs, and call out cluster-resource, checkpoint, dataset, or W&B implications.

Use git like machine learning engineer to update. For example, when there is new development, open a new branch a develop there then merge it back.
Also use wandb to record important metrics, when there is new experiments, open a project to record. Please make each training process continuable, and when resume training, please use same wandb run to record.

## Cluster & QOS Policy

Current runs should target PACE-Phoenix by default. Use the `embers` QOS for GPU jobs because it is not charged to the account, though it has lower priority. Do not submit with `inferno` unless the user explicitly approves it; `inferno` has normal priority but incurs account charges.

Default Slurm account for H100、H200、A100 work: `gts-agarg35`.
Default Slurm account for L40S work: `gts-agarg35-ideas_l40s`.

Usually current node is home node, and don't have GPUs. GPU must be accessed from submitting a job.
Don't do any GPU job, env setup at $HOME, keep it clean.
Construct venv or conda env and record it in this doc, and develop on the virtual environment afterwards.

SAM2 PoC environment (PACE CUDA 12.8):
`/storage/scratch1/9/eliu354/sra_sam2/envs/pace-cu128-py312-v1`.
It is created by `scripts/create_pace_env.sh` from `requirements/pace-cu128.lock.txt`; the resulting
`pip-freeze.txt` stays beside the environment. Official SAM2 source and checkpoints are referenced
read-only through experiment manifests rather than installed into or modified from adjacent repos.

## 5. AI Research
As a senior & rigorous AI researcher, always check with fact. Come up with hypothesis -> design experiments -> record metrics -> gain research signal -> further shape our research idea and directions.

Please document and keep updating experiments tables, each table should answer it's own question.

Design documentation system under docs/ and record how it's designed on this doc.

Focus on TOP AI venue's best papers' criteria, do mearningful work. Have research novelty. Keep idea tide and elegant, don't just integrate different ideas and think it'll be novel. Good idea should shape people's perspective. Do a great and thorough literature review and document it. Always keep updating but keep it concise and precise.

## DriftFlowWorld Project Layout

Research documentation lives under `docs/`:

- `docs/research-protocol.md`: locked claims, gates, and evaluation protocol.
- `docs/literature-review.md`: concise related-work and novelty audit.
- `docs/experiments.md`: append-only hypothesis/experiment/result tables.
- `docs/manifests/`: versioned experiment inputs; generated outputs stay on scratch.

The PACE environment and all large artifacts live under
`/storage/scratch1/9/eliu354/driftflowworld/`. The canonical environment prefix is
`envs/pace-cu128-py312-v1`; data, checkpoints, W&B files, caches, and Slurm output use
the sibling `data/`, `checkpoints/`, `runs/`, `cache/`, and `slurm_logs/` directories.
Create the environment with `scripts/create_pace_env.sh`; its `pip-freeze.txt` is
written beside the environment. Never commit datasets, checkpoints, credentials, or
generated metric arrays.
