# Weekend queue pruning decision — 2026-07-28

## Evidence boundary

This decision uses the shared company-cluster snapshot captured at commit `d797a0f`
on 2026-07-28 18:27 UTC, the completed 12-checkpoint mechanism audit, and the
corrected action-routing refresh. The PACE home node cannot inspect or signal the
company cluster's `run*` containers, so process termination there must be executed
inside each affected node.

## Decision

Stop all remaining work in the four aggressive weekend training queues:

| Queue | Stop scope | Evidence |
| --- | --- | --- |
| node-a | Current `wknd-a-grid25` extension and all waiting winner/runner-up seeds | Rollout score selected 30k; the 300k result had collapsed, with LPIPS `0.0182/0.0246/0.0314/0.0318` and substantially worse block error. |
| node-b | Current `wknd-b-grid2-p125` extension and all waiting winner/runner-up seeds | Rollout score selected 60k; the 300k result regressed at every reported NFE. |
| node-c | Any runner-up extension and all waiting seeds | The 400k winner collapsed, and the preregistered composition-amplification gate failed across the audited checkpoint families. More composition-depth/source-replay training does not test the surviving hypothesis. |
| node-d | Current grid-only extension and every waiting source/joint/K-scaling arm | Three base seeds already show deep-training regression. The grid-only 60k result is worse than its selected 30k point, while completed 100k grid/source families already reject the proposed composition mechanism. |

Do not delete completed checkpoints, metric JSON, W&B runs, or logs. The retained
30k/60k rollout-best checkpoints are evidence for early stopping and remain useful
for paired evaluation.

Keep the fifth node only if it is running
`company/run_advantage_frontier_queue.sh node-b`. This queue performs frozen
1000-video evaluations without further training and supplies the missing K32,
joint-K16, and deep-base families. Stop it only after all eight markers are present
or if it has failed and cannot create another marker.

No additional node-D mechanism audit is needed to decide the rejected composition
gate: even four positive missing runs cannot reach the preregistered 75% threshold.
It is optional protocol closure, not a reason to retain a GPU node.

## Research implication

The stopped queues target the rejected idea that deeper grid/source composition will
produce monotonic test-time scaling. The live research lead is narrower:

> A trained finite two-step map can improve risk on some checkpoint families, while
> deeper application and prolonged post-training accumulate error. Pre-inference
> action difficulty may route the second evaluation, but this must be tested with a
> preregistered split, same-code official baseline, confidence intervals, latency,
> and a second task.

Accordingly, compute should move from deep Push-T extensions to locked evaluation,
official-baseline parity, downstream policy metrics, and Robomimic replication.

## PACE cleanup

The unrelated stale PACE dependency chain was cancelled on 2026-07-28:

- `11364799` (`dfw-pilot10k`)
- `11364806` (`dw-control10k-eval`)
- `11364808` (`dfw-pilot10k-eval`)
- `11362985` (`dfw-posttrain`)

The pilot depended on failed job `11364127`; all downstream jobs were therefore
unable to start and could not provide additional evidence.
