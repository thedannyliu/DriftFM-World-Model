# Action-routing refresh — 2026-07-28

## Scope

Source: `company/refresh_advantage_frontier.sh node-a` at commit `52c8ad5`,
run on `run332013-sam4-3`.

- The command re-summarized retained raw metrics from eight completed 1000-video
  evaluations. It did not rerun inference or modify model checkpoints.
- Development examples are even indices; test examples are odd indices.
- The action-path threshold is the development-set median.
- Test examples above the threshold use NFE2; the rest use NFE1.
- Every checkpoint routes 49% of test examples to NFE2.
- `route_mse` and `route_vertex` are relative changes against random NFE1/NFE2
  allocation with the same expected compute. Negative is better.
- The same W&B run IDs were resumed.

This split was introduced after the original ordered half split produced exact-zero
route effects, so these results are exploratory rather than preregistered evidence.

## Per-checkpoint results

| Checkpoint | Family | NFE2 fraction | Route MSE | Route vertex | Decision | W&B |
| --- | --- | ---: | ---: | ---: | --- | --- |
| K1 grid=.25 s1 latest | K1 grid=.25 | 49% | `-1.874%` | `-8.066%` | Improves both risks | `jnwjr70t` |
| K1 grid=.25 s2 latest | K1 grid=.25 | 49% | `-1.802%` | `-3.039%` | Improves both risks | `syxu74v4` |
| K1 grid=.25 s3 latest | K1 grid=.25 | 49% | `-2.362%` | `-0.309%` | Improves both risks | `rysbem5u` |
| K1 grid=.25 s1 validation-best | K1 validation-best | 49% | `+0.330%` | `-1.610%` | Mixed | `4j9ancnf` |
| K1 grid=.25 s2 validation-best | K1 validation-best | 49% | `+0.320%` | `+1.114%` | Worsens both risks | `andiirbi` |
| K1 grid=.25 s3 validation-best | K1 validation-best | 49% | `+0.945%` | `-0.333%` | Mixed | `fr0plzni` |
| K32 s1 latest | K32 | 49% | `+2.004%` | `+5.257%` | Worsens both risks | `spjgkt9f` |
| K32 s2 latest | K32 | 49% | `-0.708%` | `-2.043%` | Improves both risks | `a9zro2z0` |

## Family-level signal

| Family / checkpoint rule | Coverage | Mean route MSE | Mean route vertex | Improvement frequency MSE / vertex | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| K1 grid=.25 latest | 3/3 | **`-2.013%`** | **`-3.805%`** | **3/3 / 3/3** | Action path consistently identifies examples where spending the second model evaluation is more useful than random allocation. |
| K1 grid=.25 validation-best | 3/3 | `+0.531%` | `-0.276%` | 0/3 / 2/3 | Routing does not rescue checkpoints whose NFE2 advantage is weak or inconsistent. |
| K32 latest | 2/3 | `+0.648%` | `+1.607%` | 1/2 / 1/2 | No robust routing signal; seed 3 remains missing. |

## Interpretation

The corrected result supports a narrower form of the risk-conditional-compute
hypothesis:

1. A pre-inference action statistic contains information about the benefit of NFE2
   for the three K1/grid=.25 latest checkpoints.
2. The signal is checkpoint-family dependent. It does not generalize to
   validation-best K1 or the available K32 checkpoints.
3. Routing is therefore not a generic wrapper that repairs any transport model. It
   is useful only when training has already created a real finite-depth advantage.
4. The largest consistent effect is on block dynamics, which is more relevant than
   pixel MSE to downstream Push-T decisions.

The result does not yet establish a deployable method:

- even/odd splitting was chosen after diagnosing the first split;
- adjacent ordered examples may be correlated;
- there is no confidence interval or permutation baseline;
- the status output does not report routed LPIPS;
- no planning or policy metric has been evaluated;
- node B's eight checkpoints and K32 seed 3 are missing.

## Decision

Advance **finite-depth, risk-conditional NFE2** as a research lead, not as a finished
claim. The next locked test should preregister a group-aware split by episode/domain,
compare action routing with random and constant-NFE controls, report bootstrap
confidence intervals for LPIPS/MSE/block risk, and measure downstream planning
success at fixed wall-clock budget.

