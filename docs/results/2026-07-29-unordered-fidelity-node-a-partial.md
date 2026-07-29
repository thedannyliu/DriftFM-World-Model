# Node A fixed-candidate depth result

Status: **partial, 15/16 rows complete**. The missing row is k1-grid25 NFE8.
This is a 20-seed Push-T discovery result, not a confirmatory or cross-task claim.

## Provenance

| Field | Value |
| --- | --- |
| Report time | `2026-07-29T01:30:50+00:00` |
| Host | `run330254-sam4-2` |
| Reporting checkout | `3f7ba3b438780a3e96bdc6c0d5a00c58511b9473` |
| Last code-changing ancestor | `791482410b03a1eda4c86b8abe99708da91174c4` |
| Task / policy | Push-T / Diffusion Policy `ep100` |
| Candidates | 32, fixed across NFE within family |
| Proposal RNG base seed | 5 |
| Hardware | one company node, 4xH100 |
| Trials per row | 20 environment test seeds, indices 0--19 |
| W&B project | `driftfm-unordered-fidelity-company` |
| Terminal report | `/user-volume/driftworld/logs/unordered-fidelity-node-a-report-20260729-013050.txt` |

Episode maximum IoU is the primary task metric. The candidate audit branches the
first action chunk through the real simulator and records score rank correlation,
selected-versus-oracle regret, and exact-oracle selection. Positive IoU/rank/oracle
changes are better; negative regret changes are better.

## Per-row results

| Family | NFE | Mean IoU | 95% CI | Success | Latency (s) | Rank GT | Regret | Oracle | W&B |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| k1-grid25 | 1 | 0.683641 | [0.611974, 0.759410] | 0.15 | 0.778966 | 0.082315 | 0.009659 | 0.10 | `naggjymg` |
| k1-grid25 | 2 | 0.758368 | [0.682425, 0.831903] | 0.10 | 1.466363 | 0.086429 | 0.014504 | 0.10 | `jkdf6d9p` |
| k1-grid25 | 4 | 0.779644 | [0.701974, 0.853145] | 0.15 | 2.848645 | 0.086022 | 0.014504 | 0.10 | `tp8dkn1u` |
| k1-grid25 | 8 | **missing** | — | — | — | — | — | — | — |
| k32 | 1 | 0.767752 | [0.692714, 0.838666] | 0.20 | 0.772155 | 0.075921 | 0.008055 | 0.15 | `yeo90dyr` |
| k32 | 2 | 0.752107 | [0.685585, 0.816868] | 0.10 | 1.467829 | 0.081378 | 0.014504 | 0.10 | `c4h8ik2d` |
| k32 | 4 | 0.776486 | [0.706327, 0.842977] | 0.15 | 2.848405 | 0.078527 | 0.012900 | 0.15 | `hmheg6tu` |
| k32 | 8 | 0.790843 | [0.717915, 0.857170] | 0.20 | 5.608143 | 0.080767 | 0.014504 | 0.10 | `ygwgtn3f` |
| joint-k16 | 1 | 0.710147 | [0.633120, 0.787626] | 0.15 | 0.777816 | 0.070341 | 0.009659 | 0.10 | `sopwdwcr` |
| joint-k16 | 2 | 0.734656 | [0.661017, 0.808506] | 0.15 | 1.469894 | 0.079871 | 0.009659 | 0.10 | `1i7j5y7r` |
| joint-k16 | 4 | 0.732149 | [0.651719, 0.809160] | 0.10 | 2.849844 | 0.084800 | 0.009659 | 0.10 | `vt7nqcfb` |
| joint-k16 | 8 | 0.687117 | [0.614420, 0.758028] | 0.00 | 5.617980 | 0.082641 | 0.009659 | 0.10 | `a5s7dx47` |
| deep-base-k16 | 1 | 0.736336 | [0.664795, 0.806010] | 0.10 | 0.780760 | 0.082315 | 0.011046 | 0.10 | `p00ifsg5` |
| deep-base-k16 | 2 | 0.757932 | [0.685275, 0.826163] | 0.10 | 1.474391 | 0.082193 | 0.011046 | 0.10 | `t1ktcsj3` |
| deep-base-k16 | 4 | 0.771506 | [0.702299, 0.837605] | 0.15 | 2.858676 | 0.083781 | 0.011046 | 0.10 | `1tnywf0o` |
| deep-base-k16 | 8 | 0.749913 | [0.678589, 0.817419] | 0.05 | 5.622408 | 0.084555 | 0.011046 | 0.10 | `6626vb9h` |

## Paired NFE-minus-NFE1 episode results

All available families report matching first-decision policy-action hashes and
ground-truth candidate rewards across NFE.

| Family | Comparison | Mean ΔIoU | Paired 95% CI | W/T/L |
| --- | --- | ---: | --- | ---: |
| k1-grid25 | NFE2 - NFE1 | +0.074726 | [0.015204, 0.141392] | 9/4/7 |
| k1-grid25 | NFE4 - NFE1 | +0.096003 | [0.025093, 0.175148] | 10/4/6 |
| k1-grid25 | NFE8 - NFE1 | **missing** | — | — |
| k32 | NFE2 - NFE1 | -0.015645 | [-0.088043, 0.055874] | 6/3/11 |
| k32 | NFE4 - NFE1 | +0.008734 | [-0.076550, 0.088324] | 7/4/9 |
| k32 | NFE8 - NFE1 | +0.023091 | [-0.051739, 0.097269] | 8/3/9 |
| joint-k16 | NFE2 - NFE1 | +0.024509 | [-0.031365, 0.098927] | 5/6/9 |
| joint-k16 | NFE4 - NFE1 | +0.022002 | [-0.025799, 0.083184] | 5/4/11 |
| joint-k16 | NFE8 - NFE1 | -0.023030 | [-0.072677, 0.022496] | 5/5/10 |
| deep-base-k16 | NFE2 - NFE1 | +0.021596 | [-0.024213, 0.062043] | 10/6/4 |
| deep-base-k16 | NFE4 - NFE1 | +0.035170 | [-0.018231, 0.088103] | 10/5/5 |
| deep-base-k16 | NFE8 - NFE1 | +0.013577 | [-0.045500, 0.073255] | 10/4/6 |

Only k1-grid25 NFE2 and NFE4 clear zero at 20 trials. No single depth is the
descriptive task optimum across all completed families: k1 currently peaks at NFE4,
k32 at NFE8, joint at NFE2, and deep at NFE4.

## The first-decision audit is not an adequate mechanism target

The raw audit reveals an important measurement limitation:

1. Ground-truth candidate rewards have zero variance in 11/20 first decisions.
   Rank correlation is therefore defined on only nine states per row.
2. On those nine states, individual rank correlations range from strongly positive
   (about `+.85`) to negative (about `-.47`). Their mean near `.08` hides this
   bimodality and is not a calibrated ranking score.
3. Joint and deep have exactly the same first-decision regret and oracle-selection
   vectors at every NFE. K1 and k32 change regret on only one of 20 states. These
   diagnostics cannot explain the materially different full-episode IoUs.
4. Exact-oracle selection treats one `argmax` as the oracle even when several
   candidates tie. With sparse immediate rewards, the reported 10--15% oracle rate
   is not a reliable standalone accuracy measure.

The bounded interpretation is that NFE changes full-episode planning outcomes, but
the present one-action-chunk audit does not identify why. Later receding-horizon
decisions, where the block is moving and candidate rewards have variance, may be the
relevant states.

## Compute-allocation implication

Comparing this fixed-candidate result with Node B makes the practical result sharper:

- k1 32x4 reaches IoU `.77964` at 2.85 s, while 64x1 reaches `.78725` at 1.56 s;
- k32 32x8 reaches `.79084` at 5.61 s, while 64x1 reaches `.79145` at 1.56 s;
- joint 64x1 (`.75860`) exceeds every fixed-32 depth, with latency close to NFE2;
- deep 64x1 (`.76613`) is close to the NFE4 peak (`.77151`) at roughly half latency.

Thus high NFE can help a selected family, but proposal breadth matches or exceeds the
observed gain much more efficiently. This supports a breadth-first planning baseline,
not a universal claim that extra NFE is harmful.

## Decision and next gate

- A read-only W&B audit using the recorded PACE Python 3.12 environment found all
  15 listed runs in `finished` state. Each has one aggregate history row, and the
  summary IoU/rank/regret/oracle values agree with the shared markers.
- Complete the missing k1 NFE8 row and the single missing rows on Nodes B/C/D for
  protocol closure.
- Do not scale the current first-decision rank/regret audit or train a gate from it.
- Confirm the primary full-episode NFE effect on 80 untouched environment indices
  20--99, paired across NFE, and replicate with ep300.
- A mechanism follow-up must evaluate common candidates on common later-episode
  states or use a longer-horizon counterfactual return. It should be implemented and
  smoke-tested before consuming another large wave.
