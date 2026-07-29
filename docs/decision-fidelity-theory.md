# Decision fidelity under untrusted generative evaluators

Status: proof audit and theorem draft, 2026-07-29.

This document separates mathematical facts from empirical hypotheses for the
proposed decision-calibrated compute paper. A failed proof is an acceptable and
informative outcome. Statements must not be strengthened merely to preserve the
paper story.

## Claim-status convention

| Label | Meaning |
| --- | --- |
| **Proved** | The statement follows from the definitions under the stated assumptions. |
| **Conditional** | The proof is valid only if an explicit assumption is satisfied; that assumption must be tested. |
| **Conjecture** | The statement is plausible but currently has neither a proof nor sufficient evidence. |
| **Refuted** | A counterexample or experiment invalidates the stated universal claim. |
| **Open** | A useful stronger statement for which no valid proof is currently known. |

When a proof attempt fails, record the failed step and either provide a
counterexample or add the weakest assumption that repairs it. If the repair
assumption is empirically false, the theorem and any method depending on it are
killed.

## 1. Formal setting

Fix a decision state \(x\). Let:

- \(\mathcal A(x)\) be the set of executable action chunks;
- \(U_x(a)\in[0,1]\) be the expected true continuation utility of action \(a\)
  under a fixed continuation and randomness protocol;
- \(\pi(\cdot\mid x)\) be a proposal policy;
- \(a_1,a_2,\ldots\sim\pi(\cdot\mid x)\) be a coupled proposal sequence, and
  \(A_B=\{a_1,\ldots,a_B\}\) its nested prefix;
- \(\widehat U_{x,d}(a)\) be the score obtained from a learned world model at
  inference depth \(d\);
- \(\widehat a_{B,d}\in\arg\max_{a\in A_B}\widehat U_{x,d}(a)\) be the selected
  action, with a fixed tie-breaking rule.

Define

\[
U_x^\star=\sup_{a\in\mathcal A(x)}U_x(a),\qquad
U_{x,B}^\star=\max_{a\in A_B}U_x(a).
\]

The theoretical optimum \(U_x^\star\) is generally unobservable. In simulator
experiments with a master pool \(A_M\), report the estimable pool-relative
quantity

\[
C_M(B\mid x)=U_{x,M}^\star-U_{x,B}^\star
\]

in addition to any estimate of global coverage.

Inference depth is called a **trusted fidelity order** only if an explicit error
certificate becomes tighter with depth. A larger NFE is otherwise merely a
different, more expensive evaluator.

## 2. Exact regret decomposition

### Theorem 1 — Coverage plus selection regret

**Status: Proved.**

For every state, sampled candidate set, evaluator, and selected candidate,

\[
\begin{aligned}
R(B,d\mid x)
&=U_x^\star-U_x(\widehat a_{B,d})\\
&=\underbrace{U_x^\star-U_{x,B}^\star}_{C(B\mid x)}
 +\underbrace{U_{x,B}^\star-U_x(\widehat a_{B,d})}_{S(B,d\mid x)}.
\end{aligned}
\]

Both terms are non-negative.

**Proof.** Add and subtract \(U_{x,B}^\star\). Because
\(A_B\subseteq\mathcal A(x)\), \(U_x^\star\ge U_{x,B}^\star\). Because
\(\widehat a_{B,d}\in A_B\),
\(U_{x,B}^\star\ge U_x(\widehat a_{B,d})\). \(\square\)

The identity also holds after taking expectations over states, proposal
sampling, evaluator randomness, and environment randomness.

### Corollary 1.1 — Nested breadth improves coverage

**Status: Proved.**

For a coupled proposal sequence,

\[
C(B+1\mid x)\le C(B\mid x).
\]

**Proof.** \(A_B\subseteq A_{B+1}\), so
\(U_{x,B+1}^\star\ge U_{x,B}^\star\). \(\square\)

This monotonicity applies only to coverage regret. It does not imply that total
regret decreases when an imperfect evaluator must select from the enlarged set.

### Proposition 1.2 — More proposals can worsen total regret

**Status: Proved by counterexample.**

Let three nested proposals have true utilities and evaluator scores

\[
\begin{array}{c|ccc}
 &a_1&a_2&a_3\\ \hline
U_x&0.8&1.0&0\\
\widehat U&0.8&0.7&0.9.
\end{array}
\]

With \(B=1\), the planner selects \(a_1\) and has total regret \(0.2\).
With \(B=3\), coverage regret falls to zero, but the planner selects \(a_3\)
and total regret rises to \(1.0\).

Therefore breadth has a structurally monotone effect on coverage but no
unconditional monotone effect on the final decision. This is the world-model
analogue of maximization against an imperfect verifier.

## 3. Prediction accuracy does not order decision fidelity

### Proposition 2 — Lower score MSE can make the decision worse

**Status: Proved by counterexample.**

For two candidates with true utilities \(U=(1,0)\), consider

\[
\widehat U^{(1)}=(0.9,-1.0),\qquad
\widehat U^{(2)}=(0.49,0.51).
\]

The first evaluator selects the correct candidate and has MSE

\[
\tfrac12[(0.9-1)^2+(-1-0)^2]=0.505.
\]

The second has lower MSE

\[
\tfrac12[(0.49-1)^2+(0.51-0)^2]=0.2601
\]

but selects the wrong candidate. A third evaluator equal to \(U\) has zero MSE
and selects correctly, so even a strictly decreasing MSE sequence can have a
correct-wrong-correct decision sequence.

Consequently, lower frame LPIPS, pose error, or average utility MSE cannot by
itself establish a fidelity order for action selection.

### Corollary 2.1 — No gap-free MSE threshold guarantees ranking

**Status: Proved.**

For any \(\epsilon>0\), choose \(0<\delta<\epsilon\),
\(U=(\delta,0)\), and \(\widehat U=(0,\delta)\). The evaluator is wrong while
its MSE is \(\delta^2<\epsilon^2\). Therefore an absolute average-error threshold
without a decision-gap condition cannot guarantee correct ranking.

## 4. What cannot be guaranteed without a fidelity assumption

### Theorem 3 — No-free-elimination under an untrusted evaluator

**Status: Proved.**

Let an elimination rule \(E:\mathbb R^B\rightarrow 2^{\{1,\ldots,B\}}\) use
only cheap evaluator scores and discard at least one candidate for some score
vector \(s\). If the model class imposes no relationship between \(s\) and true
utility, then \(E\) cannot guarantee retaining the true best candidate.

**Proof.** For the score vector \(s\), choose an index \(j\notin E(s)\) that the
rule discards. Construct an admissible utility vector with \(U_j=1\) and
\(U_i=0\) for all \(i\ne j\). Because there is no assumed relationship between
scores and utility, this world is allowed and \(E\) discards its unique optimum.
\(\square\)

For a randomized rule, any candidate discarded with positive probability can be
made the unique optimum, yielding positive failure probability.

This theorem is deliberately modest. It is an assumption audit, not by itself a
novel bandit result: compute-saving elimination requires either a trusted bias
bound, valid uncertainty intervals, or environment-calibrated structure.

### Proposition 3.1 — Depth disagreement is observationally non-identifying

**Status: Proved.**

Suppose two depths select different candidates \(i\) and \(j\) from the same
observed score table. Without a constraint linking either score table to true
utility, there are two observationally identical worlds: one in which \(i\) is
the unique optimum and one in which \(j\) is the unique optimum. No scheduler
using only those scores can be correct in both.

This is why counterfactual environment returns are necessary. Depth disagreement
alone cannot reveal which depth is more faithful.

## 5. Conditions that recover safe decision guarantees

### Theorem 4 — Uniform error controls selection regret

**Status: Conditional.**

Assume that, with probability at least \(1-\alpha\),

\[
\max_{a\in A_B}\left|\widehat U_{x,d}(a)-U_x(a)\right|
\le \varepsilon_d.
\]

Then, on that event,

\[
S(B,d\mid x)\le 2\varepsilon_d.
\]

If the true best candidate has gap

\[
\Delta_B=U_{x,B}^\star-\max_{a\ne a_B^\star}U_x(a)
>2\varepsilon_d,
\]

the evaluator selects the true best candidate.

**Proof.** Let \(a_B^\star\) be the true best and \(\widehat a\) the score
maximizer. Then

\[
U_x(a_B^\star)
\le\widehat U_{x,d}(a_B^\star)+\varepsilon_d
\le\widehat U_{x,d}(\widehat a)+\varepsilon_d
\le U_x(\widehat a)+2\varepsilon_d.
\]

The regret bound follows. If the selected candidate were not optimal, its true
gap would be at most \(2\varepsilon_d\), contradicting the stated gap.
\(\square\)

A trusted fidelity ladder follows only if valid radii satisfy
\(\varepsilon_{d+1}\le\varepsilon_d\). Current Push-T results do not establish
such radii, and average prediction metrics cannot substitute for them.

### Theorem 5 — Pessimistic selection and safe switching

**Status: Conditional.**

Suppose intervals \([L_i,H_i]\) simultaneously cover every candidate utility
with probability at least \(1-\alpha\). Let

\[
\widetilde i\in\arg\max_i L_i.
\]

On the coverage event,

\[
U_{x,B}^\star-U_x(a_{\widetilde i})
\le \max_i(H_i-L_i).
\]

Further, let \(b\) be a fixed baseline decision and \(c\) a proposed alternative.
Executing \(c\) only when \(L_c\ge H_b\), and otherwise executing \(b\), is never
worse than \(b\) on the simultaneous-coverage event.

**Proof.** Let \(i^\star\) be the true best candidate. Since
\(L_{\widetilde i}\ge L_{i^\star}\),

\[
U_{i^\star}-U_{\widetilde i}
\le H_{i^\star}-L_{\widetilde i}
\le H_{i^\star}-L_{i^\star}
\le\max_i(H_i-L_i).
\]

For safe switching, \(U_c\ge L_c\ge H_b\ge U_b\). \(\square\)

The proof is simple; the hard research problem is obtaining intervals that
remain calibrated under task, policy, candidate-count, and model shifts.

### Corollary 5.1 — Pool-relative regret certificate

**Status: Conditional.**

For a master proposal pool \(A_M\), pessimistic selection from \(A_B\) satisfies

\[
U_{x,M}^\star-U_x(a_{\widetilde i})
\le C_M(B\mid x)+\max_{i\le B}(H_i-L_i)
\]

on the coverage event. This exposes the two quantities an allocator would need
to reduce: unseen proposal coverage and uncertainty about the selected set.

## 6. Value of one additional computation

Let \(H\) be the current decision history, let \(\mathcal A(H)\) be its currently
available candidates, and define its conditional Bayes decision regret as

\[
\rho(H)=\min_{a\in\mathcal A(H)}
\mathbb E[U_x^\star-U_x(a)\mid H].
\]

A query \(q\) is either:

- **Expand:** sample and cheaply evaluate a new proposal;
- **Refine:** obtain another evaluator observation for an existing proposal.

Let \(Y_q\) be its random outcome and \(c(q)\) its measured cost.

### Proposition 6 — Exact one-step value-of-computation rule

**Status: Proved for a one-query horizon; open for the sequential greedy policy.**

If at most one additional query may be purchased and compute has Lagrange price
\(\lambda\), the Bayes-optimal choice maximizes

\[
V(q\mid H)
=\rho(H)-\mathbb E[\rho(H,Y_q)\mid H]-\lambda c(q),
\]

and stops when every query has non-positive value.

**Proof.** Stopping has conditional loss \(\rho(H)\). Query \(q\) has conditional
loss \(\mathbb E[\rho(H,Y_q)\mid H]+\lambda c(q)\). Comparing these losses gives
the rule. \(\square\)

Repeatedly applying this one-step rule is not known to be globally optimal.
Such a proof would require additional structure such as adaptive submodularity
or a dynamic-programming argument. We currently have neither. The method must
therefore be described as a calibrated approximation unless a stronger proof is
completed.

## 7. Open claims that experiments may kill

| ID | Claim | Current status | What failure means |
| --- | --- | --- | --- |
| C1 | Learned world-model depth lacks a stable decision-fidelity order across states, policies, tasks, and model families. | **Conjecture.** Push-T motivates it but does not prove it. | If one depth dominates direct selection regret, use that static depth and drop the adaptive-depth thesis. |
| C2 | Some states benefit more from expansion and others from refinement. | **Conjecture.** The immediate reward audit was degenerate. | If the oracle expand/refine switch has negligible value, an allocator is unnecessary. |
| C3 | Deployment-time evidence can predict marginal regret reduction with valid lower bounds under block shift. | **Conjecture.** | If calibration fails leave-one-task/policy transfer, stop method development. |
| C4 | A sequential expand/refine/stop allocator beats the best static frontier at matched latency. | **Conjecture.** | If `64x1`, `32x2`, or another static point matches it, report the negative result and do not claim DCCA. |
| C5 | The phenomenon extends beyond the current DFM conversion. | **Conjecture.** | If a diffusion world model has an ordered depth ladder, narrow the result to DriftFlowWorld. |

The universal claim “more NFE always improves decisions” is **refuted as a
mathematical statement without additional assumptions** and unsupported in the
held-out Push-T data. The opposite universal claim “more NFE always harms
decisions” is also unsupported.

## 8. Proof-failure log

| Attempt | Status | Failure or missing step | Resolution |
| --- | --- | --- | --- |
| Derive monotone decision quality from lower reconstruction loss | **Refuted** | Proposition 2 gives a lower-MSE wrong-ranking counterexample. | Require decision-gap-aware uniform utility bounds or direct selection evidence. |
| Prove higher NFE is a fidelity ladder from greater compute alone | **Failed** | Compute count places no constraint on evaluator bias or ranking. | Treat depths as separate evaluators until Theorem 4's error-order assumption is measured. |
| Prove cheap-to-expensive elimination is safe without bias bounds | **Refuted** | Theorem 3 constructs an eliminated optimum. | Use calibrated intervals or accept non-zero failure risk. |
| Prove greedy expand/refine selection is globally optimal | **Open** | One-step value of computation does not imply multi-step optimality. | Evaluate it as an approximation or identify sufficient adaptive-submodular structure. |
| Prove DCCA cannot hurt a static baseline | **Conditional only** | It requires simultaneous interval coverage under distribution shift. | Use Theorem 5's safe switch and measure its actual miscoverage. |

## 9. What would count as a theory contribution

Theorem 1 and the counterexamples are necessary exposition but are unlikely to
be sufficient novelty by themselves. A stronger paper-level theory result would
need at least one of:

1. a finite-sample regret guarantee for allocation across biased, non-ordered
   evaluators using environment-calibrated intervals;
2. a lower bound showing the cost of identifying the best candidate when
   evaluator bias is unknown but calibration data are available;
3. sufficient conditions under which greedy expand/refine allocation is
   near-optimal;
4. a shift-robust no-harm guarantee whose assumptions are satisfied on held-out
   robot tasks.

These are open targets, not promised results. The mandatory experiments that
test their assumptions are specified in
[`decision-fidelity-experiments.md`](decision-fidelity-experiments.md).
