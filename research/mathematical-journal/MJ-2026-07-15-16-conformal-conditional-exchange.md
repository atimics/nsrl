# MJ-2026-07-15-16: Conformal certificates for conditional exchange

- Date: 2026-07-15
- Status: finite-sample safety theorem established; existing surface shows
  retrospective nonvacuity only; prospective cross-source test remains open
- Extends: MJ-2026-07-15-15
- Diagnostic artifact:
  [`p10m-atomic-conformal-exchange-retrospective-v1.json`](../../benchmarks/production-model-v1/p10m-atomic-conformal-exchange-retrospective-v1.json)
- Theory checker:
  [`check-conformal-exchange-theory-v1.mjs`](../../scripts/check-conformal-exchange-theory-v1.mjs)

## Question

MJ-15 reduced the successful proposal mechanism to an exact conditional
exchange,

\[
  \Delta_d(e)=\lambda_d(e)+\rho_d(e),
  \qquad e=(B,i\to j),
\]

where `lambda` is visible from singleton probes and `rho` is the otherwise
unobserved interaction residual. How can a router use those probes to authorize
an exchange while controlling the probability that the authorized exchange is
actually harmful?

The answer does not require a stable global Ising model. It requires a
finite-sample upper prediction envelope for the residual, calibrated at the
same population level to which transfer is claimed.

## Numeric surface and notation

For a document `d`, let `L_d(S)` be the exact integer negative-log-likelihood
component after atomic action subset `S`. This entry uses Q32 values throughout:
an integer `z` represents `z 2^{-32}`. For an exchange
`e=(B,i -> j)`, with `B` disjoint from `i,j`, define

\[
\begin{aligned}
  s_d(a)&=L_d(\{a\})-L_d(\varnothing),\\
  \lambda_d(e)&=s_d(j)-s_d(i),\\
  \rho_d(e)&=\Delta_d(e)-\lambda_d(e),\\
  \Delta_d(e)&=L_d(B\cup\{j\})-L_d(B\cup\{i\}).
\end{aligned}
\]

Thus negative `Delta` is favorable. The exact Möbius formula for `rho` is
Proposition 15.1.

Let `E` be a finite candidate-exchange set frozen before calibration. Let
`phi_d` contain only features available before evaluating the candidate
multi-atom outcomes; singleton effects may be included. A proper training split
fits an arbitrary Q32-valued residual predictor `q_e(phi)`; any internal real
prediction must be deterministically converted to the frozen Q32 rule before
calibration. Conditional on that training split, every `q_e` is fixed.

For calibration unit `u`, define the simultaneous one-sided nonconformity score

\[
  A_u=\max_{e\in E}\{\rho_u(e)-q_e(\phi_u)\}.
\]

With `n` calibration units and error level `alpha`, put

\[
  k=\left\lceil(n+1)(1-\alpha)\right\rceil,
  \qquad
  Q_\alpha=\begin{cases}
    A_{(k)},&k\le n,\\
    +\infty,&k=n+1,
  \end{cases}
\]

where `A_(k)` is the `k`th smallest calibration score. The `+infinity` case is
important: it represents insufficient calibration resolution rather than a
license to substitute the sample maximum.

## Finite-sample upper envelope

**Proposition 16.1 (simultaneous split-conformal residual envelope).** Assume
the `n` calibration units and one new unit are exchangeable conditional on the
proper training split, and the score construction is fixed and permutation
symmetric. Then

\[
  \Pr\left[\forall e\in E:\
    \rho_{n+1}(e)\le q_e(\phi_{n+1})+Q_\alpha\right]
  \ge 1-\alpha.
\]

**Proof.** Apply the usual split-conformal rank argument to the `n+1`
exchangeable scalar scores `A_1,...,A_{n+1}`. With ties included on the covered
side,

\[
  \Pr[A_{n+1}\le A_{(k)}]\ge \frac{k}{n+1}\ge1-\alpha.
\]

When `k=n+1`, `Q_alpha=+infinity` makes the statement trivial. On the covered
event, the definition of the maximum gives every component inequality
simultaneously. `square`

The maximum is not decorative. Calibrating each exchange separately and then
choosing the most promising exchange would introduce a selection error unless
an additional multiplicity correction were supplied.

## A certifiable proposal operator

Let a router inspect `phi`, the singleton margins, the fitted predictors, and
`Q_alpha`, but not the new unit's candidate multi-atom losses. It may adaptively
select any exchange `e_star` in `E`, and fires only if

\[
  \lambda_{n+1}(e_\star)
  +q_{e_\star}(\phi_{n+1})+Q_\alpha<0.
\]

Otherwise it abstains and retains the control mask.

**Proposition 16.2 (unsafe-action control).** Under Proposition 16.1,

\[
  \Pr[\text{router fires and }\Delta_{n+1}(e_\star)\ge0]\le\alpha.
\]

**Proof.** On the simultaneous covered event,

\[
  \Delta_{n+1}(e_\star)
  =\lambda_{n+1}(e_\star)+\rho_{n+1}(e_\star)
  \le\lambda_{n+1}(e_\star)+q_{e_\star}(\phi_{n+1})+Q_\alpha<0.
\]

Therefore a fired non-improvement can occur only when the conformal envelope
fails, whose probability is at most `alpha`. `square`

This is an abstaining proposal operator, not a loss optimizer. It certifies a
local replacement relative to a declared control mask. Its useful operating
characteristic is the pair

\[
  (\text{unsafe-action rate},\ \text{nonvacuous firing rate}),
\]

not classification accuracy in a post-selected subgroup.

**Proposition 16.3 (bounded positive regret).** If the positive contrast of
every fired exchange is bounded by a declared real constant `R`, then the
router's positive regret relative to abstention satisfies

\[
  \mathbb E\left[
    \mathbf 1\{\text{fire}\}\,[\Delta_{n+1}(e_\star)]_+
  \right]\le\alpha R.
\]

**Proof.** The random variable is zero on every covered event by Proposition
16.2 and is at most `R` otherwise. `square`

The boundedness assumption must come from the numeric contract or a frozen
evaluation envelope. It is not implied by conformal calibration.

## The population unit must match the transfer claim

Document exchangeability within one source cannot establish a new-source
guarantee. To target source transfer, define a source panel `P_u` by a frozen
document-sampling rule and score the whole panel as

\[
  A_u^{\mathrm{src}}=
  \max_{d\in P_u}\max_{e\in E}
  \{\rho_d(e)-q_e(\phi_d)\}.
\]

**Proposition 16.4 (source-panel safety).** If the fitting sources are disjoint
from the calibration sources, and the calibration source panels and new source
panel are exchangeable, Propositions 16.1 and 16.2 hold with
`A_u=A_u^src`. On the covered event, every frozen exchange considered for every
document in the new panel obeys its upper residual envelope.

**Proof.** The source-panel maxima are exchangeable scalar scores, so the same
rank proof applies. The nested maximum yields simultaneous document and
exchange coverage inside the panel. `square`

The panel size, document sampling rule, exchange set, features, fitting
algorithm, and handling of source failures must all be frozen before
calibration. A guarantee beyond the fixed panel would require another sampling
argument or a stronger within-source assumption.

**Proposition 16.5 (95% resolution floor).** At `alpha=0.05`, ordinary
split-conformal calibration has a finite threshold only if there are at least
19 calibration units.

**Proof.** Finiteness requires

\[
  \left\lceil0.95(n+1)\right\rceil\le n.
\]

This first holds at `n=19`. `square`

For cross-source inference those are 19 independent calibration source panels,
not 19 documents. Additional disjoint sources are needed to fit a nontrivial
`q_e`; using the calibration outcomes to tune it would invalidate the simple
rank proof.

## What conformalization does not prove

The literature distinction is decisive:

- Split conformal prediction gives finite-sample **marginal** coverage under
  exchangeability ([Lei et al., 2018](https://doi.org/10.1080/01621459.2017.1307116)).
- A learned conditional-quantile predictor can make the envelope adapt to
  heteroskedastic probes, while conformal correction preserves marginal
  coverage ([Romano, Patterson, and Candès, 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)).
- Exact distribution-free conditional coverage is generally impossible except
  under restrictions that can make the result effectively marginal or
  vacuous ([Barber et al., 2021](https://doi.org/10.1093/imaiai/iaaa017)).
- Hierarchical data require calibration that respects group exchangeability;
  treating correlated observations as independent units changes the target
  ([Dunn, Wasserman, and Ramdas, 2022](https://arxiv.org/abs/1809.07441)).

Consequently Proposition 16.2 means that the joint event “fire and harm” has
probability at most `alpha` over a new exchangeable unit. It does **not** imply

\[
  \Pr[\Delta<0\mid\text{fire}]\ge1-\alpha
\]

or 95% correctness inside every probe-defined subgroup. If the router rarely
fires, the conditional fraction among fired cases can differ greatly from the
marginal unsafe-action bound.

## Retrospective substrate diagnostic

The existing artifacts can test whether the proposed certificate is
numerically vacuous, but cannot prospectively validate it. After confirmation
had already been opened, the diagnostic fixed the observed exchange

\[
  e=(B=43,\ 2\to4),\qquad 47\to59,
\]

used `q_e=0`, treated the 64 proposal documents as calibration units, and set
`alpha=1/20`. The order-statistic rank was 62 and the resulting one-sided
interaction-residual threshold was

\[
  Q_{0.05}=2193\quad\text{Q32}.
\]

On the 64 already-open confirmation documents:

| Diagnostic | Result |
| --- | ---: |
| Residual envelope covered | 63 / 64 |
| Certificate fired | 18 / 64 |
| Favorable among fired | 18 |
| Unfavorable among fired | 0 |
| Aggregate fired contrast | -92,415 Q32 |

The 18 documents are `136, 139, 141, 144, 147, 155, 156, 159, 160, 162, 168,
173, 175, 177, 178, 186, 190, 191`. This recovers all 17 favorable documents
from the confirmed medoid route and additionally certifies document 141.

This is a post-confirmation, same-source diagnostic. The rule was constructed
after those outcomes were visible, the two blocks belong to one SimpleWiki
source cluster, and documents are not independent source panels. The result
establishes only that the conformal margin is nonvacuous on the recorded
substrate. It supplies no prospective p value, no 95% source-transfer claim,
and no authorization to deploy the router.

## Revised research program

The leading theory is now **probe-visible margin plus calibrated interaction
uncertainty**:

1. The exact Möbius identity supplies local counterfactual accounting within
   the declared action cube.
2. Singleton probes supply the exchange margin `lambda` without evaluating the
   candidate multi-atom outcome.
3. A predictor `q_e(phi)` may exploit Ramanujan phase, Walsh features, source
   metadata, or other proposal-only features to reduce residual uncertainty.
4. A simultaneous conformal correction turns that imperfect predictor into a
   marginal unsafe-action guarantee.
5. Abstention preserves the control action wherever the envelope is too wide.

The key empirical quantity is no longer pairwise spectral sparsity. It is the
calibrated upper tail of `rho-q` together with the rate and value of exchanges
that remain strictly below zero after adding that tail.

## Next experiment contract

A future prospective contract should proceed at source level:

1. Acquire mutually disjoint source clusters and freeze a source-panel sampling
   rule.
2. Split entire sources into proper training, calibration, and untouched
   confirmation sets; never split documents from one source across roles.
3. Before calibration, freeze `E`, `phi`, the residual learner, `alpha`, panel
   size, missing-source policy, simultaneous maximum score, control action, and
   strict firing rule.
4. Use at least 19 calibration source panels for a finite 95% envelope, plus
   separate fitting sources. More are needed for useful tail resolution and
   model selection.
5. Freeze a nonvacuity gate and a randomized or otherwise comparable abstaining
   control before viewing confirmation outcomes.
6. On untouched sources, report unsafe firings, firing rate, aggregate
   incremental contrast, and confidence bounds at the source-panel level.

Candidate numerical firing-rate and benefit thresholds are intentionally not
set from the already-open confirmation surface. They must be chosen from a new
proposal corpus and frozen before the next confirmation.

## Conjectures and falsifiers

**C16-A (source-panel envelope validity), open.** A frozen source-level score
achieves its preregistered marginal coverage on untouched source panels.

- Falsifier: the preregistered source-level coverage test rejects the claimed
  error rate after its multiplicity and finite-source rules are applied.

**C16-B (nonvacuous safe exchange), open.** The source-calibrated envelope
authorizes a useful number of exchanges.

- Falsifier: the frozen nonvacuity gate fails on untouched sources, even if
  coverage succeeds. A perfectly safe always-abstain rule is not an
  optimization breakthrough.

**C16-C (incremental value), open.** The certified router improves the declared
objective relative to its frozen abstaining/control policy.

- Falsifier: the preregistered source-level incremental effect is nonnegative or
  fails its decision threshold.

These are distinct gates. Passing coverage while failing nonvacuity is a valid
uncertainty estimate but a useless proposal operator; passing nonvacuity while
failing coverage is unsafe; passing both without incremental value does not
justify an optimizer change.

## Decision

The conformal conditional-exchange operator is the first current proposal rule
with an exact finite-sample safety theorem that permits adaptive selection
without assuming a stable global Hamiltonian. It is promoted as the next
mathematical research target, not as a deployed optimizer.

No new experiment, optimizer change, or paid scaling is authorized by this
entry. Documents `200--212` remain sealed because another 13 documents from the
same source cannot identify the source-transfer claim.

## Replay bindings

- Retrospective artifact SHA-256:
  `ce408c92bc4739b5d5da13e2a552aa3efb34a5bb5153359b63cca5d80811ed77`
- Retrospective analyzer SHA-256:
  `1e3c339c7a096d6bf5bab69f376dd913f581e342acbc6e29ab19bf44ffe476be`
- Theory checker SHA-256:
  `66055dbd51441fd2c3c8f798889660ee060e6795daa930df95d76803c2ee9b95`

## Open work

- Design the multi-source sampling frame and define what makes source panels
  exchangeable enough for the intended deployment population.
- Compare `q_e=0`, robust quantile regression, and probe-conditioned residual
  learners using fitting sources only.
- Determine whether a source-level maximum is needlessly conservative and, if
  so, freeze a weaker target such as a random-document or bounded-fraction
  guarantee before calibration.
- Derive an anytime or weighted variant only if the actual source stream
  violates exchangeability; such methods change assumptions and do not repair a
  one-source dataset retroactively.
