# MJ-2026-07-15-19: Solomonic judgment calculus

- Date: 2026-07-15
- Status: supported on the frozen six-source, three-family prospective frame;
  symbolic hash correspondence falsified; optimizer promotion and paid scaling
  remain unauthorized
- Extends: MJ-2026-07-15-16 through MJ-2026-07-15-18
- Formal record schema:
  [`judgment-record-v1.schema.json`](../../protocol/judgment-record-v1.schema.json)
- Prospective contract:
  [`p10m-solomonic-judgment-v1-contract.json`](../../benchmarks/production-model-v1/p10m-solomonic-judgment-v1-contract.json)
- Untouched result:
  [`p10m-solomonic-judgment-v1-result.json`](../../benchmarks/production-model-v1/p10m-solomonic-judgment-v1-result.json)
- Trivalent publication:
  [`p10m-solomonic-judgment-v1-publication.json`](../../benchmarks/production-model-v1/p10m-solomonic-judgment-v1-publication.json)
- Replay checker:
  [`check-solomonic-judgment-v1.mjs`](../../scripts/check-solomonic-judgment-v1.mjs)
- Publication checker:
  [`check-solomonic-judgment-publication-v1.mjs`](../../scripts/check-solomonic-judgment-publication-v1.mjs)

## Question

MJ-16 supplied a marginal conformal safety theorem for one conditional
exchange. MJ-17 and MJ-18 tested that exchange on fixed, outcome-independent
routers. How can the same evidence be placed inside a judgment calculus with
several expert faculties, explicit abstention, costs, reversibility,
provenance, falsifiers, and later invocations that may depend on earlier
outcomes without treating adaptivity as free?

The answer has two layers that must not be conflated:

1. a simultaneous family-specific conformal envelope certifies the candidate
   move on one new four-passage source panel under within-family
   exchangeability; and
2. an e-process controls repeated, adaptively gated invocations under an
   explicit conditional unsafe-intensity null.

Marginal conformal coverage alone does not imply the conditional null needed
by the e-process. M4 records both assumptions and gives neither an exemption.

## Formal judgment record

**Definition 19.1 (judgment record).** At passage invocation `t`, a judgment is

\[
  J_t=(C_t,\mathcal A_t,\mathcal E_t,\widehat B_t,H_t,U_t,
       K_t,R_t,a_t,F_t,P_t).
\]

The fields are:

| Symbol | Required record field | Meaning |
| --- | --- | --- |
| `C_t` | `context` | source, family, passage, and history available before action |
| `mathcal A_t` | `candidate_actions` | every expert faculty plus abstention |
| `mathcal E_t(a)` | `evidence` | strength, assumptions, and hash-bound source |
| `widehat B_t(a)` | `predicted_benefit_q32` | negative of the certified upper contrast |
| `H_t(a)` | `possible_harm_q32` | nonnegative part of that upper contrast |
| `U_t(a)` | `uncertainty_envelope_q32` | one-sided conformal interval and its population unit |
| `K_t(a)` | `intervention_cost` | coordinate writes and objective penalty |
| `R_t(a)` | `reversibility` | inverse cost and the condition under which inversion is exact |
| `a_t` | `selected` | one eligible faculty or explicit abstention |
| `F_t` | `falsifier` | observable condition changing the claim state |
| `P_t` | `provenance` | paths, SHA-256 bindings, and evidentiary roles |

The JSON schema requires these fields and rejects undeclared top-level fields.
The result contains 24 concrete records, one for each untouched passage. Each
record lists five candidates:

1. a Federal Register exchange faculty;
2. an RFC exchange faculty;
3. a scientific-article exchange faculty;
4. a symbolic hash-parity hypothesis; and
5. abstention.

The three domain faculties are distinct experts with disjoint declared
contexts. They currently recommend the same physical move, `47 -> 59`; M4
does not pretend that three labels constitute three different parameter
interventions. The calculus permits distinct interventions, but this bounded
experiment tests faculty selection.

For the physical move, intervention cost is two integer-coordinate writes and
the exact inverse also costs two writes, provided no parameter update occurs
between action and rollback. The experiment is counterfactual: it evaluates
the two states without mutating the retained model.

## Evidence-bound selection

For passage `d` and exchange `e=(B=43,2->4)`, retain the MJ-16 decomposition

\[
  \Delta_d(e)=\lambda_d(e)+\rho_d(e),
\]

where negative `Delta` improves over abstention at control mask `47`. The
proper-fitting predictor `q(phi_d)` and family correction `Q_f` were fixed by
M18. Define the certified upper contrast

\[
  U_d=\lambda_d+q(\phi_d)+Q_f.
\]

The matching domain faculty is eligible exactly when the history guard is open
and `U_d<0`. Equality abstains. If several physical actions are later admitted,
the record contract selects the minimum cost-adjusted certified upper contrast
with a frozen action-id tie break; v1 has one physical move.

The inherited corrections are:

| Family | Calibration source panels | Q32 correction |
| --- | ---: | ---: |
| Federal Register | 19 | 2,326 |
| RFC | 19 | 4,307 |
| Science | 19 | 4,272 |

The simultaneous calibration score is the maximum residual error across all
four passages of a source panel. Thus candidate selection within a panel does
not multiply the marginal source-panel error.

## Sequential controller

Let `mathcal F_(t-1)` contain all prior judgment records and controller state,
plus the current source metadata and probes available before the current
candidate outcome. Let

\[
  Y_t=\mathbf 1\{\text{source panel }t\text{ contains a fired action with }
                  \Delta\ge0\}.
\]

M4 declares the sequential safety null

\[
  H_0:\quad
  \mathbb E[Y_t\mid\mathcal F_{t-1}]\le \alpha=\frac1{20}
  \quad\text{for every predictably invoked source panel }t.
\]

Choose the fixed alternative `q=1/4` and define

\[
  E_0=1,\qquad
  E_t=E_{t-1}
  \left(\frac q\alpha\right)^{Y_t}
  \left(\frac{1-q}{1-\alpha}\right)^{1-Y_t}
  =E_{t-1}
  \begin{cases}
    5,&Y_t=1,\\
    15/19,&Y_t=0.
  \end{cases}
\]

**Proposition 19.1 (adaptive e-validity).** Under `H_0`, `(E_t)` is a
nonnegative supermartingale for every predictable choice of faculty,
invocation, and stopping rule.

**Proof.** If
`p_t=E[Y_t | mathcal F_(t-1)] <= alpha`, then

\[
\begin{aligned}
  \mathbb E[E_t/E_{t-1}\mid\mathcal F_{t-1}]
  &=p_t\frac q\alpha+(1-p_t)\frac{1-q}{1-\alpha}\\
  &\le \alpha\frac q\alpha+(1-\alpha)
         \frac{1-q}{1-\alpha}=1,
\end{aligned}
\]

because `q>alpha` makes the expression increasing in `p_t`. Predictability
allows earlier records to change later faculties or force abstention without
altering the conditional calculation. `square`

By Ville's inequality,

\[
  \Pr_{H_0}\left(\sup_t E_t\ge20\right)\le\frac1{20}.
\]

The controller therefore closes its history guard and forces all later
invocations to abstain as soon as `E_t>=20`. This is an anytime-valid alarm for
the declared conditional null. It is not a proof that the null follows from
split conformal exchangeability.

## Positive-regret boundary

The Q47 observation objective supplies a deterministic magnitude bound. Each
unmasked target weight is at least `1`; with vocabulary `8,192`, the sum of
Q47 weights is at most `2^60`. A window loss is therefore in `[0,60]` bits.
Each passage has two windows, so every candidate-control contrast obeys

\[
  |\Delta_d|\le R=120\cdot2^{32}=515{,}396{,}075{,}520
  \quad\text{Q32}.
\]

At source round `t`, let `b_t` be the smallest unsafe count `s` for which

\[
  5^s(15/19)^{t-s}\ge20.
\]

If each source has at most `m=4` fired passages, then outside the Ville event,
simultaneously for every `t`,

\[
  G_t^+=\sum_{j\le t}\sum_{d\in P_j}
         \mathbf1\{\text{fire}\}[\Delta_d]_+
  \le (b_t-1)mR.
\]

For six source rounds, `b_6=3`, so the preregistered 95% boundary is

\[
  2\cdot4\cdot R=4{,}123{,}168{,}604{,}160\quad\text{Q32}.
\]

This magnitude boundary is deliberately conservative. The unsafe-count part
is informative; the conversion to Q32 harm uses a worst-case numeric bound and
is not a tight performance interval.

## Symbolic and occult hypotheses

The symbolic candidate was a frozen SHA-256 parity of source family, source
ID, and passage ordinal. Its only admissible route into the controller was
compression of already available calibration contrast signs.

For `N=304` calibration passages, let `L_0=N h(k/N)` be the empirical binary
codelength without parity and let `L_1` be the sum of empirical codelengths in
the two parity groups. M4 charged the extra binary parameter the BIC penalty
`(1/2)log_2 N` and required

\[
  L_0-L_1-\tfrac12\log_2N\ge\log_2 20
\]

before the feature could act. The result was `-4.0983` net bits versus
`+4.3219` required. The correspondence was therefore falsified before the new
outcomes opened and never became eligible. Had it passed, the same conformal
envelope, e-process, cost, regret ledger, and exact-contrast falsifier would
still have applied.

## Prospective experiment

The source frame selected the remaining eligible publications in the cached
M18 acquisition manifest after excluding every M18 source ID and independence
key. It contains no Gutenberg source:

| Family | Untouched source panels | Passage judgments |
| --- | ---: | ---: |
| Federal Register | 2 | 8 |
| RFC | 2 | 8 |
| Science | 2 | 8 |
| Total | 6 | 24 |

The source frame, passages, raw cube contract, predictor, conformal
corrections, faculties, controller, e-process, numeric bound, occult gate,
falsifiers, analyzer, checker, publisher, and publication checker were bound
before the new cube ran. The prospective contract SHA-256 is
`47c97f43a48f87cdefd2cf68f5cd4e92a5172ac6187f93487f0b7e427095c9b1`.

## Untouched result

All six four-passage source panels were inside their inherited simultaneous
envelopes. The controller fired eight times:

| Family | Fired passages | Unsafe | Signed regret Q32 | Positive regret Q32 |
| --- | ---: | ---: | ---: | ---: |
| Federal Register | 3 | 0 | -15,287 | 0 |
| RFC | 3 | 0 | -25,016 | 0 |
| Science | 2 | 0 | -12,078 | 0 |
| Total | 8 | 0 | -52,381 | 0 |

Thus the held-out policy beats always-abstain by `52,381 Q32`. Every firing is
favorable. Transfer is observed in all three source families. With six safe
source rounds,

\[
  E_6=(15/19)^6=11{,}390{,}625/47{,}045{,}881<1,
\]

and the maximum e-value is `1`; the history guard never closes. Observed
positive regret is `0`, inside the frozen 95% boundary.

The sample is intentionally small: two new sources per family establish the
frozen pass conditions, not stable family-level rate estimates or universal
transfer.

## Publication contract and decision

The publisher can emit only:

- `supported` when every pass condition holds and no hard falsifier fires;
- `falsified` when an e-process, positive-regret, coverage-rejection,
  nonnegative-regret, or feature-specific falsifier fires; or
- `inconclusive` when neither rule resolves the claim.

Unknown states fail closed. The byte-replayed publication is:

| Claim | Status |
| --- | --- |
| Sequential evidence-bound judgment on the frozen frame | `supported` |
| SHA-256 parity correspondence predicts useful exchange | `falsified` |

This supports a bounded controller, not universal wisdom. It does not
authorize optimizer promotion, retained-model mutation, paid scaling, or a
claim that one successful exchange is a general optimizer.

## Replay bindings

- Source frame SHA-256:
  `1b12ab86fffa4457680f75f623f3a13bc7202fd6512d99b0577df62aaeb45c56`
- Raw structure contract SHA-256:
  `2dfd600d519a9c9bd27edd9506b9094a00157dab9c92869f9b7fe4b1ded3fa48`
- Untouched structure result SHA-256:
  `a10c768a1197aaf637ec610a9e61d5ebf9f997b21efb176ee933583a87bbe4d5`
- Judgment result SHA-256:
  `d88e07cd7919c001a551436a95fcf24b70f01ec64746a0338037763b70f49aa0`
- Publication SHA-256:
  `501c00402c87d3e4895643a2a20f09e331e8a0d6fc2916946dc7b0b27d3f0434`
- Judgment schema SHA-256:
  `2157ff8654af04ab456b81912f9b9d98533919e79eb09a8c13f0e081441e8ea6`

## Open work

- Replicate with at least 19 new evaluation sources per family before making
  family-level rate claims.
- Replace the worst-case Q32 positive-regret conversion with a prospectively
  calibrated bounded-loss confidence sequence.
- Test genuinely distinct physical action families under one simultaneous
  source-panel score; v1 tests distinct expert faculties around one move.
- Audit the conditional unsafe-intensity null under longer adaptive sequences;
  it is an explicit sequential assumption, not a consequence of marginal
  conformal coverage.
