# MJ-2026-07-15-17: Prospective cross-source conditional exchange

- Date: 2026-07-15
- Status: preregistered source-panel certificate supported on the frozen
  distinct-author English Project Gutenberg frame; broader source transfer and
  optimizer deployment remain unauthorized
- Extends: MJ-2026-07-15-16
- Prospective contract:
  [`p10m-cross-source-exchange-v1-contract.json`](../../benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json)
- Result artifact:
  [`p10m-cross-source-exchange-v1-result.json`](../../benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json)
- Replay checker:
  [`check-production-cross-source-exchange-v1.mjs`](../../scripts/check-production-cross-source-exchange-v1.mjs)
- Deterministic publication artifact:
  [`p10m-cross-source-exchange-v1-publication.json`](../../benchmarks/production-model-v1/p10m-cross-source-exchange-v1-publication.json)
- Publication checker:
  [`check-production-cross-source-exchange-publication-v1.mjs`](../../scripts/check-production-cross-source-exchange-publication-v1.mjs)

## Question

MJ-16 established a finite-sample conformal safety theorem but tested
nonvacuity only retrospectively inside one SimpleWiki source cluster. Does the
same frozen conditional exchange remain certifiable in a prospective experiment
whose fitting, calibration, and evaluation units are disjoint source panels?

This entry answers that bounded question affirmatively for one frozen
distinct-author English Project Gutenberg frame. It does not identify transfer
to arbitrary web text, SimpleWiki, or a larger unsampled portion of each book.

## Frozen source population and firewall

**Definition 17.1 (source unit).** A source is one eligible English Project
Gutenberg ebook from one normalized author. Before model outcomes were
evaluated, the acquisition frame was reduced to at most one ebook per author,
71 distinct-author sources were selected by a frozen SHA-256 ordering, and a
second frozen SHA-256 ordering assigned whole sources to roles:

| Role | Independent source panels |
| --- | ---: |
| Proper fitting | 16 |
| Split-conformal calibration | 39 |
| Untouched evaluation | 16 |
| Total | 71 |

An **independent source panel** is exactly the experimental unit formed from
one such ebook: its Gutenberg ebook ID, normalized author key, raw-file
SHA-256, and sampled-panel SHA-256 must all be unique across the experiment,
and the complete ebook source can occur in only one role. No author, ebook, or
sampled panel crosses roles. “Independent” here names the source-level design
unit; it is not a claim that the observed books are stochastically independent.
The intended population is the frozen distinct-author English Project
Gutenberg acquisition frame. Exchangeability of calibration and evaluation
panels within that frame, conditional on the fitting sources, remains an
explicit assumption rather than an artifact fact.

**Definition 17.2 (panel sampling).** Each source panel contains one document:
a 16,384-byte UTF-8 passage selected from the cleaned ebook body by a frozen
SHA-256 byte-offset rule, advanced to UTF-8 and whitespace boundaries. The
production objective is the sum of the first two consecutive target losses
after 64-token contexts in that passage. Thus the source-panel guarantee is for
this fixed two-window panel, not every location in the ebook.

The source frame, source roles, panel hashes, and fitting token stream were
frozen before fitting action-cube outcomes. The fitting outcome then fixed the
predictor. Only after that predictor and the final analyzer, checker, score,
threshold rule, abstention rule, and falsifiers were hash-bound did the
calibration/evaluation cube run.

Documents `200--212` were not read. They are from the old SimpleWiki source
cluster and do not count as independent source panels.

## Frozen exchange, probes, and predictor

The exchange set is the singleton

\[
  E=\{e\},\qquad e=(B=43,\ 2\mathbin{\to}4),
\]

so the control is mask `47` and the candidate is mask `59`. Every loss and
contrast is an exact Q32 integer. For panel document `d`,

\[
\begin{aligned}
  \phi_d&=(s_d(0),\ldots,s_d(5)),\\
  \lambda_d&=s_d(4)-s_d(2),\\
  \Delta_d&=L_d(59)-L_d(47),\\
  \rho_d&=\Delta_d-\lambda_d.
\end{aligned}
\]

The six singleton effects in `phi_d` are evaluated without the candidate
multi-atom outcome. The predictor is the lower median residual of the three
nearest fitting source panels. Its distance is

\[
  D(\phi,\phi^{(r)})=
  \sum_{j=0}^{5}
  \left\lfloor
    \frac{|\phi_j-\phi_j^{(r)}|2^{20}}
         {\max(1,\operatorname{MAD}^{\rm fit}_j)}
  \right\rfloor,
\]

with ascending source ID as the distance tie break. The predictor output is
already an exact Q32 integer; no floating-point conversion occurs.

## Frozen score, conformal threshold, and router

For source panel `u`, the simultaneous score is the MJ-16 source maximum

\[
  A_u=\max_{d\in P_u}\max_{e\in E}
      \{\rho_d(e)-q_e(\phi_d)\}.
\]

There is one document and one exchange in v1, so both maxima are operationally
trivial, but their scope was frozen and checked. With `n=39` calibration source
panels and `alpha=1/20`,

\[
  k=\left\lceil 40\frac{19}{20}\right\rceil=38.
\]

The correction is the 38th smallest calibration source score, with ties on the
covered side. The router fires only when

\[
  \lambda_d+q(\phi_d)+Q_{1/20}<0.
\]

Equality and positive upper contrasts abstain and retain control mask `47`.
The exact candidate contrast is hidden from this decision and is opened only
for evaluation.

## Preregistered falsifiers

Four gates were frozen before calibration and evaluation outcomes:

1. **Source-envelope validity.** On 16 evaluation panels, three or more
   failures reject a `1/20` failure probability by a one-sided exact binomial
   test at `1/20`. At most one failure is required for support; two failures
   are explicitly inconclusive and do not promote the certificate.
2. **Unsafe action.** Any fired evaluation panel with `Delta >= 0` falsifies
   the bounded operational safety gate.
3. **Nonvacuity.** At least one evaluation source panel must fire.
4. **Incremental value.** The aggregate exact Q32 contrast over fired
   evaluation panels must be strictly negative.

These gates distinguish marginal envelope behavior, action safety,
nonvacuity, and value. They do not estimate conditional correctness among
firings.

## Prospective result

The calibration source scores ranged from `-2` to `6,272` Q32. The frozen rank
gave

\[
  Q_{1/20}=4{,}326\quad\text{Q32}.
\]

The untouched evaluation result was:

| Preregistered quantity | Result |
| --- | ---: |
| Evaluation source panels | 16 |
| Residual envelope covered | 16 / 16 |
| Source-panel coverage rate | 16 / 16 |
| Envelope failures | 0 |
| Certified firings | 5 / 16 |
| Firing rate | 5 / 16 |
| Favorable fired contrasts | 5 |
| Tied fired contrasts | 0 |
| Unfavorable / unsafe firings | 0 |
| Marginal unsafe-action rate | 0 / 16 |
| Unsafe-given-firing diagnostic | 0 / 5 |
| Aggregate signed regret relative to abstention | -40,769 Q32 |
| Mean signed regret relative to abstention | -40,769 / 16 Q32 |
| Aggregate positive regret relative to abstention | 0 Q32 |

The five fired panels were:

| Source | Exact exchange contrast Q32 |
| --- | ---: |
| Charlotte Brontë, *Jane Eyre* | -8,132 |
| Herman Melville, *Moby-Dick* | -8,024 |
| Niccolò Machiavelli, *The Prince* | -8,289 |
| Charles Dickens, *A Tale of Two Cities* | -7,939 |
| Michel de Montaigne, *Essays* | -8,385 |

All four preregistered gates pass. The independent checker reconstructs every
score identity, the rank-38 threshold, the strict firing rule, the source-role
disjointness, the exact-binomial falsifier boundary, and all evaluation
summaries.

Here regret relative to abstention is candidate loss minus control loss when
the router fires and zero when it abstains. Negative signed regret is therefore
an improvement over always retaining control mask `47`; the positive part is
harm. The unsafe-action rate is the marginal joint rate over all untouched
panels, matching the MJ-16 theorem. The `0/5` conditional diagnostic is also
reported, but is not a conditional 95% guarantee.

## Checked publication verdict

The prospective contract, analyzer, and checker remain byte-for-byte bound to
the pre-outcome hashes. A deterministic reporting layer added after evaluation
does not tune or replace any frozen rule; it only turns the checked rows into
the four required operating metrics and a closed verdict vocabulary:
`supported`, `falsified`, or `inconclusive`.

The publication verdict is **supported** on the frozen frame. A finite envelope
with all support gates passing maps to `supported`; a preregistered falsifier
maps to `falsified`; the two-failure coverage gray zone maps to `inconclusive`.
A `+infinity` conformal correction or rank beyond the calibration count is a
vacuous envelope and maps to `inconclusive`, never to support or falsification.
The publication checker replays the observed metrics and exercises all of
these decision branches, including the vacuous-envelope branch.

## Interpretation

**Artifact observation 17.1.** The singleton-visible conditional exchange is
not confined to the earlier SimpleWiki documents. A larger source-calibrated
residual correction than the retrospective `2,193` Q32 margin remains
nonvacuous on five of sixteen untouched, distinct-author panels, and every
fired exact contrast is favorable.

**Artifact observation 17.2.** The fitted predictor contributes little on the
five fired panels: its Q32 predictions are `0, 1, 0, -1, 0`. The transferable
signal in this experiment is therefore primarily a large negative singleton
margin plus a source-calibrated residual envelope, not a successful complex
residual regression law.

**Proposition 17.1 (bounded prospective certificate).** Conditional on the
source-panel exchangeability assumption in Definition 17.1, Proposition 16.4
applies to the frozen protocol and gives marginal simultaneous residual
coverage of at least `19/20` for one new panel from the same target population.
Consequently Proposition 16.2 bounds the marginal probability of firing a
non-improving exchange by `1/20`.

**Proof.** Fitting sources are disjoint, the predictor is fixed before
calibration, calibration and evaluation units are whole source panels, the
score is permutation-symmetric over calibration panels, and the rank is the
MJ-16 split-conformal rank. The nested maximum covers the fixed panel and
exchange set. The strict firing implication is Proposition 16.2. `square`

The observed `16/16` coverage and `5/5` favorable firings agree with the
theorem's operating claim but do not prove the exchangeability assumption.
Nor do they imply 95% conditional correctness among fired panels.

## Conjecture updates

- **C16-A (source-panel envelope validity): supported on the frozen
  distinct-author English Project Gutenberg frame.** Zero of sixteen untouched
  panels fail the preregistered envelope.
- **C16-B (nonvacuous safe exchange): supported on that frame.** Five of
  sixteen untouched panels fire and none is unsafe.
- **C16-C (incremental value): supported on that frame.** The aggregate fired
  contrast is `-40,769` Q32.

The scope qualifier is part of each update. These conjectures remain open for
arbitrary web text, SimpleWiki source clusters, larger within-book panels, and
different exchange sets.

## Decision

M3 is complete as a bounded prospective cross-source exchange certificate. The
conditional-exchange mechanism is promoted from retrospective same-source
nonvacuity to prospective source-panel support on the frozen 71-source frame.

No optimizer change or paid scaling is authorized. The evidence surface is
still one passage and two adjacent target windows per source, uses one frozen
exchange, and depends on an explicit source-exchangeability assumption.
Documents `200--212` remain sealed and are not needed for this conclusion.

## Replay bindings

- Source frame SHA-256:
  `3cf38860568320257a7af77fb615054925fb816ba98adc016ab60ddbeae01738`
- Fitting raw structure SHA-256:
  `f9de9d67fe73da7b9338987f97578f24e1a6094b774477209990920b75577da9`
- Fitted predictor SHA-256:
  `07712a0a5071e89dc217aa0c00de63beda27e058c9da3ce68d5a553485eddfac`
- Prospective contract SHA-256:
  `cfb7e59dafd49bb9a643210cf3a97e6dbd0ecd0b923b95e78af2c7abb67ac443`
- Calibration/evaluation raw structure SHA-256:
  `094d0641b843ffc322bf232605a038c690f0cd12cc757d712e6255cd87d1fa23`
- Result SHA-256:
  `69557a15d19df703a27858072c3444d1b8a5b6b98ccf92897e0d12e129191425`
- Analyzer SHA-256:
  `3a9e35aa35eb0d0be87722418079e4fe8e44c74bb965a7b9a44253e4166031a1`
- Checker SHA-256:
  `d450bb382caec220ceada5ce41a0ac8008cf1eb0a9fc1f3584123485c5e91899`
- Publication artifact SHA-256:
  `047c28da9d281f609bb33b8df73fbf712c59693305a78a8e0e0d03d351efaeb0`
- Publisher SHA-256:
  `eb97aeb294f9291bb24dccd4b184ba8a9706578c80c343cba0ff8b045ba5d596`
- Publication checker SHA-256:
  `768519337e745681254f614c1e2fce75b93782f4f84d893d73b1e472b907da21`

## Open work

- Repeat the experiment with multiple nonadjacent sampled documents per source
  while retaining the source-panel maximum.
- Freeze a second distinct source frame from a different acquisition process;
  the current curated Gutenberg frame does not identify arbitrary-source
  deployment.
- Test a larger exchange set with a genuinely simultaneous source score.
- Compare the frozen nearest-source predictor with `q=0` prospectively; the v1
  predictor's observed contribution is small and must not be post-hoc removed
  from this result.
