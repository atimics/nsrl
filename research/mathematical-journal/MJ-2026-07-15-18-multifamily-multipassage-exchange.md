# MJ-2026-07-15-18: Multi-family, multi-passage conditional exchange

- Date: 2026-07-15
- Status: prospective overall certificate inconclusive; Federal Register and
  RFC strata pass their frozen family-promotion gates, Gutenberg is withheld
  by coverage, and science abstains
- Extends: MJ-2026-07-15-17
- Prospective contract:
  [`p10m-multifamily-exchange-v1-contract.json`](../../benchmarks/production-model-v1/p10m-multifamily-exchange-v1-contract.json)
- Result artifact:
  [`p10m-multifamily-exchange-v1-result.json`](../../benchmarks/production-model-v1/p10m-multifamily-exchange-v1-result.json)
- Replay checker:
  [`check-production-multifamily-exchange-v1.mjs`](../../scripts/check-production-multifamily-exchange-v1.mjs)

## Question

MJ-17 prospectively supported the atom-2-to-atom-4 conditional exchange on a
distinct-author Project Gutenberg frame, but it sampled only one passage and
two adjacent targets from each source. Does the same frozen exchange remain
certifiable when each source contributes four nonadjacent passages and the
source frame spans literature, technical standards, regulatory documents, and
scientific articles?

The answer is mixed. The exchange remains nonvacuous, safe on every fired
passage, and valuable on three families, but the preregistered overall coverage
promotion gate fails. M4 is therefore inconclusive rather than a broader
certificate.

## Frozen source frame and firewall

Four public acquisition surfaces were frozen before model outcomes:

- the [Project Gutenberg machine-readable catalog](https://www.gutenberg.org/ebooks/offline_catalogs.html),
- the [RFC Editor index](https://www.rfc-editor.org/rfc-index/),
- the [FederalRegister.gov public API](https://www.federalregister.gov/developers/documentation/api/v1), and
- the [Europe PMC REST and open-access services](https://europepmc.org/developers).

The operational source unit is one complete publication. Within-family
metadata keys enforce distinct authors, first-listed RFC authors, most-specific
Federal Register agencies, or first-author/journal identities. These rules
reduce obvious duplication but do not prove stochastic independence: agencies,
genres, standards groups, and scientific institutions may still share latent
effects. One Gutenberg calibration record has the catalog creator `Various`,
which is retained as one publication-level source and is an explicit limitation
of the metadata-key interpretation.

Each family has the same whole-source role allocation:

| Role | Per family | Four families |
| --- | ---: | ---: |
| Proper fitting | 3 | 12 |
| Family-specific conformal calibration | 19 | 76 |
| Untouched evaluation | 4 | 16 |
| Total | 26 | 104 |

No publication crosses roles. The 71 Gutenberg sources used by MJ-17 are
excluded from the new Gutenberg acquisition frame. Documents `200--212` are
not sources, were not read, and remain sealed.

The source frame and all 416 passage hashes were frozen before the fitting
cube. The fitting cube then fixed the predictor. Before any calibration or
evaluation cube ran, the eight raw contracts, final analyzer, independent
checker, family-specific conformal rule, abstention rule, proposal ordering,
and falsifiers were hash-bound by the prospective contract.

## Four-passage panels

Each cleaned source body is divided into four consecutive byte quartiles. One
12,288-byte UTF-8 passage is sampled wholly inside each quartile by a frozen
SHA-256 rule. The four passages are nonoverlapping. Each passage objective is
the sum of the first two target losses after 64-token contexts.

For source panel `u`, the simultaneous score is

\[
  A_u=\max_{d\in P_u}\max_{e\in E}
      \{\rho_d(e)-q_e(\phi_d)\},
\]

where `P_u` contains all four passages. A source is covered only when all four
passages are covered. This maximum was operationally nontrivial in M4 and
prevents selecting the easiest passage after outcomes.

## Frozen exchange and predictor

The exchange remains

\[
  E=\{(B=43,\ 2\mathbin{\to}4)\},
\]

with control mask `47` and candidate mask `59`. The six Q32 singleton effects
are the only passage features. The fitting set contains 48 passage rows from 12
whole-source panels.

For a new passage, the predictor first finds the nearest of four fitting
passages inside each fitting source panel. It then takes the lower median
interaction residual from the three nearest distinct fitting sources. Thus
three passages from one source cannot masquerade as three independent
neighbors.

## Family-specific conformal rule

A pooled correction from only ten sources per family was rejected before model
outcomes: pooled scores would require an implausible cross-family
exchangeability assumption. M4 instead uses 19 calibration sources in each
family. At `alpha=1/20`, every family has the finite split-conformal rank

\[
  k=\left\lceil(19+1)\frac{19}{20}\right\rceil=19.
\]

The family correction is therefore that family's largest calibration
source-panel score. Conditional on within-family source-panel exchangeability,
MJ-16 gives marginal 95% simultaneous coverage for a new source panel from
that family. The passage router fires only when

\[
  \lambda_d+q(\phi_d)+Q_{1/20,\operatorname{family}(d)}<0.
\]

Equality and positive upper contrasts abstain. Candidate multi-atom outcomes
are not router inputs.

## Preregistered falsifiers

Support required all of the following:

1. At most one uncovered evaluation source overall and at most one in any
   family. Exactly two failures are `coverage_inconclusive_no_promotion`; three
   failures reject a 5% failure rate by the frozen one-sided exact binomial
   test at level 5%.
2. Zero fired passages with nonnegative exact candidate-minus-control
   contrast.
3. At least eight fired passages spanning at least two source families.
4. Strictly negative aggregate fired contrast overall and within every family
   containing a firing.
5. No family with an unsafe firing may be promoted.

Held-out proposals are ordered by a frozen SHA-256 key. Their outcomes cannot
affect later router decisions; the ordering exists to provide a deterministic
cumulative value ledger, not to claim a cumulatively mutated model state.

## Prospective result

The family corrections were:

| Family | Calibration panels | Rank | Q32 correction |
| --- | ---: | ---: | ---: |
| Federal Register | 19 | 19 | 2,326 |
| Gutenberg | 19 | 19 | 2,141 |
| RFC | 19 | 19 | 4,307 |
| Science | 19 | 19 | 4,272 |

The untouched result was:

| Quantity | Result |
| --- | ---: |
| Evaluation source panels | 16 |
| Four-passage panels covered | 14 / 16 |
| Envelope failures | 2 |
| Fired source panels | 7 / 16 |
| Fired passages | 12 / 64 |
| Families with firings | 3 / 4 |
| Favorable fired passages | 12 |
| Unsafe fired passages | 0 |
| Aggregate candidate-minus-control contrast | -63,541 Q32 |
| Net held-out improvement | 63,541 Q32 |

Both uncovered sources are Gutenberg evaluation panels: Xenophon's
*Anabasis* (`gutenberg-1170`) and John Ruskin's *Lectures on Art*
(`gutenberg-19164`). Exactly two failures do not reject the 5% target under the
frozen global test, but they fail both the overall support gate and the
Gutenberg family gate.

| Family | Covered panels | Fired panels | Fired passages | Unsafe | Net improvement Q32 | Frozen family status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Federal Register | 4 / 4 | 4 | 7 | 0 | 36,255 | promoted |
| Gutenberg | 2 / 4 | 2 | 3 | 0 | 14,534 | withheld by coverage |
| RFC | 4 / 4 | 1 | 2 | 0 | 12,752 | promoted |
| Science | 4 / 4 | 0 | 0 | 0 | 0 | abstained |

Every fired exact contrast is negative. The 12 deterministic actions have Q32
improvements `6,396`, `4,308`, `6,580`, `7,769`, `4,876`, `4,094`, `6,178`,
`3,956`, `185`, `4,243`, `8,600`, and `6,356`, summing to `63,541`.

## Interpretation

**Artifact observation 18.1.** The mechanism is not confined to literature.
It fires safely and with negative aggregate contrast on Federal Register and
RFC evaluation passages. Those families pass their frozen family-promotion
rules.

**Artifact observation 18.2.** The four-passage maximum exposes heterogeneity
hidden by MJ-17's single passage. Gutenberg remains nonvacuous and all three
firings are favorable, but two of four evaluation books exceed the rank-19
source envelope. Gutenberg promotion is withheld.

**Artifact observation 18.3.** Science has full observed panel coverage but no
certified firing. This is safe abstention, not evidence of useful transfer.

**Proposition 18.1 (bounded family-conditional safety).** Conditional on
exchangeability of fitting-fixed calibration and evaluation panels within a
given frozen family, the rank-19 correction gives marginal 95% simultaneous
coverage for the four-passage panel, and the strict router bounds the marginal
probability of a fired non-improving exchange by 5% for a new panel from that
family.

The observed 0/12 unsafe firings agree with this theorem but do not prove the
exchangeability assumptions or 95% correctness conditional on firing.

## Decision

M4 does **not** promote an overall multi-family certificate. Its frozen verdict
is `coverage_inconclusive_no_promotion`: two failures are below the exact-test
rejection boundary of three but above the promotion allowance of one.

Federal Register and RFC source families pass their preregistered local
promotion rules. Gutenberg is withheld by coverage, and science abstains. No
claim is made for arbitrary text, future publications, whole documents,
different exchange sets, or conditional correctness among firings.

No optimizer change or paid scaling is authorized. Documents `200--212`
remain sealed.

## Replay bindings

- Source frame SHA-256:
  `21e60c4d47fa91bd458342ce8ebe865950d475c1101f778252153a1bb5781666`
- Fitting raw structure SHA-256:
  `6f10129bcab6274212a13252008b45eedba364059aa25a69646f6cf8de407c8d`
- Fitted predictor SHA-256:
  `42744fb0a7bbd70b254d385d7717f586905af0a90ae4784b201ce2c6fbcea078`
- Prospective contract SHA-256:
  `b70d0250bf56ee7f89cb9d50389b0f409eb1cb744058d88d907fe33281ce0e16`
- Result SHA-256:
  `1963d8288fc18624687a239f8f6099576630c2d5ab1ff4543b3f422f3b7095c7`
- Analyzer SHA-256:
  `1bc22b7c0b47965c03851818de93f9b9e47ef10d2eaa7eea373157691c0758f1`
- Checker SHA-256:
  `a149f869579bcb52eb7d9b9290b9fa91f3026dc129db414c0cd6a1f12c576bcf`

The eight passage/shard raw result hashes, in contract order, are:

```text
38084d1017deab894216ae84cf8b34e69af253e6e8cbefd6bf28f628ea5fd73c
0f6a44bf89cfa7e4ed3a61897fa7d210e01e89ed5d50edaeafd39dea4c71c200
9ef05abeb7700d4be58b10e4002c5dd84f1592f6f45d3190932abd5e86e88fbd
fe6fa5a25241a4aeadf7d9e893fb8c87778bd1ff1a0a452a1319fa8c5c447e21
e05b6f83ddb09f767929eff1e8dd8c4e413624f99e320aca45c2faf81f02dee6
3024cc2cc5c4f25765fd18af1f499e3327405bbad7f9d91c8731af244a4a0b00
789861486b3d295a6aa2c683321aee7736ab2bcae974b5b81c963e3298c40345
1abf854364a1b8980d3c93acd8e5e99c9a040146cd7f230179eec4df0e8e5a4f
```

## Open work

- Diagnose the two Gutenberg source-envelope failures on a new prospective
  frame; do not tune M4's threshold post hoc.
- Replicate Federal Register and RFC promotion on new source panels.
- Determine whether science needs a different frozen exchange set or should
  remain an abstention-only family.
- Increase within-family evaluation counts before making family-level rate
  claims; four panels per family are deliberately modest.
