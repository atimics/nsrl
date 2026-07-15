# MJ-2026-07-15-08: Hierarchical distributional Boolean jets and certifiability

- Date: 2026-07-15
- Status: algebraic and finite-sample core established; proposal operator open
- Extends: MJ-2026-07-15-06 and MJ-2026-07-15-07
- Code binding:
  [`check-boolean-jet-stability-theory-v1.mjs`](../../scripts/check-boolean-jet-stability-theory-v1.mjs)
- Artifact binding:
  [`p10m-boolean-jet-confirmation-v1.json`](../../benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1.json)

## Question

What mathematical object should replace a single aggregate Boolean coefficient
as the target of a discrete optimizer, and when is that object statistically
and numerically identifiable?

## 1. Environment-indexed Boolean jets

Let `d` denote an independently sampled evaluation unit, such as a document,
and let `q` denote a fully specified observation objective and numeric
representation. For a finite atomic move set `G`, define

```text
ell_d^q(S) = loss on d after applying atomic subset S subseteq G,

mu_d^q(S) = sum_(T subseteq S) (-1)^(|S|-|T|) ell_d^q(T).
```

The deterministic Boolean jet of MJ-06 is recovered by holding `(d,q)` fixed.
The optimizer-relevant object is the random field

```text
(d,q,S) -> mu_d^q(S),
```

not one coefficient summed over a small evaluation surface.

### Proposition 1.1 (conditional Möbius identity)

For disjoint atomic action sets `A` and `B`,

```text
C_d^q(A | B)
  = ell_d^q(A union B) - ell_d^q(B)
  = sum mu_d^q(U),
```

where the sum ranges over

```text
U subseteq A union B  such that  U intersect A is nonempty.
```

**Proof.** Boolean inversion gives
`ell(A union B)=sum_(U subseteq A union B) mu(U)`. Subtracting
`ell(B)=sum_(U subseteq B) mu(U)` removes exactly the subsets that do not touch
`A`. The executable checker verifies the identity in 700 random cases through
rank eight.

For a singleton `A={i}` this reduces to

```text
C_d^q(i | B) = sum_(T subseteq B) mu_d^q(T union {i}).
```

For the coarse two-generator trunk/head cube,

```text
C_d^q(T | H) = bar_mu_d^q(T) + bar_mu_d^q(TH).
```

The two-vertex conditional contrast is therefore the exact operational answer
to “does trunk help after head?” It is preferable to interpreting the mixed
coefficient alone.

## 2. Block coarse-graining is a Möbius pushforward

The p10m trunk and head actions are themselves blocks of atomic parameter
moves. Let the atomic ground set be partitioned into disjoint blocks
`B_1,...,B_b`. Define the coarse loss

```text
bar_ell(J) = ell(union_(j in J) B_j),   J subseteq {1,...,b},
```

and define the block support of an atomic subset `U` by

```text
supp_B(U) = {j : U intersect B_j is nonempty}.
```

### Proposition 2.1 (block-support pushforward)

The coarse Möbius coefficient is

```text
bar_mu(J) = sum_(U : supp_B(U)=J) mu(U).
```

**Proof.** Regroup the atomic zeta expansion by block support:

```text
bar_ell(J)
  = sum_(U subseteq union_(j in J) B_j) mu(U)
  = sum_(K subseteq J) sum_(supp_B(U)=K) mu(U).
```

Uniqueness of Boolean Möbius inversion identifies the inner sum with
`bar_mu(K)`. The checker verifies this pushforward in 240 random block cubes.

### Consequence for the p10m experiment

The trunk block contains four atoms and the head block two. Consequently,

```text
bar_mu(T)  aggregates 2^4-1 = 15 atomic coefficients,
bar_mu(H)  aggregates 2^2-1 = 3 atomic coefficients,
bar_mu(TH) aggregates (2^4-1)(2^2-1) = 45 cross-support coefficients.
```

The observed `bar_mu(TH)` is not a single pair interaction. It is a signed sum
of 45 atomic-support terms of orders two through six.

### Corollary 2.2 (localization and cancellation)

- If `bar_mu(J)<0`, at least one atomic coefficient with block support `J` is
  negative.
- If `bar_mu(J)>=0`, negative atomic coefficients may still exist and cancel
  against positive ones.
- More generally,

  ```text
  |bar_mu(J)| <= sum_(supp_B(U)=J) |mu(U)|.
  ```

The checker contains an explicit zero-coarse-interaction example formed from
atomic cross coefficients `-5` and `+5`. A hierarchical search may refine a
negative coarse block, but it cannot safely prune a nonnegative block without
additional absolute-variation bounds.

## 3. Risk, direction, and visibility are different estimands

For a declared conditional action, write `C_d=C_d^q(A|B)`, with negative values
favorable. Define

```text
a = -E[C_d]                                      magnitude advantage,
v = Pr(C_d != 0)                                 objective visibility,
p = Pr(C_d < 0 | C_d != 0)                       conditional sign advantage,
m = Pr(C_d < 0) - Pr(C_d > 0) = v(2p-1)          signed visibility margin.
```

The empirical signed margin is

```text
hat_m = (number favorable - number unfavorable) / N.
```

It treats an objective tie as zero information rather than silently deleting
the document. The sign test estimates `p`; the aggregate loss estimates `a`;
`m` combines direction and information rate. These quantities are not
interchangeable.

### Proposition 3.1 (risk and directional primality are incomparable)

Define a risk-local minimum by `E[Delta_d(S)]>=0` for every admissible `S`, and
an `eta`-directional local minimum by
`Pr(Delta_d(S)<0)<=eta` for every `S`. Neither property implies the other
without a magnitude bound.

Two exact counterexamples are

```text
[-1,-1,-1,-1,-1,-1,-1,-1,-1,+20]
```

where 90% of documents improve but the mean change is harmful, and

```text
[-20,+1,+1,+1,+1,+1,+1,+1,+1,+1]
```

where 90% are harmful but the mean change improves. The checker preserves both
counterexamples.

Therefore “primes as local minima” needs an estimand qualifier:

- **risk prime:** no move improves population mean loss;
- **directional prime:** no move improves more than a declared document
  fraction;
- **representation-robust prime:** the property holds across a declared family
  of compatible observation objectives.

## 4. Finite-sample certification under ties

For a move fixed before the transfer documents are observed, condition on the
`n` non-tied documents. Under the null `p=1/2`, the favorable count is binomial,
so the exact paired sign test remains valid after discarding ties. Its effective
sample size is `n`, not the total document count `N`.

For planning, suppose `p >= 1/2+gamma`. Hoeffding's inequality gives

```text
Pr(hat_p <= 1/2) <= exp(-2 n gamma^2).
```

Thus the conservative non-tie planning count is

```text
n >= log(1/alpha) / (2 gamma^2).
```

If the visibility rate is `v`, obtaining `n` non-ties requires approximately
`N=n/v` total documents in expectation. A separate binomial tail calculation
is required for a high-probability total-document guarantee.

The failed transfer confirmation observed `v=18/64=0.28125`. At this diagnostic
visibility and `alpha=0.05`, the planning counts are

| `gamma` | required non-ties | expected documents |
| ---: | ---: | ---: |
| 0.10 | 150 | 534 |
| 0.20 | 38 | 136 |
| 0.30 | 17 | 61 |

Only 77 documents remain reserved. At the observed visibility, that budget can
support only a very large conditional sign advantage, roughly `gamma>=0.263`
in expectation. This is not a power claim for a future move—its visibility is
unknown—but it proves that another low-visibility one-LSB candidate is not a
credible use of the reserved block.

### Proposition 4.1 (all-document signed-margin bound)

Let

```text
Z_d = +1 if C_d<0,  0 if C_d=0,  -1 if C_d>0.
```

Then `E[Z_d]=m`. Since `Z_d in [-1,1]`, with probability at least `1-delta`,

```text
m >= hat_m - sqrt(2 log(1/delta)/N).
```

This is conservative but makes the cost of ties explicit. In the artifact,

```text
proposal: hat_m = (10-2)/64 = 1/8,
transfer: hat_m = (7-11)/64 = -1/16.
```

The transfer margin points in the wrong direction before any confidence bound
is applied.

## 5. Resolution-stability bounds

Suppose an observed objective is a bounded perturbation of a latent or refined
objective:

```text
ell(S) = ell_star(S) + e(S),    |e(S)| <= epsilon.
```

### Proposition 5.1 (Möbius error amplification)

For a coefficient of atomic order `k`,

```text
|mu(S)-mu_star(S)| <= 2^k epsilon.
```

**Proof.** The coefficient is an alternating sum of `2^k` vertex losses; apply
the triangle inequality. The checker verifies 20,400 coefficient bounds.

### Proposition 5.2 (conditional contrasts do not amplify with rank)

For any disjoint block actions `A,B`,

```text
|C(A|B)-C_star(A|B)| <= 2 epsilon.
```

Only `ell(A union B)` and `ell(B)` appear. The checker verifies 320 cases.
This is another reason to certify the conditional effect directly rather than
thresholding a collection of high-order coefficients.

If the observed loss is nearest-grid rounding with step `delta`, then
`epsilon=delta/2`. A conditional sign is guaranteed stable when its latent or
observed magnitude exceeds `delta`; an order-`k` coefficient requires margin
greater than `2^(k-1) delta`.

These are cross-representation robustness bounds, not objections to exact
integer evaluation. A Q20 objective is exact as a declared lattice objective.
The bounds become relevant only when that objective is interpreted as a proxy
for a refined objective. The Q47 logit-anchored confirmation still reports Q20
loss, so its output grid remains one Q20 unit. Most observed document-level
conditional effects lay between `-2` and `+3`, directly against this boundary.

If two objectives use different exponent approximations rather than one being
a bounded rounding refinement of the other, Proposition 5.1 does not apply
without an independently proved uniform error bound. Such objectives must be
treated as separate environments `q`, with a predeclared concordance rule.

## 6. Selection, controls, and valid confirmation

If a proposal operator observes only a proposal partition and freezes one move
family before transfer evaluation, the transfer sign test remains valid under
the declared independent-document model. If `M` families are inspected on the
same transfer partition and the best is reported, the nominal level is no
longer valid; a simple familywise remedy is to test each at `alpha/M`.

For a structured conditional effect `C_d` and a matched-control conditional
effect `R_d`, define

```text
D_d = C_d - R_d.
```

A paired sign test on `D_d` asks whether the structured move beats its frozen
control more often than the reverse. But seeded group/cardinality matching does
not make the structured and random moves exchangeable: the structured move was
chosen by a gradient-derived proposal. The comparison is a prospective
benchmark, not a randomization-based causal test, unless an actual exchangeable
assignment mechanism is constructed.

Controls should at minimum bind group, cardinality, stored width, boundary
margin, and pre-outcome function visibility. A control with systematically
lower visibility can make the structured proposal look informative without
showing better optimization.

## 7. The stability-aware compressed proposal operator

Let `b(X)` be a compact sketch of rounding boundaries, activation paths,
residual state, and parameter groups at state `X`. The missing operator is

```text
Q_phi : b(X) -> {at most M canonically ordered move families}.
```

It should be trained on past checkpoints and proposal partitions to predict a
certificate-relevant tuple, not merely an aggregate gradient:

```text
(expected magnitude advantage a,
 objective visibility v,
 signed visibility margin m,
 representation concordance,
 evaluation cost).
```

One prospective eligibility rule is the vector gate

```text
non_ties >= n_min,
exact sign p <= alpha/M,
signed-margin lower bound > 0,
mean conditional effect <= -tau,
structured-minus-control direction favorable,
representation rule passed,
saturation count = 0.
```

No one scalar should silently substitute for another component. In particular,
a large conditional win rate with tiny visibility is not the same result as a
material mean improvement.

The block-support pushforward suggests a tractable hierarchy:

1. propose a few semantic blocks rather than millions of coordinates;
2. evaluate their exact coarse Boolean cube;
3. retain only blocks with prospectively stable conditional effects;
4. refine retained blocks into subblocks or atoms;
5. repeat with a new proposal partition before touching transfer data.

This reduces branch cost from atomic rank to block rank, but Corollary 2.2
prevents unsafe pruning: nonnegative coarse coefficients can hide cancellation.
A complete hierarchical optimizer still needs computable absolute-variation
or boundary-support bounds for discarded blocks.

## Decision

Adopt the environment-indexed conditional effect `C_d^q(A|B)`, visibility `v`,
signed margin `m`, and block-support pushforward as the next mathematical
foundation. Stop treating a coarse aggregate `mu(TH)` as one transferable pair
interaction.

Do not spend the 77 reserved documents on another candidate unless proposal
data predicts materially higher visibility or a very large directional margin.
The next theoretical search should focus on a hierarchical boundary sketch
that can bound hidden cancellation and predict objective visibility before
transfer evaluation.

## Conjectures and falsifiers

### C08-A: hierarchical boundary sketches predict visibility

- State: `open`
- Claim: pre-outcome boundary features rank block moves by document-level
  objective visibility better than group/cardinality-matched controls.
- Falsifier: visibility ranking is no better than the frozen control on unused
  proposal partitions.

### C08-B: stable coarse negativity localizes stable atomic negativity

- State: `open`
- Claim: when a coarse block coefficient is negative with a declared stability
  margin, recursive refinement finds an atomic or smaller-block conditional
  effect with the same transfer direction at useful frequency.
- Falsifier: coarse negativity repeatedly decomposes into unstable cancellation.

### C08-C: representation refinement reduces ties without sign disorder

- State: `open`
- Claim: a predeclared finer loss grid increases `v` while preserving the signs
  of contrasts whose coarse-grid magnitude exceeds the proved error bound.
- Falsifier: added visibility is dominated by representation-dependent sign
  reversals.

## Open work

- Derive computable absolute-variation bounds for block-support coefficient
  sums from branch-local boundary counts.
- Add a higher-than-Q20 observation loss without changing deployed inference.
- Specify cluster-robust evaluation units when documents are not independent.
- Define cross-checkpoint training and cross-fitting for `Q_phi` without
  leaking reserved transfer documents.
- Preserve immutable protocol versions before evaluating matched controls or a
  new hierarchical proposal.
