# MJ-2026-07-15-11: Robust surrogate and finite-sample certificates

- Date: 2026-07-15
- Status: sharp surrogate-regret theorem established; worst-case six-atom
  population certificate is sample-limited
- Extends: MJ-2026-07-15-10
- Sharpens: MJ-2026-07-15-10 Corollary 3.2 from `2 tau_r` to `tau_r`
- Code binding:
  [`check-discrete-structure-theory-v1.mjs`](../../scripts/check-discrete-structure-theory-v1.mjs)

## Question

How can an empirical sparse or low-width Boolean surrogate authorize a move on
the population objective, and what can 64 proposal documents actually certify?

## Result

The correct comparison quantity is not the absolute error of two objectives.
It is the oscillation of their difference. Constants and shared offsets cannot
change an optimizer and must not consume certificate budget.

This yields four refinements:

1. the global regret of exact order-`r` truncation is at most `tau_r`, not
   `2 tau_r`;
2. a retained sparse support has a single additive certificate combining
   coefficient estimation error and omitted interaction mass;
3. the support may be selected from the same data when the coefficient
   intervals are simultaneous over every selectable term;
4. under only a bounded-document assumption, simultaneous inference over all
   63 coefficients of the six-atom cube is too loose at `n=64` to support a
   population optimization claim.

The exact six-atom proposal cube remains valuable for structure discovery. It
must be followed by a frozen, lower-dimensional candidate comparison on new
source-clustered units.

## 1. Objective discrepancy modulo constants

For a finite domain `D`, define the oscillation of `h` by

```text
osc_D(h) = max_(x in D) h(x) - min_(x in D) h(x).
```

### Theorem 1.1 (robust surrogate minimization)

Let `F` be the target objective, `G` a surrogate, and `x_hat` satisfy

```text
G(x_hat) <= min_(x in D) G(x) + eta.
```

Then

```text
F(x_hat) - min_(x in D) F(x)
  <= eta + osc_D(F-G).
```

**Proof.** Let `x_star` minimize `F` and write `h=F-G`. Then

```text
F(x_hat)-F(x_star)
  = G(x_hat)-G(x_star) + h(x_hat)-h(x_star)
  <= eta + max_D h-min_D h.
```

This bound is invariant under adding any constant to `F` or `G`. The familiar
uniform bound `|F-G|<=delta` gives `osc(F-G)<=2 delta`, but it can be twice as
loose as the direct oscillation.

### Corollary 1.2 (representation transfer)

After placing a coarse and fine objective on the same integer scale, a coarse
minimizer has fine-objective regret at most

```text
osc_D(F_fine-F_coarse).
```

Thus Q20/Q32 agreement should be reported as an objective-discrepancy
oscillation, not only as coefficient sign concordance.

## 2. Boolean coefficient certificate

Write target and surrogate fields on a Boolean domain as

```text
F(S) = sum_(U subseteq S) a(U),
G(S) = sum_(U subseteq S) b(U).
```

The constant coefficient cancels from every objective comparison.

### Lemma 2.1 (nonconstant coefficient envelope)

```text
osc(F-G) <= sum_(U != empty) |a(U)-b(U)|.
```

**Proof.** For any `S,T`, each nonconstant AND-basis indicator
`1[U subseteq S]` differs from `1[U subseteq T]` by at most one in absolute
value. Apply the triangle inequality to `(F-G)(S)-(F-G)(T)` and maximize.

The checker verifies the resulting regret bound on 400 random retained
surrogates. It deliberately gives the surrogate an unrelated constant term to
verify that baseline error does not enter the certificate.

### Corollary 2.2 (sharp interaction-tail certificate)

Let `F_r` retain the exact population coefficients of order at most `r`. If
`S_hat` minimizes `F_r`, then

```text
F(S_hat)-min_S F(S) <= tau_r.
```

MJ-10's `2 tau_r` statement remains valid but is not sharp. The factor of two
came from separately bounding the surrogate error at two vertices. Comparing
their shared Boolean expansion removes that duplication. The checker verifies
the sharp bound on 50,800 vertices from 700 random cubes.

## 3. Selected sparse support

Let `R` be a retained set of nonempty supports and let

```text
G_R(S) = b(empty) + sum_(U in R, U subseteq S) hat_nu(U).
```

### Theorem 3.1 (retained-support regret)

If `S_R` minimizes `G_R`, then

```text
F(S_R)-min_S F(S)
 <= sum_(U in R) |nu(U)-hat_nu(U)|
    + sum_(U notin R, U != empty) |nu(U)|.
```

Suppose simultaneous intervals

```text
|nu(U)-hat_nu(U)| <= epsilon_U
```

hold for every nonempty selectable `U`. Then, simultaneously for every support
`R`, including a data-dependent one,

```text
C(R)
 = sum_(U != empty) epsilon_U
   + sum_(U notin R, U != empty) |hat_nu(U)|
```

is a valid population-regret upper bound for the exact minimizer of `G_R`.

**Reason.** Retained terms pay `epsilon_U`. For an omitted term,
`|nu(U)|<=|hat_nu(U)|+epsilon_U`. The simultaneous event is fixed before `R`
is chosen, so selecting `R` does not invalidate it.

This creates a certificate-driven structural objective:

```text
choose R to minimize C(R) subject to induced_width(R) <= w_max.
```

The uncertainty sum is common to every `R`; under this conservative model, the
combinatorial problem is to retain as much observed absolute coefficient mass
as possible without exceeding the width budget.

### Corollary 3.2 (certified low-width optimizer)

If the primal graph of `R` has an elimination order of induced width `w`, exact
variable elimination minimizes `G_R` in

```text
O((k+|R|) 2^(w+1))
```

table operations, and the returned move has population regret at most `C(R)`
on the simultaneous confidence event.

The checker verifies 200 width-two retained surrogates against brute force and
their population certificates. Sparsity, tractability, and approximation error
are therefore one joint tradeoff rather than three separate claims.

## 4. Exchange primes under objective error

Let `D_m={S:|S|=m}`. Let `epsilon_ex(G)` be the uniform exchange defect of a
surrogate `G` on `D_m`.

### Theorem 4.1 (surrogate exchange-prime certificate)

If `X` is exchange-local for `G`, then

```text
F(X)-min_(Y in D_m) F(Y)
 <= m epsilon_ex(G) + osc_(D_m)(F-G).
```

**Proof.** MJ-10 gives
`G(X)-min_(D_m)G <= m epsilon_ex(G)`. Apply Theorem 1.1 with that quantity as
the surrogate optimization tolerance.

If `|F-G|<=delta` on the slice, the final term is at most `2 delta`. This is
tighter than first transferring the exchange axiom to `F` and then applying a
population local-to-global theorem.

### Proposition 4.2 (exchange-defect stability)

```text
|epsilon_ex(F)-epsilon_ex(G)|
 <= 2 osc_(D_m)(F-G).
```

Each four-vertex exchange contrast changes by at most twice the discrepancy
oscillation. Taking the partner minimum, outer maxima, and positive part does
not increase that uniform perturbation. The checker verifies both results on
300 surrogate/population pairs and 452 surrogate exchange-local minima.

## 5. Finite-sample coefficient envelopes

Let the sampling units be `n` independent source clusters. If documents from
one source are dependent, the source aggregate—not each document—is one unit.
Assume the complete within-unit cube obeys the predeclared bound

```text
osc_S ell_d(S) <= B
```

almost surely. An observed maximum on the current sample is not by itself a
population bound.

For a nonempty support `U` of order `u`, the alternating sum has equally many
positive and negative terms. Translation invariance therefore gives

```text
|mu_d(U)| <= 2^(u-1) B.
```

Across sampling units, `mu_d(U)` lies in an interval of width at most `2^u B`.
The checker verifies the deterministic envelope on 800 bounded cubes.

Let

```text
M = 2^k-1,
c(n,M,alpha) = sqrt(log(2M/alpha)/(2n)).
```

Applying [Hoeffding's bounded-sum inequality](https://doi.org/10.1080/01621459.1963.10500830)
and a union bound gives, with probability at least `1-alpha`, simultaneously
for every nonempty `U`,

```text
|hat_nu(U)-nu(U)| <= epsilon_u,
epsilon_u = 2^u B c(n,M,alpha).
```

Because `|mu_d(U)|` lies in `[0,2^(u-1)B]`, a separate simultaneous family of
bounds gives

```text
E|mu_d(U)|
 <= mean_d |mu_d(U)| + 2^(u-1) B c(n,M,alpha).
```

To use both families at joint level `1-alpha`, allocate failure probability
between them, for example `alpha/2` each.

These are worst-case bounds. Variance-adaptive or time-uniform confidence
sequences can be substantially tighter, but their sampling and predictability
conditions must be declared. [Howard et al.'s confidence-sequence work](https://arxiv.org/abs/1810.08240)
is relevant to sequential releases; it does not make repeated reuse of a
completed transfer set free of adaptivity.

## 6. What 64 six-atom documents can certify

For the planned diagnostic,

```text
k=6, n=64, M=63, alpha=0.05,
c=0.2473612961.
```

The simultaneous coefficient radii, divided by `B`, are:

| Order `u` | `epsilon_u/B` |
| ---: | ---: |
| 1 | 0.495 |
| 2 | 0.989 |
| 3 | 1.979 |
| 4 | 3.958 |
| 5 | 7.916 |
| 6 | 15.831 |

Summed over all 63 nonconstant coefficients, the uncertainty contribution in
`C(R)` is

```text
sum_(u=1)^6 binom(6,u) epsilon_u = 180.079 B.
```

This is vacuous relative to the trivial `B` bound on population cube
oscillation. Under the same worst-case calculation, merely obtaining
`epsilon_6<=B` would require at least 16,040 independent source units.

This is not evidence that the six-atom field lacks structure. It is evidence
that full high-order population reconstruction is the wrong confirmatory
estimand at `n=64`.

For a frozen candidate `S` against baseline `T`, the direct paired contrast

```text
Z_d = ell_d(S)-ell_d(T)
```

lies in `[-B,B]` and avoids the `2^u` Möbius amplification. One frozen contrast
has Hoeffding radius

```text
2 B sqrt(log(2/alpha)/(2n)),
```

with only a small multiplicity adjustment if a predeclared candidate family is
confirmed. Directional sign tests can avoid magnitude sensitivity but pay for
ties through low visibility.

## 7. Revised experiment logic

The six-atom proposal cube has two distinct jobs that must not be conflated:

1. **Discovery:** on proposal documents 8--71, measure exact coefficients,
   absolute mass, widths, exchange defects, Q20/Q32 discrepancy oscillation,
   and fold stability. These are descriptive or cross-fitted discovery results.
2. **Compression:** select a small support, move, or fixed-budget family whose
   empirical benefit is not created by one source cluster or one objective
   representation.
3. **Freeze:** declare the candidate, baseline, estimand, representation,
   source unit, stopping rule, and multiplicity before new outcomes are read.
4. **Confirmation:** estimate direct paired contrasts and document direction on
   a new source-clustered block. Do not attempt to reconfirm all 63 coefficients
   unless the sample size is designed for that estimand.

The completed transfer documents 72--135 and reserved documents 136--212 remain
outside structure selection.

## Conjectures and falsifiers

### C11-A: a low-width retained surrogate has a nontrivial empirical certificate

- State: `open`
- Claim: a support with induced width at most two retains most cross-fold stable
  absolute coefficient mass after Q20/Q32 alignment.
- Falsifier: omitted stable mass is comparable to total mass, or the selected
  graph changes materially across folds.

### C11-B: variance adaptation makes selected coefficients estimable

- State: `open`
- Claim: predeclared bounded empirical-Bernstein or confidence-sequence radii
  for the selected terms are materially smaller than the worst-case Hoeffding
  radii while preserving source-cluster validity.
- Falsifier: within-source dependence, variance, or rare large coefficients
  keeps the certificate at the trivial objective-oscillation bound.

### C11-C: compressed direct confirmation is more stable than coefficient transfer

- State: `open`
- Claim: a frozen move selected from the proposal cube has stable paired
  document direction on a new block even when individual Möbius coefficients
  vary.
- Falsifier: the direct candidate contrast reverses sign or remains dominated by
  ties, sources, or objective representation.

### C11-D: objective discrepancy is small on selected vertices

- State: `open`
- Claim: after exact scale alignment, Q20/Q32 discrepancy oscillation on the
  frozen candidate family is small relative to its measured advantage.
- Falsifier: the representation-transfer bound is as large as the candidate
  advantage or changes the selected minimizer.

## Decision

Replace the separate questions “is it sparse?” and “can it be optimized?” with
one certificate curve:

```text
width budget
  -> retained support
  -> omitted empirical mass plus simultaneous uncertainty
  -> exact surrogate optimizer
  -> population-regret upper bound.
```

The immediate six-atom run remains authorized only as a proposal-side
diagnostic. Its output should be a small frozen confirmatory contrast, not a
claim that 64 documents identify the full population Boolean polynomial.

## Open work

- Derive a source-cluster empirical-Bernstein version of `C(R)` with a
  predeclared range contract.
- Optimize retained mass subject to an induced-width budget without enumerating
  all supports.
- Replace coefficientwise union bounds with a direct oscillation bound for a
  selected surrogate class.
- Add backpointer reconstruction to the exact variable-elimination checker.
- Quantify how sign-test visibility and magnitude certificates combine in one
  promotion rule.
