# MJ-2026-07-15-12: Finite-group harmonic diagnostics

- Date: 2026-07-15
- Status: exact harmonic diagnostics and a spectral regret certificate
  established; phase-action predictive structure untested
- Extends: MJ-2026-07-15-09 through MJ-2026-07-15-11
- Motivation: attached finite-group/Ramanujan/Walsh research proposal
- Code binding:
  [`check-harmonic-structure-theory-v1.mjs`](../../scripts/check-harmonic-structure-theory-v1.mjs)

## Question

Can quantizer phase and Boolean action structure be analyzed in one exact
finite-group framework without mistaking spectral pseudorandomness for a safe
optimizer?

## Result

Yes, as a diagnostic and surrogate language. The natural index space is

```text
G = (Z/Delta Z) x F_2^k,
```

where `r in Z/Delta Z` is fine-log quantizer phase and `x in F_2^k` indexes an
atomic action subset. The second factor is an analysis group on cube indices;
it does not assert that applying physical actions forms a commutative group.
Canonical replay still defines the value at each index.

Two exact transforms are especially relevant:

- Walsh characters give an orthogonal action-space decomposition using integer
  additions and subtractions;
- Ramanujan subspaces give integer-valued features for exact periodic structure
  in phase.

The important correction is that low Walsh or Gowers uniformity does not by
itself control an optimizer. Optimization requires an oscillation certificate.
The robust-surrogate theorem of MJ-11 converts Walsh residual energy into such
a certificate, with an explicit dimension factor.

## 1. Product-group characters

The complex characters of the product group are

```text
chi_(a,A)(r,x)
 = exp(2 pi i a r / Delta) (-1)^(A dot x).
```

Ordinary characters are the complete orthogonal basis. NSRL need not evaluate
complex exponentials at runtime: Walsh characters are signs, and Ramanujan
sums aggregate cyclic characters of a common exact period into integer-valued
features.

A joint phase-action moment can therefore be accumulated exactly as

```text
J_(q,s,A)
 = sum_i y_i c_q(r_i-s) (-1)^(A dot x_i),
```

for integer outcome `y_i`. Normalization and projection may require an exact
rational scale even when every numerator and basis value is integer.

The observed substrate will not generally cover every point of `G`. Joint
moments are therefore sampled features unless phase and action cells are
deliberately balanced; they are not automatically a complete group transform.

## 2. Exact Walsh analysis of an action cube

Let `N=2^k`, `chi_A(x)=(-1)^(A dot x)`, and define the unnormalized transform

```text
W_d(A) = sum_x ell_d(x) chi_A(x).
```

Then

```text
ell_d(x) = N^(-1) sum_A W_d(A) chi_A(x),

sum_A W_d(A)^2 = N sum_x ell_d(x)^2.
```

Every `W_d(A)` is an integer. For `A!=empty`, adding a constant to all cube
vertices changes nothing. Explicit centering is therefore unnecessary for the
nonconstant spectrum and would only introduce a denominator.

The checker verifies integer inversion, Parseval, and offset invariance on
51,000 vertices from 800 random cubes.

### Relationship to the Möbius basis

If

```text
ell(x) = sum_U mu(U) product_(i in U) x_i,
```

then `x_i=(1-chi_{i}(x))/2`, so the normalized Walsh coefficient is

```text
hat_ell(A)
 = (-1)^|A| sum_(U superseteq A) 2^(-|U|) mu(U).
```

Thus the transforms answer different questions:

- Möbius coefficients are triangular, conditional, and tied to AND actions;
- Walsh coefficients are orthogonal and expose `L2` energy;
- one high-order Möbius term contributes to several Walsh degrees.

Walsh degree must not be reported as irreducible interaction order. A field
whose Walsh energy is mostly degree one is close in `L2` to an affine parity
model, but only the residual oscillation decides whether that approximation is
safe for minimization.

## 3. Spectral optimization certificate

Let a Walsh surrogate retain or estimate some characters. Write the target
minus surrogate residual as

```text
h(x) = sum_(A in T) e_A chi_A(x),
```

where `T` contains every nonconstant character with nonzero residual
coefficient. Define its disagreement geometry

```text
m_T = max_(z != 0) |{A in T : A dot z = 1}|.
```

### Theorem 3.1 (Walsh-energy oscillation bound)

```text
osc(h) <= 2 sqrt(m_T sum_(A in T) e_A^2).
```

**Proof.** For `z=x+y` in `F_2^k`,

```text
h(x)-h(y)
 = sum_(A in T) e_A chi_A(x) (1-chi_A(z)).
```

Only the `m_T(z)` disagreeing characters contribute, each multiplier has
absolute value two, and Cauchy--Schwarz gives

```text
|h(x)-h(y)| <= 2 sqrt(m_T(z)) ||e||_2.
```

Maximize over pairs. By MJ-11 Theorem 1.1, the exact minimizer of the Walsh
surrogate has target-objective regret bounded by the same expression.

For an exact degree-`r` truncation,

```text
e_A = hat_ell(A) for |A|>r,
E_r^W = sum_(|A|>r) hat_ell(A)^2,

regret <= 2 sqrt(m_r E_r^W).
```

This is the optimization-safe meaning of “small high-degree Walsh energy.” The
checker verifies it on 50,800 vertices from 700 random cubes.

For the rank-six cube, the tail geometry is:

| Retained Walsh degree | Residual characters | `m_r` |
| ---: | ---: | ---: |
| 0 | 63 | 32 |
| 1 | 57 | 31 |
| 2 | 42 | 26 |
| 3 | 22 | 16 |
| 4 | 7 | 6 |
| 5 | 1 | 1 |

The dimension factor is real. Orthogonality improves energy accounting but does
not make a diffuse residual harmless to its worst vertex.

## 4. What Gowers norms do and do not authorize

For a normalized Walsh transform on `F_2^k`,

```text
||f||_(U2)^4 = sum_A |hat_f(A)|^4.
```

A large `U2` value relative to `L2` indicates Fourier concentration and can
locate a correlating linear character. A large `U3` residual can motivate a
search for quadratic structure; the
[finite-abelian-group inverse theorem](https://arxiv.org/abs/2112.13759)
provides the appropriate theoretical context. At rank six, however, direct
enumeration of all characters and candidate quadratic phases is simpler than
invoking an asymptotic regularity lemma.

### Counterexample 4.1 (uniform does not mean optimization-safe)

Let `f` be a unit spike on one of `N=2^k` vertices, centered by its cube mean.
Then

```text
osc(f)=1,
||f||_(U2)^4=(N-1)/N^4.
```

The `U2` norm tends to zero while the hidden one-vertex objective excursion
stays one. At rank six, `U2` is approximately `0.0440` and the oscillation is
still exactly one. The checker preserves the counterexample through rank ten.

Therefore a small Gowers-uniform residual may be pseudorandom for arithmetic
pattern counting while remaining dangerous for minimization. It can stop a
search for *correlating low-complexity patterns*; it cannot authorize dropping
the residual without a tail, oscillation, or direct candidate certificate.

The [Green--Tao structure/error/uniform decomposition](https://arxiv.org/abs/1002.2028)
is a useful research metaphor, but its small-`L2` and small-Gowers components
serve counting theorems under their hypotheses, not NSRL global-regret bounds.

## 5. Ramanujan analysis of dyadic quantizer phase

For `q=2^j`, the Ramanujan sum simplifies to

```text
c_(2^j)(n) =
  2^(j-1)   if 2^j divides n,
 -2^(j-1)   if 2^(j-1) divides n but 2^j does not,
  0         otherwise.
```

Thus the proposed phase features are a multiscale test of two-adic residue
structure. They do not import unrelated partition or modular-form identities.

When `q` divides `Delta`, the shifted vectors span the exact-period Ramanujan
subspace of dimension `phi(q)`. Distinct divisor subspaces are orthogonal over
a complete `Delta`-cell. The projection matrices have integer numerators with
an overall scale factor, consistent with the
[Ramanujan-subspace signal-processing construction](https://authors.library.caltech.edu/records/3c8eh-xc275).

For the Q32-to-Q20 cell `Delta=4096`, the checker verifies:

- the integer power-two formula for all 13 divisors
  `q in {1,2,4,...,4096}`;
- zero mean for every nonconstant period vector;
- squared norm `Delta phi(q)`;
- 631 shifted cross-subspace orthogonality cases.

Ramanujan analysis is not automatically compressed. A localized boundary step
can require many shifts and period subspaces. Its first use should be to test a
specific claim: whether the *pre-action phase distribution* or an outcome
conditional on pre-action phase contains stable dyadic periodic structure.

## 6. Leakage-safe phase-action experiment

The exact crossing

```text
chi_Delta = floor((r+h)/Delta)
```

is already determined once realized displacement `h` is known. A model that
uses `h` to “predict” the crossing merely leaks the answer. Separate the roles:

- **proposal features:** pre-action phase `r`, action identity, semantic group,
  source, and state observables available before replay;
- **outcomes:** realized displacement `h`, crossings, component cancellation,
  Q20/Q32 direction, and document loss;
- **post-hoc explanation only:** features computed from realized `h` or the
  output trace.

For the six-atom proposal cube:

1. compute exact per-document Walsh numerators and degree energies alongside
   the existing Möbius coefficients;
2. construct source-stratified pre-action phase histograms and project them
   onto `q=2^j` Ramanujan subspaces;
3. accumulate pre-action joint features
   `c_q(r-s)(-1)^(A dot x)` and cross-fit their prediction of visibility and
   direction;
4. compare with phase-, group-, cardinality-, and source-matched controls;
5. measure the Walsh spectral regret bound and the Q20/Q32 discrepancy
   oscillation for every proposed retained surrogate;
6. freeze only a small direct candidate family for new confirmation, as required
   by MJ-11's sample-size analysis.

Ramanujan energy under a finite sample must be compared with a source-clustered
null or held-out prediction. Exact arithmetic eliminates numeric roundoff, not
sampling variation.

## 7. Empirical decisions

| Observation | Consequence |
| --- | --- |
| Stable low-degree Walsh energy and a small spectral regret bound | Test an orthogonal low-degree surrogate; factor its parity terms and audit induced width |
| Stable Walsh energy but a vacuous oscillation bound | Use the spectrum descriptively; do not optimize from it |
| Stable nonconstant pre-action Ramanujan energy | Develop phase-aware proposal features at the implicated dyadic periods |
| Ramanujan energy disappears after source controls | Treat it as corpus composition, not quantizer dynamics |
| Joint pre-action phase-action features predict held-out visibility and direction | Freeze a small phase-conditioned candidate comparison |
| Prediction works only with realized `h` or crossings as inputs | Reject it as leakage |
| Large `U3` residual after removing linear modes | Inspect exact quadratic phases and higher-order Möbius terms |
| Small Gowers norm but large residual oscillation | Call the residual pattern-uniform but optimization-unsafe |
| No stable phase or action spectrum | Return to tail/width/exchange tests or an explicitly heuristic direct search |

## 8. Compressed recovery boundary

Incomplete Fourier recovery is not licensed by observing a sparse-looking
spectrum. The cited Candès--Romberg--Tao result reconstructs a sparse signal from
randomly incomplete frequency information under explicit sparsity and sampling
conditions. By duality and related results, sparse spectral recovery from time
samples is plausible, but arbitrary adaptive NSRL vertex queries are not the
same design.

Do not compress the rank-six experiment: all 64 action vertices are available.
Consider sparse recovery only for later ranks, after proposal folds establish a
stable sparsity model and the query distribution matches a proved recovery
theorem.

## Conjectures and falsifiers

### C12-A: the action field has stable low-degree orthogonal energy

- State: `open`
- Claim: most cross-fold stable nonconstant Walsh energy lies below a small
  degree and yields a nontrivial spectral regret bound.
- Falsifier: energy or selected modes move across folds, or the dimension factor
  makes the regret bound as large as the cube oscillation.

### C12-B: quantizer phases have stable dyadic structure

- State: `open`
- Claim: one or more nonconstant `q=2^j` phase components replicate across
  source folds after controlling for source and target frequency.
- Falsifier: projection energy matches the clustered null or changes period
  across folds.

### C12-C: pre-action joint harmonics predict visibility

- State: `open`
- Claim: a small predeclared phase-action feature family improves held-out
  visibility and direction prediction over group/cardinality controls.
- Falsifier: improvement vanishes without realized displacement features or on
  a new source fold.

### C12-D: Gowers-uniform residuals are still oscillation-small on this substrate

- State: `open`
- Claim: after removing selected structured modes, both the declared Gowers
  norm and the direct residual oscillation are small.
- Falsifier: the residual is uniform by a Gowers diagnostic but retains a
  material best-vertex excursion.

## Decision

Add Walsh spectra and leakage-safe Ramanujan phase features to the exact
six-atom proposal diagnostic. Do not use Gowers uniformity, Ramanujan energy, or
spectral sparsity alone as an optimizer gate. Promotion requires the spectral
oscillation certificate, representation concordance, and prospective direct
confirmation developed in MJ-11.

## Open work

- Derive simultaneous source-cluster confidence bounds for selected Walsh and
  Ramanujan moments.
- Compute the exact relation between retained Walsh parity factors and minimum
  induced width for the observed six-atom support.
- Define a clustered null for phase-subspace energy under unequal source and
  target frequencies.
- Implement exact quadratic-phase enumeration for the rank-six `U3` diagnostic.
- Determine whether cyclic Fourier, Ramanujan, or dyadic Haar features give the
  smallest leakage-safe phase model on observed traces.
