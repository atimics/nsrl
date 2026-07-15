# MJ-2026-07-15-09: Objective-boundary phase calculus

- Date: 2026-07-15
- Status: exact output-boundary decomposition established; reverse proposal map open
- Extends: MJ-2026-07-15-08
- Code binding:
  [`attention.rs`](../../crates/nsrl-core/src/attention.rs),
  [`boolean_jet.rs`](../../crates/nsrl-train/src/production/boolean_jet.rs)
- Executable binding:
  [`check-objective-boundary-phase-theory-v1.mjs`](../../scripts/check-objective-boundary-phase-theory-v1.mjs)

## Question

Can the high tie rate in the Boolean-jet confirmation be decomposed into an
exact boundary phenomenon rather than treated as generic experimental noise?

## 1. The implemented objective has two logarithmic boundaries

For a positive integer `n`, define `lambda_f(n)` as the integer produced after
`f` fractional-bit iterations of the same deterministic fixed-point logarithm
algorithm used by the source. Its intended mathematical value is
`floor(2^f log2(n))`, but the results below require only the algorithmic prefix
property, not an unproved real-log approximation bound:

```text
lambda_f(n) = f-bit output of the declared integer log algorithm.
```

The canonical implemented window loss is

```text
L_q = lambda_q(W) - lambda_q(T),    q=20,
```

where `W` is the sum of integer exponent weights and `T` is the target weight.
This is a difference of two truncated logarithms, not one rounding of an
already-subtracted real NLL.

**Code observation 1.1.** `base2_softmax_nll_q20` constructs Q15 exponent
weights and returns `log2_u64_q20(weight_sum)-log2_u64_q20(target_weight)`.
The Q47-logit-anchored variant changes the exponent weights but calls the same
Q20 logarithm for both terms. It is a different approximation environment, but
it does not refine the final logarithmic cell width.

**Code observation 1.2.** The confirmation code sums the Q20 window losses by
document and only then forms `joint-head`. A document tie can therefore arise
either because no window component crosses a Q20 boundary or because nonzero
window components cancel.

## 2. Exact quantizer-phase identity

Fix a diagnostic precision `F>q` and let

```text
delta = 2^(F-q).
```

For fine-grid integers `y_0,y_1`, define their coarse-cell crossing count

```text
chi_delta(y_0,y_1)
  = floor(y_1/delta) - floor(y_0/delta).
```

Write `h=y_1-y_0` and let `r=y_0 mod delta`, with `0<=r<delta`.

### Proposition 2.1 (phase form)

```text
chi_delta(y_0,y_1) = floor((r+h)/delta).
```

**Proof.** Write `y_0=k delta+r`. Then
`y_1=k delta+r+h`; subtracting the two coarse quotients leaves the displayed
floor. The executable checker verifies 10,000 signed-displacement cases.

This separates two quantities that the aggregate Q20 result confounds:

- `h`, the fine-grid movement;
- `r`, the baseline phase or distance to the next coarse boundary.

A nonzero `h` can remain invisible when the phase does not expose a boundary.
Conversely, a small `h` can be visible when the baseline lies close to one.

### Proposition 2.2 (exact refinement)

For every positive integer `n`,

```text
lambda_q(n) = floor(lambda_F(n)/delta).
```

**Proof.** Both executions have the same normalized Q63 initial state. Their
first `q` squaring, truncation, threshold, and renormalization decisions are
identical; only the bit positions in the result word differ by `F-q`. Therefore
the leading `q` fractional bits of the `F`-bit execution are exactly the Q20
execution. A higher-precision diagnostic using the same state machine and
integer exponent weights is consequently an exact refinement of the declared
Q20 algorithm, not a surrogate objective. The checker verifies this prefix
property on 4,096 positive integer inputs.

## 3. Exact NLL-boundary decomposition

Consider one window under a baseline state `0` and candidate state `1`. Let
`W_0,W_1` be their weight sums and `T_0,T_1` their target weights. Define

```text
y_W,s = lambda_F(W_s),
y_T,s = lambda_F(T_s).
```

### Theorem 3.1 (two-component output boundary calculus)

The exact observed Q20 loss contrast is

```text
Delta L_q
  = chi_delta(y_W,0,y_W,1) - chi_delta(y_T,0,y_T,1).
```

**Proof.** Apply Proposition 2.2 to each of the four logarithm terms and
regroup. The checker verifies the identity on 5,000 random positive integer
weight quadruples.

For a document `d` with windows `w`, the conditional Boolean contrast therefore
has the exact component expansion

```text
C_d^q(A|B)
  = sum_w [chi_W(d,w;A|B) - chi_T(d,w;A|B)].
```

Define signed components `c_(d,j)` to include the positive denominator
crossings and negative target crossings, and define

```text
A_d = sum_j |c_(d,j)|,

kappa_d = |sum_j c_(d,j)| / A_d       when A_d>0.
```

`A_d` is output-boundary activity and `kappa_d` is output-boundary coherence.
They provide an exact tie taxonomy:

1. **fine-grid inactivity:** every component displacement `h` is zero;
2. **phase masking:** some `h` is nonzero but `A_d=0`;
3. **component cancellation:** `A_d>0` but `kappa_d=0`;
4. **objective visibility:** `A_d>0` and `kappa_d>0`.

The existing artifact records only the final document contrast, so it cannot
distinguish cases 1--3. The checker preserves an explicit example with boundary
activity two whose signed components sum to a document tie.

## 4. A phase-mixing model for proposal planning

The runtime is deterministic; phase is not random inside one replay. Across a
predeclared document population, however, phase can be studied as a covariate.
The following is a reference model, not an assumption to insert silently.

### Proposition 4.1 (uniform discrete phase)

If `r` is uniform on `{0,...,delta-1}` and the integer fine-grid displacement
`h` is fixed with `|h|<=delta`, then

```text
E[chi_delta] = h/delta,

Pr(chi_delta != 0) = |h|/delta.
```

**Proof.** For positive `h`, exactly `h` phases lie in the upper boundary
interval and produce crossing `+1`. For negative `h`, exactly `|h|` phases lie
in the lower interval and produce crossing `-1`. The checker exhausts all 129
displacements for a 64-point cell.

This suggests the pre-outcome component exposure score

```text
e_phi(h) = min(1, |h|/delta),
```

but it does not justify adding component exposure scores to obtain document
visibility. Components share logits, phases are not known to be uniform, and
opposite crossings can cancel. Empirical phase histograms, activity, coherence,
and final document visibility must all be reported.

## 5. Relation to hierarchical Boolean jets

MJ-08 showed that the coarse trunk/head interaction is a signed pushforward of
45 atomic cross-support coefficients. The phase calculus answers a different
question: given a candidate branch transition, why does its output loss become
visible or tied at Q20?

It does not yet compute the branch transition without evaluating the candidate.
The remaining breakthrough is a compressed reverse map

```text
parameter/block move
  -> bounded fine-log component displacements and phases
  -> predicted (activity, coherence, visibility, direction).
```

This is a more precise target than a generic “discrete gradient.” It is a
boundary adjoint whose output is a distributional certificate tuple. Exact
fine-log phase traces can supply labels for this map without changing deployed
inference or consuming reserved confirmation documents.

## Engineering issues implied by the theory

1. **P0 — preserve protocol immutability.** Keep the completed v1 confirmation
   replayable. Introduce phase traces and the newly required matched control
   only in a v2 contract.
2. **P0 — add an exact refined-log diagnostic.** Generalize the private integer
   log routine to a checked diagnostic precision such as Q32 and prove that
   downshifting each denominator and target term reproduces Q20 exactly.
3. **P1 — record boundary components per window.** For `B` and `A union B`,
   store fine-log denominator/target values, phases, displacements, Q20 crossing
   counts, and document cancellation summaries.
4. **P1 — separate visibility failures.** Report the fractions of documents
   assigned to fine-grid inactivity, phase masking, cancellation, and visible
   contrast. Do not collapse all four into `ties`.
5. **P1 — train the proposal on certificate targets.** Candidate ranking should
   predict magnitude, component activity, coherence, document visibility,
   direction, and representation concordance on cross-fitted proposal blocks.
6. **P1 — phase-match controls.** Group/cardinality matching is insufficient;
   controls should also match pre-outcome fine-displacement and boundary-phase
   exposure distributions.
7. **P2 — localize upstream boundary creation.** After the output trace is
   stable, record the first layer at which the four Boolean branches differ and
   recursively bind that event to the output components.

## Decision

Use quantizer phase and component cancellation as first-class variables in the
next proposal theory. Do not interpret the Q47 exponent path as higher output
resolution, and do not spend the 77 reserved documents merely to measure
another candidate whose Q20 visibility is unknown.

The next bounded experiment should run only on proposal documents: add the
exact refined-log trace, classify every existing tie, and test whether phase
exposure plus coherence predicts document visibility. This is diagnostic work,
not optimizer promotion.

## Conjectures and falsifiers

### C09-A: phase exposure predicts Q20 visibility

- State: `open`
- Claim: fine-log displacement and baseline phase predict component boundary
  crossings out of sample better than group/cardinality features alone.
- Falsifier: a frozen phase-aware predictor does not beat its matched control on
  unused proposal documents.

### C09-B: cancellation explains a material share of ties

- State: `open`
- Claim: some current document ties have nonzero component activity and zero
  coherence rather than complete functional inactivity.
- Falsifier: nearly every tie has zero component activity.

### C09-C: refined output resolution exposes concordant signal

- State: `open`
- Claim: the same-algorithm refined log grid increases visibility and its
  aggregated contrast direction agrees with Q20 whenever the Q20 contrast is
  larger than the declared refinement-error margin.
- Falsifier: finer-grid contrasts show persistent sign disorder outside the
  proved uncertainty band.

## Open work

- Derive a reverse boundary-support bound through integer linear,
  normalization, and activation nodes.
- Establish whether observed phases are sufficiently mixed for Proposition 4.1
  to be useful as a planning model.
- Derive cluster-robust confidence bounds for multiple windows per document and
  related documents per source.
- Combine phase exposure with the MJ-08 block-support variation bound so that a
  hierarchical search can prune blocks without hiding negative atomic terms.
