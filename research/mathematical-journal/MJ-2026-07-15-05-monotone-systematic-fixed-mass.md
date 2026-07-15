# MJ-2026-07-15-05: Monotone wide loss and systematic fixed-mass proposals

- Date: 2026-07-15
- Status: v3 causal control complete; rescue changes sampled trunk magnitude
  but not direction or descent; bounded systematic-`K` experiment authorized
- Refines:
  [MJ-2026-07-15-04](MJ-2026-07-15-04-three-geometry-optimization.md)
- Code binding:
  [`rsqrt_lut_8bit.rs`](../../crates/nsrl-core/src/rsqrt_lut_8bit.rs),
  [`training.rs`](../../crates/nsrl-train/src/production/training.rs), and
  [`alignment.rs`](../../crates/nsrl-train/src/production/alignment.rs) as
  inspected on 2026-07-15
- Artifact binding:
  [`p10m-gradient-lane-alignment-v2-contract.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v2-contract.json),
  [`p10m-gradient-lane-alignment-v2.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v2.json),
  [`p10m-gradient-lane-alignment-v3-contract.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v3-contract.json), and
  [`p10m-gradient-lane-alignment-v3.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v3.json)
- Executable check:
  [`check-fixed-mass-theory-v1.mjs`](../../scripts/check-fixed-mass-theory-v1.mjs)

## Question

Can the wide integer objective and a reciprocal-free fixed-mass proposal be
given exact guarantees strong enough to replace guessed scales? What does the
rescue-stratified v2 failure say those guarantees can and cannot repair?

## Result

Four open items from MJ-04 now have bounded answers.

1. Raising the target logit by one Q8 cell cannot increase the proposed
   `B=47` wide logit-anchored NLL. This follows from an exact bound on the
   committed fractional exponent table, not a floating-point approximation.
2. A single seeded systematic phase converts integer weights into exact mass
   `K`, is exactly unbiased over phases, has coordinate error strictly below
   one count, and has normalized squared error at most `V/(4K^2)`. This is a
   stronger rate than independent categorical counts.
3. For `V=8192`, `K=2^16` is the centered preflight value: it gives eight
   counts per token under a uniform distribution, normalized L2 RMS error at
   most `6.91e-4`, and conservative hidden-gradient magnitude below `2^32`.
4. On the frozen v2 coordinates, replacing nonzero rescue with plain RHU changes
   all four exposed aggregate trunk magnitudes by one count but changes no sign
   and improves neither proposal agreement nor exact descent. Rescue is a real
   perturbation, not the cause of the sampled directional failure.

These results repair the output proposal's mass and sampling semantics. They do
not repair the trunk oracle by themselves. The v2 artifact found that all five
public lanes chose the same sampled trunk signs and that the primary
mass-corrected lane lost to paired random signs on rescue-exposed proposal
coordinates. Fixed mass must therefore be evaluated beside a causal no-rescue
control, not promoted as a complete optimizer.

## 1. Exact target monotonicity of the wide objective

Use the MJ-04 notation. Logits `ell_i` are Q8 integers, `m=max_i ell_i`, and
the wide exponent weight is

```text
W_B(delta) = floor((L[f] 2^(B-15)) / 2^n),
n = floor(-delta/256),
f = (-delta) mod 256,
delta <= 0,
B = 47.
```

Let

```text
Z_B = sum_i W_B(ell_i-m),
W_0 = W_B(0),
J_B(ell,y) = log2 Z_B - log2 W_0 - (ell_y-m)/256.
```

### 1.1 Table-bound lemma

**Code observation.** Exhaustive inspection of every one-cell transition in
the committed Q15 fractional table gives

```text
W_0 = 32767 * 2^32 = 140733193388032,
max_q [W_B(-(q-1))-W_B(-q)] = 88 * 2^32,
```

where the maximum occurs at `q=1`.

**Lemma 1 (exact step sandwich).** The maximum one-cell increase, normalized by
the maximum weight, satisfies

```text
Delta W / W_0 <= 88/32767
                     < 27/10000
                     < 2^(1/256)-1.
```

**Proof.** The first inequality is the exhaustive integer table bound. The
middle inequality follows from

```text
88 * 10000 < 27 * 32767.
```

The final inequality is equivalent to

```text
(10027)^256 < 2 * (10000)^256,
```

which the executable check verifies with exact arbitrary-precision integers.
No floating-point comparison enters the proof. `QED`.

### 1.2 Monotonicity theorem

**Theorem 1 (one-cell target monotonicity).** Holding all non-target stored
Q8 logits fixed, increasing `ell_y` by one cannot increase `J_B`.

**Proof.** First suppose the target is not maximal before the change. Then `m`
does not change, only the target weight increases, and `Z_B>=W_0`. Therefore

```text
Delta J_B
  = log2(1 + Delta W/Z_B) - 1/256
  <= log2(1 + Delta W/W_0) - 1/256
  < 0
```

by Lemma 1. This also covers a target one cell below a tied maximum.

Now suppose the target is maximal before the change. Its gap remains zero. The
new maximum either stays fixed or rises with the target; in the latter case
every non-target gap weakly increases, every non-target weight weakly
decreases, and `Z_B` cannot increase. The target anchor term remains zero, so
`J_B` cannot increase. `QED`.

**Corollary 1.** Repeated target increases make `J_B` nonincreasing. Thus the
target-floor failure of the old Q15 score is removed without introducing a
one-cell reversal. This is a property of the declared integer objective; it
does not assert that every coarse backward proposal descends it.

## 2. Seeded systematic fixed-mass apportionment

Let nonnegative integer weights `w_i` have `Z=sum_i w_i>0`, let
`p_i=w_i/Z`, and choose fixed integer proposal mass `K`. In a contract-bound
vocabulary order define prefixes

```text
C_i = sum_(j=0)^i w_j,
C_-1 = 0.
```

Choose a phase `R` uniformly from the integers `{0,...,Z-1}` using a
counter-based generator bound to the training contract, step, example, and
operation. Define

```text
A_i = floor((K C_i + R)/Z),
A_-1 = 0,
a_i = A_i - A_(i-1),
g_i = a_i - K 1[i=y].
```

For a fixed seed and counter this is deterministic and exactly replayable.
Randomness below refers to the ensemble over `R`, not nondeterministic
execution.

### 2.1 Exact mass and unbiasedness

**Lemma 2 (uniform shifted floor).** For integer `x` and
`R~Uniform{0,...,Z-1}`,

```text
E_R floor((x+R)/Z) = x/Z.
```

**Proof.** Write `x=qZ+r`, `0<=r<Z`. The floor is `q` for `Z-r` phases and
`q+1` for `r` phases. Its expectation is `q+r/Z=x/Z`. `QED`.

**Theorem 2 (systematic proposal guarantees).** The construction satisfies

```text
a_i >= 0,
sum_i a_i = K,
sum_i g_i = 0,
E_R[a_i] = K p_i,
E_R[g] = K(p-e_y),
|a_i-Kp_i| < 1.
```

**Proof.** Prefix monotonicity gives nonnegative increments. The sum telescopes
to `floor((KZ+R)/Z)=K`. Lemma 2 applied to adjacent prefixes gives exact
coordinate expectations. For a fixed coordinate, subtracting the two floors
counts lattice points in an interval of length `K w_i/Z`, hence the count is
either its floor or ceiling and differs from the quota by less than one.
Subtracting the target mass gives the claims for `g`. `QED`.

The deterministic `R=0` rule from MJ-04 retains the exact-mass and error
bounds but has stable rounding bias. A seeded uniform phase removes its
marginal bias without requiring `K` independent categorical draws.

### 2.2 Exact coordinate variance and global error

Write `Kp_i=n_i+f_i` with integer `n_i` and `0<=f_i<1`.

**Theorem 3 (variance).** Under a uniform phase,

```text
P(a_i=n_i+1) = f_i,
P(a_i=n_i)   = 1-f_i,
Var(a_i) = f_i(1-f_i) <= 1/4.
```

Consequently,

```text
E ||a/K-p||_2^2
  = (1/K^2) sum_i f_i(1-f_i)
  <= V/(4K^2),

sqrt(E ||a/K-p||_2^2) <= sqrt(V)/(2K).
```

Independent categorical counts instead obey only

```text
E ||a/K-p||_2^2 = (1-||p||_2^2)/K <= 1/K.
```

Systematic apportionment therefore has an `O(K^-2)` normalized mean-square
bound rather than `O(K^-1)`. Its tradeoff is correlated coordinate error.

### 2.3 What vocabulary order changes

Let `e_i=a_i-Kp_i` and prefix error `E_i=sum_(j=0)^i e_j`. Then
`|E_i|<1`, with `E_-1=E_(V-1)=0`. For any downstream scalar coefficients
`v_i`, summation by parts gives

```text
sum_i e_i v_i
  = sum_(i=0)^(V-2) E_i (v_i-v_(i+1)),

|sum_i e_i v_i|
  < sum_(i=0)^(V-2) |v_i-v_(i+1)|.
```

Thus order does not change marginal unbiasedness or the coordinate variance
bound. It changes covariance and controls projection error through the total
variation of downstream coefficients in that order. A contract-bound
pseudorandom permutation is a legitimate covariance control; an undocumented
token-ID order is not a mathematical neutral choice.

## 3. Choosing `K` from explicit budgets

There is no universally optimal `K`. Two interpretable requirements are:

1. a uniform-distribution occupancy floor `K/V >= c`; and
2. a systematic normalized L2 RMS ceiling `sqrt(V)/(2K) <= epsilon`.

Together they give the design rule

```text
K >= max(cV, sqrt(V)/(2 epsilon)).
```

For p10m, `V=8192`. Taking `c=8` and `epsilon=10^-3` gives lower bounds
`65536` and approximately `45255`, so the next power-of-two choice is
`K=2^16`.

| `K` | Uniform count `K/V` | Systematic RMS bound | Categorical RMS bound | Hidden gradient bound |
| ---: | ---: | ---: | ---: | ---: |
| `2^15` | 4 | `1.381e-3` | `5.524e-3` | `<2^31` |
| `2^16` | 8 | `6.905e-4` | `3.906e-3` | `<2^32` |
| `2^18` | 32 | `1.726e-4` | `1.953e-3` | `<2^34` |
| `2^20` | 128 | `4.316e-5` | `9.766e-4` | `<2^36` |
| `2^23` | 1024 | `5.395e-6` | `3.453e-4` | `<2^39` |

The last column uses `||g||_1<=2K` and signed-16-bit output weights, hence a
conservative feature-gradient bound below `2K 2^15=K 2^16`. An individual
output-parameter product is below `K 2^15`. With `B=47`, `Z_B<2^60`, so
`K C_i<2^76` at `K=2^16`; the apportionment numerator requires `u128` but has
ample range.

Moving from the current Q23-like output mass to `K=2^16` removes roughly seven
bits of coefficient magnitude. Downstream shifts cannot be copied unchanged:
the bounded experiment must retune only the explicitly mass-dependent shift
and rerun zero/nonzero, rescue, and saturation measurements.

**Decision.** Preflight `K in {2^15,2^16,2^18}`, centered on `2^16`. Reject a
lane if it violates exact output mass, introduces saturation, or loses output-
head proposal fidelity against the present mass-corrected reference.

## 4. Confidence belongs to documents, not coordinates

Overlapping windows and multiple coordinates from one document are dependent.
The unit of inference for proposal-versus-random comparison is therefore an
independent document block.

For each document and stratum:

1. aggregate the predeclared coordinate/window score difference between the
   proposal and its paired random sign;
2. label the document win, loss, or tie; and
3. apply an exact paired sign/binomial test to non-tied documents.

Do not report coordinate count as sample size. A conservative two-sided
Hoeffding half-width for a Bernoulli document win rate is

```text
h(n,alpha) = sqrt(log(2/alpha)/(2n)).
```

At `alpha=0.05`:

| Independent documents `n` | Half-width bound |
| ---: | ---: |
| 64 | `0.170` |
| 128 | `0.120` |
| 213 | `0.093` |
| 738 | `0.050` |

**Data observation.** The bound dev stream contains 213 documents and 165,146
tokens; all 213 documents are eligible at context 64, yielding 151,088 sliding
windows. Those windows do not create 151,088 independent trials. The present
v2 surfaces use only four documents each and are calibration evidence, not a
confidence-qualified comparison. The existing dev set can detect a large
document-level effect but cannot achieve a conservative `+/-5` percentage-point
Hoeffding half-width. That claim requires a fresh, predeclared confirmation
surface of at least 738 independent eligible documents.

## 5. Reading the rescue-stratified v2 result

**Artifact observation.** For the primary
`mass-corrected-normalized-rhu` lane, v2 reports:

- exact zero output-gradient sum on every example, no gradient or residual
  saturation, and 222 STE rescues across 88,868,864 backward quantizations;
- rescue-exposed trunk proposal agreement of `1/3`, versus `3/3` for paired
  random signs, with the sole exact improving neighbor missed by the proposal;
- no exact improving transfer neighbor in the four rescue-exposed trunk
  samples, so the transfer stratum cannot identify descent quality; and
- output-head proposal and transfer agreement/descent of `2/2`, versus `1/2`
  for paired random signs.

All five public lanes chose the same six sampled signs. This is stronger than a
mere output-normalization diagnosis and weaker than a causal rescue diagnosis:
the sample localizes the observed failure to the trunk, but no lane intervened
by preserving natural zeros while holding the normalized mass-corrected source
fixed.

**Inference.** Systematic fixed mass can test whether output discretization and
per-example weighting are obstructing learning. It cannot, by construction,
repair a wrong sign already created in the upstream backward chain. A trunk
result must therefore be stratified and reported separately from the output
head in every `K` comparison.

### 5.1 Causal no-rescue replay v3

The v3 contract was frozen before execution. It retained the v2 model, seed,
eight windows over four proposal documents, eight windows over four different
transfer documents, and all six selected coordinates. The no-rescue lane was
excluded from coordinate selection and evaluated only after the v2 union and
source-specific rescue strata were fixed. Both historical traces remained
byte-identical, and a second v3 execution reproduced the v3 trace byte for
byte.

**Artifact observation.** The rescued and no-rescue mass-corrected lanes had
identical output-gradient health, including exact zero mass and L1 range
`16,774,752..16,775,588`. Rescue counts changed from 222 to zero. Across the
four rescue-exposed `final_rms` coordinates:

- aggregate magnitudes differed on `4/4`, by exactly one count each;
- aggregate signs differed on `0/4`;
- both lanes retained proposal agreement `1/3` versus paired random `3/3`;
- both selected zero exact proposal descents versus paired random one; and
- their transfer and output-head summaries were identical.

**Inference.** This is a valid causal intervention on the rescue operator. It
shows that rescue perturbs accumulated magnitude at the sampled coordinates,
but it does not mediate the observed sign error or descent failure. Globally
disabling rescue is therefore not an evidence-supported repair. The defect
persists in the normalized mass-corrected source or the backward chain before
the rescued/plain projection sites.

## 6. Revised experimental order

1. **Complete:** add the causal normalized, mass-corrected no-rescue lane and
   replay the exact v2 coordinates.
2. **Next:** add seeded systematic fixed mass with `K={2^15,2^16,2^18}` while preserving
   the source and no-rescue distinction.
3. Compare canonical vocabulary order with one contract-bound pseudorandom
   permutation. Reuse the same seeded phases and coordinates.
4. Score the wide objective on the same-surface proposal audit and a strictly
   document-disjoint transfer audit. Report output, natural trunk, and rescue-
   exposed trunk separately.
5. If a bounded lane beats paired random signs and the current reference,
   confirm it on at least 64 proposal and 64 transfer documents. Reserve a
   conservative five-percentage-point claim for at least 738 documents.
6. Only after direction and descent gates pass should residual resolution,
   learning rate, or paid scale be optimized.

## 7. Conjectures and falsifiers

### C05-A: systematic phase dominates categorical counts at equal `K`

- State: `open`
- Claim: systematic apportionment has lower document-level proposal error and
  no worse saturation than independent categorical counts at the same mass.
- Falsifier: the categorical lane matches or exceeds both same-surface and
  disjoint-transfer fidelity after seeds, `K`, source, and shifts are matched.

### C05-B: order covariance is measurable

- State: `open`
- Claim: a contract-bound vocabulary permutation changes projection error and
  improves the worst document stratum without changing exact mass or marginal
  phase-unbiasedness.
- Falsifier: canonical and permuted orders are byte-identical downstream or the
  permutation does not improve any predeclared document-level metric.

### C05-C: `K=2^16` is sufficient for the output head

- State: `open`
- Claim: `K=2^16` preserves output-head directional fidelity while reducing
  mass-dependent accumulator and shift pressure relative to Q23 scale.
- Falsifier: it loses output-head proposal fidelity, creates excess zeros, or
  requires compensating shifts that reproduce the original numerical pressure.

### C05-D: the v2 trunk defect is upstream of output mass correction

- State: `supported` for magnitude sensitivity and persistence of the defect;
  rescue as a directional repair is falsified on the bounded sample
- Claim: a causal no-rescue control will separate from the rescued source on
  the v2 rescue-exposed coordinates; if not, the defect lies in another
  upstream surrogate or backward-chain operation.
- Existing evidence: v2 supports localization to its sampled trunk coordinates
  but does not identify the responsible upstream operation. V3 verifies source
  equivalence, removes all 222 rescues, changes every exposed aggregate
  magnitude, changes no sign, and leaves fidelity/descent summaries identical.
- Falsifier: rescued and no-rescue sources give the same exact trunk signs and
  magnitudes on the bound coordinates after source equivalence is verified.

## Decision

The theory now authorizes a bounded p10m systematic-`K`/order comparison, not an
optimizer promotion. Wide-objective target monotonicity and systematic fixed-
mass guarantees remove two sources of guesswork. The empirical bottleneck is
the trunk proposal oracle. The causal rescue test is complete and rejects
global rescue removal as a repair; the next test must vary fixed mass while
preserving the rescued/plain distinction and document blocks. Paid scaling
remains unauthorized.

## Open work

- Prove or bound non-target one-cell behavior of `J_B`; target monotonicity does
  not imply every coordinate neighbor is consistently ordered.
- Derive covariance-sensitive downstream bounds for structured vocabulary
  permutations rather than only the total-variation inequality.
- Freeze the systematic-`K`/order contract before observing its results.
- Replace Hoeffding with an exact interval and a predeclared power calculation
  once the document-level endpoint and expected effect are fixed.

## 2026-07-15 implementation update

The centered systematic lanes are now implemented in production Rust as
explicit audit-only sources:

```text
K in {2^15, 2^16, 2^18},
phase = contract-seeded uniform residue modulo Z,
order = token ID ascending.
```

Each window checks `sum_i a_i=K` and `sum_i g_i=0`; the trace records `K`,
phase semantics, and vocabulary order. This closes the implementation gap, not
the empirical conjecture. No systematic-`K` result has authorized training or
scaling.
