# MJ-2026-07-15-04: Three-geometry optimization with fixed-mass proposals

- Date: 2026-07-15
- Status: rescue-stratified v2 fails the trunk gate; its causal rescue question
  is resolved by MJ-05/v3; optimizer refinement and paid scaling remain
  unauthorized
- Supersedes:
  - the observation map in MJ-03 that omitted deployed forward scales;
  - the interpretation of `w - Z e_y` as an objective-equivalent minibatch
    gradient;
  - the factorization-based definition of a strongly prime escape; and
  - any claim that fiber-exit prediction can generally skip gradient computation
- Preserves:
  - the full-state/fiber correction to the plain functional quotient;
  - exact mass conservation as an output-gradient invariant;
  - saturation-debt accounting; and
  - finite-difference calibration as the authority on the integer lattice
- Code binding:
  [`attention.rs`](../../crates/nsrl-core/src/attention.rs),
  [`production.rs`](../../crates/nsrl-train/src/production.rs),
  [`training.rs`](../../crates/nsrl-train/src/production/training.rs), and
  [`alignment.rs`](../../crates/nsrl-train/src/production/alignment.rs) as
  inspected on 2026-07-15
- Artifact binding:
  [`p10m-probability-normalization-signal-attribution.json`](../../benchmarks/production-model-v1/p10m-probability-normalization-signal-attribution.json),
  [`p10m-gradient-lane-alignment-v1-contract.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v1-contract.json),
  [`p10m-gradient-lane-alignment-v1.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v1.json),
  [`p10m-gradient-lane-alignment-v2-contract.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v2-contract.json), and
  [`p10m-gradient-lane-alignment-v2.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v2.json)

## Question

How can NSRL define an integer optimizer whose state, objective, proposal, batch
semantics, and acceptance test refer to the same mathematical problem?

## Executive correction

Integer training contains three distinct geometries:

1. **deployed geometry** — which full states realize the same exact forward
   function;
2. **objective geometry** — which stored-parameter neighbors improve a declared
   integer loss; and
3. **proposal geometry** — which direction a coarse backward oracle recommends.

The previous theory blurred these geometries. In particular, the raw
exponential-weight vector is a valid zero-sum per-example surrogate, but its
scale is the example-dependent normalization mass `Z`. Summing those vectors in
a batch reweights examples by `Z`; it is not generally a scalar multiple of the
normalized batch gradient.

The corrected optimization program uses a fixed integer mass `K` for every
example. Two constructions are derived below:

- deterministic cumulative apportionment, with exact mass and less than one
  count of coordinate error relative to the normalized LUT distribution; and
- counter-seeded categorical counts, which are exactly unbiased for the
  normalized distribution and require neither a reciprocal nor integer
  division in the sampling path.

The canonical objective is also revised. A wide exponent accumulator plus a
direct target-logit term removes the artificial 32-bit loss floor caused when
the Q15 target exponent annihilates.

## 1. Three geometries

### 1.1 Full Markov state

**Definition.** Split the complete state into

```text
X_t = (D_t, H_t, C_t),
```

where:

```text
D_t = (theta_t, sigma_t^f, A_t)
H_t = (r_t, sigma_t^b, controller_t, seed_t, counter_t, batch_t)
C_t = (data_cursor_t, schedule_t).
```

- `D_t` is deployed state: stored parameters, every forward projection scale,
  and architecture constants that affect inference.
- `H_t` is hidden training state: residuals, backward/update scales, adaptive
  controller state, deterministic-random counters, and partial batch sums.
- `C_t` is exogenous schedule state.

Given an admissible control `a_t`, the transition is deterministic:

```text
X_{t+1} = T(X_t, a_t).
```

Seeded stochastic rounding or categorical sampling does not break deterministic
replay when its seed and counter are components of `H_t`. Randomness enters only
when results are considered across an ensemble of seeds.

### 1.2 Correct observation map and fibers

**Definition.** For a fixed external contract `C_ext` containing tokenizer and
input semantics, define

```text
pi(X) = [F_Cext(D, .)].
```

Forward scales belong in `D`: production serializes them with the model and they
change Q/K, MLP, and output projections. Backward learning-rate shifts remain
in `H` because they do not directly change inference.

The functional fiber is

```text
Fib(v) = { X : pi(X) = v }.
```

Two states can share a fiber while differing in residual, batch, cursor, or
random-counter state. Those differences can change the next exit, so `pi(X)` is
not Markov even though `X` is.

### 1.3 Objective and proposal geometry

For an evaluation surface `E`, define exact stored-state loss

```text
L_E(X) = L_E(pi(X)).
```

This function is constant on a functional fiber. It need not be differentiable
with respect to stored parameters.

Let `G_P(X)` be the coarse proposal produced on proposal surface `P`. `G_P` is a
vector field on full states, not a derivative by definition. Calling it a
gradient is an implementation convention until finite differences establish
alignment.

This yields three different notions of nearness:

- same exact deployed function (`pi`-equivalence);
- adjacent stored parameter cells with similar loss; and
- similar proposal vectors.

None implies either of the others.

## 2. Objective hierarchy

### 2.1 Ideal reference objective

Let Q8 logits represent real values `ell_i / 256`. For target `y`, the ideal
base-2 cross-entropy in bits is

```text
J*(ell, y)
  = log2(sum_i 2^((ell_i-m)/256)) - (ell_y-m)/256,
m = max_i ell_i.
```

Its real logit gradient is proportional to

```text
p* - e_y,
p*_i = 2^((ell_i-m)/256) / sum_j 2^((ell_j-m)/256).
```

`J*` is a reference objective. NSRL does not evaluate those real exponentials.

### 2.2 Current Q15 LUT objective

The current exponential approximation returns

```text
W_15(delta) = LUT(delta),
```

with `W_15(delta)=0` once the negative logit gap reaches 15 bits. The current
canonical integer objective is essentially

```text
J_15 = log2(sum_i W_15(ell_i-m)) - log2(W_15(ell_y-m)),
```

with a caller-supplied constant when the target weight is zero.

**Proposition 1.** `W_15 - Z_15 e_y` is not the derivative of `J_15`.

**Reason.** `W_15` is a lookup table over integer logits and is piecewise
constant relative to any continuous relaxation. `J_15` is consequently flat
between integer/LUT transitions and undefined under ordinary derivative rules
at transitions. When the target weight is zero, the declared floor is also
constant over an arbitrarily large region. The vector remains a coarse tangent
surrogate; it is not an exact derivative.

### 2.3 Wide logit-anchored integer NLL

The scoring objective should retain the existing fractional LUT while removing
its shallow exponent cutoff and target floor.

Let `B=47`. For `delta<=0`, write

```text
q = -delta,
n = floor(q / 256),
f = q mod 256.
```

Let `L[f]` be the existing Q15 fractional exponential table. Define a wide
weight

```text
W_B(delta) = floor((L[f] * 2^(B-15)) / 2^n),
```

with zero only when the shift annihilates the wide representation. Let

```text
Z_B = sum_i W_B(ell_i-m),
W_0 = W_B(0).
```

Define the **wide logit-anchored NLL**

```text
J_B(ell,y)
  = log2(Z_B) - log2(W_0) - (ell_y-m)/256.
```

The target term uses the actual Q8 logit gap, not `log2(W_B(target))`. It
therefore remains informative even when the target exponent weight is too small
to represent. All terms can be accumulated and scored in integer Q20.

**Proposition 2 (p10m range).** For `V=8192` and `B=47`,

```text
0 < Z_B < V 2^B <= 2^60,
```

so `Z_B` fits in `u64`. A fixed proposal mass `K=2^23` multiplied by a
cumulative weight requires fewer than 84 bits and therefore fits in `u128`.

**Code-bound approximation audit.** Relative to `L[f]/32767`, the generated
256-entry fractional table has maximum log2 error approximately
`8.3111e-5` bits (83.11 microbits). Ignoring terms below a 47-bit relative
cutoff contributes at most

```text
log2(1 + (8191) 2^-47) < 8.4e-11 bits
```

to the denominator in the worst case. Fractional-LUT error, not the wide cutoff,
dominates this objective's approximation error. Q20 log2 evaluation adds at
most its separately tested integer approximation error.

`J_B` is still a declared integer approximation, not the exact real objective.
Its value is that it is shift-invariant, reciprocal-independent, high
resolution, and does not collapse all deeply wrong targets onto one constant
floor.

## 3. Fixed-mass output proposals

Let nonnegative integer weights be `w_i`, with

```text
Z = sum_i w_i > 0,
p_i = w_i / Z.
```

The desired normalized coarse direction is

```text
h = p - e_y.
```

### 3.1 Minimal mass repair

If the existing normalized integer probabilities are `q_i` with actual mass
`M=sum_i q_i`, define

```text
g_i = q_i        for i != y,
g_y = q_y - M.
```

Then `sum_i g_i=0` exactly and

```text
g = M(q/M - e_y).
```

This is the cheapest production lane. Its remaining per-example scale variation
is the probability-mass error. With Q47 Newton normalization that variation is
measured in tens of parts per million rather than the large variation of raw
`Z`.

### 3.2 Why raw exponential weights do not preserve a batch

For example `n`, raw weights give

```text
c_n = w_n - Z_n e_(y_n) = Z_n h_n.
```

For a batch of `N` examples,

```text
C_raw  = sum_n Z_n h_n,
C_norm = sum_n h_n.
```

**Proposition 3 (batch distortion).** `C_raw = alpha C_norm` only when

```text
sum_n (Z_n-alpha) h_n = 0.
```

Constant `Z_n` is sufficient but not necessary. Without this condition there is
no single learning-rate adjustment that makes the two batch directions equal.

For any reference mass `Z_ref`, let `a_n=Z_n/Z_ref`. Then

```text
||C_raw/Z_ref - C_norm||
  <= max_n |a_n-1| sum_n ||h_n||.
```

No useful relative bound follows when `C_norm` is small through cancellation.
Even bounded scale variation can rotate a batch direction arbitrarily near such
cancellation.

**Artifact observation.** On the 14 frozen signal-attribution windows, source
normalization sums range from `136,849,272` to `219,855,578`, a ratio of
`1.6066`. They all occupy the same power-of-two bin, so exponent-only block
normalization would retain that entire variation.

**Decision.** Raw exponential-weight gradients are a diagnostic control, not
the primary optimizer lane.

### 3.3 Deterministic cumulative apportionment

Choose a fixed integer mass `K`. In a declared vocabulary order define

```text
C_i = sum_(j=0)^i w_j,
C_-1 = 0,
a_i = floor(K C_i / Z) - floor(K C_(i-1) / Z).
```

Then define

```text
g = a - K e_y.
```

This construction uses integer division but no approximate reciprocal.

**Proposition 4 (exact mass and error bounds).** Cumulative apportionment has

```text
a_i >= 0,
sum_i a_i = K,
sum_i g_i = 0,
|a_i - K p_i| < 1,
||a-Kp||_infinity < 1,
||a-Kp||_1 < V.
```

**Proof.** Monotonicity of `C_i` gives `a_i>=0`. The sum telescopes to
`floor(KZ/Z)=K`. Let

```text
d_i = floor(K C_i/Z) - K C_i/Z,
```

so `-1<d_i<=0`, with `d_-1=d_(V-1)=0`. Coordinate error is
`a_i-Kp_i=d_i-d_(i-1)`, whose absolute value is less than one. The remaining
claims follow by summation. `QED`.

The error is order-dependent even though its bound is not. A fixed vocabulary
order may introduce stable token-ID bias. A contract-bound rotation or
permutation can distribute this bias across steps while retaining replay; the
permutation must itself be part of `H_t`.

### 3.4 Seeded categorical counts

Choose fixed mass `K`. Draw `K` indices independently from the categorical
distribution `p`, and let `a_i` be the number of draws equal to `i`. Define

```text
g = a - K e_y.
```

An exact draw does not require a reciprocal or division:

1. form cumulative integer weights `C_i` and total `Z`;
2. generate a uniform integer `U` in `[0,Z)` by rejection from
   `b=ceil(log2 Z)` random bits; and
3. select the first index with `C_i>U`.

Expected rejection attempts are less than two. A counter-based generator keyed
by contract hash, seed, step, example, draw, and operation gives exact replay.

**Proposition 5 (unbiased fixed-scale proposal).** If the draws are independent,

```text
E[g] = K(p-e_y),
Cov(g) = K(Diag(p) - p p^T),
E[||g-K(p-e_y)||_2^2] = K(1-||p||_2^2) <= K.
```

Therefore the normalized estimator `g/K` has mean-square error at most `1/K`.
For a batch, linearity gives

```text
E[sum_n g_n] = K sum_n (p_n-e_(y_n)),
```

so example weighting is preserved in expectation.

This construction introduces seed variance but can make the output gradient
sparse. It tests stochastic-rounding theory without surrendering replay for a
fixed contract.

### 3.5 Candidate comparison

| Lane | Exact zero sum | Per-example mass | Bias/variance | Normalization cost |
| --- | --- | --- | --- | --- |
| actual-mass correction | yes | approximately fixed | normalized-probability quantization bias | existing Q47 path |
| cumulative apportionment | yes | exactly `K` | deterministic, `<1` coordinate error, order bias | integer cumulative divisions |
| seeded categorical counts | yes | exactly `K` draws | unbiased, variance `<=1/K` after normalization | comparisons and rejection sampling |
| raw exponential weights | yes | `Z_n` | deterministic example reweighting | none |

The first three are legitimate bounded experiments. The fourth is useful only
to measure how harmful `Z`-reweighting is.

## 4. Honest finite-difference alignment

### 4.1 Two questions require two surfaces

Let `P` be the proposal surface and `A` a genuinely separated acceptance
surface. For sampled stored coordinate `j`, define

```text
Delta_(S,j)(d) = L_S(theta + d e_j) - L_S(theta),
d in {-1,+1}, S in {P,A}.
```

Let the coarse oracle predict

```text
dhat_j = -sign(G_(P,j)).
```

The audit must report separately:

1. **oracle fidelity** on `P`: whether `dhat_j` selects the better one-cell
   neighbor on the same data used to form the proposal;
2. **proposal descent** on `P`: whether that neighbor improves over the current
   state;
3. **transfer alignment** on `A`: whether the same sign selects the better
   neighbor on unseen data; and
4. **conditional transfer**: whether a proposal-surface descent also descends on
   `A`.

A failure of (1) diagnoses the coarse backward or its scaling. A pass on (1)
but failure on (4) diagnoses sampling noise, overfit, or distribution shift.
Collapsing them into one acceptance-surface comparison cannot identify the
cause.

### 4.2 Surface separation

**Code observation.** The current audit constructs consecutive sliding windows
and splits their indices into proposal and acceptance sets. At context length
64, adjacent windows share 63 input tokens; they are index-distinct but not
token- or document-disjoint.

**Decision.** A promotion audit must use one of:

- disjoint document IDs;
- disjoint frozen stream shards; or
- a declared gap of at least `context_tokens+1` within a document.

The trace must record document identity, start offset, target offset, and hashes
for both surfaces. “Disjoint” without naming the disjoint unit is prohibited.

### 4.3 Sampling and inference

Coordinates must be stratified by `(layer, tensor role)` rather than only tensor
role. Report rescue-free, rescue-exposed, saturated, and zero-proposal strata
separately.

The deterministic random sign is a paired control on the same coordinate and
surfaces. Because many coordinates share the same windows, coordinate outcomes
are not independent observations of generalization. Confidence should therefore
be computed by resampling or blocking over documents/surfaces, not by treating
millions of coordinates as independent Bernoulli trials.

Predeclare:

- surface hashes and separation rule;
- coordinate strata and sample counts;
- proposal construction and mass `K`;
- tie policy;
- paired test and confidence level; and
- the minimum effect over random control required for promotion.

## 5. Fiber exits and first-passage words

### 5.1 Prime escapes corrected

Let `A*` be admissible action words from full state `X`. Define the descent
language

```text
D_X = { w in A* : L_E(T_w X) < L_E(X) }.
```

**Definition.** A **first-passage escape word** is an element of `D_X` with no
proper prefix in `D_X`. “Prime escape” may be used as a mnemonic only after the
action alphabet, starting state, surface, and prefix order are declared.

The earlier factorization definition is discarded. A suffix acts from a
different intermediate state, so ordinary integer-style factorization is not
well-defined without using a category of state-indexed morphisms.

### 5.2 What event-driven search can skip

Define first functional exit time

```text
tau_pi = min { t>0 : pi(X_t) != pi(X_0) }.
```

Because `L_E` is constant inside a fiber, exact acceptance evaluation can be
deferred until `tau_pi`.

**Proposition 6.** Fiber silence does not generally permit skipping forward or
backward gradient computation.

**Reason.** Residual increments depend on the data cursor, current activations,
and coarse backward oracle. Even when the deployed function is unchanged,
different batches produce different increments. Boundary time can be advanced
analytically only when the future increment sequence is already known or is
constant under a declared control. Computing that sequence may require the same
forward/backward work as ordinary training.

Event-driven optimization therefore promises fewer acceptance evaluations and
better candidate selection—not free jumps over unknown gradients.

## 6. Attention: the positivity-selectivity tradeoff

### 6.1 Affine positivity forces a large constant

For the affine elementwise feature map

```text
phi_c(z) = z + c,
```

strict positivity over every `i16` value requires

```text
c >= 32769.
```

This is exactly the production offset. It guarantees global positivity, but it
also introduces a constant kernel component of order `d c^2`.

### 6.2 Selectivity bound

Suppose realized query and key coordinates satisfy

```text
|q_j| <= B, |k_j| <= B, B<c.
```

For head dimension `d`, every kernel weight obeys

```text
d(c-B)^2 <= kappa(q,k) <= d(c+B)^2.
```

Hence the largest possible ratio between two token weights is bounded by

```text
R(B/c) = ((c+B)/(c-B))^2.
```

For `N` tokens, the maximum normalized share is at most

```text
R / (R + N - 1).
```

When typical `B/c` is small, the kernel is forced near uniform regardless of
optimizer quality. Conversely, reducing `c` while retaining affine positivity
is impossible over the complete `i16` domain.

**Decision.** The architectural comparison should test nonlinear positive maps
such as `max(z,0)+1`, signed-split features, or an explicitly clipped/calibrated
domain. The offset should be chosen from a declared selectivity and overflow
budget, not from positivity alone.

Fixed decay changes recency weighting but does not solve content selectivity;
feature-map calibration and decay remain separate experimental factors.

## 7. Revised optimization program

Call the revised method **Fibered Objective-Calibrated Search (FOCS)**. The name
denotes a protocol, not a convergence theorem.

1. Score with the wide logit-anchored integer objective and retain the current
   Q20 objective as a frozen comparison.
2. Construct a fixed-mass output proposal using actual-mass correction,
   cumulative apportionment, or seeded categorical counts.
3. Measure same-surface oracle fidelity before testing held-out transfer.
4. Accumulate error-feedback residuals in the full Markov state.
5. Track deployed-function hashes and run exact acceptance only on functional
   exits.
6. Record saturation debt, layer/role strata, proposal mass, and seed state.
7. Promote an optimizer only if it beats the paired random control on genuinely
   separated surfaces.
8. Test attention selectivity/order capability before scaling the same
   architecture.

## 8. Conjectures and falsifiers

### C04-A: common-mode mass error matters

- State: `open`
- Claim: replacing nominal target subtraction with actual-mass subtraction
  improves same-surface alignment without introducing saturation.
- Falsifier: alignment and proper-loss descent are unchanged or worse across
  replayed, predeclared samples.

### C04-B: fixed mass beats raw weight mass

- State: `open`
- Claim: cumulative apportionment or seeded counts outperform raw `Z`-scaled
  weights because they preserve example weighting.
- Falsifier: raw weights match or exceed both fixed-mass lanes on proposal
  fidelity and genuinely separated transfer after scale and saturation are
  controlled.

### C04-C: the target floor hides useful ordering

- State: `open`
- Claim: wide logit-anchored NLL resolves stored-cell comparisons that tie under
  the current zero-target floor and predicts later proper-loss improvement.
- Falsifier: the additional ordering is unstable across surfaces or unrelated
  to ideal/float reference NLL on an audit subset.

### C04-D: affine positivity suppresses first-layer selection

- State: `open`
- Claim: a positive nonlinear feature map improves order/retrieval capability
  by escaping constant-component domination.
- Falsifier: exact integer probes find selective current heads, or alternate
  maps increase kernel variation without capability or NLL gain.

## 9. Decisions

1. The minimal actual-mass correction remains the cheapest first lane.
2. Cumulative apportionment is the canonical deterministic fixed-mass lane.
3. Seeded categorical counts are the canonical unbiased reciprocal-free lane.
4. Raw exponential weights are demoted to a reweighting ablation.
5. The alignment audit now separates same-surface fidelity from document-
   separated transfer, binds the numerical backward schedule, and can sample
   source-specific rescue strata. The v2 result rejects the current trunk
   proposal lane; it does not yet isolate rescue as the cause.
6. The wide logit-anchored objective should be implemented and compared before
   any large prime-escape or scaling run.
7. Paid p20m/p30m execution remains unauthorized.

### Predeclared p10m calibration v1

The v1 contract fixed model `0xb10996e0707ab342`, dev stream
`0xda195778ceb603ab`, seed 29, two proposal windows from document 0, two
transfer windows from document 1, and one shared hash-selected coordinate per
active tensor group. A second execution reproduced the trace byte for byte.

Only `final_rms`, `output`, and `bias` produced sampled coordinates. For the
primary actual-mass-corrected lane:

- output-gradient mass was exactly zero on both proposal examples, compared
  with legacy normalized sums from `-84` to `5`;
- both direction-comparable proposal coordinates agreed with the better exact
  Q20 neighbor and descended, versus zero for paired random signs;
- transfer agreement and exact descent were `1/2`, exactly tying the paired
  random signs;
- gradient and residual saturation were both zero; and
- none of the three sampled coordinates was rescue-exposed for the primary
  lane, so the rescue-specific proposal and transfer gates had no observations.

The raw reciprocal-free output L1 mass varied from `438,238,576` to
`444,622,028` even on this two-example surface and remains explicitly labeled a
sample-reweighted control. Late RHU and seeded stochastic rounding produced the
same sampled signs; the stochastic lane recorded 283 upward rounding events.

**Decision.** This run validates strict surface separation, exact mass
accounting, common coordinate sampling, seeded replay, and canonical lattice
evaluation. It does not validate the rescue operator, generalization, or an
optimizer. The next alignment run must deliberately stratify rescue-exposed
trunk coordinates and block inference over more documents. Optimizer refinement
and paid scaling remain unauthorized.

### Rescue-stratified p10m calibration v2

The v2 contract retained model `0xb10996e0707ab342` and dev stream
`0xda195778ceb603ab`, changed the seed to 43, and predeclared four proposal
documents and four different transfer documents. Each surface used eight
windows balanced two per document. Sampling selected one coordinate per active
tensor group from the union of public-lane activity and one coordinate per
active group from each source-specific rescued-versus-natural stratum. The six
resulting shared coordinates comprised four `final_rms`, one `output`, and one
`bias` coordinate; four were rescue-exposed for the primary lane. A second
execution reproduced the v5 trace byte for byte. The legacy v1 runner also
continued to reproduce its frozen v4 trace byte for byte.

For the primary actual-mass-corrected lane:

- output-gradient mass was exactly zero on all examples, output L1 mass stayed
  within `16,774,752..16,775,588`, and neither gradients nor residuals
  saturated;
- rescue-exposed trunk proposal agreement was `1/3` comparable coordinates,
  versus `3/3` for paired random signs;
- the one trunk coordinate with an exact improving proposal neighbor was
  missed by the proposed direction and selected by the paired random sign;
- the transfer surface contained no exact improving neighbor among the four
  rescue-exposed trunk coordinates, so it cannot independently establish
  descent quality there; and
- both output-head coordinates agreed with and selected exact descent on both
  surfaces, versus one of two for their paired random signs.

All five public proposal lanes selected the same sampled signs. This includes
the rescued normalized lanes and the two late-quantization reciprocal-free
lanes, so this audit does not identify which upstream operation created the
wrong trunk directions. In particular, rescue exposure is a selection label,
not a causal intervention: the audit did not report a normalized,
mass-corrected lane with nonzero rescue disabled.

**Decision.** Both promotion gates fail. The failure on the proposal surface
rules out explaining the trunk result only as held-out sample variance, while
the aligned output head localizes the observed defect to the trunk sample. It
does not prove that nonzero rescue is the cause. The next causal control is the
same normalized mass-corrected source with natural zeros preserved. Only if
that control separates should rescue be replaced or calibrated; if it does not,
the upstream surrogate or backward chain remains the target. Optimizer
refinement and paid scaling remain unauthorized.

**Subsequent bounded result.** MJ-05/v3 performed the requested source-matched
no-rescue intervention on the same coordinates. It changed all four exposed
aggregate magnitudes by one count, changed no signs, and left agreement and
descent unchanged. The diagnosis is therefore no longer “rescue may explain the
wrong signs”; the defect survives before or independently of that projection.

## Open work

- **Completed by MJ-05/v3:** evaluate a normalized, mass-corrected no-rescue
  reference on the same rescue-stratified coordinates.
- Prove monotonicity or characterize rare reversals of `J_B` under one-Q8-logit
  target changes with the finite fractional LUT.
- Compare cumulative apportionment with largest-remainder apportionment and
  quantify token-order bias.
- Choose `K` by a variance, accumulator, sparsity, and residual-resolution
  budget rather than copying the current Q23 scale automatically.
- Define document-blocked confidence intervals for the dual-surface audit.
- Extend the artifact-bound attention selectivity probe to exact integer Q/K
  projections; v2 now validates the model checksum, tokenizer binding, stream
  hash, and vocabulary range, but its RMS reconstruction remains approximate.
