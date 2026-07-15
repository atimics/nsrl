# MJ-2026-07-14-03: Fibered optimization and reciprocal-free descent

- Date: 2026-07-14
- Status: partially superseded by
  [MJ-2026-07-15-04](MJ-2026-07-15-04-three-geometry-optimization.md); mass
  conservation, saturation debt, and the fiber correction remain active
- Supersedes: the Markov interpretation of the functional quotient in
  [MJ-2026-07-14-02](MJ-2026-07-14-02-discrete-optimization.md), but not its
  residual or exact-acceptance derivations
- Code binding:
  [`training.rs`](../../crates/nsrl-train/src/production/training.rs),
  [`production.rs`](../../crates/nsrl-train/src/production.rs), and
  [`attention.rs`](../../crates/nsrl-core/src/attention.rs) as inspected on
  2026-07-14
- Artifact binding:
  [`p10m-probability-normalization-accuracy.json`](../../benchmarks/production-model-v1/p10m-probability-normalization-accuracy.json)
  and
  [`p10m-normalized-wide-gradient-preflight.json`](../../benchmarks/production-model-v1/p10m-normalized-wide-gradient-preflight.json)

## Question

What is the most plausible mathematical reason that NSRL can produce many
integer parameter updates, exact deterministic replay, and forward-visible
changes without obtaining held-out descent? Which optimization changes follow
from that reason before another large run is justified?

## Executive result

The system is not primarily blocked by a lack of reachable parameter cells.
Three earlier assumptions are too coarse:

1. A quotient containing only realized functions is not a Markov state space,
   because optimizer residuals can change future exits while the function is
   unchanged.
2. Backpropagation can enforce zero mass without changing minibatch weighting
   by correcting the target coordinate of the existing normalized probability
   vector. A reciprocal-free direction can also be built from raw exponential
   weights, but its example-dependent total mass makes it an explicitly
   sample-reweighted surrogate when windows are accumulated.
3. Error feedback repairs the projection of a proposed direction; it does not
   establish that the upstream integer/STE direction is a descent direction.
   NSRL's nonzero-rescue operators can have unbounded relative distortion near
   zero.

The revised object is a **fibered controlled state graph**. Residual motion
inside a fiber is hidden preparation; parameter or rounding events exit the
fiber. Optimization should first repair and calibrate the proposal direction,
then search selectively over predicted exits, and only then spend compute on
larger trajectories.

## 1. The controlled state is larger than the realized function

### 1.1 Full state and observation

**Definition.** Let the complete deterministic training state be

```text
X_t = (theta_t, r_t, s_t, c_t, m_t),
```

where:

- `theta_t` is the stored integer parameter vector;
- `r_t` is the optimizer residual vector;
- `s_t` is the set of numeric scales and shift-controller state;
- `c_t` is the data cursor and schedule state; and
- `m_t` contains any persistent recurrent or optimizer memory.

For an admissible action `a` (batch choice, shift choice, candidate update, or
architecture lane), the transition is

```text
X_{t+1} = T_a(X_t).
```

Let `pi(X)` identify the exact deployed function, including every serialized
forward scale that changes that function:

```text
pi(X) = [F_C(theta, s_forward, .)].
```

Equivalently, all deployed forward scales may be included in `theta`. Backward-
only controller state remains in `s_t` without changing the observation until
it changes a deployed forward scale.

The **functional fiber** over a realized function `v` is

```text
Fib(v) = { X : pi(X) = v }.
```

Residual accumulation, cursor movement, and some parameter movements can stay
inside the same fiber. A transition is **silent** when
`pi(T_a(X)) = pi(X)` and is an **exit** otherwise.

### 1.2 Why the plain quotient is not Markov

**Proposition 1.** In general there is no transition map on functional classes
alone that reproduces the optimizer dynamics.

**Proof sketch.** Consider one parameter with update quantum `q = 2^s`, a fixed
incoming integer gradient `g`, and error-feedback residual update

```text
u = R_s(r + g),
r' = r + g - q u.
```

Two states can have the same `theta`, hence the same realized function, but
residuals `r_1 = 0` and `r_2` close enough to a rounding boundary that
`R_s(r_1 + g) = 0` while `R_s(r_2 + g) = 1`. Their next parameter states differ.
Thus the next functional class is not determined by the current functional
class and action. The projection `pi` is an observation, not a sufficient
state. `QED`.

This corrects the strongest interpretation in MJ-02. The functional quotient
remains useful for determining whether an endpoint matters, but optimization
must retain the hidden state in each fiber.

### 1.3 Local minima and prime escapes

**Definition.** For a declared evaluation surface `E`, action alphabet `A`, and
full state `X`, define escape depth

```text
h*(X; E, A) = min { |w| : L_E(pi(T_w(X))) < L_E(pi(X)) },
```

where `w` is a finite action word over `A`. If no such word exists, the value is
infinite.

**Definition.** An improving word `w` is a **prime escape** at `X` when no proper
prefix of `w` improves on the starting loss. It is strongly prime when it also
cannot be factored into shorter accepted descent words under the same action
alphabet.

This gives a precise form to “primes as local minima.” A prime is not a special
parameter value. It is an irreducible improving action sequence relative to a
chosen generator set. Silent residual steps are the factors hidden inside the
fiber; the final boundary crossing is the observable escape.

**Local-minimum taxonomy.** A failure to improve can now mean different things:

- a **quantization plateau**: an improving fiber exit exists but has not yet
  been reached;
- a **proposal minimum**: the backward oracle proposes no aligned exit even
  though one exists;
- a **barrier minimum**: improvement requires a temporarily worse endpoint;
- an **architectural minimum**: the deployed function class cannot express the
  needed distinction; or
- a **true `H`-local minimum**: no word of length at most `H` in the declared
  action alphabet improves the declared objective.

Calling all five cases “the optimizer is stuck” discards the information needed
to select the next experiment.

## 2. Reciprocal-free, mass-conserving output descent

### 2.1 Current defect

**Code observation.** Production backward copies quantized probabilities and
subtracts the nominal scale only at the target:

```text
g_i = p_i                         for i != y
g_y = p_y - (2^F - 1).
```

Therefore

```text
sum_i g_i = sum_i p_i - (2^F - 1).
```

The probability normalization artifact measured maximum mass errors near
98,929 ppm for the legacy Q31 reciprocal path. Q47 with one Newton step reduced
the maximum to 98/83 ppm for the two compared models, close to exact division at
73/74 ppm.

**Artifact observation.** This numeric repair did not establish descent. In the
normalized-wide preflight, the control and the `up` boundary lane tied on
held-out loss despite 155 `up` updates and 84 feature-changed windows. The
`output` boundary lane changed the target logit in 64 windows and the target Q15
probability in 3, but worsened total held-out loss by 415 millibits.

The artifacts therefore separate two questions:

1. Is the reported probability distribution normalized accurately?
2. Is the backward direction aligned with the proper training objective?

Improving the first does not answer the second.

### 2.2 Direct construction

Let integer logits be `ell_i`. Let `m = max_i ell_i`, and let the existing
base-2 exponential approximation produce positive integers

```text
w_i = Exp2Neg(ell_i - m),
Z   = sum_i w_i.
```

The realized normalized distribution is `p_i = w_i / Z`. For base-2 NLL and
exact base-2 exponentiation, the continuous real-logit gradient is `p - e_y`.
The implemented LUT is piecewise constant with an annihilation boundary, so no
nonzero vector is literally the derivative of the deployed integer objective.

Choose a block shift `s` so that the retained total mass occupies a desired
number of gradient bits, and compute

```text
v_i = R_s(w_i).
```

Define the integer gradient by

```text
g_i = v_i                         for i != y,
g_y = -sum_{i != y} v_i.
```

The target assignment should use a wide accumulator and be range-checked before
storage.

For the current vocabulary `V = 8192`, the existing exponential LUT gives
`0 <= w_i <= 32767`. Hence

```text
32767 <= Z <= 8192 * 32767 = 268427264 < 2^28,
```

and the largest possible target magnitude is

```text
(V - 1) * 32767 = 268394497 < 2^28.
```

The unshifted gradient therefore fits in signed `i32`. Moreover,
`sum_i |g_i| < 2^29`; multiplying by an `i16` feature or output weight is
bounded below `2^44`, leaving ample `i64` accumulator headroom. A shift can
still be used to match current learning-rate scales, but it is not required for
representability at p10m. This proof assumes no additional vocabulary masking
semantics beyond zero weights.

**Proposition 2 (mass conservation).** `sum_i g_i = 0` exactly, independent of
rounding and reciprocal accuracy.

This is immediate from the definition.

**Proposition 3 (per-example surrogate directional equivalence).** If
`v_i = 2^{-s} w_i`, then

```text
g = 2^{-s} (w - Z e_y)
  = 2^{-s} Z (p - e_y).
```

Thus `g` is a positive scalar multiple of the continuous softmax
cross-entropy surrogate direction for one example. It needs no reciprocal.
This is an algebraic statement about the distribution induced by `w`, not a
derivative theorem for the piecewise-constant LUT objective. Integer finite
differences remain the authority for descent.

The factor `Z` is also example-dependent. For a minibatch,

```text
sum_n g_n = sum_n Z_n (p_n - e_{y_n}),
```

which is generally not a scalar multiple of the normalized batch gradient.
The inspected p10m surface varies materially in `Z`; therefore the raw-weight
lane is classified as a sample-reweighted surrogate. A fixed learning-rate
change cannot remove this weighting. The primary experimental lane is instead
the minimal mass correction `g_y = -sum_{i != y} p_i`, which preserves every
non-target normalized component and changes only the common-mode defect.

**Proposition 4 (rounding error).** Let
`e_i = v_i - 2^{-s} w_i` for non-target coordinates and set `e_y = 0`. Then

```text
g - 2^{-s}(w - Z e_y)
  = e - (sum_{i != y} e_i) e_y,
```

and

```text
||error||_1 <= 2 sum_{i != y} |e_i|.
```

For round-to-nearest with no saturation, `|e_i| <= 1/2`, so the absolute L1
error is at most `V - 1` integer units. The total retained mass can be chosen
large enough that this error is negligible while remaining inside an `i32` or
`i64` contract.

### 2.3 Minimal and stronger lanes

The minimal repair can retain the existing probabilities: compute their actual
mass `M = sum_i p_i`, preserve non-target entries, and set

```text
g_y = p_y - M = -sum_{i != y} p_i.
```

This exactly removes common-mode gradient injection. The stronger lane builds
the gradient from `w` and removes reciprocal normalization from backward
entirely. The two lanes distinguish mass conservation from reciprocal removal.

**Decision.** Q47 Newton remains useful for calibrated reporting and integer
NLL evaluation. It should no longer be a dependency of the backward direction.

## 3. The upstream coarse-gradient oracle is not yet trusted

### 3.1 Nonzero rescue

**Code observation.** Several backward paths use an operator equivalent to

```text
Q^dagger_s(z) = R_s(z)                    if R_s(z) != 0 or z = 0,
                sign(z)                   otherwise.
```

`quantized_nonzero`, `round_ratio`, and gated backward code use this kind of
straight-through rescue. It ensures liveness, but it changes the mathematical
object being optimized.

**Proposition 5.** The relative distortion of `Q^dagger_s` is unbounded near
zero.

For nonzero `z` below the ordinary rounding threshold,
`Q^dagger_s(z) = sign(z)`, while the scaled exact value is `z / 2^s`. Hence

```text
|Q^dagger_s(z) - z/2^s| / |z/2^s|
```

diverges as `|z| / 2^s` approaches zero. Applying the rescue independently to
many microterms also makes the bias dimension-dependent and destroys linearity:
in general, summing rescued terms differs from rescuing their wide sum.

This does not prove that rescue is harmful in the current model. It proves that
nonzero propagation and directional fidelity are different properties.

### 3.2 What error feedback can and cannot repair

Error feedback gives an identity for the quantization of a proposed update. It
cannot convert a systematically misaligned upstream proposal into descent.
This is the boundary between two problems:

```text
backward oracle:       data -> proposed wide direction g_t
projection optimizer: (g_t, r_t) -> stored parameter transition
```

Residual SGD addresses the second arrow. The evidence so far does not validate
the first.

### 3.3 Calibration protocol

Before more shift search, sample coordinates or small blocks and run two
separate exact integer finite-difference tests:

```text
Delta_j^proposal = L_P(theta +/- q_j e_j) - L_P(theta),
Delta_j^transfer = L_H(theta +/- q_j e_j) - L_H(theta).
```

`P` is the same proposal surface used to form the gradient and measures
implementation/directional fidelity. `H` is separated by document when
possible, otherwise by a non-overlap gap of at least one full context plus its
target, and measures transfer. A proposal-surface failure implicates the
surrogate or implementation. A transfer failure alone can be sample variance
or lack of generalization and is not proof of a bad backward oracle.

Record:

- sign agreement with the best one-cell direction;
- predicted versus realized loss ordering;
- results partitioned by tensor role and layer;
- the fraction of proposals created only by nonzero rescue; and
- ordinary RHU, late-quantized, and seeded stochastic-rounding controls.

The preferred deterministic stochastic control is counter-based: the sample is
a pure function of the training-contract hash, seed, global step, operation ID,
and coordinate. That preserves exact replay for a fixed seed while testing the
unbiased-rounding hypothesis.

**Decision.** Do not globally remove the rescue operator before this probe.
First add a wide-accumulate/quantize-once lane and measure directional alignment.

## 4. Attention appears almost content-uniform

### 4.1 Structural expansion

**Code observation.** The production feature map is

```text
phi(z) = z + c,  c = 32769,
```

applied elementwise. There is no explicit positional embedding in the inspected
production path and the available state decay is not enabled there.

For a head of dimension `d`, the attention kernel is

```text
kappa(q, k)
  = (q + c 1)^T (k + c 1)
  = d c^2 + c 1^T q + c 1^T k + q^T k.
```

When `q` and `k` are small compared with `c`, the constant term dominates and
the normalized state approaches a uniform average. Increasing optimizer motion
in `K` cannot fix this if the map suppresses the resulting contrast.

### 4.2 Read-only selectivity probe

The diagnostic
[`analyze-production-attention-selectivity-v1.mjs`](../../scripts/analyze-production-attention-selectivity-v1.mjs)
reconstructed the first-layer Q/K projections for 32 held-out contexts of
length 64 (256 head rows). It uses floating-point RMS reconstruction only as an
offline diagnostic and is not a training or quality result.

The v2 probe now validates the model checksum, requires the model and token
stream tokenizer hashes to match, rejects out-of-vocabulary tokens, and records
the model, tokenizer, and token-stream hashes. It still approximates RMSNorm and
decay rather than executing the exact integer path, so the measurements remain
reconnaissance until reproduced through exact runtime Q/K projections.

For the trained normalized-wide control model `0xd7c18cde6c8d678d` on dev
stream `0xda195778ceb603ab` (tokenizer `0xf4fe71d93c438c1a`), the current map
with no decay had:

| Statistic | Mean |
| --- | ---: |
| kernel-weight coefficient of variation | 0.00949 |
| effective attended tokens out of 64 | 63.9970 |
| maximum single-token share | 0.01604 |
| uniform single-token share | 0.015625 |
| oldest/newest weight ratio | 0.99834 |

The first-layer kernel is therefore nearly uniform on this probe.
The corresponding initialized p10m model was already in the same regime
(weight CV `0.00811`, effective tokens `63.9979`, and maximum share `0.01596`).
The trained K movement increased first-layer contrast only marginally in this
sample.

Offline counterfactuals, using the same projected Q/K values, produced:

| Feature map and decay | Weight CV | Effective tokens | Maximum share |
| --- | ---: | ---: | ---: |
| `z + 32769`, none | 0.0095 | 64.00 | 0.0160 |
| `z + 32769`, `63/64` | 0.2887 | 61.41 | 0.0247 |
| `z + 32769`, `31/32` | 0.5679 | 54.75 | 0.0360 |
| `max(z,0)+1`, none | 0.4018 | 59.10 | 0.0380 |
| signed split, none | 0.3016 | 61.39 | 0.0373 |
| `max(z,0)+1`, `63/64` | 0.5737 | 55.63 | 0.0581 |

Decay injects recency and order but not content retrieval. Alternate positive
feature maps increase content contrast in this diagnostic, but their stability,
integer overflow behavior, and held-out effect remain untested.

**Conjecture C03-A (open).** Lack of selective and order-sensitive attention is
a stronger quality bottleneck than failure to cross more optimizer boundaries.

**Falsifier.** An order/retrieval synthetic gate and a bounded language-model
lane show no gain from fixed decay or a calibrated positive feature map, while a
backward-calibrated optimizer lane improves the same frozen surfaces under the
current feature map.

## 5. Numeric-accounting corrections

### 5.1 Rounding is operator-specific

**Code observation.** Many projections use round-half-up, but output-logit
projection uses an arithmetic right shift. For negative nonmultiples this is a
floor-like operation rather than symmetric nearest rounding. The generic symbol
`R_s` in earlier entries must therefore be instantiated per operator.

**Decision.** Maintain a rounding ledger with at least:

| Site | Wide type | Projection | Saturation | Rescue |
| --- | --- | --- | --- | --- |
| output logits | `i64` | arithmetic shift | `i32` clamp | none |
| backward feature | `i64` | RHU | `i16` | nonzero |
| attention ratios | `i64/i64` | signed nearest | downstream | nonzero |
| parameter update | `i64` residual | RHU | parameter-width clamp | none |

A cheap preflight should compare arithmetic shift with RHU at the output head,
including sign-conditioned logit error and target NLL. This is an experiment,
not authorization to change the deployed contract silently.

### 5.2 Saturation creates debt

The usual residual identity assumes the requested lattice update is applied.
Let `u_t^req` be the requested stored update, `u_t^act` the clipped update, and

```text
d_t^sat = q_t (u_t^req - u_t^act).
```

Then the represented incoming mass decomposes as

```text
sum_t g_t
  = sum_t q_t u_t^act + r_T - r_0 + sum_t d_t^sat.
```

If the implementation consumes the requested residual quantum while clipping
the parameter transition, `d_t^sat` is silently discarded optimization mass.
Zero saturation remains a promotion gate, but future artifacts should record
the magnitude and tensor location of saturation debt, not only event counts.

### 5.3 Per-layer state is hidden by group aggregation

One scale shared across a tensor role and all layers can average together dead,
healthy, and saturated layers. Before introducing adaptive per-layer shifts,
record per `(layer, role)`:

- wide-gradient exponent/range;
- residual distance to the nearest update boundary;
- requested, applied, and function-visible update counts;
- saturation debt; and
- adjacent-cell sign flips or returns to the prior cell.

A bounded controller can then use

```text
s_(layer,role) = s_role + delta_layer,
```

with a small declared range for `delta_layer`. Variable shifts preserve the
error-feedback accounting when the residual and saturation terms are tracked in
represented units.

## 6. Revised optimizer: Fibered Residual Exit Search

The revised theory changes DRTO into a staged method.

### Phase I: repair the proposal oracle

1. Introduce the minimal mass-conserving normalized-probability lane as the
   primary comparison.
2. Introduce the reciprocal-free exponential-weight lane as an explicitly
   sample-reweighted surrogate comparison.
3. Promote the existing target-aligned integer NLL into candidate acceptance
   rather than using probability error, mistake count, or Q15 no-op counts.
4. Measure finite-difference alignment by layer and tensor role.
5. Compare microterm rescue with wide accumulation followed by one projection;
   include counter-based stochastic rounding as a control.

**Code observation.** `base2_softmax_nll_millibits` and
`evaluate_production_model_canonical_nll` already compute a
normalization-independent objective in the logit domain from the same weights:

```text
L_y = log2(Z) - log2(w_y),
```

using an integer Q20 `log2` approximation before rounding to millibits. This
avoids a probability reciprocal and retains changes hidden by Q15 target
probability. The cited normalized-wide artifact predates promotion of this
canonical measure into its acceptance gate, so the next experiment should
reuse the implementation rather than create another objective.

### Phase II: search fiber exits, not every silent step

For each candidate tensor block:

1. use residual distance and proposal coherence to predict the next parameter
   boundary event;
2. advance the hidden residual state analytically or by deterministic replay to
   that event;
3. reject candidates that saturate or violate the numeric contract;
4. run the forward function and acceptance surface only for actual exits; and
5. accept an exit only when the declared integer NLL decreases, with a separate
   rotating surface guarding against local overfit.

This is an event-driven search over fibers. It avoids spending a full evaluation
on every silent residual transition while retaining exact endpoint acceptance.

### Phase III: expand the architecture only after a capability gate

Run tiny exact tasks that require order and associative retrieval. If the
current map fails, compare:

1. fixed integer decay `63/64` and `31/32`;
2. a calibrated `phi_b(z) = max(z+b, 1)` rather than a fixed `b=32769`;
3. signed-split features if the state budget permits; and
4. only then delta-rule correction or learned gates.

The order matters because fixed decay and feature-map calibration are much
smaller changes than a new recurrent learning rule.

## 7. Ranked experiments

| Rank | Experiment | Cost | Information gained | Promotion condition |
| ---: | --- | --- | --- | --- |
| 1 | mass-conserving and reciprocal-free output-gradient lanes | low | isolates common-mode defect and reciprocal dependence | higher finite-difference sign agreement and lower rotating integer NLL, zero saturation |
| 2 | canonical logit-domain NLL acceptance | low | aligns search with the proper target objective | resolves changes hidden by Q15 and predicts exact-division NLL ordering |
| 3 | backward rescue/late-quantization calibration with seeded stochastic control | low-medium | tests whether the proposal oracle is biased | at least one lane improves sign/rank agreement without health regressions |
| 4 | order/retrieval gate with fixed decay and calibrated feature map | low-medium | distinguishes optimization from architectural incapacity | capability gain precedes any language-scale lane |
| 5 | per-layer instrumentation and bounded scale offsets | medium | finds layers hidden by role-level aggregation | improves useful-exit rate, not merely parameter movement |
| 6 | event-driven exact exit search | medium | tests prime escapes and barrier depth | accepted exits transfer to the rotating surface |
| 7 | larger p10m trajectory | high | quality confirmation only | authorized only after one bounded lane produces held-out proper-loss descent |

## 8. Literature constraints on the revision

The revision is consistent with, but narrower than, established work:

- [Deep Learning with Limited Numerical Precision](https://proceedings.mlr.press/v37/gupta15.html)
  and [Is Integer Arithmetic Enough for Deep Learning Training?](https://openreview.net/forum?id=G7MX_0J6JKX)
  make stochastic rounding a serious control, not an optional curiosity.
- [Error Feedback Fixes SignSGD](https://proceedings.mlr.press/v97/karimireddy19a.html)
  supports residual correction of compressed updates, but does not validate an
  arbitrary upstream coarse-gradient oracle.
- [Understanding Straight-Through Estimator](https://openreview.net/forum?id=Skh4jRcKQ)
  shows that useful coarse gradients require model- and estimator-specific
  alignment; liveness alone is not the theorem.
- [Overcoming Oscillations in Quantization-Aware Training](https://proceedings.mlr.press/v162/nagel22a.html)
  and [Hysteresis Quantization](https://openreview.net/forum?id=3HJOA-1hb0e)
  motivate measuring adjacent-cell churn before adding dampening.
- [Direct Quantized Training of Language Models with Stochastic Rounding](https://proceedings.mlr.press/v304/zhao26b.html),
  [ECO](https://arxiv.org/abs/2601.22101), and
  [Ternary Momentum](https://openreview.net/forum?id=A3mVmPlahU) narrow novelty
  claims around low-precision weights without master copies and make quantized
  optimizer-state comparisons mandatory.
- [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)
  and [Gated Linear Attention](https://proceedings.mlr.press/v235/yang24ab.html)
  support testing decay and correction when additive state behaves as an
  indiscriminate average.

## 9. Decisions

1. **Do not authorize a larger training run yet.** Forward-visible motion has
   already failed to imply held-out descent.
2. **Treat probability normalization and backward construction as separate
   contracts.** Repair reporting with Q47 if needed; test reciprocal-free
   backward independently.
3. **Make objective alignment the next gate.** The next artifact should bind a
   high-resolution integer NLL, finite-difference alignment, saturation debt,
   and per-layer visibility.
4. **Use the fibered graph as the optimization model.** Functional hashes label
   observations; residuals and controller state remain part of the state.
5. **Test architectural selectivity before blaming every plateau on the
   optimizer.** The current first-layer attention probe is nearly uniform.

## Open work

- Generalize the p10m accumulator proof to larger vocabularies and any future
  left-shifted gradient representation.
- Bind the exponential-weight and integer-log2 approximations into a single
  proper-loss error budget.
- Add layer-indexed diagnostics without changing training semantics.
- Define the action alphabet and maximum word length for the first prime-escape
  search.
- Freeze the attention-selectivity probe only after replacing its diagnostic
  RMS reconstruction with the exact integer forward path or explicitly keeping
  it as a non-gating reconnaissance artifact.
