# MJ-2026-07-14-01: Foundations of Deterministic Integer Training

- Date: 2026-07-14
- Status: active
- Supersedes: none
- Code binding: current `ProductionModelV1` forward, backward, and optimizer
- Artifact binding: checked-in p10m reachability, probability-resolution, and
  normalization checkpoints

## Question

What mathematical object is NSRL training, what does its update rule optimize,
and which measurements are necessary before an integer parameter update can be
called useful learning?

The immediate purpose is to replace undirected shift sweeps with predictions
derived from scale, residual, forward-boundary, and objective geometry.

## 1. Numeric representation

**Definition 1.1 (fixed-point interpretation).** A stored integer `z` with
fractional-bit exponent `f_z` represents

\[
\bar z = z 2^{-f_z}.
\]

The bar distinguishes a represented real quantity from its stored integer.

**Definition 1.2 (rounding).** For a non-negative right shift `s`, define

\[
R_s(z)=\left\lfloor\frac{z+2^{s-1}}{2^s}\right\rfloor
\]

for `s > 0`, and `R_0(z)=z`. This is the arithmetic implemented by
`round_shift_rhu_i64` in
[`numeric.rs`](../../crates/nsrl-core/src/numeric.rs). It rounds half-way cases
toward positive infinity. In particular,

\[
R_s(2^{s-1})=1,\qquad R_s(-2^{s-1})=0.
\]

The positive and negative update thresholds therefore differ by one stored
integer unit. Error feedback controls the long-run consequence when saturation
is absent, but the timing of an individual boundary crossing is asymmetric.

**Definition 1.3 (projection).** An integer linear projection with stored input
`x`, weight matrix `W`, output shift `s`, and signed output width `b` is

\[
y=\operatorname{Sat}_b(R_s(Wx)).
\]

Dimensional consistency requires

\[
f_y=f_x+f_W-s.
\]

This equation is the beginning of a scale ledger. A shift is meaningful only
relative to the exponents of its input, weight, and output.

**Code observation 1.1.** `ProductionProjectionScales` records projection
shifts, but the production artifact does not yet record `f_x`, `f_W`, and `f_y`
as a checked dimensional relation. Consequently, changing a forward shift
changes model gain as well as boundary visibility.

For example, changing the `up` shift from 10 to 7 multiplies its unbounded
projection output by eight:

\[
R_7(Wx)\approx 8R_{10}(Wx).
\]

This is not merely an allocation of additional observation precision.

**Resolved code observation 1.2 (executable scale and rounding ledger).** The
runtime now derives a machine-readable contract for every production forward
projection and backward linear edge. It checks

\[
f_{acc}=f_{in}+f_W,
\qquad
f_{out}=f_{acc}-s,
\]

records the output head's legacy arithmetic-right-shift rule separately from
RHU projections, exposes the seven-bit default output-backward damping, derives
the represented power-of-two update coefficient for all 13 parameter groups,
and rejects configurations whose worst-case projection, linear-attention,
RMS, or softmax accumulators exceed their declared integer width. The RHU
primitive now implements Definition 1.2 exactly at the i64 boundaries instead
of saturating the pre-shift bias addition. The ledger can be inspected with
`nsrl-production-model numeric-contract`.

## 2. Exact forward map

Let the persistent parameter tuple be

\[
\theta=(E,\Gamma_a,\Gamma_m,\Gamma_f,
         W_q,W_k,W_v,W_o,W_u,W_g,W_d,W_{out},b_{out}).
\]

The exact forward map is a composition of integer projections, integer
RMSNorm, saturating residual additions, a gated MLP, and causal linear
attention. It is deterministic for a fixed contract.

**Definition 2.1 (linear-attention feature map).** The production feature map
is

\[
\phi(z)=z+32769.
\]

For an i16 input, every component of `phi(z)` is a positive integer in
`[1, 65536]`.

For one head, define

\[
S_t=\sum_{j\le t}\phi(k_j)v_j^\top,
\qquad
z_t=\sum_{j\le t}\phi(k_j).
\]

The context is

\[
c_t=operatorname{Sat}_{16}
\left(
\operatorname{RoundDiv}
\left(
\phi(q_t)^\top S_t,
\phi(q_t)^\top z_t
\right)
\right).
\]

This is implemented in
[`attention.rs`](../../crates/nsrl-core/src/attention.rs). The corresponding
kernel is

\[
\kappa(q,k)=\phi(q)^\top\phi(k)
=q^\top k+c\mathbf 1^\top q+c\mathbf 1^\top k+dc^2,
\qquad c=32769.
\]

The constant term can dominate kernel variation. Whether the trained Q/K range
produces selective attention is an empirical question that should be measured
with attention entropy or effective attended-token count, not inferred from K
or V parameter movement.

**Definition 2.2 (output distribution).** For stored Q8 logits `ell_i`, the
ideal base-2 distribution is

\[
p_i=\frac{2^{\ell_i/256}}{\sum_j2^{\ell_j/256}}.
\]

The deployed distribution `p_hat` replaces exponentiation and reciprocal
division with integer lookup and normalization operators. The normalization
method is therefore part of the model and training contract.

## 3. Training is a deterministic state transition

**Definition 3.1 (complete training state).** Excluding administrative trace
counters, define

\[
X_t=(\theta_t,r_t,c_t),
\]

where `r_t` is the vector of i64 optimizer residuals and `c_t` contains the data
cursor and active scale/shift controls.

**Definition 3.2 (coarse output signal).** At probability precision `F`, NSRL
begins backward propagation with

\[
\tilde g_{\ell,i}=\hat p_i^{(F)}-(2^F-1)\mathbf 1[i=y].
\]

The remaining backward operator is denoted

\[
\tilde g_\theta=B_C(\theta,x,y,\tilde g_\ell).
\]

The tilde is essential: `B_C` includes integer rounding, saturation, and
straight-through rescue. It has not been shown to equal the derivative of a
scalar surrogate objective.

**Code observation 3.1 (nonzero rescue).** When a backward value is nonzero but
would round to zero, `quantized_nonzero` returns its sign. Thus its scalar
operator is

\[
Q_s^*(z)=
\begin{cases}
\operatorname{sign}(z),&R_s(z)=0\ \text{and}\ z\ne0,\\
R_s(z),&\text{otherwise}.
\end{cases}
\]

This operator prevents dead backward paths but has unbounded relative error as
`z` approaches zero. Its value must be established by descent-alignment tests,
not by gradient-liveness counts alone.

**Definition 3.3 (residual integer SGD).** For parameter coordinate `j`, batch
gradient-like sum `g_tj`, and update shift `s_j`, define

\[
a_{t,j}=r_{t,j}+g_{t,j},
\]

\[
u_{t,j}=R_{s_j}(a_{t,j}),
\]

\[
\theta_{t+1,j}=\operatorname{Sat}(\theta_{t,j}-u_{t,j}),
\]

\[
r_{t+1,j}=a_{t,j}-2^{s_j}u_{t,j}.
\]

This is the optimizer implemented in
[`training.rs`](../../crates/nsrl-train/src/production/training.rs).

**Lemma 3.1 (error-feedback identity).** If residual accumulation and parameter
application do not saturate, then after `T` steps

\[
r_{T,j}=r_{0,j}+\sum_{t<T}g_{t,j}-2^{s_j}\sum_{t<T}u_{t,j}.
\]

If `r_0=0`, the accumulated applied update is therefore

\[
\sum_{t<T}u_{t,j}
=\frac{\sum_{t<T}g_{t,j}-r_{T,j}}{2^{s_j}}.
\]

**Proof.** Substitute the residual recurrence and telescope over `t`. `□`

For a constant shift and an unsaturated rounding step,

\[
-2^{s_j-1}\le r_{t,j}\le2^{s_j-1}-1.
\]

This bounds the cumulative quantization discrepancy. If a parameter saturates,
the implementation consumes the requested residual update even when clipping
prevents the corresponding parameter movement. The identity then no longer
describes applied parameter change. Zero saturation is consequently a
mathematical precondition for interpreting residual carry as error feedback.

## 4. Objective alignment

**Definition 4.1 (canonical language-model objective).** For a frozen set `E`
of context/target pairs, define

\[
L_{NLL,E}(\theta)
=-\sum_{(x,y)\in E}\log_2 p_\theta(y\mid x).
\]

The integer implementation may approximate this in milli- or microbits, but the
normalization method and zero-probability floor must be declared.

**Code observation 4.1.** Production evaluation uses negative log2 target
probability. The frozen substrate proof instead promotes aggregate
`probability_error_q15`.

For Q15 scale `S=32767`, the per-sample probability error is

\[
L_{PE}(p,y)=(S-p_y)+\sum_{i\ne y}p_i.
\]

**Proposition 4.1 (probability error is not a proper distributional loss).** If
`sum_i p_i=S`, then

\[
L_{PE}(p,y)=2(S-p_y).
\]

For a true conditional distribution `q`, its expected value is

\[
\mathbb E_{y\sim q}L_{PE}(p,y)
=2S-2\sum_yq_yp_y.
\]

This is minimized by placing all available mass on an index attaining
`max_y q_y`, rather than by setting `p=q`. Therefore it measures confidence in
the modal class, not distributional fidelity. `□`

**Decision 4.1.** Negative log-likelihood is the canonical quality objective
for mathematical claims about language modeling. Probability error and mistake
count remain useful secondary diagnostics, but a future proof contract should
not treat probability error as a calibrated probabilistic score.

## 5. Probability-mass invariant

For exact softmax cross-entropy,

\[
\sum_i(p_i-\mathbf1[i=y])=0.
\]

For the stored NSRL output signal,

\[
\sum_i\tilde g_{\ell,i}
=\sum_i\hat p_i^{(F)}-(2^F-1).
\]

**Proposition 5.1.** Probability-mass error is exactly a common-mode component
of the initial logit-gradient signal.

**Proof.** Sum Definition 3.2 over all classes. `□`

Adding a common constant to every logit leaves softmax unchanged. A nonzero
gradient sum therefore spends optimizer state in a forward-null direction and
breaks the shift-invariance expected from cross-entropy.

**Artifact observation 5.1.** In the frozen Q23 normalization audit, worst-case
probability-mass error is approximately 98,925 ppm for the legacy Q31
reciprocal, 98 ppm after one Q47 Newton step, and 73 ppm under rounded exact
division. The Newton method reduces this common-mode error by roughly three
orders of magnitude without runtime division.

**Decision 5.1.** `q47_newton1` is the justified runtime candidate for a bounded
normalized-wide-gradient test. The test must report the gradient-sum residual,
not only probability-vector differences.

## 6. Resolution near a uniform vocabulary

Let the vocabulary size be `V` and the probability format have `F` fractional
bits.

**Proposition 6.1 (uniform-regime stored resolution).** The stored probability
count near uniform is approximately

\[
n_{uniform}=\frac{2^F}{V}.
\]

The available relative fractional resolution near uniform is therefore

\[
b_{relative}\approx F-\log_2V.
\]

For production `V=8192=2^13`:

| Format | Uniform count | Approximate relative bits |
| --- | ---: | ---: |
| Q15 | 4 | 2 |
| Q19 | 64 | 6 |
| Q23 | 1024 | 10 |

Q15 target-probability collapse was predictable from vocabulary size before an
empirical sweep. Q23 is the first tested format with roughly ten stored bits of
relative resolution around uniform.

## 7. The boundary ladder

Let `E` be a declared evaluation surface.

**Definition 7.1 (parameter-visible update).** An optimizer action is
parameter-visible when `theta' != theta`.

**Definition 7.2 (function-visible update).** It is function-visible on `E`
when

\[
\exists x\in E:F_C(\theta',x)\ne F_C(\theta,x).
\]

**Definition 7.3 (objective-visible update).** It is objective-visible on `E`
when

\[
L_E(\theta')\ne L_E(\theta)
\]

at the declared evaluation precision.

**Definition 7.4 (useful update).** It is useful at margin `delta > 0` when

\[
L_E(\theta')\le L_E(\theta)-\delta.
\]

These predicates form the diagnostic ladder

\[
\text{coherent backward signal}
\rightarrow\text{parameter-visible}
\rightarrow\text{function-visible}
\rightarrow\text{objective-visible}
\rightarrow\text{useful}.
\]

No arrow is reversible, and an earlier predicate does not guarantee a later
one. Existing p10m experiments have separately exhibited parameter movement
without function movement, function movement without target-probability
movement, and target-probability movement without held-out improvement.

## 8. Reachability and capacity

The existing training contract is

\[
C=(\text{architecture},\text{initialization},\text{data},\text{objective},
\text{dtypes},\text{scales},\text{rounding},\text{batching},\text{optimizer}).
\]

**Proposition 8.1 (fixed deterministic contract).** If `C`, initial state `X_0`,
and training budget `T` fix every transition, the reachable endpoint set has
cardinality one.

**Proof.** Each state has one successor under the fixed transition map;
induction gives one state at every step. `□`

The phrase *distinct functions reachable under a fixed deterministic contract*
therefore requires a declared set of admissible controls.

**Definition 8.1 (controlled reachable set).** Let `U` be a set of allowed
control sequences, such as predeclared per-group shift schedules or carry
policies. Then

\[
\mathcal R_T(X_0,U)
=\{X_T(u):u\in U\}.
\]

**Definition 8.2 (empirical functional diversity).** On frozen surface `E`,

\[
C_E(T,U)
=\log_2\left|
\{F_C(\theta,x)_{x\in E}:X\in\mathcal R_T(X_0,U)\}
\right|.
\]

This is an empirical diversity measure, not a general neural-network capacity
law. Useful reachable diversity additionally restricts the set to endpoints
clearing a predeclared NLL margin.

## 9. Current contradictions and baseline

**Resolved code observation 9.1 (advertised profile geometry).** The production
profiles declare

\[
d_{head}^{p20m}=384/8=48,
\qquad
d_{head}^{p30m}=448/8=56,
\]

The earlier model-level validator required a power-of-two head dimension even
though the production path uses causal linear attention, whose checked kernel
requires only exact head divisibility. The validator now follows the deployed
kernel contract, and all three advertised profiles must pass both configuration
validation and the executable accumulator-bound contract in unit tests. This
resolves the internal profile contradiction; it does not authorize paid scale.

**Artifact observation 9.1 (uniform baseline).** A uniform distribution over
8,192 tokens has NLL

\[
\log_2(8192)=13\ \text{bits/token}.
\]

The selected 1,024-window p10m checkpoint reports 13.044 bits/token and 256
mistakes on 256 dev windows. Its geometric-mean target probability is only

\[
2^{-13.044}\approx0.970\cdot\frac1{8192}.
\]

The current production result demonstrates deterministic numeric liveness, not
held-out language learning above the uniform NLL baseline.

## 10. Open conjectures

### C1: normalized-gradient invariant

- State: `open`
- Claim: Q23 probabilities with Q47 Newton normalization reduce common-mode
  output-gradient error enough to create an update distinct from the matched
  Q15 legacy lane under a scale-compensated schedule.
- Falsifier: model update tensors remain identical after the predeclared bounded
  horizon, or their only difference is common-mode/logit-shift equivalent.

### C2: coarse-gradient descent alignment

- State: `open`
- Claim: the negative NSRL coarse-gradient direction is positively aligned with
  exact one-cell integer finite-difference improvement often enough to be a
  useful optimizer signal.
- Test: sample coordinates by parameter group and compare the sign of the coarse
  gradient with

  \[
  L_E(\theta-e_j)-L_E(\theta),
  \qquad
  L_E(\theta+e_j)-L_E(\theta).
  \]
- Falsifier: agreement is at or below a sign-matched random control, or the
  `quantized_nonzero` rescue subset is systematically anti-aligned.

### C3: requantization-margin prediction

- State: `open`
- Claim: projection-accumulator distance to the nearest rounding boundary
  predicts whether an actual parameter update becomes function-visible better
  than update count or parameter delta L1.
- Falsifier: margin predictions fail to classify held-out forward changes under
  matched update vectors.

### C4: residual drift dominates diffusion for useful crossings

- State: `open`
- Claim: parameter coordinates that later improve held-out NLL have higher
  cumulative-gradient coherence before their first update than coordinates
  whose crossings are neutral or harmful.
- Required measurements: signed cumulative gradient, sum of squared gradients,
  crossing time, residual sign changes, and exact post-update NLL delta.
- Falsifier: coherence does not predict direction or quality better than maximum
  absolute residual alone.

### C5: attention selectivity is a quality bottleneck

- State: `open`
- Claim: current Q/K ranges leave the positive linear-attention kernel too close
  to its constant component for useful token selection on held-out language
  windows.
- Required measurements: per-head kernel coefficient of variation, normalized
  attention entropy, effective attended-token count, and order/repetition
  counterexamples.
- Falsifier: trained heads are selective on held-out windows and selectivity is
  uncorrelated with NLL.

## 11. Next bounded experiment

The normalized wide-gradient preflight is complete and recovered target-
probability visibility without dev gain. The next bounded experiment is the
integer-lattice direction audit now implemented by `gradient-alignment-audit`.
On a predeclared p10m model, window surface, seed, and per-group coordinate
budget, it must compare the coarse-gradient proposal with exact stored-parameter
`-1` and `+1` changes scored by canonical Q20 integer NLL. Report by parameter
group and by rescue exposure:

1. forward-visible proposal frequency;
2. direction-comparable coordinates and sign agreement;
3. exact descent frequency for the proposed direction;
4. the same measurements for a deterministic random-direction control;
5. zero-gradient, objective-tie, gradient-saturation, and residual-saturation
   counts; and
6. replay equality under the same model, data, and sampling bindings.

Only a predeclared sample that beats the random control can justify optimizer
refinement. A failed proposal-surface fidelity gate points to the backward
surrogate rather than to more forward precision or scale sweeps. A held-out
transfer failure alone does not identify the backward oracle as the cause.

### Post-implementation calibration smoke

> **Correction (2026-07-15):** “disjoint following windows” below means
> distinct sliding-window indices, not token- or document-disjoint samples.
> Adjacent contexts overlap heavily. This smoke is not a held-out alignment
> result and its interpretation is superseded by
> [MJ-2026-07-15-04](MJ-2026-07-15-04-three-geometry-optimization.md).

The audit implementation was calibrated after its acceptance surface was
changed from the proposal window to disjoint following windows. This smoke is
post-hoc and small, so it is not promotion evidence. Its bindings were p10m
model `0xb10996e0707ab342`, dev stream `0xda195778ceb603ab`, four proposal
windows, four acceptance windows, two sampled nonzero coordinates per active
group, and seed 23.

- six coordinates were sampled across `final_rms`, `output`, and `bias`;
- four had unequal `-1` and `+1` acceptance NLL, with gradient agreement `3/4`
  versus deterministic random-control agreement `1/4`;
- the proposed direction was an exact held-out descent for three coordinates,
  versus one for the random control;
- all four rescue-free output/bias coordinates were comparable, but the two
  rescue-exposed `final_rms` coordinates were objective ties; and
- gradient and residual saturation were zero, while 111 STE rescues occurred
  in the proposal batch.

The result confirms that the measurement surface has sufficient resolution for
some output coordinates. It does not establish trunk alignment, statistical
superiority, or generalization. The required next result remains a predeclared,
larger replayed audit with group-level acceptance criteria.

## 12. Gap closure status

| Original gap | Status after this work | Remaining evidence |
|---|---|---|
| Backward-direction validity | Instrumented, still critical | Predeclared per-group audit; rescue-exposed trunk agreement above control |
| Canonical objective | Implemented for integer evaluation, lattice acceptance, and successor-v2 promotion | Bind future training reports and real baseline runs to the same objective |
| Scale/rounding ledger | Implemented for production projections, backward linear edges, updates, and advertised profile bounds | Extend explicit bounds/counters to every nonlinear and saturating intermediate |
| RHU boundary mismatch | Closed | Mixed arithmetic-shift/RHU choices remain intentional and ledgered |
| Faithful float relaxation | Implemented as `integer-relaxation-v2` | Run a matched float-transformer baseline on the successor suite |
| Invalid p20m/p30m profiles | Closed | Scaling remains gated on p10m quality, not geometry |
| Integer Adam lattice behavior | Open by design | Analyze moments only after coarse-gradient alignment passes |
| Attention selectivity | Open | Per-head kernel variation, entropy, effective-token count, and order controls |
| Profile-wide overflow coverage | Partial | Eliminate or separately count remaining internal saturating operations |
| Narrow empirical scope | Open | Fresh non-overlapping multi-corpus evaluation and actual float transformer |
| DRTO convergence/generalization | Research | Rotating-surface experiments and scale-aware trust-region norms |

## Decision

Paid p20m or p30m scaling is not mathematically authorized by current evidence.
The checked scale ledger, canonical NLL evaluator, faithful float relaxation,
successor-v2 contract, and valid profile geometry are now implemented as
infrastructure. Before scaling, NSRL still needs:

- normalized-gradient invariant reporting;
- a predeclared coarse-gradient alignment result that beats its random control;
- rescue-exposed trunk evidence rather than output-head-only alignment;
- and a p10m result below the uniform held-out NLL baseline.

The immediate work is bounded lattice calibration, interpreted through the
boundary ladder rather than through movement counts.

## Internal evidence

- [`quantized-optimization.md`](../quantized-optimization.md)
- [`production-model-v1.md`](../../docs/production-model-v1.md)
- [`integer-transformer-proof-v1.md`](../../docs/integer-transformer-proof-v1.md)
- [`p10m-probability-normalization-accuracy.json`](../../benchmarks/production-model-v1/p10m-probability-normalization-accuracy.json)
- [`p10m-probability-normalization-signal-attribution.json`](../../benchmarks/production-model-v1/p10m-probability-normalization-signal-attribution.json)
- [`p10m-up-forward-scale-training.json`](../../benchmarks/production-model-v1/p10m-up-forward-scale-training.json)

## Primary literature anchors

- Gupta et al., *Deep Learning with Limited Numerical Precision* (2015)
- Li et al., *Training Quantized Nets: A Deeper Understanding* (2017)
- Yin et al., *Understanding Straight-Through Estimator in Training Activation
  Quantized Neural Nets* (2019)
- Karimireddy et al., *Error Feedback Fixes SignSGD and other Gradient
  Compression Schemes* (2019)
- Gneiting and Raftery, *Strictly Proper Scoring Rules, Prediction, and
  Estimation* (2007)

Links and numeric-path classifications are maintained in
[`paper-catalog.md`](../paper-catalog.md).
