# MJ-2026-07-15-21: Correlated lattice optimization and scale-stable neural updates

- Date: 2026-07-15
- Status: recent low-precision and recurrent-network results synthesized into
  five bounded NSRL conjectures; no optimizer change, architecture promotion,
  or paid scaling authorized
- Extends: MJ-2026-07-14-02 and MJ-2026-07-15-04 through
  MJ-2026-07-15-13
- Code bindings:
  [`training.rs`](../../crates/nsrl-train/src/production/training.rs),
  [`nsrl-train-core`](../../crates/nsrl-train-core/src/lib.rs), and
  [`attention.rs`](../../crates/nsrl-core/src/attention.rs)
- Artifact bindings:
  [`integer-reachable-capacity-v1/longitudinal.json`](../../benchmarks/integer-reachable-capacity-v1/longitudinal.json),
  [`nsrl-mme-v0.json`](../../data/processed/nsrl-mme-v0.json), and
  [`solomon-latent-planner-eval.tsv`](../../docs/solomon-latent-planner-eval.tsv)
- Primary literature bindings:
  [ECO (2026)](https://arxiv.org/abs/2601.22101),
  [QuEST (2025)](https://arxiv.org/abs/2502.05003),
  [DiscQuant (2025)](https://proceedings.mlr.press/v291/chee25a.html),
  [mu-nit Scaling (2025)](https://proceedings.mlr.press/v267/narayan25b.html),
  [SigLIP 2 (2025)](https://arxiv.org/abs/2502.14786),
  [Global Minimizers of Sigmoid Contrastive Loss (2025)](https://arxiv.org/abs/2509.18552),
  [Preconditioned DeltaNet (2026)](https://arxiv.org/abs/2604.21100), and
  [Gated DeltaNet-2 (2026)](https://arxiv.org/abs/2605.22791)

## Question

Which recent mathematical and neural-network results can address NSRL's
current failures without weakening its native-integer, exact-replay, and
held-out-evidence contracts?

The useful answer is not a list of architectures. Five mechanisms map onto
measured NSRL bottlenecks:

1. embed quantization error in momentum rather than adding a second error-
   feedback buffer;
2. round a block of lattice proposals jointly against a document influence
   matrix rather than independently by coordinate;
3. use fixed-seed Walsh-Hadamard mixing and a fitting-only trust mask to reduce
   outliers and reject unstable directions;
4. derive static linear and residual scales from variance identities, and use
   a pairwise sigmoid objective for the failing multimodal binding surface; and
5. precondition the existing integer delta-rule memory while separating erase
   and write strength.

Every mechanism is an experiment candidate, not an imported result. The cited
papers operate under different numeric formats, objectives, data scales, or
model families. Their empirical findings do not establish an NSRL quality
gain.

## 1. Bound substrate observations

**Code observation 21.1 (production residual feedback).** For each production
parameter, `training.rs` accumulates an integer gradient in an `i64` residual,
rounds that residual by the parameter-group update shift, changes the stored
integer parameter, and subtracts the exact fine-grid mass consumed by the
coarse update. The optimizer artifact stores one `i64` residual per model
parameter.

**Code observation 21.2 (three-array integer Adam state).** Integer Adam stores
an `i64` first moment, a `u64` second moment, and a separate `i64` update
residual for every parameter. The residual is added after moment
normalization, rounded at the final effective update shift, and replaced by
the exact rounding remainder.

**Code observation 21.3 (integer fast-weight substrate).** The streaming TTT
attention path already implements an outer-product correction proportional to

\[
  \phi(k_t)(v_t-\widehat v_t)^\top,
\]

and the core exposes a fixed-point recurrent-state decay. The current TTT path
uses one learning-rate shift for both removal of the predicted association and
writing of the observed association. It does not use a diagonal curvature
state.

**Artifact observation 21.1 (reachability is not stability).** In the frozen
longitudinal reachable-capacity experiment, every one of the 16 early-
reachable configurations later improved the declared held-out probability
error, but all 16 recorded weight or hidden saturation; 20 of 30 total cells
saturated. Early movement is therefore a useful queue signal and an
insufficient optimizer certificate.

**Artifact observation 21.2 (multimodal binding failure).** The current
`NSRL-MME v0` headline is 371 per mille against a 700-per-mille gate. The
latent planner identifies only 3 of 72 held-out prompts at top one. This binds
the first multimodal intervention to representation alignment and hard-
negative discrimination before higher-resolution rendering.

## 2. Residual feedback identity and the ECO distinction

Consider one unsaturated stored integer parameter `theta_t`, an `s`-fractional-
bit fine update `g_t`, and a persistent fine-grid residual `r_t`. Define the
current residual-feedback recurrence abstractly as

\[
\begin{aligned}
  a_t &= r_t+g_t,\\
  \delta_t &= R_s(a_t),\\
  \theta_{t+1} &= \theta_t+\delta_t,\\
  r_{t+1} &= a_t-2^s\delta_t.
\end{aligned}
\]

Here `theta` and `delta` are stored integer units; `g`, `a`, and `r` are signed
`i64` fine-grid units representing real values multiplied by `2^s`. The sign
of `g_t` includes the optimizer's descent convention.

**Proposition 21.1 (exact cumulative-mass invariant).** Before accumulator or
parameter saturation,

\[
  2^s\theta_T+r_T
   =2^s\theta_0+r_0+\sum_{t=0}^{T-1}g_t.
\]

**Proof.** One recurrence step gives

\[
\begin{aligned}
2^s\theta_{t+1}+r_{t+1}
 &=2^s(\theta_t+\delta_t)
   +(r_t+g_t-2^s\delta_t)\\
 &=2^s\theta_t+r_t+g_t.
\end{aligned}
\]

Telescoping proves the claim. `square`

Thus NSRL already has exact error feedback. ECO is not evidence that NSRL
should merely add a residual.

**Literature result 21.1 (momentum-embedded compensation).** In the real-
valued analysis of ECO, a quantized iterate is formed from

\[
  \widetilde\theta_{t+1}=\widehat\theta_t-\eta\widetilde m_{t+1},
  \qquad
  \widehat\theta_{t+1}=Q(\widetilde\theta_{t+1}),
\]

with error

\[
  e_{t+1}=\widetilde\theta_{t+1}-\widehat\theta_{t+1}.
\]

The paper injects a scaled version of `e_(t+1)` into momentum, avoiding an
additional error buffer. Under its stated assumptions and decaying learning
rate, ECO converges to a constant-radius neighborhood while naive master-
weight removal has a worst-case error proportional to `1/eta`. Its experiments
use FP8 and INT4 quantization, not NSRL's stored `i8` lattice.

**Definition 21.1 (NSRL momentum-embedded comparator).** An admissible NSRL
ECO comparator must declare integer scales

```text
(f_m, f_e, eta_num, eta_shift, beta_num, beta_shift,
 rounding, saturation, injection_order)
```

for momentum, quantization error, learning rate, momentum coefficient, and
the injection arithmetic. It is a different discrete contract `C`; calling it
ECO does not make its integer trajectory equivalent to the paper's real-
valued recurrence.

**Conjecture C21.1 (state-reuse benefit, open).** On the frozen mini-
transformer corpus, a prospectively fixed momentum-embedded comparator can
remove the separate `update_residuals` vector while preserving exact replay
and matching or improving held-out canonical NLL, zero-probability count,
parameter saturation, and optimizer-state bytes relative to integer Adam.

**Falsifiers.** C21.1 is falsified on the bounded frame if any of the following
holds:

- the comparator needs an error buffer with the same asymptotic size as the
  removed residual vector;
- an exact replay mismatch occurs;
- its best proper-training-fixed checkpoint loses on held-out canonical NLL;
- it increases zero-probability windows or saturation; or
- apparent improvement disappears when the current residual lane receives
  the same state-byte budget.

## 3. Correlated lattice rounding from discrepancy

Let `E_fit={1,...,m}` be proper-fitting documents and let
`A_1,...,A_p` be an ordered family of admissible one-cell parameter actions.
For Q32 document loss, define the exact singleton influence matrix

\[
  B_{di}=\ell_d(\{i\})-\ell_d(\varnothing)\in\mathbb Z.
\]

Let `u` in `Q^p` be a fine-grid proposal, expressed in coarse parameter-cell
units, and let `x` in `Z^p` be its deployable lattice rounding. The admissible
coordinate set may be `{floor(u_i),ceil(u_i)}` or a declared subset of
`{-1,0,+1}`.

**Definition 21.2 (document discrepancy).** The linearized document
discrepancy of `x` relative to `u` is

\[
  \operatorname{Disc}_{E_{fit}}(x;u)
    =\lVert B(x-u)\rVert_\infty.
\]

Coordinatewise round-to-nearest minimizes each `|x_i-u_i|`; it does not in
general minimize this document-level quantity.

For binary `x`, define the exact interaction remainder

\[
  \rho_d(x)
   =\ell_d(x)-\ell_d(0)-\sum_{i=1}^p B_{di}x_i.
\]

This is the Boolean-jet mass above singleton order evaluated at `x`.

**Lemma 21.2 (exact surrogate error decomposition).** For every fitting
document `d`,

\[
  \left|\ell_d(x)-
  \left(\ell_d(0)+\sum_iB_{di}u_i\right)\right|
  \le |(B(x-u))_d|+|\rho_d(x)|.
\]

Consequently,

\[
  \max_d|\ell_d(x)-\widehat\ell_d(u)|
  \le \operatorname{Disc}_{E_{fit}}(x;u)
     +\lVert\rho(x)\rVert_\infty.
\]

**Proof.** Add and subtract the singleton surrogate at `x`, use the definition
of `rho_d(x)`, and apply the triangle inequality. `square`

**Literature result 21.2 (data-dependent correlated rounding).** DiscQuant
studies rounding on a fixed quantization grid through discrepancy theory. It
proves a distributional approximation result when the original model's
gradient space is approximately low rank and reports that data-dependent joint
rounding can substantially outperform coordinatewise rounding. The result is
post-training quantization from a continuous model; it does not establish an
optimizer for a native-integer trajectory.

**Definition 21.3 (NSRL discrepancy proposal operator).** A prospective NSRL
operator may use `B` and `u` from proper-fitting sources to choose

\[
  x^*\in\arg\min_{x\in\mathcal X}
  \left(
    \lVert B(x-u)\rVert_\infty
    +\lambda\lVert x\rVert_0
  \right),
\]

with an exact lexicographic tie break. `lambda` is a nonnegative integer Q32
write-cost penalty. The operator is only a generator. The selected move must
still pass exact interaction reconstruction and untouched source-level
evaluation.

**Conjecture C21.2 (correlated-rounding stability, open).** At equal write
mass, a fitting-only discrepancy proposal has a lower worst-document proposal
error and a higher untouched-source favorable-sign rate than independent
residual boundary crossings.

**Falsifiers.** Reject or narrow the proposal if its fitted influence matrix
is not stably low rank, the exact interaction remainder dominates the
discrepancy term, the selected `x` is unchanged from coordinatewise rounding,
or the untouched favorable-sign rate or aggregate canonical objective does
not improve.

## 4. Walsh-Hadamard mixing and fitting-only trust

**Literature result 21.3 (distribution normalization and trust).** QuEST uses
Hadamard normalization to reduce quantization distortion and a trust estimator
to suppress gradient components whose quantized computation is a poor proxy
for the unobserved full-precision gradient. Its stable one-bit results use
quantization-aware training with continuous latent quantities and therefore
do not transfer automatically to NSRL.

Let `n` be a power of two and let
`H_n` in `{-1,+1}^{n x n}` be the
unnormalized Walsh-Hadamard matrix. Let `D_sigma` be a diagonal fixed-sign
matrix whose signs are generated by a bound counter-based seed. For an integer
vector `z`, define

\[
  y=H_nD_\sigma z.
\]

All butterflies accumulate in a declared wide integer type before one final
requantization.

**Lemma 21.3 (exact pre-quantization energy identity).** Before saturation or
requantization,

\[
  \lVert y\rVert_2^2=n\lVert z\rVert_2^2.
\]

**Proof.** `D_sigma` is orthogonal and
`H_n^T H_n=nI`. Therefore

\[
  y^Ty=z^TD_\sigma H_n^TH_nD_\sigma z=nz^Tz.
\]

`square`

The identity preserves total energy up to the known factor `n`; it does not
guarantee a smaller infinity norm for every vector or every fixed sign seed.
Peak reduction is an empirical gate.

For fitting source panels `u=1,...,N` and coordinate `i`, let
`a_ui` in `{-1,0,+1}` be the sign of an exact integer finite difference and let
`q_ui` be the sign proposed by the quantized backward path. Define the Q15
trust score

\[
  \tau_i^{(15)}
   =R_0\left(
       \frac{2^{15}}{N_i}
       \sum_{u:a_{ui}q_{ui}\ne0}a_{ui}q_{ui}
     \right),
\]

where `N_i` is the number of nonzero comparable source panels. Coordinates
with `N_i=0` are ineligible. A threshold `tau_min` and all tie breaks must be
frozen before calibration or evaluation outcomes open.

**Definition 21.4 (trusted mixed proposal).** A trusted proposal is generated
by:

1. applying the fixed-seed integer Hadamard transform to a declared gradient
   block;
2. requantizing with a declared `FixedScale`;
3. undoing the transform, when required by the parameter basis; and
4. admitting only coordinates whose proper-fitting trust score meets
   `tau_min`.

Hadamard mixing and trust masking are distinct interventions and require
separate ablation rows.

**Conjecture C21.3 (outlier and sign stability, open).** For the K projection
and gated-MLP trunk blocks, fixed-seed Hadamard mixing reduces peak-to-RMS
gradient ratio and saturation, while the fitting-only trust mask increases
proposal/transfer sign agreement without collapsing update reachability.

**Falsifiers.** The mechanism fails if peak or saturation does not fall, exact
replay fails, the inverse transform creates more error than it removes, the
trust mask retains no useful coordinates, or matched untouched performance is
no better than the unmasked random-coordinate control.

## 5. Static scale and residual variance contracts

**Literature result 21.4 (mu-nit scaling).** Mu-nit Scaling combines static
fan-in scaling, unit-variance initialization, post-branch normalization,
variance-preserving residual weights, and width-aware learning-rate transfer.
It reports FP8 hidden-layer training matching higher-precision baselines
without dynamic scale estimation. Its FP8 result is evidence for the scaling
principles, not for NSRL's integer coefficients.

Consider a linear output

\[
  y_j=c_n\sum_{i=1}^n w_{ji}x_i.
\]

Under the declared initialization assumptions

```text
E[w_ji]=E[x_i]=0,
Var(w_ji)=Var(x_i)=1,
and all summed products are independent,
```

choosing `c_n=1/sqrt(n)` gives `Var(y_j)=1`.

For one residual branch, let `x` and `f(x)` have unit variance and covariance
zero. Then

\[
  y=ax+bf(x)
  \quad\Longrightarrow\quad
  \operatorname{Var}(y)=a^2+b^2.
\]

**Definition 21.5 (Q15 residual variance defect).** Let integer coefficients
`a_15,b_15` represent `a=a_15 2^-15` and `b=b_15 2^-15`. Define

\[
  \epsilon_{res}
   =\frac{|a_{15}^2+b_{15}^2-2^{30}|}{2^{30}}.
\]

This is an exact coefficient defect. It is not a bound on observed variance
when the unit-variance or zero-covariance assumptions fail.

**Proposition 21.4 (idealized depth drift).** If every one of `L` residual
blocks satisfies the same independence assumptions and has multiplicative
variance factor in `[1-epsilon,1+epsilon]`, then its variance relative to the
initial stream lies in

\[
  [(1-\epsilon)^L,(1+\epsilon)^L].
\]

**Proof.** Multiply the per-layer variance factors and use the endpoint factor
at every layer. `square`

**Decision 21.1.** Static fan-in scales, post-branch normalization, and Q15
weighted residuals must be tested as separate, fresh-model architecture lanes.
They cannot modify the frozen successor-v2 model or inherit its promotion
status.

## 6. Pairwise sigmoid geometry for the multimodal floor

**Literature result 21.5 (sigmoid multimodal alignment).** SigLIP 2 combines
pairwise sigmoid image-text alignment with captioning, self-distillation, and
masked prediction and reports improvements in retrieval and dense visual
features. The later geometry paper characterizes zero-loss sigmoid
configurations using margin/relative-bias constellations and proposes an
explicit relative-bias parameterization. Both papers train much larger
continuous models than NSRL.

Let `u_i` be a Q15 text embedding and `v_j` a Q15 image embedding. Their wide
dot product is Q30. Let `s_ij` be that dot product requantized to declared Q`f`,
let `y_ij=+1` for a true pair and `-1` for a declared negative, and let `t` and
`b` be integer inverse-temperature and relative-bias parameters with explicit
scales. Define

\[
  \operatorname{softplus}_2(z)=\log_2(1+2^z)
\]

and the Q32 audit objective

\[
  L_{sig}
   =\sum_{i,j}\omega_{ij}
      \operatorname{softplus}_2
       \left(-y_{ij}(t s_{ij}-b)\right).
\]

`omega_ij` are nonnegative integer weights. All positive pairs, ordinary
negatives, and hard negatives are enumerated by the contract. The base-2
exponential, logarithm, rounding, and overflow limits must be canonical.

**Lemma 21.5 (margin loss bound).** If every declared pair satisfies

\[
  y_{ij}(t s_{ij}-b)\ge m>0,
\]

then

\[
  L_{sig}
  \le\left(\sum_{i,j}\omega_{ij}\right)
     \log_2(1+2^{-m}).
\]

**Proof.** `softplus_2` is increasing, so each term is at most
`omega_ij log_2(1+2^-m)`. Sum the terms. `square`

Each pairwise term has no normalization denominator coupling it to the number
of other candidates. This avoids the target-resolution failure mode of a
single large-vocabulary softmax, but introduces its own temperature, bias, and
negative-weight contract.

**Conjecture C21.4 (multimodal binding objective, open).** A small native-
integer text/image alignment head trained with the declared pairwise Q32
objective improves the minimum of `identity_source_binding` and
`hard_negative_match` on untouched NSRL-MME rows without reducing any other
headline component or relying on retrieval during generation.

**Falsifiers.** C21.4 fails if the held-out floor does not strictly improve,
hard-negative gains come from identity leakage, a generative component
regresses, the embeddings collapse to a constant or lookup table, or the
claimed score requires memory assistance excluded by NSRL-MME.

## 7. Diagonally preconditioned erase/write memory

**Literature result 21.6 (curvature-aware and decoupled recurrence).**
Preconditioned DeltaNet derives delta-rule recurrences from online least
squares and reports consistent gains from a diagonal curvature approximation.
Gated DeltaNet-2 separates channel-wise erase and write gates and reports
stronger language-modeling and retrieval results than several matched
recurrent alternatives. Neither paper evaluates fixed-point native-integer
state updates.

Let `k_t=phi(key_t)` be a nonnegative Q`f_k` key feature vector, `v_t` a Q15
value, `vhat_t` the current Q15 prediction, and `d_t` a nonnegative wide
diagonal curvature state. Define

\[
  d_t=R_{s_d}(\gamma_d d_{t-1})+k_t\odot k_t,
\]

with declared decay `gamma_d`, shift `s_d`, accumulator width, and saturation.
Using the existing deterministic reciprocal machinery, construct a Q15
preconditioned key

\[
  p_t^{(15)}
   =Q_{15}\left(k_t\oslash(d_t+\varepsilon)\right).
\]

The proposed split recurrent update is

\[
  S_t=\operatorname{Decay}(S_{t-1})
      -R_{s_e}\left(\eta_e p_t\widehat v_t^\top\right)
      +R_{s_w}\left(\eta_w p_t v_t^\top\right).
\]

`S_t` remains the existing signed `i64` fast-weight state. Products are wide
integer outer products; `eta_e,eta_w,s_e,s_w` are contract-bound integer
scales. Equal erase/write scales reduce the two final terms to a preconditioned
delta correction. Replacing `p_t` by `phi(k_t)` and using equal scales recovers
the current unsplit TTT correction, apart from its separately declared state
decay order.

**Conjecture C21.5 (preconditioned memory editing, open).** On a frozen
synthetic binding suite, diagonal preconditioning reduces stale-association
error and state saturation relative to the current TTT delta rule; allowing
separate erase and write scales then improves multi-key overwrite accuracy
without contaminating independent sessions.

**Falsifiers.** Reject the mechanism if it fails streaming/batch parity,
requires an unbounded state, increases saturation or stale-memory error, harms
fresh-key retrieval, changes state across session resets, or gains only from a
larger parameter/state budget than the matched control.

## 8. Prospective experiment ladder

### E21-A: momentum-embedded quantization error

Use the existing mini-transformer integer-Adam path before touching production.
Freeze model initialization, corpus, window order, batch geometry, objective,
checkpoint rule, and state-byte budget. Compare:

1. current integer Adam with explicit update residual;
2. the declared integer momentum-embedded comparator using round-half-up; and
3. a counter-seeded stochastic-rounding audit control that remains exactly
   replayable from its bound seed and counter.

Primary endpoint: untouched canonical NLL. Joint gates: zero-probability
windows, exact replay, parameter and moment saturation, model/function update
hashes, delayed activation, and serialized optimizer bytes.

### E21-B: trusted correlated proposal

On multiple proper-fitting source clusters, build the singleton influence
matrix and freeze a small atom family. Run a factorial comparison:

```text
rounding: independent | discrepancy-correlated
mixing:  none | fixed-seed Hadamard
mask:    none | fitting-only trust
```

The proposal generator sees only fitting sources. Exact Boolean reconstruction
on separate calibration sources measures interaction remainder. Untouched
source panels decide favorable-sign rate and canonical aggregate effect.

### E21-C: static scale and residual lane

Train fresh, matched small transformers with independently ablated fan-in
scales, post-branch normalization, and Q15 variance-preserving residual
weights. Record per-layer RMS, peak, saturation, zero-update fraction, and
held-out canonical NLL. Do not transfer a successful lane's status to the
frozen model.

### E21-D: multimodal pair geometry

Build all declared positive and negative pairs for the 72-identity Solomon
surface, including the frozen NSRL-MME hard negatives. Compare the existing
latent objective with an auxiliary Q32 sigmoid head under identical train/
held-out identity splits. Report every NSRL-MME component, not only retrieval
top one.

### E21-E: preconditioned recurrent memory

Use a synthetic suite with insert, overwrite, collision, long-gap recall,
session reset, and adversarial repeated keys. Compare additive state, current
delta TTT, diagonal preconditioning, and split erase/write under equal state
and compute budgets. Solomon generation remains out of scope until this suite
passes.

## 9. Decision and promotion boundary

**Decision 21.2.** The next optimizer experiment is E21-A because it is the
smallest direct test of a recent result against an already implemented NSRL
state layout. It does not authorize replacing production residual SGD.

**Decision 21.3.** E21-B is the next admissible proposal-family program. It
implements the post-MJ-07 requirement for a genuinely new, stability-aware
proposal generator and retains exact Boolean and cross-source falsification.

**Decision 21.4.** E21-D is the first product-facing objective experiment
because it directly targets two headline components and the measured latent-
planner failure. Better retrieval alone is not a headline pass.

No cited literature result authorizes:

- modifying the frozen successor-v2 artifact;
- changing the default production optimizer;
- treating float, FP8, or post-training quantization evidence as native-
  integer evidence;
- opening sealed documents for proposal design;
- paid scaling; or
- a release claim.

## Open work

1. Specify the exact integer ECO injection recurrence and prove its scalar
   no-saturation invariant, if one exists.
2. Implement a deterministic discrepancy solver with a canonical tie break
   and bounded runtime.
3. Measure the singular spectrum and source stability of the fitting influence
   matrix before relying on the low-rank premise.
4. Add exact wide Walsh-Hadamard butterflies and inverse/parity tests for the
   128- and 256-dimensional blocks.
5. Freeze Q15 residual coefficient pairs and their exact variance defects.
6. Specify a canonical base-2 Q32 softplus evaluator and overflow bounds.
7. Define the diagonal curvature scale and reciprocal accuracy contract for
   recurrent memory.
8. Execute no promotion experiment until fitting, calibration, and untouched
   source roles are byte-bound in a preregistration.
