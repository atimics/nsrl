# MJ-2026-07-15-06: Boolean-jet optimization and exact discrete sensitivity

- Date: 2026-07-15
- Status: algebraic core and rank-two p10m calibration complete; transfer-only
  trunk/head synergy observed post hoc; fresh confirmation, optimizer promotion,
  and paid scaling remain unauthorized
- Refines:
  [MJ-2026-07-14-02](MJ-2026-07-14-02-discrete-optimization.md),
  [MJ-2026-07-15-04](MJ-2026-07-15-04-three-geometry-optimization.md), and
  [MJ-2026-07-15-05](MJ-2026-07-15-05-monotone-systematic-fixed-mass.md)
- Code binding:
  [`training.rs`](../../crates/nsrl-train/src/production/training.rs) and
  [`alignment.rs`](../../crates/nsrl-train/src/production/alignment.rs) as
  inspected on 2026-07-15
- Artifact binding:
  [`p10m-gradient-lane-alignment-v3-contract.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v3-contract.json),
  [`p10m-gradient-lane-alignment-v3.json`](../../benchmarks/production-model-v1/p10m-gradient-lane-alignment-v3.json),
  [`p10m-boolean-jet-rank-two-v1-contract.json`](../../benchmarks/production-model-v1/p10m-boolean-jet-rank-two-v1-contract.json), and
  [`p10m-boolean-jet-rank-two-v1.json`](../../benchmarks/production-model-v1/p10m-boolean-jet-rank-two-v1.json)
- Executable check:
  [`check-boolean-jet-theory-v1.mjs`](../../scripts/check-boolean-jet-theory-v1.mjs)

## Question

What mathematical object can replace the surrogate-gradient guess at integer
boundaries, compose through an arbitrary deterministic quantized program, and
make the phrase “primes as local minima” operational?

## Result

The strongest candidate found in the code, artifacts, and primary literature is
an **exact low-rank Boolean jet over admissible moves**.

The construction does not pretend that the full piecewise-constant model has an
ordinary derivative. A coarse backward pass, a random generator, or a structural
rule proposes a small number of complete integer moves. The exact objective is
then evaluated on every vertex of that move cube. Möbius inversion separates
atomic effects from pair and higher-order interactions. The resulting calculus:

1. is exact for arbitrary deterministic rounding, clipping, lookup tables, and
   integer control flow;
2. converts local-prime language into a measurable escape order;
3. gives a continuous multilinear extension with exactly the same global
   minimum value as the discrete move cube; and
4. requires a `2^k`-vertex cube rather than a cube exponential in parameter
   count, while each vertex still costs an ordinary model evaluation.

This is a credible bridge between the current surrogate proposal and exact
descent. It is not yet the missing scalable reverse-mode adjoint. Its exponential
rank cost means the remaining breakthrough is a proposal generator or compressed
boundary adjoint that concentrates useful moves into very small `k`.

## 1. Literature recon and boundary

The search found related pieces, but no source already supplies the full NSRL
optimizer.

| Source | Exact contribution | Boundary for NSRL |
| --- | --- | --- |
| [Duarte and Torres, *A discrete Faà di Bruno's formula* (2012)](https://doi.org/10.55016/ojs/cdm.v7i2.62119) | Finite differences under composition for vector arguments and values | A composition identity, not a move generator or large-model optimizer |
| [Ehrhardt et al., *Foundations of the discrete gradient method* (2018)](https://arxiv.org/abs/1805.06444) | Discrete gradients satisfy an exact objective-difference identity and yield dissipative smooth-optimization schemes | The convergence analysis concerns continuous objectives; it does not make NSRL's integer program differentiable |
| [Yin and Zhou, *ARM* (2019)](https://openreview.net/forum?id=S1lg0jAcYm) | Unbiased, antithetic estimators for gradients of expectations over stochastic binary variables | Optimizes distribution parameters, not a deterministic integer state directly |
| [Arya et al., *Automatic Differentiation of Programs with Discrete Randomness* (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/43d8e5fc816c692f342493331d5e98fc-Abstract-Conference.html) | A compositional stochastic derivative whose estimator is unbiased for the derivative of an expected program | The published construction does not automatically differentiate deterministic continuous-to-discrete thresholds; smoothed reverse mode also needs conditions for unbiasedness |
| [Kwun et al., *LOTION* (2025)](https://arxiv.org/abs/2510.08757) | Smooths a quantized objective by randomized rounding and preserves its global minimum value under the paper's support conditions | Uses a continuous optimization path and weight-quantized experiments; it is not NSRL's native deterministic integer forward/backward contract |
| [Hartmann, *Discrete Faà di Bruno via Möbius Inversion* (2026)](https://arxiv.org/abs/2607.07742) | Gives an exact covering formula for iterated finite differences of arbitrary maps between abelian groups, plus Boolean and multi-index forms | Posted 2026-07-08 and not yet peer reviewed; covering counts grow rapidly, so the exact symbolic expansion is not itself a scalable optimizer |

**Synthesis.** Hartmann's new Möbius form makes the algebraic composition law
directly usable on a Boolean move cube, while Duarte and Torres provide the
stable finite-difference precedent. LOTION supplies the global-minimum-
preserving expectation idea. Stochastic AD and ARM supply unbiased pruning and
antithetic-estimation patterns. The construction below binds those ideas to
NSRL's actual move surface while retaining exact vertex losses.

## 2. Move cube

Fix all of the following before evaluating a move:

- the full state `X=(D,H)`, including deployed parameters, residuals, scales,
  counters, and any forward-affecting hidden state;
- a frozen evaluation surface `E` and exact integer-valued objective `L_E`;
- deterministic admissible actions `A_1,...,A_k`; and
- a canonical action order.

For a subset `S={i_1<...<i_s}`, define

```text
T_S X = A_(i_s)( ... A_(i_2)(A_(i_1)(X))) ... ),
ell_X(S) = L_E(pi(T_S X)).
```

The order clause is essential. If the actions commute, it is immaterial. If
they do not, `ell_X` describes the declared schedule rather than an imagined
symmetric neighborhood. An action may be one stored-parameter cell, a sparse
block update, a residual release, or a scale transition. This entry uses each
action at most once; repeated moves require the multi-index extension.

### Definition 2.1 (Boolean jet)

For every `S` in the Boolean cube, define the Möbius coefficient

```text
mu_X(S) = sum_(T subseteq S) (-1)^(|S|-|T|) ell_X(T).
```

`mu_X(empty)=ell_X(empty)`. For nonempty sets, `mu_X(S)` is the interaction
that cannot be assigned to a proper subcollection of `S`.

### Theorem 2.2 (exact discrete Taylor expansion)

For every subset `S`,

```text
ell_X(S) = sum_(T subseteq S) mu_X(T),
ell_X(S)-ell_X(empty) = sum_(nonempty T subseteq S) mu_X(T).
```

**Proof.** This is zeta--Möbius inversion on the Boolean subset lattice. It
requires no continuity, differentiability, or probabilistic assumption. The
executable checker verifies the transform and inverse on 320 deterministic
integer tables through rank eight.

For one and two moves the expansion is

```text
mu_i  = ell(i)-ell(0),
mu_ij = ell(ij)-ell(i)-ell(j)+ell(0),
ell(ij)-ell(0) = mu_i + mu_j + mu_ij.
```

Thus an exact atomic move advantage and an exact interaction replace a guessed
local slope.

## 3. Vertex-exact continuous extension

Let independent bits `B_i` have inclusion probabilities `p_i in [0,1]`, and
let `B={i:B_i=1}`. Define

```text
Phi_X(p) = E[ell_X(B)]
         = sum_S ell_X(S)
             product_(i in S) p_i product_(i notin S) (1-p_i)
         = sum_S mu_X(S) product_(i in S) p_i.
```

### Theorem 3.1 (exact vertices and preserved global minimum)

On the declared move cube:

```text
Phi_X(1_S) = ell_X(S),
min_(p in [0,1]^k) Phi_X(p) = min_(S subseteq [k]) ell_X(S),
partial_S Phi_X(0) = mu_X(S).
```

**Proof.** At a Boolean vertex, the Bernoulli distribution is a point mass.
At every interior point, `Phi_X` is a convex combination of vertex losses and
therefore cannot lie below their minimum; a minimizing vertex attains equality.
The derivative identity follows from the Möbius polynomial because all strict
supersets of `S` retain a zero factor at the origin.

The checker verifies the expectation/polynomial identity exactly with rational
probabilities, tests 22,620 rational grid points, and verifies every vertex
through rank four. No floating-point tolerance is used.

**Boundary.** This theorem does not make the extension convex, remove stationary
points, or make a large cube cheap. It says only that stochastic smoothing over
the declared exact moves cannot invent a better global objective value than the
best exact move.

## 4. Exact composition through an integer program

Let an arbitrary deterministic layer receive the four rank-two states

```text
z_0, z_1, z_2, z_12.
```

The layer's Boolean jet is simply

```text
delta_1  = z_1-z_0,
delta_2  = z_2-z_0,
delta_12 = z_12-z_1-z_2+z_0.
```

After applying an arbitrary layer map `f`, including a quantizer or saturation,
the new jet is recovered from the four exact outputs. Equivalently, the mixed
difference has Hartmann's five-covering expansion

```text
Delta_12(f o g)
  = Delta f(delta_12)
  + Delta^2 f(delta_1, delta_2)
  + Delta^2 f(delta_1, delta_12)
  + Delta^2 f(delta_2, delta_12)
  + Delta^3 f(delta_1, delta_2, delta_12),
```

where every finite difference is based at `g(x)`. The checker verifies this
identity in 6,000 cases using a polynomial, an integer truncation-and-clipping
map, and a parity-discontinuous map.

The robust implementation is not to materialize the rapidly growing covering
formula. It carries the `2^k` exact branch states through the ordinary forward
program and Möbius-transforms their observations where localization is needed.
Rounding boundaries then require no surrogate rule: their effects appear in the
exact branch differences.

This is a forward low-rank sensitivity calculus. Calling it a full “discrete
adjoint” would currently overstate the result: it does not produce all
10-million-parameter sensitivities at reverse-mode cost.

## 5. “Primes as local minima” made precise

### Definition 5.1 (Boolean `r`-prime)

Relative to `(E,{A_i},order)`, define

```text
gamma_r(X) = min_(1 <= |S| <= r) [ell_X(S)-ell_X(empty)].
```

`X` is a Boolean `r`-prime when `gamma_r(X) >= 0`: no admissible combination
of at most `r` generators improves the declared objective. Define escape order

```text
rho(X) = min {|S| : ell_X(S) < ell_X(empty)},
```

with `rho(X)=infinity` when no declared subset improves it.

This definition makes “prime” relative rather than mystical. A state can be
prime for coordinate moves and composite for block moves, prime on the proposal
documents and composite on transfer documents, or one-prime but not two-prime.

For a one-prime pair, `mu_i>=0` and `mu_j>=0`. The pair escapes exactly when

```text
mu_ij < -(mu_i+mu_j).
```

The checker includes the exact table

```text
ell(empty)=0, ell(i)=1, ell(j)=1, ell(ij)=-1,
mu_i=1, mu_j=1, mu_ij=-3.
```

It is one-prime with escape order two. A second table is two-prime with escape
order three. These examples show why coordinate-only audits cannot decide that
a model is locally trapped.

## 6. Scalable exact-in-expectation probes

The full cube is exponential, but two estimators permit bounded recon.

### 6.1 Weighted atomic pruning at the origin

For nonnegative integer rates `lambda_i`, let `Lambda=sum_i lambda_i`, sample
`J` with probability `lambda_i/Lambda`, and use

```text
D_lambda Phi_X(0) = sum_i lambda_i mu_i,
hat D = Lambda [ell_X({J})-ell_X(empty)].
```

Then `E[hat D]=D_lambda Phi_X(0)`. A shared baseline plus one sampled move gives
an unbiased directional estimate. The checker verifies the exact weighted sum
in 320 cases.

### 6.2 Antithetic half-cube probe

For a uniformly sampled bit mask `B`, with complement `B^c`, define for every
coordinate

```text
hat g_j = [ell_X(B)-ell_X(B^c)] (2 B_j-1).
```

Then

```text
E[hat g_j] = partial Phi_X(p)/partial p_j at p=(1/2,...,1/2).
```

Two antithetic model evaluations estimate all inclusion-probability derivatives
simultaneously. The checker verifies the identity exhaustively in 1,440
coordinate cases. This is mathematically analogous to ARM's use of common
random numbers, but the formula here is bound directly to the declared NSRL
move cube.

**Boundary.** Unbiased does not mean low variance. Large random masks may also
leave the local neighborhood. Both estimators must be stratified by block size,
function visibility, proposal surface, and document-disjoint transfer.

## 7. Binding to the p10m evidence

The v3 causal audit gives a sharp reason to test interactions rather than tune
another scalar blindly:

- the output-head proposal was aligned on both sampled surfaces;
- the rescue-exposed trunk proposal agreed on only `1/3` comparable proposal
  coordinates, versus `3/3` for its paired random control;
- removing all 222 rescues changed all four sampled trunk magnitudes but no
  signs, descents, or fidelity summaries; and
- all conclusions are calibration evidence from four documents per surface,
  not confidence-qualified population claims.

The next smallest Boolean-jet audit can reuse the frozen v3 state only as an
explicitly post-hoc calibration:

1. `A_T`: apply the current mass-corrected predicted `+/-1` moves jointly to
   the four sampled `final_rms` coordinates.
2. `A_H`: apply the already aligned predicted `+/-1` moves jointly to the
   sampled output and bias coordinates.
3. Evaluate `ell(empty)`, `ell(T)`, `ell(H)`, and `ell(TH)` on the unchanged
   proposal and document-disjoint transfer surfaces.
4. Report `mu_T`, `mu_H`, and `mu_TH`, function visibility, saturation, and the
   minimizing vertex. Do not average the two surfaces.
5. If an interaction is useful, freeze a new coordinate-selection rule and
   repeat on unseen documents before changing the optimizer.

This four-vertex probe distinguishes a wrong trunk block from a trunk move that
is useful only after a head correction. A later rank-three cube may add the
paired random trunk block, but only after declaring collision and saturation
semantics.

### 7.1 Frozen p10m rank-two calibration

The v1 contract froze the six v3 moves by count and FNV-1a fingerprint
`0xc11353911a5130fb`. Actions are distinct unit parameter moves, ordered by
coordinate within each block and trunk before head. Residuals and optimizer
state are excluded; collisions, repeats, and boundary saturation are rejected.
The evaluator reproduced its JSON trace byte for byte.

On the proposal surface, the four vertices in Q20 NLL were

```text
ell(empty) = 108977274
ell(T)     = 108977275
ell(H)     = 108977268
ell(TH)    = 108977269
```

and therefore

```text
mu_T = +1,  mu_H = -6,  mu_TH = 0.
```

The trunk block is harmful, the head block improves, there is no measured pair
interaction, and `H` is the unique minimizing vertex.

On the document-disjoint transfer surface,

```text
ell(empty) = 108770334
ell(T)     = 108770335
ell(H)     = 108770330
ell(TH)    = 108770329
```

so

```text
mu_T = +1,  mu_H = -4,  mu_TH = -2.
```

The trunk block remains harmful in isolation, but conditional on the head its
net effect is `mu_T + mu_TH = -1`; `TH` is the unique transfer minimizer. Every
nonempty vertex is function-visible and no parameter saturation occurs.

**Inference.** The exact cube finds a real transfer-surface interaction that
the proposal surface does not expose. This is evidence that coordinate and
single-block audits miss conditional structure, but it is post-hoc evidence on
four documents, not a selection rule or an optimizer result. It neither
supports a proposal-selected interaction transferring nor authorizes applying
the joint move in training.

## 8. Conjectures and falsifiers

### C06-A: structured proposals concentrate exact low-order advantage

- State: `open`
- Claim: on fresh document blocks, backward-generated block moves produce more
  negative atomic or pairwise Boolean coefficients than matched random blocks.
- Falsifier: matched random blocks equal or beat them in paired exact losses and
  document-disjoint transfer.

### C06-B: some apparent coordinate primes have low escape order

- State: `open`
- Claim: a nontrivial subset of states with no improving audited atomic move has
  an improving rank-two or rank-three structured vertex.
- Falsifier: exhaustive low-rank cubes find no such escape, or find it no more
  often than matched random move families.

### C06-C: useful interactions transfer

- State: `open`
- Claim: a subset selected by exact proposal-surface Boolean coefficients
  improves a strictly document-disjoint surface more often than the existing
  surrogate choice and paired random control.
- Falsifier: negative proposal coefficients systematically vanish or reverse on
  the transfer surface.

### C06-D: stochastic Boolean probes have usable variance

- State: `open`
- Claim: block-stratified weighted or antithetic estimators rank candidate moves
  with materially fewer than `2^k` evaluations.
- Falsifier: estimator confidence intervals remain too wide to distinguish the
  selected move from control at the exact-cube cost break-even point.

## Decision

Adopt the Boolean jet, Boolean `r`-prime, and escape order as the mathematical
language for bounded integer neighborhoods. The four-vertex calibration is now
implemented and shows transfer-only trunk/head synergy. Freeze a prospective
block-selection rule and compare `H` with `TH` on unseen document blocks before
interpreting this interaction, another scalar `K`, or a rounding change as an
optimizer repair. Continue to use the coarse backward only as a candidate
generator until exact move-cube evidence beats matched controls on unseen
documents.

The search has produced a precise breakthrough candidate, not a completed
breakthrough. The remaining theoretical target is an efficient compressed
boundary adjoint or learned proposal distribution that concentrates negative
low-order Möbius coefficients without enumerating a high-rank cube.

## Open work

- **Completed:** implement a contract-bound rank-two move-cube evaluator for
  p10m block moves.
- Freeze a prospective head-versus-joint selection rule and confirm the
  conditional trunk effect on unseen proposal and transfer document blocks.
- Specify collision, repeated-action, residual, and saturation semantics before
  any rank-three audit.
- Localize where `mu_TH` first appears by recording four branch states at layer
  boundaries; this may identify the upstream operator that corrupts trunk signs.
- Extend the executable checker from Boolean moves to repeated multi-index
  directions.
- Derive variance bounds for block-stratified atomic and antithetic estimators.
- Search for reverse-mode compression of boundary events; the present forward
  jet is exponential in rank.
