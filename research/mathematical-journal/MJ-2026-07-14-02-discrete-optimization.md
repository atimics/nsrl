# MJ-2026-07-14-02: Optimization on the Functional Quotient

- Date: 2026-07-14
- Status: proposed theory
- Supersedes: none
- Depends on: [MJ-2026-07-14-01](MJ-2026-07-14-01-foundations.md)
- Code binding: `ProductionModelV1` residual integer SGD
- Artifact binding: integer reachable-capacity matrix and longitudinal result;
  p10m boundary, forward-scale, probability-resolution, and normalization audits

## Question

What optimizer follows from the behavior observed on the experimental
substrate, once integer training is treated as discrete function search rather
than as a low-precision imitation of smooth SGD?

The theory developed here has two goals:

1. preserve the useful property of residual error feedback; and
2. stop equating gradient liveness, integer parameter movement, or wider
   probability observation with objective improvement.

## 1. Empirical constraints on any replacement theory

The checked-in substrate rules out several simple optimization stories.

**Artifact observation 1.1 (parameter states overcount functions).** The
30-cell rank × shift × carry matrix produced 19 parameter-update hashes but only
15 functional-update hashes. Fourteen configurations were functional no-ops.
Optimization must therefore operate on, or at least measure, functional
equivalence classes rather than counting parameter states.

**Artifact observation 1.2 (carry is useful but not sufficient).** Residual
carry exposed function-visible updates in cells that were dead without carry.
But distinct rank-32 movement at shift 0 with carry finished worse than rank 16.
Error feedback expands the reachable neighborhood; it does not identify a
descent direction.

**Artifact observation 1.3 (early function movement is a filter).** In the
longitudinal matrix, early function movement had no false positive under the
declared later-gain classification, but it missed six delayed activations. Its
precision was 1.0, recall 0.727, Matthews correlation 0.645, and early
functional-delta L1 had Spearman correlation 0.828 with later held-out gain.
Long runs also saturated. Early function movement is therefore a useful
priority signal, not a complete stopping rule or safety proof.

**Artifact observation 1.4 (movement can be hidden by the forward lattice).**
The p10m `up` group recorded 50,568 nonzero update events in a bounded run after
its learning-rate boundary was crossed. Earlier matched models with materially
different `up` update counts still produced identical final features, logits,
probabilities, and losses until the forward shift was lowered to 7.

**Artifact observation 1.5 (forward visibility can be hidden by the objective
lattice).** At `up` forward shift 7, 250 of 256 feature/logit vectors and 124
probability vectors changed, while zero Q15 target probabilities changed. Q23
made target differences observable, but corrected normalization showed that
only four windows changed under exact division and five under Q47 Newton.

**Artifact observation 1.6 (objective visibility is not usefulness).** Wider
probability information and corrected normalization did not yet produce a
distinct quality-improving p10m checkpoint. The selected 1,024-window p10m
candidate remained worse than the 13-bit uniform NLL baseline.

These observations imply that an update proposal must pass several different
tests. No scalar such as update count, active rank, residual magnitude, or
probability-vector delta can stand in for all of them.

## 2. Optimization space

Let `P` be a diagnostic input surface and `A` an acceptance surface drawn from
training data but disjoint from the batch that generated the current update.
The frozen development and test surfaces are not optimizer inputs.

**Definition 2.1 (functional equivalence).** For a fixed numeric contract `C`,

\[
\theta\sim_P\theta'
\quad\Longleftrightarrow\quad
F_C(\theta,x)=F_C(\theta',x)\ \text{for every}\ x\in P.
\]

Write `[theta]_P` for the resulting equivalence class.

**Definition 2.2 (functional quotient graph).** Let each node be one class
`[theta]_P`. A directed edge

\[
[\theta]_P\longrightarrow[\theta']_P
\]

exists when `theta'` can be produced by an admissible integer optimizer action
without numeric saturation.

The practical problem is not continuous minimization over `R^d`. It is graph
search:

\[
\min_{[\theta]_P\in\mathcal G_C}L_A(\theta),
\]

where the graph is far too large to enumerate and the coarse backward pass is a
proposal oracle for a small local neighborhood.

**Principle 2.1.** The straight-through backward operator is not promoted to a
gradient. It proposes edges in the quotient graph. Exact deterministic forward
evaluation decides whether a proposed edge exists and whether its endpoint
improves the declared objective.

This division of labor matches the substrate:

- error feedback preserves weak repeated evidence;
- boundary geometry predicts which evidence can become an action;
- exact forward comparison detects functional equivalence; and
- exact integer NLL evaluates the action.

## 3. Boundary-aware proposal geometry

### 3.1 Residual drift and diffusion

For coordinate `j`, let the accumulated coarse-gradient samples over an
observation horizon be `g_1j, ..., g_nj`. Define

\[
G_j=\sum_{i=1}^ng_{ij},
\qquad
A_j=\sum_{i=1}^n|g_{ij}|,
\qquad
Q_j=\sum_{i=1}^ng_{ij}^2.
\]

**Definition 3.1 (signed coherence).** For `A_j > 0`,

\[
\rho_j=\frac{|G_j|}{A_j}\in[0,1].
\]

`rho_j=1` means every nonzero sample agreed in sign. Values near zero indicate
cancellation. A group summary should report weighted quantiles rather than only
the maximum residual.

If `mu_j` and `sigma_j^2` denote estimated per-sample drift and variance, then
the parameter half-cell `h_j=2^{s_j-1}` suggests the diagnostic time scales

\[
T_{drift,j}\approx\frac{h_j}{|\mu_j|},
\qquad
T_{diffusion,j}\approx\frac{h_j^2}{\sigma_j^2}.
\]

These are estimates, not convergence theorems. Their purpose is to distinguish
a predictable coherent boundary crossing from a noise-driven first passage.

**Proposal rule 3.1.** Prioritize coordinates or blocks with high coherence and
short drift crossing time. Do not lower a group shift solely because one
coordinate has a large absolute residual.

### 3.2 Parameter-boundary distance

With pre-update residual `a_j` and update shift `s_j`, the next positive and
negative one-cell crossings occur at

\[
a_j\ge2^{s_j-1}
\]

and

\[
a_j\le-2^{s_j-1}-1,
\]

respectively, under the current rounding rule.

For a candidate shift `s`, the exact proposed update is

\[
u_j(s)=R_s(a_j).
\]

There is no need to infer its parameter movement from a scalar summary; the
candidate vector can be constructed exactly before mutating optimizer state.

### 3.3 Forward-boundary distance

For one scalar projection accumulator `a` with forward shift `f`, define

\[
m_f(a,\delta)
=\mathbf1[R_f(a+\delta)\ne R_f(a)].
\]

For a weight update vector `u` and input `x`, the accumulator perturbation is

\[
\delta a=-u^\top x.
\]

Thus first-layer projection visibility can be predicted exactly from stored
accumulators and candidate updates. Define the projection visibility rate on a
probe surface `P` as

\[
V_{proj}(u;P)
=\frac{1}{N_P}
\sum_{a\in P}\mathbf1[R_f(a+\delta a)\ne R_f(a)].
\]

This is cheaper than a complete candidate forward and is used only to prune
obvious no-ops. The complete forward comparison remains authoritative because
later rounding, normalization, residual addition, and nonlinearities can mask
or amplify an earlier change.

## 4. Discrete residual trust-region optimizer

The proposed optimizer is called **discrete residual trust-region optimization**
(`DRTO`). The name is descriptive: residual error feedback supplies candidate
moves, while an exact finite neighborhood and acceptance rule replace an
assumed smooth descent step.

### 4.1 State

For each parameter group, DRTO stores:

\[
(r_g,s_g,H_g),
\]

where `r_g` is the ordinary error-feedback residual, `s_g` is the current
proposal shift, and `H_g` is a small rejection/cooldown ledger containing the
last rejected candidate hash and the residual epoch at which it was tested.

The global optimizer state also binds:

- the model and data hashes;
- the probability precision and normalization method;
- the diagnostic surface `P`;
- the rotating training acceptance schedule `A_t`;
- candidate masks and shift offsets; and
- deterministic tie-breaking rules.

### 4.2 Candidate neighborhood

Let `M_t` be a small predeclared family of masks. A mask may select one parameter
group, layer, matrix block, or deterministic top-boundary subset. Let `S_g` be a
small set of shifts around the current shift, normally

\[
S_g=\{s_g-1,s_g,s_g+1,s_g+2\}
\]

after clamping to valid bounds.

From the current accumulated residual `a`, construct

\[
u_{g,m,s}=m\odot R_s(a_g).
\]

The trust region imposes

\[
\|u\|_0\le k_t,
\qquad
\|u\|_\infty\le b_t,
\qquad
\|u\|_1\le d_t,
\]

with integer budgets recorded in the contract. These bounds control the number
and magnitude of simultaneous lattice jumps. They are not float norms used by
the runtime.

Candidate generation is hierarchical:

1. rank groups by residual drift/coherence and saturation headroom;
2. construct exact update vectors for a few group/shift/mask actions;
3. discard zero updates and predicted projection no-ops;
4. run exact forward comparison on `P`;
5. evaluate surviving candidates on `A_t`.

The no-update action is always present.

### 4.3 Canonical candidate score

For a candidate `u`, define

\[
\Delta_A(u)=L_{NLL,A_t}(\operatorname{Sat}(\theta-u))-L_{NLL,A_t}(\theta).
\]

Candidates with any parameter, activation, residual, or accumulator saturation
are infeasible during the bounded proof phase.

Selection is lexicographic:

1. smallest exact `Delta_A`;
2. smallest `||u||_0`;
3. smallest `||u||_1`;
4. largest update shift;
5. stable group, block, and candidate index.

Lexicographic selection avoids inventing penalty coefficients whose units would
themselves need tuning.

Accept `u*` only when

\[
\Delta_A(u^*)\le-\delta_A,
\qquad
F_C(\theta-u^*,P)\ne F_C(\theta,P),
\]

for a predeclared positive integer loss margin `delta_A`.

On acceptance:

- apply `theta <- Sat(theta-u*)`;
- consume `2^s u*` only for the selected coordinates;
- retain unselected residual coordinates exactly; and
- record the candidate, forward, loss, model, and optimizer hashes.

On rejection:

- leave model parameters and residuals unchanged;
- record the rejected candidate hash; and
- do not retest the identical candidate until its selected residual block has
  changed by a declared amount, changed sign, or reached the next acceptance
  epoch.

Rejection must not erase residual evidence. It also must not repeatedly spend
the same exact evaluation on an unchanged proposal.

### 4.4 Deterministic algorithm

```text
accumulate coarse gradients into residual state

at a proposal boundary:
    summarize drift, coherence, headroom, and boundary distance
    enumerate the bounded group × mask × shift neighborhood
    construct exact integer update vectors
    prune zero updates and predicted projection no-ops
    reject candidates with any saturation
    compare exact forward hashes on the diagnostic surface
    evaluate exact NLL on the rotating training acceptance surface
    choose by the frozen lexicographic rule

    if the winner clears the NLL margin:
        apply it and consume only its represented residual quantum
    else:
        preserve model and residual state
```

The acceptance surface is drawn from training data and rotates by a frozen
schedule. Development data remains checkpoint evidence; test data remains
untouched.

## 5. Basic guarantees

**Lemma 5.1 (variable-shift residual identity).** With zero saturation and
accepted update shifts `s_t`, DRTO satisfies

\[
r_T=r_0+\sum_{t<T}g_t-\sum_{t<T}2^{s_t}u_t,
\]

where rejected candidates have `u_t=0` and consume no residual.

**Proof.** Each accepted action subtracts exactly its represented residual
quantum; each rejected action leaves the residual unchanged. Telescope the
recurrence. `□`

**Proposition 5.1 (monotone acceptance loss).** On a fixed acceptance surface
`A`, if `L_A` is an integer and every accepted update decreases it by at least
`delta_A >= 1`, then DRTO accepts at most

\[
\left\lfloor\frac{L_A(\theta_0)-L_{A,min}}{\delta_A}\right\rfloor
\]

updates before no further accepted move is possible.

**Proof.** Sum the minimum loss decrease over accepted actions and use the lower
bound `L_A >= L_A,min`. `□`

This is a local termination statement, not global convergence. A rotating
acceptance schedule weakens monotonicity to the active surface, so checkpoint
development NLL must still be tracked without feeding it into update choice.

**Proposition 5.2 (no accepted functional no-ops).** If the exact diagnostic
forward comparison is part of the acceptance predicate, every accepted action
changes the quotient node on `P`.

This guarantee directly excludes the parameter-only movement observed in the
p10m `up` comparison.

**Proposition 5.3 (deterministic replay).** If candidate enumeration, acceptance
surfaces, integer evaluation, and tie breaking are fixed in the optimizer
contract, DRTO is byte-replayable.

None of these propositions proves generalization or validates the coarse
gradient. They convert three important desired behaviors—residual conservation,
function visibility, and local objective descent—into checked invariants.

## 6. Plateaus, exploration, and delayed activation

A strict one-step descent rule can stop at a local minimum of the candidate
graph. The substrate also contains delayed activations: six cells with no early
function movement later improved. DRTO must therefore distinguish *preserving
evidence* from *accepting movement*.

Residuals continue to accumulate while a group is inactive. A group is not
permanently pruned because its present candidate is a no-op. It is reconsidered
when drift, sign, boundary distance, or trust-region geometry changes.

For research beyond the bounded proof phase, define an `H`-step macro-action

\[
U=(u_1,\ldots,u_H)
\]

and accept it when its endpoint improves NLL even if an intermediate state lies
on the same functional plateau:

\[
L_A(\theta-U)\le L_A(\theta)-\delta_A.
\]

`H=2` with a very small beam is the first admissible extension. Intermediate
saturation remains forbidden. Temporary objective regressions should not be
authorized until monotone DRTO has been measured, because otherwise the theory
again collapses into an unconstrained sweep.

## 7. Scale control as model-predictive control

The shift controller should not optimize movement. It should choose which
finite candidate neighborhood DRTO evaluates.

At proposal epoch `t`, define the observation

\[
o_t=(\rho_g,T_{drift,g},T_{diffusion,g},
     V_{proj,g},\text{headroom}_g,
     \text{mass-error},\text{recent accept/reject history}).
\]

The action is a bounded change to group shift, mask budget, or proposal dwell
time. The immediate action value is exact candidate NLL after integer forward
evaluation, not delta L1 or update count.

This is a deterministic, one-step model-predictive controller. A learned
holographic controller is unnecessary until the logged state/action/outcome
tuples demonstrate a reward that predicts held-out quality and cannot be gamed
by freezing all movement.

## 8. Coarse-gradient calibration

DRTO can function with an imperfect proposal oracle, but proposal cost depends
on its quality. The oracle should be calibrated on the integer lattice.

For an admissible update vector `u`, define the exact discrete directional
change

\[
D_uL_A=L_A(\operatorname{Sat}(\theta-u))-L_A(\theta).
\]

The coarse backward signal predicts descent when

\[
\langle\tilde g,u\rangle>0
\]

because the parameter action subtracts `u`.

Record by group and by rescue status:

- sign agreement between predicted and exact direction;
- exact descent frequency;
- median and quantiles of `D_u L_A`;
- results for coordinates affected by `quantized_nonzero` rescue;
- a sign-matched deterministic random-control baseline; and
- calibration conditional on coherence and forward visibility.

This turns the question “does the STE work?” into a measurable conditional
proposal-quality model.

## 9. Experiment program

### Experiment A: retrospective substrate replay

Use the frozen reachable-capacity matrix without retraining. Test whether the
DRTO filters—early function visibility, coherence when available, zero
saturation, and exact early objective movement—would have prioritized later
held-out gains. This calibrates selection rules but cannot serve as prospective
confirmation.

### Experiment B: integer finite-difference calibration

On a small frozen p10m surface, sample boundary-ready coordinates and blocks
from every parameter group. Compare coarse-gradient predictions with exact
`+1`, `-1`, and proposed-block NLL changes. Separate rescued from naturally
nonzero backward values.

### Experiment C: normalized-wide proposal comparison

Run the already authorized Q15 legacy versus Q23 + Q47 Newton preflight. Before
applying either update, construct their candidate update vectors and measure:

1. probability mass and output-gradient sum;
2. residual and update-vector difference outside common-mode null directions;
3. projection and complete-forward visibility;
4. exact training-acceptance NLL; and
5. frozen development NLL after the predeclared horizon.

This experiment can falsify C1 from the foundations entry and supplies the
first DRTO candidate pair.

### Experiment D: DRTO versus static residual SGD

On the smallest production-shaped profile that passes all numeric contracts,
compare matched initialization and data order:

- current static per-group residual SGD;
- DRTO with one group, one-step candidates; and
- a no-update baseline.

Hold total optimizer steps and exact forward-evaluation budget fixed. Primary
metric: held-out NLL. Secondary metrics: accepted updates, rejected proposals,
function-visible rate, saturation, compute, and replay equality.

## 10. Promotion and falsification

DRTO is supported only if it:

- improves held-out NLL over matched static residual SGD;
- produces a higher useful-update rate, not merely fewer total updates;
- remains zero-saturation under the proof contract;
- replays byte-identically;
- does not access development or test data for update acceptance; and
- reaches held-out NLL below the uniform vocabulary baseline.

The theory is falsified in its current form if exact candidate acceptance fails
to improve held-out NLL at matched compute, if the acceptance surface is
systematically overfit, or if useful progress requires frequent objective-
regressing actions outside the bounded macro-action extension.

## Decision

The next optimizer should not be another manually tuned global shift schedule.
Implement only the instrumentation required to test DRTO's premises first:

1. per-group residual coherence and signed moment summaries;
2. candidate update construction without state mutation;
3. projection-boundary visibility;
4. exact function comparison;
5. canonical training-acceptance NLL; and
6. accept/reject replay hashes.

The existing residual optimizer remains the control. DRTO is a proposed theory,
not yet the default training path.

## Internal evidence

- [`integer-reachable-capacity-v1/matrix.json`](../../benchmarks/integer-reachable-capacity-v1/matrix.json)
- [`integer-reachable-capacity-v1/longitudinal.json`](../../benchmarks/integer-reachable-capacity-v1/longitudinal.json)
- [`p10m-up-functional-comparison.json`](../../benchmarks/production-model-v1/p10m-up-functional-comparison.json)
- [`p10m-up-forward-scale-sensitivity.json`](../../benchmarks/production-model-v1/p10m-up-forward-scale-sensitivity.json)
- [`p10m-up-forward-scale-training.json`](../../benchmarks/production-model-v1/p10m-up-forward-scale-training.json)
- [`p10m-target-probability-resolution.json`](../../benchmarks/production-model-v1/p10m-target-probability-resolution.json)
- [`p10m-probability-normalization-signal-attribution.json`](../../benchmarks/production-model-v1/p10m-probability-normalization-signal-attribution.json)
