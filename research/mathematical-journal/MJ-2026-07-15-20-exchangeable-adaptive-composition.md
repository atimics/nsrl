# MJ-2026-07-15-20: Exchangeable adaptive composition and its finite-sample price

- Date: 2026-07-15
- Status: MJ-19 conditional-null bridge falsified; bounded-horizon
  exchangeability-valid composition theorem established; the preregistered
  persistent optimizer execution is falsified by a zero-fire exact replay
- Supersedes: the interpretation of a non-crossing MJ-19 unsafe-action
  e-process as affirmative sequential support
- Extends: MJ-2026-07-15-16 through MJ-2026-07-15-19
- Executable theory binding:
  [`check-adaptive-composition-theory-v1.mjs`](../../scripts/check-adaptive-composition-theory-v1.mjs)
- Preregistration:
  [`p10m-adaptive-composition-v1-preregistration.json`](../../protocol/examples/p10m-adaptive-composition-v1-preregistration.json)
- Execution publication:
  [`p10m-adaptive-composition-v1-publication.json`](../../benchmarks/production-model-v1/p10m-adaptive-composition-v1-publication.json)

## Question

Can marginal source-panel conditional-exchange certificates be composed into
a persistent, history-dependent integer optimizer with an anytime-valid harm
guarantee, without assuming the conditional unsafe intensity that MJ-19
explicitly left unproved?

The answer has two parts.

1. **No, not by the MJ-19 likelihood-ratio argument.** Marginal exchangeable
   coverage does not imply its conditional null. Failure to cross that
   e-process is not evidence for the null.
2. **Yes, on a finite predeclared horizon, at an explicit sample price.** A
   simultaneous conformal score over every reachable state, physical action,
   and panel passage, combined with predeclared alpha spending, gives a
   time-uniform zero-positive-regret event by a union bound. This requires no
   independence between evaluation rounds and permits history-dependent choice
   among noncommuting actions.

The second result is intentionally bounded. It is not an indefinite optimizer
theorem and it does not make a fixed 19-panel calibration reusable forever.

## 1. Persistent state and physical actions

Let `theta_0` be the retained integer model. Let `A={H,T}` contain two
physically distinct proposal families and let `0` denote abstention. For every
reachable state `theta`:

- `H(theta)` is a frozen, state-specific output-head lattice-exit action whose
  writes are restricted to `output` or `bias`;
- `T(theta)` is a frozen, state-specific trunk lattice-exit action whose writes
  are restricted to parameter groups before `output`; and
- `0(theta)=theta`.

The concrete coordinate lists and directions must be selected using proper
fitting sources only, before calibration and evaluation outcomes are opened.
The two action families are genuinely distinct only if their ordered delta
fingerprints and affected group sets differ.

Each source panel has a frozen order of passages. At a passage, the controller
computes the two fitting-fixed certified upper contrasts at the current state,
chooses at most one eligible physical action by the frozen tie-break, or
abstains, and immediately persists an accepted action before the next passage.
After `K` accepts, every remaining passage and panel must abstain. Thus the
first experiment has six source-panel error-control rounds and 24 ordered
passage decisions, but no more than two persistent model transitions.

**Definition 20.1 (persistent transition).** If `a_j` is the `j`th accepted
physical action (passage and source-panel abstentions do not increment `j`),

\[
  \theta_j=T_{a_j}(\theta_{j-1}).
\]

There is no rollback before the next passage. The stored model and function
hashes before and after every accepted action are part of the trace.

**Definition 20.2 (bounded reachable set).** For a maximum of `K` accepted
actions, let

\[
  \mathcal S_K=\{T_{a_j}\circ\cdots\circ T_{a_1}(\theta_0):
                    0\le j\le K,\ a_i\in\{H,T\}\}.
\]

The preregistered first experiment uses `K=2`, so it must bind at most the seven
labelled paths

```text
empty, H, T, HH, HT, TH, TT.
```

Path labels are chronological: `HT` means accept `H` first and `T` second,
so its state is `T_T(T_H(theta_0))`.

Equal model states may be deduplicated for execution, but their path aliases
remain in the contract.

**Definition 20.3 (noncommutativity witness).** The physical proposal
families are noncommuting on the bound substrate when

\[
  \theta_{HT}=T_T(T_H(\theta_0))
  \ne T_H(T_T(\theta_0))=\theta_{TH}
\]

in both serialized model hash and a frozen function hash. If either comparison
ties, the noncommuting-composition hypothesis is falsified before evaluation;
renaming two commuting masks does not pass.

## 2. Why the MJ-19 conditional bridge fails

MJ-19 defines a source-panel unsafe indicator `Y_t` and assumes

\[
  E[Y_t\mid\mathcal F_{t-1}]\le 1/20.
\]

Its e-process is valid under that assumption. The assumption does not follow
from marginal exchangeable coverage.

**Proposition 20.1 (exact six-panel exchangeable counterexample).** There is
an exchangeable sequence `(Y_1,...,Y_6)` such that

\[
  Pr(Y_t=1)=1/20\quad\text{for every }t,
\]

but

\[
  Pr(Y_6=1\mid Y_1=\cdots=Y_5=0)=1/15>1/20.
\]

**Proof.** Assign probability `7/10` to the all-zero vector and probability
`1/20` to each of the six vectors having exactly one unsafe coordinate. The
probabilities sum to one, the law is invariant under coordinate permutations,
and every coordinate is unsafe with probability `1/20`. The event that the
first five coordinates are safe has probability `7/10+1/20=3/4`; its
intersection with `Y_6=1` has probability `1/20`. The conditional probability
is therefore `(1/20)/(3/4)=1/15`. `square`

On that history, the MJ-19 unsafe multiplier is `5` and its safe multiplier is
`15/19`, so its conditional expected multiplier is

\[
  (1/15)5+(14/15)(15/19)=61/57>1.
\]

The process is not a supermartingale under marginal exchangeability alone,
even over MJ-19's actual six-panel horizon.

**Decision 20.1.** MJ-19's empirical frozen-frame result remains an exact
record of eight favorable firings. Its maximum e-value of one is an absence of
an alarm under an additional null, not affirmative evidence that the null is
true. Future publication schemas must distinguish:

- `alarm_not_crossed`;
- `conditional_null_assumed`; and
- `conditional_null_supported_by_design`.

Only the last can support a sequential safety theorem.

## 3. Simultaneous state-action conformal score

For source panel `u`, passage `d`, reachable state `s`, and non-abstaining
action `a`, let the exact current-state contrast be

\[
  \Delta_{u,d}(s,a)
   =L_{u,d}(T_a(s))-L_{u,d}(s),
\]

where `L` is the declared Q32 observation objective. Decompose it using a
proper-fitting predictor:

\[
  \Delta_{u,d}(s,a)=\lambda_{u,d}(s,a)
                   +q_{s,a}(\phi_{u,d})
                   +r_{u,d}(s,a).
\]

The calibration unit is a whole source panel. Define

\[
  A_u=\max_{s\in\mathcal S_K}
      \max_{a\in\{H,T\}}
      \max_{d\in P_u} r_{u,d}(s,a).
\]

This maximum is essential. A score computed only at `theta_0` does not cover
the state reached after a persistent action. Separate per-action corrections
do not cover adaptive action selection unless their error budgets are also
adjusted.

For family `f`, let `A_1,...,A_n` be the fitting-fixed calibration scores and
let `epsilon_t` be a predeclared error spend for evaluation round `t`. Define

\[
  k_t=\lceil(n+1)(1-\epsilon_t)\rceil,
\]

and let `Q_{f,t}` be the `k_t`th calibration order statistic when `k_t<=n`, or
`+infinity` otherwise.

At current state `s_t`, action `a` may fire on passage `d` only when

\[
  \lambda_{t,d}(s_t,a)+q_{s_t,a}(\phi_{t,d})+Q_{f,t}<0.
\]

If both actions are eligible, select the smaller certified upper contrast,
then the lower write cost, then the frozen action-family ID. No current
multi-action outcome is a controller input.

## 4. Exchangeability-valid anytime guarantee

**Assumption E20 (explicit source-panel exchangeability).** Conditional on the
proper-fitting artifacts, define each complete potential-outcome and predictor
cube as

\[
  \mathcal Z_u=
  \{(\Delta_{u,d}(s,a),\lambda_{u,d}(s,a),\phi_{u,d},r_{u,d}(s,a)):
    s\in\mathcal S_K,a\in\{H,T\},d\in P_u\}.
\]

For each fresh evaluation source `t`, the `n` calibration cubes together with
`Z_t` are exchangeable within the declared source family. Equivalently, the
raw panels and every fitting-fixed feature map used to derive these cubes must
be permutation-equivariant. The source frame and order do not depend on
action-cube outcomes. Joint exchangeability or independence among the several
fresh evaluation sources is not required.

This is stronger data acquisition than observing the selected action alone:
calibration must evaluate every bound state-action branch. It is still a
marginal exchangeability assumption, not a conditional unsafe-intensity null.

**Proposition 20.2 (one-round adaptive selection safety).** Under E20, for
every source-panel round `t`,

\[
  Pr(\text{some fired passage action in panel }t\text{ is non-improving})
  \le\epsilon_t.
\]

**Proof.** The maximum defining `A_u` is the same permutation-invariant map of
every cube, so E20 makes the `n` calibration scores and `A_t` exchangeable.
Split conformal rank coverage gives
`Pr(A_t>Q_{f,t})<=epsilon_t`. On `A_t<=Q_{f,t}`, the correction simultaneously
bounds every state, action, and passage in the panel. Therefore every passage
action satisfying the strict firing rule has `Delta<0`, including actions at
states reached by earlier passages in the same panel or by prior panels. Thus
any unsafe firing in panel `t` implies `A_t>Q_{f,t}`. `square`

**Theorem 20.3 (bounded-horizon anytime zero-harm guarantee).** For a
predeclared horizon `T`, if

\[
  \sum_{t=1}^T\epsilon_t\le\alpha,
\]

then, without independence between evaluation rounds,

\[
  Pr(\exists t\le T:\text{unsafe firing in panel }t)\le\alpha.
\]

Consequently, with probability at least `1-alpha`, simultaneously for every
prefix `r<=T`,

\[
  G_r^+=\sum_{t\le r}\sum_{d\in P_t}
  1\{\text{action accepted at }d\}[\Delta_{t,d}]_+=0.
\]

**Proof.** Proposition 20.2 bounds each unsafe event by `epsilon_t`.
The union bound gives the first statement for arbitrary dependence. On its
complement every fired contrast is strictly negative, so every positive part
and every prefix sum is zero. `square`

This theorem is anytime-valid over the frozen finite horizon because it holds
simultaneously at all prefixes. It does not claim validity after `T` or after
the reachable-state/action manifest changes.

## 5. The finite-sample price

The conformal correction is finite only when `k_t<=n`, equivalently

\[
  \epsilon_t\ge\frac1{n+1}.
\]

With equal spending `epsilon_t=alpha/T`, nonvacuity requires

\[
  n\ge\left\lceil\frac T\alpha\right\rceil-1.
\]

For the preregistered `T=6` and `alpha=1/20`, each family needs at least

\[
  n=119
\]

calibration source panels for a finite rank-119 correction at per-round spend
`1/120`. The earlier 19-panel family calibrations can support one marginal 5%
decision, not six globally 5%-safe adaptive rounds.

For 19 evaluation sources in each of three families under a global 5% budget,
equal spending would require 1,139 calibration source panels per family. This
is a theorem-implied acquisition cost, not a recommendation to hide the cost
with the MJ-19 conditional null.

## 6. Canonical retained-model comparison

Per-panel negative contrasts are safety evidence, not the endpoint optimizer
claim. Persistent composition must be evaluated on a sealed endpoint source
surface `E*` that is not used for fitting, calibration, action selection, or
early stopping.

Using the identical source order and action manifests, replay four trajectories:

1. `pi`: the adaptive certified policy;
2. `0`: always abstain, retaining `theta_0`;
3. `H`: select only the head family whenever its certificate fires; and
4. `T`: select only the trunk family whenever its certificate fires.

Let `theta_T^pi`, `theta_T^0`, `theta_T^H`, and `theta_T^T` be their retained
endpoints. The primary estimand is normalization-independent canonical integer
base-2 NLL in total millibits on `E*`, implemented by
`evaluate_production_model_canonical_nll` with objective ID
`integer_base2_softmax_nll_millibits`, aggregation field
`total_nll_millibits`, and a 32,000-millibit zero-probability floor.

**Definition 20.4 (adaptive composition promotion).** The bounded optimizer
result is supported only if all of the following are true:

```text
NLL_E*(theta_T^pi) < NLL_E*(theta_T^0)
NLL_E*(theta_T^pi) < min(NLL_E*(theta_T^H), NLL_E*(theta_T^T))
observed cumulative positive regret = 0
at least one action from each physical family fires
theta_T^pi is persisted and byte-replays from the same ordered source stream
zero-probability windows do not increase relative to theta_T^0
```

Calibration success, a non-crossing alarm, or improvement over abstention alone
cannot substitute for the best-fixed-family comparison.

## 7. Preregistered experiment M5

The machine-readable preregistration fixes the following design before source
acquisition:

- model profile: p10m;
- physical action families: state-specific `head_lattice_exit` and
  `trunk_lattice_exit`, plus abstention;
- maximum accepted actions: two;
- reachable labelled paths: seven;
- noncommutativity gate: both model and function hashes for `HT` and `TH` must
  differ before calibration outcomes open;
- source families: Federal Register, RFC, and open-access science;
- proper fitting: 12 whole-publication panels per family;
- simultaneous calibration: 119 panels per family;
- adaptive horizon: two panels per family, six globally;
- sealed endpoint comparison: 19 panels per family;
- four nonoverlapping passages and two 64-token-context targets per source;
- global alpha: `1/20`, spent as `1/120` per adaptive source round;
- observation objective for certificates: Q47-weight Q32 NLL;
- primary endpoint: canonical normalization-independent total NLL millibits;
- controls: always abstain, head-only policy, and trunk-only policy; and
- no optimizer or paid-scaling authorization from calibration alone.

All M18 and M19 source IDs and independence keys are excluded. If the required
fresh calibration count cannot be acquired, the experiment is
`inconclusive_insufficient_exchangeable_calibration`, not silently downgraded
to a marginal or conditionally assumed guarantee.

## Falsifiers

The adaptive composition claim is falsified if any of these occurs:

1. the two physical action families have equal delta fingerprints or equal
   affected group sets;
2. `HT` and `TH` have equal model or function hashes;
3. an executed state leaves the bound reachable set;
4. any selected action uses a current candidate outcome, endpoint source, or
   post-fitting calibration outcome;
5. any fired action has nonnegative exact passage contrast;
6. the retained adaptive endpoint fails to beat always abstain strictly in
   canonical NLL;
7. the retained adaptive endpoint fails to beat the best fixed physical action
   policy strictly;
8. zero-probability windows increase; or
9. replay changes any state, action, source-order, or endpoint hash.

Failure to acquire 119 exchangeable calibration panels per family is
inconclusive rather than falsifying. Failure of the noncommutativity gate is a
falsification of this experiment's composition hypothesis, not evidence that
commuting optimizers are impossible.

## Numeric correction to MJ-19

MJ-19's 60-bit window bound remains valid, but its proof omitted the
annihilated-target branch. If the Q47 target weight is nonzero, it is at least
one and the denominator is below `2^60`, so NLL is at most 60 bits. If the
target weight is zero, the declared evaluator returns the 32-bit floor, which
is also at most 60 bits. Two-window passage loss is therefore in `[0,120]`
bits and the absolute difference of two such losses is at most 120 bits.

## 8. Frozen M5 execution result

The full preregistered source design was executed without changing its
rank-119 corrections or selection threshold after opening adaptive outcomes.
Each of Federal Register, RFC, and science contributed 12 proper-fitting, 119
calibration, two adaptive, and 19 sealed-endpoint whole-publication panels.
The calibration cube contained 19,992 state-action rows over 357 calibration
source panels.

The simultaneous corrections were 16,785,479 Q32 for Federal Register,
16,837,634 Q32 for RFC, and 16,698,609 Q32 for science. None of the 24 ordered
adaptive passage decisions had a certified upper contrast below zero. The
adaptive, always-abstain, head-only, and trunk-only trajectories therefore all
retained the empty state with canonical endpoint NLL 5,930,001 millibits and
zero zero-probability windows.

**Decision 20.2.** The empirical M5 optimizer claim is falsified. It failed
the frozen nonvacuity, strict-abstention-improvement, strict-best-fixed-policy,
and both-physical-families gates. Zero observed positive regret is vacuous
because no action fired. The theorem remains valid; this proposal family did
not produce useful certified actions at its theorem-implied finite-sample
price.

Calibration, decisions, retained models, and the result JSON replayed byte for
byte. The tracked replay receipt binds that one-time full replay without
checking 194 MB of ignored corpus and model intermediates into Git. The
publication is fail-closed and explicitly forbids post-outcome threshold
retuning, optimizer promotion, paid scaling, successor mutation, or product
release.

## Decision

The MJ-19 conditional-null bridge is falsified under marginal exchangeability.
Do not report a non-crossing unsafe-action e-process as support for safety.

Theorem 20.3 is accepted as the bounded replacement: simultaneous
state-action-source conformalization plus alpha spending gives a valid
finite-horizon adaptive composition guarantee. Its price is 119 calibration
source panels per family for the six-round M5 design.

M5 has now been executed and falsified. Its state-specific action manifests,
noncommutativity witness, fresh source frame, full calibration count, sealed
endpoint, and exact replay are preserved as negative evidence. Optimizer
mutation and paid scaling remain unauthorized. A new proposal must be frozen
prospectively; the M5 correction or firing threshold must not be tuned after
this result.

## Open work

- Preserve M5 as a terminal falsification; do not retune its correction or
  threshold.
- Run E21-A on the small transformer before proposing any production optimizer
  change.
- Run the product-facing E21-D pair-geometry experiment against the frozen
  72-identity multimodal split.
- Replace the union-bound design only if a stronger theorem derives from an
  equally explicit data-generating assumption; do not reintroduce an assumed
  conditional hazard under another name.
