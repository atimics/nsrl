# Quantized Optimization and Reachable Capacity

This note connects established low-precision optimization research to NSRL's
observed integer-training behavior. It deliberately separates prior art from a
new hypothesis.

## Established results

### Rounding can decide whether learning happens

[Gupta et al. (2015)](https://proceedings.mlr.press/v37/gupta15.html)
showed that 16-bit fixed-point training could preserve accuracy when paired with
stochastic rounding. Deterministic round-to-nearest can erase an update whose
magnitude stays below half a quantization cell.

NSRL uses deterministic arithmetic, so ordinary stochastic rounding is not the
default escape hatch. Wide batch accumulation and residual carry instead allow
sub-cell evidence to persist across samples or optimizer steps.

### Error feedback is established prior art

[Karimireddy et al. (2019)](https://proceedings.mlr.press/v97/karimireddy19a.html)
showed that storing compression error and adding it to the next update repairs a
large class of biased gradient compressors. Earlier communication-compression
work also used accumulated quantization error.

NSRL's carried integer residuals are best described as an integer optimizer
application of error feedback. The repository can claim a specific deterministic
implementation and empirical result, not invention of error feedback itself.

### Quantized training can lose a search phase

[Li et al. (2017)](https://proceedings.neurips.cc/paper/7163-training-quantized-nets-a-deeper-understanding)
argue that algorithms with high-precision latent weights have a greedy search
phase unavailable to purely quantized updates. This is a useful interpretation
of the observed i8 cliff:

- one shift produces a whole-cell update that overshoots;
- the next shift produces no parameter update;
- neither action can explore the interval between them.

### Quantized weights can oscillate

[Nagel et al. (2022)](https://proceedings.mlr.press/v162/nagel22a.html)
document weights oscillating between adjacent quantization grid points during
QAT. [Lee et al. (2022)](https://openreview.net/forum?id=3HJOA-1hb0e)
use hysteresis to make leaving a cell harder than staying in it.

These results matter even though they retain floating-point latent weights:
they show that quantization creates discrete state transitions with dynamics
that parameter count alone does not describe.

### Precision should be allocated by role

[Banner et al. (2018)](https://proceedings.neurips.cc/paper/2018/hash/e82c4b19b8151ddc25d4d93baf7b908f-Abstract.html),
[AMPA](https://proceedings.mlr.press/v235/ding24b.html), and
[Jetfire](https://openreview.net/forum?id=ltzTHGFF5i) independently support the
same broad conclusion: weights, activations, input gradients, weight gradients,
and different layers do not need the same numeric format.

NSRL's p10m result makes this concrete. K saturation dominated the parameter
saturation count, so globally widening or slowing every group would spend
precision where it is not needed.

The later 2,048-window K/V readiness run strengthens the role-allocation
argument. K and V crossed update boundaries in every chunk at shifts 26 and 30
without saturation, while ten other groups accumulated nonzero residuals but
did not move. Exact final residual inspection selects the gated-MLP `gate`
projection at shift 23 as the smallest isolated next action, with only 12
predicted crossings. Validation was not monotone across integer chunks, so
boundary reachability remains a safety/prioritization signal rather than a
claim that every additional update improves quality.

The isolated follow-up confirms the boundary policy while clarifying its unit
of prediction. Lowering only `gate` from shift 25 to 23 caused its first 7
updates at window 768 and 26 cumulative updates by window 2,048, with no new
moving groups, saturation, or held-out endpoint change relative to the K/V
source. The original 12-crossing estimate described one residual snapshot, not
cumulative crossings across later optimizer chunks. Final residual state then
selects `up` at shift 23 with 6 predicted crossings, preserving the policy of
one bounded role-specific change at a time.

The next two experiments falsify the stronger interpretation of that policy.
At shift 23, `up` made 26 safe updates but did not beat its source. At shift 22,
it made 101,543 safe updates—about 3,900 times as many—yet the selected dev
checkpoint only tied and the one-shot test was worse by 1,245 total millibits.
The matched window-1,024 functional comparison is more specific: despite
different model hashes and roughly 50,825 versus 3 cumulative `up` updates,
the two models produced identical final features, logits, probabilities, and
per-window losses on all 256 dev windows. Here reachable parameter capacity did
not become reachable functional capacity because forward quantization masked
the changed weights before the final feature boundary.

The follow-up scale sweep locates the next boundary. Common `up` forward shifts
10 through 8 preserve the complete mask; shift 7 changes 250 of 256 final
feature/logit vectors and 124 probability vectors without saturation. Yet it
changes the target probability on zero windows, and fresh 1,024-window training
at that scale still ties source dev after 50,568 exact `up` updates. Reachable
functional capacity therefore remains weaker than reachable objective capacity:
the next precision allocation question is the Q15 target probability emitted
by an 8,192-way softmax near the uniform regime.

That audit now separates observation precision from update precision. Across
the identical frozen logits, Q15 represents source targets with only three
values and hides all target deltas. Q19 exposes one changed target, while Q23
exposes 13 and matches the support visible at Q27/Q31. This confirms that Q15
was hiding objective variation. It does not establish useful gradient
direction: the wider comparison is slightly worse overall at Q23 (5,497 total
microbits), with six improved and seven worsened windows.

Scale-compensated Q19 and Q23 training then preserve the effective learning
rate rather than multiplying it by the extra fractional bits. After 256
windows both lanes are zero-saturation and end at exactly the Q15 model bytes
and dev loss, although their optimizer artifacts differ. Wider probability
information is therefore reachable in residual state but not yet reachable as
a distinct parameter function. The audit also exposes maximum probability-mass
error near ten percent at wider precision, localizing the next experiment to
reciprocal normalization accuracy rather than another undirected precision or
trunk sweep.

## NSRL hypothesis: reachable capacity

Define a training contract

```text
C = (architecture, initialization, data order, objective,
     dtypes, scales, rounding, batch geometry, optimizer state)
```

and let `U_C(g, s)` be the exact parameter update produced from gradient-like
signal `g` and optimizer state `s`. Two nominal architectures are
**update-equivalent for an experiment** when their mapped updates induce the
same function over the frozen evaluation surface, even if their parameter
counts differ.

A stricter and cheaper diagnostic is **update identity**: corresponding update
tensors have the same shape-aware hash after one or more matched steps.

The proposed quantity of interest is:

> **Reachable capacity**: the number or structure of distinct useful model
> functions reachable under a fixed discrete training contract and budget.

This is not yet a formal capacity measure and is not established by the cited
literature. It is motivated by NSRL observations:

- some rank-4 and rank-8 residual projections produced zero updates;
- a rank-32 probe produced the same integer update as rank 16;
- direct i8 continuation alternated between destructive whole-cell movement and
  no movement;
- i16 residual paths exposed useful improvements while preserving a frozen i8
  trunk.

## Measurement protocol

For every architecture/optimizer probe, record:

1. pre-quantization gradient or product magnitude summaries;
2. residual/carry magnitude before and after the step;
3. zero-update count and fraction by tensor;
4. saturation count and fraction by tensor;
5. update `L1`, `L∞`, nonzero count, and shape-aware hash;
6. numerical rank or a deterministic rank surrogate for matrix updates;
7. model hash after the update;
8. exact train-objective delta;
9. frozen held-out objective delta;
10. whether a smaller candidate produced an identical embedded update.

An architecture increase has **not exposed capacity** when its additional
parameters receive gradients but all additional updates remain zero, repeat a
smaller candidate's update, or fail to change the frozen evaluation function.

## First falsifiable experiment

Use the existing rank-16/rank-32 residual pair and hold initialization,
projection seed, data order, and objective constant. Sweep only:

- accumulator width;
- batch windows;
- update shift;
- residual error feedback on/off;
- expansion dtype (`i8` versus `i16`);
- deterministic round-half-up versus seeded stochastic rounding as an offline
  control.

The hypothesis predicts a phase diagram with regions such as:

```text
both dead -> identical nonzero update -> distinct updates -> saturation
```

The most important test is whether error feedback moves rank 32 from the
identical region into the distinct region and whether that distinction predicts
held-out improvement.

## First bounded result

The repository now runs a 30-cell rank × shift × carry matrix with ranks
8/16/32, shifts 0–4, and residual error feedback on/off. Every cell records an
exact parameter-update fingerprint plus a function-level fingerprint over the
same 256 frozen hidden states. The frozen report is
`benchmarks/integer-reachable-capacity-v1/matrix.json`.

- 30 nominal configurations produced only 15 distinct functional updates.
- 14 configurations were exact functional no-ops.
- Without carry, every rank was dead by shift 2.
- Carry exposed nonzero updates through shift 2 for all ranks and through shift
  3 for ranks 16 and 32.
- Rank 32 and rank 16 were functionally identical at shift 1 without carry and
  at shift 3 with carry.
- At shift 0 with carry, rank 32 was distinct but finished 17,247 Q15 worse
  than rank 16.

This supports the narrow claim that nominal rank, optimizer scale, and error
feedback jointly determine which updates are reachable. It does not establish
a general capacity law: distinctness sometimes helps, sometimes does nothing,
and in one matched pair hurts the bounded training objective.

## Longitudinal held-out result

The predeclared follow-up keeps the same 30 configurations, expands training
from 256 to 2,048 windows (8 to 64 optimizer steps), and evaluates every final
expert on 4,096 windows from the separately materialized literary holdout. The
contract and frozen result are
`benchmarks/integer-reachable-capacity-v1/longitudinal-contract.json` and
`benchmarks/integer-reachable-capacity-v1/longitudinal.json`.

- All 16 cells with an early nonzero functional update later improved held-out
  probability error; there were no false positives.
- Six of 14 early functional no-ops activated later and improved, yielding
  precision 1.0, recall 0.727, accuracy 0.8, and MCC 0.645.
- Early functional-update L1 correlated with held-out gain at Spearman ρ 0.828;
  a deterministic 10,000-permutation one-sided test gave p = 0.0001.
- Early-reachable cells averaged 12,759,487 Q15 more held-out gain than early
  no-op cells.
- All 16 early-reachable long runs recorded weight or hidden saturation, and
  20 of 30 cells saturated overall.

The result supports early reachability as a conservative prioritization signal,
not as a sufficient optimizer-health or pruning rule. A no-op after eight steps
does not prove a candidate is dead, while a reachable update does not prove its
longer trajectory is saturation-free or functionally visible. The p10m `up`
comparison supplies the direct counterexample: tens of thousands of safe weight
updates can collapse to zero observable feature change under the frozen forward
scales.

## Falsifiers

The reachable-capacity framing should be rejected or narrowed if:

- update hashes differ but never predict objective or function differences;
- effective update rank adds no predictive power beyond zero-update and
  saturation counts;
- rank-16/rank-32 identity was an isolated initialization artifact;
- float and integer controls show the same equivalence at the same frequency;
- or extra integer update diversity consistently harms held-out quality.

Even under those outcomes, the instrumentation remains useful as optimizer
health evidence.

## Roadmap consequence

Do not approve a larger paid run merely because it has more parameters. Require
a bounded preflight showing that the extra parameter groups produce distinct,
non-saturating updates and improve a frozen objective. Use early fingerprints
to prioritize the queue, retain a longer delayed-activation check for apparent
no-ops, and reject trajectories whose eventual gain depends on unchecked
saturation. Scaling follows exposed functional capacity; it does not substitute
for it. The p10m forward-scale boundary is now measured at `up` shift 7; the
next gate must audit target-probability resolution before spending more trunk
update resolution. The probability audit and compensated wide-gradient
preflight are now complete; the next gate is normalization accuracy because
Q19/Q23 information remains residual-only at 256 windows.
