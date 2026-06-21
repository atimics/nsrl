# Adaptive Integer Shift Control

NSRL uses integer learning-rate shifts instead of floating-point learning
rates. A lower shift applies a larger update; a higher shift applies a smaller
update. Static shift sweeps are useful for discovery, but they are the wrong
long-term abstraction because training is non-stationary.

Early training needs aggression. Weights are near random, gradients are noisy,
and many accumulated gradients do not cross the i8 update threshold unless the
shift is low enough. Late training needs conservation. The model is closer to a
minimum, and the same shift can overshoot, trigger rollbacks, or cause output
collapse. The correct shift schedule is therefore a curriculum, not a constant.

## Diagnosis

Adaptive shifts are a control problem:

```text
observe training health -> choose shift delta -> observe outcome
```

Useful observations already exist in NSRL traces:

- rollback or rejected-batch rate,
- per-component delta L1,
- gradient saturation count,
- zero-delta count,
- probability-error or bits-per-token movement when available,
- component asymmetry such as dead Q movement while K/V/O move.

The deeper architectural idea is elegant: the controller can use the same
integer linear-attention machinery as the primary model. A tiny meta-state can
bind component statistics to future shift adjustments with i64 outer-product
updates. This would make the learning-rate controller an NSRL-native integer
memory, not an external floating-point optimizer.

## Load-Bearing Problem

The learned holographic controller does not yet have a clean reward signal.
Every obvious reward is compromised:

- `-rollback_count` rewards excessive conservatism.
- `delta_l1` rewards movement, including oscillation.
- BPT improvement is expensive and delayed.
- "Did the previous adjustment help?" requires a counterfactual action that was
  not taken.

Without a reward that is per-batch, correlated with quality, and not directly
gameable by the action, the controller has no reliable thing to learn.

There is also a non-stationarity problem. A plain holographic memory accumulates
all past bindings with equal weight. Early-training advice is actively harmful
late in training. The correct learned controller therefore needs forgetting,
which points toward gated linear attention / RetNet-style decay before it points
toward a larger meta-network.

## Implemented Runnable Experiment

The first version is a simple rule-based controller exposed through
`--adaptive-rule-shifts`. It evaluates component health over an observation
window controlled by `--adaptive-rule-interval-batches`:

```text
if rollback_count over the observation window > 0:
    shift += 1

if saturation_count is sustained over the observation window:
    shift += 1

if zero_delta_count is near-total and no rollback/saturation happened:
    shift -= 1

clamp shift to [min_shift, max_shift]
```

This captures most of the real control surface:

- rollbacks mean the shift is too low,
- near-total zero deltas mean the shift is too high,
- saturation means gradients are too energetic before narrowing.

It has no reward-signal ambiguity, no counterfactual requirement, and no learned
memory staleness. It runs per component: output, MLP, embeddings, attention O/V,
attention K, and attention Q. Q gets peer-aware handling because Q movement is
structurally delayed in causal linear attention; if K is moving and Q is badly
lagging, the controller can lower Q's shift independently.

The trace records:

- `adaptive_rule_shift_adjustment_count`
- `adaptive_rule_update_count`
- `adaptive_rule_event_count`
- `adaptive_shift_events`, with component, reason, previous/next shift,
  observation batches, rejected batches, saturation count, zero-delta count,
  and weight-delta L1.

## Success Criteria

The rule-based controller is worth keeping if it:

- reduces the number of manual sweep runs needed to find a stable schedule,
- improves held-out BPT or accuracy at fixed training windows,
- reduces rollbacks without driving weight movement to zero,
- increases dead-component movement, especially Q, without destabilizing K/V/O,
- and emits deterministic trace rows that replay byte-identically.

These criteria moved enough to justify a first holographic implementation, but
the rule controller remains the safety baseline.

## Holographic Controller

The learned version is now implemented behind `--adaptive-holographic-shifts`.
Each parameter group maintains a small integer memory state and observes its
own Q15 statistics:

```text
s_t = encode(rollback_rate, delta_l1, saturation_ratio, zero_delta_ratio)
delta_shift = query(memory, s_t)
memory <- memory + bind(teacher_{t+1}, s_t)
```

The binding is intentionally lagged by one observation: state `s_t` is stored,
and only the next batch's teacher signal is bound to it. This makes the memory a
predictor of the next control decision rather than a same-step echo of the
rule. When the explicit teacher is nonzero, it still overrides recall. When the
teacher is silent, recalled memory may apply a conservative shift delta in
`[-1, 1]`.

This makes the holographic controller reachable in ordinary training states,
not only in already-dead components. The current implementation also adds
authority gates:

- recalled memory needs a minimum amount of observed history before acting,
- accepted-batch holographic adjustments are cooldown-limited per actuator,
- in combined rule+holographic mode, the rule controller owns nonzero teacher
  actions and holographic memory acts only when the teacher is silent.

The cost is that memory authority still matters. A reachable memory can improve
short smokes but over-act over longer runs unless it gains forgetting or
confidence gating.

The remaining work is to add an integer forgetting primitive for stale
meta-experience, most likely through gated linear attention / retention.
