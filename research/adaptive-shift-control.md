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

## Next Runnable Experiment

Implement the simple rule-based controller first:

```text
if rollback_count over the last N batches > K:
    shift += 1

if zero_delta_count over the last N batches > J:
    shift -= 1

if saturation_count over the last N batches > S:
    shift += 1

clamp shift to [min_shift, max_shift]
```

This captures most of the real control surface:

- rollbacks mean the shift is too low,
- zero deltas mean the shift is too high,
- saturation means gradients are too energetic before narrowing.

It has no reward-signal ambiguity, no counterfactual requirement, and no learned
memory staleness. It should be tested per component: output, MLP, embeddings,
attention O/V/K/Q, with Q allowed a more aggressive floor because Q movement is
structurally delayed in causal linear attention.

## Success Criteria

The rule-based controller is worth keeping if it:

- reduces the number of manual sweep runs needed to find a stable schedule,
- improves held-out BPT or accuracy at fixed training windows,
- reduces rollbacks without driving weight movement to zero,
- increases dead-component movement, especially Q, without destabilizing K/V/O,
- and emits deterministic trace rows that replay byte-identically.

Only if these criteria move should the holographic controller be promoted from
research note to implementation target.

## Holographic Version, Later

The learned version remains interesting. Each parameter group could maintain a
small integer memory state and observe its own Q15 statistics:

```text
s_t = encode(rollback_rate, delta_l1, saturation_ratio, zero_delta_ratio)
delta_shift = query(memory, s_t)
memory <- gated_decay(memory) + bind(reward_signal, s_t)
```

But this should wait until two things are true:

1. The rule-based controller proves adaptive shifts help.
2. NSRL has an integer forgetting primitive for stale meta-experience.

The insight is real. The timing is premature.
