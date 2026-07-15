# Test-Time Training and Inference-Time Adaptation

Test-time training (TTT) covers several different mechanisms. This note keeps
them separate because they imply different integer implementations.

## Three mechanisms

### Parameter adaptation on retrieved data

[Test-Time Training on Nearest Neighbors for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html)
(Hardt & Sun, ICLR 2024) retrieves nearby sequences and applies ordinary
fine-tuning steps before evaluating an input. It reports improvements across
Pile domains with as few as 20 retrieved neighbors.

This is genuine parameter adaptation. It requires a retrieval index, a backward
pass, optimizer state, and a policy for retaining or reverting the update. A
[2025 reproducibility report](https://arxiv.org/abs/2511.16691) supports the
central result on additional model families.

### Small sample-specific residuals

[SLOT](https://arxiv.org/abs/2505.12392) (Hu et al., 2025) caches final-layer
features and optimizes a small sample-specific vector from the prompt. This is
closer to NSRL's successful frozen-trunk residual strategy than mutating the
whole trunk at inference.

### Trainable recurrent state

[Learning to (Learn at Test Time)](https://proceedings.mlr.press/v267/sun25h.html)
(Sun et al., ICML 2025) uses a model as the recurrent hidden state and updates
that model with a self-supervised objective while processing the sequence.

[Test-Time Training with KV Binding Is Secretly Linear Attention](https://arxiv.org/abs/2602.21204)
(Liu et al., ICML 2026) shows that a broad class of KV-binding TTT architectures
can be rewritten as learned linear-attention operators. The result covers more
than a single linear fast-weight layer and explains why several apparently
different TTT updates admit parallel forms.

The correct implication is narrower than the repository's previous wording:

> Some structured KV-binding TTT layers can be represented as learned linear
> attention. This does not make every form of test-time parameter optimization
> equivalent to NSRL's existing additive KV accumulator.

The equivalence depends on the inner model, loss, update rule, and the layer
being adapted.

## Earlier fast-weight connection

[Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)
(Schlag et al., ICML 2021) already established that linear attention's
outer-product state can be interpreted as fast weights. It also introduced a
delta-rule update that corrects an existing key/value association rather than
only adding another association.

For NSRL, the additive state

```text
S_t = S_{t-1} + k_t outer_product v_t
```

is therefore best described as an integer fast-weight memory. Calling it
“training” is reasonable under the fast-weight interpretation, but it is not a
substitute for an experiment with an explicit self-supervised TTT objective.

## NSRL opportunities

### 1. Transactional residual adaptation

Freeze the trunk, adapt a small `i16` residual, evaluate an exact guard set, and
commit or revert the update. This bounds optimizer memory and avoids the known
i8 trunk cliff.

Required evidence:

- update and revert are byte-identical;
- the guard set is disjoint from the adapted prompt tokens;
- the residual improves the target task without degrading the guard set;
- the same prompt and seed produce the same artifact hash.

### 2. Integer delta-rule memory

Compare additive linear-attention state with a bounded integer delta-rule
correction. Measure retrieval accuracy, state saturation, stale-memory error,
and cross-platform replay. This is a state-update experiment, not ordinary
backpropagation.

### 3. Retrieval-conditioned adaptation

Reproduce the nearest-neighbor TTT protocol first with a float reference, then
with a frozen integer trunk plus residual adapter. Keep retrieval-only,
adaptation-only, and combined rows separate.

### 4. Prompt-only adaptation

Use a SLOT-like residual vector as the smallest test of sample-specific integer
optimization. A full-trunk update is unjustified until the small residual has a
measured ceiling.

## Risks

- **Self-reinforcement:** optimizing on the prompt can increase confidence
  without improving correctness.
- **Catastrophic local updates:** exact replay makes a bad update reproducible,
  not safe.
- **State contamination:** persistent fast weights can leak information between
  requests unless session boundaries are explicit.
- **Metric leakage:** selecting an update on the same tokens used to report the
  gain is not held-out evidence.
- **Optimizer memory:** integer Adam or per-parameter residual carry can exceed
  the inference model size even though no state is floating point.

## Recommended order

1. Additive state versus delta-rule state on a synthetic binding task.
2. Frozen-trunk prompt residual with transactional rollback.
3. Retrieval-conditioned residual adaptation with matched ablations.
4. Only then test persistent session learning or full-trunk TTT.
