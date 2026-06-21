# Linear & Subquadratic Attention

Papers on attention mechanisms that reduce or eliminate the O(n²) complexity of standard softmax attention.

---

## Foundational Linear Attention

### Katharopoulos et al. (2020) — Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- **arXiv**: https://arxiv.org/abs/2006.16236
- **Key insight**: Replace softmax(QKᵀ)V with φ(Q)(φ(K)ᵀV) using the associativity of matrix multiplication. Complexity drops from O(n²d) to O(nd²). The feature map φ can be as simple as ReLU or identity+shift.
- **Recurrent form**: At inference, maintain a running state S = Σ φ(kₜ)⊗vₜ and normalization z = Σ φ(kₜ). Each new token: O(d²) update, O(d²) query. Constant memory regardless of context length.
- **Relevance to NSRL**: This is the paper that defines the computation NSRL now has implemented. The `linear_attention_i16_q15_checked` primitive uses φ(x) = x (identity, ReLU-shifted), making kₜ⊗vₜ an i8×i16→i32 outer product, accumulated in i64.
- **Quality gap**: Linear attention underperforms softmax on selective retrieval tasks. The feature map choice matters.

---

## Hardware-Efficient & Gated Variants

### Gated Linear Attention (2023) — GLA Transformers with Hardware-Efficient Training
- **arXiv**: https://arxiv.org/abs/2312.06635
- **Authors**: Yang, Wang, Shen, Panda, Kim (MIT/IBM)
- **Key insight**: Add data-dependent scalar or vector gates to the linear attention state update: S_t = G_t ⊙ S_{t-1} + kₜ⊗vₜ. Gates modulate how much history to retain per token. Implemented with FlashLinearAttention — faster than FlashAttention-2 even on short sequences.
- **Relevance**: The gating mechanism is the main quality improvement over vanilla linear attention. Integer-friendly: scalar gate per head could be a Q15 multiplier, making state decay a right-shift operation.
- **Quality**: Closes most of the gap with softmax attention on language modeling benchmarks.

### Parallelizing Linear Transformers with the Delta Rule (2024)
- **arXiv**: https://arxiv.org/abs/2406.06484
- **Key insight**: Delta rule update (S_t = S_{t-1} + β_t(vₜ - kₜᵀS_{t-1})kₜ) gives the state associative memory semantics. Allows parallel training via a custom CUDA kernel.
- **Relevance**: Alternative state update rule that may be more amenable to integer arithmetic (the correction term prevents unbounded growth).

---

## Recurrent / SSM Approaches

### RetNet (2023) — Retentive Network: A Successor to Transformer for Large Language Models
- **arXiv**: https://arxiv.org/abs/2307.08621
- **Authors**: Sun, Li, Dong, et al. (Microsoft Research)
- **Three computation modes**: Parallel (training), recurrent (O(1) inference per token), chunkwise (balanced).
- **Key numbers**: 7B model, 8k context: 8.4× faster decode than Transformer with KV cache, 70% memory reduction.
- **Retention mechanism**: Decayed outer-product accumulation — each k⊗v contribution decays exponentially with distance. This is GLA with a fixed decay pattern.
- **Relevance**: Shows that recurrent reformulation of attention at scale produces practical speedups. The decay factor is a scalar per head, integer-friendly as a right-shift.

### RWKV — Receptance Weighted Key Value
- **Survey**: https://arxiv.org/html/2412.14847v1
- **Key insight**: WKV attention mechanism with time-mixing (recurrent) and channel-mixing (local). Linear complexity. Can be parallelized during training via parallel scan, runs as an RNN at inference.
- **Versions**: RWKV-4 through RWKV-6, progressively adding matrix-valued states and data-dependent mixing.
- **Relevance**: Shows linear attention is competitive with softmax at language model scale (multi-billion parameter models). Architecture closer to RNN than transformer.

### Mamba (2023) — Linear-Time Sequence Modeling with Selective State Spaces
- **arXiv**: https://arxiv.org/abs/2312.00752
- **Authors**: Gu, Dao (CMU/Princeton)
- **Key insight**: Input-dependent (selective) SSM parameters. The selection mechanism allows the model to focus on relevant tokens, addressing the quality gap of time-invariant SSMs. Hardware-aware parallel scan.
- **Relevance**: State-of-the-art alternative to attention for sequence modeling. The selective scan is float-specific (requires hardware-aware recomputation during backward). Not obviously amenable to integer arithmetic.
- **Integer barrier**: The input-dependent parameters require per-token floating-point operations that are hard to quantize without significant quality loss.

---

## NSRL's Linear Attention Status

**Implemented**: `linear_attention_i16_q15_checked` in nsrl-core. The codebase
now has full linear attention, incremental streaming linear attention, and a
training path using a straight-through numerator backward with the denominator
treated as constant. The state uses causal i64 outer-product accumulation.

**Current training result**: a d=32, seq_len=32, 8192-window run reached
`final_accuracy_per_mille: 184` with zero rollbacks and zero rejected batches
under the `linear_numerator_straight_through_denominator_constant` backward.
The per-projection movement showed the expected causal learning-order
asymmetry: O moved most, K/V moved meaningfully, and Q remained much smaller.
This is consistent with retrieve-after-store dynamics: K/V write the associative
state before Q has a useful state to query.

**Not implemented / not solved**: integer-friendly gating, chunkwise training,
and a full denominator-gradient backward. Gating is especially important because
ungated holographic or linear-attention memories accumulate stale early-training
bindings indefinitely.

**Next steps in order of impact**:

1. **Integer-friendly gating** — Q15 decay factor per head, applied as a
   multiplier or shift. This corresponds to GLA/RetNet retention and gives the
   state a forgetting mechanism.
2. **Denominator-gradient experiment** — compare the current straight-through
   denominator-constant backward against a fuller integer approximation to see
   whether Q movement improves.
3. **TTT-style state update** — treat the streaming state correction as the
   integer delta-rule update and measure whether it improves generation without
   ordinary backprop.
4. **Chunkwise computation** — process in chunks for balanced
   parallelism/memory during training. RetNet's third mode.

**Open question**: Does integer arithmetic hurt or help linear attention quality? The state S in i64 accumulates signal over the entire context. Float linear attention suffers from numerical instability in the key_sum normalization denominator. i64 accumulation with exact integer arithmetic may actually be more stable.
